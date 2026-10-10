from __future__ import annotations

import argparse
import importlib.util
import io
import json
import socket
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
except ImportError as error:
    raise ImportError(
        "fastapi is required for tests.integration.test_tier2_canary_test"
    ) from error

from llm_behavior_ci.config import ConfigError, run_configuration_hash
from llm_behavior_ci.experiments.protocol import task_selection_allowance_from_dict
from llm_behavior_ci.lifecycle.connected import ConnectedLifecycleError, run_three_tier_dev
from llm_behavior_ci.records import RecordError, assert_public_payload
from llm_behavior_ci.runtime.actions import ActionRejected
from llm_behavior_ci.service import create_app
from llm_behavior_ci.storage import EpisodeStore

_ROOT = Path(__file__).resolve().parents[2]
_IDENTIFIER_SUFFIXES = (
    "pair_id",
    "pair_ids",
    "episode_id",
    "episode_ids",
    "task_id",
    "task_ids",
)


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


demo = _load("run_three_tier_dev_for_tier2", "scripts/demo/run_three_tier_dev.py")
harness = _load("run_tier2_canary_test", "scripts/demo/run_tier2_canary_test.py")


def _keys(payload: object) -> list[str]:
    if isinstance(payload, dict):
        return [key for name, value in payload.items() for key in (name, *_keys(value))]
    if isinstance(payload, (list, tuple)):
        return [key for item in payload for key in _keys(item)]
    return []


class NoOutcomeAgent(demo.ScenarioAgent):
    """A candidate whose every action is rejected, so its episode has no evaluator outcome."""

    def next_turn(self, *, tool_output):
        raise ActionRejected("rejected before any tool call")


class CountingFactory:
    def __init__(self, inner) -> None:
        self._inner = inner
        self.calls = 0

    def __call__(self, configuration):
        self.calls += 1
        return self._inner(configuration)


def _no_outcome_candidate(inner, candidate_hash: str):
    def factory(configuration):
        dependencies = inner(configuration)
        if run_configuration_hash(configuration) != candidate_hash:
            return dependencies
        return replace(
            dependencies,
            agent=NoOutcomeAgent(dependencies.clock, faulted=lambda task_id: False),
        )

    return factory


class _HarnessCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.root = Path(self._tmpdir.name)

    def _lifecycle(self, scenario: str):
        lifecycle = demo.build_synthetic(scenario, self.root)
        self.inputs = harness.Tier2Inputs(
            gate_reference=lifecycle.gate.reference,
            gate_candidate=lifecycle.gate.candidate,
            traffic=replace(lifecycle.traffic, production_task_ids=()),
        )
        self.allowance = task_selection_allowance_from_dict(
            json.loads((self.root / "task_selection_allowance.json").read_text())
        )
        self.store_path = lifecycle.store_path
        return lifecycle

    def _run(self, scenario: str, *, expected_from=None, **dependency_changes):
        lifecycle = self._lifecycle(scenario)
        changes = {
            key: value(lifecycle) if callable(value) else value
            for key, value in dependency_changes.items()
        }
        dependencies = replace(lifecycle.dependencies, **changes)
        intended = dependencies if expected_from is None else expected_from(lifecycle)
        with TestClient(harness.isolated_app(dependencies)) as client:
            return harness.run_tier2_test(
                self.inputs,
                store_path=self.store_path,
                service=client,
                allowance=self.allowance,
                expected_identity=harness.identity_from_dependencies(intended),
                git_commit=self.inputs.traffic.production.git_commit,
            )

    def _rows(self):
        reader = EpisodeStore(self.store_path)
        try:
            return (
                reader.load_validation_artifact_ids(),
                reader.load_deployment_decisions(),
                reader.load_finished_episodes(),
            )
        finally:
            reader.close()


class Tier2CanaryTestHarnessTests(_HarnessCase):
    def test_test_only_admission_runs_the_canary_to_promotion(self) -> None:
        summary = self._run("healthy")
        count = len(self.inputs.traffic.canary_task_ids)
        execution = summary["canary_execution"]
        self.assertEqual(summary["status"], "promoted")
        self.assertTrue(summary["test_only_admission"])
        self.assertFalse(summary["gate_executed"])
        self.assertEqual(summary["admission_evidence_source"], "synthetic_fixture")
        self.assertTrue(summary["integrity_verified"], summary["integrity_checks"])
        self.assertTrue(summary["tier2_executed"])
        self.assertTrue(summary["admission_verification"]["admitted"])
        self.assertEqual(execution["pairs_executed"], count)
        self.assertEqual(execution["pairs_persisted"], count)
        self.assertEqual(execution["pairs_with_both_evaluator_outcomes"], count)
        self.assertEqual(execution["pairs_eligible_for_stopping_rule"], count)
        self.assertEqual(execution["stopping_rule_observations"], count)
        self.assertTrue(execution["horizon_reached"])
        self.assertEqual(execution["controller_decision"], "promote")
        self.assertEqual(execution["controller_decision_method"], "stopping_rule")
        self.assertEqual(execution["final_state"], "PROMOTED")
        self.assertEqual(execution["statistical_integration"], "complete")
        self.assertEqual(harness.exit_code_for(summary), 0)
        self.assertEqual(
            summary["deployment"]["serving_configuration_hash"],
            run_configuration_hash(self.inputs.traffic.candidate),
        )
        self.assertNotIn("tier3", summary)
        self.assertNotIn("dev-", json.dumps(summary))
        self.assertFalse(
            [key for key in _keys(summary) if key.endswith(_IDENTIFIER_SUFFIXES)]
        )
        artifacts, decisions, episodes = self._rows()
        self.assertEqual(artifacts, (summary["fixture_artifact_id"],))
        self.assertEqual(decisions[0].evidence_source, "synthetic_fixture")
        self.assertEqual(len(episodes), 2 * count)

    def test_a_regressed_candidate_is_rolled_back_by_the_stopping_rule(self) -> None:
        summary = self._run("regression")
        execution = summary["canary_execution"]
        self.assertEqual(summary["status"], "rolled_back")
        self.assertTrue(summary["integrity_verified"], summary["integrity_checks"])
        self.assertTrue(summary["integrity_checks"]["candidate_traffic_closed"])
        self.assertEqual(execution["controller_decision"], "rollback")
        self.assertEqual(execution["controller_decision_method"], "stopping_rule_alarm")
        self.assertEqual(execution["final_state"], "ROLLED_BACK")
        self.assertEqual(execution["statistical_integration"], "complete")
        self.assertEqual(
            execution["stopping_rule_observations"], execution["pairs_executed"]
        )
        self.assertEqual(harness.exit_code_for(summary), 0)

    def test_an_expected_admission_refusal_passes_safety_but_fails_the_run(self) -> None:
        summary = self._run("healthy", admission_mode="release")
        self.assertEqual(summary["status"], "admission_refused")
        self.assertTrue(summary["integrity_verified"], summary["integrity_checks"])
        self.assertTrue(
            summary["admission_verification"]["release_admission_refuses_fixture"]
        )
        self.assertFalse(summary["admission_verification"]["admitted"])
        self.assertEqual(summary["admission_verification"]["status_code"], 409)
        self.assertFalse(summary["tier2_executed"])
        self.assertEqual(summary["canary_execution"]["pairs_executed"], 0)
        self.assertEqual(
            summary["canary_execution"]["statistical_integration"], "incomplete"
        )
        self.assertEqual(harness.exit_code_for(summary), 1)
        artifacts, decisions, episodes = self._rows()
        self.assertEqual(len(artifacts), 1)
        self.assertEqual((decisions, episodes), ((), ()))

    def test_missing_evaluator_outcomes_are_not_statistical_observations(self) -> None:
        summary = self._run(
            "healthy",
            runtime_factory=lambda lifecycle: _no_outcome_candidate(
                lifecycle.dependencies.runtime_factory,
                run_configuration_hash(lifecycle.traffic.candidate),
            ),
        )
        count = len(self.inputs.traffic.canary_task_ids)
        execution = summary["canary_execution"]
        self.assertEqual(summary["status"], "canary_incomplete")
        self.assertTrue(summary["integrity_verified"], summary["integrity_checks"])
        self.assertTrue(summary["integrity_checks"]["candidate_traffic_closed"])
        self.assertEqual(execution["pairs_executed"], count)
        self.assertEqual(execution["pairs_persisted"], count)
        self.assertEqual(execution["pairs_with_both_evaluator_outcomes"], 0)
        self.assertEqual(execution["pairs_missing_an_evaluator_outcome"], count)
        self.assertEqual(execution["stopping_rule_observations"], 0)
        self.assertFalse(execution["horizon_reached"])
        self.assertEqual(execution["controller_decision_method"], "manual_rollback")
        self.assertEqual(execution["statistical_integration"], "incomplete")
        self.assertEqual(summary["cleanup"]["rollback"], "manual_rollback")
        self.assertEqual(harness.exit_code_for(summary), 3)

    def test_observed_pairs_short_of_the_horizon_stay_statistically_incomplete(self) -> None:
        lifecycle = self._lifecycle("healthy")
        self.inputs = replace(
            self.inputs,
            traffic=replace(
                self.inputs.traffic, canary_task_ids=self.inputs.traffic.canary_task_ids[:2]
            ),
        )
        with TestClient(harness.isolated_app(lifecycle.dependencies)) as client:
            summary = harness.run_tier2_test(
                self.inputs,
                store_path=self.store_path,
                service=client,
                allowance=self.allowance,
                expected_identity=harness.identity_from_dependencies(lifecycle.dependencies),
                git_commit=self.inputs.traffic.production.git_commit,
            )
        execution = summary["canary_execution"]
        self.assertEqual(summary["status"], "canary_incomplete")
        self.assertTrue(summary["integrity_verified"], summary["integrity_checks"])
        self.assertEqual(execution["stopping_rule_observations"], 2)
        self.assertFalse(execution["horizon_reached"])
        self.assertEqual(execution["controller_decision_method"], "manual_rollback")
        self.assertEqual(execution["statistical_integration"], "incomplete")
        self.assertEqual(harness.exit_code_for(summary), 3)

    def test_a_used_store_is_refused_before_admission(self) -> None:
        self._run("healthy")
        lifecycle = demo.build_synthetic("healthy", self.root)
        with self.assertRaisesRegex(ConnectedLifecycleError, "needs its own store"):
            with TestClient(harness.isolated_app(lifecycle.dependencies)) as client:
                harness.run_tier2_test(
                    self.inputs,
                    store_path=lifecycle.store_path,
                    service=client,
                    allowance=None,
                    expected_identity=harness.identity_from_dependencies(
                        lifecycle.dependencies
                    ),
                    git_commit=self.inputs.traffic.production.git_commit,
                )

    def test_pair_checks_fail_when_roles_carry_the_wrong_hashes(self) -> None:
        summary = self._run("healthy")
        reader = EpisodeStore(self.store_path)
        try:
            pair_ids = [
                str(item.episode.pair_id)
                for item in reader.load_finished_episodes()
                if item.role == "candidate"
            ]
        finally:
            reader.close()
        production = summary["configurations"]["production"]
        candidate = summary["configurations"]["candidate"]
        _, checks = harness._pair_checks(self.store_path, pair_ids, production, candidate)
        self.assertTrue(checks["pairs_use_registered_hashes"])
        _, swapped = harness._pair_checks(self.store_path, pair_ids, candidate, production)
        self.assertFalse(swapped["pairs_use_registered_hashes"])

    def test_admission_checks_tell_the_fixture_from_gate_evidence(self) -> None:
        self._lifecycle("healthy")
        fixture = harness.fixture_artifact(self.inputs)
        self.assertEqual(
            harness.admission_checks(fixture, self.inputs, self.allowance),
            {
                "fixture_labelled_synthetic": True,
                "test_admission_accepts_fixture": True,
                "release_admission_refuses_fixture": True,
            },
        )
        relabelled = replace(fixture, evidence_source="gate_run")
        self.assertEqual(
            harness.admission_checks(relabelled, self.inputs, self.allowance),
            {
                "fixture_labelled_synthetic": False,
                "test_admission_accepts_fixture": True,
                "release_admission_refuses_fixture": False,
            },
        )


class ServiceIdentityTests(_HarnessCase):
    def test_a_settings_mismatch_aborts_before_any_model_episode(self) -> None:
        def rule(lifecycle, **changes):
            settings = lifecycle.dependencies.canary_settings
            return replace(settings, stopping_rule=replace(settings.stopping_rule, **changes))

        def canary(lifecycle, **changes):
            return replace(lifecycle.dependencies.canary_settings, **changes)

        cases = {
            "horizon": {"canary_settings": lambda lc: rule(lc, horizon_episodes=3)},
            "alpha": {"canary_settings": lambda lc: rule(lc, alpha=0.1)},
            "harm margin": {"canary_settings": lambda lc: canary(lc, harm_margin=0.3)},
            "fraction": {"canary_settings": lambda lc: canary(lc, fraction=0.5)},
            "outcome delay": {
                "canary_settings": lambda lc: canary(lc, outcome_delay_seconds=1.0)
            },
            "assignment seed": {"canary_assignment_seed": 99},
            "allowance": {"task_selection_allowance": None},
            "release admission": {"admission_mode": "release"},
        }
        for name, changes in cases.items():
            with self.subTest(case=name):
                self.root = Path(self._tmpdir.name) / name.replace(" ", "_")
                self.root.mkdir()
                counting: list[CountingFactory] = []

                def wrap(lifecycle):
                    factory = CountingFactory(lifecycle.dependencies.runtime_factory)
                    counting.append(factory)
                    return factory

                with self.assertRaises(harness.ServiceIdentityError):
                    self._run(
                        "healthy",
                        expected_from=lambda lifecycle: lifecycle.dependencies,
                        runtime_factory=wrap,
                        **changes,
                    )
                self.assertEqual(counting[0].calls, 0)
                self.assertEqual(self._rows(), ((), (), ()))

    def test_a_service_without_the_test_identity_is_refused(self) -> None:
        lifecycle = self._lifecycle("healthy")
        with TestClient(create_app(lifecycle.dependencies)) as client:
            with self.assertRaisesRegex(harness.ServiceIdentityError, "not the isolated"):
                harness.run_tier2_test(
                    self.inputs,
                    store_path=self.store_path,
                    service=client,
                    allowance=self.allowance,
                    expected_identity=harness.identity_from_dependencies(
                        lifecycle.dependencies
                    ),
                    git_commit=self.inputs.traffic.production.git_commit,
                )
        self.assertEqual(self._rows(), ((), (), ()))

    def test_the_identity_names_every_bound_setting(self) -> None:
        lifecycle = self._lifecycle("healthy")
        identity = harness.identity_from_dependencies(lifecycle.dependencies)
        canary = identity["canary_settings"]
        self.assertEqual(identity["admission_mode"], "test")
        self.assertEqual(
            identity["production_configuration_hash"],
            run_configuration_hash(self.inputs.traffic.production),
        )
        self.assertEqual(
            identity["candidate_configuration_hash"],
            run_configuration_hash(self.inputs.traffic.candidate),
        )
        self.assertEqual(identity["canary_assignment_seed"], 0)
        self.assertEqual(
            set(canary),
            {
                "fraction",
                "harm_margin",
                "metric_orientation",
                "outcome_delay_seconds",
                "promotion_policy",
                "stopping_rule",
            },
        )
        self.assertEqual(
            set(canary["stopping_rule"]), {"alpha", "horizon_episodes", "name", "threshold"}
        )
        self.assertIsNotNone(identity["task_selection_allowance"])
        self.assertEqual(
            identity["monitor_reference"]["configuration_hash"],
            run_configuration_hash(self.inputs.traffic.production),
        )
        self.assertEqual(len(identity["identity_sha256"]), 64)


class PublicOutputPrivacyTests(unittest.TestCase):
    def test_plural_and_nested_identifier_fields_are_rejected(self) -> None:
        for payload in (
            {"canary_pair_ids": ["p"]},
            {"pair_ids": ["p"]},
            {"stage": {"episode_ids": ["e"]}},
            {"items": [{"reference_pair_id": "p"}]},
            {"task_ids": ["t"]},
            {"deep": [{"inner": {"candidate_episode_id": "e"}}]},
            {"scenario_ids": ["s"]},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(RecordError):
                    assert_public_payload(payload)
        assert_public_payload({"invalid_action": 1, "fixture_artifact_id": "a", "pairs": 2})

    def test_the_three_tier_lifecycle_summary_carries_no_identifiers(self) -> None:
        with TemporaryDirectory() as directory:
            lifecycle = demo.build_synthetic("healthy", Path(directory))
            with TestClient(create_app(lifecycle.dependencies)) as client:
                summary = run_three_tier_dev(
                    lifecycle.gate,
                    lifecycle.traffic,
                    store_path=lifecycle.store_path,
                    service=client,
                )
        self.assertEqual(summary["status"], "promoted")
        self.assertFalse(
            [key for key in _keys(summary) if key.endswith(_IDENTIFIER_SUFFIXES)]
        )
        self.assertEqual(
            set(summary),
            {
                "record",
                "tier1",
                "admission",
                "tier2",
                "tier3",
                "status",
                "deployment",
                "evidence",
            },
        )


class Tier2CanaryTestIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.root = Path(self._tmpdir.name)

    def test_the_test_service_needs_a_new_store_and_a_free_local_port(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
            busy.bind((harness.HOST, 0))
            busy.listen()
            taken = busy.getsockname()[1]
            existing = self.root / "used.sqlite"
            existing.write_bytes(b"")
            results = self.root / "results"
            results.mkdir()
            for store, port in (
                (existing, 8031),
                (results / "new.sqlite", 8031),
                (self.root / "new.sqlite", taken),
                (self.root / "new.sqlite", 0),
            ):
                with self.subTest(store=store.name, port=port):
                    with self.assertRaises(ConfigError):
                        harness.require_isolated_target(store, port)
        free = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        free.bind((harness.HOST, 0))
        port = free.getsockname()[1]
        free.close()
        harness.require_isolated_target(self.root / "new.sqlite", port)
        self.assertEqual(harness.HOST, "127.0.0.1")

    def _argv(self, command: str, **overrides: str) -> list[str]:
        values = {
            "--gate-reference": "x",
            "--gate-candidate": "x",
            "--production-config": "x",
            "--candidate-config": "x",
            "--dev-task-set": "x",
            "--task-selection-allowance": "x",
            "--canary-arrivals": "4",
            "--canary-settings": "x",
            "--monitor-settings": "x",
            "--frozen-reference": "x",
            "--store": str(self.root / "store.sqlite"),
            "--port": "8031",
        }
        values.update(overrides)
        argv = [command]
        for flag, value in values.items():
            argv += [flag, value]
        return argv

    def _inputs(self):
        lifecycle = demo.build_synthetic("healthy", self.root)
        return harness.Tier2Inputs(
            gate_reference=lifecycle.gate.reference,
            gate_candidate=lifecycle.gate.candidate,
            traffic=replace(lifecycle.traffic, production_task_ids=()),
        )

    def test_serve_refuses_a_used_store_without_starting(self) -> None:
        used = self.root / "used.sqlite"
        used.write_bytes(b"")
        with patch.object(harness, "load_inputs", return_value=self._inputs()), patch.object(
            harness, "isolated_service_dependencies"
        ) as build, redirect_stderr(io.StringIO()):
            self.assertEqual(harness.main(self._argv("serve", **{"--store": str(used)})), 2)
        build.assert_not_called()

    def test_drive_exit_codes_follow_the_verdicts(self) -> None:
        (self.root / "store.sqlite").write_bytes(b"")

        def verdict(integrity: bool, executed: bool, statistical: str) -> dict[str, object]:
            return {
                "return_value": {
                    "integrity_verified": integrity,
                    "tier2_executed": executed,
                    "canary_execution": {"statistical_integration": statistical},
                }
            }

        cases = {
            "identity mismatch": (
                {"side_effect": harness.ServiceIdentityError("mismatch")},
                2,
            ),
            "admission refused": (verdict(True, False, "incomplete"), 1),
            "integrity failure": (verdict(False, True, "complete"), 1),
            "statistically incomplete": (verdict(True, True, "incomplete"), 3),
            "complete": (verdict(True, True, "complete"), 0),
            "execution failure": ({"side_effect": ConnectionError("refused")}, 4),
        }
        inputs = self._inputs()
        for name, (effect, code) in cases.items():
            with self.subTest(case=name):
                summary_path = self.root / f"{name.replace(' ', '_')}.json"
                with patch.object(harness, "load_inputs", return_value=inputs), patch.object(
                    harness, "load_allowance", return_value=None
                ), patch.object(
                    harness, "intended_identity", return_value={}
                ), patch.object(
                    harness, "run_tier2_test", **effect
                ), redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                    exit_code = harness.main(
                        self._argv("drive", **{"--summary": str(summary_path)})
                    )
                self.assertEqual(exit_code, code)

    def test_preflight_builds_the_test_service_without_any_episode(self) -> None:
        inputs = self._inputs()
        args = argparse.Namespace(
            store=str(self.root / "fresh.sqlite"),
            port="0",
            task_selection_allowance=str(self.root / "task_selection_allowance.json"),
            production_config=str(self.root / "production.json"),
            candidate_config=str(self.root / "candidate.json"),
            canary_settings=str(self.root / "canary.json"),
            monitor_settings=str(self.root / "monitor.json"),
            frozen_reference=str(self.root / "frozen_reference.json"),
            canary_assignment_seed="0",
        )
        original = harness.service_arguments

        def with_endpoints(namespace):
            return argparse.Namespace(
                **{
                    **vars(original(namespace)),
                    "production_base_url": "http://production.invalid",
                    "candidate_base_url": "http://candidate.invalid",
                }
            )

        with patch.object(harness, "load_inputs", return_value=inputs), patch.object(
            harness, "service_arguments", side_effect=with_endpoints
        ), patch.object(harness, "require_isolated_target"):
            summary = harness.preflight(args)
            built = harness.isolated_service_dependencies
            with patch.object(
                harness,
                "isolated_service_dependencies",
                side_effect=lambda namespace: replace(
                    built(namespace), canary_assignment_seed=99
                ),
            ):
                drifted = harness.preflight(args)
        self.assertFalse(drifted["checks"]["service_identity_matches_intended"])
        self.assertFalse(drifted["verified"])
        self.assertTrue(summary["verified"], summary["checks"])
        self.assertTrue(summary["checks"]["service_identity_matches_intended"])
        self.assertEqual(summary["provider_calls"], 0)
        self.assertEqual(
            summary["max_execute_episodes"], 2 * len(inputs.traffic.canary_task_ids)
        )
        self.assertEqual(summary["service_identity"]["admission_mode"], "test")
        self.assertFalse((self.root / "fresh.sqlite").exists())

    def test_bare_invocation_exits_2(self) -> None:
        self.assertEqual(harness.main([]), 2)
        with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            self.assertEqual(harness.main(["drive"]), 2)


if __name__ == "__main__":
    unittest.main()
