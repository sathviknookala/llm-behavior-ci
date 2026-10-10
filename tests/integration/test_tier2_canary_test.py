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
from llm_behavior_ci.lifecycle.connected import ConnectedLifecycleError
from llm_behavior_ci.runtime.actions import ActionRejected
from llm_behavior_ci.service import create_app
from llm_behavior_ci.storage import EpisodeStore

_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


demo = _load("run_three_tier_dev_for_tier2", "scripts/demo/run_three_tier_dev.py")
harness = _load("run_tier2_canary_test", "scripts/demo/run_tier2_canary_test.py")


class NoOutcomeAgent(demo.ScenarioAgent):
    """A candidate whose every action is rejected, so its episode has no evaluator outcome."""

    def next_turn(self, *, tool_output):
        raise ActionRejected("rejected before any tool call")


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


class Tier2CanaryTestHarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.root = Path(self._tmpdir.name)

    def _run(self, scenario: str, **dependency_changes):
        lifecycle = demo.build_synthetic(scenario, self.root)
        inputs = harness.Tier2Inputs(
            gate_reference=lifecycle.gate.reference,
            gate_candidate=lifecycle.gate.candidate,
            traffic=replace(lifecycle.traffic, production_task_ids=()),
        )
        allowance = task_selection_allowance_from_dict(
            json.loads((self.root / "task_selection_allowance.json").read_text())
        )
        changes = {
            key: value(lifecycle) if callable(value) else value
            for key, value in dependency_changes.items()
        }
        dependencies = replace(lifecycle.dependencies, **changes)
        with TestClient(create_app(dependencies)) as client:
            summary = harness.run_tier2_test(
                inputs,
                store_path=lifecycle.store_path,
                service=client,
                allowance=allowance,
                git_commit=inputs.traffic.production.git_commit,
            )
        self.store_path = lifecycle.store_path
        self.inputs = inputs
        return summary

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

    def test_test_only_admission_runs_the_canary_to_promotion(self) -> None:
        summary = self._run("healthy")
        self.assertEqual(summary["status"], "promoted")
        self.assertTrue(summary["test_only_admission"])
        self.assertFalse(summary["gate_executed"])
        self.assertEqual(summary["admission_evidence_source"], "synthetic_fixture")
        self.assertTrue(summary["verified"], summary["checks"])
        self.assertTrue(summary["checks"]["release_admission_refuses_fixture"])
        traffic = self.inputs.traffic
        self.assertEqual(summary["pairs"]["pairs"], len(traffic.canary_task_ids))
        self.assertEqual(summary["pairs"]["eligible_pairs"], len(traffic.canary_task_ids))
        self.assertEqual(
            [row["decision"] for row in summary["lifecycle_decisions"]], ["admit", "promote"]
        )
        self.assertEqual(
            summary["deployment"]["serving_configuration_hash"],
            run_configuration_hash(traffic.candidate),
        )
        self.assertNotIn("tier3", summary)
        self.assertNotIn("dev-", json.dumps(summary))

        artifacts, decisions, episodes = self._rows()
        self.assertEqual(artifacts, (summary["fixture_artifact_id"],))
        self.assertEqual(decisions[0].evidence_source, "synthetic_fixture")
        self.assertEqual(len(episodes), 2 * len(traffic.canary_task_ids))
        hashes = {item.run.configuration_hash for item in episodes}
        self.assertEqual(
            hashes,
            {run_configuration_hash(traffic.production), run_configuration_hash(traffic.candidate)},
        )

    def test_admission_checks_tell_the_fixture_from_gate_evidence(self) -> None:
        lifecycle = demo.build_synthetic("healthy", self.root)
        inputs = harness.Tier2Inputs(
            gate_reference=lifecycle.gate.reference,
            gate_candidate=lifecycle.gate.candidate,
            traffic=replace(lifecycle.traffic, production_task_ids=()),
        )
        allowance = task_selection_allowance_from_dict(
            json.loads((self.root / "task_selection_allowance.json").read_text())
        )
        fixture = harness.fixture_artifact(inputs)
        self.assertEqual(
            harness.admission_checks(fixture, inputs, allowance),
            {
                "fixture_labelled_synthetic": True,
                "test_admission_accepts_fixture": True,
                "release_admission_refuses_fixture": True,
            },
        )
        relabelled = replace(fixture, evidence_source="gate_run")
        self.assertEqual(
            harness.admission_checks(relabelled, inputs, allowance),
            {
                "fixture_labelled_synthetic": False,
                "test_admission_accepts_fixture": True,
                "release_admission_refuses_fixture": False,
            },
        )

    def test_pair_checks_fail_when_roles_carry_the_wrong_hashes(self) -> None:
        summary = self._run("healthy")
        reader = EpisodeStore(self.store_path)
        try:
            pair_ids = [
                item.episode.pair_id
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

    def test_a_regressed_candidate_is_rolled_back_and_closed(self) -> None:
        summary = self._run("regression")
        self.assertEqual(summary["status"], "rolled_back")
        self.assertTrue(summary["verified"], summary["checks"])
        self.assertTrue(summary["checks"]["candidate_traffic_closed"])
        self.assertEqual(summary["deployment"]["state"], "ROLLED_BACK")
        self.assertEqual(
            [row["decision"] for row in summary["lifecycle_decisions"]], ["admit", "rollback"]
        )
        self.assertLess(summary["pairs"]["pairs"], len(self.inputs.traffic.canary_task_ids) + 1)

    def test_release_admission_refuses_the_fixture(self) -> None:
        summary = self._run("healthy", admission_mode="release")
        self.assertEqual(summary["status"], "admission_refused")
        self.assertEqual(summary["admission"]["status_code"], 409)
        self.assertIn("synthetic_fixture", summary["admission"]["detail"])
        self.assertTrue(summary["verified"], summary["checks"])
        artifacts, decisions, episodes = self._rows()
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(decisions, ())
        self.assertEqual(episodes, ())

    def test_pairs_without_evaluator_outcomes_end_incomplete_and_closed(self) -> None:
        summary = self._run(
            "healthy",
            runtime_factory=lambda lifecycle: _no_outcome_candidate(
                lifecycle.dependencies.runtime_factory,
                run_configuration_hash(lifecycle.traffic.candidate),
            ),
        )
        count = len(self.inputs.traffic.canary_task_ids)
        self.assertEqual(summary["status"], "canary_incomplete")
        self.assertTrue(summary["verified"], summary["checks"])
        self.assertEqual(summary["pairs"]["pairs"], count)
        self.assertEqual(summary["pairs"]["eligible_pairs"], 0)
        self.assertEqual(summary["pairs"]["missing_evaluator_outcome"], count)
        self.assertEqual(summary["cleanup"]["rollback"], "manual_rollback")
        self.assertEqual(summary["deployment"]["admission"], "rollback_requested")
        self.assertEqual(
            [row["method"] for row in summary["lifecycle_decisions"]][-1], "manual_rollback"
        )

    def test_a_used_store_is_refused_before_admission(self) -> None:
        self._run("healthy")
        with self.assertRaisesRegex(ConnectedLifecycleError, "needs its own store"):
            lifecycle = demo.build_synthetic("healthy", self.root)
            with TestClient(create_app(lifecycle.dependencies)) as client:
                harness.run_tier2_test(
                    self.inputs,
                    store_path=lifecycle.store_path,
                    service=client,
                    allowance=None,
                    git_commit=self.inputs.traffic.production.git_commit,
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

    def test_serve_refuses_a_used_store_without_starting(self) -> None:
        lifecycle = demo.build_synthetic("healthy", self.root)
        used = self.root / "used.sqlite"
        used.write_bytes(b"")
        argv = ["serve", "--store", str(used), "--port", "8031"]
        for flag in (
            "--gate-reference", "--gate-candidate", "--production-config", "--candidate-config",
            "--dev-task-set", "--task-selection-allowance", "--canary-arrivals",
            "--canary-settings", "--monitor-settings", "--frozen-reference",
        ):
            argv += [flag, "x"]
        inputs = harness.Tier2Inputs(
            gate_reference=lifecycle.gate.reference,
            gate_candidate=lifecycle.gate.candidate,
            traffic=replace(lifecycle.traffic, production_task_ids=()),
        )
        with patch.object(harness, "load_inputs", return_value=inputs), patch.object(
            harness, "isolated_service_dependencies"
        ) as build, redirect_stderr(io.StringIO()):
            self.assertEqual(harness.main(argv), 2)
        build.assert_not_called()

    def test_preflight_builds_the_test_service_without_any_episode(self) -> None:
        lifecycle = demo.build_synthetic("healthy", self.root)
        inputs = harness.Tier2Inputs(
            gate_reference=lifecycle.gate.reference,
            gate_candidate=lifecycle.gate.candidate,
            traffic=replace(lifecycle.traffic, production_task_ids=()),
        )
        args = argparse.Namespace(
            store=str(self.root / "fresh.sqlite"),
            port="0",
            task_selection_allowance=str(self.root / "task_selection_allowance.json"),
            production_config=str(self.root / "production.json"),
            candidate_config=str(self.root / "candidate.json"),
            canary_settings=str(self.root / "canary.json"),
            monitor_settings=str(self.root / "monitor.json"),
            frozen_reference=str(self.root / "frozen_reference.json"),
            canary_assignment_seed="3",
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

        built = []
        original_build = harness.serve._build_dependencies

        def spy_build(namespace):
            dependencies = original_build(namespace)
            built.append(dependencies)
            return dependencies

        with patch.object(harness, "load_inputs", return_value=inputs), patch.object(
            harness, "service_arguments", side_effect=with_endpoints
        ), patch.object(harness.serve, "_build_dependencies", side_effect=spy_build), patch.object(
            harness, "require_isolated_target"
        ):
            summary = harness.preflight(args)
        self.assertTrue(summary["verified"], summary["checks"])
        self.assertEqual(summary["provider_calls"], 0)
        self.assertEqual(summary["max_execute_episodes"], 2 * len(inputs.traffic.canary_task_ids))
        self.assertTrue(summary["checks"]["release_admission_refuses_fixture"])
        self.assertTrue(summary["checks"]["test_admission_accepts_fixture"])
        self.assertFalse((self.root / "fresh.sqlite").exists())
        self.assertEqual(len(built), 1)
        self.assertEqual(built[0].admission_mode, "release")

    def test_bare_invocation_exits_2(self) -> None:
        self.assertEqual(harness.main([]), 2)
        with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            self.assertEqual(harness.main(["drive"]), 2)


if __name__ == "__main__":
    unittest.main()
