from __future__ import annotations

import importlib.util
import io
import json
import sqlite3
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
        "fastapi is required for tests.integration.test_three_tier_dev"
    ) from error

from llm_behavior_ci.config import TaskConfiguration, run_configuration_hash
from llm_behavior_ci.experiments.run_config import build_run_configuration
from llm_behavior_ci.lifecycle.connected import (
    ConnectedLifecycleError,
    DevTraffic,
    run_three_tier_dev,
)
from llm_behavior_ci.lifecycle.monitoring import monitoring_period_id
from llm_behavior_ci.lifecycle.offline_gate import GateCapabilityError
from llm_behavior_ci.runtime.episode import RuntimeUnavailable
from llm_behavior_ci.service import create_app
from llm_behavior_ci.storage import EpisodeStore

_ROOT = Path(__file__).resolve().parents[2]


def _load_demo():
    path = _ROOT / "scripts" / "demo" / "run_three_tier_dev.py"
    spec = importlib.util.spec_from_file_location("run_three_tier_dev", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


demo = _load_demo()


class CountingFactory:
    def __init__(self, inner) -> None:
        self._inner = inner
        self.hashes: list[str] = []

    def __call__(self, configuration):
        self.hashes.append(run_configuration_hash(configuration))
        return self._inner(configuration)


class FakeResponse:
    def __init__(self, status_code: int, body: object) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> object:
        return self._body


class RollbackOverride:
    def __init__(self, inner, respond) -> None:
        self._inner = inner
        self._respond = respond
        self.rollback_calls = 0

    def get(self, url: str):
        return self._inner.get(url)

    def post(self, url: str, *, json: object):
        if url == "/deployment/rollback":
            self.rollback_calls += 1
            return self._respond(self._inner)
        return self._inner.post(url, json=json)


class MisroutedProduction:
    def __init__(self, inner, configuration_hash: str) -> None:
        self._inner = inner
        self._hash = configuration_hash

    def get(self, url: str):
        return self._inner.get(url)

    def post(self, url: str, *, json: object):
        response = self._inner.post(url, json=json)
        key = json.get("assignment_key", "") if isinstance(json, dict) else ""
        if url == "/episodes" and key.startswith("production:"):
            return FakeResponse(
                response.status_code, {**response.json(), "configuration_hash": self._hash}
            )
        return response


class ThreeTierDevLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.root = Path(self._tmpdir.name)

    def _run(self, scenario: str, **changes):
        lifecycle = demo.build_synthetic(scenario, self.root)
        counting = CountingFactory(lifecycle.dependencies.runtime_factory)
        dependencies = replace(
            lifecycle.dependencies,
            **{"runtime_factory": counting, **changes.pop("dependencies", {})},
        )
        traffic = changes.pop("traffic", lifecycle.traffic)
        gate = changes.pop("gate", lifecycle.gate)
        self.assertEqual(changes, {})
        self.production_hash = run_configuration_hash(lifecycle.traffic.production)
        self.candidate_hash = run_configuration_hash(lifecycle.traffic.candidate)
        self.store_path = lifecycle.store_path
        self.counting = counting
        self.client = TestClient(create_app(dependencies))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        return run_three_tier_dev(
            gate, traffic, store_path=lifecycle.store_path, service=self.client
        )

    def _store(self) -> EpisodeStore:
        reader = EpisodeStore(self.store_path)
        self.addCleanup(reader.close)
        return reader

    def _artifact_count(self) -> int:
        connection = sqlite3.connect(self.store_path)
        try:
            return connection.execute(
                "SELECT COUNT(*) FROM validation_artifacts"
            ).fetchone()[0]
        finally:
            connection.close()

    def test_blocked_gate_stops_before_admission_and_execution(self) -> None:
        summary = self._run("blocked")

        self.assertEqual(summary["status"], "blocked")
        self.assertEqual(summary["tier1"]["outcome"], "BLOCK")
        self.assertEqual(summary["tier1"]["reason_codes"], ["plan_run_failed"])
        self.assertNotIn("admission", summary)
        self.assertNotIn("tier2", summary)
        self.assertNotIn("tier3", summary)
        self.assertEqual(self.counting.hashes, [])

        store = self._store()
        artifact = store.load_validation_artifact(summary["tier1"]["artifact_id"])
        self.assertIsNotNone(artifact)
        self.assertEqual(artifact.outcome, "BLOCK")
        self.assertEqual(store.load_deployment_decisions(), ())
        self.assertEqual(store.load_finished_episodes(), ())
        self.assertEqual(store.load_alerts(), ())

        deployment = self.client.get("/deployment").json()
        self.assertNotIn("state", deployment)
        self.assertEqual(deployment["production_configuration_hash"], self.production_hash)
        refused = self.client.post(
            "/candidates", json={"artifact_id": artifact.artifact_id}
        )
        self.assertEqual(refused.status_code, 409)
        self.assertIn("PASS", refused.json()["detail"])

    def test_healthy_candidate_promotes_and_reaches_the_production_monitor(self) -> None:
        summary = self._run("healthy")

        self.assertEqual(summary["status"], "promoted")
        self.assertNotIn("fallback_verification", summary)
        self.assertNotIn("cleanup", summary)
        tier1 = summary["tier1"]
        self.assertEqual(tier1["outcome"], "PASS")
        self.assertEqual(tier1["evidence_source"], "synthetic_fixture")
        self.assertNotEqual(tier1["candidate_configuration_hash"], self.candidate_hash)
        self.assertEqual(summary["admission"], {"status_code": 200, "state": "GATE_PASSED"})

        self.assertEqual(summary["tier2"]["by_role"], {"candidate": 4})
        self.assertEqual(summary["tier2"]["state"], "PROMOTED")
        tier3 = summary["tier3"]
        self.assertEqual(tier3["by_role"], {"production": 7})
        self.assertEqual(tier3["by_configuration_hash"], {self.candidate_hash: 7})
        self.assertEqual(tier3["by_monitoring_status"], {"updated": 7})

        deployment = self.client.get("/deployment").json()
        self.assertEqual(deployment["state"], "PROMOTED")
        self.assertEqual(deployment["serving_configuration_hash"], self.candidate_hash)
        self.assertEqual(deployment["production_configuration_hash"], self.candidate_hash)
        self.assertEqual(
            deployment["previous_production_configuration_hash"], self.production_hash
        )
        period = monitoring_period_id(self.candidate_hash, self.production_hash)
        self.assertEqual(deployment["monitor_period_id"], period)

        store = self._store()
        decisions = store.load_deployment_decisions()
        self.assertEqual(
            [item.decision for item in decisions], ["admit", "promote", "alert"]
        )
        admit, promote, _ = decisions
        self.assertEqual(admit.evidence_artifact_id, tier1["artifact_id"])
        self.assertEqual(admit.evidence_source, "synthetic_fixture")
        self.assertEqual(promote.configuration_hash, self.candidate_hash)
        self.assertEqual(promote.reference_configuration_hash, self.production_hash)
        self.assertEqual(promote.sample_size, 4)

        alerts = store.load_alerts()
        self.assertEqual(len(alerts), 1)
        alert = alerts[0]
        self.assertEqual(alert.signal, "task_success")
        self.assertEqual(alert.configuration_hash, self.candidate_hash)
        self.assertEqual(alert.reference_configuration_hash, self.production_hash)
        self.assertEqual(alert.period_id, period)

        episodes = store.load_finished_episodes()
        by_hash = {}
        for episode in episodes:
            by_hash.setdefault(episode.run.configuration_hash, []).append(episode)
        self.assertEqual(len(by_hash[self.production_hash]), 4)
        self.assertEqual(len(by_hash[self.candidate_hash]), 11)
        self.assertEqual(
            summary["evidence"]["episodes_by_configuration_hash"],
            {self.production_hash: 4, self.candidate_hash: 11},
        )

    def test_steady_promoted_traffic_raises_no_alert(self) -> None:
        lifecycle = demo.build_synthetic("healthy", self.root)
        steady = replace(lifecycle.traffic, production_task_ids=demo.STEADY_TASK_IDS)
        summary = self._run("healthy", traffic=steady)

        self.assertEqual(summary["status"], "promoted")
        self.assertEqual(summary["tier3"]["by_monitoring_status"], {"updated": 3})
        self.assertEqual(self._store().load_alerts(), ())

    def test_regression_rolls_back_and_keeps_the_known_good_configuration(self) -> None:
        summary = self._run("regression")

        self.assertEqual(summary["status"], "rolled_back")
        self.assertEqual(summary["tier2"]["by_role"], {"candidate": 4})
        self.assertEqual(summary["tier2"]["state"], "ROLLED_BACK")
        self.assertNotIn("tier3", summary)
        self.assertNotIn("cleanup", summary)
        fallback = summary["fallback_verification"]
        self.assertEqual(fallback["by_role"], {"production": 7})
        self.assertEqual(fallback["by_configuration_hash"], {self.production_hash: 7})

        deployment = self.client.get("/deployment").json()
        self.assertEqual(deployment["state"], "ROLLED_BACK")
        self.assertEqual(deployment["admission"], "rollback_requested")
        self.assertEqual(deployment["serving_configuration_hash"], self.production_hash)
        self.assertEqual(deployment["production_configuration_hash"], self.production_hash)
        self.assertIsNone(deployment["promoted_configuration_hash"])

        store = self._store()
        decisions = store.load_deployment_decisions()
        self.assertEqual([item.decision for item in decisions], ["admit", "rollback"])
        self.assertEqual(decisions[1].method, "stopping_rule_alarm")
        self.assertEqual(decisions[1].reference_configuration_hash, self.production_hash)
        self.assertEqual(store.load_alerts(), ())
        self.assertEqual(
            summary["evidence"]["episodes_by_configuration_hash"],
            {self.production_hash: 11, self.candidate_hash: 4},
        )

        served_before = list(self.counting.hashes)
        closed = self.client.post(
            "/episodes",
            json={"task_id": "dev-canary-0", "mode": "execute", "role": "candidate"},
        )
        self.assertEqual(closed.status_code, 409)
        self.assertEqual(self.counting.hashes, served_before)

    def test_an_incomplete_canary_is_rolled_back_and_known_good_keeps_serving(self) -> None:
        lifecycle = demo.build_synthetic("healthy", self.root)
        short = replace(lifecycle.traffic, canary_task_ids=demo.CANARY_TASK_IDS[:2])
        summary = self._run("healthy", traffic=short)

        self.assertEqual(summary["status"], "canary_incomplete")
        self.assertEqual(summary["tier2"]["by_role"], {"candidate": 2})
        self.assertEqual(summary["tier2"]["state"], "CANARY_ACTIVE")
        self.assertNotIn("tier3", summary)
        self.assertNotIn("fallback_verification", summary)
        self.assertEqual(
            summary["cleanup"],
            {
                "rollback": "manual_rollback",
                "reason": "canary_arrivals_exhausted",
                "state": "ROLLED_BACK",
                "admission": "rollback_requested",
                "serving_configuration_hash": self.production_hash,
            },
        )

        store = self._store()
        decisions = store.load_deployment_decisions()
        self.assertEqual([item.decision for item in decisions], ["admit", "rollback"])
        self.assertEqual(decisions[1].method, "manual_rollback")
        self.assertEqual(decisions[1].sample_size, 2)
        self.assertEqual(
            summary["evidence"]["deployment_decisions"][1]["method"], "manual_rollback"
        )

        deployment = self.client.get("/deployment").json()
        self.assertEqual(deployment["state"], "ROLLED_BACK")
        self.assertEqual(deployment["serving_configuration_hash"], self.production_hash)
        self.assertIsNone(deployment["promoted_configuration_hash"])
        served_before = list(self.counting.hashes)
        closed = self.client.post(
            "/episodes",
            json={"task_id": "dev-canary-2", "mode": "execute", "role": "candidate"},
        )
        self.assertEqual(closed.status_code, 409)
        self.assertEqual(self.counting.hashes, served_before)
        routed = self.client.post(
            "/episodes",
            json={"task_id": "dev-canary-2", "mode": "execute", "assignment_key": "after"},
        )
        self.assertEqual(routed.status_code, 200)
        self.assertEqual(routed.json()["role"], "production")
        self.assertEqual(routed.json()["configuration_hash"], self.production_hash)

    def test_a_canary_execution_failure_rolls_back_then_fails(self) -> None:
        world = demo.build_synthetic("healthy", self.root).dependencies.runtime_factory
        calls = {"n": 0}

        def failing(configuration):
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeUnavailable("injected canary failure")
            return world(configuration)

        with self.assertRaises(ConnectedLifecycleError) as caught:
            self._run("healthy", dependencies={"runtime_factory": failing})
        self.assertIn("injected canary failure", str(caught.exception))
        self.assertNotIn("cleanup rollback failed", str(caught.exception))

        decisions = self._store().load_deployment_decisions()
        self.assertEqual([item.decision for item in decisions], ["admit", "rollback"])
        self.assertEqual(decisions[1].method, "manual_rollback")
        deployment = self.client.get("/deployment").json()
        self.assertEqual(deployment["state"], "ROLLED_BACK")
        self.assertEqual(deployment["admission"], "rollback_requested")
        self.assertEqual(deployment["serving_configuration_hash"], self.production_hash)
        self.assertIsNone(deployment["promoted_configuration_hash"])
        closed = self.client.post(
            "/episodes",
            json={"task_id": "dev-canary-1", "mode": "execute", "role": "candidate"},
        )
        self.assertEqual(closed.status_code, 409)

    def test_a_failed_cleanup_rollback_is_an_error_not_a_safe_completion(self) -> None:
        cases = {
            "refused": lambda real: FakeResponse(503, {"detail": "unavailable"}),
            "unconfirmed": lambda real: FakeResponse(200, real.get("/deployment").json()),
        }
        for name, respond in cases.items():
            with self.subTest(case=name):
                root = self.root / name
                root.mkdir()
                lifecycle = demo.build_synthetic("healthy", root)
                short = replace(lifecycle.traffic, canary_task_ids=demo.CANARY_TASK_IDS[:1])
                with TestClient(create_app(lifecycle.dependencies)) as real:
                    service = RollbackOverride(real, respond)
                    with self.assertRaises(ConnectedLifecycleError) as caught:
                        run_three_tier_dev(
                            lifecycle.gate,
                            short,
                            store_path=lifecycle.store_path,
                            service=service,
                        )
                    self.assertIn("cleanup", str(caught.exception))
                    self.assertEqual(service.rollback_calls, 1)
                    self.assertEqual(real.get("/deployment").json()["state"], "CANARY_ACTIVE")

    def test_fallback_traffic_on_the_candidate_is_an_error(self) -> None:
        lifecycle = demo.build_synthetic("regression", self.root)
        candidate_hash = run_configuration_hash(lifecycle.traffic.candidate)
        with TestClient(create_app(lifecycle.dependencies)) as real:
            service = MisroutedProduction(real, candidate_hash)
            with self.assertRaises(ConnectedLifecycleError) as caught:
                run_three_tier_dev(
                    lifecycle.gate,
                    lifecycle.traffic,
                    store_path=lifecycle.store_path,
                    service=service,
                )
        self.assertIn("ROLLED_BACK", str(caught.exception))

    def test_release_admission_refuses_the_synthetic_pass(self) -> None:
        summary = self._run("healthy", dependencies={"admission_mode": "release"})

        self.assertEqual(summary["tier1"]["outcome"], "PASS")
        self.assertEqual(summary["status"], "admission_refused")
        self.assertEqual(summary["admission"]["status_code"], 409)
        self.assertIn("synthetic_fixture", summary["admission"]["detail"])
        self.assertNotIn("tier2", summary)
        self.assertEqual(self.counting.hashes, [])
        self.assertEqual(self._store().load_deployment_decisions(), ())

    def test_a_required_statistic_the_provider_lacks_fails_closed(self) -> None:
        lifecycle = demo.build_synthetic("healthy", self.root)
        document = json.loads(
            (_ROOT / "configs" / "models" / "glm_5_3_spotify_capability.json").read_text(
                encoding="utf-8"
            )
        )
        hosted = build_run_configuration(
            document,
            TaskConfiguration.from_dict(lifecycle.gate.reference.task.to_dict()),
            run_seed=7,
            git_commit="a" * 40,
        )
        self.assertIn("kl", lifecycle.gate.plan_evidence.required_statistics)
        with self.assertRaises(GateCapabilityError):
            self._run("healthy", gate=replace(lifecycle.gate, candidate=hosted))
        self.assertEqual(self._artifact_count(), 0)
        self.assertEqual(self.counting.hashes, [])

    def test_a_service_serving_other_configurations_is_refused_before_the_gate(self) -> None:
        lifecycle = demo.build_synthetic("healthy", self.root)
        swapped = replace(
            lifecycle.traffic,
            production=lifecycle.traffic.candidate,
            candidate=lifecycle.traffic.production,
        )
        with self.assertRaises(ConnectedLifecycleError):
            self._run("healthy", traffic=swapped)
        self.assertEqual(self._artifact_count(), 0)


def _live_argv(root: Path, **overrides: str) -> list[str]:
    values = {
        "--gate-reference": str(root / "gate_reference.json"),
        "--gate-candidate": str(root / "gate_candidate.json"),
        "--task-set": str(root / "train.json"),
        "--gate-settings": str(root / "gate_settings.json"),
        "--plan-evidence": str(root / "plan_evidence.json"),
        "--production-config": str(root / "production.json"),
        "--candidate-config": str(root / "candidate.json"),
        "--dev-task-set": str(root / "dev.json"),
        "--canary-arrivals": "4",
        "--production-arrivals": "4",
        "--service-url": "http://127.0.0.1:9",
        "--store": str(root / "lifecycle.sqlite"),
    }
    values.update(overrides)
    argv = ["live"]
    for name, value in values.items():
        argv.extend([name, value])
    return argv


class ThreeTierDevCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.root = Path(self._tmpdir.name)

    def _main(self, argv: list[str]) -> tuple[int, str]:
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            code = demo.main(argv)
        return code, stdout.getvalue()

    def test_bare_invocation_and_bad_arguments_exit_2(self) -> None:
        self.assertEqual(demo.main([]), 2)
        with redirect_stderr(io.StringIO()):
            self.assertEqual(demo.main(["live", "--store", "x.sqlite"]), 2)
            self.assertEqual(demo.main(["synthetic", "--scenario", "unknown"]), 2)

    def test_live_exit_code_for_each_lifecycle_status(self) -> None:
        expected = {
            "promoted": 0,
            "blocked": 1,
            "rolled_back": 1,
            "admission_refused": 1,
            "canary_incomplete": 3,
        }
        for status, code in expected.items():
            with self.subTest(status=status):
                with patch.object(demo, "live_inputs", return_value=(None, None)), patch.object(
                    demo, "run_live", return_value={"status": status}
                ) as run:
                    exit_code, output = self._main(_live_argv(self.root))
                self.assertEqual(exit_code, code)
                self.assertEqual(json.loads(output)["status"], status)
                run.assert_called_once()

    def test_live_failures_exit_4_and_print_no_summary(self) -> None:
        for name, effect in {
            "lifecycle": {"side_effect": ConnectedLifecycleError("cleanup failed")},
            "transport": {"side_effect": ConnectionError("refused")},
            "unknown status": {"return_value": {"status": "unexpected"}},
        }.items():
            with self.subTest(case=name):
                with patch.object(demo, "live_inputs", return_value=(None, None)), patch.object(
                    demo, "run_live", **effect
                ):
                    exit_code, output = self._main(_live_argv(self.root))
                self.assertEqual(exit_code, 4)
                self.assertEqual(output, "")

    def test_invalid_live_inputs_exit_2_before_any_model_call(self) -> None:
        results = self.root / "results"
        results.mkdir()
        for name, argv in {
            "missing files": _live_argv(self.root),
            "store under results": _live_argv(
                self.root, **{"--store": str(results / "lifecycle.sqlite")}
            ),
        }.items():
            with self.subTest(case=name):
                with patch.object(demo, "run_live") as run:
                    exit_code, output = self._main(argv)
                self.assertEqual(exit_code, 2)
                self.assertEqual(output, "")
                run.assert_not_called()

    def test_synthetic_scenarios_succeed_as_engineering_tests(self) -> None:
        for scenario, status in demo.EXPECTED_SYNTHETIC_STATUS.items():
            with self.subTest(scenario=scenario):
                exit_code, output = self._main(["synthetic", "--scenario", scenario])
                summary = json.loads(output)
                self.assertEqual(exit_code, 0)
                self.assertEqual(summary["status"], status)
                self.assertTrue(summary["verified"])
                self.assertEqual(summary["provenance"], "synthetic_fixture")

    def test_a_synthetic_scenario_that_misses_its_outcome_exits_1(self) -> None:
        with patch.object(demo, "run_synthetic", return_value={"status": "promoted"}):
            exit_code, output = self._main(["synthetic", "--scenario", "regression"])
        self.assertEqual(exit_code, 1)
        self.assertFalse(json.loads(output)["verified"])


if __name__ == "__main__":
    unittest.main()
