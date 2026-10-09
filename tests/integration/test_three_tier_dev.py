from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

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
            runtime_factory=counting,
            **changes.pop("dependencies", {}),
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
        tier3 = summary["tier3"]
        self.assertEqual(tier3["by_role"], {"production": 7})
        self.assertEqual(tier3["by_configuration_hash"], {self.production_hash: 7})

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


class ThreeTierDevCommandTests(unittest.TestCase):
    def test_bare_invocation_exits_2(self) -> None:
        self.assertEqual(demo.main([]), 2)

    def test_synthetic_command_reports_each_scenario(self) -> None:
        expected = {"blocked": "blocked", "healthy": "promoted", "regression": "rolled_back"}
        for scenario, status in expected.items():
            with self.subTest(scenario=scenario):
                with TemporaryDirectory() as temporary:
                    summary = demo.run_synthetic(scenario, Path(temporary))
                self.assertEqual(summary["status"], status)
                self.assertEqual(summary["provenance"], "synthetic_fixture")


if __name__ == "__main__":
    unittest.main()
