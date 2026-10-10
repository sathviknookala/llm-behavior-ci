from __future__ import annotations

import importlib.util
import io
import json
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from typing import Any
from tempfile import TemporaryDirectory
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
except ImportError as error:
    raise ImportError(
        "fastapi is required for tests.integration.test_production_rollback"
    ) from error

from llm_behavior_ci.config import run_configuration_hash
from llm_behavior_ci.lifecycle.connected import (
    ConnectedLifecycleError,
    ReleaseLineageError,
    release_summary,
    require_release_lineage,
    run_three_tier_dev,
)
from llm_behavior_ci.lifecycle.monitoring import Alert, monitoring_period_id
from llm_behavior_ci.service import (
    PRODUCTION_ROLLED_BACK,
    ServiceError,
    _apply_rollback_policy,
    _request_rollback,
    create_app,
)
from llm_behavior_ci.storage import DeploymentDecisionRecord, EpisodeStore, StorageError

_ROOT = Path(__file__).resolve().parents[2]


def _load_demo():
    path = _ROOT / "scripts" / "demo" / "run_three_tier_dev.py"
    spec = importlib.util.spec_from_file_location("run_three_tier_dev_rollback", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


demo = _load_demo()


class FailingRollbackStore(demo.synthetic.LazyStore):
    """A store whose ``production_rollback`` write fails while ``failing`` is set."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.failing = True

    def append_deployment_decision(self, decision):
        if self.failing and decision.decision == "production_rollback":
            raise StorageError("append_deployment_decision failed")
        return super().append_deployment_decision(decision)


class MisroutedFallback:
    def __init__(self, inner, configuration_hash: str) -> None:
        self._inner = inner
        self._hash = configuration_hash

    def get(self, url: str):
        return self._inner.get(url)

    def post(self, url: str, *, json: object):
        response = self._inner.post(url, json=json)
        state = self._inner.get("/deployment").json().get("state")
        key = json.get("assignment_key", "") if isinstance(json, dict) else ""
        if url == "/episodes" and key.startswith("production:") and state == PRODUCTION_ROLLED_BACK:
            body = {**response.json(), "configuration_hash": self._hash}
            return type(response)(response.status_code, json=body)
        return response


class _Rollback(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.root = Path(self._tmpdir.name)
        self.client: TestClient | None = None
        self.addCleanup(self._close)

    def _close(self) -> None:
        if self.client is not None:
            client, self.client = self.client, None
            client.__exit__(None, None, None)

    def _lifecycle(self, scenario: str = "healthy", **dependency_changes):
        lifecycle = demo.build_synthetic(scenario, self.root)
        self.production_hash = run_configuration_hash(lifecycle.traffic.production)
        self.candidate_hash = run_configuration_hash(lifecycle.traffic.candidate)
        self.store_path = lifecycle.store_path
        self.candidate_period = monitoring_period_id(self.candidate_hash, self.production_hash)
        self.production_period = monitoring_period_id(self.production_hash, self.production_hash)
        dependencies = replace(lifecycle.dependencies, **dependency_changes)
        return replace(lifecycle, dependencies=dependencies)

    def _open(self, dependencies) -> TestClient:
        self._close()
        client = TestClient(create_app(dependencies))
        client.__enter__()
        self.client = client
        return client

    def _run(self, scenario: str = "healthy", *, traffic_changes=None, **dependency_changes):
        lifecycle = self._lifecycle(scenario, **dependency_changes)
        traffic = replace(lifecycle.traffic, **(traffic_changes or {}))
        self._open(lifecycle.dependencies)
        summary = run_three_tier_dev(
            lifecycle.gate, traffic, store_path=lifecycle.store_path, service=self.client
        )
        return summary, replace(lifecycle, traffic=traffic)

    def _restart(self, **dependency_changes) -> TestClient:
        self._close()
        lifecycle = demo.build_synthetic("healthy", self.root)
        return self._open(replace(lifecycle.dependencies, **dependency_changes))

    def _rows(self):
        reader = EpisodeStore(self.store_path)
        try:
            return (
                reader.load_deployment_decisions(),
                reader.load_alerts(),
                reader.load_finished_episodes(),
            )
        finally:
            reader.close()

    def _decisions(self) -> list[str]:
        return [row.decision for row in self._rows()[0]]

    def _arrive(self, client: TestClient, task_id: str, key: str) -> dict[str, object]:
        response = client.post(
            "/episodes", json={"task_id": task_id, "mode": "execute", "assignment_key": key}
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _assert_production_restored(self, client: TestClient) -> dict[str, object]:
        deployment = client.get("/deployment").json()
        self.assertEqual(deployment["state"], PRODUCTION_ROLLED_BACK)
        self.assertEqual(deployment["admission"], "rollback_requested")
        self.assertEqual(deployment["serving_configuration_hash"], self.production_hash)
        self.assertEqual(deployment["production_configuration_hash"], self.production_hash)
        self.assertEqual(deployment["rolled_back_configuration_hash"], self.candidate_hash)
        self.assertEqual(deployment["monitor_period_id"], self.production_period)
        rollback = deployment["production_rollback"]
        self.assertEqual(rollback["restored_configuration_hash"], self.production_hash)
        self.assertEqual(rollback["rolled_back_period_id"], self.candidate_period)
        return deployment

    def _alert(self, **changes) -> Alert:
        values: dict[str, Any] = dict(
            configuration_hash=self.candidate_hash,
            reference_configuration_hash=self.production_hash,
            signal="task_success",
            slice_name="task_success",
            method="cusum",
            estimate=3.0,
            boundary=2.0,
            sample_size=9,
            raised_at=demo.synthetic._START,
            period_id=self.candidate_period,
        )
        values.update(changes)
        return Alert(**values)


class PostPromotionRollbackTests(_Rollback):
    def test_a_healthy_promotion_with_no_alert_keeps_the_candidate_serving(self) -> None:
        summary, _ = self._run(
            traffic_changes={"production_task_ids": demo.STEADY_TASK_IDS},
            post_promotion_rollback="rollback_on_alert",
        )
        self.assertEqual(summary["status"], "promoted")
        self.assertNotIn("tier3_fallback", summary)
        self.assertEqual(summary["tier3"]["by_configuration_hash"], {self.candidate_hash: 3})
        deployment = self.client.get("/deployment").json()
        self.assertEqual(deployment["state"], "PROMOTED")
        self.assertEqual(deployment["serving_configuration_hash"], self.candidate_hash)
        self.assertEqual(self._decisions(), ["admit", "promote"])

    def test_record_only_persists_the_alert_and_keeps_the_candidate_serving(self) -> None:
        summary, _ = self._run()
        self.assertEqual(summary["status"], "promoted")
        self.assertEqual(summary["deployment"]["post_promotion_rollback"], "record_only")
        self.assertEqual(summary["tier3"]["by_configuration_hash"], {self.candidate_hash: 7})
        self.assertEqual(self._decisions(), ["admit", "promote", "alert"])
        self.assertEqual(
            self.client.get("/deployment").json()["serving_configuration_hash"],
            self.candidate_hash,
        )

    def test_a_qualifying_alert_restores_the_previous_production(self) -> None:
        summary, _ = self._run("production_regression")

        self.assertEqual(summary["status"], "production_rolled_back")
        self.assertEqual(summary["tier2"]["state"], "PROMOTED")
        self.assertEqual(summary["tier3"]["by_configuration_hash"], {self.candidate_hash: 6})
        fallback = summary["tier3_fallback"]
        self.assertEqual(fallback["by_configuration_hash"], {self.production_hash: 1})
        self.assertEqual(fallback["by_monitoring_status"], {"updated": 1})
        rollback = summary["production_rollback"]
        self.assertEqual(rollback["method"], "monitor_alert")
        self.assertEqual(rollback["source"], "production_monitor")
        self.assertEqual(rollback["candidate_arrivals_before_rollback"], 6)
        self.assertEqual(rollback["trigger"]["signal"], "task_success")
        self.assertEqual(rollback["trigger"]["method"], "cusum")

        self._assert_production_restored(self.client)
        decisions, alerts, _ = self._rows()
        self.assertEqual(
            [row.decision for row in decisions], ["admit", "promote", "alert", "production_rollback"]
        )
        admit, promote, alert_row, rollback_row = decisions
        self.assertEqual(promote.configuration_hash, self.candidate_hash)
        self.assertEqual(rollback_row.configuration_hash, self.candidate_hash)
        self.assertEqual(rollback_row.reference_configuration_hash, self.production_hash)
        self.assertEqual(rollback_row.slice_name, "production")
        self.assertEqual(rollback_row.method, "monitor_alert")
        self.assertEqual(rollback_row.period_id, self.candidate_period)
        self.assertEqual(rollback_row.trigger_signal, "task_success")
        self.assertEqual(rollback_row.sample_size, alert_row.sample_size)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].period_id, self.candidate_period)

        artifact_id = summary["tier1"]["artifact_id"]
        readmit = self.client.post("/candidates", json={"artifact_id": artifact_id})
        self.assertEqual(readmit.status_code, 409)
        self.assertIn("rolled back after promotion", readmit.json()["detail"])
        candidate = self.client.post(
            "/episodes", json={"task_id": "dev-canary-0", "mode": "execute", "role": "candidate"}
        )
        self.assertEqual(candidate.status_code, 409)

    def test_a_manual_rollback_after_promotion_restores_the_previous_production(self) -> None:
        summary, _ = self._run(traffic_changes={"production_task_ids": demo.STEADY_TASK_IDS})
        self.assertEqual(summary["status"], "promoted")

        response = self.client.post("/deployment/rollback")
        self.assertEqual(response.status_code, 200)
        deployment = self._assert_production_restored(self.client)
        self.assertEqual(response.json(), deployment)
        self.assertEqual(deployment["production_rollback"]["method"], "manual_rollback")
        self.assertEqual(deployment["production_rollback"]["source"], "operator")
        self.assertIsNone(deployment["production_rollback"]["trigger"])
        self.assertEqual(self._decisions(), ["admit", "promote", "production_rollback"])

        receipt = self._arrive(self.client, demo.REGRESS_TASK_IDS[0], "after-manual")
        self.assertEqual(receipt["configuration_hash"], self.production_hash)

    def test_arrivals_after_rollback_are_served_and_monitored_as_production(self) -> None:
        self._run("production_regression")
        _, _, before = self._rows()
        receipts = [
            self._arrive(self.client, task_id, f"after:{index}")
            for index, task_id in enumerate(demo.REGRESS_TASK_IDS)
        ]
        self.assertEqual({item["configuration_hash"] for item in receipts}, {self.production_hash})
        self.assertEqual({item["monitoring_status"] for item in receipts}, {"updated"})
        decisions, alerts, after = self._rows()
        self.assertEqual(len(after) - len(before), len(demo.REGRESS_TASK_IDS))
        self.assertEqual(
            {episode.run.configuration_hash for episode in after[len(before):]},
            {self.production_hash},
        )
        self.assertEqual(len(alerts), 1)
        self.assertEqual(decisions[-1].decision, "production_rollback")

    def test_repeated_alerts_and_rollback_requests_do_not_repeat_the_transition(self) -> None:
        self._run("production_regression")
        decisions_before, alerts_before, _ = self._rows()
        state = self.client.app.state.service
        trigger = state.production_rollback.trigger
        for _ in range(2):
            self.assertEqual(self.client.post("/deployment/rollback").status_code, 200)
            with state.deployment_lock:
                _apply_rollback_policy(state, (trigger, self._alert()))
        decisions_after, alerts_after, _ = self._rows()
        self.assertEqual(decisions_after, decisions_before)
        self.assertEqual(alerts_after, alerts_before)
        deployment = self._assert_production_restored(self.client)
        self.assertEqual(deployment["production_rollback"]["method"], "monitor_alert")

    def test_stale_or_unrelated_alerts_do_not_roll_back(self) -> None:
        self._run(
            traffic_changes={"production_task_ids": demo.STEADY_TASK_IDS},
            post_promotion_rollback="rollback_on_alert",
        )
        state = self.client.app.state.service
        unrelated = {
            "other signal": self._alert(signal="requirement_fraction", slice_name="requirement_fraction"),
            "slice alarm": self._alert(slice_name="difficulty:1"),
            "previous period": self._alert(period_id=self.production_period),
            "other configuration": self._alert(configuration_hash=self.production_hash),
            "other reference": self._alert(reference_configuration_hash="f" * 64),
            "no period": self._alert(period_id=None),
        }
        for name, alert in unrelated.items():
            with self.subTest(case=name):
                with state.deployment_lock:
                    _apply_rollback_policy(state, (alert,))
                self.assertEqual(self.client.get("/deployment").json()["state"], "PROMOTED")
        self.assertEqual(self._decisions(), ["admit", "promote"])

        record_only = replace(state.dependencies, post_promotion_rollback="record_only")
        state.dependencies = record_only
        with state.deployment_lock:
            _apply_rollback_policy(state, (self._alert(),))
        self.assertEqual(self.client.get("/deployment").json()["state"], "PROMOTED")

    def test_an_in_flight_request_finishes_on_its_configuration_without_entering_fallback_monitoring(
        self,
    ) -> None:
        lifecycle = self._lifecycle()
        inner = lifecycle.dependencies.runtime_factory
        holder: dict[str, object] = {}

        def factory(configuration):
            runtime = inner(configuration)
            if holder.pop("roll_back_now", False):
                state = holder["state"]
                holder["in_flight"] = state.in_flight()
                holder["during"] = _request_rollback(state)
            return runtime

        dependencies = replace(lifecycle.dependencies, runtime_factory=factory, max_in_flight=1)
        traffic = replace(lifecycle.traffic, production_task_ids=())
        self._open(dependencies)
        summary = run_three_tier_dev(
            lifecycle.gate, traffic, store_path=lifecycle.store_path, service=self.client
        )
        self.assertEqual(summary["status"], "promoted")
        state = self.client.app.state.service
        holder["state"] = state
        holder["roll_back_now"] = True

        in_flight = self._arrive(self.client, demo.STEADY_TASK_IDS[0], "in-flight")
        self.assertEqual(in_flight["configuration_hash"], self.candidate_hash)
        self.assertEqual(in_flight["monitoring_status"], "superseded")
        self.assertEqual(holder["in_flight"], 1)
        self.assertEqual(holder["during"]["state"], PRODUCTION_ROLLED_BACK)
        self.assertEqual(state.in_flight(), 0)

        after = self._arrive(self.client, demo.STEADY_TASK_IDS[1], "after")
        self.assertEqual(after["configuration_hash"], self.production_hash)
        self.assertEqual(after["monitoring_status"], "updated")

        reader = EpisodeStore(self.store_path)
        try:
            episodes = {
                episode.task.task_id: episode
                for episode in reader.load_finished_episodes()
                if episode.task.task_id in demo.STEADY_TASK_IDS
            }
            self.assertEqual(
                episodes[demo.STEADY_TASK_IDS[0]].run.configuration_hash, self.candidate_hash
            )
            self.assertEqual(
                reader.load_monitor_metadata_for_episode(
                    episodes[demo.STEADY_TASK_IDS[0]].episode.episode_id
                ),
                (),
            )
            self.assertNotEqual(
                reader.load_monitor_metadata_for_episode(
                    episodes[demo.STEADY_TASK_IDS[1]].episode.episode_id
                ),
                (),
            )
        finally:
            reader.close()
        self.assertEqual(self._decisions(), ["admit", "promote", "production_rollback"])


class ProductionRollbackRestartTests(_Rollback):
    def test_a_production_rollback_survives_restart(self) -> None:
        summary, _ = self._run("production_regression")
        decisions_before, alerts_before, _ = self._rows()
        for policy in ("record_only", "rollback_on_alert"):
            with self.subTest(policy=policy):
                client = self._restart(post_promotion_rollback=policy)
                deployment = self._assert_production_restored(client)
                self.assertTrue(deployment["recovered_from_store"])
                self.assertTrue(deployment["production_rollback"]["recovered_from_store"])
                self.assertEqual(deployment["production_rollback"]["method"], "monitor_alert")
                self.assertEqual(deployment["production_rollback"]["trigger"]["method"], "cusum")

                receipt = self._arrive(client, demo.REGRESS_TASK_IDS[0], f"restart:{policy}")
                self.assertEqual(receipt["configuration_hash"], self.production_hash)
                readmit = client.post(
                    "/candidates", json={"artifact_id": summary["tier1"]["artifact_id"]}
                )
                self.assertEqual(readmit.status_code, 409)
                self.assertEqual(client.post("/deployment/rollback").status_code, 200)
                decisions_after, alerts_after, _ = self._rows()
                self.assertEqual(decisions_after, decisions_before)
                self.assertEqual(alerts_after, alerts_before)

    def test_an_alert_persisted_without_its_rollback_is_rolled_back_at_restart(self) -> None:
        store = FailingRollbackStore(self.root / "lifecycle.sqlite")
        lifecycle = self._lifecycle("production_regression", store=store)
        self._open(lifecycle.dependencies)
        with self.assertRaises(ConnectedLifecycleError):
            run_three_tier_dev(
                lifecycle.gate,
                lifecycle.traffic,
                store_path=lifecycle.store_path,
                service=self.client,
            )
        deployment = self.client.get("/deployment").json()
        self.assertEqual(deployment["state"], PRODUCTION_ROLLED_BACK)
        self.assertEqual(deployment["serving_configuration_hash"], self.production_hash)
        self.assertFalse(deployment["production_rollback"]["persisted"])
        self.assertEqual(self._decisions(), ["admit", "promote", "alert"])

        client = self._restart(post_promotion_rollback="rollback_on_alert")
        deployment = self._assert_production_restored(client)
        rollback = deployment["production_rollback"]
        self.assertEqual(rollback["source"], "restart_reconciliation")
        self.assertEqual(rollback["method"], "monitor_alert")
        self.assertTrue(rollback["persisted"])
        self.assertEqual(self._decisions(), ["admit", "promote", "alert", "production_rollback"])
        self.assertEqual(len(self._rows()[1]), 1)
        receipt = self._arrive(client, demo.REGRESS_TASK_IDS[0], "after-restart")
        self.assertEqual(receipt["configuration_hash"], self.production_hash)

        client = self._restart(post_promotion_rollback="rollback_on_alert")
        rollback = client.get("/deployment").json()["production_rollback"]
        self.assertTrue(rollback["recovered_from_store"])
        self.assertEqual(rollback["source"], "restart_reconciliation")
        self.assertEqual(self._decisions(), ["admit", "promote", "alert", "production_rollback"])

    def test_a_failed_rollback_write_is_retried_by_the_next_request(self) -> None:
        store = FailingRollbackStore(self.root / "lifecycle.sqlite")
        lifecycle = self._lifecycle("production_regression", store=store)
        self._open(lifecycle.dependencies)
        with self.assertRaises(ConnectedLifecycleError):
            run_three_tier_dev(
                lifecycle.gate,
                lifecycle.traffic,
                store_path=lifecycle.store_path,
                service=self.client,
            )
        receipt = self._arrive(self.client, demo.STEADY_TASK_IDS[0], "while-unpersisted")
        self.assertEqual(receipt["configuration_hash"], self.production_hash)
        store.failing = False
        for _ in range(2):
            response = self.client.post("/deployment/rollback")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["production_rollback"]["persisted"])
            self.assertEqual(response.json()["production_rollback"]["method"], "monitor_alert")
        self.assertEqual(self._decisions(), ["admit", "promote", "alert", "production_rollback"])

    def test_an_alert_recorded_under_record_only_rolls_back_once_enabled(self) -> None:
        self._run()
        self.assertEqual(self._decisions(), ["admit", "promote", "alert"])
        client = self._restart(post_promotion_rollback="record_only")
        self.assertEqual(client.get("/deployment").json()["state"], "PROMOTED")
        client = self._restart(post_promotion_rollback="rollback_on_alert")
        self._assert_production_restored(client)
        self.assertEqual(self._decisions(), ["admit", "promote", "alert", "production_rollback"])

    def test_restart_ignores_an_alert_from_another_period(self) -> None:
        self._run(traffic_changes={"production_task_ids": demo.STEADY_TASK_IDS})
        writer = EpisodeStore(self.store_path)
        try:
            writer.append_alert(self._alert(period_id=self.production_period).to_record(), dedup_seconds=0.0)
            writer.append_alert(
                self._alert(signal="requirement_fraction", slice_name="requirement_fraction").to_record(),
                dedup_seconds=0.0,
            )
        finally:
            writer.close()
        client = self._restart(post_promotion_rollback="rollback_on_alert")
        deployment = client.get("/deployment").json()
        self.assertEqual(deployment["state"], "PROMOTED")
        self.assertEqual(deployment["serving_configuration_hash"], self.candidate_hash)
        self.assertEqual(self._decisions(), ["admit", "promote"])

    def test_corrupted_rollback_lineage_refuses_startup(self) -> None:
        self._run("production_regression")
        decisions, _, _ = self._rows()
        rollback = decisions[-1]
        corruptions = {
            "other period": replace(rollback, period_id=self.production_period),
            "other configuration": replace(rollback, configuration_hash=self.production_hash),
            "unknown method": replace(rollback, method="mystery"),
            "missing alert signal": replace(rollback, trigger_signal=None),
        }
        for name, corrupted in corruptions.items():
            with self.subTest(case=name):
                self._replace_last_decision(corrupted)
                with self.assertRaises(ServiceError):
                    self._restart()
        self._replace_last_decision(rollback)
        self._append_decision(rollback)
        with self.assertRaises(ServiceError):
            self._restart()

    def _replace_last_decision(self, record: DeploymentDecisionRecord) -> None:
        import sqlite3

        connection = sqlite3.connect(self.store_path)
        try:
            (last,) = connection.execute(
                "SELECT MAX(decision_id) FROM deployment_decisions"
            ).fetchone()
            connection.execute(
                "UPDATE deployment_decisions SET decision_json = ? WHERE decision_id = ?",
                (json.dumps(record.to_dict(), sort_keys=True), last),
            )
            connection.commit()
        finally:
            connection.close()

    def _append_decision(self, record: DeploymentDecisionRecord) -> None:
        writer = EpisodeStore(self.store_path)
        try:
            writer.append_deployment_decision(record)
        finally:
            writer.close()


class ProductionRollbackReleaseTests(_Rollback):
    def test_the_release_summary_reports_the_rollback_and_the_restored_serving(self) -> None:
        summary, lifecycle = self._run("production_regression")
        record = release_summary(summary, lifecycle.traffic)
        self.assertEqual(record["version"], "release-summary-v2")
        self.assertEqual(record["status"], "production_rolled_back")
        self.assertEqual(
            record["configurations"],
            {
                "production": self.production_hash,
                "candidate": self.candidate_hash,
                "serving": self.production_hash,
                "known_good": self.production_hash,
                "frozen_reference": self.production_hash,
                "rolled_back": self.candidate_hash,
            },
        )
        self.assertEqual(
            record["deployment"]["decisions"], ["admit", "promote", "production_rollback"]
        )
        self.assertEqual(record["deployment"]["decision"], "production_rollback")
        self.assertEqual(record["deployment"]["state"], PRODUCTION_ROLLED_BACK)
        self.assertEqual(record["deployment"]["production_rollback"]["method"], "monitor_alert")
        self.assertEqual(record["monitoring"]["tier3_arrivals"], 6)
        self.assertEqual(record["monitoring"]["tier3_fallback_arrivals"], 1)
        self.assertEqual(record["gate"]["outcome"], "PASS")
        self.assertNotIn("dev-", json.dumps(record))

        lineage = require_release_lineage(record, lifecycle.traffic.production)
        self.assertEqual(lineage["previous_status"], "production_rolled_back")
        self.assertEqual(lineage["previous_serving_configuration_hash"], self.production_hash)
        with self.assertRaises(ReleaseLineageError):
            require_release_lineage(record, lifecycle.traffic.candidate)
        with self.assertRaises(ConnectedLifecycleError):
            release_summary({**summary, "status": "promoted"}, lifecycle.traffic)

    def test_a_summary_claiming_a_rollback_the_service_did_not_make_is_refused(self) -> None:
        summary, lifecycle = self._run("regression")
        with self.assertRaises(ConnectedLifecycleError):
            release_summary({**summary, "status": "production_rolled_back"}, lifecycle.traffic)

    def test_fallback_traffic_on_the_rolled_back_candidate_is_an_error(self) -> None:
        lifecycle = self._lifecycle("production_regression")
        real = self._open(lifecycle.dependencies)
        with self.assertRaises(ConnectedLifecycleError) as caught:
            run_three_tier_dev(
                lifecycle.gate,
                lifecycle.traffic,
                store_path=lifecycle.store_path,
                service=MisroutedFallback(real, self.candidate_hash),
            )
        self.assertIn("PRODUCTION_ROLLED_BACK", str(caught.exception))

    def test_a_production_rollback_exits_5_and_writes_its_release_summary(self) -> None:
        traffic = demo.build_synthetic("healthy", self.root).traffic
        path = self.root / "release_summary.json"
        argv = ["live"]
        for name in (
            "--gate-reference",
            "--gate-candidate",
            "--task-set",
            "--gate-settings",
            "--plan-evidence",
            "--production-config",
            "--candidate-config",
            "--dev-task-set",
            "--service-url",
            "--store",
        ):
            argv += [name, str(self.root / name.strip("-"))]
        argv += ["--canary-arrivals", "4", "--production-arrivals", "7"]
        argv += ["--release-summary", str(path)]
        status = {"status": "production_rolled_back"}
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(demo, "live_inputs", return_value=(None, traffic)), patch.object(
            demo, "run_live", return_value=status
        ), patch.object(demo, "release_summary", return_value=status):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = demo.main(argv)
        self.assertEqual(code, 5)
        self.assertEqual(json.loads(path.read_text())["status"], "production_rolled_back")

    def test_the_synthetic_production_regression_is_verified(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            code = demo.main(["synthetic", "--scenario", "production_regression"])
        summary = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertTrue(summary["verified"])
        self.assertEqual(summary["status"], "production_rolled_back")
        self.assertEqual(summary["deployment"]["state"], PRODUCTION_ROLLED_BACK)


if __name__ == "__main__":
    unittest.main()
