from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from llm_behavior_ci.config import (
    ConfigError,
    DistributionalMonitorSettings,
    EpisodeIdentity,
    RunConfiguration,
    TaskConfiguration,
    new_run_identity,
    run_configuration_hash,
)
from llm_behavior_ci.experiments.power import empirical_harm_power
from llm_behavior_ci.experiments.protocol import (
    ProtocolError,
    ProtocolLock,
    _digest_for_payload,
    protocol_commitment,
)
from llm_behavior_ci.experiments.run_config import build_run_configuration
from llm_behavior_ci.experiments.validation import BaselineOutcome, ValidationError
from llm_behavior_ci.export import AggregateResults, export_public_results
from llm_behavior_ci.lifecycle.monitoring import (
    DistributionalMonitor,
    MissingSliceReference,
    monitoring_period_id,
)
from llm_behavior_ci.lifecycle.offline_gate import GateCapabilityError, require_gate_capabilities
from llm_behavior_ci.records import (
    NamedCount,
    ProviderCall,
    ToolSelectionObservation,
)
from llm_behavior_ci.runtime.episode import RuntimeUnavailable
from llm_behavior_ci.runtime.factory import (
    ConfigurationRoutedRuntimeFactory,
    LiveRuntimeFactory,
    RuntimeFactoryError,
)
from llm_behavior_ci.storage import AlertRecord, EpisodeStore
from llm_behavior_ci.usage import aggregate_usage, pricing_from_dict

_ROOT = Path(__file__).resolve().parents[2]
_KEY = "zai-test-secret-value-0123456789"
_NOW = datetime(2026, 10, 2, 9, 0, tzinfo=timezone.utc)


def _task() -> TaskConfiguration:
    document = json.loads(
        (_ROOT / "configs" / "tasks" / "train_smoke.json").read_text(encoding="utf-8")
    )
    document.pop("scenario_count", None)
    return TaskConfiguration.from_dict(document)


def _config(template: str, *, run_seed: int = 17) -> RunConfiguration:
    document = json.loads(
        (_ROOT / "configs" / "models" / f"{template}.json").read_text(encoding="utf-8")
    )
    return build_run_configuration(document, _task(), run_seed=run_seed, git_commit="a" * 40)


class RoutedFactoryTests(unittest.TestCase):
    def test_hosted_needs_no_url_and_reports_missing_key_without_secret(self) -> None:
        glm = _config("glm_5_3_general_experimental")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeFactoryError) as caught:
                ConfigurationRoutedRuntimeFactory.for_configurations({"production": (glm, None)})
        self.assertIn("ZAI_API_KEY", str(caught.exception))
        with patch.dict(os.environ, {"ZAI_API_KEY": _KEY}, clear=True):
            factory = ConfigurationRoutedRuntimeFactory.for_configurations(
                {"production": (glm, None)}
            )
            with self.assertRaises(RuntimeFactoryError) as given_url:
                ConfigurationRoutedRuntimeFactory.for_configurations(
                    {"production": (glm, "http://127.0.0.1:8000")}
                )
        self.assertNotIn(_KEY, str(given_url.exception))
        self.assertEqual(factory.routes, ((run_configuration_hash(glm), None),))

    def test_vllm_routes_need_urls_and_distinct_servers(self) -> None:
        production = _config("qwen3_4b_production")
        candidate = _config("qwen3_4b_production", run_seed=18)
        with self.assertRaises(RuntimeFactoryError):
            ConfigurationRoutedRuntimeFactory.for_configurations({"production": (production, None)})
        with self.assertRaises(RuntimeFactoryError):
            ConfigurationRoutedRuntimeFactory.for_configurations(
                {
                    "production": (production, "http://127.0.0.1:8000"),
                    "candidate": (candidate, "http://127.0.0.1:8000"),
                }
            )
        factory = ConfigurationRoutedRuntimeFactory.for_configurations(
            {
                "production": (production, "http://127.0.0.1:8000"),
                "candidate": (candidate, "http://127.0.0.1:8001"),
            }
        )
        stranger = _config("qwen3_4b_production", run_seed=19)
        with self.assertRaises(RuntimeUnavailable):
            factory(stranger, mode="execute", role="production")

    def test_mixed_preflight_names_variable_not_value(self) -> None:
        glm = _config("glm_5_3_spotify_capability")
        qwen = _config("qwen3_4b_production")
        factory = LiveRuntimeFactory.from_endpoints(reference="http://127.0.0.1:8000")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeFactoryError) as caught:
                factory.preflight({"reference": qwen, "candidate": glm})
        self.assertIn("ZAI_API_KEY", str(caught.exception))
        with patch.dict(os.environ, {"ZAI_API_KEY": _KEY}, clear=True):
            factory.preflight({"reference": qwen, "candidate": glm})
        with self.assertRaises(RuntimeFactoryError):
            LiveRuntimeFactory.from_endpoints().preflight({"reference": qwen})


class GateCapabilityTests(unittest.TestCase):
    def test_hosted_kl_fails_preflight_and_bootstrap_mmd_passes(self) -> None:
        glm = _config("glm_5_3_spotify_capability")
        qwen = _config("qwen3_4b_production")
        with self.assertRaises(GateCapabilityError):
            require_gate_capabilities(qwen, glm, ("plan_quality", "kl", "mmd"))
        require_gate_capabilities(glm, glm, ("plan_quality", "mmd"))
        require_gate_capabilities(qwen, qwen, ("plan_quality", "kl", "mmd"))


class AlertIncidentTests(unittest.TestCase):
    def _alert(self, method: str, period: str | None) -> AlertRecord:
        return AlertRecord(
            configuration_hash="b" * 64,
            reference_configuration_hash="c" * 64,
            signal="task_success",
            slice_name="all",
            method=method,
            estimate=0.4,
            boundary=0.5,
            sample_size=9,
            raised_at=_NOW,
            period_id=period,
        )

    def test_incident_dedup_survives_reopen_and_new_period_alerts(self) -> None:
        period = monitoring_period_id("b" * 64, "c" * 64)
        self.assertEqual(period, monitoring_period_id("b" * 64, "c" * 64))
        self.assertNotEqual(period, monitoring_period_id("c" * 64, "b" * 64))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.sqlite"
            store = EpisodeStore(path)
            _record, inserted = store.append_alert_with_status(
                self._alert("cusum", period), dedup_seconds=0.0
            )
            store.close()
            self.assertTrue(inserted)
            reopened = EpisodeStore(path)
            existing, again = reopened.append_alert_with_status(
                self._alert("adwin", period), dedup_seconds=0.0
            )
            self.assertFalse(again)
            self.assertEqual(existing.method, "cusum")
            _other, fresh = reopened.append_alert_with_status(
                self._alert("cusum", monitoring_period_id("d" * 64, "c" * 64)),
                dedup_seconds=0.0,
            )
            self.assertTrue(fresh)
            self.assertEqual(len(reopened.load_alerts()), 2)
            reopened.close()


class SliceReferenceTests(unittest.TestCase):
    def _observation(self, token: str) -> ToolSelectionObservation:
        run = new_run_identity(_config("qwen3_4b_production"))
        return ToolSelectionObservation(
            episode=EpisodeIdentity(
                episode_id=f"{run.run_id}.{token * 32}", run_id=run.run_id, pair_id=None
            ),
            run=run,
            split="train",
            counts=(NamedCount(name="mail.send", count=1),),
            observed_at=_NOW,
            completion_index=0,
        )

    def test_absent_slice_is_refused_when_per_slice_counts_exist(self) -> None:
        with self.assertRaises(ConfigError):
            DistributionalMonitorSettings(
                signal="tool_selection",
                reference_counts=(("mail.send", 10),),
                window_episodes=2,
                alpha=0.01,
                correction="none",
                slice_reference_counts=(("difficulty:1", (("mail.send", 5),)),),
            )
        settings = DistributionalMonitorSettings(
            signal="tool_selection",
            reference_counts=(("mail.send", 10),),
            window_episodes=2,
            alpha=0.01,
            correction="none",
            slice_reference_counts=(("difficulty:1", (("mail.send", 5),)),),
            reference_source="dev-baseline-v1",
        )
        self.assertEqual(DistributionalMonitorSettings.from_dict(settings.to_dict()), settings)
        monitor = DistributionalMonitor(
            settings,
            reference_configuration_hash="c" * 64,
            clock=lambda: _NOW,
            dedup_seconds=0.0,
        )
        monitor.update(self._observation("2"), slice_name="difficulty:1")
        with self.assertRaises(MissingSliceReference):
            monitor.update(self._observation("3"), slice_name="difficulty:3")


def _call(**usage: int | None) -> ProviderCall:
    return ProviderCall(
        provider="zai",
        model_id="glm-5.3",
        mode="execute",
        request_index=0,
        attempts=1,
        status="succeeded",
        latency_seconds=0.5,
        **usage,
    )


class _Episode:
    def __init__(self, calls: tuple[ProviderCall, ...]) -> None:
        self.provider_calls = calls


class UsageTests(unittest.TestCase):
    def test_unknown_stays_unknown_and_cost_needs_pricing(self) -> None:
        from llm_behavior_ci import usage as usage_module

        episodes = (
            _Episode((_call(input_tokens=1000, output_tokens=100),)),
            _Episode((_call(input_tokens=500, output_tokens=None),)),
        )
        with patch.object(usage_module, "EpisodeResult", _Episode):
            bare = aggregate_usage(episodes, split="train", configuration_hash="b" * 64)
            pricing = pricing_from_dict(
                {
                    "pricing_version": "test-v1",
                    "currency": "USD",
                    "entries": [
                        {
                            "provider": "zai",
                            "model_id": "glm-5.3",
                            "per_million_tokens": {"input_tokens": 2.0},
                        }
                    ],
                }
            )
            priced = aggregate_usage(
                episodes, split="train", configuration_hash="b" * 64, pricing=pricing
            )
            output_priced = aggregate_usage(
                episodes,
                split="train",
                configuration_hash="b" * 64,
                pricing=replace(
                    pricing,
                    entries=(
                        replace(pricing.entries[0], per_million_tokens=(("output_tokens", 1.0),)),
                    ),
                ),
            )
        (row,) = bare
        self.assertEqual(row.input_tokens, 1500)
        self.assertEqual(row.output_tokens, 100)
        self.assertEqual(row.output_tokens_unknown_requests, 1)
        self.assertIsNone(row.reasoning_tokens)
        self.assertEqual(row.reasoning_tokens_unknown_requests, 2)
        self.assertIsNone(row.cost)
        self.assertAlmostEqual(priced[0].cost, 0.003)
        self.assertEqual(priced[0].pricing_version, "test-v1")
        self.assertIsNone(output_priced[0].cost)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "usage.json"
            export_public_results(
                AggregateResults(aggregates=(), evidence=(), decisions=(), usage=priced),
                output_path=path,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            empty = Path(directory) / "empty.json"
            export_public_results(
                AggregateResults(aggregates=(), evidence=(), decisions=()), output_path=empty
            )
            self.assertNotIn("usage", json.loads(empty.read_text(encoding="utf-8")))
        self.assertEqual(payload["usage"][0]["input_tokens"], 1500)
        self.assertNotIn("task_id", json.dumps(payload))


def _outcome(scenario: str, task: str, repetition: int, success: bool | None) -> BaselineOutcome:
    return BaselineOutcome(
        role="production",
        success=success,
        requirement_fraction=None,
        scenario_id=scenario,
        app="spotify",
        difficulty=1,
        pair_key=f"{task}#r{repetition}",
    )


class EmpiricalPowerTests(unittest.TestCase):
    def _rows(self, *, noisy: bool) -> list[BaselineOutcome]:
        rows = []
        for scenario in range(8):
            for task in range(2):
                for repetition in range(3):
                    success = not (noisy and (scenario + task + repetition) % 3 == 0)
                    rows.append(_outcome(f"s{scenario}", f"t{scenario}-{task}", repetition, success))
        rows.append(_outcome("s0", "t0-0", 9, None))
        return rows

    def test_power_grows_and_null_rate_reflects_repetition_noise(self) -> None:
        report = empirical_harm_power(
            self._rows(noisy=True),
            harm_margin=0.3,
            alpha=0.05,
            sample_sizes=(10, 80),
            simulations=200,
            seed=4,
        )
        small, large = report.rows
        self.assertLess(small.power, large.power)
        self.assertGreater(large.power, 0.8)
        self.assertLess(large.null_rejection_rate, 0.2)
        self.assertEqual(report.task_count, 16)
        self.assertEqual(report.repeated_task_count, 16)
        self.assertEqual(report.to_dict()["normal_approximation"], "advisory")
        quiet = empirical_harm_power(
            self._rows(noisy=False),
            harm_margin=0.3,
            alpha=0.05,
            sample_sizes=(40,),
            simulations=50,
            seed=4,
        )
        self.assertEqual(quiet.rows[0].null_rejection_rate, 0.0)
        again = empirical_harm_power(
            self._rows(noisy=True),
            harm_margin=0.3,
            alpha=0.05,
            sample_sizes=(10, 80),
            simulations=200,
            seed=4,
        )
        self.assertEqual(again, report)

    def test_refuses_one_scenario_and_oversized_margin(self) -> None:
        one = [_outcome("s0", "t0", repetition, True) for repetition in range(3)]
        with self.assertRaises(ValidationError):
            empirical_harm_power(
                one, harm_margin=0.1, alpha=0.05, sample_sizes=(10,), simulations=5, seed=1
            )
        with self.assertRaises(ValidationError):
            empirical_harm_power(
                self._rows(noisy=True),
                harm_margin=0.95,
                alpha=0.05,
                sample_sizes=(10,),
                simulations=5,
                seed=1,
            )


class ProtocolCommitmentTests(unittest.TestCase):
    def test_commitment_is_public_and_binds_sections(self) -> None:
        configuration = _config("glm_5_3_spotify_capability")
        payload = {
            "analysis_version": "draft-v0",
            "configurations": [configuration.to_dict()],
            "task_selections": [{"task_set_hash": "c" * 64}],
            "faults": [{"fault_id": "glm_reasoning_disabled", "version": "glm_reasoning_disabled.v1"}],
            "validation_reports": [{"method": "cusum", "validated": False}],
            "plan_evidence": {"task_id": "secret-task"},
        }
        lock = ProtocolLock(digest=_digest_for_payload(payload), payload=payload)
        commitment = protocol_commitment(lock)
        text = json.dumps(commitment)
        self.assertNotIn("secret-task", text)
        self.assertEqual(commitment["protocol_digest"], lock.digest)
        self.assertEqual(
            commitment["configuration_hashes"], [run_configuration_hash(configuration)]
        )
        self.assertEqual(sorted(commitment["section_digests"]), sorted(payload))
        self.assertEqual(commitment["fault_versions"], ["glm_reasoning_disabled.v1"])
        tampered = ProtocolLock(digest=lock.digest, payload={**payload, "analysis_version": "x"})
        with self.assertRaises(ProtocolError):
            protocol_commitment(tampered)


def _script(relative: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        relative.replace("/", "_").removesuffix(".py"), _ROOT / relative
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(directory: Path, apps: list[str]) -> Path:
    from llm_behavior_ci.tasks.selection import canonical_task_set_bytes, task_set_hash_from_bytes

    tasks = (("task-x1", "scen-x"), ("task-y1", "scen-y"))
    digest = task_set_hash_from_bytes(
        canonical_task_set_bytes(
            appworld_version="0.1.3.post1",
            split="train",
            selection_rule="deterministic_sample",
            selection_seed=4,
            tasks=tasks,
        )
    )
    path = directory / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "appworld_version": "0.1.3.post1",
                "split": "train",
                "selection_rule": "deterministic_sample",
                "selection_seed": 4,
                "task_count": 2,
                "scenario_count": 2,
                "task_ids": [task for task, _ in tasks],
                "scenario_ids": [scenario for _, scenario in tasks],
                "task_set_hash": digest,
                "difficulty_by_task": {"task-x1": 1, "task-y1": 2},
                "required_apps_by_task": {task: apps for task, _ in tasks},
            }
        ),
        encoding="utf-8",
    )
    return path


class BuildConfigurationCliTests(unittest.TestCase):
    def test_builds_without_task_ids_and_refuses_dirty_or_mismatched_access(self) -> None:
        from llm_behavior_ci.runtime.provenance import RepositoryState

        module = _script("scripts/evaluation/build_run_configuration.py")
        head = "b" * 40
        clean = RepositoryState(head=head, dirty_paths=())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _manifest(root, ["gmail", "spotify"])
            empty = root / "tasks"
            empty.mkdir()
            output = root / "config.json"
            arguments = [
                "--template", str(_ROOT / "configs/models/glm_5_3_general_experimental.json"),
                "--task-set", str(manifest),
                "--run-seed", "17",
                "--output", str(output),
                "--committed-tasks-dir", str(empty),
            ]
            from contextlib import redirect_stdout
            from io import StringIO

            buffer = StringIO()
            with redirect_stdout(buffer):
                self.assertEqual(module.main(arguments, repository_state=clean), 0)
            built = RunConfiguration.from_dict(json.loads(output.read_text(encoding="utf-8")))
            self.assertEqual(built.git_commit, head)
            self.assertNotIn("task-x1", buffer.getvalue())
            self.assertNotIn("task-x1", output.read_text(encoding="utf-8"))
            again = root / "again.json"
            with redirect_stdout(StringIO()):
                module.main(arguments[:-4] + ["--output", str(again)] + arguments[-2:], repository_state=clean)
            self.assertEqual(
                run_configuration_hash(built),
                run_configuration_hash(
                    RunConfiguration.from_dict(json.loads(again.read_text(encoding="utf-8")))
                ),
            )
            dirty = RepositoryState(head=head, dirty_paths=("src/llm_behavior_ci/config.py",))
            self.assertEqual(module.main(arguments, repository_state=dirty), 1)
            spotify = list(arguments)
            spotify[1] = str(_ROOT / "configs/models/glm_5_3_spotify_capability.json")
            self.assertEqual(module.main(spotify, repository_state=clean), 1)

    def test_fault_flag_changes_only_the_fault_leaves(self) -> None:
        from contextlib import redirect_stderr, redirect_stdout
        from io import StringIO

        from llm_behavior_ci.experiments.faults import apply_fault, load_fault
        from llm_behavior_ci.runtime.provenance import RepositoryState

        module = _script("scripts/evaluation/build_run_configuration.py")
        clean = RepositoryState(head="b" * 40, dirty_paths=())
        fault_path = _ROOT / "configs/faults/hosted_zai/glm_reasoning_disabled.v1.json"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _manifest(root, ["gmail", "spotify"])
            empty = root / "tasks"
            empty.mkdir()
            base_path = root / "base.json"
            faulted_path = root / "faulted.json"
            common = [
                "--template", str(_ROOT / "configs/models/glm_5_3_general_experimental.json"),
                "--task-set", str(manifest),
                "--run-seed", "17",
                "--committed-tasks-dir", str(empty),
            ]
            with redirect_stdout(StringIO()):
                self.assertEqual(
                    module.main([*common, "--output", str(base_path)], repository_state=clean), 0
                )
                self.assertEqual(
                    module.main(
                        [*common, "--output", str(faulted_path), "--fault", str(fault_path)],
                        repository_state=clean,
                    ),
                    0,
                )
            base = RunConfiguration.from_dict(json.loads(base_path.read_text(encoding="utf-8")))
            faulted = RunConfiguration.from_dict(
                json.loads(faulted_path.read_text(encoding="utf-8"))
            )
            self.assertEqual(
                run_configuration_hash(faulted),
                run_configuration_hash(apply_fault(base, load_fault(fault_path))),
            )
            self.assertNotEqual(run_configuration_hash(faulted), run_configuration_hash(base))
            missing = root / "missing_fault.json"
            with redirect_stderr(StringIO()):
                self.assertEqual(
                    module.main(
                        [*common, "--output", str(root / "x.json"), "--fault", str(missing)],
                        repository_state=clean,
                    ),
                    1,
                )
            self.assertFalse((root / "x.json").exists())


if __name__ == "__main__":
    unittest.main()
