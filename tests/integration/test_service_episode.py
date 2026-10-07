from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    from fastapi.testclient import TestClient
except ImportError as error:
    raise ImportError(
        "fastapi is required for tests.integration.test_service_episode"
    ) from error

from llm_behavior_ci.config import (
    CanarySettings,
    DistributionalMonitorSettings,
    MonitorSettings,
    RunConfiguration,
    StoppingRule,
    run_configuration_hash,
)
from llm_behavior_ci.lifecycle.monitoring import (
    DistributionalMonitor,
    FrozenReference,
    ProductionMonitor,
    TaskMetadata,
)
from llm_behavior_ci.records import TokenLogprob
from llm_behavior_ci.runtime.agent import AgentTurn
from llm_behavior_ci.runtime.appworld import (
    EvaluationResult,
    TaskContext,
    ToolResult,
)
from llm_behavior_ci.runtime.episode import RuntimeDependencies
from llm_behavior_ci.storage import EpisodeStore


class LazyStore(EpisodeStore):
    """EpisodeStore that opens on first use so TestClient's thread owns the connection."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._opened = False
        self._connection = None

    def _ensure(self) -> None:
        if not self._opened:
            EpisodeStore.__init__(self, self._path)
            self._opened = True

    def start_episode(self, identity, run, task_id: str) -> None:
        self._ensure()
        return super().start_episode(identity, run, task_id)

    def append_step(self, episode_id: str, step) -> None:
        self._ensure()
        return super().append_step(episode_id, step)

    def finish_episode(self, episode) -> None:
        self._ensure()
        return super().finish_episode(episode)

    def load_episode(self, episode_id: str):
        self._ensure()
        return super().load_episode(episode_id)

    def load_open_episode(self, episode_id: str):
        self._ensure()
        return super().load_open_episode(episode_id)

    def append_pair(self, pair) -> None:
        self._ensure()
        return super().append_pair(pair)

    def append_alert_with_status(self, alert, *, dedup_seconds: float = 0.0):
        self._ensure()
        return super().append_alert_with_status(
            alert,
            dedup_seconds=dedup_seconds,
        )

    def append_deployment_decision(self, decision):
        self._ensure()
        return super().append_deployment_decision(decision)

    def load_alerts(self, *, signal=None):
        self._ensure()
        return super().load_alerts(signal=signal)

    def load_deployment_decisions(self):
        self._ensure()
        return super().load_deployment_decisions()

    def load_validation_artifact(self, artifact_id: str):
        self._ensure()
        return super().load_validation_artifact(artifact_id)

    def close(self) -> None:
        if self._opened:
            super().close()
            self._opened = False

_START = datetime(2026, 9, 27, 15, 0, tzinfo=timezone.utc)
_LOGPROBS = ((TokenLogprob(token_id=7, logprob=-0.5, rank=0),),)
_PROMPT = "plan the next action"
_ACTION = "calendar.lookup()"


def _payload() -> dict[str, object]:
    return {
        "model": {
            "model": {
                "repository": "Qwen/Qwen3-4B",
                "revision": "0123456789abcdef0123456789abcdef01234567",
            },
            "tokenizer": {
                "repository": "Qwen/Qwen3-4B",
                "revision": "fedcba9876543210fedcba9876543210fedcba98",
            },
            "quantization": {"method": "none"},
            "vllm_version": "0.30.0",
            "serving": {
                "dtype": "bfloat16",
                "max_model_len": 8192,
                "gpu_memory_utilization": 0.9,
                "max_num_seqs": 16,
                "max_num_batched_tokens": 8192,
                "kv_cache_dtype": "bfloat16",
                "enable_prefix_caching": False,
                "enable_chunked_prefill": False,
                "enforce_eager": False,
                "tensor_parallel_size": 1,
                "max_logprobs": 20,
                "batch_invariant": False,
                "sampler_backend": "native",
            },
        },
        "agent": {
            "smolagents_version": "1.22.0",
            "action_interface": "code",
            "prompt": {
                "prompt_version": "prompt-v1",
                "plan_format_version": "plan-v1",
                "thinking_enabled": False,
            },
            "step_limit": 40,
            "sampling": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 20,
                "min_p": 0.0,
                "seed": 17,
                "max_tokens": 512,
            },
        },
        "task": {
            "appworld_version": "0.1.3.post1",
            "split": "train",
            "selection_rule": "deterministic_sample",
            "selection_seed": 20260926,
            "task_count": 50,
            "task_set_hash": "c" * 64,
        },
        "run_seed": 7,
        "git_commit": "a" * 40,
        "protocol_hash": "e" * 64,
    }


def _config() -> RunConfiguration:
    return RunConfiguration.from_dict(_payload())


def _candidate(reference: RunConfiguration) -> RunConfiguration:
    return replace(
        reference,
        agent=replace(
            reference.agent,
            prompt=replace(reference.agent.prompt, prompt_version="prompt-v2"),
        ),
        run_seed=reference.run_seed + 1,
    )


def _clock() -> callable:
    current = _START

    def tick() -> datetime:
        nonlocal current
        value = current
        current = current + timedelta(seconds=1)
        return value

    return tick


def _turn(
    output_text: str,
    *,
    action: str | None,
    app_name: str | None = None,
    api_name: str | None = None,
    prompt_text: str = _PROMPT,
    started_at: datetime = _START,
) -> AgentTurn:
    return AgentTurn(
        prompt_text=prompt_text,
        output_text=output_text,
        top_k_logprobs=_LOGPROBS,
        generated_token_count=len(_LOGPROBS),
        latency_seconds=0.1,
        started_at=started_at,
        action=action,
        app_name=app_name,
        api_name=api_name,
    )


class FakeSession:
    def __init__(self, task_id: str = "task-1") -> None:
        self.task_id = task_id
        self.evaluation = EvaluationResult(
            success=True,
            passed_requirements=1,
            total_requirements=1,
            difficulty=1,
        )

    def initial_state_identity(self) -> str:
        return "same-state"

    def context(self) -> TaskContext:
        return TaskContext(
            task_id=self.task_id,
            instruction="solve the task",
            api_documentation="calendar docs",
        )

    def execute(self, action: str) -> ToolResult:
        del action
        return ToolResult(
            output_text="ok",
            error_message=None,
            recoverable=False,
            app_name=None,
            api_name="lookup",
        )

    def evaluate(self) -> EvaluationResult:
        return self.evaluation

    def close(self) -> None:
        return None


class FakeAgent:
    def __init__(self, turns: list[AgentTurn] | None = None, *, clock=None) -> None:
        self._turns = list(turns or [])
        self._clock = clock
        self._index = 0

    def begin(self, context: TaskContext, config: RunConfiguration) -> None:
        del context, config

    def next_turn(self, *, tool_output: str | None) -> AgentTurn:
        del tool_output
        if self._index >= len(self._turns):
            source = _turn("STOP", action=None)
        else:
            source = self._turns[self._index]
            self._index += 1
        return _turn(
            source.output_text,
            action=source.action,
            app_name=source.app_name,
            api_name=source.api_name,
            prompt_text=source.prompt_text,
            started_at=self._clock(),
        )


class SpyMonitor(ProductionMonitor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.updates: list[object] = []

    def update(self, observation, **kwargs):
        self.updates.append(observation)
        return super().update(observation, **kwargs)


class ServiceEpisodeIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        from llm_behavior_ci.service import (
            ConfigurationRegistry,
            ServiceDependencies,
            create_app,
        )

        self.create_app = create_app
        self.ConfigurationRegistry = ConfigurationRegistry
        self.ServiceDependencies = ServiceDependencies
        self._tmpdir = TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.production = _config()
        self.candidate = _candidate(self.production)
        self.clock = _clock()
        self.store_path = Path(self._tmpdir.name) / "episodes.sqlite"
        self.store = LazyStore(self.store_path)
        digest = run_configuration_hash(self.production)
        settings = MonitorSettings(
            reference_configuration_hash=digest,
            outcome_delay_seconds=0.0,
            signals=("task_success",),
            stopping_rules=(
                StoppingRule(
                    name="cusum",
                    alpha=0.1,
                    horizon_episodes=20,
                    threshold=5.0,
                ),
            ),
        )
        self.monitor = SpyMonitor(
            settings,
            FrozenReference(
                configuration_hash=digest,
                baselines=(("task_success", 0.9),),
            ),
            clock=self.clock,
            dedup_seconds=0.0,
        )
        self._completion = {"index": 0}

    def _metadata_for(self, episode) -> TaskMetadata:
        del episode
        index = self._completion["index"]
        self._completion["index"] = index + 1
        return TaskMetadata(signal="task_success", completion_index=index)

    def _runtime(self) -> RuntimeDependencies:
        return RuntimeDependencies(
            session_factory=lambda task_id: FakeSession(task_id),
            agent=FakeAgent(
                [
                    _turn(_ACTION, action=_ACTION, api_name="lookup"),
                    _turn("STOP", action=None),
                ],
                clock=self.clock,
            ),
            clock=self.clock,
        )

    def _client(self):
        dependencies = self.ServiceDependencies(
            registry=self.ConfigurationRegistry(
                production=self.production,
                candidate=self.candidate,
            ),
            runtime_factory=lambda config: self._runtime(),
            store=self.store,
            monitor=self.monitor,
            clock=self.clock,
            max_in_flight=4,
            shutdown_timeout_seconds=1.0,
            metadata_for=self._metadata_for,
            canary_settings=CanarySettings(
                fraction=1.0,
                outcome_delay_seconds=0.0,
                harm_margin=0.1,
                stopping_rule=StoppingRule(
                    name="fixed_window",
                    alpha=0.05,
                    horizon_episodes=10,
                ),
                metric_orientation="higher_is_better",
                promotion_policy="horizon_reached_without_harm",
            ),
            canary_assignment_seed=0,
        )
        app = self.create_app(dependencies)
        client = TestClient(app)
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        return client

    def test_execute_production_persists_and_monitors(self) -> None:
        client = self._client()
        response = client.post(
            "/episodes",
            json={
                "task_id": "task-1",
                "mode": "execute",
                "role": "production",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["monitoring_status"], "updated")
        self.assertEqual(body["role"], "production")
        self.assertEqual(body["mode"], "execute")
        self.assertNotIn("instruction", body)
        self.assertNotIn("plan_text", body)
        episode_id = body["episode_id"]
        reader = EpisodeStore(self.store_path)
        self.addCleanup(reader.close)
        loaded = reader.load_episode(episode_id)
        self.assertEqual(loaded.task.task_id, "task-1")
        self.assertIsNotNone(loaded.evaluator_outcome)
        self.assertTrue(loaded.evaluator_outcome.success)
        self.assertEqual(len(self.monitor.updates), 1)
        self.assertEqual(
            self.monitor.updates[0].signal,
            "task_success",
        )
        self.assertEqual(self.monitor.updates[0].value, 1.0)


class MultiSignalServiceMonitoringTests(unittest.TestCase):
    """Every configured scalar signal, plus tool_selection, from one episode."""

    def setUp(self) -> None:
        from llm_behavior_ci.service import (
            ConfigurationRegistry,
            ServiceDependencies,
            create_app,
        )

        self.create_app = create_app
        self.ConfigurationRegistry = ConfigurationRegistry
        self.ServiceDependencies = ServiceDependencies
        self._tmpdir = TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.production = _config()
        self.candidate = _candidate(self.production)
        self.clock = _clock()
        self.store_path = Path(self._tmpdir.name) / "episodes.sqlite"
        self.store = LazyStore(self.store_path)
        digest = run_configuration_hash(self.production)
        settings = MonitorSettings(
            reference_configuration_hash=digest,
            outcome_delay_seconds=0.0,
            signals=("task_success", "tool_error_count"),
            stopping_rules=(
                StoppingRule(
                    name="cusum",
                    alpha=0.1,
                    horizon_episodes=20,
                    threshold=5.0,
                ),
            ),
        )
        self.monitor = SpyMonitor(
            settings,
            FrozenReference(
                configuration_hash=digest,
                baselines=(("task_success", 0.9), ("tool_error_count", 0.0)),
            ),
            clock=self.clock,
            dedup_seconds=0.0,
        )
        self.tool_selection_monitor = DistributionalMonitor(
            DistributionalMonitorSettings(
                signal="tool_selection",
                reference_counts=(("lookup", 1),),
                window_episodes=10,
                alpha=0.01,
                correction="none",
            ),
            reference_configuration_hash=digest,
            clock=self.clock,
            dedup_seconds=0.0,
        )

    def _metadata_for(self, episode) -> TaskMetadata:
        del episode
        return TaskMetadata(signal="task_success", completion_index=0)

    def _runtime(self) -> RuntimeDependencies:
        return RuntimeDependencies(
            session_factory=lambda task_id: FakeSession(task_id),
            agent=FakeAgent(
                [
                    _turn(_ACTION, action=_ACTION, api_name="lookup"),
                    _turn("STOP", action=None),
                ],
                clock=self.clock,
            ),
            clock=self.clock,
        )

    def _client(self):
        dependencies = self.ServiceDependencies(
            registry=self.ConfigurationRegistry(
                production=self.production,
                candidate=self.candidate,
            ),
            runtime_factory=lambda config: self._runtime(),
            store=self.store,
            monitor=self.monitor,
            clock=self.clock,
            max_in_flight=4,
            shutdown_timeout_seconds=1.0,
            metadata_for=self._metadata_for,
            canary_settings=CanarySettings(
                fraction=1.0,
                outcome_delay_seconds=0.0,
                harm_margin=0.1,
                stopping_rule=StoppingRule(
                    name="fixed_window",
                    alpha=0.05,
                    horizon_episodes=10,
                ),
                metric_orientation="higher_is_better",
                promotion_policy="horizon_reached_without_harm",
            ),
            canary_assignment_seed=0,
            tool_selection_monitor=self.tool_selection_monitor,
        )
        app = self.create_app(dependencies)
        client = TestClient(app)
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        return client

    def test_every_configured_signal_updates_and_persists_metadata(self) -> None:
        client = self._client()
        response = client.post(
            "/episodes",
            json={
                "task_id": "task-1",
                "mode": "execute",
                "role": "production",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["monitoring_status"], "updated")
        episode_id = body["episode_id"]

        self.assertEqual(len(self.monitor.updates), 2)
        fed_signals = {observation.signal for observation in self.monitor.updates}
        self.assertEqual(fed_signals, {"task_success", "tool_error_count"})

        reader = EpisodeStore(self.store_path)
        self.addCleanup(reader.close)
        persisted = reader.load_monitor_metadata_for_episode(episode_id)
        persisted_signals = {record.signal for record in persisted}
        self.assertEqual(
            persisted_signals,
            {"task_success", "tool_error_count", "tool_selection"},
        )


def _load_serve_module():
    import importlib.util

    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "service"
        / "serve.py"
    )
    spec = importlib.util.spec_from_file_location("serve_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ServeBuiltDependenciesFeedToolSelectionTests(unittest.TestCase):
    """A service built the way ``scripts/service/serve.py`` builds it.

    Exercises ``serve._build_dependencies`` end to end, including the
    distributional-monitor factory wiring, rather than constructing a
    ``DistributionalMonitor`` by hand the way other tests in this module
    do. Only ``runtime_factory`` is swapped afterward for a fake, since a
    real one would dial a live vLLM base URL this suite never starts.
    """

    def setUp(self) -> None:
        self.serve = _load_serve_module()
        self._tmpdir = TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.root = Path(self._tmpdir.name)
        self.clock = _clock()
        self.production = _config()
        self.digest = run_configuration_hash(self.production)

    def _write(self, name: str, document: object) -> Path:
        import json

        path = self.root / name
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def _args(self):
        import argparse

        production_path = self._write("production.json", _payload())
        canary_path = self._write(
            "canary.json",
            CanarySettings(
                fraction=1.0,
                outcome_delay_seconds=0.0,
                harm_margin=0.1,
                stopping_rule=StoppingRule(
                    name="fixed_window",
                    alpha=0.05,
                    horizon_episodes=10,
                ),
                metric_orientation="higher_is_better",
                promotion_policy="horizon_reached_without_harm",
            ).to_dict(),
        )
        monitor_path = self._write(
            "monitor.json",
            MonitorSettings(
                reference_configuration_hash=self.digest,
                outcome_delay_seconds=0.0,
                signals=("task_success",),
                stopping_rules=(
                    StoppingRule(
                        name="cusum",
                        alpha=0.1,
                        horizon_episodes=20,
                        threshold=5.0,
                    ),
                ),
            ).to_dict(),
        )
        frozen_path = self._write(
            "frozen_reference.json",
            {
                "configuration_hash": self.digest,
                "baselines": [["task_success", 0.9]],
            },
        )
        distributional_path = self._write(
            "distributional_monitors.json",
            [
                DistributionalMonitorSettings(
                    signal="tool_selection",
                    reference_counts=(("lookup", 1),),
                    window_episodes=10,
                    alpha=0.01,
                    correction="none",
                ).to_dict()
            ],
        )
        store_path = self.root / "episodes.sqlite"
        return argparse.Namespace(
            production_config=str(production_path),
            candidate_config=None,
            store=str(store_path),
            max_in_flight=4,
            shutdown_timeout_seconds=1.0,
            canary_settings=str(canary_path),
            monitor_settings=str(monitor_path),
            frozen_reference=str(frozen_path),
            dedup_seconds=0.0,
            canary_assignment_seed=0,
            production_base_url="http://production.invalid",
            candidate_base_url="http://candidate.invalid",
            distributional_monitors=str(distributional_path),
            slice_attribution=False,
        ), store_path

    def test_serve_hosted_production_needs_key_not_url(self) -> None:
        import json
        import os
        from unittest.mock import patch

        from llm_behavior_ci.config import ConfigError, TaskConfiguration
        from llm_behavior_ci.experiments.run_config import build_run_configuration

        root = Path(__file__).resolve().parents[2]
        template = json.loads(
            (root / "configs/models/glm_5_3_general_experimental.json").read_text(
                encoding="utf-8"
            )
        )
        hosted = build_run_configuration(
            template,
            TaskConfiguration.from_dict(_payload()["task"]),
            run_seed=7,
            git_commit="a" * 40,
        )
        args, _store_path = self._args()
        digest = run_configuration_hash(hosted)
        args.production_config = str(self._write("hosted.json", hosted.to_dict()))
        monitor = json.loads(Path(args.monitor_settings).read_text(encoding="utf-8"))
        monitor["reference_configuration_hash"] = digest
        args.monitor_settings = str(self._write("hosted_monitor.json", monitor))
        args.frozen_reference = str(
            self._write(
                "hosted_frozen.json",
                {"configuration_hash": digest, "baselines": [["task_success", 0.9]]},
            )
        )
        args.production_base_url = None
        args.candidate_base_url = None
        secret = "zai-test-secret-value-0123456789"
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ConfigError) as caught:
                self.serve._build_dependencies(args)
        self.assertIn("ZAI_API_KEY", str(caught.exception))
        with patch.dict(os.environ, {"ZAI_API_KEY": secret}, clear=True):
            args.production_base_url = "http://production.invalid"
            with self.assertRaises(ConfigError) as given_url:
                self.serve._build_dependencies(args)
            self.assertNotIn(secret, str(given_url.exception))
            args.production_base_url = None
            dependencies = self.serve._build_dependencies(args)
        dependencies.store.close()
        self.assertEqual(dependencies.runtime_factory.routes, ((digest, None),))

    def test_serve_built_service_feeds_tool_selection_observation(self) -> None:
        from dataclasses import replace as dc_replace

        args, store_path = self._args()
        dependencies = self.serve._build_dependencies(args)
        self.assertIsNotNone(dependencies.tool_selection_monitor)
        dependencies.store.close()

        def fake_runtime_factory(config: RunConfiguration) -> RuntimeDependencies:
            del config
            return RuntimeDependencies(
                session_factory=lambda task_id: FakeSession(task_id),
                agent=FakeAgent(
                    [
                        _turn(_ACTION, action=_ACTION, api_name="lookup"),
                        _turn("STOP", action=None),
                    ],
                    clock=self.clock,
                ),
                clock=self.clock,
            )

        dependencies = dc_replace(
            dependencies,
            runtime_factory=fake_runtime_factory,
            store=LazyStore(store_path),
            clock=self.clock,
            metadata_for=lambda episode: TaskMetadata(
                signal="task_success", completion_index=0
            ),
        )
        app = self.serve.create_app(dependencies)
        with TestClient(app) as client:
            response = client.post(
                "/episodes",
                json={
                    "task_id": "task-1",
                    "mode": "execute",
                    "role": "production",
                },
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["monitoring_status"], "updated")

        reader = EpisodeStore(store_path)
        self.addCleanup(reader.close)
        persisted = reader.load_monitor_metadata_for_episode(body["episode_id"])
        persisted_signals = {record.signal for record in persisted}
        self.assertIn("tool_selection", persisted_signals)


_DEV_TASK = {
    "appworld_version": "0.1.3.post1",
    "split": "dev",
    "selection_rule": "deterministic_sample",
    "selection_seed": 20261001,
    "task_count": 20,
    "task_set_hash": "d" * 64,
}


class ServeTrainToDevAdmissionTests(unittest.TestCase):
    """``serve.py --task-selection-allowance`` admits a train gate PASS on dev.

    The gate ran on train configurations; the service serves dev ones that
    differ only in task-selection leaves. Admission must rebuild the gate
    hashes from the allowance, keep every other binding exact, and still
    refuse synthetic evidence in release mode.
    """

    def setUp(self) -> None:
        import json

        from llm_behavior_ci.config import TaskConfiguration
        from llm_behavior_ci.experiments.protocol import (
            task_selection_allowance_document,
            task_selection_allowance_for,
        )

        self.json = json
        self.serve = _load_serve_module()
        self._tmpdir = TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.root = Path(self._tmpdir.name)
        self.clock = _clock()
        self.train_reference = _config()
        self.train_candidate = _candidate(self.train_reference)
        dev_task = TaskConfiguration.from_dict(_DEV_TASK)
        self.dev_reference = replace(self.train_reference, task=dev_task)
        self.dev_candidate = replace(self.train_candidate, task=dev_task)
        self.store_path = self.root / "episodes.sqlite"
        self.allowance_document = task_selection_allowance_document(
            task_selection_allowance_for(self.train_reference)
        )

    def _write(self, name: str, document: object) -> Path:
        path = self.root / name
        path.write_text(self.json.dumps(document), encoding="utf-8")
        return path

    def _record_gate(self, *, evidence_source: str = "gate_run") -> str:
        from llm_behavior_ci.config import new_run_identity
        from llm_behavior_ci.lifecycle.validation_artifact import (
            build_validation_artifact,
        )
        from llm_behavior_ci.records import StatisticalEvidence

        reference = self.train_reference
        candidate = self.train_candidate
        artifact = build_validation_artifact(
            outcome="PASS",
            reason_codes=(),
            reference=reference,
            candidate=candidate,
            reference_run=new_run_identity(reference),
            candidate_run=new_run_identity(candidate),
            task_set_hash=reference.task.task_set_hash,
            task_split=reference.task.split,
            statistics=(
                StatisticalEvidence(
                    method="plan_quality_bootstrap",
                    split=reference.task.split,
                    configuration_hash=run_configuration_hash(candidate),
                    estimate=0.05,
                    sample_size=2,
                    unit="score_delta",
                    reference_configuration_hash=run_configuration_hash(reference),
                ),
            ),
            evidence_source=evidence_source,
            created_at=datetime.now(timezone.utc),
        )
        writer = EpisodeStore(self.store_path)
        try:
            writer.append_validation_artifact(artifact)
        finally:
            writer.close()
        return artifact.artifact_id

    def _args(
        self,
        *,
        production: RunConfiguration | None = None,
        candidate: RunConfiguration | None = None,
        allowance: object | None = None,
        with_candidate: bool = True,
    ):
        import argparse

        production = self.dev_reference if production is None else production
        candidate = self.dev_candidate if candidate is None else candidate
        digest = run_configuration_hash(production)
        canary_path = self._write(
            "canary.json",
            CanarySettings(
                fraction=1.0,
                outcome_delay_seconds=0.0,
                harm_margin=0.1,
                stopping_rule=StoppingRule(
                    name="fixed_window",
                    alpha=0.05,
                    horizon_episodes=10,
                ),
                metric_orientation="higher_is_better",
                promotion_policy="horizon_reached_without_harm",
            ).to_dict(),
        )
        monitor_path = self._write(
            "monitor.json",
            MonitorSettings(
                reference_configuration_hash=digest,
                outcome_delay_seconds=0.0,
                signals=("task_success",),
                stopping_rules=(
                    StoppingRule(
                        name="cusum",
                        alpha=0.1,
                        horizon_episodes=20,
                        threshold=5.0,
                    ),
                ),
            ).to_dict(),
        )
        frozen_path = self._write(
            "frozen_reference.json",
            {"configuration_hash": digest, "baselines": [["task_success", 0.9]]},
        )
        allowance_path = (
            None
            if allowance is None
            else str(self._write("task_selection_allowance.json", allowance))
        )
        return argparse.Namespace(
            production_config=str(self._write("production.json", production.to_dict())),
            candidate_config=(
                str(self._write("candidate.json", candidate.to_dict()))
                if with_candidate
                else None
            ),
            store=str(self.store_path),
            max_in_flight=4,
            shutdown_timeout_seconds=1.0,
            canary_settings=str(canary_path),
            monitor_settings=str(monitor_path),
            frozen_reference=str(frozen_path),
            dedup_seconds=0.0,
            canary_assignment_seed=0,
            production_base_url="http://production.invalid",
            candidate_base_url="http://candidate.invalid",
            distributional_monitors=None,
            slice_attribution=False,
            task_selection_allowance=allowance_path,
        )

    def _admit(self, args, artifact_id: str):
        from dataclasses import replace as dc_replace

        dependencies = self.serve._build_dependencies(args)
        dependencies.store.close()
        dependencies = dc_replace(
            dependencies,
            store=LazyStore(self.store_path),
            clock=self.clock,
        )
        app = self.serve.create_app(dependencies)
        with TestClient(app) as client:
            return client.post("/candidates", json={"artifact_id": artifact_id})

    def test_train_gate_pass_admits_the_dev_candidate(self) -> None:
        artifact_id = self._record_gate()
        args = self._args(allowance=self.allowance_document)
        dependencies = self.serve._build_dependencies(args)
        dependencies.store.close()
        allowance = dependencies.task_selection_allowance
        self.assertIsNotNone(allowance)
        self.assertEqual(
            allowance.train_task_set_hash, self.train_reference.task.task_set_hash
        )
        response = self._admit(args, artifact_id)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["admission"], "open")
        self.assertEqual(
            body["candidate_configuration_hash"],
            run_configuration_hash(self.dev_candidate),
        )
        reader = EpisodeStore(self.store_path)
        self.addCleanup(reader.close)
        decisions = reader.load_deployment_decisions()
        self.assertEqual(decisions[-1].evidence_artifact_id, artifact_id)
        self.assertEqual(decisions[-1].evidence_source, "gate_run")

    def test_without_an_allowance_admission_stays_strict(self) -> None:
        artifact_id = self._record_gate()
        response = self._admit(self._args(), artifact_id)
        self.assertEqual(response.status_code, 409)
        self.assertIn("does not match the gate", response.json()["detail"])

    def test_release_admission_refuses_synthetic_evidence_with_an_allowance(self) -> None:
        artifact_id = self._record_gate(evidence_source="synthetic_fixture")
        response = self._admit(self._args(allowance=self.allowance_document), artifact_id)
        self.assertEqual(response.status_code, 409)
        self.assertIn("synthetic_fixture", response.json()["detail"])

    def test_unrelated_configuration_changes_are_refused(self) -> None:
        artifact_id = self._record_gate()
        sampling = self.dev_candidate.agent.sampling
        changes = {
            "sampling": (
                None,
                replace(
                    self.dev_candidate,
                    agent=replace(
                        self.dev_candidate.agent,
                        sampling=replace(sampling, temperature=0.7),
                    ),
                ),
            ),
            "model": (
                None,
                replace(
                    self.dev_candidate,
                    model=replace(
                        self.dev_candidate.model,
                        model=replace(
                            self.dev_candidate.model.model, revision="1" * 40
                        ),
                    ),
                ),
            ),
            "agent": (
                None,
                replace(
                    self.dev_candidate,
                    agent=replace(self.dev_candidate.agent, step_limit=12),
                ),
            ),
            "source_revision": (
                replace(self.dev_reference, git_commit="b" * 40),
                None,
            ),
            "run_seed": (
                None,
                replace(self.dev_candidate, run_seed=self.dev_candidate.run_seed + 5),
            ),
        }
        for label, (production, candidate) in changes.items():
            with self.subTest(change=label):
                response = self._admit(
                    self._args(
                        production=production,
                        candidate=candidate,
                        allowance=self.allowance_document,
                    ),
                    artifact_id,
                )
                self.assertEqual(response.status_code, 409)
                self.assertIn("does not match the gate", response.json()["detail"])

    def test_protocol_binding_change_is_refused(self) -> None:
        artifact_id = self._record_gate()
        response = self._admit(
            self._args(
                candidate=replace(self.dev_candidate, protocol_hash="f" * 64),
                allowance=self.allowance_document,
            ),
            artifact_id,
        )
        self.assertEqual(response.status_code, 409)
        self.assertIn("protocol hash does not match the gate", response.json()["detail"])

    def test_an_allowance_that_misstates_the_train_selection_is_refused(self) -> None:
        artifact_id = self._record_gate()
        forged = self.json.loads(self.json.dumps(self.allowance_document))
        forged["train_values"]["task.selection_seed"] = 1
        response = self._admit(self._args(allowance=forged), artifact_id)
        self.assertEqual(response.status_code, 409)
        self.assertIn("does not match the gate", response.json()["detail"])

    def test_invalid_allowance_documents_fail_at_startup(self) -> None:
        from llm_behavior_ci.config import ConfigError

        def edited(**changes):
            document = self.json.loads(self.json.dumps(self.allowance_document))
            for key, value in changes.items():
                if value is _DELETE:
                    del document[key]
                else:
                    document[key] = value
            return document

        def edited_values(**values):
            document = self.json.loads(self.json.dumps(self.allowance_document))
            document["train_values"].update(
                {key.replace("__", "."): value for key, value in values.items()}
            )
            return document

        train_only = {
            "version": "task-selection-allowance-v1",
            "allowed_leaves": ["task.task_set_hash"],
            "train_task_set_hash": "c" * 64,
            "train_values": {"task.task_set_hash": "c" * 64},
        }
        cases = {
            "not an object": ["task.split"],
            "unknown key": edited(note="x"),
            "missing key": edited(train_values=_DELETE),
            "wrong version": edited(version="task-selection-allowance-v0"),
            "unknown leaf": edited(
                allowed_leaves=[*self.allowance_document["allowed_leaves"], "agent.step_limit"]
            ),
            "repeated leaf": edited(
                allowed_leaves=[*self.allowance_document["allowed_leaves"], "task.split"]
            ),
            "empty leaves": edited(allowed_leaves=[], train_values={}),
            "values do not match leaves": edited_values(agent__step_limit=3),
            "boolean seed": edited_values(task__selection_seed=True),
            "string count": edited_values(task__task_count="20"),
            "hash mismatch": edited(train_task_set_hash="0" * 64),
            "uppercase hash": edited(train_task_set_hash="C" * 64),
            "train split not restored": edited_values(task__split="dev"),
            "split leaf missing": train_only,
        }
        for label, document in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ConfigError):
                    self.serve._build_dependencies(self._args(allowance=document))

    def test_allowance_requires_a_shared_dev_binding_and_a_candidate(self) -> None:
        from llm_behavior_ci.config import ConfigError, TaskConfiguration

        other_dev = TaskConfiguration.from_dict({**_DEV_TASK, "task_set_hash": "e" * 64})
        train_task = self.train_reference.task
        cases = {
            "no candidate": {"with_candidate": False},
            "production on train": {
                "production": replace(self.dev_reference, task=train_task),
                "candidate": replace(self.dev_candidate, task=train_task),
            },
            "different dev task sets": {
                "candidate": replace(self.dev_candidate, task=other_dev),
            },
        }
        for label, overrides in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ConfigError):
                    self.serve._build_dependencies(
                        self._args(allowance=self.allowance_document, **overrides)
                    )

    def test_cli_writes_the_allowance_from_the_train_configuration(self) -> None:
        import importlib.util
        import io
        from contextlib import redirect_stderr, redirect_stdout

        path = (
            Path(__file__).resolve().parents[2]
            / "scripts/evaluation/build_task_selection_allowance.py"
        )
        spec = importlib.util.spec_from_file_location("allowance_cli", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        train_path = self._write("train.json", self.train_reference.to_dict())
        output = self.root / "allowance.json"
        with redirect_stdout(io.StringIO()):
            code = module.main(["--train-config", str(train_path), "--output", str(output)])
        self.assertEqual(code, 0)
        self.assertEqual(
            self.json.loads(output.read_text(encoding="utf-8")), self.allowance_document
        )
        dev_path = self._write("dev.json", self.dev_reference.to_dict())
        refused = self.root / "refused.json"
        with redirect_stderr(io.StringIO()):
            code = module.main(["--train-config", str(dev_path), "--output", str(refused)])
        self.assertEqual(code, 2)
        self.assertFalse(refused.exists())


_DELETE = object()


if __name__ == "__main__":
    unittest.main()
