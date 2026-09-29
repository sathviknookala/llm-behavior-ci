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


if __name__ == "__main__":
    unittest.main()
