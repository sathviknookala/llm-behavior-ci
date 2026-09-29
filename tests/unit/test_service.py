from __future__ import annotations

import threading
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    from fastapi.testclient import TestClient
except ImportError as error:
    raise ImportError(
        "fastapi is required for tests.unit.test_service"
    ) from error

from llm_behavior_ci.config import (
    CanarySettings,
    MonitorSettings,
    RunConfiguration,
    StoppingRule,
    new_run_identity,
    run_configuration_hash,
)
from llm_behavior_ci.lifecycle.monitoring import (
    FrozenReference,
    ProductionMonitor,
    TaskMetadata,
)
from llm_behavior_ci.lifecycle.validation_artifact import build_validation_artifact
from llm_behavior_ci.records import StatisticalEvidence, TokenLogprob
from llm_behavior_ci.runtime.agent import AgentTurn
from llm_behavior_ci.runtime.appworld import (
    EvaluationResult,
    TaskContext,
    ToolResult,
)
from llm_behavior_ci.runtime.episode import (
    EpisodeRejected,
    RuntimeDependencies,
    RuntimeUnavailable,
)
from llm_behavior_ci.storage import EpisodeStore

_START = datetime(2026, 9, 27, 15, 0, tzinfo=timezone.utc)
_LOGPROBS = ((TokenLogprob(token_id=7, logprob=-0.5, rank=0),),)
_PROMPT = "plan the next action"
_PLAN = "1. open the calendar"
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


def _gate_artifact(
    reference: RunConfiguration,
    candidate: RunConfiguration,
    *,
    task_set_hash: str | None = None,
    validation_provenance: str = "validated",
):
    statistics = (
        StatisticalEvidence(
            method="plan_quality_bootstrap",
            split=reference.task.split,
            configuration_hash=run_configuration_hash(candidate),
            estimate=0.05,
            sample_size=2,
            unit="score_delta",
            reference_configuration_hash=run_configuration_hash(reference),
        ),
    )
    return build_validation_artifact(
        outcome="PASS",
        reason_codes=(),
        reference=reference,
        candidate=candidate,
        reference_run=new_run_identity(reference),
        candidate_run=new_run_identity(candidate),
        task_set_hash=task_set_hash or reference.task.task_set_hash,
        task_split=reference.task.split,
        statistics=statistics,
        evidence_source=(
            "synthetic_fixture"
            if validation_provenance == "synthetic_fixture"
            else "gate_run"
        ),
        created_at=datetime.now(timezone.utc),
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
        self.execute_count = 0
        self.evaluate_count = 0
        self.close_count = 0
        self.actions: list[str] = []
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
        self.execute_count += 1
        self.actions.append(action)
        return ToolResult(
            output_text="ok",
            error_message=None,
            recoverable=False,
            app_name=None,
            api_name=None,
        )

    def evaluate(self) -> EvaluationResult:
        self.evaluate_count += 1
        return self.evaluation

    def close(self) -> None:
        self.close_count += 1


class FakeAgent:
    def __init__(self, turns: list[AgentTurn] | None = None, *, clock=None) -> None:
        self._turns = list(turns or [])
        self._clock = clock
        self._index = 0
        self.config: RunConfiguration | None = None
        self.context: TaskContext | None = None

    def begin(self, context: TaskContext, config: RunConfiguration) -> None:
        self.context = context
        self.config = config

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


class SpyStore(EpisodeStore):
    """EpisodeStore that opens on first use so TestClient's thread owns the connection."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._opened = False
        self.close_calls = 0
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

    def append_validation_artifact(self, artifact):
        self._ensure()
        return super().append_validation_artifact(artifact)

    def load_validation_artifact(self, artifact_id: str):
        self._ensure()
        return super().load_validation_artifact(artifact_id)

    def load_alerts(self, *, signal=None):
        self._ensure()
        return super().load_alerts(signal=signal)

    def load_deployment_decisions(self):
        self._ensure()
        return super().load_deployment_decisions()

    def close(self) -> None:
        self.close_calls += 1
        if not self._opened:
            return
        connection = self._connection
        self._connection = None
        self._opened = False
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


class SpyMonitor(ProductionMonitor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.updates: list[object] = []

    def update(self, observation, **kwargs):
        self.updates.append(observation)
        return super().update(observation, **kwargs)


def _canary_settings() -> CanarySettings:
    return CanarySettings(
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
    )


def _monitor(production: RunConfiguration, clock) -> SpyMonitor:
    digest = run_configuration_hash(production)
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
    reference = FrozenReference(
        configuration_hash=digest,
        baselines=(("task_success", 0.9),),
    )
    return SpyMonitor(
        settings,
        reference,
        clock=clock,
        dedup_seconds=0.0,
    )


def _metadata_for(episode) -> TaskMetadata:
    del episode
    return TaskMetadata(signal="task_success", completion_index=0)


def _monitor_with_plan_signals(production: RunConfiguration, clock) -> SpyMonitor:
    digest = run_configuration_hash(production)
    settings = MonitorSettings(
        reference_configuration_hash=digest,
        outcome_delay_seconds=0.0,
        signals=("task_success", "plan_quality_score", "plan_kl_mean_nats"),
        stopping_rules=(
            StoppingRule(
                name="cusum",
                alpha=0.1,
                horizon_episodes=20,
                threshold=5.0,
            ),
        ),
    )
    reference = FrozenReference(
        configuration_hash=digest,
        baselines=(
            ("task_success", 0.9),
            ("plan_quality_score", 0.8),
            ("plan_kl_mean_nats", 0.1),
        ),
    )
    return SpyMonitor(
        settings,
        reference,
        clock=clock,
        dedup_seconds=0.0,
    )


class ServiceUnitTests(unittest.TestCase):
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
        self.store = SpyStore(Path(self._tmpdir.name) / "episodes.sqlite")
        self.monitor = _monitor(self.production, self.clock)
        self.seen_configs: list[RunConfiguration] = []

    def _runtime(self) -> RuntimeDependencies:
        sessions: list[FakeSession] = []

        def factory(task_id: str) -> FakeSession:
            session = FakeSession(task_id)
            sessions.append(session)
            return session

        return RuntimeDependencies(
            session_factory=factory,
            agent=FakeAgent(
                [
                    _turn(_ACTION, action=_ACTION, api_name="lookup"),
                    _turn("STOP", action=None),
                ],
                clock=self.clock,
            ),
            clock=self.clock,
        )

    def _plan_runtime(self) -> RuntimeDependencies:
        return RuntimeDependencies(
            session_factory=lambda task_id: FakeSession(task_id),
            agent=FakeAgent([_turn(_PLAN, action=None)], clock=self.clock),
            clock=self.clock,
        )

    def _dependencies(
        self,
        *,
        runtime_factory=None,
        max_in_flight: int = 4,
        candidate: RunConfiguration | None = None,
        shutdown_timeout_seconds: float = 1.0,
        admission_mode: str = "release",
    ):
        registry = self.ConfigurationRegistry(
            production=self.production,
            candidate=self.candidate if candidate is None else candidate,
        )
        seen = self.seen_configs

        def default_factory(config: RunConfiguration) -> RuntimeDependencies:
            seen.append(config)
            return self._runtime()

        return self.ServiceDependencies(
            registry=registry,
            runtime_factory=runtime_factory or default_factory,
            store=self.store,
            monitor=self.monitor,
            clock=self.clock,
            max_in_flight=max_in_flight,
            shutdown_timeout_seconds=shutdown_timeout_seconds,
            metadata_for=_metadata_for,
            canary_settings=_canary_settings(),
            canary_assignment_seed=0,
            admission_mode=admission_mode,
        )

    def _client(self, dependencies=None):
        app = self.create_app(dependencies or self._dependencies())
        client = TestClient(app)
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        return client

    def _persist_artifact(self, artifact) -> None:
        writer = EpisodeStore(Path(self._tmpdir.name) / "episodes.sqlite")
        try:
            writer.append_validation_artifact(artifact)
        finally:
            writer.close()

    def _read_decisions(self):
        reader = EpisodeStore(Path(self._tmpdir.name) / "episodes.sqlite")
        try:
            return reader.load_deployment_decisions()
        finally:
            reader.close()

    def _dependencies_with_monitor(self, monitor, **overrides):
        registry = self.ConfigurationRegistry(
            production=self.production,
            candidate=self.candidate,
        )
        base = dict(
            registry=registry,
            runtime_factory=lambda config: self._runtime(),
            store=self.store,
            monitor=monitor,
            clock=self.clock,
            max_in_flight=4,
            shutdown_timeout_seconds=1.0,
            metadata_for=_metadata_for,
            canary_settings=_canary_settings(),
            canary_assignment_seed=0,
        )
        base.update(overrides)
        return self.ServiceDependencies(**base)

    def test_plan_signals_reach_detector_when_supplied(self) -> None:
        monitor = _monitor_with_plan_signals(self.production, self.clock)

        def plan_quality_features_for(episode):
            del episode
            return {"requirement_coverage_fraction": 0.75}

        def plan_kl_mean_nats_for(episode):
            del episode
            return 0.2

        dependencies = self._dependencies_with_monitor(
            monitor,
            plan_quality_features_for=plan_quality_features_for,
            plan_kl_mean_nats_for=plan_kl_mean_nats_for,
        )
        with TestClient(self.create_app(dependencies)) as client:
            response = client.post(
                "/episodes",
                json={"task_id": "task-1", "mode": "execute", "role": "production"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["monitoring_status"], "updated")
        by_signal = {observation.signal: observation.value for observation in monitor.updates}
        self.assertEqual(by_signal.get("plan_quality_score"), 0.75)
        self.assertEqual(by_signal.get("plan_kl_mean_nats"), 0.2)

    def test_plan_signals_skip_without_crashing_when_not_supplied(self) -> None:
        monitor = _monitor_with_plan_signals(self.production, self.clock)
        dependencies = self._dependencies_with_monitor(monitor)
        with TestClient(self.create_app(dependencies)) as client:
            response = client.post(
                "/episodes",
                json={"task_id": "task-1", "mode": "execute", "role": "production"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["monitoring_status"], "updated")
        signals_seen = {observation.signal for observation in monitor.updates}
        self.assertNotIn("plan_quality_score", signals_seen)
        self.assertNotIn("plan_kl_mean_nats", signals_seen)

    def test_rejects_extra_model_fields_and_keeps_registry_hash(self) -> None:
        before = run_configuration_hash(self.production)
        client = self._client()
        response = client.post(
            "/episodes",
            json={
                "task_id": "task-1",
                "mode": "plan",
                "role": "production",
                "repository": "other/model",
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(run_configuration_hash(self.production), before)

    def test_rejects_bad_mode(self) -> None:
        client = self._client()
        response = client.post(
            "/episodes",
            json={
                "task_id": "task-1",
                "mode": "train",
                "role": "production",
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_rejects_missing_task_id(self) -> None:
        client = self._client()
        response = client.post(
            "/episodes",
            json={"mode": "plan", "role": "production"},
        )
        self.assertEqual(response.status_code, 422)

    def test_production_uses_registry_configuration(self) -> None:
        before = run_configuration_hash(self.production)

        def factory(config: RunConfiguration) -> RuntimeDependencies:
            self.seen_configs.append(config)
            self.assertIs(config, self.production)
            return self._plan_runtime()

        client = self._client(self._dependencies(runtime_factory=factory))
        response = client.post(
            "/episodes",
            json={
                "task_id": "task-1",
                "mode": "plan",
                "role": "production",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["configuration_hash"], before)
        self.assertEqual(body["role"], "production")
        self.assertNotIn("plan_text", body)
        self.assertNotIn("instruction", body)
        self.assertEqual(len(self.seen_configs), 1)
        self.assertIs(self.seen_configs[0], self.production)

    def test_body_repository_does_not_override_registry(self) -> None:
        client = self._client()
        response = client.post(
            "/episodes",
            json={
                "task_id": "task-1",
                "mode": "plan",
                "role": "production",
                "model": {"repository": "Evil/Model"},
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_candidate_admission_hash_mismatch_returns_409(self) -> None:
        client = self._client()
        mismatched_reference = replace(
            self.production, run_seed=self.production.run_seed + 999
        )
        artifact = _gate_artifact(mismatched_reference, self.candidate)
        self._persist_artifact(artifact)
        response = client.post(
            "/candidates",
            json={"artifact_id": artifact.artifact_id},
        )
        self.assertEqual(response.status_code, 409)
        deployment = client.get("/deployment").json()
        self.assertNotIn("state", deployment)

    def test_candidate_admission_missing_artifact_returns_409(self) -> None:
        client = self._client()
        response = client.post(
            "/candidates",
            json={"artifact_id": "0" * 64},
        )
        self.assertEqual(response.status_code, 409)

    def test_candidate_admission_starts_controller(self) -> None:
        client = self._client()
        artifact = _gate_artifact(self.production, self.candidate)
        self._persist_artifact(artifact)
        response = client.post(
            "/candidates",
            json={"artifact_id": artifact.artifact_id},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["state"], "GATE_PASSED")
        self.assertEqual(body["admission"], "open")
        decisions = self._read_decisions()
        self.assertEqual(decisions[-1].decision, "admit")
        self.assertEqual(decisions[-1].evidence_artifact_id, artifact.artifact_id)
        self.assertEqual(decisions[-1].evidence_source, "gate_run")

    def test_candidate_admission_rejects_synthetic_fixture_evidence(self) -> None:
        client = self._client()
        artifact = _gate_artifact(
            self.production,
            self.candidate,
            validation_provenance="synthetic_fixture",
        )
        self._persist_artifact(artifact)
        response = client.post(
            "/candidates",
            json={"artifact_id": artifact.artifact_id},
        )
        self.assertEqual(response.status_code, 409)

    def test_test_mode_admission_accepts_synthetic_fixture_evidence(self) -> None:
        client = self._client(self._dependencies(admission_mode="test"))
        artifact = _gate_artifact(
            self.production,
            self.candidate,
            validation_provenance="synthetic_fixture",
        )
        self._persist_artifact(artifact)
        response = client.post(
            "/candidates",
            json={"artifact_id": artifact.artifact_id},
        )
        self.assertEqual(response.status_code, 200)
        decisions = self._read_decisions()
        self.assertEqual(decisions[-1].decision, "admit")
        self.assertEqual(decisions[-1].evidence_source, "synthetic_fixture")

    def test_runtime_unavailable_returns_503(self) -> None:
        def factory(config: RunConfiguration) -> RuntimeDependencies:
            del config
            raise RuntimeUnavailable("vllm is down")

        client = self._client(self._dependencies(runtime_factory=factory))
        response = client.post(
            "/episodes",
            json={
                "task_id": "task-1",
                "mode": "plan",
                "role": "production",
            },
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "vllm is down")

    def test_episode_rejected_returns_400(self) -> None:
        def factory(config: RunConfiguration) -> RuntimeDependencies:
            del config

            def boom(task_id: str):
                del task_id
                raise EpisodeRejected("task is not in the set")

            return RuntimeDependencies(
                session_factory=boom,
                agent=FakeAgent([_turn(_PLAN, action=None)], clock=self.clock),
                clock=self.clock,
            )

        client = self._client(self._dependencies(runtime_factory=factory))
        response = client.post(
            "/episodes",
            json={
                "task_id": "task-1",
                "mode": "plan",
                "role": "production",
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "task is not in the set")

    def test_shutdown_closes_store_and_rejects_new_episodes(self) -> None:
        dependencies = self._dependencies()
        app = self.create_app(dependencies)
        with TestClient(app) as client:
            response = client.get("/deployment")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["admission"], "open")
            app.state.service.admission = "shutting_down"
            blocked = client.post(
                "/episodes",
                json={
                    "task_id": "task-1",
                    "mode": "plan",
                    "role": "production",
                },
            )
            self.assertEqual(blocked.status_code, 503)
        self.assertEqual(self.store.close_calls, 1)
        self.assertEqual(app.state.service.admission, "shutting_down")

    def test_rollback_blocks_candidate_allows_production(self) -> None:
        client = self._client()
        artifact = _gate_artifact(self.production, self.candidate)
        self._persist_artifact(artifact)
        admit = client.post(
            "/candidates",
            json={"artifact_id": artifact.artifact_id},
        )
        self.assertEqual(admit.status_code, 200)
        rollback = client.post("/deployment/rollback")
        self.assertEqual(rollback.status_code, 200)
        self.assertEqual(rollback.json()["admission"], "rollback_requested")
        candidate = client.post(
            "/episodes",
            json={
                "task_id": "task-1",
                "mode": "execute",
                "role": "candidate",
            },
        )
        self.assertEqual(candidate.status_code, 409)
        production = client.post(
            "/episodes",
            json={
                "task_id": "task-1",
                "mode": "plan",
                "role": "production",
            },
        )
        self.assertEqual(production.status_code, 200)
        deployment = client.get("/deployment").json()
        self.assertEqual(deployment["admission"], "rollback_requested")

    def test_concurrency_returns_429(self) -> None:
        from llm_behavior_ci.service import _EpisodeRequest, _run_production_episode

        started = threading.Event()
        release = threading.Event()

        class BlockingAgent(FakeAgent):
            def next_turn(self, *, tool_output: str | None) -> AgentTurn:
                started.set()
                if not release.wait(timeout=5):
                    raise TimeoutError("release event was not set")
                return super().next_turn(tool_output=tool_output)

        def factory(config: RunConfiguration) -> RuntimeDependencies:
            del config
            return RuntimeDependencies(
                session_factory=lambda task_id: FakeSession(task_id),
                agent=BlockingAgent(
                    [_turn(_PLAN, action=None)],
                    clock=self.clock,
                ),
                clock=self.clock,
            )

        dependencies = self._dependencies(
            runtime_factory=factory,
            max_in_flight=1,
        )
        app = self.create_app(dependencies)
        errors: list[BaseException] = []

        with TestClient(app) as client:
            state = app.state.service
            request = _EpisodeRequest(
                task_id="task-1",
                mode="plan",
                role="production",
                assignment_key=None,
            )

            def first() -> None:
                if not state.try_acquire_slot():
                    errors.append(RuntimeError("expected a free slot"))
                    return
                try:
                    _run_production_episode(state, request)
                except BaseException as error:
                    errors.append(error)
                finally:
                    state.release_slot()

            worker = threading.Thread(target=first)
            worker.start()
            self.assertTrue(started.wait(timeout=5))
            second = client.post(
                "/episodes",
                json={
                    "task_id": "task-2",
                    "mode": "plan",
                    "role": "production",
                },
            )
            self.assertEqual(second.status_code, 429)
            release.set()
            worker.join(timeout=5)
        self.assertEqual(errors, [])

    def test_dependencies_reject_non_positive_max_in_flight(self) -> None:
        from llm_behavior_ci.service import ServiceError

        with self.assertRaises(ServiceError):
            self._dependencies(max_in_flight=0)


if __name__ == "__main__":
    unittest.main()
