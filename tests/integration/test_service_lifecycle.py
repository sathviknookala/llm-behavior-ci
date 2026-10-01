from __future__ import annotations

import sqlite3
import threading
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
except ImportError as error:
    raise ImportError(
        "fastapi is required for tests.integration.test_service_lifecycle"
    ) from error

from llm_behavior_ci.config import (
    CanarySettings,
    MonitorSettings,
    RunConfiguration,
    StoppingRule,
    new_run_identity,
    run_configuration_hash,
)
from llm_behavior_ci.lifecycle.canary import assign_canary
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
    def __init__(
        self,
        task_id: str = "task-1",
        *,
        success: bool = True,
    ) -> None:
        self.task_id = task_id
        self.evaluation = EvaluationResult(
            success=success,
            passed_requirements=1 if success else 0,
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
            app_name="calendar",
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
        self.endpoint = "unset"

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


class LazyStore(EpisodeStore):
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
        if self._opened:
            super().close()
            self._opened = False


class ServiceLifecycleTests(unittest.TestCase):
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
        self._completion = {"index": 0}
        self.seen_hashes: list[str] = []

    def _metadata_for(self, episode) -> TaskMetadata:
        del episode
        index = self._completion["index"]
        self._completion["index"] = index + 1
        return TaskMetadata(
            signal="task_success",
            completion_index=index,
            task_mix="difficulty:1",
        )

    def _monitor(
        self,
        production: RunConfiguration,
        *,
        dedup_seconds: float = 60.0,
        period_id: str | None = "period-0",
        threshold: float = 0.5,
    ) -> ProductionMonitor:
        digest = run_configuration_hash(production)
        settings = MonitorSettings(
            reference_configuration_hash=digest,
            outcome_delay_seconds=0.0,
            signals=("task_success",),
            stopping_rules=(
                StoppingRule(
                    name="cusum",
                    alpha=0.1,
                    horizon_episodes=50,
                    threshold=threshold,
                ),
            ),
        )
        return ProductionMonitor(
            settings,
            FrozenReference(
                configuration_hash=digest,
                baselines=(("task_success", 0.9),),
            ),
            clock=self.clock,
            dedup_seconds=dedup_seconds,
            period_id=period_id,
        )

    def _execute_runtime(self, *, success: bool = True) -> RuntimeDependencies:
        return RuntimeDependencies(
            session_factory=lambda task_id: FakeSession(
                task_id,
                success=success,
            ),
            agent=FakeAgent(
                [
                    _turn(
                        _ACTION,
                        action=_ACTION,
                        app_name="calendar",
                        api_name="lookup",
                    ),
                    _turn("STOP", action=None),
                ],
                clock=self.clock,
            ),
            clock=self.clock,
        )

    def _gate_body(
        self,
        *,
        outcome: str = "PASS",
        reason_codes: list[str] | None = None,
        reference: RunConfiguration | None = None,
        candidate: RunConfiguration | None = None,
    ) -> dict[str, object]:
        reference = self.production if reference is None else reference
        candidate = self.candidate if candidate is None else candidate
        codes = tuple(reason_codes) if reason_codes is not None else ()
        statistics = (
            (
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
            if outcome == "PASS"
            else ()
        )
        artifact = build_validation_artifact(
            outcome=outcome,
            reason_codes=codes,
            reference=reference,
            candidate=candidate,
            reference_run=new_run_identity(reference),
            candidate_run=new_run_identity(candidate),
            task_set_hash=reference.task.task_set_hash,
            task_split=reference.task.split,
            statistics=statistics,
            evidence_source="gate_run",
            created_at=datetime.now(timezone.utc),
        )
        writer = EpisodeStore(self.store_path)
        try:
            writer.append_validation_artifact(artifact)
        finally:
            writer.close()
        return {"artifact_id": artifact.artifact_id}

    def _dependencies(
        self,
        *,
        candidate: RunConfiguration | None = None,
        canary_settings: CanarySettings | None = None,
        runtime_factory=None,
        monitor: ProductionMonitor | None = None,
        canary_assignment_seed: int = 0,
        fraction: float = 1.0,
        horizon_episodes: int = 10,
        harm_margin: float = 0.1,
        store: EpisodeStore | None = None,
    ):
        chosen_candidate = self.candidate if candidate is None else candidate
        settings = canary_settings or CanarySettings(
            fraction=fraction,
            outcome_delay_seconds=0.0,
            harm_margin=harm_margin,
            stopping_rule=StoppingRule(
                name="fixed_window",
                alpha=0.05,
                horizon_episodes=horizon_episodes,
            ),
            metric_orientation="higher_is_better",
            promotion_policy="horizon_reached_without_harm",
        )
        seen = self.seen_hashes

        def default_factory(config: RunConfiguration) -> RuntimeDependencies:
            digest = run_configuration_hash(config)
            seen.append(digest)
            runtime = self._execute_runtime()
            runtime.agent.endpoint = digest
            return runtime

        return self.ServiceDependencies(
            registry=self.ConfigurationRegistry(
                production=self.production,
                candidate=chosen_candidate,
            ),
            runtime_factory=runtime_factory or default_factory,
            store=self.store if store is None else store,
            monitor=monitor or self._monitor(self.production),
            clock=self.clock,
            max_in_flight=4,
            shutdown_timeout_seconds=1.0,
            metadata_for=self._metadata_for,
            canary_settings=settings,
            canary_assignment_seed=canary_assignment_seed,
        )

    def _client(self, dependencies=None):
        app = self.create_app(dependencies or self._dependencies())
        client = TestClient(app)
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        return client, app

    def test_gate_block_prevents_candidate_admission(self) -> None:
        client, _app = self._client()
        response = client.post(
            "/candidates",
            json=self._gate_body(outcome="BLOCK", reason_codes=["kl"]),
        )
        self.assertEqual(response.status_code, 409)
        deployment = client.get("/deployment").json()
        self.assertNotIn("state", deployment)
        self.assertEqual(deployment["admission"], "open")

    def test_gate_cannot_authorize_changed_candidate(self) -> None:
        approved = self.candidate
        changed_sampling = replace(
            approved,
            agent=replace(
                approved.agent,
                sampling=replace(approved.agent.sampling, temperature=1.0),
            ),
        )
        client, _app = self._client(
            self._dependencies(
                candidate=changed_sampling,
                store=LazyStore(self.store_path),
            )
        )
        response = client.post(
            "/candidates",
            json=self._gate_body(candidate=approved),
        )
        self.assertEqual(response.status_code, 409)
        changed_prompt = replace(
            approved,
            agent=replace(
                approved.agent,
                prompt=replace(
                    approved.agent.prompt,
                    prompt_version="prompt-v3",
                ),
            ),
        )
        client_b, _app_b = self._client(
            self._dependencies(
                candidate=changed_prompt,
                store=LazyStore(self.store_path),
            )
        )
        response_b = client_b.post(
            "/candidates",
            json=self._gate_body(candidate=approved),
        )
        self.assertEqual(response_b.status_code, 409)

    def test_reference_and_candidate_hit_distinct_endpoints(self) -> None:
        agents: dict[str, FakeAgent] = {}

        def factory(config: RunConfiguration) -> RuntimeDependencies:
            digest = run_configuration_hash(config)
            self.seen_hashes.append(digest)
            agent = FakeAgent(
                [
                    _turn(
                        _ACTION,
                        action=_ACTION,
                        app_name="calendar",
                        api_name="lookup",
                    ),
                    _turn("STOP", action=None),
                ],
                clock=self.clock,
            )
            agent.endpoint = f"endpoint://{digest[:8]}"
            agents[digest] = agent
            return RuntimeDependencies(
                session_factory=lambda task_id: FakeSession(task_id),
                agent=agent,
                clock=self.clock,
            )

        client, _app = self._client(
            self._dependencies(runtime_factory=factory, horizon_episodes=5)
        )
        admit = client.post("/candidates", json=self._gate_body())
        self.assertEqual(admit.status_code, 200)
        response = client.post(
            "/episodes",
            json={
                "task_id": "task-1",
                "mode": "execute",
                "assignment_key": "always-canary",
            },
        )
        self.assertEqual(response.status_code, 200)
        production_hash = run_configuration_hash(self.production)
        candidate_hash = run_configuration_hash(self.candidate)
        self.assertIn(production_hash, self.seen_hashes)
        self.assertIn(candidate_hash, self.seen_hashes)
        self.assertNotEqual(
            agents[production_hash].endpoint,
            agents[candidate_hash].endpoint,
        )
        self.assertEqual(response.json()["role"], "candidate")
        self.assertEqual(
            response.json()["configuration_hash"],
            candidate_hash,
        )

    def test_traffic_fraction_selects_without_caller_role(self) -> None:
        seed = 11
        fraction = 0.5
        canary_key = None
        production_key = None
        for index in range(1000):
            key = f"assign-{index}"
            if assign_canary(key, fraction=fraction, seed=seed):
                canary_key = key
            else:
                production_key = key
            if canary_key is not None and production_key is not None:
                break
        self.assertIsNotNone(canary_key)
        self.assertIsNotNone(production_key)
        client, _app = self._client(
            self._dependencies(
                fraction=fraction,
                canary_assignment_seed=seed,
                horizon_episodes=20,
            )
        )
        admit = client.post("/candidates", json=self._gate_body())
        self.assertEqual(admit.status_code, 200)
        canary = client.post(
            "/episodes",
            json={
                "task_id": "task-canary",
                "mode": "execute",
                "assignment_key": canary_key,
            },
        )
        self.assertEqual(canary.status_code, 200)
        self.assertEqual(canary.json()["role"], "candidate")
        production = client.post(
            "/episodes",
            json={
                "task_id": "task-prod",
                "mode": "execute",
                "assignment_key": production_key,
            },
        )
        self.assertEqual(production.status_code, 200)
        self.assertEqual(production.json()["role"], "production")
        self.assertEqual(
            production.json()["configuration_hash"],
            run_configuration_hash(self.production),
        )

    def test_harm_evidence_triggers_rollback_to_known_good(self) -> None:
        production_hash = run_configuration_hash(self.production)
        world_index = {"n": 0}

        def factory(config: RunConfiguration) -> RuntimeDependencies:
            digest = run_configuration_hash(config)
            self.seen_hashes.append(digest)

            def session_factory(task_id: str) -> FakeSession:
                if digest == production_hash:
                    world_index["n"] += 1
                    return FakeSession(
                        task_id,
                        success=world_index["n"] == 1,
                    )
                return FakeSession(task_id, success=False)

            return RuntimeDependencies(
                session_factory=session_factory,
                agent=FakeAgent(
                    [
                        _turn(
                            _ACTION,
                            action=_ACTION,
                            app_name="calendar",
                            api_name="lookup",
                        ),
                        _turn("STOP", action=None),
                    ],
                    clock=self.clock,
                ),
                clock=self.clock,
            )

        client, app = self._client(
            self._dependencies(
                runtime_factory=factory,
                horizon_episodes=1,
                harm_margin=0.1,
            )
        )
        admit = client.post("/candidates", json=self._gate_body())
        self.assertEqual(admit.status_code, 200)
        harm = client.post(
            "/episodes",
            json={
                "task_id": "task-harm",
                "mode": "execute",
                "assignment_key": "canary-harm",
            },
        )
        self.assertEqual(harm.status_code, 200)
        deployment = client.get("/deployment").json()
        self.assertEqual(deployment["state"], "ROLLED_BACK")
        self.assertEqual(deployment["admission"], "rollback_requested")
        follow = client.post(
            "/episodes",
            json={"task_id": "task-after", "mode": "execute"},
        )
        self.assertEqual(follow.status_code, 200)
        self.assertEqual(follow.json()["role"], "production")
        self.assertEqual(
            follow.json()["configuration_hash"],
            run_configuration_hash(self.production),
        )
        self.assertIs(
            app.state.service.serving_configuration,
            self.production,
        )

    def test_promotion_changes_routing_and_resets_monitoring(self) -> None:
        client, app = self._client(
            self._dependencies(horizon_episodes=1, harm_margin=0.1)
        )
        state = app.state.service
        before_period = state.monitor_period_id
        previous_hash = run_configuration_hash(self.production)
        admit = client.post("/candidates", json=self._gate_body())
        self.assertEqual(admit.status_code, 200)
        promote = client.post(
            "/episodes",
            json={
                "task_id": "task-promote",
                "mode": "execute",
                "assignment_key": "canary-promote",
            },
        )
        self.assertEqual(promote.status_code, 200)
        deployment = client.get("/deployment").json()
        self.assertEqual(deployment["state"], "PROMOTED")
        self.assertEqual(
            deployment["promoted_configuration_hash"],
            run_configuration_hash(self.candidate),
        )
        self.assertEqual(
            deployment["previous_production_configuration_hash"],
            previous_hash,
        )
        self.assertNotEqual(state.monitor_period_id, before_period)
        self.assertEqual(
            state.dependencies.monitor.period_id,
            state.monitor_period_id,
        )
        self.assertEqual(
            state.dependencies.monitor.reference.configuration_hash,
            previous_hash,
        )
        follow = client.post(
            "/episodes",
            json={"task_id": "task-served", "mode": "execute"},
        )
        self.assertEqual(follow.status_code, 200)
        self.assertEqual(follow.json()["role"], "production")
        self.assertEqual(
            follow.json()["configuration_hash"],
            run_configuration_hash(self.candidate),
        )
        reader = EpisodeStore(self.store_path)
        self.addCleanup(reader.close)
        decisions = reader.load_deployment_decisions()
        self.assertTrue(
            any(item.decision == "promote" for item in decisions)
        )

    def test_one_persisted_alert_inside_dedup_window(self) -> None:
        def factory(config: RunConfiguration) -> RuntimeDependencies:
            return self._execute_runtime(success=False)

        client, _app = self._client(
            self._dependencies(
                runtime_factory=factory,
                monitor=self._monitor(
                    self.production,
                    dedup_seconds=60.0,
                    period_id=None,
                ),
                fraction=0.01,
                canary_assignment_seed=0,
            )
        )
        for index in range(3):
            response = client.post(
                "/episodes",
                json={
                    "task_id": f"task-fault-{index}",
                    "mode": "execute",
                    "role": "production",
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["monitoring_status"], "updated")
        reader = EpisodeStore(self.store_path)
        self.addCleanup(reader.close)
        alerts = reader.load_alerts()
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].signal, "task_success")
        self.assertNotIn("instruction", response.json())
        self.assertNotIn("plan_text", response.json())

    def test_mid_episode_failure_keeps_written_steps(self) -> None:
        from llm_behavior_ci.service import _EpisodeRequest, _run_production_episode

        class BoomAgent(FakeAgent):
            def next_turn(self, *, tool_output: str | None) -> AgentTurn:
                if self._index == 0:
                    self._index += 1
                    return _turn(
                        _ACTION,
                        action=_ACTION,
                        app_name="calendar",
                        api_name="lookup",
                        started_at=self._clock(),
                    )
                raise EpisodeRejected("mid-episode failure")

        def factory(config: RunConfiguration) -> RuntimeDependencies:
            del config
            return RuntimeDependencies(
                session_factory=lambda task_id: FakeSession(task_id),
                agent=BoomAgent(clock=self.clock),
                clock=self.clock,
            )

        dependencies = self._dependencies(runtime_factory=factory)
        app = self.create_app(dependencies)
        state = app.state.service
        request = _EpisodeRequest(
            task_id="task-open",
            mode="execute",
            role="production",
            assignment_key=None,
        )
        with self.assertRaises(EpisodeRejected):
            _run_production_episode(state, request)
        opened_id = state.dependencies.store._connection.execute(
            "SELECT episode_id FROM episodes WHERE state = 'open'"
        ).fetchone()[0]
        opened = state.dependencies.store.load_open_episode(opened_id)
        self.assertGreaterEqual(len(opened.steps), 1)
        state.close_store_once()

    def test_in_flight_completion_after_rollback_reduces_outstanding(self) -> None:
        from llm_behavior_ci.service import (
            _EpisodeRequest,
            _admit_candidate,
            _parse_candidate_admission,
            _request_rollback,
            _run_candidate_episode,
        )

        started = threading.Event()
        release = threading.Event()

        class BlockingAgent(FakeAgent):
            def next_turn(self, *, tool_output: str | None) -> AgentTurn:
                if self._index == 0:
                    started.set()
                    if not release.wait(timeout=5):
                        raise TimeoutError("release was not set")
                return super().next_turn(tool_output=tool_output)

        def factory(config: RunConfiguration) -> RuntimeDependencies:
            del config
            return RuntimeDependencies(
                session_factory=lambda task_id: FakeSession(task_id),
                agent=BlockingAgent(
                    [
                        _turn(
                            _ACTION,
                            action=_ACTION,
                            app_name="calendar",
                            api_name="lookup",
                        ),
                        _turn("STOP", action=None),
                    ],
                    clock=self.clock,
                ),
                clock=self.clock,
            )

        real_connect = sqlite3.connect

        def connect_any_thread(*args, **kwargs):
            kwargs["check_same_thread"] = False
            return real_connect(*args, **kwargs)

        with patch("sqlite3.connect", side_effect=connect_any_thread):
            dependencies = self._dependencies(
                runtime_factory=factory,
                horizon_episodes=10,
            )
            app = self.create_app(dependencies)
            state = app.state.service
            _admit_candidate(state, _parse_candidate_admission(self._gate_body()))
            errors: list[BaseException] = []
            result: dict[str, object] = {}

            def worker() -> None:
                try:
                    result["receipt"] = _run_candidate_episode(
                        state,
                        _EpisodeRequest(
                            task_id="task-inflight",
                            mode="execute",
                            role="candidate",
                            assignment_key=None,
                        ),
                    )
                except BaseException as error:
                    errors.append(error)

            thread = threading.Thread(target=worker)
            thread.start()
            self.assertTrue(started.wait(timeout=5))
            rollback = _request_rollback(state)
            self.assertEqual(rollback["state"], "ROLLED_BACK")
            outstanding_at_rollback = rollback["outstanding"]
            self.assertGreaterEqual(outstanding_at_rollback, 1)
            release.set()
            thread.join(timeout=5)
            self.assertEqual(errors, [])
            self.assertIn("receipt", result)
            snapshot = state.controller.snapshot()
            self.assertEqual(snapshot.state, "ROLLED_BACK")
            self.assertEqual(snapshot.outstanding, outstanding_at_rollback - 1)
            self.assertGreaterEqual(snapshot.candidate_episodes_served, 1)
            state.close_store_once()


if __name__ == "__main__":
    unittest.main()
