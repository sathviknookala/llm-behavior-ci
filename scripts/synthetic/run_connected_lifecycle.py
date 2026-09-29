"""In-process synthetic connected lifecycle. No network, GPU, or AppWorld."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from llm_behavior_ci.config import (
    CanarySettings,
    GateSettings,
    MonitorSettings,
    RunConfiguration,
    StoppingRule,
    run_configuration_hash,
)
from llm_behavior_ci.export import AggregateResults, export_public_results
from llm_behavior_ci.lifecycle.offline_gate import PlanEvidenceInputs, run_offline_gate
from llm_behavior_ci.lifecycle.monitoring import (
    FrozenReference,
    ProductionMonitor,
    TaskMetadata,
)
from llm_behavior_ci.records import (
    AggregateRecord,
    LifecycleDecision,
    StatisticalEvidence,
    TokenLogprob,
    assert_public_payload,
)
from llm_behavior_ci.runtime.agent import AgentTurn
from llm_behavior_ci.runtime.appworld import EvaluationResult, TaskContext, ToolResult
from llm_behavior_ci.runtime.episode import RuntimeDependencies
from llm_behavior_ci.service import (
    ConfigurationRegistry,
    ServiceDependencies,
    create_app,
)
from llm_behavior_ci.storage import EpisodeStore
from llm_behavior_ci.tasks.plan_specs import task_plan_specs_from_mapping
from llm_behavior_ci.tasks.selection import (
    TaskSet,
    canonical_task_set_bytes,
    task_set_hash_from_bytes,
)

_REPO = Path(__file__).resolve().parents[2]
_PLAN_EVIDENCE = _REPO / "configs" / "fixtures" / "synthetic_plan_evidence.json"
_START = datetime(2026, 9, 27, 15, 0, tzinfo=timezone.utc)
_LOGPROBS = ((TokenLogprob(token_id=7, logprob=-0.5, rank=0),),)
_PROMPT = "plan the next action"
_PLAN = "1. open the calendar"
_ACTION = "calendar.lookup()"
_GIT = "a" * 40
_REVISION = "0123456789abcdef0123456789abcdef01234567"
_TOKENIZER_REVISION = "fedcba9876543210fedcba9876543210fedcba98"


class FakeSession:
    def __init__(self, task_id: str, *, success: bool = True) -> None:
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
    def __init__(self, turns: list[AgentTurn], *, clock) -> None:
        self._turns = list(turns)
        self._clock = clock
        self._index = 0
        self.config = None
        self._context = None

    def begin(self, context: TaskContext, config: RunConfiguration) -> None:
        self._context = context
        self.config = config

    def messages(self, tool_output: str | None = None) -> list[dict[str, str]]:
        del tool_output
        assert self._context is not None
        return [
            {"role": "system", "content": "Emit a plan."},
            {
                "role": "user",
                "content": (
                    f"{self._context.instruction}\n"
                    f"{self._context.api_documentation}"
                ),
            },
        ]

    def next_turn(self, *, tool_output: str | None) -> AgentTurn:
        del tool_output
        if self._index >= len(self._turns):
            output_text = "STOP"
            action = None
            app_name = None
            api_name = None
        else:
            source = self._turns[self._index]
            self._index += 1
            output_text = source.output_text
            action = source.action
            app_name = source.app_name
            api_name = source.api_name
        return AgentTurn(
            prompt_text=_PROMPT,
            output_text=output_text,
            top_k_logprobs=_LOGPROBS,
            latency_seconds=0.1,
            started_at=self._clock(),
            action=action,
            app_name=app_name,
            api_name=api_name,
        )

    def teacher_force_plan(
        self,
        *,
        messages: list[dict[str, str]],
        plan_text: str,
    ) -> tuple[tuple[TokenLogprob, ...], ...]:
        del messages, plan_text
        return _LOGPROBS


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


def _clock():
    current = _START

    def tick() -> datetime:
        nonlocal current
        value = current
        current = current + timedelta(seconds=1)
        return value

    return tick


def _task_set() -> TaskSet:
    tasks = (("task-a", "scenario-1"), ("task-b", None))
    payload = canonical_task_set_bytes(
        appworld_version="0.1.3.post1",
        split="train",
        selection_rule="deterministic_sample",
        selection_seed=20260926,
        tasks=tasks,
    )
    return TaskSet(
        appworld_version="0.1.3.post1",
        split="train",
        selection_rule="deterministic_sample",
        selection_seed=20260926,
        task_count=2,
        scenario_count=2,
        task_ids=("task-a", "task-b"),
        scenario_ids=("scenario-1", None),
        task_set_hash=task_set_hash_from_bytes(payload),
    )


def _config(task_set: TaskSet, *, run_seed: int = 7) -> RunConfiguration:
    return RunConfiguration.from_dict(
        {
            "model": {
                "model": {
                    "repository": "Qwen/Qwen3-4B",
                    "revision": _REVISION,
                },
                "tokenizer": {
                    "repository": "Qwen/Qwen3-4B",
                    "revision": _TOKENIZER_REVISION,
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
                "appworld_version": task_set.appworld_version,
                "split": task_set.split,
                "selection_rule": task_set.selection_rule,
                "selection_seed": task_set.selection_seed,
                "task_count": task_set.task_count,
                "task_set_hash": task_set.task_set_hash,
            },
            "run_seed": run_seed,
            "git_commit": _GIT,
            "protocol_hash": "e" * 64,
        }
    )


def _candidate(reference: RunConfiguration) -> RunConfiguration:
    return replace(
        reference,
        agent=replace(
            reference.agent,
            prompt=replace(reference.agent.prompt, prompt_version="prompt-v2"),
        ),
        run_seed=reference.run_seed + 1,
    )


def _plan_evidence() -> PlanEvidenceInputs:
    payload = json.loads(_PLAN_EVIDENCE.read_text(encoding="utf-8"))
    if payload.get("validation_provenance") != "synthetic_fixture":
        raise RuntimeError("synthetic plan evidence must use synthetic_fixture")
    task_plan_specs = task_plan_specs_from_mapping(payload.get("task_plan_specs", []))
    return PlanEvidenceInputs(
        plan_format_version=str(payload["plan_format_version"]),
        plan_quality_features=tuple(payload["plan_quality_features"]),
        plan_quality_weights=tuple(float(item) for item in payload["plan_quality_weights"]),
        mmd_features=tuple(payload["mmd_features"]),
        kl_approximation=str(payload["kl_approximation"]),  # type: ignore[arg-type]
        required_statistics=tuple(payload["required_statistics"]),
        validation_provenance=str(payload["validation_provenance"]),
        task_plan_specs=task_plan_specs,
    )


class PlanAgent:
    def __init__(self, clock) -> None:
        self._clock = clock
        self._context = None
        self.config = None

    def begin(self, context: TaskContext, config: RunConfiguration) -> None:
        self._context = context
        self.config = config

    def messages(self, tool_output: str | None = None) -> list[dict[str, str]]:
        del tool_output
        assert self._context is not None
        return [
            {"role": "system", "content": "Emit a plan."},
            {
                "role": "user",
                "content": (
                    f"{self._context.instruction}\n"
                    f"{self._context.api_documentation}"
                ),
            },
        ]

    def next_turn(self, *, tool_output: str | None) -> AgentTurn:
        del tool_output
        return AgentTurn(
            prompt_text=_PROMPT,
            output_text=_PLAN,
            top_k_logprobs=_LOGPROBS,
            latency_seconds=0.1,
            started_at=self._clock(),
            action=None,
            app_name=None,
            api_name=None,
        )

    def teacher_force_plan(
        self,
        *,
        messages: list[dict[str, str]],
        plan_text: str,
    ) -> tuple[tuple[TokenLogprob, ...], ...]:
        del messages, plan_text
        return _LOGPROBS


def _plan_runtime(clock) -> RuntimeDependencies:
    return RuntimeDependencies(
        session_factory=lambda task_id: FakeSession(task_id),
        agent=PlanAgent(clock),
        clock=clock,
    )


def _execute_runtime(clock, *, success: bool = True) -> RuntimeDependencies:
    return RuntimeDependencies(
        session_factory=lambda task_id: FakeSession(task_id, success=success),
        agent=FakeAgent(
            [
                AgentTurn(
                    prompt_text=_PROMPT,
                    output_text=_ACTION,
                    top_k_logprobs=_LOGPROBS,
                    latency_seconds=0.1,
                    started_at=_START,
                    action=_ACTION,
                    app_name="calendar",
                    api_name="lookup",
                ),
                AgentTurn(
                    prompt_text=_PROMPT,
                    output_text="STOP",
                    top_k_logprobs=_LOGPROBS,
                    latency_seconds=0.1,
                    started_at=_START,
                    action=None,
                    app_name=None,
                    api_name=None,
                ),
            ],
            clock=clock,
        ),
        clock=clock,
    )


def _persist_artifact(path: Path, artifact) -> dict[str, object]:
    """Write a real gate's ``ValidationArtifact`` into the store a client reads.

    Uses a short-lived writer connection to the same SQLite file rather
    than the client's own (lazily opened, thread-bound) store object, so
    the write always lands before the app's portal thread ever touches
    that file. Returns the ``/candidates`` request body: just the content
    hash, never the artifact's fields, mirroring what a real caller sends.
    """

    writer = EpisodeStore(path)
    try:
        writer.append_validation_artifact(artifact)
    finally:
        writer.close()
    return {"artifact_id": artifact.artifact_id}


def _dependencies(
    *,
    production: RunConfiguration,
    candidate: RunConfiguration,
    store: EpisodeStore,
    clock,
    runtime_factory,
    monitor: ProductionMonitor,
    canary_assignment_seed: int = 0,
    fraction: float = 1.0,
    horizon_episodes: int = 10,
    harm_margin: float = 0.1,
) -> ServiceDependencies:
    settings = CanarySettings(
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
    completion = {"index": 0}

    def metadata_for(episode) -> TaskMetadata:
        del episode
        index = completion["index"]
        completion["index"] = index + 1
        return TaskMetadata(signal="task_success", completion_index=index)

    return ServiceDependencies(
        registry=ConfigurationRegistry(
            production=production,
            candidate=candidate,
        ),
        runtime_factory=runtime_factory,
        store=store,
        monitor=monitor,
        clock=clock,
        max_in_flight=4,
        shutdown_timeout_seconds=1.0,
        metadata_for=metadata_for,
        canary_settings=settings,
        canary_assignment_seed=canary_assignment_seed,
    )


def _monitor(
    production: RunConfiguration,
    clock,
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
        clock=clock,
        dedup_seconds=dedup_seconds,
        period_id=period_id,
    )


def main() -> int:
    if not _PLAN_EVIDENCE.is_file():
        print("missing synthetic plan evidence fixture", file=sys.stderr)
        return 1
    evidence = _plan_evidence()
    task_set = _task_set()
    production = _config(task_set, run_seed=7)
    candidate = _candidate(production)

    blocked = run_offline_gate(
        production,
        candidate,
        task_set,
        settings=GateSettings(
            confidence_level=0.9,
            bootstrap_resamples=40,
            score_margin=0.01,
            kl_limit_nats=0.05,
            mmd_bandwidth=1.0,
            mmd_permutations=19,
            mmd_alpha=0.05,
            plan_format_version="plan-v1",
        ),
        runtime=_plan_runtime(_clock()),
        plan_evidence=evidence,
    )
    if blocked.outcome != "BLOCK":
        print("expected synthetic train gate BLOCK", file=sys.stderr)
        return 1

    passed = run_offline_gate(
        production,
        candidate,
        task_set,
        settings=GateSettings(
            confidence_level=0.9,
            bootstrap_resamples=40,
            score_margin=-0.02,
            kl_limit_nats=0.05,
            mmd_bandwidth=1.0,
            mmd_permutations=19,
            mmd_alpha=0.05,
            plan_format_version="plan-v1",
        ),
        runtime=_plan_runtime(_clock()),
        plan_evidence=evidence,
    )
    if passed.outcome != "PASS":
        print("expected synthetic train gate PASS shape", file=sys.stderr)
        return 1
    if passed.artifact.evidence_source != "synthetic_fixture":
        print("expected synthetic_fixture evidence_source", file=sys.stderr)
        return 1

    real_evidence = replace(evidence, validation_provenance="connected_lifecycle_demo")
    real_passed = run_offline_gate(
        production,
        candidate,
        task_set,
        settings=GateSettings(
            confidence_level=0.9,
            bootstrap_resamples=40,
            score_margin=-0.02,
            kl_limit_nats=0.05,
            mmd_bandwidth=1.0,
            mmd_permutations=19,
            mmd_alpha=0.05,
            plan_format_version="plan-v1",
        ),
        runtime=_plan_runtime(_clock()),
        plan_evidence=real_evidence,
    )
    if real_passed.outcome != "PASS":
        print("expected non-synthetic train gate PASS shape", file=sys.stderr)
        return 1
    if real_passed.artifact.evidence_source != "gate_run":
        print("expected gate_run evidence_source", file=sys.stderr)
        return 1

    summary: dict[str, object] = {
        "provenance": "synthetic_fixture",
        "gate_block_outcome": blocked.outcome,
        "gate_pass_outcome": passed.outcome,
        "candidate_configuration_hash": passed.candidate_configuration_hash,
        "reference_configuration_hash": passed.reference_configuration_hash,
    }

    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        clock = _clock()
        store = LazyStore(root / "episodes.sqlite")

        def factory(config: RunConfiguration) -> RuntimeDependencies:
            del config
            return _execute_runtime(clock)

        client = TestClient(
            create_app(
                _dependencies(
                    production=production,
                    candidate=candidate,
                    store=store,
                    clock=clock,
                    runtime_factory=factory,
                    monitor=_monitor(production, clock),
                )
            )
        )
        with client:
            block_admit = client.post(
                "/candidates",
                json=_persist_artifact(root / "episodes.sqlite", blocked.artifact),
            )
            if block_admit.status_code != 409:
                print("BLOCK gate must not admit", file=sys.stderr)
                return 1
            summary["block_admission_status"] = block_admit.status_code

            synthetic_admit = client.post(
                "/candidates",
                json=_persist_artifact(root / "episodes.sqlite", passed.artifact),
            )
            if synthetic_admit.status_code != 409:
                print(
                    "synthetic_fixture PASS evidence must not admit",
                    file=sys.stderr,
                )
                return 1
            summary["synthetic_fixture_admission_status"] = synthetic_admit.status_code

            missing_admit = client.post(
                "/candidates",
                json={"artifact_id": "0" * 64},
            )
            if missing_admit.status_code != 409:
                print("missing artifact must not admit", file=sys.stderr)
                return 1
            summary["missing_artifact_admission_status"] = missing_admit.status_code

            changed = replace(
                candidate,
                agent=replace(
                    candidate.agent,
                    sampling=replace(candidate.agent.sampling, temperature=1.0),
                ),
            )
            changed_store_path = root / "changed.sqlite"
            changed_client = TestClient(
                create_app(
                    _dependencies(
                        production=production,
                        candidate=changed,
                        store=LazyStore(changed_store_path),
                        clock=clock,
                        runtime_factory=factory,
                        monitor=_monitor(production, clock),
                    )
                )
            )
            with changed_client:
                reject_changed = changed_client.post(
                    "/candidates",
                    json=_persist_artifact(changed_store_path, real_passed.artifact),
                )
            if reject_changed.status_code != 409:
                print("PASS gate must not admit a changed candidate", file=sys.stderr)
                return 1
            summary["changed_candidate_status"] = reject_changed.status_code

            production_hash = run_configuration_hash(production)
            world_index = {"n": 0}

            def harm_factory(config: RunConfiguration) -> RuntimeDependencies:
                digest = run_configuration_hash(config)

                def session_factory(task_id: str) -> FakeSession:
                    if digest == production_hash:
                        world_index["n"] += 1
                        return FakeSession(task_id, success=world_index["n"] == 1)
                    return FakeSession(task_id, success=False)

                return RuntimeDependencies(
                    session_factory=session_factory,
                    agent=FakeAgent(
                        [
                            AgentTurn(
                                prompt_text=_PROMPT,
                                output_text=_ACTION,
                                top_k_logprobs=_LOGPROBS,
                                latency_seconds=0.1,
                                started_at=_START,
                                action=_ACTION,
                                app_name="calendar",
                                api_name="lookup",
                            ),
                            AgentTurn(
                                prompt_text=_PROMPT,
                                output_text="STOP",
                                top_k_logprobs=_LOGPROBS,
                                latency_seconds=0.1,
                                started_at=_START,
                                action=None,
                                app_name=None,
                                api_name=None,
                            ),
                        ],
                        clock=clock,
                    ),
                    clock=clock,
                )

            rollback_store_path = root / "rollback.sqlite"
            rollback_store = LazyStore(rollback_store_path)
            rollback_client = TestClient(
                create_app(
                    _dependencies(
                        production=production,
                        candidate=candidate,
                        store=rollback_store,
                        clock=clock,
                        runtime_factory=harm_factory,
                        monitor=_monitor(production, clock),
                        horizon_episodes=1,
                        harm_margin=0.1,
                    )
                )
            )
            with rollback_client:
                admit = rollback_client.post(
                    "/candidates",
                    json=_persist_artifact(rollback_store_path, real_passed.artifact),
                )
                if admit.status_code != 200:
                    print("PASS gate must admit matching candidate", file=sys.stderr)
                    return 1
                harm = rollback_client.post(
                    "/episodes",
                    json={
                        "task_id": "task-harm",
                        "mode": "execute",
                        "assignment_key": "canary-harm",
                    },
                )
                if harm.status_code != 200:
                    print("canary harm episode failed", file=sys.stderr)
                    return 1
                deployment = rollback_client.get("/deployment").json()
                if deployment.get("state") != "ROLLED_BACK":
                    print("expected canary rollback", file=sys.stderr)
                    return 1
                summary["canary_rollback_state"] = deployment["state"]

            promote_store_path = root / "promote.sqlite"
            promote_store = LazyStore(promote_store_path)
            promote_client = TestClient(
                create_app(
                    _dependencies(
                        production=production,
                        candidate=candidate,
                        store=promote_store,
                        clock=clock,
                        runtime_factory=factory,
                        monitor=_monitor(production, clock),
                        horizon_episodes=1,
                        harm_margin=0.1,
                    )
                )
            )
            with promote_client:
                admit = promote_client.post(
                    "/candidates",
                    json=_persist_artifact(promote_store_path, real_passed.artifact),
                )
                if admit.status_code != 200:
                    print("promotion admit failed", file=sys.stderr)
                    return 1
                promote = promote_client.post(
                    "/episodes",
                    json={
                        "task_id": "task-promote",
                        "mode": "execute",
                        "assignment_key": "canary-promote",
                    },
                )
                if promote.status_code != 200:
                    print("promotion episode failed", file=sys.stderr)
                    return 1
                deployment = promote_client.get("/deployment").json()
                if deployment.get("state") != "PROMOTED":
                    print("expected canary promotion", file=sys.stderr)
                    return 1
                summary["canary_promotion_state"] = deployment["state"]
                summary["promoted_configuration_hash"] = deployment[
                    "promoted_configuration_hash"
                ]

            alert_store_path = root / "alerts.sqlite"
            alert_store = LazyStore(alert_store_path)
            alert_client = TestClient(
                create_app(
                    _dependencies(
                        production=production,
                        candidate=candidate,
                        store=alert_store,
                        clock=clock,
                        runtime_factory=lambda config: _execute_runtime(
                            clock, success=False
                        ),
                        monitor=_monitor(
                            production,
                            clock,
                            dedup_seconds=60.0,
                            period_id=None,
                        ),
                        fraction=0.01,
                        canary_assignment_seed=0,
                    )
                )
            )
            with alert_client:
                for index in range(3):
                    response = alert_client.post(
                        "/episodes",
                        json={
                            "task_id": f"task-fault-{index}",
                            "mode": "execute",
                            "role": "production",
                        },
                    )
                    if response.status_code != 200:
                        print("monitor episode failed", file=sys.stderr)
                        return 1
            reader = EpisodeStore(alert_store_path)
            try:
                alerts = reader.load_alerts()
            finally:
                reader.close()
            if len(alerts) != 1:
                print("expected one deduplicated alert", file=sys.stderr)
                return 1
            summary["persisted_alert_count"] = len(alerts)

            production_hash = run_configuration_hash(production)
            aggregate = AggregateResults(
                aggregates=(
                    AggregateRecord(
                        split="train",
                        configuration_hash=production_hash,
                        task_set_hash=task_set.task_set_hash,
                        scenario_count=task_set.scenario_count,
                        task_count=task_set.task_count,
                        episode_count=1,
                        metric="task_success",
                        value=0.0,
                    ),
                ),
                evidence=(
                    StatisticalEvidence(
                        method="paired_bootstrap",
                        split="train",
                        configuration_hash=passed.candidate_configuration_hash,
                        estimate=0.0,
                        sample_size=2,
                        unit="task_success",
                        reference_configuration_hash=passed.reference_configuration_hash,
                    ),
                ),
                decisions=(
                    LifecycleDecision(
                        tier="offline_gate",
                        decision="block",
                        split="train",
                        candidate_configuration_hash=blocked.candidate_configuration_hash,
                        reference_configuration_hash=blocked.reference_configuration_hash,
                        evidence=(
                            StatisticalEvidence(
                                method="paired_bootstrap",
                                split="train",
                                configuration_hash=blocked.candidate_configuration_hash,
                                estimate=0.0,
                                sample_size=2,
                                unit="task_success",
                                reference_configuration_hash=(
                                    blocked.reference_configuration_hash
                                ),
                            ),
                        ),
                        decided_at=_START,
                    ),
                ),
            )
            export_path = root / "public_aggregate.json"
            export_public_results(aggregate, output_path=export_path)
            exported = json.loads(export_path.read_text(encoding="utf-8"))
            assert_public_payload(exported)
            summary["public_export_aggregates"] = len(exported["aggregates"])
            summary["public_export_decisions"] = len(exported["decisions"])

    assert_public_payload(summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
