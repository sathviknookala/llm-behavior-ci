from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest import mock

from llm_behavior_ci.config import (
    CanarySettings,
    GateSettings,
    MonitorSettings,
    RunConfiguration,
    StoppingRule,
    StreamSettings,
    run_configuration_hash,
)
from llm_behavior_ci.experiments import benchmark as benchmark_module
from llm_behavior_ci.experiments.attempts import AttemptBudget, AttemptLedger
from llm_behavior_ci.experiments.benchmark import (
    BenchmarkError,
    BenchmarkProgress,
    reconcile_checkpoint,
    run_lifecycle_benchmark,
)
from llm_behavior_ci.experiments.faults import FaultPatch, FaultSpec, HarmLabel, load_fault
from llm_behavior_ci.experiments.schedule import BenchmarkSchedule
from llm_behavior_ci.experiments.protocol import ProtocolSettings, lock_protocol
from llm_behavior_ci.experiments.validation import AADependenceReport, ValidationReport
from llm_behavior_ci.lifecycle.monitoring import FrozenReference
from llm_behavior_ci.lifecycle.offline_gate import PlanEvidenceInputs, plan_evidence_to_dict
from llm_behavior_ci.records import TokenLogprob, assert_public_payload
from llm_behavior_ci.runtime.agent import AgentTurn
from llm_behavior_ci.runtime.appworld import EvaluationResult, TaskContext, ToolResult
from llm_behavior_ci.runtime.episode import RuntimeDependencies
from llm_behavior_ci.tasks.selection import (
    TaskSet,
    canonical_task_set_bytes,
    task_set_hash_from_bytes,
)

_CATALOG = Path(__file__).resolve().parents[2] / "configs" / "faults"

_START = datetime(2026, 9, 27, 15, 0, tzinfo=timezone.utc)
_PLAN = "1. open the calendar"
_LOGPROBS = (
    (
        TokenLogprob(token_id=7, logprob=-0.2, rank=0),
        TokenLogprob(token_id=9, logprob=-1.5, rank=1),
    ),
)
_GIT = "a" * 40
_REVISION = "0123456789abcdef0123456789abcdef01234567"
_TOKENIZER_REVISION = "fedcba9876543210fedcba9876543210fedcba98"
_HASH_A = "1" * 64
_HASH_B = "2" * 64


def _make_task_set(*, split: str, tasks: tuple[tuple[str, str | None], ...]) -> TaskSet:
    payload = canonical_task_set_bytes(
        appworld_version="0.1.3.post1",
        split=split,
        selection_rule="deterministic_sample",
        selection_seed=20260926,
        tasks=tasks,
    )
    named = {item[1] for item in tasks if item[1] is not None}
    lone = sum(1 for item in tasks if item[1] is None)
    return TaskSet(
        appworld_version="0.1.3.post1",
        split=split,
        selection_rule="deterministic_sample",
        selection_seed=20260926,
        task_count=len(tasks),
        scenario_count=len(named) + lone,
        task_ids=tuple(item[0] for item in tasks),
        scenario_ids=tuple(item[1] for item in tasks),
        task_set_hash=task_set_hash_from_bytes(payload),
    )


def _payload(*, split: str, task_set: TaskSet) -> dict[str, object]:
    return {
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
            "appworld_version": task_set.appworld_version,
            "split": split,
            "selection_rule": task_set.selection_rule,
            "selection_seed": task_set.selection_seed,
            "task_count": task_set.task_count,
            "task_set_hash": task_set.task_set_hash,
        },
        "run_seed": 7,
        "git_commit": _GIT,
        "protocol_hash": None,
    }


class Clock:
    def __init__(self) -> None:
        self.current = _START

    def __call__(self) -> datetime:
        value = self.current
        self.current += timedelta(seconds=1)
        return value


class World:
    def __init__(self, task_id: str, *, success: bool) -> None:
        self.task_id = task_id
        self.success = success

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
            api_name=None,
        )

    def evaluate(self) -> EvaluationResult:
        if self.success:
            return EvaluationResult(
                success=True,
                passed_requirements=1,
                total_requirements=1,
                difficulty=1,
            )
        return EvaluationResult(
            success=False,
            passed_requirements=0,
            total_requirements=1,
            difficulty=1,
        )

    def close(self) -> None:
        return None


class BenchmarkAgent:
    def __init__(self, clock: Clock, *, block_on_temperature: bool = False) -> None:
        self._clock = clock
        self._block_on_temperature = block_on_temperature
        self._config: RunConfiguration | None = None
        self._context: TaskContext | None = None

    def begin(self, context: TaskContext, config: RunConfiguration) -> None:
        self._context = context
        self._config = config

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
        assert self._config is not None
        temperature = self._config.agent.sampling.temperature
        empty = self._block_on_temperature and temperature > 0.0
        return AgentTurn(
            prompt_text="plan the next action",
            output_text="" if empty else _PLAN,
            top_k_logprobs=_LOGPROBS,
            generated_token_count=len(_LOGPROBS),
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


def _plan_evidence() -> PlanEvidenceInputs:
    return PlanEvidenceInputs(
        plan_format_version="plan-v1",
        plan_quality_features=("numbered_step_count", "token_count"),
        plan_quality_weights=(1.0, 0.1),
        mmd_features=("char_count", "numbered_step_count", "model_step_count"),
        kl_approximation="top_k",
        required_statistics=("plan_quality", "kl", "mmd"),
        validation_provenance="synthetic_fixture",
    )


class CountingFactory:
    """Pair worlds succeed/fail by role; after a cutoff, singles fail for monitor."""

    def __init__(
        self,
        *,
        candidate_success: bool,
        monitor_success: bool,
        pair_world_cutoff: int,
    ) -> None:
        self._candidate_success = candidate_success
        self._monitor_success = monitor_success
        self._pair_world_cutoff = pair_world_cutoff
        self.count = 0

    def __call__(self, task_id: str) -> World:
        self.count += 1
        if self.count > self._pair_world_cutoff:
            return World(task_id, success=self._monitor_success)
        if self.count % 2 == 1:
            return World(task_id, success=True)
        return World(task_id, success=self._candidate_success)


class FailedStartWorld(World):
    def prepare(self) -> None:
        raise RuntimeError("environment setup failed")


class FailedStartCandidateFactory:
    """Gate and reference worlds start; every canary candidate world fails setup."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self, task_id: str) -> World:
        self.count += 1
        if self.count > _GATE_WORLDS and self.count % 2 == 0:
            return FailedStartWorld(task_id, success=True)
        return World(task_id, success=True)


def _aa() -> AADependenceReport:
    return AADependenceReport(
        status="unavailable",
        evidence_accepted=False,
        provenance="unavailable",
        hardware_observed=False,
        observation_count=0,
        pair_count=0,
        repeated_task_effect=None,
        scenario_clustering_effect=None,
        inference_variation=None,
        inference_source=None,
        inference_low=None,
        inference_high=None,
        trajectory_divergence_rate=None,
        interval_width_ratio=None,
        concurrency_effect=None,
        concurrency_levels=(),
        series_alarm=None,
        memory_used_mib=None,
        wall_seconds=None,
        reason="unavailable",
    )


def _report() -> ValidationReport:
    return ValidationReport(
        method="cusum",
        implemented=True,
        validated=True,
        benchmark_eligible=True,
        calibration="none",
        study="cpu_fast",
        null_claim="type_i",
        null_draw="gaussian",
        input_hash=_HASH_A,
        seeds=(1,),
        sample_count=1,
        null_sample_size=1,
        uncertainty_level=0.95,
        alpha=None,
        horizon=None,
        false_alarm_tolerance=None,
        coverage_tolerance=None,
        parameters=(),
        required_checks=(),
        omitted_checks=(),
        checks=(),
        reference_agreements=(),
        aa=_aa(),
        libraries=(),
        configuration_hashes=(),
        gpu_evidence=False,
        gpu_floor_measured=False,
        split=None,
    )


def _temperature_fault() -> FaultSpec:
    return FaultSpec(
        fault_id="sampling_temperature_one",
        version="1",
        kind="sampling",
        patches=(FaultPatch(path="agent.sampling.temperature", value=1.0),),
    )


def _harm_label(*, harmful: bool, task_set_hash: str) -> HarmLabel:
    if harmful:
        effect = -0.2
        margin = 0.1
    else:
        effect = -0.01
        margin = 0.1
    return HarmLabel(
        fault_version="sampling_temperature_one:1",
        base_configuration_hash=_HASH_A,
        candidate_configuration_hash=_HASH_B,
        task_set_hash=task_set_hash,
        effect_estimate=effect,
        interval_low=effect - 0.05,
        interval_high=effect + 0.05,
        margin=margin,
        harmful=harmful,
        split="dev",
        confidence_level=0.9,
        resamples=25,
        seed=3,
    )


def _build_lock(
    root: Path,
    *,
    train_tasks: TaskSet,
    dev_tasks: TaskSet,
    test_normal_tasks: TaskSet,
    harmful: bool,
    canary_horizon: int,
    canary_alpha: float,
    monitor_horizon: int,
    with_replacement: bool,
):
    root.mkdir(parents=True, exist_ok=True)
    train = RunConfiguration.from_dict(_payload(split="train", task_set=train_tasks))
    dev = RunConfiguration.from_dict(_payload(split="dev", task_set=dev_tasks))
    test_normal = RunConfiguration.from_dict(
        _payload(split="test_normal", task_set=test_normal_tasks)
    )
    reference_hash = run_configuration_hash(test_normal)
    settings = ProtocolSettings(
        configurations=(train, dev, test_normal),
        task_selections=(train.task, dev.task, test_normal.task),
        harm_labels=(
            _harm_label(harmful=harmful, task_set_hash=dev_tasks.task_set_hash),
        ),
        validation_reports=(_report(),),
        faults=(_temperature_fault(),),
        plan_evidence=_plan_evidence(),
        gate=GateSettings(
            confidence_level=0.9,
            bootstrap_resamples=20,
            score_margin=-0.02,
            kl_limit_nats=0.05,
            mmd_bandwidth=1.0,
            mmd_permutations=99,
            mmd_alpha=0.05,
            plan_format_version="plan-v1",
        ),
        canary=CanarySettings(
            fraction=1.0,
            outcome_delay_seconds=0.0,
            harm_margin=0.1,
            stopping_rule=StoppingRule(
                name="sequential_canary",
                alpha=canary_alpha,
                horizon_episodes=canary_horizon,
            ),
            metric_orientation="higher_is_better",
            promotion_policy="horizon_reached_without_harm",
        ),
        monitor=MonitorSettings(
            reference_configuration_hash=reference_hash,
            outcome_delay_seconds=0.0,
            signals=("task_success",),
            stopping_rules=(
                StoppingRule(
                    name="cusum",
                    alpha=0.1,
                    horizon_episodes=monitor_horizon,
                    threshold=0.5,
                ),
            ),
        ),
        stream=StreamSettings(
            split=test_normal_tasks.split,
            selection_rule=test_normal_tasks.selection_rule,
            selection_seed=test_normal_tasks.selection_seed,
            task_set_hash=test_normal_tasks.task_set_hash,
            stream_seed=3,
            arrival_rate_per_second=1.0,
            concurrency=1,
            with_replacement=with_replacement,
            task_mix_rule="uniform",
        ),
        analysis_version="benchmark-test-v1",
        seeds=(7,),
    )
    lock = lock_protocol(settings, root / "protocol.lock.json")
    baselines = FrozenReference(
        configuration_hash=reference_hash,
        baselines=(("task_success", 0.9),),
    )
    return lock, baselines


def _sets():
    train_tasks = _make_task_set(
        split="train",
        tasks=(
            ("train-a", "scenario-1"),
            ("train-b", None),
            ("train-c", "scenario-3"),
            ("train-d", "scenario-4"),
            ("train-e", "scenario-5"),
            ("train-f", "scenario-6"),
        ),
    )
    dev_tasks = _make_task_set(split="dev", tasks=(("dev-a", None),))
    test_tasks = _make_task_set(
        split="test_normal", tasks=(("test-a", "scenario-t"),)
    )
    return train_tasks, dev_tasks, test_tasks


_GATE_WORLDS = 18


class LifecycleBenchmarkIntegrationTests(unittest.TestCase):
    def test_benchmark_plan_evidence_loader_round_trips_vocabulary_size(self) -> None:
        evidence = PlanEvidenceInputs(
            plan_format_version="plan-v1",
            plan_quality_features=("numbered_step_count",),
            plan_quality_weights=(1.0,),
            mmd_features=(),
            kl_approximation="full",
            required_statistics=("plan_quality",),
            validation_provenance="synthetic_fixture",
            kl_vocabulary_size=32000,
        )
        payload = plan_evidence_to_dict(evidence)
        restored = benchmark_module._plan_evidence_from_dict(payload)
        self.assertEqual(restored.kl_vocabulary_size, 32000)

    def test_lifecycle_benchmark_cli_plan_evidence_loader_round_trips_vocabulary_size(
        self,
    ) -> None:
        path = (
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "benchmark"
            / "run_lifecycle_benchmark.py"
        )
        spec = importlib.util.spec_from_file_location(
            "run_lifecycle_benchmark_cli", path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        evidence = PlanEvidenceInputs(
            plan_format_version="plan-v1",
            plan_quality_features=("numbered_step_count",),
            plan_quality_weights=(1.0,),
            mmd_features=(),
            kl_approximation="full",
            required_statistics=("plan_quality",),
            validation_provenance="synthetic_fixture",
            kl_vocabulary_size=32000,
        )
        payload = plan_evidence_to_dict(evidence)
        restored = module._load_plan_evidence(payload)
        self.assertEqual(restored.kl_vocabulary_size, 32000)

    def test_gate_block_catch_not_reached_later_tiers(self) -> None:
        train_tasks, dev_tasks, test_tasks = _sets()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock, baselines = _build_lock(
                root,
                train_tasks=train_tasks,
                dev_tasks=dev_tasks,
                test_normal_tasks=test_tasks,
                harmful=True,
                canary_horizon=3,
                canary_alpha=0.5,
                monitor_horizon=2,
                with_replacement=True,
            )
            clock = Clock()
            runtime = RuntimeDependencies(
                session_factory=lambda task_id: World(task_id, success=True),
                agent=BenchmarkAgent(clock, block_on_temperature=True),
                clock=clock,
            )
            result = run_lifecycle_benchmark(
                lock,
                (_temperature_fault(),),
                runtime=runtime,
                train_tasks=train_tasks,
                test_normal_tasks=test_tasks,
                reference_baselines=baselines,
                plan_evidence=_plan_evidence(),
                admission_mode="test",
                checkpoint_path=root / "checkpoint.json",
                export_path=root / "export.json",
            )
        self.assertEqual(result.status, "completed")
        replicate = result.replicates[0]
        self.assertEqual(replicate.gate.status, "completed")
        self.assertEqual(replicate.gate.outcome, "BLOCK")
        self.assertEqual(replicate.gate.classification, "catch")
        self.assertEqual(replicate.gate.validation_provenance, "synthetic_fixture")
        self.assertEqual(replicate.canary.status, "not_reached")
        self.assertEqual(replicate.canary.reason, "gate_block")
        self.assertEqual(replicate.monitor.status, "not_reached")
        self.assertEqual(replicate.monitor.reason, "gate_block")
        self.assertFalse(replicate.monitor.miss)
        self.assertEqual(result.gate_catch_count, 1)
        self.assertEqual(result.monitor_miss_count, 0)
        self.assertEqual(result.canary_not_reached_count, 1)
        self.assertEqual(result.monitor_not_reached_count, 1)

    def test_gate_false_block_not_monitor_false_alarm(self) -> None:
        train_tasks, dev_tasks, test_tasks = _sets()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock, baselines = _build_lock(
                root,
                train_tasks=train_tasks,
                dev_tasks=dev_tasks,
                test_normal_tasks=test_tasks,
                harmful=False,
                canary_horizon=3,
                canary_alpha=0.5,
                monitor_horizon=2,
                with_replacement=True,
            )
            clock = Clock()
            runtime = RuntimeDependencies(
                session_factory=lambda task_id: World(task_id, success=True),
                agent=BenchmarkAgent(clock, block_on_temperature=True),
                clock=clock,
            )
            result = run_lifecycle_benchmark(
                lock,
                (_temperature_fault(),),
                runtime=runtime,
                train_tasks=train_tasks,
                test_normal_tasks=test_tasks,
                reference_baselines=baselines,
                plan_evidence=_plan_evidence(),
                admission_mode="test",
                checkpoint_path=root / "checkpoint.json",
            )
        replicate = result.replicates[0]
        self.assertEqual(replicate.gate.classification, "false_block")
        self.assertEqual(replicate.canary.status, "not_reached")
        self.assertEqual(replicate.monitor.status, "not_reached")
        self.assertFalse(replicate.monitor.false_alarm)
        self.assertEqual(result.gate_false_block_count, 1)
        self.assertEqual(result.monitor_false_alarm_count, 0)

    def test_canary_rollback_monitor_not_reached(self) -> None:
        train_tasks, dev_tasks, test_tasks = _sets()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock, baselines = _build_lock(
                root,
                train_tasks=train_tasks,
                dev_tasks=dev_tasks,
                test_normal_tasks=test_tasks,
                harmful=True,
                canary_horizon=4,
                canary_alpha=0.5,
                monitor_horizon=2,
                with_replacement=True,
            )
            clock = Clock()
            factory = CountingFactory(
                candidate_success=False,
                monitor_success=True,
                pair_world_cutoff=_GATE_WORLDS + 8,
            )
            runtime = RuntimeDependencies(
                session_factory=factory,
                agent=BenchmarkAgent(clock, block_on_temperature=False),
                clock=clock,
            )
            result = run_lifecycle_benchmark(
                lock,
                (_temperature_fault(),),
                runtime=runtime,
                train_tasks=train_tasks,
                test_normal_tasks=test_tasks,
                reference_baselines=baselines,
                plan_evidence=_plan_evidence(),
                admission_mode="test",
                checkpoint_path=root / "checkpoint.json",
            )
        replicate = result.replicates[0]
        self.assertEqual(replicate.gate.outcome, "PASS")
        self.assertEqual(replicate.canary.status, "completed")
        self.assertIsNotNone(replicate.canary.rollback_delay_episodes)
        self.assertEqual(replicate.canary.candidate_episodes_served, 4)
        self.assertEqual(replicate.canary.candidate_episodes_failed, 0)
        self.assertEqual(replicate.monitor.status, "not_reached")
        self.assertEqual(replicate.monitor.reason, "canary_rollback")
        self.assertEqual(result.candidate_episodes_served, 4)

    def test_scheduled_canary_failed_starts_count_against_its_pair_horizon(self) -> None:
        train_tasks, dev_tasks, test_tasks = _sets()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock, baselines = _build_lock(
                root,
                train_tasks=train_tasks,
                dev_tasks=dev_tasks,
                test_normal_tasks=test_tasks,
                harmful=True,
                canary_horizon=3,
                canary_alpha=0.5,
                monitor_horizon=8,
                with_replacement=True,
            )
            schedule = BenchmarkSchedule(
                stream=StreamSettings.from_dict(lock.payload["stream"]),
                healthy_prefix_episodes=2,
                onset_mode="abrupt",
                ramp_episodes=0,
                analysis_horizon_episodes=8,
                canary_fraction=1.0,
                canary_assignment_seed=3,
                clock_start=_START,
            )
            clock = Clock()
            runtime = RuntimeDependencies(
                session_factory=FailedStartCandidateFactory(),
                agent=BenchmarkAgent(clock, block_on_temperature=False),
                clock=clock,
            )
            result = run_lifecycle_benchmark(
                lock,
                (_temperature_fault(),),
                runtime=runtime,
                train_tasks=train_tasks,
                test_normal_tasks=test_tasks,
                reference_baselines=baselines,
                plan_evidence=_plan_evidence(),
                admission_mode="test",
                checkpoint_path=root / "checkpoint.json",
                schedule=schedule,
            )
        replicate = result.replicates[0]
        self.assertEqual(replicate.gate.outcome, "PASS")
        self.assertEqual(replicate.canary.status, "horizon_exhausted")
        self.assertEqual(replicate.canary.candidate_episodes_served, 3)
        self.assertEqual(replicate.monitor.status, "not_reached")
        self.assertEqual(replicate.monitor.reason, "canary_horizon_exhausted")

    def test_release_admission_mode_rejects_synthetic_gate_evidence(self) -> None:
        train_tasks, dev_tasks, test_tasks = _sets()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock, baselines = _build_lock(
                root,
                train_tasks=train_tasks,
                dev_tasks=dev_tasks,
                test_normal_tasks=test_tasks,
                harmful=True,
                canary_horizon=4,
                canary_alpha=0.5,
                monitor_horizon=2,
                with_replacement=True,
            )
            clock = Clock()
            factory = CountingFactory(
                candidate_success=False,
                monitor_success=True,
                pair_world_cutoff=_GATE_WORLDS + 8,
            )
            runtime = RuntimeDependencies(
                session_factory=factory,
                agent=BenchmarkAgent(clock, block_on_temperature=False),
                clock=clock,
            )
            with self.assertRaisesRegex(BenchmarkError, "attempt caps"):
                run_lifecycle_benchmark(
                    lock,
                    (_temperature_fault(),),
                    runtime=runtime,
                    train_tasks=train_tasks,
                    test_normal_tasks=test_tasks,
                    reference_baselines=baselines,
                    plan_evidence=_plan_evidence(),
                    admission_mode="release",
                    checkpoint_path=root / "uncapped.json",
                )
            with self.assertRaisesRegex(BenchmarkError, "synthetic_fixture"):
                run_lifecycle_benchmark(
                    lock,
                    (_temperature_fault(),),
                    runtime=runtime,
                    train_tasks=train_tasks,
                    test_normal_tasks=test_tasks,
                    reference_baselines=baselines,
                    plan_evidence=_plan_evidence(),
                    admission_mode="release",
                    checkpoint_path=root / "checkpoint.json",
                    attempt_budget=AttemptBudget(plan_generations=120, executions=98),
                )

    def test_promote_then_monitor_alert(self) -> None:
        train_tasks, dev_tasks, test_tasks = _sets()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock, baselines = _build_lock(
                root,
                train_tasks=train_tasks,
                dev_tasks=dev_tasks,
                test_normal_tasks=test_tasks,
                harmful=True,
                canary_horizon=3,
                canary_alpha=0.5,
                monitor_horizon=2,
                with_replacement=True,
            )
            clock = Clock()
            factory = CountingFactory(
                candidate_success=True,
                monitor_success=False,
                pair_world_cutoff=_GATE_WORLDS + 6,
            )
            runtime = RuntimeDependencies(
                session_factory=factory,
                agent=BenchmarkAgent(clock, block_on_temperature=False),
                clock=clock,
            )
            result = run_lifecycle_benchmark(
                lock,
                (_temperature_fault(),),
                runtime=runtime,
                train_tasks=train_tasks,
                test_normal_tasks=test_tasks,
                reference_baselines=baselines,
                plan_evidence=_plan_evidence(),
                admission_mode="test",
                checkpoint_path=root / "checkpoint.json",
            )
        replicate = result.replicates[0]
        self.assertEqual(replicate.gate.outcome, "PASS")
        self.assertEqual(replicate.canary.status, "promoted")
        self.assertEqual(replicate.monitor.status, "completed")
        self.assertIsNotNone(replicate.monitor.delay_episodes)
        self.assertGreaterEqual(replicate.monitor.delay_episodes, 1)
        self.assertFalse(replicate.monitor.miss)

    def test_interrupt_and_resume_does_not_double_count(self) -> None:
        train_tasks, dev_tasks, test_tasks = _sets()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock, baselines = _build_lock(
                root,
                train_tasks=train_tasks,
                dev_tasks=dev_tasks,
                test_normal_tasks=test_tasks,
                harmful=True,
                canary_horizon=3,
                canary_alpha=0.5,
                monitor_horizon=2,
                with_replacement=True,
            )
            checkpoint = root / "checkpoint.json"

            def make_runtime() -> RuntimeDependencies:
                clock = Clock()
                factory = CountingFactory(
                    candidate_success=True,
                    monitor_success=False,
                    pair_world_cutoff=_GATE_WORLDS + 6,
                )
                return RuntimeDependencies(
                    session_factory=factory,
                    agent=BenchmarkAgent(clock, block_on_temperature=False),
                    clock=clock,
                )

            first = run_lifecycle_benchmark(
                lock,
                (_temperature_fault(),),
                runtime=make_runtime(),
                train_tasks=train_tasks,
                test_normal_tasks=test_tasks,
                reference_baselines=baselines,
                plan_evidence=_plan_evidence(),
                admission_mode="test",
                checkpoint_path=checkpoint,
                should_interrupt=lambda progress: progress.tier == "canary"
                and progress.stream_index >= 1,
            )
            self.assertEqual(first.status, "interrupted")
            self.assertTrue(checkpoint.exists())
            checkpoint_doc = json.loads(checkpoint.read_text(encoding="utf-8"))
            canary_items = checkpoint_doc["replicates"][
                "sampling_temperature_one:1|7"
            ]["canary"]["items"]
            first_item_count = len(canary_items)

            second = run_lifecycle_benchmark(
                lock,
                (_temperature_fault(),),
                runtime=make_runtime(),
                train_tasks=train_tasks,
                test_normal_tasks=test_tasks,
                reference_baselines=baselines,
                plan_evidence=_plan_evidence(),
                admission_mode="test",
                checkpoint_path=checkpoint,
                should_interrupt=None,
                export_path=root / "export.json",
            )
            self.assertEqual(second.status, "completed")
            replicate = second.replicates[0]
            self.assertEqual(replicate.canary.status, "promoted")
            self.assertEqual(replicate.canary.candidate_episodes_served, 3)
            self.assertEqual(second.candidate_episodes_served, 3)
            resumed = json.loads(checkpoint.read_text(encoding="utf-8"))
            resumed_items = resumed["replicates"]["sampling_temperature_one:1|7"][
                "canary"
            ]["items"]
            self.assertGreaterEqual(len(resumed_items), first_item_count)
            indexes = [item["stream_index"] for item in resumed_items]
            self.assertEqual(len(indexes), len(set(indexes)))

    def test_public_export_and_skip_on_interrupt(self) -> None:
        train_tasks, dev_tasks, test_tasks = _sets()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock, baselines = _build_lock(
                root,
                train_tasks=train_tasks,
                dev_tasks=dev_tasks,
                test_normal_tasks=test_tasks,
                harmful=True,
                canary_horizon=3,
                canary_alpha=0.5,
                monitor_horizon=2,
                with_replacement=True,
            )
            clock = Clock()
            runtime = RuntimeDependencies(
                session_factory=lambda task_id: World(task_id, success=True),
                agent=BenchmarkAgent(clock, block_on_temperature=True),
                clock=clock,
            )
            export_path = root / "export.json"
            result = run_lifecycle_benchmark(
                lock,
                (_temperature_fault(),),
                runtime=runtime,
                train_tasks=train_tasks,
                test_normal_tasks=test_tasks,
                reference_baselines=baselines,
                plan_evidence=_plan_evidence(),
                admission_mode="test",
                checkpoint_path=root / "checkpoint.json",
                export_path=export_path,
            )
            self.assertEqual(result.status, "completed")
            self.assertTrue(export_path.exists())
            raw = export_path.read_text(encoding="utf-8")
            payload = json.loads(raw)
            assert_public_payload(payload)
            metrics = {item["metric"] for item in payload["aggregates"]}
            self.assertIn("canary_not_reached_count", metrics)
            self.assertIn("monitor_not_reached_count", metrics)
            self.assertNotIn("plan_text", raw)
            self.assertNotIn("instruction", raw)
            self.assertNotIn(_PLAN, raw)
            self.assertNotIn("solve the task", raw)
            self.assertNotIn("calendar docs", raw)

            lock2, baselines2 = _build_lock(
                root / "lock2",
                train_tasks=train_tasks,
                dev_tasks=dev_tasks,
                test_normal_tasks=test_tasks,
                harmful=True,
                canary_horizon=3,
                canary_alpha=0.5,
                monitor_horizon=2,
                with_replacement=True,
            )
            interrupt_export = root / "interrupted-export.json"
            clock2 = Clock()
            factory = CountingFactory(
                candidate_success=True,
                monitor_success=False,
                pair_world_cutoff=_GATE_WORLDS + 6,
            )
            runtime2 = RuntimeDependencies(
                session_factory=factory,
                agent=BenchmarkAgent(clock2, block_on_temperature=False),
                clock=clock2,
            )
            interrupted = run_lifecycle_benchmark(
                lock2,
                (_temperature_fault(),),
                runtime=runtime2,
                train_tasks=train_tasks,
                test_normal_tasks=test_tasks,
                reference_baselines=baselines2,
                plan_evidence=_plan_evidence(),
                admission_mode="test",
                checkpoint_path=root / "checkpoint2.json",
                export_path=interrupt_export,
                should_interrupt=lambda progress: progress.tier == "canary"
                and progress.stream_index >= 1,
            )
            self.assertEqual(interrupted.status, "interrupted")
            self.assertFalse(interrupt_export.exists())

    def test_live_unavailable_fault_reported_not_omitted(self) -> None:
        train_tasks, dev_tasks, test_tasks = _sets()
        fault = load_fault(_CATALOG / "fp8_weights.v1.json")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = RunConfiguration.from_dict(
                _payload(split="train", task_set=train_tasks)
            )
            dev = RunConfiguration.from_dict(_payload(split="dev", task_set=dev_tasks))
            test_normal = RunConfiguration.from_dict(
                _payload(split="test_normal", task_set=test_tasks)
            )
            reference_hash = run_configuration_hash(test_normal)
            settings = ProtocolSettings(
                configurations=(train, dev, test_normal),
                task_selections=(train.task, dev.task, test_normal.task),
                harm_labels=(
                    HarmLabel(
                        fault_version=fault.fault_version,
                        base_configuration_hash=_HASH_A,
                        candidate_configuration_hash=_HASH_B,
                        task_set_hash=dev_tasks.task_set_hash,
                        effect_estimate=-0.2,
                        interval_low=-0.25,
                        interval_high=-0.15,
                        margin=0.1,
                        harmful=True,
                        split="dev",
                        confidence_level=0.9,
                        resamples=25,
                        seed=3,
                    ),
                ),
                validation_reports=(_report(),),
                faults=(fault,),
                plan_evidence=_plan_evidence(),
                gate=GateSettings(
                    confidence_level=0.9,
                    bootstrap_resamples=20,
                    score_margin=-0.02,
                    kl_limit_nats=0.05,
                    mmd_bandwidth=1.0,
                    mmd_permutations=99,
                    mmd_alpha=0.05,
                    plan_format_version="plan-v1",
                ),
                canary=CanarySettings(
                    fraction=1.0,
                    outcome_delay_seconds=0.0,
                    harm_margin=0.1,
                    stopping_rule=StoppingRule(
                        name="sequential_canary",
                        alpha=0.5,
                        horizon_episodes=3,
                    ),
                    metric_orientation="higher_is_better",
                    promotion_policy="horizon_reached_without_harm",
                ),
                monitor=MonitorSettings(
                    reference_configuration_hash=reference_hash,
                    outcome_delay_seconds=0.0,
                    signals=("task_success",),
                    stopping_rules=(
                        StoppingRule(
                            name="cusum",
                            alpha=0.1,
                            horizon_episodes=2,
                            threshold=0.5,
                        ),
                    ),
                ),
                stream=StreamSettings(
                    split=test_tasks.split,
                    selection_rule=test_tasks.selection_rule,
                    selection_seed=test_tasks.selection_seed,
                    task_set_hash=test_tasks.task_set_hash,
                    stream_seed=3,
                    arrival_rate_per_second=1.0,
                    concurrency=1,
                    with_replacement=True,
                    task_mix_rule="uniform",
                ),
                analysis_version="benchmark-test-v1",
                seeds=(7,),
            )
            lock = lock_protocol(settings, root / "protocol.lock.json")
            baselines = FrozenReference(
                configuration_hash=reference_hash,
                baselines=(("task_success", 0.9),),
            )
            clock = Clock()
            runtime = RuntimeDependencies(
                session_factory=lambda task_id: World(task_id, success=True),
                agent=BenchmarkAgent(clock, block_on_temperature=False),
                clock=clock,
            )
            result = run_lifecycle_benchmark(
                lock,
                (fault,),
                runtime=runtime,
                train_tasks=train_tasks,
                test_normal_tasks=test_tasks,
                reference_baselines=baselines,
                plan_evidence=_plan_evidence(),
                admission_mode="test",
                checkpoint_path=root / "checkpoint.json",
            )
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.replicates), 1)
        replicate = result.replicates[0]
        self.assertEqual(replicate.fault_version, fault.fault_version)
        self.assertEqual(replicate.gate.status, "unavailable")
        self.assertIsNotNone(replicate.gate.reason)
        self.assertIn("quantization", replicate.gate.reason)
        self.assertEqual(replicate.canary.status, "unavailable")
        self.assertEqual(replicate.monitor.status, "unavailable")


class Crash(BaseException):
    """Stands in for a process kill: no ``except Exception`` handler sees it."""


class CrashingAgent(BenchmarkAgent):
    def __init__(self, clock: Clock, *, crash_on_turn: int) -> None:
        super().__init__(clock)
        self._crash_on_turn = crash_on_turn
        self.turns = 0

    def next_turn(self, *, tool_output: str | None) -> AgentTurn:
        self.turns += 1
        if self.turns == self._crash_on_turn:
            raise Crash()
        return super().next_turn(tool_output=tool_output)


def _crash_ledger(method: str, predicate: Callable[[Any], bool]):
    original = getattr(AttemptLedger, method)

    def wrapped(self, subject, *args, **kwargs):
        if predicate(subject):
            raise Crash()
        return original(self, subject, *args, **kwargs)

    return mock.patch.object(AttemptLedger, method, wrapped)


def _in_tier(tier: str) -> Callable[[Any], bool]:
    def matches(subject: Any) -> bool:
        attempts = [subject] if isinstance(subject, dict) else list(subject)
        return any(f"|{tier}|" in attempt["scope"] for attempt in attempts)

    return matches


_GATE_PLAN_TURNS = 12


class AttemptAccountingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.train_tasks, dev_tasks, self.test_tasks = _sets()
        self.lock, self.baselines = _build_lock(
            self.root,
            train_tasks=self.train_tasks,
            dev_tasks=dev_tasks,
            test_normal_tasks=self.test_tasks,
            harmful=True,
            canary_horizon=3,
            canary_alpha=0.5,
            monitor_horizon=8,
            with_replacement=True,
        )
        self.schedule = BenchmarkSchedule(
            stream=StreamSettings.from_dict(self.lock.payload["stream"]),
            healthy_prefix_episodes=2,
            onset_mode="abrupt",
            ramp_episodes=0,
            analysis_horizon_episodes=8,
            canary_fraction=1.0,
            canary_assignment_seed=3,
            clock_start=_START,
        )
        self.checkpoint = self.root / "checkpoint.json"

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _run(self, crash_on_turn: int | None = None, budget: AttemptBudget | None = None):
        clock = Clock()
        runtime = RuntimeDependencies(
            session_factory=lambda task_id: World(task_id, success=True),
            agent=(
                BenchmarkAgent(clock)
                if crash_on_turn is None
                else CrashingAgent(clock, crash_on_turn=crash_on_turn)
            ),
            clock=clock,
        )
        return run_lifecycle_benchmark(
            self.lock,
            (_temperature_fault(),),
            runtime=runtime,
            train_tasks=self.train_tasks,
            test_normal_tasks=self.test_tasks,
            reference_baselines=self.baselines,
            plan_evidence=_plan_evidence(),
            admission_mode="test",
            checkpoint_path=self.checkpoint,
            schedule=self.schedule,
            attempt_budget=budget,
        )

    def _records(self, tier: str | None = None) -> list[dict[str, Any]]:
        records = json.loads(self.checkpoint.read_text(encoding="utf-8"))["attempts"]["records"]
        if tier is None:
            return records
        return [record for record in records if f"|{tier}|" in str(record["scope"])]

    def test_uninterrupted_run_accounts_every_plan_generation_and_execution(self) -> None:
        result = self._run(budget=AttemptBudget(plan_generations=12, executions=14))
        replicate = result.replicates[0]
        self.assertEqual(replicate.canary.status, "promoted")
        self.assertEqual(replicate.monitor.status, "completed")
        self.assertEqual(result.attempts["plan"]["consumed"], 12)
        self.assertEqual(result.attempts["plan"]["completed"], 12)
        self.assertEqual(result.attempts["execute"]["consumed"], 14)
        self.assertEqual(result.attempts["execute"]["completed"], 14)
        self.assertEqual(result.attempts["execute"]["remaining"], 0)
        for record in self._records():
            self.assertIsNotNone(record["started_at"])
            self.assertIsNotNone(record["finished_at"])
            self.assertGreaterEqual(record["duration_seconds"], 0.0)
            self.assertNotEqual(str(record["reserved_at"])[:10], "2023-05-18")

    def test_crash_between_reservation_and_start_spends_the_slot_without_a_start(self) -> None:
        with _crash_ledger("start", _in_tier("canary")):
            with self.assertRaises(Crash):
                self._run()
        self.assertEqual(
            [record["state"] for record in self._records("canary")], ["reserved", "reserved"]
        )
        result = self._run()
        canary = self._records("canary")
        scopes = [record["scope"] for record in canary]
        self.assertEqual(len(set(scopes)), 3)
        self.assertEqual([record["state"] for record in canary[:2]], ["interrupted"] * 2)
        self.assertEqual(scopes.count(canary[0]["scope"]), 2)
        self.assertEqual(result.replicates[0].canary.status, "horizon_exhausted")
        self.assertEqual(result.replicates[0].canary.candidate_episodes_served, 2)
        self.assertEqual(result.replicates[0].monitor.reason, "canary_horizon_exhausted")
        self.assertEqual(result.attempts["execute"]["interrupted_before_start"], 2)
        self.assertEqual(result.attempts["execute"]["known_starts"], 4)
        self.assertEqual(result.attempts["execute"]["consumed"], 6)

    def test_crash_during_provider_dispatch_is_never_rerun_and_counts_against_the_pair_horizon(
        self,
    ) -> None:
        with self.assertRaises(Crash):
            self._run(crash_on_turn=_GATE_PLAN_TURNS + 1)
        self.assertEqual(
            [record["state"] for record in self._records("canary")], ["started", "started"]
        )
        summary = reconcile_checkpoint(self.checkpoint)
        self.assertEqual(summary.reconciled, 2)
        self.assertEqual(summary.attempts["execute"]["known_starts"], 2)
        self.assertEqual(reconcile_checkpoint(self.checkpoint).reconciled, 0)
        result = self._run()
        canary = self._records("canary")
        self.assertEqual(len({record["scope"] for record in canary}), 3)
        self.assertEqual(result.replicates[0].canary.status, "horizon_exhausted")
        self.assertEqual(result.replicates[0].canary.candidate_episodes_served, 2)
        self.assertEqual(result.attempts["execute"]["interrupted"], 2)
        self.assertEqual(result.attempts["execute"]["known_starts"], 6)
        self.assertEqual(result.attempts["execute"]["consumed"], 6)

    def test_crash_before_completion_is_persisted_spends_the_slot_and_is_not_repeated(self) -> None:
        with _crash_ledger("finish", _in_tier("canary")):
            with self.assertRaises(Crash):
                self._run()
        checkpoint = json.loads(self.checkpoint.read_text(encoding="utf-8"))
        replicate = next(iter(checkpoint["replicates"].values()))
        self.assertEqual(replicate["canary"].get("items", []), [])
        first = self._run()
        second = self._run()
        self.assertEqual(first.attempts, second.attempts)
        self.assertEqual(first.attempts["execute"]["interrupted"], 2)
        self.assertEqual(first.attempts["execute"]["known_starts"], 6)
        self.assertEqual(len(self._records()), 18)

    def test_scheduled_monitor_crash_records_the_arrival_as_interrupted(self) -> None:
        with _crash_ledger("finish", _in_tier("monitor")):
            with self.assertRaises(Crash):
                self._run()
        result = self._run()
        monitor = self._records("monitor")
        self.assertEqual(len({record["scope"] for record in monitor}), 8)
        self.assertEqual(monitor[0]["state"], "interrupted")
        self.assertEqual(result.replicates[0].monitor.status, "completed")
        checkpoint = json.loads(self.checkpoint.read_text(encoding="utf-8"))
        replicate = next(iter(checkpoint["replicates"].values()))
        statuses = [item["episode_status"] for item in replicate["monitor"]["items"]]
        self.assertEqual(statuses[0], "interrupted")
        self.assertEqual(len(statuses), 8)

    def test_gate_crash_ends_the_gate_without_new_plan_generations(self) -> None:
        def third_pair(subject: Any) -> bool:
            return any(str(attempt["scope"]).endswith("|gate|2") for attempt in subject)

        with _crash_ledger("start", third_pair):
            with self.assertRaises(Crash):
                self._run()
        result = self._run()
        replicate = result.replicates[0]
        self.assertEqual(replicate.gate.status, "execution_failed")
        self.assertEqual(replicate.canary.reason, "gate_execution_failed")
        self.assertEqual(result.attempts["plan"]["consumed"], 6)
        self.assertEqual(result.attempts["plan"]["known_starts"], 4)
        self.assertEqual(result.attempts["plan"]["interrupted_before_start"], 2)
        self.assertEqual(result.attempts["execute"]["consumed"], 0)

    def test_execution_cap_refuses_the_pair_that_would_exceed_it(self) -> None:
        budget = AttemptBudget(plan_generations=12, executions=3)
        with self.assertRaisesRegex(BenchmarkError, "execute cap reached"):
            self._run(budget=budget)
        self.assertEqual(len(self._records("canary")), 2)
        with self.assertRaisesRegex(BenchmarkError, "execute cap reached"):
            self._run(budget=budget)
        self.assertEqual(len(self._records("canary")), 2)

    def test_plan_cap_refuses_the_gate_before_any_generation(self) -> None:
        with self.assertRaisesRegex(BenchmarkError, "plan cap reached"):
            self._run(budget=AttemptBudget(plan_generations=11, executions=98))
        self.assertFalse(self.checkpoint.exists())

    def test_resume_keeps_the_original_caps(self) -> None:
        with self.assertRaises(Crash):
            self._run(
                crash_on_turn=_GATE_PLAN_TURNS + 1,
                budget=AttemptBudget(plan_generations=120, executions=98),
            )
        with self.assertRaisesRegex(BenchmarkError, "original caps"):
            self._run(budget=AttemptBudget(plan_generations=120, executions=100))
        result = self._run()
        self.assertEqual(result.attempts["execute"]["cap"], 98)
        self.assertEqual(result.attempts["execute"]["remaining"], 98 - 6)


if __name__ == "__main__":
    unittest.main()

