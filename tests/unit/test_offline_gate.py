from __future__ import annotations

import copy
import math
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from llm_behavior_ci.config import GateSettings, RunConfiguration, run_configuration_hash
from llm_behavior_ci.lifecycle.offline_gate import (
    GateExecutionError,
    PlanEvidenceInputs,
    plan_evidence_from_dict,
    plan_evidence_to_dict,
    run_offline_gate,
)
from llm_behavior_ci.records import TokenLogprob
from llm_behavior_ci.runtime.agent import AgentTurn
from llm_behavior_ci.runtime.appworld import EvaluationResult, TaskContext, ToolResult
from llm_behavior_ci.runtime.episode import RuntimeDependencies
from llm_behavior_ci.runtime.scoring import score_top_k
from llm_behavior_ci.tasks.plan_specs import TaskPlanSpec
from llm_behavior_ci.tasks.selection import (
    TaskSet,
    canonical_task_set_bytes,
    task_set_hash_from_bytes,
)

_START = datetime(2026, 9, 27, 15, 0, tzinfo=timezone.utc)
_PLAN = "1. open the calendar"
_PLAN_B = "1. open the calendar\n2. create an event"
_MATCHING_LOGPROBS = (
    (
        TokenLogprob(token_id=7, logprob=-0.2, rank=0),
        TokenLogprob(token_id=9, logprob=-1.5, rank=1),
    ),
)
_DIVERGENT_LOGPROBS = (
    (
        TokenLogprob(token_id=7, logprob=math.log(0.05), rank=0),
        TokenLogprob(token_id=9, logprob=math.log(0.95), rank=1),
    ),
)
_MISALIGNED_LOGPROBS = (
    (
        TokenLogprob(token_id=11, logprob=-0.2, rank=0),
        TokenLogprob(token_id=13, logprob=-1.5, rank=1),
    ),
)
_FULL_LOGPROBS = (
    (
        TokenLogprob(token_id=0, logprob=math.log(0.5), rank=0),
        TokenLogprob(token_id=1, logprob=math.log(0.5), rank=1),
    ),
)
_FULL_DIVERGENT = (
    (
        TokenLogprob(token_id=0, logprob=math.log(0.25), rank=0),
        TokenLogprob(token_id=1, logprob=math.log(0.75), rank=1),
    ),
)


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


def _config(
    task_set: TaskSet,
    *,
    run_seed: int = 7,
    tokenizer_revision: str | None = None,
) -> RunConfiguration:
    payload = copy.deepcopy(_payload())
    payload["run_seed"] = run_seed
    payload["task"] = {
        "appworld_version": task_set.appworld_version,
        "split": task_set.split,
        "selection_rule": task_set.selection_rule,
        "selection_seed": task_set.selection_seed,
        "task_count": task_set.task_count,
        "task_set_hash": task_set.task_set_hash,
    }
    if tokenizer_revision is not None:
        payload["model"]["tokenizer"]["revision"] = tokenizer_revision
    return RunConfiguration.from_dict(payload)


def _settings(**overrides: object) -> GateSettings:
    values: dict[str, object] = {
        "confidence_level": 0.9,
        "bootstrap_resamples": 40,
        "score_margin": -0.02,
        "kl_limit_nats": 0.05,
        "mmd_bandwidth": 1.0,
        "mmd_permutations": 19,
        "mmd_alpha": 0.05,
        "plan_format_version": "plan-v1",
    }
    values.update(overrides)
    return GateSettings(**values)


def _task_plan_specs() -> tuple[TaskPlanSpec, ...]:
    spec = TaskPlanSpec(
        task_id="task-a",
        available_tools=("calendar.open_calendar", "calendar.create_event"),
        subgoal_keywords=(
            ("open the calendar",),
            ("create an event", "add an event"),
        ),
        required_entities=("calendar",),
        dependency_pairs=(("calendar.open_calendar", "calendar.create_event"),),
    )
    other = TaskPlanSpec(
        task_id="task-b",
        available_tools=("calendar.open_calendar", "calendar.create_event"),
        subgoal_keywords=(
            ("open the calendar",),
            ("create an event", "add an event"),
        ),
        required_entities=("calendar",),
        dependency_pairs=(("calendar.open_calendar", "calendar.create_event"),),
    )
    return (spec, other)


def _evidence(**overrides: object) -> PlanEvidenceInputs:
    values: dict[str, object] = {
        "plan_format_version": "plan-v1",
        "plan_quality_features": ("numbered_step_count", "token_count"),
        "plan_quality_weights": (1.0, 0.1),
        "mmd_features": ("char_count", "numbered_step_count", "model_step_count"),
        "kl_approximation": "top_k",
        "required_statistics": ("plan_quality", "kl", "mmd"),
        "validation_provenance": "synthetic_fixture",
    }
    values.update(overrides)
    return PlanEvidenceInputs(**values)


class FakeSession:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.close_count = 0

    def context(self) -> TaskContext:
        return TaskContext(
            task_id=self.task_id,
            instruction="solve the task",
            api_documentation="calendar docs",
        )

    def execute(self, action: str) -> ToolResult:
        del action
        raise AssertionError("plan mode must not execute tools")

    def evaluate(self) -> EvaluationResult:
        raise AssertionError("plan mode must not evaluate")

    def close(self) -> None:
        self.close_count += 1


class PlanAgent:
    def __init__(
        self,
        clock,
        *,
        plan_text: str = _PLAN,
        logprobs=_MATCHING_LOGPROBS,
        by_run_seed: dict[int, tuple[str, object]] | None = None,
        teacher_force_by_seed: dict[int, object] | None = None,
        teacher_force_logprobs=None,
        empty_plan: bool = False,
        support_teacher_force: bool = True,
    ) -> None:
        self._clock = clock
        self._plan_text = plan_text
        self._logprobs = logprobs
        self._by_run_seed = by_run_seed or {}
        self._teacher_force_by_seed = teacher_force_by_seed or {}
        self._teacher_force_logprobs = (
            teacher_force_logprobs
            if teacher_force_logprobs is not None
            else _MATCHING_LOGPROBS
        )
        self._empty_plan = empty_plan
        self._support_teacher_force = support_teacher_force
        self.config: RunConfiguration | None = None
        self._context: TaskContext | None = None
        self.generation_logprobs_used_for_kl = False

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
        assert self.config is not None
        plan_text = self._plan_text
        logprobs = self._logprobs
        if self.config.run_seed in self._by_run_seed:
            plan_text, logprobs = self._by_run_seed[self.config.run_seed]
        if self._empty_plan:
            plan_text = "   "
        return AgentTurn(
            prompt_text="plan the next action",
            output_text=plan_text,
            top_k_logprobs=logprobs,
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
        if not self._support_teacher_force:
            raise RuntimeError("unsupported teacher force")
        assert self.config is not None
        if self.config.run_seed in self._teacher_force_by_seed:
            return self._teacher_force_by_seed[self.config.run_seed]
        return self._teacher_force_logprobs


class GenerationOnlyAgent(PlanAgent):
    def __init__(self, clock, **kwargs) -> None:
        super().__init__(clock, support_teacher_force=True, **kwargs)

    def __getattribute__(self, name: str):
        if name == "teacher_force_plan":
            raise AttributeError(name)
        return super().__getattribute__(name)


def _clock():
    current = _START

    def tick() -> datetime:
        nonlocal current
        value = current
        current = current + timedelta(seconds=1)
        return value

    return tick


def _runtime(agent=None, *, clock=None, **agent_kwargs) -> RuntimeDependencies:
    active_clock = clock or _clock()
    active_agent = agent if agent is not None else PlanAgent(active_clock, **agent_kwargs)
    return RuntimeDependencies(
        session_factory=lambda task_id: FakeSession(task_id),
        agent=active_agent,
        clock=active_clock,
    )


class OfflineGateUnitTests(unittest.TestCase):
    def test_matching_healthy_inputs_pass_with_three_statistics(self) -> None:
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(task_set, run_seed=8)
        decision = run_offline_gate(
            reference,
            candidate,
            task_set,
            settings=_settings(),
            runtime=_runtime(),
            plan_evidence=_evidence(),
        )
        self.assertEqual(decision.outcome, "PASS")
        self.assertEqual(decision.reason_codes, ())
        self.assertEqual(len(decision.statistics), 3)
        methods = tuple(item.method for item in decision.statistics)
        self.assertEqual(
            methods,
            ("plan_quality_bootstrap", "plan_kl_top_k", "plan_mmd"),
        )
        self.assertIsNotNone(decision.public_decision)
        self.assertEqual(decision.public_decision.decision, "allow_canary")
        self.assertEqual(decision.public_decision.tier, "offline_gate")
        self.assertEqual(decision.statistics[0].unit, "score_delta")
        self.assertEqual(decision.statistics[1].unit, "nats")
        self.assertEqual(decision.statistics[2].unit, "mmd_squared")
        self.assertIsNone(decision.statistics[1].seed)
        self.assertIsNotNone(decision.statistics[0].seed)
        self.assertIsNotNone(decision.statistics[2].seed)
        self.assertEqual(decision.validation_provenance, "synthetic_fixture")
        self.assertNotEqual(decision.statistics[0].estimate, 1.0)
        self.assertEqual(len(decision.artifact.scoring_contract_hashes), 4)
        self.assertEqual(
            decision.artifact.scoring_contract_hashes,
            tuple(sorted(decision.artifact.scoring_contract_hashes)),
        )
        for digest in decision.artifact.scoring_contract_hashes:
            self.assertEqual(len(digest), 64)
            int(digest, 16)

    def test_plan_quality_scores_vary_with_plan_content(self) -> None:
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(task_set, run_seed=8)
        short = run_offline_gate(
            reference,
            candidate,
            task_set,
            settings=_settings(),
            runtime=_runtime(plan_text=_PLAN),
            plan_evidence=_evidence(required_statistics=("plan_quality",)),
        )
        long = run_offline_gate(
            reference,
            candidate,
            task_set,
            settings=_settings(),
            runtime=_runtime(
                by_run_seed={
                    7: (_PLAN, _MATCHING_LOGPROBS),
                    8: (_PLAN_B, _MATCHING_LOGPROBS),
                }
            ),
            plan_evidence=_evidence(required_statistics=("plan_quality",)),
        )
        self.assertEqual(short.outcome, "PASS")
        self.assertNotEqual(
            short.statistics[0].estimate,
            long.statistics[0].estimate,
        )
        self.assertEqual(short.artifact.scoring_contract_hashes, ())

    def test_identical_plans_do_not_require_constant_one(self) -> None:
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(task_set, run_seed=8)
        decision = run_offline_gate(
            reference,
            candidate,
            task_set,
            settings=_settings(),
            runtime=_runtime(plan_text=_PLAN_B),
            plan_evidence=_evidence(
                plan_quality_features=("numbered_step_count",),
                plan_quality_weights=(1.0,),
                required_statistics=("plan_quality",),
            ),
        )
        self.assertEqual(decision.outcome, "PASS")
        self.assertEqual(decision.statistics[0].estimate, 0.0)
        self.assertNotEqual(decision.statistics[0].estimate, 1.0)

    def test_plan_quality_margin_rejection(self) -> None:
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(task_set, run_seed=8)
        decision = run_offline_gate(
            reference,
            candidate,
            task_set,
            settings=_settings(score_margin=0.01),
            runtime=_runtime(),
            plan_evidence=_evidence(),
        )
        self.assertEqual(decision.outcome, "BLOCK")
        self.assertIn("plan_quality_margin", decision.reason_codes)
        self.assertEqual(decision.public_decision.decision, "block")

    def test_kl_limit_rejection_uses_teacher_force(self) -> None:
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(task_set, run_seed=8)
        decision = run_offline_gate(
            reference,
            candidate,
            task_set,
            settings=_settings(kl_limit_nats=0.0),
            runtime=_runtime(
                teacher_force_by_seed={
                    7: _MATCHING_LOGPROBS,
                    8: _DIVERGENT_LOGPROBS,
                }
            ),
            plan_evidence=_evidence(),
        )
        self.assertEqual(decision.outcome, "BLOCK")
        self.assertIn("kl_limit", decision.reason_codes)
        kl = next(
            item for item in decision.statistics if item.method == "plan_kl_top_k"
        )
        self.assertGreater(kl.estimate, 0.0)

    def test_missing_teacher_force_does_not_use_generation_logprobs(self) -> None:
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(task_set, run_seed=8)
        clock = _clock()
        agent = GenerationOnlyAgent(
            clock,
            by_run_seed={
                7: (_PLAN, _MATCHING_LOGPROBS),
                8: (_PLAN, _DIVERGENT_LOGPROBS),
            },
        )
        with patch(
            "llm_behavior_ci.lifecycle.offline_gate.score_top_k",
            wraps=score_top_k,
        ) as scored:
            decision = run_offline_gate(
                reference,
                candidate,
                task_set,
                settings=_settings(),
                runtime=_runtime(agent=agent, clock=clock),
                plan_evidence=_evidence(),
            )
        scored.assert_not_called()
        self.assertEqual(decision.outcome, "BLOCK")
        self.assertIn("teacher_force_unavailable", decision.reason_codes)
        methods = tuple(item.method for item in decision.statistics)
        self.assertEqual(methods, ("plan_quality_bootstrap", "plan_mmd"))

    def test_top_k_versus_full_labels(self) -> None:
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(task_set, run_seed=8)
        top_k = run_offline_gate(
            reference,
            candidate,
            task_set,
            settings=_settings(),
            runtime=_runtime(),
            plan_evidence=_evidence(
                kl_approximation="top_k",
                required_statistics=("kl",),
            ),
        )
        full = run_offline_gate(
            reference,
            candidate,
            task_set,
            settings=_settings(),
            runtime=_runtime(
                teacher_force_logprobs=_FULL_LOGPROBS,
                teacher_force_by_seed={7: _FULL_LOGPROBS, 8: _FULL_LOGPROBS},
            ),
            plan_evidence=_evidence(
                kl_approximation="full",
                required_statistics=("kl",),
                kl_vocabulary_size=2,
            ),
        )
        self.assertEqual(top_k.statistics[0].method, "plan_kl_top_k")
        self.assertEqual(full.statistics[0].method, "plan_kl_full")

    def test_mmd_receives_clusters(self) -> None:
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(task_set, run_seed=8)
        with patch(
            "llm_behavior_ci.lifecycle.offline_gate.mmd_permutation_test",
        ) as mmd_call:
            mmd_call.return_value = SimpleNamespace(mmd_squared=0.0, p_value=1.0)
            run_offline_gate(
                reference,
                candidate,
                task_set,
                settings=_settings(),
                runtime=_runtime(),
                plan_evidence=_evidence(required_statistics=("mmd",)),
            )
        kwargs = mmd_call.call_args.kwargs
        self.assertEqual(kwargs["clusters"], ("scenario-1", "task-b"))

    def test_mmd_rejected(self) -> None:
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(task_set, run_seed=8)
        decision = run_offline_gate(
            reference,
            candidate,
            task_set,
            settings=_settings(mmd_alpha=0.99, mmd_permutations=99),
            runtime=_runtime(
                by_run_seed={
                    7: ("1. a", _MATCHING_LOGPROBS),
                    8: ("1. " + ("b" * 80), _MATCHING_LOGPROBS),
                }
            ),
            plan_evidence=_evidence(),
        )
        self.assertEqual(decision.outcome, "BLOCK")
        self.assertIn("mmd_rejected", decision.reason_codes)

    def test_unsupported_tokenizer_skips_kl(self) -> None:
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(
            task_set,
            run_seed=8,
            tokenizer_revision="abcdabcdabcdabcdabcdabcdabcdabcdabcdabcd",
        )
        with patch(
            "llm_behavior_ci.lifecycle.offline_gate.score_top_k",
            wraps=score_top_k,
        ) as kl_call:
            decision = run_offline_gate(
                reference,
                candidate,
                task_set,
                settings=_settings(),
                runtime=_runtime(),
                plan_evidence=_evidence(),
            )
        kl_call.assert_not_called()
        self.assertEqual(decision.outcome, "BLOCK")
        self.assertIn("unsupported_tokenizer", decision.reason_codes)
        methods = tuple(item.method for item in decision.statistics)
        self.assertEqual(methods, ("plan_quality_bootstrap", "plan_mmd"))

    def test_kl_alignment_failed(self) -> None:
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(task_set, run_seed=8)
        with patch(
            "llm_behavior_ci.lifecycle.offline_gate.score_top_k",
            wraps=score_top_k,
        ) as kl_call:
            decision = run_offline_gate(
                reference,
                candidate,
                task_set,
                settings=_settings(),
                runtime=_runtime(
                    teacher_force_by_seed={
                        7: _MATCHING_LOGPROBS,
                        8: _MISALIGNED_LOGPROBS,
                    }
                ),
                plan_evidence=_evidence(),
            )
        kl_call.assert_not_called()
        self.assertEqual(decision.outcome, "BLOCK")
        self.assertIn("kl_alignment_failed", decision.reason_codes)
        methods = tuple(item.method for item in decision.statistics)
        self.assertEqual(methods, ("plan_quality_bootstrap", "plan_mmd"))

    def test_missing_required_feature_is_execution_error(self) -> None:
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(task_set, run_seed=8)
        with self.assertRaises(GateExecutionError):
            run_offline_gate(
                reference,
                candidate,
                task_set,
                settings=_settings(),
                runtime=_runtime(),
                plan_evidence=_evidence(
                    plan_quality_features=("not_a_feature",),
                    plan_quality_weights=(1.0,),
                    required_statistics=("plan_quality",),
                ),
            )

    def test_missing_task_plan_spec_is_execution_error(self) -> None:
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(task_set, run_seed=8)
        with self.assertRaises(GateExecutionError):
            run_offline_gate(
                reference,
                candidate,
                task_set,
                settings=_settings(),
                runtime=_runtime(),
                plan_evidence=_evidence(
                    plan_quality_features=("requirement_coverage_fraction",),
                    plan_quality_weights=(1.0,),
                    required_statistics=("plan_quality",),
                    task_plan_specs=(),
                ),
            )

    def test_semantic_plan_quality_reflects_requirement_coverage(self) -> None:
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(task_set, run_seed=8)
        decision = run_offline_gate(
            reference,
            candidate,
            task_set,
            settings=_settings(score_margin=-1.0),
            runtime=_runtime(
                by_run_seed={
                    7: ("1. open the calendar", _MATCHING_LOGPROBS),
                    8: (
                        "1. open the calendar\n2. create an event",
                        _MATCHING_LOGPROBS,
                    ),
                }
            ),
            plan_evidence=_evidence(
                plan_quality_features=("requirement_coverage_fraction",),
                plan_quality_weights=(1.0,),
                required_statistics=("plan_quality",),
                task_plan_specs=_task_plan_specs(),
            ),
        )
        statistic = decision.statistics[0]
        self.assertEqual(statistic.method, "plan_quality_bootstrap")
        self.assertGreater(statistic.estimate, 0.0)

    def test_semantic_mmd_features_use_task_plan_specs_and_preserve_clusters(
        self,
    ) -> None:
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(task_set, run_seed=8)
        with patch(
            "llm_behavior_ci.lifecycle.offline_gate.mmd_permutation_test",
        ) as mmd_call:
            mmd_call.return_value = SimpleNamespace(mmd_squared=0.0, p_value=1.0)
            run_offline_gate(
                reference,
                candidate,
                task_set,
                settings=_settings(),
                runtime=_runtime(
                    by_run_seed={
                        7: ("1. open the calendar", _MATCHING_LOGPROBS),
                        8: (
                            "1. open the calendar\n2. create an event",
                            _MATCHING_LOGPROBS,
                        ),
                    }
                ),
                plan_evidence=_evidence(
                    mmd_features=(
                        "requirement_coverage_fraction",
                        "invalid_tool_reference_fraction",
                    ),
                    required_statistics=("mmd",),
                    task_plan_specs=_task_plan_specs(),
                ),
            )
        kwargs = mmd_call.call_args.kwargs
        self.assertEqual(kwargs["clusters"], ("scenario-1", "task-b"))
        self.assertEqual(kwargs["production"], ((0.5, 0.0), (0.5, 0.0)))
        self.assertEqual(kwargs["candidate"], ((1.0, 0.0), (1.0, 0.0)))

    def test_semantic_features_are_deterministic_across_repeated_calls(self) -> None:
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(task_set, run_seed=8)
        evidence = _evidence(
            plan_quality_features=("requirement_coverage_fraction",),
            plan_quality_weights=(1.0,),
            mmd_features=("requirement_coverage_fraction", "tool_reference_fraction"),
            required_statistics=("plan_quality", "mmd"),
            task_plan_specs=_task_plan_specs(),
        )
        first = run_offline_gate(
            reference,
            candidate,
            task_set,
            settings=_settings(),
            runtime=_runtime(
                plan_text="1. open the calendar\n2. create an event"
            ),
            plan_evidence=evidence,
        )
        second = run_offline_gate(
            reference,
            candidate,
            task_set,
            settings=_settings(),
            runtime=_runtime(
                plan_text="1. open the calendar\n2. create an event"
            ),
            plan_evidence=evidence,
        )
        self.assertEqual(
            tuple(item.to_dict() for item in first.statistics),
            tuple(item.to_dict() for item in second.statistics),
        )

    def test_plan_run_failed_blocks_without_statistics(self) -> None:
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(task_set, run_seed=8)
        decision = run_offline_gate(
            reference,
            candidate,
            task_set,
            settings=_settings(),
            runtime=_runtime(empty_plan=True),
            plan_evidence=_evidence(),
        )
        self.assertEqual(decision.outcome, "BLOCK")
        self.assertEqual(decision.reason_codes, ("plan_run_failed",))
        self.assertEqual(decision.statistics, ())
        self.assertIsNone(decision.public_decision)

    def test_non_plan_pair_is_execution_error(self) -> None:
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(task_set, run_seed=8)

        def fake_pair(*_args, **_kwargs):
            episode = SimpleNamespace(
                mode="execute",
                tool_steps=(),
                status="completed",
                termination_reason="agent_stopped",
            )
            return SimpleNamespace(reference=episode, candidate=episode)

        with patch(
            "llm_behavior_ci.lifecycle.offline_gate.run_pair",
            side_effect=fake_pair,
        ):
            with self.assertRaises(GateExecutionError):
                run_offline_gate(
                    reference,
                    candidate,
                    task_set,
                    settings=_settings(),
                    runtime=_runtime(),
                    plan_evidence=_evidence(),
                )

    def test_tool_steps_block_with_tool_execution(self) -> None:
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(task_set, run_seed=8)
        tool_step = SimpleNamespace(index=1)

        def fake_pair(*_args, **_kwargs):
            episode = SimpleNamespace(
                mode="plan",
                tool_steps=(tool_step,),
                status="completed",
                termination_reason="plan_emitted",
            )
            return SimpleNamespace(reference=episode, candidate=episode)

        with patch(
            "llm_behavior_ci.lifecycle.offline_gate.run_pair",
            side_effect=fake_pair,
        ):
            decision = run_offline_gate(
                reference,
                candidate,
                task_set,
                settings=_settings(),
                runtime=_runtime(),
                plan_evidence=_evidence(),
            )
        self.assertEqual(decision.outcome, "BLOCK")
        self.assertEqual(decision.reason_codes, ("tool_execution",))
        self.assertEqual(decision.statistics, ())
        self.assertIsNone(decision.public_decision)

    def test_runtime_failure_is_gate_execution_error(self) -> None:
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(task_set, run_seed=8)

        def broken_factory(task_id: str) -> FakeSession:
            del task_id
            raise RuntimeError("session unavailable")

        clock = _clock()
        runtime = RuntimeDependencies(
            session_factory=broken_factory,
            agent=PlanAgent(clock),
            clock=clock,
        )
        with self.assertRaises(GateExecutionError):
            run_offline_gate(
                reference,
                candidate,
                task_set,
                settings=_settings(),
                runtime=runtime,
                plan_evidence=_evidence(),
            )

    def test_split_and_format_and_hash_mismatches_are_execution_errors(self) -> None:
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(task_set, run_seed=8)
        with self.assertRaises(GateExecutionError):
            run_offline_gate(
                reference,
                candidate,
                replace(task_set, split="dev"),
                settings=_settings(),
                runtime=_runtime(),
                plan_evidence=_evidence(),
            )
        with self.assertRaises(GateExecutionError):
            run_offline_gate(
                reference,
                candidate,
                task_set,
                settings=_settings(plan_format_version="plan-v2"),
                runtime=_runtime(),
                plan_evidence=_evidence(plan_format_version="plan-v2"),
            )
        with self.assertRaises(GateExecutionError):
            run_offline_gate(
                reference,
                candidate,
                task_set,
                settings=_settings(plan_format_version="plan-v1"),
                runtime=_runtime(),
                plan_evidence=_evidence(plan_format_version="plan-v2"),
            )
        other = _task_set()
        mismatched = replace(
            reference,
            task=replace(reference.task, task_set_hash="d" * 64),
        )
        with self.assertRaises(GateExecutionError):
            run_offline_gate(
                mismatched,
                candidate,
                other,
                settings=_settings(),
                runtime=_runtime(),
                plan_evidence=_evidence(),
            )

    def test_plan_evidence_round_trip_preserves_vocabulary_size_and_specs(
        self,
    ) -> None:
        evidence = _evidence(
            kl_approximation="full",
            kl_vocabulary_size=32000,
            task_plan_specs=_task_plan_specs(),
        )
        payload = plan_evidence_to_dict(evidence)
        restored = plan_evidence_from_dict(payload)
        self.assertEqual(restored.kl_vocabulary_size, 32000)
        self.assertEqual(restored.task_plan_specs, _task_plan_specs())

    def test_unknown_validation_provenance_is_rejected(self) -> None:
        with self.assertRaises(GateExecutionError):
            _evidence(validation_provenance="not-a-real-provenance")

    def test_validated_provenance_with_fake_runtime_is_execution_error(self) -> None:
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(task_set, run_seed=8)
        with self.assertRaises(GateExecutionError):
            run_offline_gate(
                reference,
                candidate,
                task_set,
                settings=_settings(),
                runtime=_runtime(),
                plan_evidence=_evidence(validation_provenance="validated"),
            )

    def test_fake_runtime_never_yields_gate_run_evidence(self) -> None:
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(task_set, run_seed=8)
        decision = run_offline_gate(
            reference,
            candidate,
            task_set,
            settings=_settings(),
            runtime=_runtime(),
            plan_evidence=_evidence(),
        )
        self.assertEqual(decision.artifact.evidence_source, "synthetic_fixture")

    def test_repeated_calls_are_deterministic(self) -> None:
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(task_set, run_seed=8)
        settings = _settings()
        evidence = _evidence()
        first = run_offline_gate(
            reference,
            candidate,
            task_set,
            settings=settings,
            runtime=_runtime(),
            plan_evidence=evidence,
        )
        second = run_offline_gate(
            reference,
            candidate,
            task_set,
            settings=settings,
            runtime=_runtime(),
            plan_evidence=evidence,
        )
        self.assertEqual(first.outcome, second.outcome)
        self.assertEqual(first.reason_codes, second.reason_codes)
        self.assertEqual(
            tuple(item.to_dict() for item in first.statistics),
            tuple(item.to_dict() for item in second.statistics),
        )
        self.assertEqual(
            first.reference_configuration_hash,
            run_configuration_hash(reference),
        )
        self.assertEqual(
            first.candidate_configuration_hash,
            run_configuration_hash(candidate),
        )

    def test_source_omits_demo_constants_and_experiments(self) -> None:
        root = (
            __import__("pathlib").Path(__file__).resolve().parents[2]
            / "src"
            / "llm_behavior_ci"
            / "lifecycle"
            / "offline_gate.py"
        )
        source = root.read_text(encoding="utf-8")
        self.assertNotIn("DEMO_", source)
        self.assertNotIn("experiments", source)


if __name__ == "__main__":
    unittest.main()
