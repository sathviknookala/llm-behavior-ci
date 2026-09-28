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
    run_offline_gate,
)
from llm_behavior_ci.records import TokenLogprob
from llm_behavior_ci.runtime.agent import AgentTurn
from llm_behavior_ci.runtime.appworld import EvaluationResult, TaskContext, ToolResult
from llm_behavior_ci.runtime.episode import RuntimeDependencies
from llm_behavior_ci.stats.kl import truncated_next_token_kl
from llm_behavior_ci.tasks.selection import (
    TaskSet,
    canonical_task_set_bytes,
    task_set_hash_from_bytes,
)

_START = datetime(2026, 9, 27, 15, 0, tzinfo=timezone.utc)
_PLAN = "1. open the calendar"
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
        empty_plan: bool = False,
    ) -> None:
        self._clock = clock
        self._plan_text = plan_text
        self._logprobs = logprobs
        self._by_run_seed = by_run_seed or {}
        self._empty_plan = empty_plan
        self.config: RunConfiguration | None = None

    def begin(self, context: TaskContext, config: RunConfiguration) -> None:
        del context
        self.config = config

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
        )
        self.assertEqual(decision.outcome, "PASS")
        self.assertEqual(decision.reason_codes, ())
        self.assertEqual(len(decision.statistics), 3)
        methods = tuple(item.method for item in decision.statistics)
        self.assertEqual(
            methods,
            ("plan_quality_bootstrap", "truncated_plan_kl", "plan_mmd"),
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
        )
        self.assertEqual(decision.outcome, "BLOCK")
        self.assertIn("plan_quality_margin", decision.reason_codes)
        self.assertEqual(decision.public_decision.decision, "block")

    def test_kl_limit_rejection(self) -> None:
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(task_set, run_seed=8)
        decision = run_offline_gate(
            reference,
            candidate,
            task_set,
            settings=_settings(kl_limit_nats=0.0),
            runtime=_runtime(
                by_run_seed={
                    7: (_PLAN, _MATCHING_LOGPROBS),
                    8: (_PLAN, _DIVERGENT_LOGPROBS),
                }
            ),
        )
        self.assertEqual(decision.outcome, "BLOCK")
        self.assertIn("kl_limit", decision.reason_codes)
        kl = next(
            item for item in decision.statistics if item.method == "truncated_plan_kl"
        )
        self.assertGreater(kl.estimate, 0.0)

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
                    7: ("a", _MATCHING_LOGPROBS),
                    8: ("b" * 80, _MATCHING_LOGPROBS),
                }
            ),
        )
        self.assertEqual(decision.outcome, "BLOCK")
        self.assertIn("mmd_rejected", decision.reason_codes)

    def test_unsupported_tokenizer_skips_truncated_kl(self) -> None:
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(
            task_set,
            run_seed=8,
            tokenizer_revision="abcdabcdabcdabcdabcdabcdabcdabcdabcdabcd",
        )
        with patch(
            "llm_behavior_ci.lifecycle.offline_gate.truncated_next_token_kl",
            wraps=truncated_next_token_kl,
        ) as kl_call:
            decision = run_offline_gate(
                reference,
                candidate,
                task_set,
                settings=_settings(),
                runtime=_runtime(),
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
            "llm_behavior_ci.lifecycle.offline_gate.truncated_next_token_kl",
            wraps=truncated_next_token_kl,
        ) as kl_call:
            decision = run_offline_gate(
                reference,
                candidate,
                task_set,
                settings=_settings(),
                runtime=_runtime(
                    by_run_seed={
                        7: (_PLAN, _MATCHING_LOGPROBS),
                        8: (_PLAN, _MISALIGNED_LOGPROBS),
                    }
                ),
            )
        kl_call.assert_not_called()
        self.assertEqual(decision.outcome, "BLOCK")
        self.assertIn("kl_alignment_failed", decision.reason_codes)
        methods = tuple(item.method for item in decision.statistics)
        self.assertEqual(methods, ("plan_quality_bootstrap", "plan_mmd"))

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
            )
        with self.assertRaises(GateExecutionError):
            run_offline_gate(
                reference,
                candidate,
                task_set,
                settings=_settings(plan_format_version="plan-v2"),
                runtime=_runtime(),
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
            )

    def test_repeated_calls_are_deterministic(self) -> None:
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(task_set, run_seed=8)
        settings = _settings()
        first = run_offline_gate(
            reference,
            candidate,
            task_set,
            settings=settings,
            runtime=_runtime(),
        )
        second = run_offline_gate(
            reference,
            candidate,
            task_set,
            settings=settings,
            runtime=_runtime(),
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
