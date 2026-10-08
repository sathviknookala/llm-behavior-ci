import copy
import math
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from llm_behavior_ci.config import GateSettings, RunConfiguration
from llm_behavior_ci.lifecycle.offline_gate import PlanEvidenceInputs, run_offline_gate
from llm_behavior_ci.records import TokenLogprob
from llm_behavior_ci.runtime.agent import AgentTurn
from llm_behavior_ci.runtime.appworld import EvaluationResult, TaskContext, ToolResult
from llm_behavior_ci.runtime.episode import RuntimeDependencies
from llm_behavior_ci.runtime.scoring import score_top_k
from llm_behavior_ci.stats.bootstrap import clustered_paired_bootstrap, paired_bootstrap
from llm_behavior_ci.stats.kl import NextTokenKLError, next_token_kl
from llm_behavior_ci.stats.mmd import MMDError, mmd_permutation_test
from llm_behavior_ci.tasks.selection import (
    TaskSet,
    canonical_task_set_bytes,
    task_set_hash_from_bytes,
)

_START = datetime(2026, 9, 27, 15, 0, tzinfo=timezone.utc)
_PLAN = "1. open the calendar"
_LOGPROBS = (
    (
        TokenLogprob(token_id=7, logprob=-0.2, rank=0),
        TokenLogprob(token_id=9, logprob=-1.5, rank=1),
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


_GATE_TASKS = (
    ("task-a", "scenario-1"),
    ("task-b", None),
    ("task-c", "scenario-3"),
    ("task-d", "scenario-4"),
    ("task-e", "scenario-5"),
    ("task-f", "scenario-6"),
)
_GATE_CLUSTERS = ("scenario-1", "task-b", "scenario-3", "scenario-4", "scenario-5", "scenario-6")


def _task_set() -> TaskSet:
    tasks = _GATE_TASKS
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
        task_count=len(tasks),
        scenario_count=len(tasks),
        task_ids=tuple(task_id for task_id, _ in tasks),
        scenario_ids=tuple(scenario_id for _, scenario_id in tasks),
        task_set_hash=task_set_hash_from_bytes(payload),
    )


def _config(task_set: TaskSet, *, run_seed: int = 7) -> RunConfiguration:
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
    return RunConfiguration.from_dict(payload)


def _settings() -> GateSettings:
    return GateSettings(
        confidence_level=0.9,
        bootstrap_resamples=40,
        score_margin=-0.02,
        kl_limit_nats=0.05,
        mmd_bandwidth=1.0,
        mmd_permutations=99,
        mmd_alpha=0.05,
        plan_format_version="plan-v1",
    )


def _evidence() -> PlanEvidenceInputs:
    return PlanEvidenceInputs(
        plan_format_version="plan-v1",
        plan_quality_features=("numbered_step_count", "token_count"),
        plan_quality_weights=(1.0, 0.1),
        mmd_features=("char_count", "numbered_step_count", "model_step_count"),
        kl_approximation="top_k",
        required_statistics=("plan_quality", "kl", "mmd"),
        validation_provenance="synthetic_fixture",
    )


class FakeSession:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id

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
        return None


class PlanAgent:
    def __init__(self, clock) -> None:
        self._clock = clock
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
        return AgentTurn(
            prompt_text="plan the next action",
            output_text=_PLAN,
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


def _clock():
    current = _START

    def tick() -> datetime:
        nonlocal current
        value = current
        current = current + timedelta(seconds=1)
        return value

    return tick


class OfflineGateTests(unittest.TestCase):
    def test_paired_bootstrap_candidate_vs_production(self) -> None:
        result = paired_bootstrap(
            candidate=(0.6, 0.8, 1.0, 1.2),
            production=(1.0, 1.0, 1.0, 1.0),
            confidence_level=0.95,
            resamples=1_000,
            seed=7,
        )
        self.assertAlmostEqual(result.mean_difference, -0.1)
        self.assertLess(result.confidence_low, result.mean_difference)
        self.assertGreater(result.confidence_high, result.mean_difference)

    def test_next_token_kl_candidate_vs_production(self) -> None:
        result = next_token_kl(
            production_log_probabilities=((math.log(0.5), math.log(0.5)),),
            candidate_log_probabilities=((math.log(0.25), math.log(0.75)),),
        )
        expected = 0.5 * math.log(2.0) + 0.5 * math.log(2.0 / 3.0)
        self.assertAlmostEqual(result.mean_kl_nats, expected)
        self.assertEqual(result.position_kl_nats, (result.mean_kl_nats,))

    def test_next_token_kl_rejects_underflowed_support_mismatch(self) -> None:
        with self.assertRaises(NextTokenKLError):
            next_token_kl(
                production_log_probabilities=((0.0, -1_000.0),),
                candidate_log_probabilities=((0.0, -math.inf),),
            )

    def test_mmd_candidate_vs_production(self) -> None:
        production = tuple((value / 10.0,) for value in range(8))
        candidate = tuple((5.0 + value / 10.0,) for value in range(8))
        result = mmd_permutation_test(
            production=production,
            candidate=candidate,
            bandwidth=1.0,
            permutations=999,
            seed=7,
        )
        self.assertGreater(result.mmd_squared, 0.0)
        self.assertLessEqual(result.p_value, 0.05)

        null_result = mmd_permutation_test(
            production=production,
            candidate=production,
            bandwidth=1.0,
            permutations=999,
            seed=7,
        )
        self.assertAlmostEqual(null_result.mmd_squared, 0.0)
        self.assertAlmostEqual(null_result.p_value, 1.0)

    def test_mmd_rejects_unrepresentable_bandwidths(self) -> None:
        samples = ((0.0,), (1.0,))
        with self.assertRaises(MMDError):
            mmd_permutation_test(
                production=samples,
                candidate=samples,
                bandwidth=1e-300,
                permutations=9,
                seed=7,
            )
        with self.assertRaises(MMDError):
            mmd_permutation_test(
                production=samples,
                candidate=samples,
                bandwidth=1e308,
                permutations=9,
                seed=7,
            )

    def test_orchestrator_calls_required_checks(self) -> None:
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(task_set, run_seed=8)
        clock = _clock()
        runtime = RuntimeDependencies(
            session_factory=lambda task_id: FakeSession(task_id),
            agent=PlanAgent(clock),
            clock=clock,
        )
        with (
            patch(
                "llm_behavior_ci.lifecycle.offline_gate.clustered_paired_bootstrap",
                wraps=clustered_paired_bootstrap,
            ) as bootstrap_call,
            patch(
                "llm_behavior_ci.lifecycle.offline_gate.score_top_k",
                wraps=score_top_k,
            ) as kl_call,
            patch(
                "llm_behavior_ci.lifecycle.offline_gate.mmd_permutation_test",
                wraps=mmd_permutation_test,
            ) as mmd_call,
        ):
            decision = run_offline_gate(
                reference,
                candidate,
                task_set,
                settings=_settings(),
                runtime=runtime,
                plan_evidence=_evidence(),
            )
        bootstrap_call.assert_called_once()
        kl_call.assert_called_once()
        mmd_call.assert_called_once()
        self.assertEqual(mmd_call.call_args.kwargs["clusters"], _GATE_CLUSTERS)
        self.assertEqual(decision.outcome, "PASS")
        self.assertEqual(decision.reason_codes, ())

    def test_orchestrator_blocks_when_score_margin_rejects(self) -> None:
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(task_set, run_seed=8)
        clock = _clock()
        runtime = RuntimeDependencies(
            session_factory=lambda task_id: FakeSession(task_id),
            agent=PlanAgent(clock),
            clock=clock,
        )
        decision = run_offline_gate(
            reference,
            candidate,
            task_set,
            settings=GateSettings(
                confidence_level=0.9,
                bootstrap_resamples=40,
                score_margin=0.01,
                kl_limit_nats=0.05,
                mmd_bandwidth=1.0,
                mmd_permutations=99,
                mmd_alpha=0.05,
                plan_format_version="plan-v1",
            ),
            runtime=runtime,
            plan_evidence=_evidence(),
        )
        self.assertEqual(decision.outcome, "BLOCK")
        self.assertIn("plan_quality_margin", decision.reason_codes)


if __name__ == "__main__":
    unittest.main()
