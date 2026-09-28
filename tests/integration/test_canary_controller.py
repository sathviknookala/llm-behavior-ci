import unittest
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

from llm_behavior_ci.config import (
    CanarySettings,
    RunConfiguration,
    StoppingRule,
    new_run_identity,
    run_configuration_hash,
)
from llm_behavior_ci.lifecycle.canary import CanaryController, CanaryRejected
from llm_behavior_ci.runtime.agent import AgentTurn
from llm_behavior_ci.runtime.appworld import (
    EvaluationResult,
    TaskContext,
    ToolResult,
)
from llm_behavior_ci.runtime.episode import (
    RuntimeDependencies,
    pair_execution,
    run_pair,
)
from llm_behavior_ci.records import TokenLogprob

_START = datetime(2026, 9, 27, 15, 0, tzinfo=timezone.utc)
_LOGPROBS = ((TokenLogprob(token_id=7, logprob=-0.4, rank=0),),)


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


@dataclass(frozen=True)
class FakeGate:
    outcome: str
    reason_codes: tuple[str, ...]
    reference_configuration_hash: str
    candidate_configuration_hash: str
    task_set_hash: str
    reference_protocol_hash: str
    candidate_protocol_hash: str


class Clock:
    def __init__(self) -> None:
        self.current = _START

    def __call__(self) -> datetime:
        value = self.current
        self.current += timedelta(seconds=1)
        return value


class World:
    def __init__(self, task_id: str, *, success: bool = True) -> None:
        self.task_id = task_id
        self.success = success
        self.close_count = 0

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
        self.close_count += 1


class PairAgent:
    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._begins = 0
        self._step = 0
        self._reference = True

    def begin(self, context: TaskContext, config: RunConfiguration) -> None:
        del context, config
        self._begins += 1
        self._reference = self._begins % 2 == 1
        self._step = 0

    def next_turn(self, *, tool_output: str | None) -> AgentTurn:
        del tool_output
        self._step += 1
        started_at = self._clock()
        if self._step > 1:
            return AgentTurn(
                prompt_text="plan",
                output_text="STOP",
                top_k_logprobs=_LOGPROBS,
                latency_seconds=0.1,
                started_at=started_at,
                action=None,
                app_name=None,
                api_name=None,
            )
        action = "reference.lookup()" if self._reference else "candidate.lookup()"
        return AgentTurn(
            prompt_text="plan",
            output_text=action,
            top_k_logprobs=_LOGPROBS,
            latency_seconds=0.1,
            started_at=started_at,
            action=action,
            app_name=None,
            api_name="lookup",
        )


def _configs() -> tuple[RunConfiguration, RunConfiguration]:
    reference = RunConfiguration.from_dict(_payload())
    candidate = replace(
        reference,
        agent=replace(
            reference.agent,
            prompt=replace(reference.agent.prompt, prompt_version="prompt-v2"),
        ),
        run_seed=reference.run_seed + 1,
    )
    return reference, candidate


def _runtime(clock: Clock, *, candidate_success: bool) -> RuntimeDependencies:
    worlds: list[World] = []

    def factory(task_id: str) -> World:
        if not worlds:
            world = World(task_id, success=True)
        else:
            world = World(task_id, success=candidate_success)
        worlds.append(world)
        return world

    return RuntimeDependencies(
        session_factory=factory,
        agent=PairAgent(clock),
        clock=clock,
    )


class CanaryControllerIntegrationTests(unittest.TestCase):
    def test_run_pair_feeds_controller_through_pair_execution(self) -> None:
        reference, candidate = _configs()
        clock = Clock()
        settings = CanarySettings(
            fraction=1.0,
            outcome_delay_seconds=0.0,
            harm_margin=0.1,
            stopping_rule=StoppingRule(
                name="sequential_canary",
                alpha=0.05,
                horizon_episodes=2,
            ),
        )
        controller = CanaryController(
            reference,
            candidate,
            settings=settings,
            clock=clock,
        )
        gate = FakeGate(
            outcome="PASS",
            reason_codes=(),
            reference_configuration_hash=run_configuration_hash(reference),
            candidate_configuration_hash=run_configuration_hash(candidate),
            task_set_hash=reference.task.task_set_hash,
            reference_protocol_hash=reference.protocol_hash,
            candidate_protocol_hash=candidate.protocol_hash,
        )
        controller.start(gate)
        controller.begin_candidate_episode()
        first = run_pair(
            "task-1",
            reference,
            candidate,
            reference_run=new_run_identity(reference),
            candidate_run=new_run_identity(candidate),
            runtime=_runtime(clock, candidate_success=True),
            mode="execute",
        )
        recorded = pair_execution(first)
        self.assertEqual(recorded.pair_id, first.reference.episode.pair_id)
        decision = controller.observe(first)
        self.assertEqual(decision.action, "continue")
        self.assertIsNotNone(decision.evidence)
        self.assertEqual(decision.evidence.method, "sequential_canary")
        self.assertEqual(
            decision.snapshot.previous_production_configuration_hash,
            run_configuration_hash(reference),
        )
        self.assertEqual(
            decision.snapshot.serving_configuration_hash,
            run_configuration_hash(reference),
        )

        controller.begin_candidate_episode()
        second = run_pair(
            "task-2",
            reference,
            candidate,
            reference_run=new_run_identity(reference),
            candidate_run=new_run_identity(candidate),
            runtime=_runtime(clock, candidate_success=True),
            mode="execute",
        )
        promoted = controller.observe(second)
        self.assertEqual(promoted.action, "promote")
        self.assertEqual(promoted.state, "PROMOTED")
        self.assertEqual(
            promoted.snapshot.previous_production_configuration_hash,
            run_configuration_hash(reference),
        )
        self.assertEqual(
            promoted.snapshot.serving_configuration_hash,
            run_configuration_hash(candidate),
        )
        with self.assertRaises(CanaryRejected):
            controller.observe(
                run_pair(
                    "task-3",
                    reference,
                    candidate,
                    reference_run=new_run_identity(reference),
                    candidate_run=new_run_identity(candidate),
                    runtime=_runtime(clock, candidate_success=True),
                    mode="execute",
                )
            )

    def test_rollback_with_outstanding_work_via_run_pair(self) -> None:
        reference, candidate = _configs()
        clock = Clock()
        controller = CanaryController(
            reference,
            candidate,
            settings=CanarySettings(
                fraction=1.0,
                outcome_delay_seconds=0.0,
                harm_margin=0.1,
                stopping_rule=StoppingRule(
                    name="fixed_window",
                    alpha=0.05,
                    horizon_episodes=1,
                ),
            ),
            clock=clock,
        )
        controller.start(
            FakeGate(
                outcome="PASS",
                reason_codes=(),
                reference_configuration_hash=run_configuration_hash(reference),
                candidate_configuration_hash=run_configuration_hash(candidate),
                task_set_hash=reference.task.task_set_hash,
                reference_protocol_hash=reference.protocol_hash,
                candidate_protocol_hash=candidate.protocol_hash,
            )
        )
        controller.begin_candidate_episode()
        controller.begin_candidate_episode()
        pair = run_pair(
            "task-1",
            reference,
            candidate,
            reference_run=new_run_identity(reference),
            candidate_run=new_run_identity(candidate),
            runtime=_runtime(clock, candidate_success=False),
            mode="execute",
        )
        pair_execution(pair)
        decision = controller.observe(pair)
        self.assertEqual(decision.action, "rollback")
        self.assertEqual(decision.snapshot.in_flight_at_rollback, 1)
        self.assertEqual(decision.snapshot.served_before_rollback, 2)
        self.assertEqual(
            decision.snapshot.previous_production_configuration_hash,
            run_configuration_hash(reference),
        )
        self.assertEqual(
            decision.snapshot.serving_configuration_hash,
            run_configuration_hash(reference),
        )


if __name__ == "__main__":
    unittest.main()
