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
from llm_behavior_ci.runtime.episode import RuntimeDependencies, run_pair
from llm_behavior_ci.records import TokenLogprob

_START = datetime(2026, 9, 27, 15, 0, tzinfo=timezone.utc)
_LOGPROBS = ((TokenLogprob(token_id=7, logprob=-0.5, rank=0),),)


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


def _reference() -> RunConfiguration:
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


def _settings(**overrides: object) -> CanarySettings:
    values: dict[str, object] = {
        "fraction": 1.0,
        "outcome_delay_seconds": 0.0,
        "harm_margin": 0.1,
        "stopping_rule": StoppingRule(
            name="fixed_window",
            alpha=0.05,
            horizon_episodes=1,
        ),
    }
    values.update(overrides)
    return CanarySettings(**values)


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
    def __init__(
        self,
        task_id: str,
        *,
        success: bool = True,
        fail: bool = False,
        skip_eval: bool = False,
    ) -> None:
        self.task_id = task_id
        self.success = success
        self.fail = fail
        self.skip_eval = skip_eval
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
        if self.fail:
            raise RuntimeError("forced failure")
        return ToolResult(
            output_text="ok",
            error_message=None,
            recoverable=False,
            app_name=None,
            api_name=None,
        )

    def evaluate(self) -> EvaluationResult:
        if self.skip_eval:
            raise AssertionError("evaluate should not run")
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
    def __init__(self, clock: Clock, *, candidate_fail: bool = False) -> None:
        self._clock = clock
        self._candidate_fail = candidate_fail
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


def _runtime(
    clock: Clock,
    *,
    candidate_success: bool = True,
    candidate_fail: bool = False,
) -> RuntimeDependencies:
    worlds: list[World] = []

    def factory(task_id: str) -> World:
        if not worlds:
            world = World(task_id, success=True)
        else:
            world = World(
                task_id,
                success=candidate_success,
                fail=candidate_fail,
            )
        worlds.append(world)
        return world

    return RuntimeDependencies(
        session_factory=factory,
        agent=PairAgent(clock, candidate_fail=candidate_fail),
        clock=clock,
    )


def _passing_gate(reference: RunConfiguration, candidate: RunConfiguration) -> FakeGate:
    return FakeGate(
        outcome="PASS",
        reason_codes=(),
        reference_configuration_hash=run_configuration_hash(reference),
        candidate_configuration_hash=run_configuration_hash(candidate),
        task_set_hash=reference.task.task_set_hash,
        reference_protocol_hash=reference.protocol_hash,
        candidate_protocol_hash=candidate.protocol_hash,
    )


def _controller(
    *,
    settings: CanarySettings | None = None,
    clock: Clock | None = None,
) -> tuple[CanaryController, RunConfiguration, RunConfiguration, Clock]:
    reference = _reference()
    candidate = _candidate(reference)
    tick = clock or Clock()
    controller = CanaryController(
        reference,
        candidate,
        settings=settings or _settings(),
        clock=tick,
    )
    return controller, reference, candidate, tick


def _run_execute_pair(
    reference: RunConfiguration,
    candidate: RunConfiguration,
    clock: Clock,
    *,
    candidate_success: bool = True,
    candidate_fail: bool = False,
):
    return run_pair(
        "task-1",
        reference,
        candidate,
        reference_run=new_run_identity(reference),
        candidate_run=new_run_identity(candidate),
        runtime=_runtime(
            clock,
            candidate_success=candidate_success,
            candidate_fail=candidate_fail,
        ),
        mode="execute",
    )


class CanaryControllerTests(unittest.TestCase):
    def test_rejects_unsupported_stopping_rule(self) -> None:
        reference = _reference()
        candidate = _candidate(reference)
        with self.assertRaises(CanaryRejected):
            CanaryController(
                reference,
                candidate,
                settings=_settings(
                    stopping_rule=StoppingRule(
                        name="cusum",
                        alpha=0.05,
                        horizon_episodes=3,
                    )
                ),
                clock=Clock(),
            )

    def test_invalid_transitions(self) -> None:
        controller, reference, candidate, clock = _controller()
        with self.assertRaises(CanaryRejected):
            controller.begin_candidate_episode()
        with self.assertRaises(CanaryRejected):
            controller.observe(
                _run_execute_pair(reference, candidate, clock)
            )
        controller.start(_passing_gate(reference, candidate))
        self.assertEqual(controller.snapshot().state, "GATE_PASSED")
        with self.assertRaises(CanaryRejected):
            controller.start(_passing_gate(reference, candidate))
        controller.begin_candidate_episode()
        pair = _run_execute_pair(reference, candidate, clock)
        decision = controller.observe(pair)
        self.assertEqual(decision.action, "promote")
        with self.assertRaises(CanaryRejected):
            controller.begin_candidate_episode()
        with self.assertRaises(CanaryRejected):
            controller.observe(_run_execute_pair(reference, candidate, clock))

    def test_mismatched_gate_hashes_leave_created(self) -> None:
        controller, reference, candidate, _clock = _controller()
        bad = FakeGate(
            outcome="PASS",
            reason_codes=(),
            reference_configuration_hash="f" * 64,
            candidate_configuration_hash=run_configuration_hash(candidate),
            task_set_hash=reference.task.task_set_hash,
            reference_protocol_hash=reference.protocol_hash,
            candidate_protocol_hash=candidate.protocol_hash,
        )
        with self.assertRaises(CanaryRejected):
            controller.start(bad)
        self.assertEqual(controller.snapshot().state, "CREATED")
        blocked = FakeGate(
            outcome="BLOCK",
            reason_codes=("kl",),
            reference_configuration_hash=run_configuration_hash(reference),
            candidate_configuration_hash=run_configuration_hash(candidate),
            task_set_hash=reference.task.task_set_hash,
            reference_protocol_hash=reference.protocol_hash,
            candidate_protocol_hash=candidate.protocol_hash,
        )
        with self.assertRaises(CanaryRejected):
            controller.start(blocked)
        self.assertEqual(controller.snapshot().state, "CREATED")

    def test_duplicate_observation_rejected(self) -> None:
        controller, reference, candidate, clock = _controller(
            settings=_settings(
                stopping_rule=StoppingRule(
                    name="fixed_window",
                    alpha=0.05,
                    horizon_episodes=5,
                )
            )
        )
        controller.start(_passing_gate(reference, candidate))
        controller.begin_candidate_episode()
        controller.begin_candidate_episode()
        pair = _run_execute_pair(reference, candidate, clock)
        first = controller.observe(pair)
        self.assertEqual(first.action, "continue")
        with self.assertRaises(CanaryRejected):
            controller.observe(pair)
        self.assertEqual(controller.snapshot().candidate_episodes_served, 1)
        self.assertEqual(controller.snapshot().outstanding, 1)

    def test_candidate_failure_counted_and_not_imputed(self) -> None:
        controller, reference, candidate, clock = _controller(
            settings=_settings(
                stopping_rule=StoppingRule(
                    name="fixed_window",
                    alpha=0.05,
                    horizon_episodes=2,
                )
            )
        )
        controller.start(_passing_gate(reference, candidate))
        controller.begin_candidate_episode()
        pair = _run_execute_pair(
            reference,
            candidate,
            clock,
            candidate_fail=True,
        )
        self.assertEqual(pair.candidate.status, "failed")
        self.assertIsNone(pair.candidate.evaluator_outcome)
        decision = controller.observe(pair)
        self.assertEqual(decision.action, "continue")
        self.assertIsNone(decision.evidence)
        self.assertIsNone(decision.public_decision)
        snap = controller.snapshot()
        self.assertEqual(snap.candidate_episodes_served, 1)
        self.assertEqual(snap.candidate_episodes_failed, 1)
        self.assertEqual(snap.state, "CANARY_ACTIVE")

    def test_rollback_includes_outstanding_in_served_before_rollback(self) -> None:
        controller, reference, candidate, clock = _controller()
        controller.start(_passing_gate(reference, candidate))
        controller.begin_candidate_episode()
        controller.begin_candidate_episode()
        controller.begin_candidate_episode()
        pair = _run_execute_pair(
            reference,
            candidate,
            clock,
            candidate_success=False,
        )
        decision = controller.observe(pair)
        self.assertEqual(decision.action, "rollback")
        self.assertEqual(decision.state, "ROLLED_BACK")
        snap = decision.snapshot
        self.assertEqual(snap.outstanding, 2)
        self.assertEqual(snap.in_flight_at_rollback, 2)
        self.assertEqual(snap.served_before_rollback, 3)
        self.assertEqual(snap.candidate_episodes_served, 1)
        self.assertIsNotNone(snap.rollback_at)
        self.assertEqual(
            snap.serving_configuration_hash,
            run_configuration_hash(reference),
        )

    def test_promotion_at_horizon_keeps_previous_production_hash(self) -> None:
        controller, reference, candidate, clock = _controller()
        reference_hash = run_configuration_hash(reference)
        candidate_hash = run_configuration_hash(candidate)
        controller.start(_passing_gate(reference, candidate))
        controller.begin_candidate_episode()
        pair = _run_execute_pair(reference, candidate, clock)
        decision = controller.observe(pair)
        self.assertEqual(decision.action, "promote")
        self.assertEqual(decision.state, "PROMOTED")
        snap = decision.snapshot
        self.assertEqual(snap.previous_production_configuration_hash, reference_hash)
        self.assertEqual(snap.serving_configuration_hash, candidate_hash)
        self.assertEqual(snap.candidate_configuration_hash, candidate_hash)
        self.assertIsNotNone(snap.promoted_at)
        self.assertIsNotNone(decision.public_decision)
        self.assertEqual(decision.public_decision.decision, "promote")
        self.assertEqual(decision.public_decision.tier, "canary")

    def test_observe_without_pair_execution_rejected(self) -> None:
        controller, reference, candidate, clock = _controller(
            settings=_settings(
                stopping_rule=StoppingRule(
                    name="fixed_window",
                    alpha=0.05,
                    horizon_episodes=5,
                )
            )
        )
        controller.start(_passing_gate(reference, candidate))
        controller.begin_candidate_episode()
        pair = _run_execute_pair(reference, candidate, clock)
        from llm_behavior_ci.records import PairedResult

        forged = PairedResult(
            reference=replace(
                pair.reference,
                episode=replace(pair.reference.episode, pair_id="a" * 32),
            ),
            candidate=replace(
                pair.candidate,
                episode=replace(pair.candidate.episode, pair_id="a" * 32),
            ),
        )
        before = controller.snapshot()
        with self.assertRaises(CanaryRejected):
            controller.observe(forged)
        after = controller.snapshot()
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
