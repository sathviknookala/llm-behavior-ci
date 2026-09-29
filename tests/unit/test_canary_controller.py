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
from llm_behavior_ci.lifecycle.canary import (
    CanaryController,
    CanaryRejected,
    assign_canary,
)
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
        "metric_orientation": "higher_is_better",
        "promotion_policy": "horizon_reached_without_harm",
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
    reference_success: bool = True,
) -> RuntimeDependencies:
    worlds: list[World] = []

    def factory(task_id: str) -> World:
        if not worlds:
            world = World(task_id, success=reference_success)
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
    reference_success: bool = True,
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
            reference_success=reference_success,
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
        self.assertEqual(
            decision.public_decision.decision,
            "promote_horizon_reached_without_harm",
        )
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

    def test_assign_canary_stable_and_tracks_fraction(self) -> None:
        key = "task-42:scenario-a"
        first = assign_canary(key, fraction=0.5, seed=17)
        second = assign_canary(key, fraction=0.5, seed=17)
        self.assertEqual(first, second)
        self.assertTrue(
            any(
                assign_canary(f"key-{index}", fraction=0.5, seed=17)
                != assign_canary(f"key-{index}", fraction=0.5, seed=18)
                for index in range(64)
            )
        )
        with self.assertRaises(CanaryRejected):
            assign_canary(key, fraction=0.0, seed=1)
        with self.assertRaises(CanaryRejected):
            assign_canary(key, fraction=1.1, seed=1)
        hits = sum(
            1
            for index in range(400)
            if assign_canary(f"key-{index}", fraction=0.5, seed=99)
        )
        self.assertGreater(hits, 50)
        self.assertLess(hits, 350)

    def test_manual_and_automatic_rollback_share_transition(self) -> None:
        reference_hash = run_configuration_hash(_reference())

        manual, reference, candidate, _clock = _controller(
            settings=_settings(
                stopping_rule=StoppingRule(
                    name="fixed_window",
                    alpha=0.05,
                    horizon_episodes=5,
                )
            )
        )
        manual.start(_passing_gate(reference, candidate))
        manual.begin_candidate_episode()
        before = manual.snapshot()
        self.assertEqual(before.outstanding, 1)
        decision = manual.rollback("operator_request", manual=True)
        self.assertEqual(decision.action, "rollback")
        self.assertEqual(decision.state, "ROLLED_BACK")
        snap = decision.snapshot
        self.assertTrue(snap.rollback_manual)
        self.assertEqual(snap.rollback_reason, "operator_request")
        self.assertEqual(snap.serving_configuration_hash, reference_hash)
        self.assertEqual(snap.previous_production_configuration_hash, reference_hash)
        self.assertEqual(snap.in_flight_at_rollback, 1)
        self.assertEqual(snap.served_before_rollback, 1)
        with self.assertRaises(CanaryRejected):
            manual.begin_candidate_episode()

        automatic, reference, candidate, clock = _controller()
        automatic.start(_passing_gate(reference, candidate))
        automatic.begin_candidate_episode()
        automatic.begin_candidate_episode()
        pair = _run_execute_pair(
            reference,
            candidate,
            clock,
            candidate_success=False,
        )
        auto_decision = automatic.observe(pair)
        self.assertEqual(auto_decision.action, "rollback")
        auto_snap = auto_decision.snapshot
        self.assertFalse(auto_snap.rollback_manual)
        self.assertEqual(auto_snap.rollback_reason, "stopping_rule_alarm")
        self.assertEqual(auto_snap.serving_configuration_hash, reference_hash)
        with self.assertRaises(CanaryRejected):
            automatic.begin_candidate_episode()

    def test_complete_outstanding_after_rollback_does_not_double_count(self) -> None:
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
        first = _run_execute_pair(reference, candidate, clock)
        decision = controller.observe(first)
        self.assertEqual(decision.action, "continue")
        rolled = controller.rollback("operator_request", manual=True)
        self.assertEqual(rolled.snapshot.outstanding, 1)
        self.assertEqual(rolled.snapshot.candidate_episodes_served, 1)
        second = _run_execute_pair(reference, candidate, clock)
        after = controller.complete_outstanding(second)
        self.assertEqual(after.outstanding, 0)
        self.assertEqual(after.candidate_episodes_served, 2)
        self.assertEqual(after.state, "ROLLED_BACK")
        with self.assertRaises(CanaryRejected):
            controller.complete_outstanding(second)
        self.assertEqual(controller.snapshot().candidate_episodes_served, 2)
        self.assertEqual(controller.snapshot().outstanding, 0)

    def test_abort_outstanding_clears_begin_without_pair(self) -> None:
        controller, reference, candidate, _clock = _controller(
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
        self.assertEqual(controller.snapshot().outstanding, 1)
        after = controller.abort_outstanding()
        self.assertEqual(after.outstanding, 0)
        self.assertEqual(after.candidate_episodes_served, 0)
        self.assertEqual(after.candidate_episodes_failed, 0)
        self.assertEqual(after.candidate_episodes_evaluator_unsuccessful, 0)
        self.assertEqual(after.state, "CANARY_ACTIVE")

    def test_evaluator_and_runtime_failures_use_separate_counters(self) -> None:
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
        runtime_pair = _run_execute_pair(
            reference,
            candidate,
            clock,
            candidate_fail=True,
        )
        runtime_decision = controller.observe(runtime_pair)
        self.assertEqual(runtime_decision.action, "continue")
        runtime_snap = runtime_decision.snapshot
        self.assertEqual(runtime_snap.candidate_episodes_failed, 1)
        self.assertEqual(runtime_snap.candidate_episodes_evaluator_unsuccessful, 0)

        controller.begin_candidate_episode()
        eval_pair = _run_execute_pair(
            reference,
            candidate,
            clock,
            candidate_success=False,
        )
        self.assertEqual(eval_pair.candidate.status, "completed")
        self.assertIsNotNone(eval_pair.candidate.evaluator_outcome)
        assert eval_pair.candidate.evaluator_outcome is not None
        self.assertFalse(eval_pair.candidate.evaluator_outcome.success)
        eval_decision = controller.observe(eval_pair)
        self.assertEqual(eval_decision.action, "continue")
        eval_snap = eval_decision.snapshot
        self.assertEqual(eval_snap.candidate_episodes_failed, 1)
        self.assertEqual(eval_snap.candidate_episodes_evaluator_unsuccessful, 1)

    def test_promotion_records_monitor_reset_identity(self) -> None:
        controller, reference, candidate, clock = _controller()
        reference_hash = run_configuration_hash(reference)
        candidate_hash = run_configuration_hash(candidate)
        controller.start(_passing_gate(reference, candidate))
        controller.begin_candidate_episode()
        decision = controller.observe(_run_execute_pair(reference, candidate, clock))
        self.assertEqual(decision.action, "promote")
        snap = decision.snapshot
        self.assertTrue(snap.monitoring_reset_required)
        self.assertEqual(snap.previous_production_configuration_hash, reference_hash)
        self.assertEqual(snap.promoted_configuration_hash, candidate_hash)
        self.assertEqual(snap.serving_configuration_hash, candidate_hash)
        with self.assertRaises(CanaryRejected):
            controller.begin_candidate_episode()

    def test_paired_difference_cs_rollback_on_clear_harm(self) -> None:
        controller, reference, candidate, clock = _controller(
            settings=_settings(
                harm_margin=0.05,
                stopping_rule=StoppingRule(
                    name="paired_difference_cs",
                    alpha=0.05,
                    horizon_episodes=50,
                ),
            )
        )
        controller.start(_passing_gate(reference, candidate))
        controller.begin_candidate_episode()
        decision = None
        for _ in range(30):
            controller.begin_candidate_episode()
            pair = _run_execute_pair(
                reference,
                candidate,
                clock,
                candidate_success=False,
            )
            decision = controller.observe(pair)
            if decision.action == "rollback":
                break
        assert decision is not None
        self.assertEqual(decision.action, "rollback")
        self.assertIsNotNone(decision.evidence)
        self.assertEqual(decision.evidence.direction, "harmful")
        self.assertTrue(decision.evidence.alarm)
        self.assertIsNotNone(decision.public_decision)
        self.assertEqual(decision.public_decision.decision, "rollback")
        self.assertEqual(
            decision.public_decision.evidence[0].direction, "harmful"
        )
        self.assertEqual(decision.snapshot.in_flight_at_rollback, 1)
        self.assertGreaterEqual(decision.snapshot.served_before_rollback, 1)

    def test_paired_difference_cs_improvement_never_rolls_back(self) -> None:
        controller, reference, candidate, clock = _controller(
            settings=_settings(
                harm_margin=0.05,
                stopping_rule=StoppingRule(
                    name="paired_difference_cs",
                    alpha=0.05,
                    horizon_episodes=15,
                ),
            )
        )
        controller.start(_passing_gate(reference, candidate))
        decisions = []
        for _ in range(15):
            controller.begin_candidate_episode()
            pair = _run_execute_pair(
                reference,
                candidate,
                clock,
                candidate_success=True,
                reference_success=False,
            )
            decision = controller.observe(pair)
            decisions.append(decision)
            self.assertNotEqual(decision.action, "rollback")
            self.assertNotEqual(decision.evidence.direction, "harmful")
            self.assertFalse(decision.evidence.alarm)
        self.assertEqual(decisions[0].action, "continue")
        self.assertEqual(decisions[-1].evidence.direction, "beneficial")
        self.assertEqual(decisions[-1].action, "promote")
        self.assertEqual(
            decisions[-1].public_decision.decision,
            "promote_horizon_reached_without_harm",
        )

    def test_paired_difference_cs_continues_when_evidence_insufficient(self) -> None:
        controller, reference, candidate, clock = _controller(
            settings=_settings(
                harm_margin=0.05,
                stopping_rule=StoppingRule(
                    name="paired_difference_cs",
                    alpha=0.05,
                    horizon_episodes=3,
                ),
            )
        )
        controller.start(_passing_gate(reference, candidate))
        controller.begin_candidate_episode()
        pair = _run_execute_pair(reference, candidate, clock, candidate_success=True)
        decision = controller.observe(pair)
        self.assertEqual(decision.action, "continue")
        self.assertEqual(decision.evidence.direction, "insufficient")
        self.assertFalse(decision.evidence.alarm)

    def test_metric_orientation_reversal_flips_harm_direction(self) -> None:
        controller, reference, candidate, clock = _controller(
            settings=_settings(
                harm_margin=0.05,
                metric_orientation="lower_is_better",
                stopping_rule=StoppingRule(
                    name="fixed_window",
                    alpha=0.05,
                    horizon_episodes=1,
                ),
            )
        )
        controller.start(_passing_gate(reference, candidate))
        controller.begin_candidate_episode()
        pair = _run_execute_pair(
            reference,
            candidate,
            clock,
            candidate_success=True,
            reference_success=False,
        )
        decision = controller.observe(pair)
        self.assertEqual(decision.action, "rollback")
        self.assertEqual(decision.evidence.direction, "harmful")

    def test_metric_orientation_default_reads_same_pair_as_improvement(self) -> None:
        controller, reference, candidate, clock = _controller(
            settings=_settings(
                harm_margin=0.05,
                stopping_rule=StoppingRule(
                    name="fixed_window",
                    alpha=0.05,
                    horizon_episodes=1,
                ),
            )
        )
        controller.start(_passing_gate(reference, candidate))
        controller.begin_candidate_episode()
        pair = _run_execute_pair(
            reference,
            candidate,
            clock,
            candidate_success=True,
            reference_success=False,
        )
        decision = controller.observe(pair)
        self.assertEqual(decision.action, "promote")
        self.assertEqual(
            decision.public_decision.decision,
            "promote_horizon_reached_without_harm",
        )


if __name__ == "__main__":
    unittest.main()
