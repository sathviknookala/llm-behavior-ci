import json
import unittest
from datetime import datetime, timezone
from pathlib import Path

import llm_behavior_ci
from llm_behavior_ci.config import EpisodeIdentity, RunIdentity
from llm_behavior_ci.records import (
    PROTECTED_FIELDS,
    AggregateRecord,
    EpisodeResult,
    EvaluatorOutcome,
    LifecycleDecision,
    LocalTaskRef,
    ModelStep,
    MonitorObservation,
    PairedResult,
    RecordError,
    RecordedError,
    StatisticalEvidence,
    TokenLogprob,
    ToolStep,
    assert_public_payload,
    public_record_dict,
)
from llm_behavior_ci.stats.bootstrap import paired_bootstrap
from llm_behavior_ci.stats.kl import next_token_kl
from llm_behavior_ci.stats.mmd import mmd_permutation_test

_HASH_A = "a" * 64
_HASH_B = "b" * 64
_TASK_HASH = "c" * 64
_GIT = "d" * 40
_PAIR = "e" * 32
_START = datetime(2026, 9, 27, 15, 0, tzinfo=timezone.utc)
_MID = datetime(2026, 9, 27, 15, 1, tzinfo=timezone.utc)
_LATER = datetime(2026, 9, 27, 15, 2, tzinfo=timezone.utc)
_END = datetime(2026, 9, 27, 15, 5, tzinfo=timezone.utc)


def _run(config: str = _HASH_A, token: str = "1" * 32) -> RunIdentity:
    return RunIdentity(
        run_id=f"{config}.{token}",
        configuration_hash=config,
        task_set_hash=_TASK_HASH,
        protocol_hash=None,
        git_commit=_GIT,
    )


def _task(
    task_id: str = "local-task",
    scenario_id: str | None = "scenario-1",
    split: str = "dev",
) -> LocalTaskRef:
    return LocalTaskRef(task_id=task_id, scenario_id=scenario_id, split=split)


def _model_step(index: int = 0, started_at: datetime = _MID) -> ModelStep:
    return ModelStep(
        index=index,
        prompt_text="plan the task",
        output_text="open the app",
        top_k_logprobs=(
            (
                TokenLogprob(token_id=3, logprob=-0.5, rank=0),
                TokenLogprob(token_id=4, logprob=float("-inf"), rank=1),
            ),
        ),
        latency_seconds=0.1,
        started_at=started_at,
    )


def _tool_step(
    index: int,
    error: RecordedError | None = None,
    started_at: datetime = _LATER,
) -> ToolStep:
    return ToolStep(
        index=index,
        action="app.lookup()",
        app_name="calendar",
        api_name="lookup",
        output_text="",
        error=error,
        latency_seconds=0.2,
        started_at=started_at,
    )


def _episode(**overrides: object) -> EpisodeResult:
    run = overrides.pop("run") if "run" in overrides else _run()
    if not isinstance(run, RunIdentity):
        raise AssertionError("run")
    pair_id = overrides.pop("pair_id", None)
    episode_token = overrides.pop("episode_token", "2" * 32)
    if not isinstance(episode_token, str):
        raise AssertionError("episode_token")
    if "episode" not in overrides:
        if pair_id is not None and not isinstance(pair_id, str):
            raise AssertionError("pair_id")
        overrides["episode"] = EpisodeIdentity(
            episode_id=f"{run.run_id}.{episode_token}",
            run_id=run.run_id,
            pair_id=pair_id if isinstance(pair_id, str) else None,
        )
    values: dict[str, object] = {
        "episode": overrides["episode"],
        "run": run,
        "task": _task(),
        "mode": "execute",
        "execution_seed": 7,
        "status": "completed",
        "started_at": _START,
        "ended_at": _END,
        "model_steps": (_model_step(),),
        "tool_steps": (),
        "plan_text": None,
        "evaluator_outcome": EvaluatorOutcome(
            success=False,
            passed_requirements=1,
            total_requirements=2,
            difficulty=2,
        ),
        "termination_reason": "agent_stopped",
        "episode_errors": (),
        "role": None,
    }
    values.update(overrides)
    return EpisodeResult(**values)


def _keys(payload: object) -> set[str]:
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            found.add(key)
            found.update(_keys(value))
    elif isinstance(payload, list):
        for item in payload:
            found.update(_keys(item))
    return found


class EvaluatorOutcomeTests(unittest.TestCase):
    def test_requirement_fraction_is_derived_from_counts(self) -> None:
        outcome = EvaluatorOutcome(
            success=False,
            passed_requirements=1,
            total_requirements=2,
            difficulty=1,
        )
        self.assertEqual(outcome.requirement_fraction, 0.5)
        self.assertFalse(outcome.success)
        empty = EvaluatorOutcome(
            success=True,
            passed_requirements=0,
            total_requirements=0,
            difficulty=None,
        )
        self.assertIsNone(empty.requirement_fraction)
        self.assertTrue(empty.success)
        self.assertEqual(
            EvaluatorOutcome(
                success=True,
                passed_requirements=3,
                total_requirements=3,
                difficulty=3,
            ).requirement_fraction,
            1.0,
        )
        self.assertEqual(
            EvaluatorOutcome(
                success=False,
                passed_requirements=0,
                total_requirements=4,
                difficulty=None,
            ).requirement_fraction,
            0.0,
        )

    def test_zero_denominator_rejects_an_imputed_fraction(self) -> None:
        payload = EvaluatorOutcome(
            success=True,
            passed_requirements=0,
            total_requirements=0,
            difficulty=None,
        ).to_dict()
        self.assertIsNone(payload["requirement_fraction"])
        payload["requirement_fraction"] = 0.0
        with self.assertRaises(RecordError):
            EvaluatorOutcome.from_dict(payload)
        payload["requirement_fraction"] = 1.0
        with self.assertRaises(RecordError):
            EvaluatorOutcome.from_dict(payload)

    def test_success_must_match_the_counts(self) -> None:
        with self.assertRaises(RecordError):
            EvaluatorOutcome(
                success=True,
                passed_requirements=1,
                total_requirements=2,
                difficulty=None,
            )
        with self.assertRaises(RecordError):
            EvaluatorOutcome(
                success=False,
                passed_requirements=0,
                total_requirements=0,
                difficulty=None,
            )

    def test_round_trip_checks_the_stored_fraction(self) -> None:
        outcome = EvaluatorOutcome(
            success=False,
            passed_requirements=1,
            total_requirements=2,
            difficulty=None,
        )
        payload = json.loads(json.dumps(outcome.to_dict()))
        self.assertEqual(EvaluatorOutcome.from_dict(payload), outcome)
        payload["requirement_fraction"] = 0.25
        with self.assertRaises(RecordError):
            EvaluatorOutcome.from_dict(payload)
        del payload["requirement_fraction"]
        with self.assertRaises(RecordError):
            EvaluatorOutcome.from_dict(payload)


class EpisodeRecordTests(unittest.TestCase):
    def test_completed_episode_can_carry_a_recoverable_tool_error(self) -> None:
        error = RecordedError(
            source="tool",
            recoverable=True,
            message="calendar was busy",
            step_index=1,
        )
        episode = _episode(
            tool_steps=(_tool_step(1, error),),
        )
        self.assertEqual(episode.status, "completed")
        self.assertEqual(episode.termination_reason, "agent_stopped")
        self.assertFalse(episode.evaluator_outcome.success)
        self.assertEqual(episode.recorded_errors, (error,))
        self.assertTrue(episode.reached_evaluation)
        self.assertIsNotNone(episode.evaluator_outcome)
        self.assertEqual(episode.evaluator_outcome.requirement_fraction, 0.5)

    def test_unrecoverable_tool_error_fails_the_episode(self) -> None:
        error = RecordedError(
            source="tool",
            recoverable=False,
            message="calendar rejected the call",
            step_index=1,
        )
        episode = _episode(
            status="failed",
            termination_reason="unrecoverable_tool_error",
            tool_steps=(_tool_step(1, error),),
            evaluator_outcome=None,
        )
        self.assertEqual(episode.recorded_errors, (error,))
        self.assertFalse(episode.reached_evaluation)
        with self.assertRaises(RecordError):
            _episode(
                status="completed",
                termination_reason="agent_stopped",
                tool_steps=(_tool_step(1, error),),
            )
        with self.assertRaises(RecordError):
            _episode(
                status="failed",
                termination_reason="unrecoverable_tool_error",
                evaluator_outcome=None,
                tool_steps=(
                    _tool_step(
                        1,
                        RecordedError(
                            source="tool",
                            recoverable=True,
                            message="retry",
                            step_index=1,
                        ),
                    ),
                ),
            )

    def test_failed_termination_requires_a_matching_episode_error(self) -> None:
        timeout = RecordedError(
            source="timeout",
            recoverable=False,
            message="deadline exceeded",
            step_index=None,
        )
        episode = _episode(
            mode="plan",
            status="failed",
            termination_reason="timeout",
            plan_text=None,
            evaluator_outcome=None,
            model_steps=(),
            episode_errors=(timeout,),
        )
        self.assertEqual(episode.recorded_errors, (timeout,))
        with self.assertRaises(RecordError):
            _episode(
                status="failed",
                termination_reason="timeout",
                evaluator_outcome=None,
                model_steps=(),
                episode_errors=(),
            )
        with self.assertRaises(RecordError):
            RecordedError(
                source="runtime",
                recoverable=True,
                message="still running",
                step_index=None,
            )

    def test_plan_episode_has_plan_text_and_no_evaluation(self) -> None:
        episode = _episode(
            mode="plan",
            status="completed",
            termination_reason="plan_emitted",
            plan_text="1. open the calendar",
            evaluator_outcome=None,
            tool_steps=(),
        )
        self.assertEqual(episode.plan_text, "1. open the calendar")
        self.assertIsNone(episode.evaluator_outcome)
        self.assertEqual(episode.tool_steps, ())
        with self.assertRaises(RecordError):
            _episode(
                mode="plan",
                status="completed",
                termination_reason="plan_emitted",
                plan_text="1. open the calendar",
                evaluator_outcome=EvaluatorOutcome(
                    success=True,
                    passed_requirements=1,
                    total_requirements=1,
                    difficulty=None,
                ),
            )
        with self.assertRaises(RecordError):
            _episode(
                mode="plan",
                status="completed",
                termination_reason="plan_emitted",
                plan_text="1. open the calendar",
                evaluator_outcome=None,
                tool_steps=(_tool_step(1),),
            )

    def test_agent_stop_is_not_task_success(self) -> None:
        episode = _episode(evaluator_outcome=None)
        self.assertEqual(episode.status, "completed")
        self.assertEqual(episode.termination_reason, "agent_stopped")
        self.assertFalse(episode.reached_evaluation)
        with self.assertRaises(RecordError):
            _episode(status="failed", termination_reason="agent_stopped")

    def test_episode_binds_its_run_and_rejects_bad_traces(self) -> None:
        episode = _episode()
        self.assertEqual(episode.episode.run_id, episode.run.run_id)
        self.assertTrue(episode.episode.episode_id.startswith(episode.run.run_id))
        other = _run(_HASH_B, "9" * 32)
        with self.assertRaises(RecordError):
            EpisodeResult(
                episode=episode.episode,
                run=other,
                task=episode.task,
                mode=episode.mode,
                execution_seed=episode.execution_seed,
                status=episode.status,
                started_at=episode.started_at,
                ended_at=episode.ended_at,
                model_steps=episode.model_steps,
                tool_steps=episode.tool_steps,
                plan_text=episode.plan_text,
                evaluator_outcome=episode.evaluator_outcome,
                termination_reason=episode.termination_reason,
                episode_errors=episode.episode_errors,
                role=episode.role,
            )
        with self.assertRaises(RecordError):
            _episode(model_steps=(_model_step(0), _model_step(2)))
        with self.assertRaises(RecordError):
            _episode(started_at=datetime(2026, 9, 27, 15, 0))
        with self.assertRaises(RecordError):
            TokenLogprob(token_id=1, logprob=float("nan"), rank=0)
        with self.assertRaises(RecordError):
            TokenLogprob(token_id=1, logprob=float("inf"), rank=0)

    def test_local_round_trip_keeps_protected_fields_local(self) -> None:
        episode = _episode(plan_text="1. open the calendar")
        payload = json.loads(json.dumps(episode.to_dict()))
        restored = EpisodeResult.from_dict(payload)
        self.assertEqual(restored, episode)
        self.assertEqual(payload["visibility"], "local")
        self.assertIn("task_id", _keys(payload))
        self.assertIn("plan_text", _keys(payload))
        with self.assertRaises(RecordError):
            public_record_dict(episode)
        with self.assertRaises(RecordError):
            assert_public_payload(payload)
        payload["instruction"] = "book a ride"
        with self.assertRaises(RecordError):
            EpisodeResult.from_dict(payload)


class PairedResultTests(unittest.TestCase):
    def _pair(self, **candidate_overrides: object) -> PairedResult:
        shared = _task()
        reference = _episode(
            run=_run(_HASH_A, "1" * 32),
            episode_token="4" * 32,
            pair_id=_PAIR,
            role="reference",
            task=shared,
            mode="execute",
            execution_seed=11,
        )
        candidate_values: dict[str, object] = {
            "run": _run(_HASH_B, "3" * 32),
            "episode_token": "5" * 32,
            "pair_id": _PAIR,
            "role": "candidate",
            "task": shared,
            "mode": "execute",
            "execution_seed": 11,
        }
        candidate_values.update(candidate_overrides)
        return PairedResult(
            reference=reference,
            candidate=_episode(**candidate_values),
        )

    def test_pair_accepts_distinct_runs_with_explicit_roles(self) -> None:
        pair = self._pair()
        self.assertEqual(pair.reference.role, "reference")
        self.assertEqual(pair.candidate.role, "candidate")
        self.assertEqual(pair.reference.task, pair.candidate.task)
        self.assertEqual(pair.reference.episode.pair_id, _PAIR)
        self.assertEqual(pair.candidate.episode.pair_id, _PAIR)
        self.assertNotEqual(
            pair.reference.episode.episode_id,
            pair.candidate.episode.episode_id,
        )
        self.assertNotEqual(
            pair.reference.run.configuration_hash,
            pair.candidate.run.configuration_hash,
        )
        self.assertEqual(pair.visibility, "local")

    def test_pair_rejects_mismatched_identity(self) -> None:
        cases = {
            "task": {"task": _task(task_id="other-task")},
            "scenario": {"task": _task(scenario_id="other-scenario")},
            "split": {"task": _task(split="train")},
            "pair": {"pair_id": "f" * 32},
            "mode": {
                "mode": "plan",
                "status": "completed",
                "termination_reason": "plan_emitted",
                "plan_text": "1. open the calendar",
                "evaluator_outcome": None,
                "tool_steps": (),
            },
            "seed": {"execution_seed": 12},
            "role": {"role": "reference"},
        }
        for label, overrides in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(RecordError):
                    self._pair(**overrides)
        reference = _episode(
            run=_run(_HASH_A, "1" * 32),
            episode_token="4" * 32,
            pair_id=_PAIR,
            role="reference",
        )
        with self.assertRaises(RecordError):
            PairedResult(reference=reference, candidate=reference)
        with self.assertRaises(RecordError):
            self._pair(pair_id=None, role="candidate")


class PublicRecordTests(unittest.TestCase):
    def _evidence(self) -> StatisticalEvidence:
        return StatisticalEvidence(
            method="paired_bootstrap",
            split="dev",
            configuration_hash=_HASH_B,
            estimate=-0.1,
            sample_size=8,
            unit="success_difference",
            reference_configuration_hash=_HASH_A,
            confidence_low=-0.2,
            confidence_high=0.0,
            confidence_level=0.95,
            threshold=-0.02,
            seed=7,
        )

    def test_decision_and_aggregate_round_trip_without_task_content(self) -> None:
        evidence = self._evidence()
        decision = LifecycleDecision(
            tier="canary",
            decision="rollback",
            split="dev",
            candidate_configuration_hash=_HASH_B,
            reference_configuration_hash=_HASH_A,
            evidence=(evidence,),
            decided_at=_END,
        )
        payload = public_record_dict(decision)
        restored = LifecycleDecision.from_dict(json.loads(json.dumps(payload)))
        self.assertEqual(restored, decision)
        self.assertTrue(PROTECTED_FIELDS.isdisjoint(_keys(payload)))
        aggregate = AggregateRecord(
            split="dev",
            configuration_hash=_HASH_B,
            task_set_hash=_TASK_HASH,
            scenario_count=4,
            task_count=6,
            episode_count=8,
            metric="task_success_rate",
            value=0.5,
        )
        aggregate_payload = public_record_dict(aggregate)
        self.assertEqual(
            AggregateRecord.from_dict(aggregate_payload),
            aggregate,
        )
        self.assertTrue(PROTECTED_FIELDS.isdisjoint(_keys(aggregate_payload)))
        self.assertIsNone(aggregate.confidence_level)

    def test_public_payload_rejects_nested_task_content(self) -> None:
        with self.assertRaises(RecordError):
            assert_public_payload(
                {"visibility": "public", "nested": {"task_id": "local-task"}}
            )
        with self.assertRaises(RecordError):
            LifecycleDecision(
                tier="offline_gate",
                decision="rollback",
                split="train",
                candidate_configuration_hash=_HASH_B,
                reference_configuration_hash=_HASH_A,
                evidence=(self._evidence(),),
                decided_at=_END,
            )
        with self.assertRaises(RecordError):
            self._evidence().__class__.from_dict(
                {
                    **self._evidence().to_dict(),
                    "confidence_high": None,
                }
            )

    def test_monitor_observation_stays_local_and_typed(self) -> None:
        run = _run()
        observation = MonitorObservation(
            episode=EpisodeIdentity(
                episode_id=f"{run.run_id}.{'6' * 32}",
                run_id=run.run_id,
            ),
            run=run,
            split="dev",
            signal="task_success",
            value=0.0,
            observed_at=_END,
        )
        self.assertEqual(observation.visibility, "local")
        with self.assertRaises(RecordError):
            public_record_dict(observation)
        with self.assertRaises(RecordError):
            MonitorObservation(
                episode=observation.episode,
                run=run,
                split="dev",
                signal="task_success",
                value=1,
                observed_at=_END,
            )
        with self.assertRaises(RecordError):
            MonitorObservation(
                episode=observation.episode,
                run=run,
                split="dev",
                signal="tool_error_count",
                value=1.5,
                observed_at=_END,
            )
        fraction = MonitorObservation(
            episode=observation.episode,
            run=run,
            split="dev",
            signal="requirement_fraction",
            value=0.0,
            observed_at=_END,
        )
        self.assertEqual(fraction.value, 0.0)


class PackageContractTests(unittest.TestCase):
    def test_package_exports_the_shared_contracts_and_one_statistic_each(self) -> None:
        self.assertIs(llm_behavior_ci.paired_bootstrap, paired_bootstrap)
        self.assertIs(llm_behavior_ci.next_token_kl, next_token_kl)
        self.assertIs(llm_behavior_ci.mmd_permutation_test, mmd_permutation_test)
        self.assertIs(llm_behavior_ci.EpisodeResult, EpisodeResult)
        self.assertIs(llm_behavior_ci.RunIdentity, RunIdentity)
        root = Path(__file__).parents[2] / "src"
        for name in ("bootstrap.py", "kl.py", "mmd.py"):
            found = list(root.rglob(name))
            self.assertEqual(found, [root / "llm_behavior_ci" / "stats" / name])
        for relative in (
            "llm_behavior_ci/records.py",
            "llm_behavior_ci/config.py",
            "llm_behavior_ci/lifecycle/offline_gate.py",
        ):
            source = (root / relative).read_text(encoding="utf-8")
            self.assertNotIn("experiments", source)
