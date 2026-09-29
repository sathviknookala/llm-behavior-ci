from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from llm_behavior_ci.config import (
    EpisodeIdentity,
    MonitorSettings,
    RunIdentity,
    StoppingRule,
)
from llm_behavior_ci.experiments.replay import monitoring_detector_factories
from llm_behavior_ci.lifecycle.detectors import (
    DetectorConstructionError,
    build_detector,
)
from llm_behavior_ci.lifecycle.monitoring import (
    Alert,
    FrozenReference,
    LocalAlertSink,
    MissingEvaluatorOutcome,
    MonitorRejected,
    NormalizedEpisode,
    ProductionMonitor,
    RepeatedEpisode,
    TaskMetadata,
    UndefinedRequirementFraction,
    normalize_episode,
    observation_from_episode,
    task_mix_observation_from_episode,
    tool_selection_observation_from_episode,
)
from llm_behavior_ci.records import (
    EpisodeResult,
    EvaluatorOutcome,
    LocalTaskRef,
    ModelStep,
    RecordError,
    RecordedError,
    TokenLogprob,
    ToolStep,
    assert_public_payload,
)
from llm_behavior_ci.storage import EpisodeStore

_HASH_A = "a" * 64
_HASH_B = "b" * 64
_TASK_HASH = "c" * 64
_GIT = "d" * 40
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


def _task(split: str = "dev") -> LocalTaskRef:
    return LocalTaskRef(task_id="local-task", scenario_id="scenario-1", split=split)


def _model_step(index: int = 0) -> ModelStep:
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
        started_at=_MID,
    )


def _tool_step(
    index: int,
    *,
    app_name: str | None = "calendar",
    api_name: str | None = "lookup",
    error: RecordedError | None = None,
) -> ToolStep:
    return ToolStep(
        index=index,
        action="app.lookup()",
        app_name=app_name,
        api_name=api_name,
        output_text="",
        error=error,
        latency_seconds=0.2,
        started_at=_LATER,
    )


def _episode(**overrides: object) -> EpisodeResult:
    run = overrides.pop("run") if "run" in overrides else _run()
    if not isinstance(run, RunIdentity):
        raise AssertionError("run")
    episode_token = overrides.pop("episode_token", "2" * 32)
    if not isinstance(episode_token, str):
        raise AssertionError("episode_token")
    if "episode" not in overrides:
        overrides["episode"] = EpisodeIdentity(
            episode_id=f"{run.run_id}.{episode_token}",
            run_id=run.run_id,
            pair_id=None,
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


def _metadata(signal: str = "task_success", **overrides: object) -> TaskMetadata:
    values: dict[str, object] = {
        "signal": signal,
        "completion_index": 0,
        "difficulty": 2,
        "task_mix": "difficulty:2",
    }
    values.update(overrides)
    return TaskMetadata(**values)


class NormalizeEpisodeTests(unittest.TestCase):
    def test_normalizes_all_five_signals_plus_tool_selection_and_task_mix(self) -> None:
        error = RecordedError(
            source="tool",
            recoverable=True,
            message="lookup failed",
            step_index=1,
        )
        episode = _episode(
            tool_steps=(
                _tool_step(1, api_name="lookup", error=error),
                _tool_step(2, app_name="mail", api_name=None),
                _tool_step(3, app_name=None, api_name=None),
                _tool_step(4, api_name="lookup"),
            ),
            evaluator_outcome=EvaluatorOutcome(
                success=True,
                passed_requirements=2,
                total_requirements=2,
                difficulty=1,
            ),
        )
        normalized = normalize_episode(
            episode,
            task_metadata=_metadata(task_mix="difficulty:1", completion_index=3),
        )
        self.assertIsInstance(normalized, NormalizedEpisode)
        self.assertEqual(normalized.task_success, 1.0)
        self.assertEqual(normalized.requirement_fraction, 1.0)
        self.assertEqual(normalized.tool_error_count, 1.0)
        self.assertEqual(normalized.invalid_tool_call_count, 1.0)
        self.assertEqual(normalized.trajectory_length, 5.0)
        self.assertEqual(
            normalized.tool_selection,
            (("lookup", 2), ("mail", 1), ("unparsed", 1)),
        )
        self.assertEqual(normalized.task_mix, "difficulty:1")
        self.assertEqual(normalized.completion_index, 3)
        self.assertFalse(normalized.missing_outcome)

    def test_missing_outcome_leaves_success_and_fraction_none(self) -> None:
        episode = _episode(
            mode="plan",
            tool_steps=(),
            plan_text="call calendar.lookup",
            evaluator_outcome=None,
            termination_reason="plan_emitted",
            model_steps=(_model_step(),),
        )
        normalized = normalize_episode(episode, task_metadata=_metadata())
        self.assertTrue(normalized.missing_outcome)
        self.assertIsNone(normalized.task_success)
        self.assertIsNone(normalized.requirement_fraction)
        self.assertEqual(normalized.tool_error_count, 0.0)
        self.assertEqual(normalized.invalid_tool_call_count, 0.0)
        self.assertEqual(normalized.trajectory_length, 1.0)

    def test_observation_missing_outcome_raises_for_success_and_fraction(self) -> None:
        episode = _episode(
            mode="plan",
            plan_text="call calendar.lookup",
            evaluator_outcome=None,
            termination_reason="plan_emitted",
        )
        with self.assertRaises(MissingEvaluatorOutcome):
            observation_from_episode(
                episode,
                task_metadata=_metadata("task_success"),
            )
        with self.assertRaises(MissingEvaluatorOutcome):
            observation_from_episode(
                episode,
                task_metadata=_metadata("requirement_fraction"),
            )
        length = observation_from_episode(
            episode,
            task_metadata=_metadata("trajectory_length"),
        )
        self.assertEqual(length.value, 1.0)
        self.assertEqual(length.signal, "trajectory_length")

    def test_zero_total_requirements_raises_undefined_fraction(self) -> None:
        episode = _episode(
            evaluator_outcome=EvaluatorOutcome(
                success=True,
                passed_requirements=0,
                total_requirements=0,
                difficulty=None,
            )
        )
        with self.assertRaises(UndefinedRequirementFraction):
            observation_from_episode(
                episode,
                task_metadata=_metadata("requirement_fraction"),
            )
        success = observation_from_episode(
            episode,
            task_metadata=_metadata("task_success"),
        )
        self.assertEqual(success.value, 1.0)


class ProductionMonitorUnitTests(unittest.TestCase):
    def test_repeated_episode_does_not_double_update(self) -> None:
        settings = MonitorSettings(
            reference_configuration_hash=_HASH_A,
            outcome_delay_seconds=0.0,
            signals=("task_success",),
            stopping_rules=(
                StoppingRule(
                    name="cusum",
                    alpha=0.1,
                    horizon_episodes=20,
                    threshold=0.5,
                ),
            ),
        )
        reference = FrozenReference(
            configuration_hash=_HASH_A,
            baselines=(("task_success", 0.9),),
        )
        clock_time = {"now": _END + timedelta(seconds=1)}

        def clock() -> datetime:
            return clock_time["now"]

        monitor = ProductionMonitor(
            settings,
            reference,
            clock=clock,
            dedup_seconds=0.0,
        )
        observation = observation_from_episode(
            _episode(
                evaluator_outcome=EvaluatorOutcome(
                    success=False,
                    passed_requirements=0,
                    total_requirements=2,
                    difficulty=2,
                )
            ),
            task_metadata=_metadata("task_success", completion_index=0),
        )
        first = monitor.update(observation)
        self.assertEqual(len(first), 1)
        with self.assertRaises(RepeatedEpisode):
            monitor.update(observation)
        second_episode = _episode(
            episode_token="3" * 32,
            evaluator_outcome=EvaluatorOutcome(
                success=False,
                passed_requirements=0,
                total_requirements=2,
                difficulty=2,
            ),
        )
        second = monitor.update(
            observation_from_episode(
                second_episode,
                task_metadata=_metadata("task_success", completion_index=1),
            )
        )
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0].sample_size, 2)

    def test_alert_public_dict_passes_assert_public_payload(self) -> None:
        alert = Alert(
            configuration_hash=_HASH_A,
            reference_configuration_hash=_HASH_B,
            signal="task_success",
            slice_name="task_success",
            method="cusum",
            estimate=1.0,
            boundary=0.5,
            sample_size=2,
            raised_at=_END,
        )
        payload = alert.to_public_dict()
        assert_public_payload(payload)
        self.assertEqual(payload["slice_name"], "task_success")
        self.assertNotIn("task_id", payload)
        self.assertNotIn("episode_id", payload)

    def test_shared_factory_matches_replay_and_rejects_unsupported(self) -> None:
        rule = StoppingRule(
            name="cusum",
            alpha=0.1,
            horizon_episodes=20,
            threshold=0.5,
        )
        left = build_detector(rule, signal="task_success", baseline=0.9)
        settings = MonitorSettings(
            reference_configuration_hash=_HASH_A,
            outcome_delay_seconds=0.0,
            signals=("task_success",),
            stopping_rules=(rule,),
        )
        reference = FrozenReference(
            configuration_hash=_HASH_A,
            baselines=(("task_success", 0.9),),
        )
        factories = monitoring_detector_factories(
            settings,
            reference,
            signal="task_success",
            alpha=0.05,
            window_episodes=2,
            reference_sample=(1.0, 0.0),
            harm_margin=0.1,
            corrections=("none",),
        )
        right = factories["cusum"]()
        self.assertEqual(left.snapshot(), right.snapshot())
        with self.assertRaises(DetectorConstructionError):
            build_detector(rule, signal="tool_selection", baseline=0.9)

    def test_distributional_observation_rejected_as_scalar(self) -> None:
        episode = _episode(
            tool_steps=(_tool_step(1),),
        )
        with self.assertRaises(MonitorRejected):
            observation_from_episode(
                episode,
                task_metadata=_metadata("tool_selection"),
            )
        with self.assertRaises(RecordError):
            from llm_behavior_ci.records import MonitorObservation

            MonitorObservation(
                episode=episode.episode,
                run=episode.run,
                split=episode.task.split,
                signal="tool_selection",
                value=1.0,
                observed_at=_END,
            )
        selection = tool_selection_observation_from_episode(
            episode,
            task_metadata=_metadata(task_mix="difficulty:2"),
        )
        mix = task_mix_observation_from_episode(
            episode,
            task_metadata=_metadata(task_mix="difficulty:2"),
        )
        settings = MonitorSettings(
            reference_configuration_hash=_HASH_A,
            outcome_delay_seconds=0.0,
            signals=("task_success",),
            stopping_rules=(
                StoppingRule(
                    name="cusum",
                    alpha=0.1,
                    horizon_episodes=20,
                    threshold=0.5,
                ),
            ),
        )
        monitor = ProductionMonitor(
            settings,
            FrozenReference(
                configuration_hash=_HASH_A,
                baselines=(("task_success", 0.9),),
            ),
            clock=lambda: _END,
            dedup_seconds=0.0,
        )
        with self.assertRaises(MonitorRejected):
            monitor.update(selection)  # type: ignore[arg-type]
        with self.assertRaises(MonitorRejected):
            monitor.update(mix)  # type: ignore[arg-type]

    def test_baseline_unchanged_after_updates(self) -> None:
        settings = MonitorSettings(
            reference_configuration_hash=_HASH_A,
            outcome_delay_seconds=0.0,
            signals=("task_success",),
            stopping_rules=(
                StoppingRule(
                    name="cusum",
                    alpha=0.1,
                    horizon_episodes=20,
                    threshold=0.5,
                ),
            ),
        )
        reference = FrozenReference(
            configuration_hash=_HASH_A,
            baselines=(("task_success", 0.9),),
        )
        monitor = ProductionMonitor(
            settings,
            reference,
            clock=lambda: _END + timedelta(seconds=1),
            dedup_seconds=0.0,
        )
        original = reference.baselines
        monitor.update(
            observation_from_episode(
                _episode(
                    evaluator_outcome=EvaluatorOutcome(
                        success=False,
                        passed_requirements=0,
                        total_requirements=2,
                        difficulty=2,
                    )
                ),
                task_metadata=_metadata("task_success"),
            )
        )
        monitor.update(
            observation_from_episode(
                _episode(
                    episode_token="3" * 32,
                    evaluator_outcome=EvaluatorOutcome(
                        success=True,
                        passed_requirements=2,
                        total_requirements=2,
                        difficulty=2,
                    ),
                ),
                task_metadata=_metadata("task_success"),
            )
        )
        self.assertEqual(monitor.reference.baselines, original)
        self.assertIs(monitor.reference, reference)

    def test_period_mismatch_rejected(self) -> None:
        settings = MonitorSettings(
            reference_configuration_hash=_HASH_A,
            outcome_delay_seconds=0.0,
            signals=("task_success",),
            stopping_rules=(
                StoppingRule(
                    name="cusum",
                    alpha=0.1,
                    horizon_episodes=20,
                    threshold=0.5,
                ),
            ),
        )
        monitor = ProductionMonitor(
            settings,
            FrozenReference(
                configuration_hash=_HASH_A,
                baselines=(("task_success", 0.9),),
            ),
            clock=lambda: _END,
            dedup_seconds=0.0,
            period_id="production-1",
        )
        observation = observation_from_episode(
            _episode(
                evaluator_outcome=EvaluatorOutcome(
                    success=True,
                    passed_requirements=2,
                    total_requirements=2,
                    difficulty=2,
                )
            ),
            task_metadata=_metadata("task_success"),
        )
        with self.assertRaises(MonitorRejected):
            monitor.update(observation, period_id="canary-1")
        accepted = monitor.update(observation, period_id="production-1")
        self.assertEqual(accepted, ())

    def test_local_alert_sink_delivery_and_store_dedup(self) -> None:
        alert = Alert(
            configuration_hash=_HASH_A,
            reference_configuration_hash=_HASH_B,
            signal="task_success",
            slice_name="task_success",
            method="cusum",
            estimate=1.0,
            boundary=0.5,
            sample_size=2,
            raised_at=_END,
        )
        duplicate = Alert(
            configuration_hash=_HASH_A,
            reference_configuration_hash=_HASH_B,
            signal="task_success",
            slice_name="task_success",
            method="cusum",
            estimate=3.0,
            boundary=0.5,
            sample_size=4,
            raised_at=_END + timedelta(seconds=1),
        )
        with TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "alerts.sqlite")
            sink = LocalAlertSink(store, dedup_seconds=60.0)
            first = sink.deliver((alert,))
            second = sink.deliver((duplicate,))
            self.assertEqual(first, (alert,))
            self.assertEqual(second, (alert,))
            self.assertEqual(sink.delivered, (alert, alert))
            self.assertEqual(store.load_alerts(), (alert.to_record(),))
            self.assertEqual(len(store.load_deployment_decisions()), 1)
            store.close()


if __name__ == "__main__":
    unittest.main()
