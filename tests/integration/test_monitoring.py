from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from llm_behavior_ci.config import MonitorSettings, StoppingRule
from llm_behavior_ci.lifecycle.detectors import build_detector
from llm_behavior_ci.lifecycle.monitoring import (
    FrozenReference,
    LocalAlertSink,
    ProductionMonitor,
    TaskMetadata,
    observation_from_episode,
)
from llm_behavior_ci.experiments.replay import monitoring_detector_factories
from llm_behavior_ci.records import (
    EpisodeIdentity,
    EpisodeResult,
    EvaluatorOutcome,
    LocalTaskRef,
    ModelStep,
    RunIdentity,
    TokenLogprob,
    assert_public_payload,
)

_HASH_A = "a" * 64
_HASH_B = "b" * 64
_TASK_HASH = "c" * 64
_GIT = "d" * 40
_START = datetime(2026, 9, 27, 15, 0, tzinfo=timezone.utc)
_MID = datetime(2026, 9, 27, 15, 1, tzinfo=timezone.utc)
_END = datetime(2026, 9, 27, 15, 5, tzinfo=timezone.utc)


def _run(config: str = _HASH_A, token: str = "1" * 32) -> RunIdentity:
    return RunIdentity(
        run_id=f"{config}.{token}",
        configuration_hash=config,
        task_set_hash=_TASK_HASH,
        protocol_hash=None,
        git_commit=_GIT,
    )


def _model_step() -> ModelStep:
    return ModelStep(
        index=0,
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


def _episode(
    *,
    success: bool,
    episode_token: str,
    config: str = _HASH_A,
    ended_at: datetime = _END,
) -> EpisodeResult:
    run = _run(config=config)
    if success:
        outcome = EvaluatorOutcome(
            success=True,
            passed_requirements=2,
            total_requirements=2,
            difficulty=1,
        )
    else:
        outcome = EvaluatorOutcome(
            success=False,
            passed_requirements=0,
            total_requirements=2,
            difficulty=1,
        )
    return EpisodeResult(
        episode=EpisodeIdentity(
            episode_id=f"{run.run_id}.{episode_token}",
            run_id=run.run_id,
            pair_id=None,
        ),
        run=run,
        task=LocalTaskRef(task_id="local-task", scenario_id="scenario-1", split="dev"),
        mode="execute",
        execution_seed=7,
        status="completed",
        started_at=_START,
        ended_at=ended_at,
        model_steps=(_model_step(),),
        tool_steps=(),
        plan_text=None,
        evaluator_outcome=outcome,
        termination_reason="agent_stopped",
        episode_errors=(),
        role=None,
    )


class ProductionMonitorIntegrationTests(unittest.TestCase):
    def test_cusum_alarms_on_supplied_fault_and_dedups_persistent_fault(self) -> None:
        settings = MonitorSettings(
            reference_configuration_hash=_HASH_A,
            outcome_delay_seconds=0.0,
            signals=("task_success",),
            stopping_rules=(
                StoppingRule(
                    name="cusum",
                    alpha=0.1,
                    horizon_episodes=50,
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
            dedup_seconds=60.0,
        )
        alerts = []
        for index, token in enumerate(("2" * 32, "3" * 32, "4" * 32)):
            observation = observation_from_episode(
                _episode(success=False, episode_token=token),
                task_metadata=TaskMetadata(
                    signal="task_success",
                    completion_index=index,
                    task_mix="difficulty:1",
                ),
            )
            alerts.extend(monitor.update(observation))
            clock_time["now"] = clock_time["now"] + timedelta(seconds=1)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].signal, "task_success")
        self.assertEqual(alerts[0].slice_name, "task_success")
        self.assertEqual(alerts[0].method, "cusum")
        self.assertEqual(alerts[0].reference_configuration_hash, _HASH_A)
        assert_public_payload(alerts[0].to_public_dict())

    def test_reset_for_promotion_changes_baseline_without_absorbing_stream(
        self,
    ) -> None:
        settings = MonitorSettings(
            reference_configuration_hash=_HASH_A,
            outcome_delay_seconds=0.0,
            signals=("task_success",),
            stopping_rules=(
                StoppingRule(
                    name="cusum",
                    alpha=0.1,
                    horizon_episodes=50,
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
        prior = monitor.update(
            observation_from_episode(
                _episode(success=False, episode_token="2" * 32),
                task_metadata=TaskMetadata(signal="task_success", completion_index=0),
            )
        )
        self.assertEqual(len(prior), 1)
        self.assertEqual(prior[0].sample_size, 1)
        self.assertEqual(prior[0].reference_configuration_hash, _HASH_A)
        monitor.update(
            observation_from_episode(
                _episode(success=False, episode_token="3" * 32),
                task_metadata=TaskMetadata(signal="task_success", completion_index=1),
            )
        )
        monitor.reset_for_promotion(
            FrozenReference(
                configuration_hash=_HASH_B,
                baselines=(("task_success", 0.8),),
            )
        )
        healthy = monitor.update(
            observation_from_episode(
                _episode(success=True, episode_token="4" * 32, config=_HASH_B),
                task_metadata=TaskMetadata(signal="task_success", completion_index=0),
            )
        )
        self.assertEqual(healthy, ())
        promoted = monitor.update(
            observation_from_episode(
                _episode(success=False, episode_token="5" * 32, config=_HASH_B),
                task_metadata=TaskMetadata(signal="task_success", completion_index=1),
            )
        )
        self.assertEqual(len(promoted), 1)
        self.assertEqual(promoted[0].sample_size, 2)
        self.assertEqual(promoted[0].estimate, 0.8)
        self.assertEqual(promoted[0].reference_configuration_hash, _HASH_B)
        self.assertEqual(promoted[0].configuration_hash, _HASH_B)
        assert_public_payload(promoted[0].to_public_dict())

    def test_shared_factory_and_local_sink_on_alarm_stream(self) -> None:
        rule = StoppingRule(
            name="cusum",
            alpha=0.1,
            horizon_episodes=50,
            threshold=0.5,
        )
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
        shared = build_detector(rule, signal="task_success", baseline=0.9)
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
        self.assertEqual(shared.snapshot(), factories["cusum"]().snapshot())
        clock_time = {"now": _END + timedelta(seconds=1)}

        def clock() -> datetime:
            return clock_time["now"]

        monitor = ProductionMonitor(
            settings,
            reference,
            clock=clock,
            dedup_seconds=60.0,
            period_id="production-window",
        )
        sink = LocalAlertSink(dedup_seconds=60.0)
        for index, token in enumerate(("2" * 32, "3" * 32)):
            alerts = monitor.update(
                observation_from_episode(
                    _episode(success=False, episode_token=token),
                    task_metadata=TaskMetadata(
                        signal="task_success",
                        completion_index=index,
                    ),
                ),
                completion_index=index,
                period_id="production-window",
            )
            sink.deliver(alerts)
            clock_time["now"] = clock_time["now"] + timedelta(seconds=1)
        self.assertEqual(len(sink.delivered), 1)
        self.assertEqual(monitor.reference.baselines, reference.baselines)


if __name__ == "__main__":
    unittest.main()
