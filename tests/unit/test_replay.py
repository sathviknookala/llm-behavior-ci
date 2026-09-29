from __future__ import annotations

import os
import subprocess
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from llm_behavior_ci.config import (
    EpisodeIdentity,
    MonitorSettings,
    RunIdentity,
    StoppingRule,
)
from llm_behavior_ci.experiments.replay import (
    ReplaySchedule,
    monitoring_detector_factories,
    replay_detectors,
)
from llm_behavior_ci.lifecycle.detectors import (
    DetectorConstructionError,
    build_detector,
)
from llm_behavior_ci.lifecycle.monitoring import FrozenReference
from llm_behavior_ci.records import MonitorObservation
from llm_behavior_ci.stats.cusum import CUSUM
from llm_behavior_ci.stats.evidence import Evidence

_HASH_A = "a" * 64
_TASK_HASH = "c" * 64
_GIT = "d" * 40
_START = datetime(2026, 9, 27, 15, 0, tzinfo=timezone.utc)
_ROOT = Path(__file__).resolve().parents[2]
_CLI = _ROOT / "scripts" / "replay" / "replay_detectors.py"


def _run(token: str = "1" * 32) -> RunIdentity:
    return RunIdentity(
        run_id=f"{_HASH_A}.{token}",
        configuration_hash=_HASH_A,
        task_set_hash=_TASK_HASH,
        protocol_hash=None,
        git_commit=_GIT,
    )


def _observation(
    *,
    token: str,
    value: float,
    observed_at: datetime,
    signal: str = "task_success",
    run: RunIdentity | None = None,
) -> MonitorObservation:
    identity = run if run is not None else _run()
    return MonitorObservation(
        episode=EpisodeIdentity(
            episode_id=f"{identity.run_id}.{token}",
            run_id=identity.run_id,
        ),
        run=identity,
        split="dev",
        signal=signal,
        value=value,
        observed_at=observed_at,
    )


def _schedule(
    count: int,
    *,
    outcome_delay_seconds: float = 0.0,
    horizon_episodes: int = 3,
    onset_index: int = 1,
    start: datetime = _START,
) -> ReplaySchedule:
    return ReplaySchedule(
        outcome_delay_seconds=outcome_delay_seconds,
        horizon_episodes=horizon_episodes,
        onset_index=onset_index,
        scenario_keys=tuple(f"scenario-{index}" for index in range(count)),
        arrival_times=tuple(start + timedelta(seconds=index) for index in range(count)),
    )


class _IndexAlarm:
    def __init__(self, alarm_at: int) -> None:
        self._alarm_at = alarm_at
        self._count = 0

    def update(self, observation: float) -> Evidence:
        del observation
        self._count += 1
        alarm = self._count - 1 == self._alarm_at
        return Evidence(
            method="index_alarm",
            estimate=float(self._count),
            sample_size=self._count,
            alarm=alarm,
            boundary=None,
            p_value=None,
            details=(),
        )

    def reset(self) -> None:
        self._count = 0

    def snapshot(self) -> dict[str, object]:
        return {"sample_size": self._count}


class _SeededThreshold:
    def __init__(self, *, seed: int, threshold: float) -> None:
        self._seed = seed
        self._threshold = threshold
        self._total = 0.0
        self._count = 0

    def update(self, observation: float) -> Evidence:
        self._total += float(observation) + (self._seed * 0.0)
        self._count += 1
        estimate = self._total
        return Evidence(
            method="seeded",
            estimate=estimate,
            sample_size=self._count,
            alarm=estimate >= self._threshold,
            boundary=self._threshold,
            p_value=None,
            details=(),
        )

    def reset(self) -> None:
        self._total = 0.0
        self._count = 0

    def snapshot(self) -> dict[str, object]:
        return {"sample_size": self._count, "total": self._total}


class ReplayUnitTests(unittest.TestCase):
    def test_identical_input_shares_stream_hash_and_preserves_caller_list(self) -> None:
        observations = [
            _observation(token="2" * 32, value=1.0, observed_at=_START),
            _observation(
                token="3" * 32,
                value=0.0,
                observed_at=_START + timedelta(seconds=1),
            ),
        ]
        list_id = id(observations)
        first_id = id(observations[0])
        original_values = [item.value for item in observations]
        schedule = _schedule(len(observations), onset_index=1, horizon_episodes=2)

        def left_factory() -> CUSUM:
            return CUSUM(target=0.9, slack=0.0, threshold=0.5, direction="decrease")

        def right_factory() -> CUSUM:
            return CUSUM(target=0.9, slack=0.0, threshold=0.5, direction="decrease")

        results = replay_detectors(
            observations,
            {"left": left_factory, "right": right_factory},
            schedule=schedule,
        )
        self.assertEqual(results["left"].stream_hash, results["right"].stream_hash)
        self.assertEqual(id(observations), list_id)
        self.assertEqual(id(observations[0]), first_id)
        self.assertEqual([item.value for item in observations], original_values)

    def test_independent_detector_state_across_factories(self) -> None:
        observations = [
            _observation(
                token=token,
                value=0.0,
                observed_at=_START + timedelta(seconds=index),
            )
            for index, token in enumerate(("2" * 32, "3" * 32, "4" * 32))
        ]
        schedule = _schedule(len(observations), onset_index=0, horizon_episodes=3)
        stashed: list[CUSUM] = []

        def first_factory() -> CUSUM:
            detector = CUSUM(
                target=0.9, slack=0.0, threshold=0.5, direction="decrease"
            )
            stashed.append(detector)
            return detector

        def second_factory() -> CUSUM:
            detector = CUSUM(
                target=0.9, slack=0.0, threshold=0.5, direction="decrease"
            )
            stashed.append(detector)
            return detector

        replay_detectors(
            observations,
            {"first": first_factory, "second": second_factory},
            schedule=schedule,
        )
        self.assertEqual(len(stashed), 2)
        first_snapshot = dict(stashed[0].snapshot())
        second_snapshot = dict(stashed[1].snapshot())
        self.assertGreater(first_snapshot["sample_size"], 0)
        self.assertGreater(second_snapshot["sample_size"], 0)
        stashed[0].update(0.0)
        self.assertEqual(stashed[1].snapshot(), second_snapshot)
        self.assertNotEqual(stashed[0].snapshot(), first_snapshot)

    def test_deterministic_seeded_replay(self) -> None:
        observations = [
            _observation(
                token=f"{index + 2}" * 32,
                value=0.0,
                observed_at=_START + timedelta(seconds=index),
            )
            for index in range(4)
        ]
        schedule = _schedule(len(observations), onset_index=1, horizon_episodes=3)

        def factory() -> _SeededThreshold:
            return _SeededThreshold(seed=7, threshold=0.5)

        first = replay_detectors(
            observations, {"seeded": factory}, schedule=schedule
        )
        second = replay_detectors(
            observations, {"seeded": factory}, schedule=schedule
        )
        self.assertEqual(
            first["seeded"].detection_delay_episodes,
            second["seeded"].detection_delay_episodes,
        )
        self.assertEqual(
            first["seeded"].healthy_false_alarm_count,
            second["seeded"].healthy_false_alarm_count,
        )
        self.assertEqual(first["seeded"].stream_hash, second["seeded"].stream_hash)
        self.assertEqual(
            first["seeded"].subsequent_alert_count,
            second["seeded"].subsequent_alert_count,
        )

    def test_delay_accounting_withholds_and_releases_in_order(self) -> None:
        first_observed = _START + timedelta(seconds=5)
        second_arrival = _START + timedelta(seconds=1)
        third_arrival = _START + timedelta(seconds=20)
        observations = [
            _observation(token="2" * 32, value=0.0, observed_at=first_observed),
            _observation(
                token="3" * 32,
                value=1.0,
                observed_at=_START + timedelta(seconds=2),
                signal="trajectory_length",
            ),
            _observation(
                token="4" * 32,
                value=2.0,
                observed_at=_START + timedelta(seconds=21),
                signal="trajectory_length",
            ),
        ]
        schedule = ReplaySchedule(
            outcome_delay_seconds=2.0,
            horizon_episodes=3,
            onset_index=0,
            scenario_keys=("a", "b", "c"),
            arrival_times=(
                _START,
                second_arrival,
                third_arrival,
            ),
        )
        applied_values: list[float] = []

        class _Recorder:
            def update(self, observation: float) -> Evidence:
                applied_values.append(float(observation))
                return Evidence(
                    method="recorder",
                    estimate=float(observation),
                    sample_size=len(applied_values),
                    alarm=False,
                    boundary=None,
                    p_value=None,
                    details=(),
                )

            def reset(self) -> None:
                return None

            def snapshot(self) -> dict[str, object]:
                return {"sample_size": len(applied_values)}

        short = ReplaySchedule(
            outcome_delay_seconds=2.0,
            horizon_episodes=3,
            onset_index=0,
            scenario_keys=("a", "b"),
            arrival_times=(_START, second_arrival),
        )
        withheld = replay_detectors(
            observations[:2],
            {"recorder": _Recorder},
            schedule=short,
        )
        self.assertGreaterEqual(withheld["recorder"].observations_withheld, 1)
        self.assertNotIn(0.0, applied_values)

        applied_values.clear()
        released = replay_detectors(
            observations,
            {"recorder": _Recorder},
            schedule=schedule,
        )
        self.assertEqual(released["recorder"].observations_withheld, 0)
        self.assertEqual(applied_values, [1.0, 0.0, 2.0])

    def test_post_horizon_exclusion(self) -> None:
        observations = [
            _observation(
                token=f"{index + 2}" * 32,
                value=1.0,
                observed_at=_START + timedelta(seconds=index),
            )
            for index in range(4)
        ]
        schedule = _schedule(
            len(observations),
            onset_index=0,
            horizon_episodes=2,
        )

        def factory() -> _IndexAlarm:
            return _IndexAlarm(alarm_at=3)

        result = replay_detectors(
            observations, {"late": factory}, schedule=schedule
        )["late"]
        self.assertTrue(result.missed_horizon)
        self.assertIsNone(result.first_detection_index)
        self.assertGreaterEqual(result.post_horizon_detection_count, 1)

    def test_healthy_false_alarm_is_not_first_detection(self) -> None:
        observations = [
            _observation(
                token=f"{index + 2}" * 32,
                value=1.0,
                observed_at=_START + timedelta(seconds=index),
            )
            for index in range(3)
        ]
        schedule = _schedule(
            len(observations),
            onset_index=2,
            horizon_episodes=1,
        )

        def factory() -> _IndexAlarm:
            return _IndexAlarm(alarm_at=0)

        result = replay_detectors(
            observations, {"early": factory}, schedule=schedule
        )["early"]
        self.assertEqual(result.healthy_false_alarm_count, 1)
        self.assertIsNone(result.first_detection_index)
        self.assertTrue(result.missed_horizon)

    def test_agent_execution_seconds_is_none_and_compute_is_finite(self) -> None:
        observations = [
            _observation(token="2" * 32, value=0.0, observed_at=_START),
        ]
        schedule = _schedule(1, onset_index=0, horizon_episodes=1)

        def factory() -> CUSUM:
            return CUSUM(target=0.9, slack=0.0, threshold=0.5, direction="decrease")

        result = replay_detectors(
            observations, {"cusum": factory}, schedule=schedule
        )["cusum"]
        self.assertIsNone(result.agent_execution_seconds)
        self.assertIsInstance(result.replay_compute_seconds, float)
        self.assertTrue(result.replay_compute_seconds >= 0.0)
        self.assertTrue(result.replay_compute_seconds == result.replay_compute_seconds)

    def test_cli_without_arguments_exits_two(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        completed = subprocess.run(
            [sys.executable, str(_CLI)],
            cwd=_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)

    def test_monitoring_factories_use_shared_detector_parameters(self) -> None:
        rule = StoppingRule(
            name="cusum",
            alpha=0.1,
            horizon_episodes=20,
            threshold=0.75,
        )
        settings = MonitorSettings(
            reference_configuration_hash=_HASH_A,
            outcome_delay_seconds=0.0,
            signals=("task_success",),
            stopping_rules=(rule,),
        )
        reference = FrozenReference(
            configuration_hash=_HASH_A,
            baselines=(("task_success", 0.85),),
        )
        factories = monitoring_detector_factories(
            settings,
            reference,
            signal="task_success",
            alpha=0.05,
            window_episodes=2,
            reference_sample=(1.0, 1.0, 0.0),
            harm_margin=0.05,
            corrections=("none", "bonferroni"),
        )
        shared = build_detector(rule, signal="task_success", baseline=0.85)
        from_replay = factories["cusum"]()
        self.assertEqual(shared.snapshot(), from_replay.snapshot())
        self.assertIn("ks_hourly_none", factories)
        self.assertIn("chi_square_hourly_bonferroni", factories)
        with self.assertRaises(DetectorConstructionError):
            build_detector(rule, signal="task_mix", baseline=0.5)
        with self.assertRaises(DetectorConstructionError):
            build_detector(
                StoppingRule(
                    name="unknown_rule",
                    alpha=0.1,
                    horizon_episodes=1,
                    threshold=0.1,
                ),
                signal="task_success",
                baseline=0.9,
            )


if __name__ == "__main__":
    unittest.main()
