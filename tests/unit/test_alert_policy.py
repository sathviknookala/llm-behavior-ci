from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from llm_behavior_ci.config import EpisodeIdentity, MonitorSettings, RunIdentity, StoppingRule
from llm_behavior_ci.lifecycle.monitoring import (
    FrozenReference,
    LocalAlertSink,
    MonitorRejected,
    ProductionMonitor,
)
from llm_behavior_ci.records import MonitorObservation
from llm_behavior_ci.storage import EpisodeStore

_HASH = "a" * 64
_TASK_HASH = "b" * 64
_GIT = "c" * 40
_START = datetime(2026, 10, 8, tzinfo=timezone.utc)
_BASELINE = 8.0 / 9.0


class _Clock:
    def __init__(self) -> None:
        self.now = _START

    def __call__(self) -> datetime:
        self.now += timedelta(minutes=1)
        return self.now


def _settings() -> MonitorSettings:
    return MonitorSettings(
        reference_configuration_hash=_HASH,
        outcome_delay_seconds=0.0,
        signals=("tool_error_count",),
        stopping_rules=(
            StoppingRule(name="cusum", alpha=0.05, horizon_episodes=50, threshold=5.0),
        ),
    )


def _monitor(clock: _Clock, period_id: str = "period-1") -> ProductionMonitor:
    return ProductionMonitor(
        _settings(),
        FrozenReference(configuration_hash=_HASH, baselines=(("tool_error_count", _BASELINE),)),
        clock=clock,
        period_id=period_id,
        use_slice_attribution=True,
    )


def _observation(index: int, value: float) -> MonitorObservation:
    run_id = f"{_HASH}.{index:032x}"
    return MonitorObservation(
        episode=EpisodeIdentity(episode_id=f"{run_id}.{index:032x}", run_id=run_id),
        run=RunIdentity(
            run_id=run_id,
            configuration_hash=_HASH,
            task_set_hash=_TASK_HASH,
            protocol_hash=None,
            git_commit=_GIT,
        ),
        split="dev",
        signal="tool_error_count",
        value=value,
        observed_at=_START,
    )


_PLANTED = ((1, 11.0, "difficulty:1"), (2, 1.0, "difficulty:2"), (3, 11.0, "difficulty:1"))


def _feed(monitor: ProductionMonitor, sink: LocalAlertSink, start: int, rounds: int):
    raised = []
    delivered = []
    index = start
    for _ in range(rounds):
        for _offset, value, slice_name in _PLANTED:
            alerts = monitor.update(_observation(index, value), slice_name=slice_name)
            raised.extend(alerts)
            delivered.extend(sink.deliver(alerts))
            index += 1
    return raised, delivered


class AlertPolicyTests(unittest.TestCase):
    def test_aggregate_and_slice_alarm_on_one_observation_is_one_incident(self) -> None:
        clock = _Clock()
        monitor = _monitor(clock)
        raised, delivered = _feed(monitor, LocalAlertSink(), 1, 3)
        self.assertEqual(len(raised), 1)
        self.assertEqual(delivered, raised)
        alert = raised[0]
        self.assertEqual(alert.slice_name, "tool_error_count")
        self.assertEqual(alert.attributed_slices, ("difficulty:1",))
        self.assertEqual(alert.period_id, "period-1")
        self.assertEqual(alert.to_public_dict()["attributed_slices"], ["difficulty:1"])

    def test_repeat_delivery_returns_nothing_new(self) -> None:
        clock = _Clock()
        monitor = _monitor(clock)
        raised, _ = _feed(monitor, LocalAlertSink(), 1, 1)
        with tempfile.TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "alerts.sqlite")
            sink = LocalAlertSink(store)
            self.assertEqual(sink.deliver(raised), tuple(raised))
            self.assertEqual(sink.deliver(raised), ())
            self.assertEqual(len(store.load_alerts()), 1)
            self.assertEqual(
                [item.decision for item in store.load_deployment_decisions()], ["alert"]
            )
            store.close()

    def test_incident_persists_across_a_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alerts.sqlite"
            store = EpisodeStore(path)
            _, delivered = _feed(_monitor(_Clock()), LocalAlertSink(store), 1, 3)
            self.assertEqual(len(delivered), 1)
            store.close()

            late_clock = _Clock()
            late_clock.now = _START + timedelta(days=2)
            reopened = EpisodeStore(path)
            unrestored = _monitor(late_clock)
            raised, delivered = _feed(unrestored, LocalAlertSink(reopened), 100, 3)
            self.assertEqual(len(raised), 1)
            self.assertEqual(delivered, [])

            restored = _monitor(late_clock)
            self.assertEqual(restored.restore_open_incidents(reopened.load_alerts()), 1)
            raised, delivered = _feed(restored, LocalAlertSink(reopened), 200, 3)
            self.assertEqual(raised, [])
            self.assertEqual(delivered, [])
            self.assertEqual(len(reopened.load_alerts()), 1)
            self.assertEqual(len(reopened.load_deployment_decisions()), 1)
            reopened.close()

    def test_new_period_opens_a_new_incident(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "alerts.sqlite")
            sink = LocalAlertSink(store)
            clock = _Clock()
            monitor = _monitor(clock)
            _, first = _feed(monitor, sink, 1, 2)
            reference = FrozenReference(
                configuration_hash=_HASH, baselines=(("tool_error_count", _BASELINE),)
            )
            with self.assertRaises(MonitorRejected):
                monitor.reset_for_promotion(reference, period_id="period-1")
            monitor.reset_for_promotion(reference, period_id="period-2")
            with self.assertRaises(MonitorRejected):
                monitor.update(_observation(50, 11.0), period_id="period-1")
            _, second = _feed(monitor, sink, 60, 2)
            self.assertEqual([alert.period_id for alert in first + second], ["period-1", "period-2"])
            self.assertEqual(len(store.load_alerts()), 2)
            restored = _monitor(clock, period_id="period-2")
            self.assertEqual(restored.restore_open_incidents(store.load_alerts()), 1)
            store.close()

    def test_slice_alarm_without_aggregate_alarm_raises_nothing(self) -> None:
        clock = _Clock()
        monitor = _monitor(clock)
        sink = LocalAlertSink()
        raised = []
        for index in range(1, 7):
            raised.extend(monitor.update(_observation(index, 3.0), slice_name="difficulty:3"))
            for offset in range(3):
                raised.extend(
                    monitor.update(
                        _observation(100 * index + offset, 0.0), slice_name="difficulty:1"
                    )
                )
        self.assertEqual(raised, [])
        self.assertEqual(sink.deliver(raised), ())

    def test_period_is_required(self) -> None:
        with self.assertRaises(MonitorRejected):
            ProductionMonitor(
                _settings(),
                FrozenReference(configuration_hash=_HASH, baselines=(("tool_error_count", 1.0),)),
                clock=_Clock(),
                period_id="",
            )


if __name__ == "__main__":
    unittest.main()
