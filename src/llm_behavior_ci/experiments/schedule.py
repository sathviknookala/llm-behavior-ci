"""Explicit, hashed arrival schedules for the monitor and canary tiers.

A ``BenchmarkSchedule`` fixes everything that decides which episode runs
when: the seeded stream (task order, task-mix rule, arrival offsets), the
healthy prefix, the onset shape (abrupt, or a deterministic ramp), the
analysis horizon, the canary fraction and assignment seed, and the start
of a simulated clock. ``schedule_hash`` binds all of it, and a checkpoint
records the hash so a resumed run refuses a different schedule.

``plan_arrivals`` expands a schedule into ``ScheduledArrival`` rows. Every
decision is a pure function of the schedule and the arrival index, so an
uninterrupted run and a resumed run make the same decisions.

``run_scheduled_monitor`` drives the production monitor over those
arrivals: arrivals before onset are served by the healthy configuration,
arrivals after onset by the faulted one (all of them for an abrupt onset,
a deterministic, growing share across a ramp). The monitor's clock is the
simulated clock, every observation is stamped with its arrival's simulated
time, and outcome delay is the monitor's own ``outcome_delay_seconds``
measured on that clock. Each processed arrival stores its exposure, its
detector inputs, and the alerts it released; on resume the stored inputs
are replayed through fresh detectors before new arrivals run. Alarms
before onset are reported apart from post-onset detection.

The same function serves the frozen benchmark and a dev rehearsal; only
the task set, the configurations, and the lock differ. Synthetic runs of
this code are software evidence, not measurements.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any

from llm_behavior_ci.config import (
    RunConfiguration,
    RunIdentity,
    StreamSettings,
    new_run_identity,
    run_configuration_hash,
)
from llm_behavior_ci.lifecycle.canary import assign_canary
from llm_behavior_ci.lifecycle.monitoring import (
    Alert,
    DistributionalMonitor,
    MissingEvaluatorOutcome,
    PLAN_KL_SIGNAL,
    PLAN_QUALITY_SIGNAL,
    ProductionMonitor,
    TaskMetadata,
    UndefinedRequirementFraction,
    observation_from_episode,
    task_mix_observation_from_episode,
    tool_selection_observation_from_episode,
)
from llm_behavior_ci.records import (
    MonitorObservation,
    TaskMixObservation,
    ToolSelectionObservation,
)
from llm_behavior_ci.runtime.episode import run_episode
from llm_behavior_ci.tasks.selection import TaskSet
from llm_behavior_ci.tasks.streams import generate_stream

SCHEDULE_VERSION = "benchmark-schedule-v1"
ONSET_MODES = frozenset({"abrupt", "ramp"})


class ScheduleError(ValueError):
    """A schedule is malformed, does not fit its task set, or disagrees with a checkpoint."""


def _nonnegative(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ScheduleError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class BenchmarkSchedule:
    """Every input that decides which configuration serves which arrival."""

    stream: StreamSettings
    healthy_prefix_episodes: int
    onset_mode: str
    ramp_episodes: int
    analysis_horizon_episodes: int
    canary_fraction: float
    canary_assignment_seed: int
    clock_start: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.stream, StreamSettings):
            raise ScheduleError("stream must be StreamSettings")
        _nonnegative(self.healthy_prefix_episodes, "healthy_prefix_episodes")
        if self.onset_mode not in ONSET_MODES:
            raise ScheduleError("onset_mode must be abrupt or ramp")
        _nonnegative(self.ramp_episodes, "ramp_episodes")
        if self.onset_mode == "abrupt" and self.ramp_episodes != 0:
            raise ScheduleError("an abrupt onset has ramp_episodes 0")
        if self.onset_mode == "ramp" and self.ramp_episodes < 1:
            raise ScheduleError("a ramp onset needs ramp_episodes >= 1")
        horizon = _nonnegative(self.analysis_horizon_episodes, "analysis_horizon_episodes")
        if horizon <= self.healthy_prefix_episodes:
            raise ScheduleError("analysis_horizon_episodes must exceed the healthy prefix")
        if (
            isinstance(self.canary_fraction, bool)
            or not isinstance(self.canary_fraction, float)
            or not 0.0 < self.canary_fraction <= 1.0
        ):
            raise ScheduleError("canary_fraction must be a float in (0, 1]")
        if isinstance(self.canary_assignment_seed, bool) or not isinstance(
            self.canary_assignment_seed, int
        ):
            raise ScheduleError("canary_assignment_seed must be an integer")
        if not isinstance(self.clock_start, datetime) or self.clock_start.tzinfo is None:
            raise ScheduleError("clock_start must be a timezone-aware datetime")

    @property
    def onset_index(self) -> int:
        return self.healthy_prefix_episodes

    def to_dict(self) -> dict[str, object]:
        return {
            "schedule_version": SCHEDULE_VERSION,
            "stream": self.stream.to_dict(),
            "healthy_prefix_episodes": self.healthy_prefix_episodes,
            "onset_mode": self.onset_mode,
            "ramp_episodes": self.ramp_episodes,
            "analysis_horizon_episodes": self.analysis_horizon_episodes,
            "canary_fraction": self.canary_fraction,
            "canary_assignment_seed": self.canary_assignment_seed,
            "clock_start": self.clock_start.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: object) -> BenchmarkSchedule:
        if not isinstance(payload, Mapping):
            raise ScheduleError("schedule must be an object")
        expected = set(cls.__dataclass_fields__) | {"schedule_version"}
        if set(payload) != expected:
            raise ScheduleError(
                "schedule fields must be exactly: " + ", ".join(sorted(expected))
            )
        if payload["schedule_version"] != SCHEDULE_VERSION:
            raise ScheduleError(f"schedule_version must be {SCHEDULE_VERSION}")
        try:
            start = datetime.fromisoformat(str(payload["clock_start"]))
        except ValueError as error:
            raise ScheduleError("clock_start must be an ISO timestamp") from error
        fraction = payload["canary_fraction"]
        if isinstance(fraction, int) and not isinstance(fraction, bool):
            fraction = float(fraction)
        return cls(
            stream=StreamSettings.from_dict(payload["stream"]),
            healthy_prefix_episodes=payload["healthy_prefix_episodes"],
            onset_mode=payload["onset_mode"],
            ramp_episodes=payload["ramp_episodes"],
            analysis_horizon_episodes=payload["analysis_horizon_episodes"],
            canary_fraction=fraction,
            canary_assignment_seed=payload["canary_assignment_seed"],
            clock_start=start,
        )


def schedule_hash(schedule: BenchmarkSchedule) -> str:
    text = json.dumps(schedule.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unit(*parts: object) -> float:
    material = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") / 2**64


@dataclass(frozen=True)
class ScheduledArrival:
    """One arrival and every decision the schedule makes about it."""

    index: int
    task_id: str
    scenario_id: str | None
    simulated_at: datetime
    phase: str
    exposure: str
    canary_assigned: bool

    def decision_dict(self) -> dict[str, object]:
        """The persisted decisions, without the local task id."""

        return {
            "index": self.index,
            "simulated_at": self.simulated_at.isoformat(),
            "phase": self.phase,
            "exposure": self.exposure,
            "canary_assigned": self.canary_assigned,
        }


def exposure_for(schedule: BenchmarkSchedule, index: int) -> str:
    """``healthy`` or ``faulted`` for arrival ``index``; pure in the schedule."""

    onset = schedule.onset_index
    if index < onset:
        return "healthy"
    if schedule.onset_mode == "abrupt" or index >= onset + schedule.ramp_episodes:
        return "faulted"
    share = (index - onset + 1) / (schedule.ramp_episodes + 1)
    seed = schedule.stream.stream_seed
    return "faulted" if _unit("ramp", seed, index) < share else "healthy"


def plan_arrivals(task_set: TaskSet, schedule: BenchmarkSchedule) -> tuple[ScheduledArrival, ...]:
    """Expand the schedule over ``task_set`` up to the analysis horizon."""

    if not isinstance(task_set, TaskSet):
        raise ScheduleError("plan_arrivals requires a task set")
    rows: list[ScheduledArrival] = []
    for arrival in generate_stream(task_set, schedule.stream):
        if arrival.index >= schedule.analysis_horizon_episodes:
            break
        rows.append(
            ScheduledArrival(
                index=arrival.index,
                task_id=arrival.task_id,
                scenario_id=arrival.scenario_id,
                simulated_at=schedule.clock_start
                + timedelta(seconds=float(arrival.scheduled_offset_seconds)),
                phase="healthy_prefix" if arrival.index < schedule.onset_index else "post_onset",
                exposure=exposure_for(schedule, arrival.index),
                canary_assigned=assign_canary(
                    f"{schedule.stream.stream_seed}:{arrival.index}",
                    fraction=schedule.canary_fraction,
                    seed=schedule.canary_assignment_seed,
                ),
            )
        )
    if len(rows) < schedule.analysis_horizon_episodes:
        raise ScheduleError(
            "the stream ends before the analysis horizon; use a stream with replacement "
            "or a shorter horizon"
        )
    return tuple(rows)


class SimulatedClock:
    """A clock that moves only when the schedule moves it."""

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise ScheduleError("simulated clock start must be timezone-aware")
        self._now = start

    def __call__(self) -> datetime:
        return self._now

    def advance_to(self, moment: datetime) -> None:
        if moment < self._now:
            raise ScheduleError("simulated clock cannot move backward")
        self._now = moment

    def advance_by(self, seconds: float) -> None:
        self.advance_to(self._now + timedelta(seconds=float(seconds)))


@dataclass
class StreamCounters:
    """Separate counts; one number never stands in for another."""

    arrivals: int = 0
    healthy_exposures: int = 0
    faulted_exposures: int = 0
    candidate_exposures: int = 0
    paired_outcomes: int = 0
    completed_evaluator_outcomes: int = 0
    missing_evaluator_outcomes: int = 0
    production_only_arrivals: int = 0

    def to_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class ScheduledMonitorResult:
    status: str
    healthy_prefix_alarms: int
    first_post_onset_alert_index: int | None
    post_onset_delay_episodes: int | None
    counters: StreamCounters
    alerts: tuple[dict[str, object], ...] = field(default_factory=tuple)
    prefix_incident_open_at_onset: bool = False
    post_onset_suppressed_alarms: int = 0

    def detected(self) -> bool:
        return self.first_post_onset_alert_index is not None


def _alert_summary(alert: Alert, index: int) -> dict[str, object]:
    return {
        "index": index,
        "signal": alert.signal,
        "slice_name": alert.slice_name,
        "method": alert.method,
        "configuration_hash": alert.configuration_hash,
    }


class _Detectors:
    def __init__(
        self,
        monitor: ProductionMonitor,
        distributional: Mapping[str, DistributionalMonitor],
    ) -> None:
        self.monitor = monitor
        self.tool = distributional.get("tool_selection")
        self.mix = distributional.get("task_mix")

    def suppressed(self) -> dict[str, int]:
        return self.monitor.suppressed_alarms

    def feed(self, item: Mapping[str, Any]) -> list[Alert]:
        alerts: list[Alert] = []
        for payload in item.get("observations", ()):
            alerts.extend(self.monitor.update(MonitorObservation.from_dict(payload)))
        alerts.extend(self.monitor.release_due())
        tool = item.get("tool_selection")
        if self.tool is not None and tool is not None:
            alerts.extend(self.tool.update(ToolSelectionObservation.from_dict(tool)))
        mix = item.get("task_mix")
        if self.mix is not None and mix is not None:
            alerts.extend(self.mix.update(TaskMixObservation.from_dict(mix)))
        return alerts


def _detector_inputs(
    episode: object,
    *,
    index: int,
    simulated_at: datetime,
    signals: Sequence[str],
    difficulty: int | None,
    task_mix: str | None,
) -> dict[str, Any]:
    observations: list[dict[str, object]] = []
    for signal in signals:
        if signal in (PLAN_QUALITY_SIGNAL, PLAN_KL_SIGNAL):
            continue
        try:
            observation = observation_from_episode(
                episode,
                task_metadata=TaskMetadata(
                    signal=signal,
                    completion_index=index,
                    difficulty=difficulty,
                    task_mix=task_mix,
                ),
            )
        except (MissingEvaluatorOutcome, UndefinedRequirementFraction):
            continue
        observations.append(replace(observation, observed_at=simulated_at).to_dict())
    metadata = TaskMetadata(
        signal="tool_selection",
        completion_index=index,
        difficulty=difficulty,
        task_mix=task_mix,
    )
    tool = replace(
        tool_selection_observation_from_episode(episode, task_metadata=metadata),
        observed_at=simulated_at,
    ).to_dict()
    mix = None
    if task_mix is not None:
        mix = replace(
            task_mix_observation_from_episode(
                episode, task_metadata=replace(metadata, signal="task_mix")
            ),
            observed_at=simulated_at,
        ).to_dict()
    return {"observations": observations, "tool_selection": tool, "task_mix": mix}


def run_scheduled_monitor(
    *,
    schedule: BenchmarkSchedule,
    arrivals: Sequence[ScheduledArrival],
    healthy: RunConfiguration,
    faulted: RunConfiguration,
    runtime_for: Callable[[RunConfiguration], object],
    monitor: ProductionMonitor,
    distributional: Mapping[str, DistributionalMonitor],
    clock: SimulatedClock,
    state: dict[str, Any],
    persist: Callable[[], None],
    difficulty_for: Callable[[str], int | None] | None = None,
    should_interrupt: Callable[[int], bool] | None = None,
) -> ScheduledMonitorResult:
    """Serve the scheduled arrivals and feed every configured monitor.

    ``runtime_for`` builds a fresh runtime for one configuration.
    ``monitor`` and ``distributional`` must use ``clock``. ``state`` is the
    checkpoint section for this stream; ``persist`` writes it after every
    arrival. A task-mix label is ``difficulty:<n>`` when ``difficulty_for``
    knows the task.
    """

    digest = schedule_hash(schedule)
    recorded = state.get("schedule_hash")
    if recorded is not None and recorded != digest:
        raise ScheduleError("checkpoint was written under a different schedule")
    state["schedule_hash"] = digest
    runs = state.setdefault("runs", {})
    identities: dict[str, RunIdentity] = {}
    for label, configuration in (("healthy", healthy), ("faulted", faulted)):
        if label in runs:
            identities[label] = RunIdentity.from_dict(runs[label])
        else:
            identities[label] = new_run_identity(configuration)
            runs[label] = identities[label].to_dict()
    items: list[dict[str, Any]] = state.setdefault("items", [])
    counters = StreamCounters()
    detectors = _Detectors(monitor, distributional)
    alerts: list[dict[str, object]] = []
    suppressed_post_onset = [0]
    prefix_signals: set[str] = set()

    def suppressed_by_prefix(before: Mapping[str, int], index: int) -> None:
        if index < schedule.onset_index:
            return
        after = detectors.suppressed()
        suppressed_post_onset[0] += sum(
            after.get(signal, 0) - before.get(signal, 0) for signal in prefix_signals
        )

    def account(item: Mapping[str, Any], fired: Sequence[Alert], before: Mapping[str, int]) -> None:
        index = int(item["index"])
        suppressed_by_prefix(before, index)
        if index < schedule.onset_index:
            prefix_signals.update(alert.signal for alert in fired)
        counters.arrivals += 1
        if item["exposure"] == "healthy":
            counters.healthy_exposures += 1
        else:
            counters.faulted_exposures += 1
        if item.get("evaluator_outcome_present"):
            counters.completed_evaluator_outcomes += 1
        else:
            counters.missing_evaluator_outcomes += 1
        alerts.extend(_alert_summary(alert, int(item["index"])) for alert in fired)

    by_index = {int(item["index"]): item for item in items}
    for arrival in arrivals:
        stored = by_index.get(arrival.index)
        if stored is not None:
            if stored.get("decision") != arrival.decision_dict():
                raise ScheduleError("checkpoint decisions disagree with the schedule")
            clock.advance_to(arrival.simulated_at)
            before = detectors.suppressed()
            fired = detectors.feed(stored)
            account(stored, fired, before)
            continue
        if should_interrupt is not None and should_interrupt(arrival.index):
            persist()
            return _monitor_result(
                "interrupted", schedule, alerts, counters, suppressed_post_onset[0]
            )
        clock.advance_to(arrival.simulated_at)
        configuration = healthy if arrival.exposure == "healthy" else faulted
        episode = run_episode(
            arrival.task_id,
            configuration,
            "execute",
            run=identities[arrival.exposure],
            runtime=runtime_for(configuration),
            scenario_id=arrival.scenario_id,
        )
        difficulty = None if difficulty_for is None else difficulty_for(arrival.task_id)
        item: dict[str, Any] = {
            "index": arrival.index,
            "decision": arrival.decision_dict(),
            "exposure": arrival.exposure,
            "configuration_hash": run_configuration_hash(configuration),
            "episode_status": episode.status,
            "evaluator_outcome_present": episode.evaluator_outcome is not None,
            **_detector_inputs(
                episode,
                index=arrival.index,
                simulated_at=arrival.simulated_at,
                signals=monitor.signals,
                difficulty=difficulty,
                task_mix=None if difficulty is None else f"difficulty:{difficulty}",
            ),
        }
        before = detectors.suppressed()
        fired = detectors.feed(item)
        items.append(item)
        by_index[arrival.index] = item
        account(item, fired, before)
        persist()
    if arrivals:
        clock.advance_by(monitor.outcome_delay_seconds)
        before = detectors.suppressed()
        tail = monitor.release_due()
        suppressed_by_prefix(before, arrivals[-1].index)
        alerts.extend(_alert_summary(alert, arrivals[-1].index) for alert in tail)
    return _monitor_result("completed", schedule, alerts, counters, suppressed_post_onset[0])


def _monitor_result(
    status: str,
    schedule: BenchmarkSchedule,
    alerts: Sequence[dict[str, object]],
    counters: StreamCounters,
    suppressed_post_onset: int,
) -> ScheduledMonitorResult:
    onset = schedule.onset_index
    prefix = sum(1 for alert in alerts if int(alert["index"]) < onset)
    post = [int(alert["index"]) for alert in alerts if int(alert["index"]) >= onset]
    first = min(post) if post else None
    return ScheduledMonitorResult(
        status=status,
        healthy_prefix_alarms=prefix,
        first_post_onset_alert_index=first,
        post_onset_delay_episodes=None if first is None else first - onset + 1,
        counters=counters,
        alerts=tuple(alerts),
        prefix_incident_open_at_onset=prefix > 0,
        post_onset_suppressed_alarms=suppressed_post_onset,
    )
