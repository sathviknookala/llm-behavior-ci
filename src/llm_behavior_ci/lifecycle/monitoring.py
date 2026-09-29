from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

from llm_behavior_ci.config import MONITOR_SIGNALS, MonitorSettings
from llm_behavior_ci.lifecycle.detectors import (
    BOUNDED_SIGNALS,
    DetectorConstructionError,
    build_detectors,
)
from llm_behavior_ci.records import (
    EpisodeResult,
    MonitorObservation,
    NamedCount,
    TaskMixObservation,
    ToolSelectionObservation,
    assert_public_payload,
)
from llm_behavior_ci.stats.evidence import Detector, Evidence
from llm_behavior_ci.storage import (
    AlertRecord,
    DeploymentDecisionRecord,
    EpisodeStore,
)

_DELAYED_SIGNALS = BOUNDED_SIGNALS
_DISTRIBUTIONAL_SIGNALS = frozenset({"tool_selection", "task_mix"})


class MissingEvaluatorOutcome(ValueError):
    """Raised when a delayed evaluator signal is requested without an outcome."""


class UndefinedRequirementFraction(ValueError):
    """Raised when requirement_fraction is requested with a zero total."""


class RepeatedEpisode(ValueError):
    """Raised when the same episode id and signal are observed twice."""


class MonitorRejected(ValueError):
    """Raised when monitor construction or an update is rejected."""


@dataclass(frozen=True)
class TaskMetadata:
    """Caller-supplied signal selection and public task-mix labels."""

    signal: str
    completion_index: int
    difficulty: int | None = None
    task_mix: str | None = None
    slice_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.completion_index, int) or isinstance(
            self.completion_index, bool
        ):
            raise MonitorRejected("completion_index must be a nonnegative integer")
        if self.completion_index < 0:
            raise MonitorRejected("completion_index must be a nonnegative integer")
        if self.difficulty is not None and self.difficulty not in (1, 2, 3):
            raise MonitorRejected("difficulty must be 1, 2, or 3")
        if self.task_mix is not None and (
            not isinstance(self.task_mix, str) or self.task_mix == ""
        ):
            raise MonitorRejected("task_mix must be a non-empty string when set")
        if self.slice_id is not None and (
            not isinstance(self.slice_id, str) or self.slice_id == ""
        ):
            raise MonitorRejected("slice_id must be a non-empty string when set")


@dataclass(frozen=True)
class NormalizedEpisode:
    """Local normalized monitor inputs for one episode.

    ``tool_selection`` and ``task_mix`` are normalized here and are not
    members of ``MONITOR_SIGNALS``. Use the typed distributional
    observation builders when the caller asks for those series.
    """

    task_success: float | None
    requirement_fraction: float | None
    tool_error_count: float
    invalid_tool_call_count: float
    trajectory_length: float
    tool_selection: tuple[tuple[str, int], ...]
    task_mix: str | None
    completion_index: int
    missing_outcome: bool


@dataclass(frozen=True)
class FrozenReference:
    """Caller-supplied frozen baselines for the previous known-good config."""

    configuration_hash: str
    baselines: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class Alert:
    """A public production-monitor alert without protected task content."""

    configuration_hash: str
    reference_configuration_hash: str
    signal: str
    slice_name: str
    method: str
    estimate: float
    boundary: float | None
    sample_size: int
    raised_at: datetime

    def to_public_dict(self) -> dict:
        payload = {
            "configuration_hash": self.configuration_hash,
            "reference_configuration_hash": self.reference_configuration_hash,
            "signal": self.signal,
            "slice_name": self.slice_name,
            "method": self.method,
            "estimate": self.estimate,
            "boundary": self.boundary,
            "sample_size": self.sample_size,
            "raised_at": self.raised_at.isoformat(),
        }
        assert_public_payload(payload)
        return payload

    def to_record(self) -> AlertRecord:
        return AlertRecord(
            configuration_hash=self.configuration_hash,
            reference_configuration_hash=self.reference_configuration_hash,
            signal=self.signal,
            slice_name=self.slice_name,
            method=self.method,
            estimate=self.estimate,
            boundary=self.boundary,
            sample_size=self.sample_size,
            raised_at=self.raised_at,
        )

    @classmethod
    def from_record(cls, record: AlertRecord) -> Alert:
        return cls(
            configuration_hash=record.configuration_hash,
            reference_configuration_hash=record.reference_configuration_hash,
            signal=record.signal,
            slice_name=record.slice_name,
            method=record.method,
            estimate=record.estimate,
            boundary=record.boundary,
            sample_size=record.sample_size,
            raised_at=record.raised_at,
        )


class LocalAlertSink:
    """In-process alert delivery without external notification accounts."""

    def __init__(
        self,
        store: EpisodeStore | None = None,
        *,
        dedup_seconds: float = 0.0,
    ) -> None:
        if store is not None and not isinstance(store, EpisodeStore):
            raise MonitorRejected("store must be an EpisodeStore or None")
        if isinstance(dedup_seconds, bool) or not isinstance(
            dedup_seconds, (int, float)
        ):
            raise MonitorRejected("dedup_seconds must be a finite float >= 0")
        window = float(dedup_seconds)
        if not math.isfinite(window) or window < 0.0:
            raise MonitorRejected("dedup_seconds must be a finite float >= 0")
        self._store = store
        self._dedup_seconds = window
        self._delivered: list[Alert] = []

    def deliver(self, alerts: Sequence[Alert]) -> tuple[Alert, ...]:
        if not isinstance(alerts, Sequence) or isinstance(alerts, (str, bytes)):
            raise MonitorRejected("alerts must be a sequence of Alert")
        delivered: list[Alert] = []
        for alert in alerts:
            if not isinstance(alert, Alert):
                raise MonitorRejected("alerts must be a sequence of Alert")
            chosen = alert
            if self._store is not None:
                stored, inserted = self._store.append_alert_with_status(
                    alert.to_record(),
                    dedup_seconds=self._dedup_seconds,
                )
                chosen = Alert.from_record(stored)
                if inserted:
                    self._store.append_deployment_decision(
                        DeploymentDecisionRecord(
                            configuration_hash=chosen.configuration_hash,
                            reference_configuration_hash=(
                                chosen.reference_configuration_hash
                            ),
                            signal=chosen.signal,
                            slice_name=chosen.slice_name,
                            decision="alert",
                            method=chosen.method,
                            estimate=chosen.estimate,
                            boundary=chosen.boundary,
                            sample_size=chosen.sample_size,
                            decided_at=chosen.raised_at,
                        )
                    )
            self._delivered.append(chosen)
            delivered.append(chosen)
        return tuple(delivered)

    @property
    def delivered(self) -> tuple[Alert, ...]:
        return tuple(self._delivered)


def normalize_episode(
    episode: EpisodeResult,
    *,
    task_metadata: TaskMetadata,
) -> NormalizedEpisode:
    """Normalize episode fields for monitoring without substituting missing outcomes."""

    missing = episode.evaluator_outcome is None
    if missing:
        task_success: float | None = None
        requirement_fraction: float | None = None
    else:
        outcome = episode.evaluator_outcome
        task_success = 1.0 if outcome.success else 0.0
        requirement_fraction = outcome.requirement_fraction
    tool_error_count = float(
        sum(1 for step in episode.tool_steps if step.error is not None)
    )
    invalid_tool_call_count = float(
        sum(
            1
            for step in episode.tool_steps
            if step.app_name is None and step.api_name is None
        )
    )
    trajectory_length = float(len(episode.model_steps) + len(episode.tool_steps))
    counts: Counter[str] = Counter()
    for step in episode.tool_steps:
        if step.api_name is not None:
            name = step.api_name
        elif step.app_name is not None:
            name = step.app_name
        else:
            name = "unparsed"
        counts[name] += 1
    tool_selection = tuple(sorted(counts.items(), key=lambda item: item[0]))
    return NormalizedEpisode(
        task_success=task_success,
        requirement_fraction=requirement_fraction,
        tool_error_count=tool_error_count,
        invalid_tool_call_count=invalid_tool_call_count,
        trajectory_length=trajectory_length,
        tool_selection=tool_selection,
        task_mix=task_metadata.task_mix,
        completion_index=task_metadata.completion_index,
        missing_outcome=missing,
    )


def observation_from_episode(
    episode: EpisodeResult,
    *,
    task_metadata: TaskMetadata,
) -> MonitorObservation:
    """Build one ``MonitorObservation`` for a configured monitor signal."""

    signal = task_metadata.signal
    if signal in _DISTRIBUTIONAL_SIGNALS:
        raise MonitorRejected(
            "distributional signals require typed observations, not MonitorObservation"
        )
    if signal not in MONITOR_SIGNALS:
        raise MonitorRejected("signal must be a monitor signal")
    normalized = normalize_episode(episode, task_metadata=task_metadata)
    if signal == "task_success":
        if normalized.missing_outcome:
            raise MissingEvaluatorOutcome("task_success requires an evaluator outcome")
        assert normalized.task_success is not None
        value = normalized.task_success
    elif signal == "requirement_fraction":
        if normalized.missing_outcome:
            raise MissingEvaluatorOutcome(
                "requirement_fraction requires an evaluator outcome"
            )
        if normalized.requirement_fraction is None:
            raise UndefinedRequirementFraction(
                "requirement_fraction is undefined when total_requirements is zero"
            )
        value = normalized.requirement_fraction
    elif signal == "tool_error_count":
        value = normalized.tool_error_count
    elif signal == "invalid_tool_call_count":
        value = normalized.invalid_tool_call_count
    elif signal == "trajectory_length":
        value = normalized.trajectory_length
    else:
        raise MonitorRejected("signal must be a monitor signal")
    return MonitorObservation(
        episode=episode.episode,
        run=episode.run,
        split=episode.task.split,
        signal=signal,
        value=value,
        observed_at=episode.ended_at,
    )


def tool_selection_observation_from_episode(
    episode: EpisodeResult,
    *,
    task_metadata: TaskMetadata,
) -> ToolSelectionObservation:
    """Build a typed tool-selection observation when the caller asks."""

    normalized = normalize_episode(episode, task_metadata=task_metadata)
    return ToolSelectionObservation(
        episode=episode.episode,
        run=episode.run,
        split=episode.task.split,
        counts=tuple(
            NamedCount(name=name, count=count)
            for name, count in normalized.tool_selection
        ),
        observed_at=episode.ended_at,
        completion_index=task_metadata.completion_index,
    )


def task_mix_observation_from_episode(
    episode: EpisodeResult,
    *,
    task_metadata: TaskMetadata,
) -> TaskMixObservation:
    """Build a typed task-mix observation when the caller asks."""

    if task_metadata.task_mix is None:
        raise MonitorRejected("task_mix observation requires task_metadata.task_mix")
    return TaskMixObservation(
        episode=episode.episode,
        run=episode.run,
        split=episode.task.split,
        label=task_metadata.task_mix,
        observed_at=episode.ended_at,
        completion_index=task_metadata.completion_index,
    )


@dataclass(frozen=True)
class _HeldObservation:
    observation: MonitorObservation
    completion_index: int | None
    slice_name: str


class ProductionMonitor:
    """Sequential production monitors against a frozen known-good reference.

    Reference baselines are caller-supplied and are never refit from
    ``update`` observations. ``tool_selection`` and ``task_mix`` use typed
    distributional observations and are not scalar ``MONITOR_SIGNALS``.
    When ``period_id`` is set, every update must supply the same period.
    """

    def __init__(
        self,
        settings: MonitorSettings,
        reference: FrozenReference,
        *,
        clock: Callable[[], datetime],
        dedup_seconds: float,
        period_id: str | None = None,
        use_slice_attribution: bool = False,
    ) -> None:
        if not isinstance(settings, MonitorSettings):
            raise MonitorRejected("settings must be MonitorSettings")
        if not callable(clock):
            raise MonitorRejected("clock must be callable")
        if isinstance(dedup_seconds, bool) or not isinstance(dedup_seconds, (int, float)):
            raise MonitorRejected("dedup_seconds must be a finite float >= 0")
        dedup_value = float(dedup_seconds)
        if not math.isfinite(dedup_value) or dedup_value < 0.0:
            raise MonitorRejected("dedup_seconds must be a finite float >= 0")
        if period_id is not None and (
            not isinstance(period_id, str) or period_id == ""
        ):
            raise MonitorRejected("period_id must be a non-empty string when set")
        if not isinstance(use_slice_attribution, bool):
            raise MonitorRejected("use_slice_attribution must be a boolean")
        self._settings = settings
        self._clock = clock
        self._dedup_seconds = dedup_value
        self._period_id = period_id
        self._use_slice_attribution = use_slice_attribution
        self._signals = frozenset(settings.signals)
        self._outcome_delay = timedelta(seconds=settings.outcome_delay_seconds)
        self._held: list[_HeldObservation] = []
        self._seen: set[tuple[str, str]] = set()
        self._last_alert_at: dict[str, datetime] = {}
        if reference.configuration_hash != settings.reference_configuration_hash:
            raise MonitorRejected(
                "reference.configuration_hash must equal settings.reference_configuration_hash"
            )
        self._reference_configuration_hash = settings.reference_configuration_hash
        self._reference = reference
        self._detectors = self._detectors_from(reference)

    @property
    def reference(self) -> FrozenReference:
        return self._reference

    @property
    def period_id(self) -> str | None:
        return self._period_id

    def update(
        self,
        observation: MonitorObservation,
        *,
        completion_index: int | None = None,
        period_id: str | None = None,
        slice_name: str | None = None,
    ) -> tuple[Alert, ...]:
        if isinstance(observation, (ToolSelectionObservation, TaskMixObservation)):
            raise MonitorRejected(
                "distributional observations cannot update a scalar monitor signal"
            )
        if not isinstance(observation, MonitorObservation):
            raise MonitorRejected("observation must be MonitorObservation")
        self._check_period(period_id)
        if observation.signal not in self._signals:
            raise MonitorRejected("signal is outside settings.signals")
        if completion_index is not None and (
            isinstance(completion_index, bool)
            or not isinstance(completion_index, int)
            or completion_index < 0
        ):
            raise MonitorRejected("completion_index must be a nonnegative integer")
        if slice_name is not None and (
            not isinstance(slice_name, str) or slice_name == ""
        ):
            raise MonitorRejected("slice_name must be a non-empty string when set")
        key = (observation.episode.episode_id, observation.signal)
        if key in self._seen:
            raise RepeatedEpisode("episode id and signal were already observed")
        self._seen.add(key)
        resolved_slice = observation.signal if slice_name is None else slice_name
        now = self._clock()
        delayed = (
            observation.signal in _DELAYED_SIGNALS
            and now < observation.observed_at + self._outcome_delay
        )
        alerts: list[Alert] = []
        if delayed:
            self._held.append(
                _HeldObservation(
                    observation=observation,
                    completion_index=completion_index,
                    slice_name=resolved_slice,
                )
            )
            alerts.extend(self._release_ready(now))
            return tuple(alerts)
        alerts.extend(self._release_ready(now))
        alerts.extend(self._apply(observation, slice_name=resolved_slice))
        return tuple(alerts)

    def update_from_episode(
        self,
        episode: EpisodeResult,
        *,
        task_metadata: TaskMetadata,
        period_id: str | None = None,
    ) -> tuple[Alert, ...]:
        observation = observation_from_episode(episode, task_metadata=task_metadata)
        slice_name = None
        if self._use_slice_attribution and task_metadata.slice_id is not None:
            slice_name = task_metadata.slice_id
        return self.update(
            observation,
            completion_index=task_metadata.completion_index,
            period_id=period_id,
            slice_name=slice_name,
        )

    def reset_for_promotion(self, reference: FrozenReference) -> None:
        if not isinstance(reference, FrozenReference):
            raise MonitorRejected("reference must be FrozenReference")
        self._reference = reference
        self._reference_configuration_hash = reference.configuration_hash
        self._detectors = self._detectors_from(reference)
        self._held.clear()
        self._seen.clear()
        self._last_alert_at.clear()

    def _check_period(self, period_id: str | None) -> None:
        if self._period_id is None:
            return
        if period_id != self._period_id:
            raise MonitorRejected("period_id does not match the monitoring period")

    def _detectors_from(self, reference: FrozenReference) -> dict[str, list[Detector]]:
        baselines = _baselines_for_signals(reference.baselines, self._settings.signals)
        try:
            return build_detectors(
                signals=self._settings.signals,
                stopping_rules=self._settings.stopping_rules,
                baselines=baselines,
            )
        except DetectorConstructionError as error:
            raise MonitorRejected(str(error)) from error

    def _release_ready(self, now: datetime) -> list[Alert]:
        ready: list[_HeldObservation] = []
        remaining: list[_HeldObservation] = []
        for held in self._held:
            if now >= held.observation.observed_at + self._outcome_delay:
                ready.append(held)
            else:
                remaining.append(held)
        self._held = remaining
        ready.sort(
            key=lambda item: (
                item.completion_index
                if item.completion_index is not None
                else 10**18,
                item.observation.observed_at,
            )
        )
        alerts: list[Alert] = []
        for held in ready:
            alerts.extend(
                self._apply(held.observation, slice_name=held.slice_name)
            )
        return alerts

    def _apply(
        self,
        observation: MonitorObservation,
        *,
        slice_name: str,
    ) -> list[Alert]:
        detectors = self._detectors[observation.signal]
        alerts: list[Alert] = []
        for detector in detectors:
            evidence = detector.update(observation.value)
            if not evidence.alarm:
                continue
            alert = self._alert_from_evidence(
                observation,
                evidence,
                slice_name=slice_name,
            )
            if alert is None:
                continue
            alerts.append(alert)
        return alerts

    def _alert_from_evidence(
        self,
        observation: MonitorObservation,
        evidence: Evidence,
        *,
        slice_name: str,
    ) -> Alert | None:
        raised_at = self._clock()
        dedup_key = f"{observation.signal}:{slice_name}"
        last = self._last_alert_at.get(dedup_key)
        if (
            self._dedup_seconds > 0.0
            and last is not None
            and (raised_at - last).total_seconds() < self._dedup_seconds
        ):
            return None
        self._last_alert_at[dedup_key] = raised_at
        return Alert(
            configuration_hash=observation.run.configuration_hash,
            reference_configuration_hash=self._reference_configuration_hash,
            signal=observation.signal,
            slice_name=slice_name,
            method=evidence.method,
            estimate=evidence.estimate,
            boundary=evidence.boundary,
            sample_size=evidence.sample_size,
            raised_at=raised_at,
        )


def _baselines_for_signals(
    baselines: tuple[tuple[str, float], ...],
    signals: tuple[str, ...],
) -> dict[str, float]:
    if not isinstance(baselines, tuple):
        raise MonitorRejected("baselines must be a tuple of (signal, estimate) pairs")
    mapping: dict[str, float] = {}
    for item in baselines:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
        ):
            raise MonitorRejected("baselines must be a tuple of (signal, estimate) pairs")
        signal, estimate = item
        if signal not in MONITOR_SIGNALS:
            raise MonitorRejected("baseline signal must be a monitor signal")
        if signal in mapping:
            raise MonitorRejected("baselines must include each signal exactly once")
        if isinstance(estimate, bool) or not isinstance(estimate, (int, float)):
            raise MonitorRejected("baseline estimate must be a finite float")
        value = float(estimate)
        if not math.isfinite(value):
            raise MonitorRejected("baseline estimate must be a finite float")
        mapping[signal] = value
    if set(mapping) != set(signals):
        raise MonitorRejected("baselines must include every settings.signals entry exactly once")
    return mapping
