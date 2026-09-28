from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

from llm_behavior_ci.config import MONITOR_SIGNALS, MonitorSettings, StoppingRule
from llm_behavior_ci.records import (
    EpisodeResult,
    MonitorObservation,
    assert_public_payload,
)
from llm_behavior_ci.stats.adwin import ADWIN
from llm_behavior_ci.stats.confidence_sequence import BoundedMeanCS
from llm_behavior_ci.stats.cusum import CUSUM
from llm_behavior_ci.stats.e_detector import BettingEDetector
from llm_behavior_ci.stats.evidence import Detector, Evidence

_BOUNDED_SIGNALS = frozenset({"task_success", "requirement_fraction"})
_DELAYED_SIGNALS = _BOUNDED_SIGNALS


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


@dataclass(frozen=True)
class NormalizedEpisode:
    """Local normalized monitor inputs for one episode.

    ``tool_selection`` and ``task_mix`` are normalized here and are not
    members of ``MONITOR_SIGNALS``.
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


@dataclass(frozen=True)
class _HeldObservation:
    observation: MonitorObservation
    completion_index: int | None


class ProductionMonitor:
    """Sequential production monitors against a frozen known-good reference.

    Reference baselines are caller-supplied and are never refit from
    ``update`` observations. ``tool_selection`` and ``task_mix`` stay on
    ``NormalizedEpisode`` and are not monitored series.
    """

    def __init__(
        self,
        settings: MonitorSettings,
        reference: FrozenReference,
        *,
        clock: Callable[[], datetime],
        dedup_seconds: float,
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
        self._settings = settings
        self._clock = clock
        self._dedup_seconds = dedup_value
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

    def update(self, observation: MonitorObservation) -> tuple[Alert, ...]:
        if not isinstance(observation, MonitorObservation):
            raise MonitorRejected("observation must be MonitorObservation")
        if observation.signal not in self._signals:
            raise MonitorRejected("signal is outside settings.signals")
        key = (observation.episode.episode_id, observation.signal)
        if key in self._seen:
            raise RepeatedEpisode("episode id and signal were already observed")
        self._seen.add(key)
        now = self._clock()
        delayed = (
            observation.signal in _DELAYED_SIGNALS
            and now < observation.observed_at + self._outcome_delay
        )
        alerts: list[Alert] = []
        if delayed:
            self._held.append(
                _HeldObservation(observation=observation, completion_index=None)
            )
            alerts.extend(self._release_ready(now))
            return tuple(alerts)
        alerts.extend(self._release_ready(now))
        alerts.extend(self._apply(observation))
        return tuple(alerts)

    def reset_for_promotion(self, reference: FrozenReference) -> None:
        if not isinstance(reference, FrozenReference):
            raise MonitorRejected("reference must be FrozenReference")
        self._reference = reference
        self._reference_configuration_hash = reference.configuration_hash
        self._detectors = self._detectors_from(reference)
        self._held.clear()
        self._seen.clear()
        self._last_alert_at.clear()

    def _detectors_from(self, reference: FrozenReference) -> dict[str, list[Detector]]:
        baselines = _baselines_for_signals(reference.baselines, self._settings.signals)
        return _build_detectors(
            signals=self._settings.signals,
            stopping_rules=self._settings.stopping_rules,
            baselines=baselines,
        )

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
            alerts.extend(self._apply(held.observation))
        return alerts

    def _apply(self, observation: MonitorObservation) -> list[Alert]:
        detectors = self._detectors[observation.signal]
        alerts: list[Alert] = []
        for detector in detectors:
            evidence = detector.update(observation.value)
            if not evidence.alarm:
                continue
            alert = self._alert_from_evidence(observation, evidence)
            if alert is None:
                continue
            alerts.append(alert)
        return alerts

    def _alert_from_evidence(
        self,
        observation: MonitorObservation,
        evidence: Evidence,
    ) -> Alert | None:
        raised_at = self._clock()
        last = self._last_alert_at.get(observation.signal)
        if (
            self._dedup_seconds > 0.0
            and last is not None
            and (raised_at - last).total_seconds() < self._dedup_seconds
        ):
            return None
        self._last_alert_at[observation.signal] = raised_at
        return Alert(
            configuration_hash=observation.run.configuration_hash,
            reference_configuration_hash=self._reference_configuration_hash,
            signal=observation.signal,
            slice_name=observation.signal,
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


def _bounds(signal: str, baseline: float) -> tuple[float, float]:
    if signal in _BOUNDED_SIGNALS:
        return (0.0, 1.0)
    return (0.0, max(baseline * 4.0, 1.0))


def _build_detector(
    rule: StoppingRule,
    *,
    signal: str,
    baseline: float,
) -> Detector:
    if rule.name == "cusum":
        if rule.threshold is None:
            raise MonitorRejected("cusum requires a threshold")
        direction = "decrease" if signal in _BOUNDED_SIGNALS else "increase"
        return CUSUM(
            target=baseline,
            slack=0.0,
            threshold=rule.threshold,
            direction=direction,
        )
    if rule.name == "adwin":
        return ADWIN(delta=rule.alpha)
    lower, upper = _bounds(signal, baseline)
    if rule.name == "e_detector":
        if not lower < baseline < upper:
            raise MonitorRejected(
                "e_detector null_mean must lie strictly inside (lower, upper)"
            )
        direction = "below" if signal in _BOUNDED_SIGNALS else "above"
        return BettingEDetector(
            null_mean=baseline,
            alpha=rule.alpha,
            lower=lower,
            upper=upper,
            direction=direction,
        )
    if rule.name == "bounded_mean_cs":
        return BoundedMeanCS(
            alpha=rule.alpha,
            lower=lower,
            upper=upper,
            null_mean=baseline,
        )
    raise MonitorRejected(f"unknown stopping rule: {rule.name}")


def _build_detectors(
    *,
    signals: tuple[str, ...],
    stopping_rules: tuple[StoppingRule, ...],
    baselines: dict[str, float],
) -> dict[str, list[Detector]]:
    detectors: dict[str, list[Detector]] = {}
    for signal in signals:
        baseline = baselines[signal]
        built: list[Detector] = []
        for rule in stopping_rules:
            built.append(_build_detector(rule, signal=signal, baseline=baseline))
        detectors[signal] = built
    return detectors
