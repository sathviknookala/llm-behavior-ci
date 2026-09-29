"""Offline detector replay on a frozen observation stream.

Factories produce one detector per method on a single-signal stream.
``Detector.update`` takes a float, so mixed-signal streams are the
caller's split before replay. Outcome delay matches production holding
for ``task_success`` and ``requirement_fraction``; non-delayed signals
apply immediately. Replay does not run an agent and uses no RNG.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from llm_behavior_ci.config import (
    DistributionalMonitorSettings,
    MonitorSettings,
    StoppingRule,
)
from llm_behavior_ci.lifecycle.detectors import (
    BOUNDED_SIGNALS,
    CORRECTIONS,
    MONITOR_RULE_NAMES,
    DetectorConstructionError,
    build_detector,
    build_distributional_detector,
    build_harmful_shift_adapter,
    build_hourly_window_detector,
)
from llm_behavior_ci.lifecycle.monitoring import FrozenReference
from llm_behavior_ci.records import MonitorObservation
from llm_behavior_ci.stats.evidence import Detector

_DELAYED_SIGNALS = BOUNDED_SIGNALS


class ReplayError(ValueError):
    """Raised when replay inputs or detector factories are rejected."""


@dataclass(frozen=True)
class ReplaySchedule:
    """Caller-supplied delay, horizon, onset, and aligned stream labels."""

    outcome_delay_seconds: float
    horizon_episodes: int
    onset_index: int
    scenario_keys: tuple[str, ...]
    arrival_times: tuple[datetime, ...]

    def __post_init__(self) -> None:
        if isinstance(self.outcome_delay_seconds, bool) or not isinstance(
            self.outcome_delay_seconds, (int, float)
        ):
            raise ReplayError("outcome_delay_seconds must be a finite float >= 0")
        delay = float(self.outcome_delay_seconds)
        if not math.isfinite(delay) or delay < 0.0:
            raise ReplayError("outcome_delay_seconds must be a finite float >= 0")
        if isinstance(self.horizon_episodes, bool) or not isinstance(
            self.horizon_episodes, int
        ):
            raise ReplayError("horizon_episodes must be an integer >= 1")
        if self.horizon_episodes < 1:
            raise ReplayError("horizon_episodes must be an integer >= 1")
        if isinstance(self.onset_index, bool) or not isinstance(self.onset_index, int):
            raise ReplayError("onset_index must be an integer >= 0")
        if self.onset_index < 0:
            raise ReplayError("onset_index must be an integer >= 0")
        if not isinstance(self.scenario_keys, tuple):
            raise ReplayError("scenario_keys must be a tuple of strings")
        for key in self.scenario_keys:
            if not isinstance(key, str):
                raise ReplayError("scenario_keys must be a tuple of strings")
        if not isinstance(self.arrival_times, tuple):
            raise ReplayError("arrival_times must be a tuple of timezone-aware datetimes")
        for arrival in self.arrival_times:
            if not isinstance(arrival, datetime):
                raise ReplayError(
                    "arrival_times must be a tuple of timezone-aware datetimes"
                )
            if arrival.tzinfo is None or arrival.utcoffset() is None:
                raise ReplayError("arrival_times must be timezone-aware")


@dataclass(frozen=True)
class ReplayResult:
    """Public replay metrics for one detector method on one stream hash."""

    method: str
    stream_hash: str
    first_detection_index: int | None
    detection_delay_episodes: int | None
    detection_delay_hours: float | None
    subsequent_alert_count: int
    missed_horizon: bool
    healthy_false_alarm_count: int
    post_horizon_detection_count: int
    replay_compute_seconds: float
    agent_execution_seconds: float | None
    scenario_alarm_counts: Mapping[str, int]
    observations_applied: int
    observations_withheld: int


@dataclass(frozen=True)
class _HeldItem:
    index: int
    observation: MonitorObservation
    scenario_key: str


@dataclass(frozen=True)
class _AppliedItem:
    index: int
    value: float
    scenario_key: str


def _canonical_json(document: object) -> bytes:
    try:
        text = json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ReplayError("stream hash input is not finite JSON") from error
    return text.encode("utf-8")


def _stream_hash(
    observations: Sequence[MonitorObservation],
    schedule: ReplaySchedule,
) -> str:
    rows: list[list[object]] = []
    for index, observation in enumerate(observations):
        rows.append(
            [
                observation.signal,
                observation.value,
                observation.observed_at.isoformat(),
                observation.split,
                schedule.arrival_times[index].isoformat(),
                schedule.scenario_keys[index],
                schedule.onset_index,
                schedule.horizon_episodes,
                schedule.outcome_delay_seconds,
            ]
        )
    return hashlib.sha256(_canonical_json(rows)).hexdigest()


def _validate_detector(detector: object) -> Detector:
    for name in ("update", "reset", "snapshot"):
        if not callable(getattr(detector, name, None)):
            raise ReplayError(
                "factory must return an object with update, reset, and snapshot"
            )
    return detector  # type: ignore[return-value]


def _validate_inputs(
    observations: Sequence[MonitorObservation],
    detector_factories: Mapping[str, Callable[[], Detector]],
    schedule: ReplaySchedule,
) -> tuple[MonitorObservation, ...]:
    if not isinstance(schedule, ReplaySchedule):
        raise ReplayError("schedule must be ReplaySchedule")
    if not isinstance(detector_factories, Mapping) or not detector_factories:
        raise ReplayError("detector_factories must be a non-empty mapping")
    for name, factory in detector_factories.items():
        if not isinstance(name, str) or name == "":
            raise ReplayError("factory names must be non-empty strings")
        if not callable(factory):
            raise ReplayError("each factory must be callable")
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise ReplayError("observations must be a non-empty sequence")
    if len(observations) == 0:
        raise ReplayError("observations must be a non-empty sequence")
    count = len(observations)
    if len(schedule.scenario_keys) != count:
        raise ReplayError("scenario_keys length must match observations")
    if len(schedule.arrival_times) != count:
        raise ReplayError("arrival_times length must match observations")
    if schedule.onset_index > count:
        raise ReplayError("onset_index must be <= len(observations)")
    materialised: list[MonitorObservation] = []
    for item in observations:
        if not isinstance(item, MonitorObservation):
            raise ReplayError("observations must contain MonitorObservation values")
        materialised.append(item)
    return tuple(materialised)


def _apply_schedule(
    observations: Sequence[MonitorObservation],
    schedule: ReplaySchedule,
) -> tuple[tuple[_AppliedItem, ...], int]:
    delay = float(schedule.outcome_delay_seconds)
    delay_delta = timedelta(seconds=delay)
    held: list[_HeldItem] = []
    applied: list[_AppliedItem] = []

    def release_ready(arrival: datetime) -> None:
        nonlocal held
        remaining: list[_HeldItem] = []
        for item in held:
            if arrival >= item.observation.observed_at + delay_delta:
                value = float(item.observation.value)
                applied.append(
                    _AppliedItem(
                        index=item.index,
                        value=value,
                        scenario_key=item.scenario_key,
                    )
                )
            else:
                remaining.append(item)
        held = remaining

    for index, observation in enumerate(observations):
        arrival = schedule.arrival_times[index]
        scenario_key = schedule.scenario_keys[index]
        hold = (
            delay > 0.0
            and observation.signal in _DELAYED_SIGNALS
            and arrival < observation.observed_at + delay_delta
        )
        if hold:
            held.append(
                _HeldItem(
                    index=index,
                    observation=observation,
                    scenario_key=scenario_key,
                )
            )
            release_ready(arrival)
            continue
        release_ready(arrival)
        value = float(observation.value)
        applied.append(
            _AppliedItem(index=index, value=value, scenario_key=scenario_key)
        )
    return tuple(applied), len(held)


def _score_alarms(
    *,
    method: str,
    stream_hash: str,
    alarm_indices: Sequence[tuple[int, str]],
    schedule: ReplaySchedule,
    compute_seconds: float,
    observations_applied: int,
    observations_withheld: int,
) -> ReplayResult:
    onset = schedule.onset_index
    horizon_end = onset + schedule.horizon_episodes
    healthy_false_alarm_count = 0
    post_horizon_detection_count = 0
    first_detection_index: int | None = None
    subsequent_alert_count = 0
    scenario_alarm_counts: dict[str, int] = {}
    for index, scenario_key in alarm_indices:
        scenario_alarm_counts[scenario_key] = (
            scenario_alarm_counts.get(scenario_key, 0) + 1
        )
        if index < onset:
            healthy_false_alarm_count += 1
            continue
        if index >= horizon_end:
            post_horizon_detection_count += 1
            continue
        if first_detection_index is None:
            first_detection_index = index
        else:
            subsequent_alert_count += 1
    if first_detection_index is None:
        detection_delay_episodes = None
        detection_delay_hours = None
    else:
        detection_delay_episodes = first_detection_index - onset + 1
        if onset < len(schedule.arrival_times):
            seconds = (
                schedule.arrival_times[first_detection_index]
                - schedule.arrival_times[onset]
            ).total_seconds()
            detection_delay_hours = seconds / 3600.0
        else:
            detection_delay_hours = None
    return ReplayResult(
        method=method,
        stream_hash=stream_hash,
        first_detection_index=first_detection_index,
        detection_delay_episodes=detection_delay_episodes,
        detection_delay_hours=detection_delay_hours,
        subsequent_alert_count=subsequent_alert_count,
        missed_horizon=first_detection_index is None,
        healthy_false_alarm_count=healthy_false_alarm_count,
        post_horizon_detection_count=post_horizon_detection_count,
        replay_compute_seconds=compute_seconds,
        agent_execution_seconds=None,
        scenario_alarm_counts=scenario_alarm_counts,
        observations_applied=observations_applied,
        observations_withheld=observations_withheld,
    )


def replay_detectors(
    observations: Sequence[MonitorObservation],
    detector_factories: Mapping[str, Callable[[], Detector]],
    *,
    schedule: ReplaySchedule,
) -> Mapping[str, ReplayResult]:
    """Replay independent detectors on one frozen observation stream.

    Preserves caller order. Hashes the stream once before any detector
    runs. Each factory is called exactly once. Outcome-delayed signals
    stay withheld when no later arrival releases them.
    """

    materialised = _validate_inputs(observations, detector_factories, schedule)
    stream_hash = _stream_hash(materialised, schedule)
    detectors: dict[str, Detector] = {}
    for name, factory in detector_factories.items():
        detectors[name] = _validate_detector(factory())
    applied, withheld = _apply_schedule(materialised, schedule)
    results: dict[str, ReplayResult] = {}
    for name, detector in detectors.items():
        alarm_indices: list[tuple[int, str]] = []
        started = time.perf_counter()
        for item in applied:
            value = float(item.value)
            evidence = detector.update(value)
            if evidence.alarm:
                alarm_indices.append((item.index, item.scenario_key))
        compute_seconds = time.perf_counter() - started
        results[name] = _score_alarms(
            method=name,
            stream_hash=stream_hash,
            alarm_indices=alarm_indices,
            schedule=schedule,
            compute_seconds=compute_seconds,
            observations_applied=len(applied),
            observations_withheld=withheld,
        )
    return results


def _baseline_for_signal(
    reference: FrozenReference,
    signal: str,
) -> float:
    matches = [value for name, value in reference.baselines if name == signal]
    if len(matches) == 0:
        raise ReplayError("signal is absent from reference.baselines")
    if len(matches) > 1:
        raise ReplayError("signal is duplicated in reference.baselines")
    value = matches[0]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayError("baseline estimate must be a finite float")
    number = float(value)
    if not math.isfinite(number):
        raise ReplayError("baseline estimate must be a finite float")
    return number


def _finite_alpha(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayError("alpha must be a finite float in (0, 1)")
    number = float(value)
    if not math.isfinite(number) or not 0.0 < number < 1.0:
        raise ReplayError("alpha must be a finite float in (0, 1)")
    return number


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ReplayError(f"{name} must be an integer >= 1")
    return value


def _finite_sample(values: Sequence[float], name: str) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ReplayError(f"{name} must be a sequence of finite floats")
    converted: list[float] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ReplayError(f"{name} must be a sequence of finite floats")
        number = float(item)
        if not math.isfinite(number):
            raise ReplayError(f"{name} must be a sequence of finite floats")
        converted.append(number)
    return tuple(converted)


def monitoring_detector_factories(
    settings: MonitorSettings,
    reference: FrozenReference,
    *,
    signal: str,
    alpha: float,
    window_episodes: int,
    reference_sample: Sequence[float],
    harm_margin: float,
    corrections: tuple[str, ...],
) -> dict[str, Callable[[], Detector]]:
    """Build fresh detector factories for one monitor signal.

    Scalar sequential detectors and fixed-window adapters are built
    through ``lifecycle.detectors``.
    """

    if not isinstance(settings, MonitorSettings):
        raise ReplayError("settings must be MonitorSettings")
    if not isinstance(reference, FrozenReference):
        raise ReplayError("reference must be FrozenReference")
    if reference.configuration_hash != settings.reference_configuration_hash:
        raise ReplayError(
            "reference.configuration_hash must equal settings.reference_configuration_hash"
        )
    if not isinstance(signal, str) or signal == "":
        raise ReplayError("signal must be a non-empty string")
    if signal not in settings.signals:
        raise ReplayError("signal must be present in settings.signals")
    alpha_value = _finite_alpha(alpha)
    window_value = _positive_int(window_episodes, "window_episodes")
    sample = _finite_sample(reference_sample, "reference_sample")
    if isinstance(harm_margin, bool) or not isinstance(harm_margin, (int, float)):
        raise ReplayError("harm_margin must be a finite float >= 0")
    margin = float(harm_margin)
    if not math.isfinite(margin) or margin < 0.0:
        raise ReplayError("harm_margin must be a finite float >= 0")
    if not isinstance(corrections, tuple):
        raise ReplayError("corrections must be a tuple")
    for item in corrections:
        if item not in CORRECTIONS:
            raise ReplayError("corrections may contain only none and/or bonferroni")
    baseline = _baseline_for_signal(reference, signal)
    factories: dict[str, Callable[[], Detector]] = {}
    seen_rules: set[str] = set()
    for rule in settings.stopping_rules:
        if rule.name not in MONITOR_RULE_NAMES:
            continue
        if rule.name in seen_rules:
            raise ReplayError(f"duplicate stopping rule name: {rule.name}")
        seen_rules.add(rule.name)
        rule_copy = rule

        def _factory(
            bound_rule: StoppingRule = rule_copy,
            bound_signal: str = signal,
            bound_baseline: float = baseline,
        ) -> Detector:
            try:
                return build_detector(
                    bound_rule,
                    signal=bound_signal,
                    baseline=bound_baseline,
                )
            except DetectorConstructionError as error:
                raise ReplayError(str(error)) from error

        factories[rule.name] = _factory
    unique_corrections = tuple(dict.fromkeys(corrections))
    for correction in unique_corrections:
        if correction == "none":
            correction_token: str = "none"
        else:
            correction_token = "bonferroni"
        for kind, prefix in (
            ("ks", "ks_hourly"),
            ("chi_square", "chi_square_hourly"),
        ):
            name = f"{prefix}_{correction_token}"

            def _window_factory(
                bound_kind: str = kind,
                bound_correction: str = correction_token,
                bound_alpha: float = alpha_value,
                bound_window: int = window_value,
                bound_sample: tuple[float, ...] = sample,
                bound_baseline: float = baseline,
                bound_signal: str = signal,
            ) -> Detector:
                try:
                    if bound_kind == "ks":
                        return build_hourly_window_detector(
                            kind="ks",
                            correction=(
                                "none"
                                if bound_correction == "none"
                                else "bonferroni"
                            ),
                            alpha=bound_alpha,
                            window_episodes=bound_window,
                            reference_sample=bound_sample,
                            baseline=bound_baseline,
                            signal=bound_signal,
                        )
                    return build_hourly_window_detector(
                        kind="chi_square",
                        correction=(
                            "none"
                            if bound_correction == "none"
                            else "bonferroni"
                        ),
                        alpha=bound_alpha,
                        window_episodes=bound_window,
                        reference_sample=bound_sample,
                        baseline=bound_baseline,
                        signal=bound_signal,
                    )
                except DetectorConstructionError as error:
                    raise ReplayError(str(error)) from error

            factories[name] = _window_factory

    def _harmful_factory(
        bound_alpha: float = alpha_value,
        bound_margin: float = margin,
    ) -> Detector:
        try:
            return build_harmful_shift_adapter(
                alpha=bound_alpha,
                harm_margin=bound_margin,
            )
        except DetectorConstructionError as error:
            raise ReplayError(str(error)) from error

    factories["harmful_shift"] = _harmful_factory
    return factories


def distributional_detector_factories(
    settings: DistributionalMonitorSettings,
) -> dict[str, Callable[[], Detector]]:
    """Build a fresh distributional detector factory from shared settings.

    Shares ``lifecycle.detectors.build_distributional_detector`` with
    ``lifecycle.monitoring.DistributionalMonitor``, the same way
    ``monitoring_detector_factories`` shares scalar detector construction
    with ``ProductionMonitor``: a replayed ``tool_selection`` or
    ``task_mix`` stream is scored by the identical construction a live
    monitor would have used for the same ``DistributionalMonitorSettings``.
    """

    if not isinstance(settings, DistributionalMonitorSettings):
        raise ReplayError("settings must be DistributionalMonitorSettings")

    def _factory(
        bound_settings: DistributionalMonitorSettings = settings,
    ) -> Detector:
        try:
            return build_distributional_detector(
                signal=bound_settings.signal,
                reference_counts=dict(bound_settings.reference_counts),
                window_episodes=bound_settings.window_episodes,
                alpha=bound_settings.alpha,
                correction=bound_settings.correction,
            )
        except DetectorConstructionError as error:
            raise ReplayError(str(error)) from error

    name = (
        "chi_square_hourly"
        if settings.correction == "none"
        else "chi_square_hourly_bonferroni"
    )
    return {name: _factory}


def _distributional_stream_hash(
    counts: Sequence[Mapping[str, int]],
    schedule: ReplaySchedule,
) -> str:
    rows: list[list[object]] = []
    for index, item in enumerate(counts):
        rows.append(
            [
                sorted(item.items()),
                schedule.arrival_times[index].isoformat(),
                schedule.scenario_keys[index],
                schedule.onset_index,
                schedule.horizon_episodes,
            ]
        )
    return hashlib.sha256(_canonical_json(rows)).hexdigest()


def replay_distributional_detectors(
    counts: Sequence[Mapping[str, int]],
    detector_factories: Mapping[str, Callable[[], Detector]],
    *,
    schedule: ReplaySchedule,
) -> Mapping[str, ReplayResult]:
    """Replay independent distributional detectors on one frozen counts stream.

    ``counts`` is one named-category count mapping per episode (a
    ``tool_selection`` episode's ``{name: count}``, or a ``task_mix``
    episode's single-label ``{label: 1}``), in caller order. Distribution-
    valued behavior signals are never outcome-delayed the way
    ``task_success``/``requirement_fraction`` are, so every item applies as
    soon as it arrives; detection-delay and false-alarm scoring reuse the
    same ``_score_alarms`` scalar replay uses.
    """

    if not isinstance(schedule, ReplaySchedule):
        raise ReplayError("schedule must be ReplaySchedule")
    if not isinstance(detector_factories, Mapping) or not detector_factories:
        raise ReplayError("detector_factories must be a non-empty mapping")
    for name, factory in detector_factories.items():
        if not isinstance(name, str) or name == "":
            raise ReplayError("factory names must be non-empty strings")
        if not callable(factory):
            raise ReplayError("each factory must be callable")
    if not isinstance(counts, Sequence) or isinstance(counts, (str, bytes)):
        raise ReplayError("counts must be a non-empty sequence")
    if len(counts) == 0:
        raise ReplayError("counts must be a non-empty sequence")
    total = len(counts)
    if len(schedule.scenario_keys) != total:
        raise ReplayError("scenario_keys length must match counts")
    if len(schedule.arrival_times) != total:
        raise ReplayError("arrival_times length must match counts")
    if schedule.onset_index > total:
        raise ReplayError("onset_index must be <= len(counts)")
    materialised: list[dict[str, int]] = []
    for item in counts:
        if not isinstance(item, Mapping):
            raise ReplayError("counts entries must be mappings of name to count")
        row: dict[str, int] = {}
        for name, value in item.items():
            if not isinstance(name, str) or name == "":
                raise ReplayError("counts entries must name non-empty categories")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ReplayError("counts entries must be non-negative integers")
            row[name] = value
        materialised.append(row)
    stream_hash = _distributional_stream_hash(materialised, schedule)
    results: dict[str, ReplayResult] = {}
    for name, factory in detector_factories.items():
        detector = _validate_detector(factory())
        alarm_indices: list[tuple[int, str]] = []
        started = time.perf_counter()
        for index, item in enumerate(materialised):
            evidence = detector.update(item)
            if evidence.alarm:
                alarm_indices.append((index, schedule.scenario_keys[index]))
        compute_seconds = time.perf_counter() - started
        results[name] = _score_alarms(
            method=name,
            stream_hash=stream_hash,
            alarm_indices=alarm_indices,
            schedule=schedule,
            compute_seconds=compute_seconds,
            observations_applied=total,
            observations_withheld=0,
        )
    return results


def replay_result_public_dict(result: ReplayResult) -> dict[str, Any]:
    """Serialize one replay result without protected task or episode fields."""

    return {
        "method": result.method,
        "stream_hash": result.stream_hash,
        "first_detection_index": result.first_detection_index,
        "detection_delay_episodes": result.detection_delay_episodes,
        "detection_delay_hours": result.detection_delay_hours,
        "subsequent_alert_count": result.subsequent_alert_count,
        "missed_horizon": result.missed_horizon,
        "healthy_false_alarm_count": result.healthy_false_alarm_count,
        "post_horizon_detection_count": result.post_horizon_detection_count,
        "replay_compute_seconds": result.replay_compute_seconds,
        "agent_execution_seconds": result.agent_execution_seconds,
        "scenario_alarm_counts": dict(result.scenario_alarm_counts),
        "observations_applied": result.observations_applied,
        "observations_withheld": result.observations_withheld,
    }
