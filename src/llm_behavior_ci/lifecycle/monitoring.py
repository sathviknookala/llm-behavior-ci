from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Mapping

from llm_behavior_ci.config import (
    DISTRIBUTIONAL_SIGNALS,
    MONITOR_SIGNALS,
    DistributionalMonitorSettings,
    MonitorSettings,
)
from llm_behavior_ci.lifecycle.detectors import (
    BOUNDED_SIGNALS,
    DetectorConstructionError,
    build_detectors,
    build_distributional_detector,
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
    MonitorMetadataRecord,
)

_DELAYED_SIGNALS = BOUNDED_SIGNALS
_DISTRIBUTIONAL_SIGNALS = DISTRIBUTIONAL_SIGNALS
PLAN_QUALITY_SIGNAL = "plan_quality_score"
PLAN_KL_SIGNAL = "plan_kl_mean_nats"


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

    def to_record(self, episode_id: str) -> MonitorMetadataRecord:
        """Serialize this metadata so a stored episode's inputs can be rebuilt.

        Paired with the persisted ``EpisodeResult`` (which already carries
        the evaluator outcome, tool steps, and model steps), this record is
        everything ``normalize_episode``/``observation_from_episode`` and
        the typed distributional builders need to reconstruct the exact
        detector input a live update once used.
        """

        return MonitorMetadataRecord(
            episode_id=episode_id,
            signal=self.signal,
            completion_index=self.completion_index,
            difficulty=self.difficulty,
            task_mix=self.task_mix,
            slice_id=self.slice_id,
        )

    @classmethod
    def from_record(cls, record: MonitorMetadataRecord) -> TaskMetadata:
        return cls(
            signal=record.signal,
            completion_index=record.completion_index,
            difficulty=record.difficulty,
            task_mix=record.task_mix,
            slice_id=record.slice_id,
        )


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
    difficulty: int | None


class MissingSliceReference(MonitorRejected):
    """Raised when a per-slice reference is required but the slice has none."""


@dataclass(frozen=True)
class FrozenReference:
    """Caller-supplied frozen baselines for the previous known-good config.

    ``slice_baselines`` maps a slice name to that slice's own baselines.
    When it is empty, a slice detector starts from the aggregate
    ``baselines`` (the aggregate-reference approximation). When it is
    non-empty, ``source`` must name where the baselines were measured, and
    a slice that has no entry raises ``MissingSliceReference`` rather than
    falling back to the aggregate.
    """

    configuration_hash: str
    baselines: tuple[tuple[str, float], ...]
    slice_baselines: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()
    source: str | None = None

    def __post_init__(self) -> None:
        names = [name for name, _values in self.slice_baselines]
        if len(set(names)) != len(names):
            raise MonitorRejected("slice_baselines must name each slice once")
        if any(not isinstance(name, str) or name == "" for name in names):
            raise MonitorRejected("slice_baselines names must be non-empty strings")
        if self.slice_baselines and (
            not isinstance(self.source, str) or self.source == ""
        ):
            raise MonitorRejected("per-slice baselines require a non-empty source")

    @property
    def per_slice(self) -> bool:
        return bool(self.slice_baselines)

    def baselines_for_slice(self, slice_name: str) -> tuple[tuple[str, float], ...]:
        if not self.per_slice:
            return self.baselines
        for name, values in self.slice_baselines:
            if name == slice_name:
                return values
        raise MissingSliceReference(
            f"frozen reference from {self.source} has no baseline for slice {slice_name}"
        )


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
    period_id: str | None = None
    attributed_slices: tuple[str, ...] = ()

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
        if self.attributed_slices:
            payload["attributed_slices"] = list(self.attributed_slices)
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
            period_id=self.period_id,
            attributed_slices=self.attributed_slices,
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
            period_id=record.period_id,
            attributed_slices=record.attributed_slices,
        )


class LocalAlertSink:
    """In-process alert delivery without external notification accounts.

    ``deliver`` returns only the alerts it newly delivered. An alert whose
    incident (``AlertRecord.incident_key``: period, configuration, signal,
    slice) was already delivered is suppressed: with a store the check reads
    SQLite, so it holds across a restart; without one it is kept in memory.
    An alert with no period falls back to the ``dedup_seconds`` window.
    """

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

    def _seen_in_memory(self, alert: Alert) -> bool:
        record = alert.to_record()
        for previous in self._delivered:
            earlier = previous.to_record()
            if record.incident_key is not None:
                if earlier.incident_key == record.incident_key:
                    return True
                continue
            if (
                self._dedup_seconds > 0.0
                and earlier.incident_key is None
                and (earlier.signal, earlier.slice_name, earlier.method)
                == (record.signal, record.slice_name, record.method)
                and 0.0
                <= (record.raised_at - earlier.raised_at).total_seconds()
                < self._dedup_seconds
            ):
                return True
        return False

    def deliver(self, alerts: Sequence[Alert]) -> tuple[Alert, ...]:
        if not isinstance(alerts, Sequence) or isinstance(alerts, (str, bytes)):
            raise MonitorRejected("alerts must be a sequence of Alert")
        delivered: list[Alert] = []
        for alert in alerts:
            if not isinstance(alert, Alert):
                raise MonitorRejected("alerts must be a sequence of Alert")
            if self._store is None:
                if self._seen_in_memory(alert):
                    continue
                self._delivered.append(alert)
                delivered.append(alert)
                continue
            stored, inserted = self._store.append_alert_with_status(
                alert.to_record(),
                dedup_seconds=self._dedup_seconds,
            )
            if not inserted:
                continue
            chosen = Alert.from_record(stored)
            self._store.append_deployment_decision(
                DeploymentDecisionRecord(
                    configuration_hash=chosen.configuration_hash,
                    reference_configuration_hash=chosen.reference_configuration_hash,
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


def _resolved_difficulty(
    episode: EpisodeResult,
    task_metadata: TaskMetadata,
) -> int | None:
    """Prefer the evaluator's own difficulty over a caller-supplied label.

    ``task_metadata.difficulty`` is a launcher-side hint, not a claim about
    execution. When AppWorld's evaluator has already recorded a difficulty
    for this episode, that is the value monitoring uses and slices by; a
    caller label that disagrees is rejected rather than silently
    overriding actual evaluator-derived task metadata with a launcher-
    supplied synthetic value.
    """

    evaluator_difficulty = (
        None
        if episode.evaluator_outcome is None
        else episode.evaluator_outcome.difficulty
    )
    if (
        evaluator_difficulty is not None
        and task_metadata.difficulty is not None
        and task_metadata.difficulty != evaluator_difficulty
    ):
        raise MonitorRejected(
            "task_metadata.difficulty must match the episode's evaluator difficulty"
        )
    return (
        evaluator_difficulty
        if evaluator_difficulty is not None
        else task_metadata.difficulty
    )


def resolved_slice_name(
    episode: EpisodeResult,
    task_metadata: TaskMetadata,
    *,
    use_slice_attribution: bool,
) -> str | None:
    """The slice label a monitor should attribute this episode's signal to.

    Prefers an explicit ``task_metadata.slice_id``. Otherwise, when slice
    attribution is enabled, falls back to the evaluator-derived difficulty
    (``f"difficulty:{difficulty}"``): the only per-task cluster the split
    policy releases on ``test_normal`` (``docs/DATA.md``), so it is the
    project's default defined cluster rather than a ground-truth app label.
    Returns ``None`` when attribution is off or neither label is available,
    leaving the caller at aggregate-only attribution.
    """

    if not use_slice_attribution:
        return None
    if task_metadata.slice_id is not None:
        return task_metadata.slice_id
    difficulty = _resolved_difficulty(episode, task_metadata)
    if difficulty is not None:
        return f"difficulty:{difficulty}"
    return None


def normalize_episode(
    episode: EpisodeResult,
    *,
    task_metadata: TaskMetadata,
) -> NormalizedEpisode:
    """Normalize episode fields for monitoring without substituting missing outcomes."""

    difficulty = _resolved_difficulty(episode, task_metadata)
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
        difficulty=difficulty,
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


def plan_quality_observation_from_features(
    episode: EpisodeResult,
    *,
    features: Mapping[str, float],
) -> MonitorObservation:
    """Fold one semantic plan-feature vector into a bounded monitor observation.

    ``features`` is whatever ``lifecycle.plan_features.semantic_plan_features``
    computed for this episode's plan text against its ``TaskPlanSpec``: an
    offline, pre-execution representation. This adapter reads only
    ``requirement_coverage_fraction``, which is already bounded in [0, 1]
    and defined even for a task with no declared subgoals (vacuous 1.0), so
    the fold needs no extra caller weighting to stay a valid ``plan_quality_
    score`` observation. Plan-quality scoring is scheduled by the caller,
    not implied by every episode: an episode with no plan representation
    simply never calls this.
    """

    if "requirement_coverage_fraction" not in features:
        raise MonitorRejected(
            "features must include requirement_coverage_fraction"
        )
    value = features["requirement_coverage_fraction"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MonitorRejected("requirement_coverage_fraction must be a float")
    return MonitorObservation(
        episode=episode.episode,
        run=episode.run,
        split=episode.task.split,
        signal=PLAN_QUALITY_SIGNAL,
        value=float(value),
        observed_at=episode.ended_at,
    )


def plan_kl_observation(
    episode: EpisodeResult,
    *,
    mean_kl_nats: float,
) -> MonitorObservation:
    """Fold one teacher-forced plan-KL result into a monitor observation.

    Takes the bare ``mean_kl_nats`` float rather than a KL result type
    directly: ``stats.kl`` and ``runtime.scoring`` are a concurrently
    developed contract that may still change shape, so the coupling here
    stays to one scalar a caller reads off ``NextTokenKLResult.mean_kl_nats``
    or ``TruncatedKLResult.mean_kl_nats`` (or an equivalent
    ``DistributionScore.mean_kl_nats``) and passes explicitly. Teacher-forced
    scoring is scheduled by the caller alongside production traffic, not run
    on every episode.
    """

    if isinstance(mean_kl_nats, bool) or not isinstance(mean_kl_nats, (int, float)):
        raise MonitorRejected("mean_kl_nats must be a finite nonnegative float")
    value = float(mean_kl_nats)
    if not math.isfinite(value) or value < 0.0:
        raise MonitorRejected("mean_kl_nats must be a finite nonnegative float")
    return MonitorObservation(
        episode=episode.episode,
        run=episode.run,
        split=episode.task.split,
        signal=PLAN_KL_SIGNAL,
        value=value,
        observed_at=episode.ended_at,
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

    Alert policy: the aggregate detector chain of a signal opens one
    incident per monitoring period, and only that raises an ``Alert``.
    Slice chains run beside it for attribution only: the alert names every
    slice whose chain has alarmed in the period so far, and a slice alarm
    alone raises nothing. ``period_id`` is explicit and required; an update
    that names a different period is rejected. Within a period a signal
    alerts at most once; ``restore_open_incidents`` reloads the incidents a
    previous process already raised, and ``reset_for_promotion`` with a new
    period starts fresh incidents.
    """

    def __init__(
        self,
        settings: MonitorSettings,
        reference: FrozenReference,
        *,
        clock: Callable[[], datetime],
        period_id: str,
        use_slice_attribution: bool = False,
    ) -> None:
        if not isinstance(settings, MonitorSettings):
            raise MonitorRejected("settings must be MonitorSettings")
        if not callable(clock):
            raise MonitorRejected("clock must be callable")
        if not isinstance(period_id, str) or period_id == "":
            raise MonitorRejected("period_id must be a non-empty string")
        if not isinstance(use_slice_attribution, bool):
            raise MonitorRejected("use_slice_attribution must be a boolean")
        self._settings = settings
        self._clock = clock
        self._period_id = period_id
        self._use_slice_attribution = use_slice_attribution
        self._signals = frozenset(settings.signals)
        self._outcome_delay = timedelta(seconds=settings.outcome_delay_seconds)
        self._held: list[_HeldObservation] = []
        self._seen: set[tuple[str, str]] = set()
        self._open_incidents: dict[str, Alert] = {}
        self._alarmed_slices: dict[str, set[str]] = {}
        self._slice_detectors: dict[tuple[str, str], list[Detector]] = {}
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
    def period_id(self) -> str:
        return self._period_id

    @property
    def open_incidents(self) -> tuple[Alert, ...]:
        return tuple(self._open_incidents[signal] for signal in sorted(self._open_incidents))

    def restore_open_incidents(self, alerts: Sequence[Alert | AlertRecord]) -> int:
        """Mark this period's already-raised aggregate incidents as open.

        A restarted process reads its store's alerts and passes them here, so
        a signal whose incident was raised before the restart does not alert
        again in the same period. Alerts from other periods, other reference
        configurations, or slice-named records are ignored. Returns how many
        incidents were restored.
        """

        restored = 0
        for item in alerts:
            alert = Alert.from_record(item) if isinstance(item, AlertRecord) else item
            if not isinstance(alert, Alert):
                raise MonitorRejected("alerts must be Alert or AlertRecord")
            if (
                alert.period_id != self._period_id
                or alert.reference_configuration_hash != self._reference_configuration_hash
                or alert.signal not in self._signals
                or alert.slice_name != alert.signal
                or alert.signal in self._open_incidents
            ):
                continue
            self._open_incidents[alert.signal] = alert
            restored += 1
        return restored

    @property
    def signals(self) -> tuple[str, ...]:
        """The configured scalar series, in the order ``MonitorSettings`` names them."""

        return self._settings.signals

    @property
    def use_slice_attribution(self) -> bool:
        return self._use_slice_attribution

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
        slice_name = resolved_slice_name(
            episode,
            task_metadata,
            use_slice_attribution=self._use_slice_attribution,
        )
        return self.update(
            observation,
            completion_index=task_metadata.completion_index,
            period_id=period_id,
            slice_name=slice_name,
        )

    def reset_for_promotion(
        self,
        reference: FrozenReference,
        *,
        period_id: str,
    ) -> None:
        """Rebuild every detector against ``reference`` and start a new period.

        ``period_id`` must differ from the current one, so incidents from
        before the promotion cannot suppress alerts after it.
        """

        if not isinstance(reference, FrozenReference):
            raise MonitorRejected("reference must be FrozenReference")
        if not isinstance(period_id, str) or period_id == "":
            raise MonitorRejected("period_id must be a non-empty string")
        if period_id == self._period_id:
            raise MonitorRejected("a promotion must start a new monitoring period")
        self._period_id = period_id
        self._reference = reference
        self._reference_configuration_hash = reference.configuration_hash
        self._detectors = self._detectors_from(reference)
        self._held.clear()
        self._seen.clear()
        self._open_incidents.clear()
        self._alarmed_slices.clear()
        self._slice_detectors.clear()

    def release_due(self, *, period_id: str | None = None) -> tuple[Alert, ...]:
        """Apply every held delayed observation whose outcome delay has passed.

        A scheduled stream calls this after advancing its simulated clock, so
        outcomes become visible to the detectors exactly when the schedule
        says they arrive, even when no new observation follows.
        """

        self._check_period(period_id)
        return tuple(self._release_ready(self._clock()))

    @property
    def held_count(self) -> int:
        return len(self._held)

    @property
    def outcome_delay_seconds(self) -> float:
        return float(self._settings.outcome_delay_seconds)

    def _check_period(self, period_id: str | None) -> None:
        if period_id is not None and period_id != self._period_id:
            raise MonitorRejected("period_id does not match the monitoring period")

    def _detectors_from(
        self,
        reference: FrozenReference,
        baseline_pairs: tuple[tuple[str, float], ...] | None = None,
    ) -> dict[str, list[Detector]]:
        baselines = _baselines_for_signals(
            reference.baselines if baseline_pairs is None else baseline_pairs,
            self._settings.signals,
        )
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
        """Run the aggregate chain and, when named, the slice chain for attribution.

        The aggregate chain (``self._detectors[signal]``) sees every episode
        for that signal. A caller-resolved ``slice_name`` other than the bare
        signal also feeds an independent chain scoped to that slice, built
        from the same reference. Every detector updates on every observation.
        A slice alarm only records the slice as alarmed for the period. The
        first aggregate alarm in the period opens the signal's incident and
        returns its one alert, naming the alarmed slices; later alarms in the
        period return nothing.
        """

        evidence = self._first_alarm(self._detectors[observation.signal], observation.value)
        if slice_name != observation.signal:
            key = (observation.signal, slice_name)
            if key not in self._slice_detectors:
                self._slice_detectors[key] = self._detectors_from(
                    self._reference,
                    self._reference.baselines_for_slice(slice_name),
                )[observation.signal]
            if self._first_alarm(self._slice_detectors[key], observation.value) is not None:
                self._alarmed_slices.setdefault(observation.signal, set()).add(slice_name)
        if evidence is None or observation.signal in self._open_incidents:
            return []
        alert = Alert(
            configuration_hash=observation.run.configuration_hash,
            reference_configuration_hash=self._reference_configuration_hash,
            signal=observation.signal,
            slice_name=observation.signal,
            method=evidence.method,
            estimate=evidence.estimate,
            boundary=evidence.boundary,
            sample_size=evidence.sample_size,
            raised_at=self._clock(),
            period_id=self._period_id,
            attributed_slices=tuple(
                sorted(self._alarmed_slices.get(observation.signal, ()))
            ),
        )
        self._open_incidents[observation.signal] = alert
        return [alert]

    @staticmethod
    def _first_alarm(detectors: list[Detector], value: float) -> Evidence | None:
        first: Evidence | None = None
        for detector in detectors:
            evidence = detector.update(value)
            if evidence.alarm and first is None:
                first = evidence
        return first


DistributionalObservation = ToolSelectionObservation | TaskMixObservation


def _counts_from_observation(
    observation: DistributionalObservation,
) -> dict[str, int]:
    if isinstance(observation, ToolSelectionObservation):
        return {item.name: item.count for item in observation.counts}
    return {observation.label: 1}


class DistributionalMonitor:
    """Windowed categorical drift monitor for ``tool_selection`` or ``task_mix``.

    Distribution-valued observations are never forced through a scalar
    ``Detector.update(float)`` interface: this monitor routes them to
    ``lifecycle.detectors.build_distributional_detector``, the same
    canonical construction ``experiments.replay.distributional_detector_
    factories`` uses, so monitoring and replay never diverge on how a
    ``tool_selection`` or ``task_mix`` detector is built. ``task_mix`` is
    monitored independently of every behavior signal, so a shift in the
    incoming task composition is attributed as input drift rather than
    misread as a change in model behavior.
    """

    def __init__(
        self,
        settings: DistributionalMonitorSettings,
        *,
        reference_configuration_hash: str,
        clock: Callable[[], datetime],
        dedup_seconds: float,
        period_id: str | None = None,
    ) -> None:
        if not isinstance(settings, DistributionalMonitorSettings):
            raise MonitorRejected(
                "settings must be DistributionalMonitorSettings"
            )
        if not isinstance(reference_configuration_hash, str) or not reference_configuration_hash:
            raise MonitorRejected(
                "reference_configuration_hash must be a non-empty string"
            )
        if not callable(clock):
            raise MonitorRejected("clock must be callable")
        if isinstance(dedup_seconds, bool) or not isinstance(
            dedup_seconds, (int, float)
        ):
            raise MonitorRejected("dedup_seconds must be a finite float >= 0")
        window = float(dedup_seconds)
        if not math.isfinite(window) or window < 0.0:
            raise MonitorRejected("dedup_seconds must be a finite float >= 0")
        self._settings = settings
        self._reference_configuration_hash = reference_configuration_hash
        self._clock = clock
        self._dedup_seconds = window
        self._detector = build_distributional_detector(
            signal=settings.signal,
            reference_counts=dict(settings.reference_counts),
            window_episodes=settings.window_episodes,
            alpha=settings.alpha,
            correction=settings.correction,
        )
        self._slice_detectors: dict[str, object] = {}
        self._last_alert_at: dict[str, datetime] = {}
        if period_id is not None and (not isinstance(period_id, str) or period_id == ""):
            raise MonitorRejected("period_id must be a non-empty string when set")
        self._period_id = period_id

    @property
    def signal(self) -> str:
        return self._settings.signal

    def update(
        self,
        observation: DistributionalObservation,
        *,
        slice_name: str | None = None,
    ) -> tuple[Alert, ...]:
        expected = ToolSelectionObservation if self.signal == "tool_selection" else TaskMixObservation
        if not isinstance(observation, expected):
            raise MonitorRejected(f"{self.signal} monitor requires {expected.__name__}")
        counts = _counts_from_observation(observation)
        resolved_slice = self.signal if slice_name is None else slice_name
        alerts: list[Alert] = []
        alerts.extend(
            self._apply(self._detector, counts, observation, slice_name=self.signal)
        )
        if resolved_slice != self.signal:
            detector = self._slice_detectors.get(resolved_slice)
            if detector is None:
                detector = build_distributional_detector(
                    signal=self._settings.signal,
                    reference_counts=self._reference_counts_for(resolved_slice),
                    window_episodes=self._settings.window_episodes,
                    alpha=self._settings.alpha,
                    correction=self._settings.correction,
                )
                self._slice_detectors[resolved_slice] = detector
            alerts.extend(
                self._apply(detector, counts, observation, slice_name=resolved_slice)
            )
        return tuple(alerts)

    @property
    def period_id(self) -> str | None:
        return self._period_id

    def _reference_counts_for(self, slice_name: str) -> dict[str, int]:
        if not self._settings.slice_reference_counts:
            return dict(self._settings.reference_counts)
        for name, counts in self._settings.slice_reference_counts:
            if name == slice_name:
                return dict(counts)
        raise MissingSliceReference(
            f"{self.signal} reference from {self._settings.reference_source} "
            f"has no distribution for slice {slice_name}"
        )

    def _apply(
        self,
        detector: object,
        counts: Mapping[str, int],
        observation: DistributionalObservation,
        *,
        slice_name: str,
    ) -> list[Alert]:
        evidence = detector.update(counts)
        if not evidence.alarm:
            return []
        raised_at = self._clock()
        dedup_key = f"{self.signal}:{slice_name}"
        last = self._last_alert_at.get(dedup_key)
        if (
            self._dedup_seconds > 0.0
            and last is not None
            and (raised_at - last).total_seconds() < self._dedup_seconds
        ):
            return []
        self._last_alert_at[dedup_key] = raised_at
        return [
            Alert(
                configuration_hash=observation.run.configuration_hash,
                reference_configuration_hash=self._reference_configuration_hash,
                signal=self.signal,
                slice_name=slice_name,
                method=evidence.method,
                estimate=evidence.estimate,
                boundary=evidence.boundary,
                sample_size=evidence.sample_size,
                raised_at=raised_at,
                period_id=self._period_id,
            )
        ]

    def reset(
        self,
        *,
        reference_configuration_hash: str | None = None,
        period_id: str | None = None,
    ) -> None:
        """Start a fresh window; on promotion also rebind reference and period."""

        if reference_configuration_hash is not None:
            if reference_configuration_hash == "":
                raise MonitorRejected("reference_configuration_hash must be non-empty")
            self._reference_configuration_hash = reference_configuration_hash
        if period_id is not None:
            if period_id == "":
                raise MonitorRejected("period_id must be non-empty when set")
            self._period_id = period_id
        self._detector.reset()
        self._slice_detectors.clear()
        self._last_alert_at.clear()


def build_distributional_monitors(
    settings: tuple[DistributionalMonitorSettings, ...],
    *,
    reference_configuration_hash: str,
    clock: Callable[[], datetime],
    dedup_seconds: float,
    period_id: str | None = None,
) -> dict[str, DistributionalMonitor]:
    """Build one ``DistributionalMonitor`` per configured distributional signal.

    Keyed by ``signal`` (``tool_selection`` or ``task_mix``) so a caller
    wires ``monitors.get("tool_selection")``/``monitors.get("task_mix")``
    straight into ``ServiceDependencies.tool_selection_monitor``/
    ``task_mix_monitor`` or an equivalent benchmark or replay binding,
    the same shared construction every ``DistributionalMonitor`` uses.
    """

    if not isinstance(settings, tuple):
        raise MonitorRejected(
            "settings must be a tuple of DistributionalMonitorSettings"
        )
    monitors: dict[str, DistributionalMonitor] = {}
    for entry in settings:
        if not isinstance(entry, DistributionalMonitorSettings):
            raise MonitorRejected(
                "settings must contain DistributionalMonitorSettings"
            )
        if entry.signal in monitors:
            raise MonitorRejected(
                "settings contains a duplicate distributional signal"
            )
        monitors[entry.signal] = DistributionalMonitor(
            entry,
            reference_configuration_hash=reference_configuration_hash,
            clock=clock,
            dedup_seconds=dedup_seconds,
            period_id=period_id,
        )
    return monitors


def monitoring_period_id(serving_configuration_hash: str, reference_configuration_hash: str) -> str:
    """Deterministic monitoring period for one serving/reference pair.

    The same deployment restarted keeps its period, so stored incident keys
    still de-duplicate; a promotion changes the serving hash and opens a new
    period.
    """

    digest = hashlib.sha256(
        f"{serving_configuration_hash}:{reference_configuration_hash}".encode("utf-8")
    ).hexdigest()
    return f"period-{digest[:32]}"


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
