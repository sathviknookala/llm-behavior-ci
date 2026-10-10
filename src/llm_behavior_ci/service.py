"""HTTP service for plan/execute episodes, canary admission, and deployment status."""

from __future__ import annotations

import asyncio
import math
import re
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Callable, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from llm_behavior_ci.config import (
    CanarySettings,
    ConfigError,
    EpisodeIdentity,
    RunConfiguration,
    RunIdentity,
    new_run_identity,
    run_configuration_hash,
)
from llm_behavior_ci.experiments.protocol import (
    ProtocolError,
    TaskSelectionAllowance,
    authorize_gated_candidate,
    authorize_test_gated_candidate,
)
from llm_behavior_ci.lifecycle.canary import (
    CanaryController,
    CanaryDecision,
    CanaryRejected,
    DeploymentSnapshot,
    assign_canary,
)
from llm_behavior_ci.lifecycle.monitoring import (
    Alert,
    DistributionalMonitor,
    FrozenReference,
    LocalAlertSink,
    MissingEvaluatorOutcome,
    MonitorRejected,
    PLAN_KL_SIGNAL,
    PLAN_QUALITY_SIGNAL,
    ProductionMonitor,
    TaskMetadata,
    UndefinedRequirementFraction,
    monitoring_period_id,
    plan_kl_observation,
    plan_quality_observation_from_features,
    resolved_slice_name,
    task_mix_observation_from_episode,
    tool_selection_observation_from_episode,
)
from llm_behavior_ci.records import EpisodeResult, ModelStep, ToolStep
from llm_behavior_ci.runtime.budget import ModelCallBudget
from llm_behavior_ci.runtime.clock import monotonic
from llm_behavior_ci.runtime.episode import (
    EpisodeRejected,
    RuntimeDependencies,
    RuntimeUnavailable,
    run_episode,
    run_pair,
)
from llm_behavior_ci.runtime.factory import RuntimeFactory
from llm_behavior_ci.storage import DeploymentDecisionRecord, EpisodeStore, StorageError

_EPISODE_REQUIRED = frozenset({"task_id", "mode"})
_EPISODE_FIELDS = frozenset({"task_id", "mode", "role", "assignment_key"})
_CANDIDATE_ADMISSION_FIELDS = frozenset({"artifact_id"})
_ARTIFACT_ID = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_OVERRIDE_FIELDS = frozenset(
    {
        "model",
        "agent",
        "sampling",
        "prompt",
        "temperature",
        "max_tokens",
        "step_limit",
        "quantization",
        "repository",
        "protocol",
        "protocol_hash",
    }
)
_TERMINAL_CANARY = frozenset({"ROLLED_BACK", "PROMOTED"})
_PRODUCTION_ROLLBACK = "production_rollback"
_LIFECYCLE_DECISIONS = frozenset({"admit", "promote", "rollback", _PRODUCTION_ROLLBACK})
_UNFINISHED_AT_RESTART = "restart_with_unfinished_canary"
_ACTIVE_CANARY = frozenset({"GATE_PASSED", "CANARY_ACTIVE"})
_PRODUCTION_ROLLBACK_METHODS = frozenset({"monitor_alert", "manual_rollback"})
_SUPERSEDED = "superseded"
POST_PROMOTION_ROLLBACK_POLICIES = ("record_only", "rollback_on_alert")
ROLLBACK_SIGNAL = "task_success"
PRODUCTION_ROLLED_BACK = "PRODUCTION_ROLLED_BACK"


ModeName = Literal["plan", "execute"]


class ServiceError(ValueError):
    """Raised when service construction or request validation fails."""


@dataclass(frozen=True)
class ConfigurationRegistry:
    production: RunConfiguration
    candidate: RunConfiguration | None


@dataclass(frozen=True)
class ServiceDependencies:
    registry: ConfigurationRegistry
    runtime_factory: Callable[[RunConfiguration], RuntimeDependencies]
    store: EpisodeStore
    monitor: ProductionMonitor
    clock: Callable[[], datetime]
    max_in_flight: int
    shutdown_timeout_seconds: float
    metadata_for: Callable[[EpisodeResult], TaskMetadata]
    canary_settings: CanarySettings
    canary_assignment_seed: int
    task_selection_allowance: TaskSelectionAllowance | None = None
    tool_selection_monitor: DistributionalMonitor | None = None
    task_mix_monitor: DistributionalMonitor | None = None
    plan_quality_features_for: (
        Callable[[EpisodeResult], dict[str, float] | None] | None
    ) = None
    plan_kl_mean_nats_for: Callable[[EpisodeResult], float | None] | None = None
    admission_mode: Literal["release", "test"] = "release"
    alert_dedup_seconds: float = 0.0
    post_promotion_rollback: Literal["record_only", "rollback_on_alert"] = "record_only"
    model_call_budget: ModelCallBudget | None = None

    def __post_init__(self) -> None:
        try:
            if not isinstance(self.registry, ConfigurationRegistry):
                raise ValueError("registry must be a ConfigurationRegistry")
            if not isinstance(self.registry.production, RunConfiguration):
                raise ValueError("registry.production must be a RunConfiguration")
            if self.registry.candidate is not None and not isinstance(
                self.registry.candidate, RunConfiguration
            ):
                raise ValueError(
                    "registry.candidate must be a RunConfiguration or None"
                )
            if not callable(self.runtime_factory):
                raise ValueError("runtime_factory must be callable")
            if not isinstance(self.store, EpisodeStore):
                raise ValueError("store must be an EpisodeStore")
            if not isinstance(self.monitor, ProductionMonitor):
                raise ValueError("monitor must be a ProductionMonitor")
            if not callable(self.clock):
                raise ValueError("clock must be callable")
            if isinstance(self.max_in_flight, bool) or not isinstance(
                self.max_in_flight, int
            ):
                raise ValueError("max_in_flight must be a positive int")
            if self.max_in_flight < 1:
                raise ValueError("max_in_flight must be a positive int")
            if isinstance(self.shutdown_timeout_seconds, bool) or not isinstance(
                self.shutdown_timeout_seconds, (int, float)
            ):
                raise ValueError(
                    "shutdown_timeout_seconds must be a finite float >= 0"
                )
            timeout = float(self.shutdown_timeout_seconds)
            if not math.isfinite(timeout) or timeout < 0.0:
                raise ValueError(
                    "shutdown_timeout_seconds must be a finite float >= 0"
                )
            if not callable(self.metadata_for):
                raise ValueError("metadata_for must be callable")
            if not isinstance(self.canary_settings, CanarySettings):
                raise ValueError("canary_settings must be CanarySettings")
            if isinstance(self.canary_assignment_seed, bool) or not isinstance(
                self.canary_assignment_seed, int
            ):
                raise ValueError("canary_assignment_seed must be an integer")
            if self.task_selection_allowance is not None and not isinstance(
                self.task_selection_allowance, TaskSelectionAllowance
            ):
                raise ValueError(
                    "task_selection_allowance must be TaskSelectionAllowance or None"
                )
            if self.tool_selection_monitor is not None and not isinstance(
                self.tool_selection_monitor, DistributionalMonitor
            ):
                raise ValueError(
                    "tool_selection_monitor must be DistributionalMonitor or None"
                )
            if self.task_mix_monitor is not None and not isinstance(
                self.task_mix_monitor, DistributionalMonitor
            ):
                raise ValueError(
                    "task_mix_monitor must be DistributionalMonitor or None"
                )
            if self.plan_quality_features_for is not None and not callable(
                self.plan_quality_features_for
            ):
                raise ValueError(
                    "plan_quality_features_for must be callable or None"
                )
            if self.plan_kl_mean_nats_for is not None and not callable(
                self.plan_kl_mean_nats_for
            ):
                raise ValueError("plan_kl_mean_nats_for must be callable or None")
            if self.admission_mode not in {"release", "test"}:
                raise ValueError("admission_mode must be release or test")
            if self.model_call_budget is not None and not isinstance(
                self.model_call_budget, ModelCallBudget
            ):
                raise ValueError("model_call_budget must be ModelCallBudget or None")
            if self.post_promotion_rollback not in POST_PROMOTION_ROLLBACK_POLICIES:
                raise ValueError(
                    "post_promotion_rollback must be one of: "
                    + ", ".join(POST_PROMOTION_ROLLBACK_POLICIES)
                )
        except ValueError as error:
            raise ServiceError(error) from error


@dataclass(frozen=True)
class _EpisodeRequest:
    task_id: str
    mode: ModeName
    role: str | None
    assignment_key: str | None


@dataclass(frozen=True)
class _RecoveredDeployment:
    """The terminal deployment state a restarted service read back from its store."""

    state: Literal["PROMOTED", "ROLLED_BACK"]
    previous_production_configuration_hash: str
    promoted_configuration_hash: str | None
    rejected_candidate_configuration_hash: str | None
    method: str


@dataclass(frozen=True)
class _ProductionRollback:
    """A post-promotion rollback: the promoted configuration stopped serving.

    ``source`` is ``production_monitor`` (an eligible alert in this
    process), ``operator`` (``POST /deployment/rollback``), or
    ``restart_reconciliation`` (an eligible alert found at startup with no
    rollback after it). ``persisted`` is false only while the decision row
    could not be written; the known-good configuration serves regardless.
    """

    rolled_back_configuration_hash: str
    restored_configuration_hash: str
    rolled_back_period_id: str
    method: str
    source: str
    trigger: Alert | None
    decided_at: datetime
    persisted: bool
    recovered: bool


class _ServiceState:
    def __init__(self, dependencies: ServiceDependencies) -> None:
        self.dependencies = dependencies
        self.admission = "open"
        self.controller: CanaryController | None = None
        self.recovered: _RecoveredDeployment | None = None
        self.production_rollback: _ProductionRollback | None = None
        self.serving_configuration = dependencies.registry.production
        self.monitor_period_id = dependencies.monitor.period_id
        self.alert_sink = LocalAlertSink(
            dependencies.store,
            dedup_seconds=dependencies.alert_dedup_seconds,
        )
        self.incidents_restored = False
        self.deployment_lock = threading.Lock()
        self._slot_lock = threading.Lock()
        self._in_flight = 0
        self._slots = threading.BoundedSemaphore(dependencies.max_in_flight)
        self._store_closed = False

    def try_acquire_slot(self) -> bool:
        if not self._slots.acquire(blocking=False):
            return False
        with self._slot_lock:
            self._in_flight += 1
        return True

    def release_slot(self) -> None:
        with self._slot_lock:
            self._in_flight -= 1
        self._slots.release()

    def in_flight(self) -> int:
        with self._slot_lock:
            return self._in_flight

    def close_store_once(self) -> None:
        if self._store_closed:
            return
        self._store_closed = True
        self.dependencies.store.close()


def _message(error: BaseException) -> str:
    text = str(error)
    if text:
        return text
    return type(error).__name__


def _reject_overrides(body: dict[str, object]) -> None:
    blocked = _FORBIDDEN_OVERRIDE_FIELDS.intersection(body)
    if blocked:
        raise ServiceError(
            "request must not override model or agent fields: "
            + ", ".join(sorted(blocked))
        )


def _parse_episode_request(body: object) -> _EpisodeRequest:
    if not isinstance(body, dict):
        raise ServiceError("body must be a JSON object")
    if not all(isinstance(key, str) for key in body):
        raise ServiceError("body keys must be strings")
    _reject_overrides(body)
    extra = set(body) - _EPISODE_FIELDS
    if extra:
        raise ServiceError(
            "unexpected fields: " + ", ".join(sorted(extra))
        )
    missing = _EPISODE_REQUIRED - set(body)
    if missing:
        raise ServiceError(
            "missing fields: " + ", ".join(sorted(missing))
        )
    task_id = body["task_id"]
    mode = body["mode"]
    if not isinstance(task_id, str) or task_id == "":
        raise ServiceError("task_id must be a non-empty string")
    if mode == "plan" or mode == "execute":
        chosen_mode = mode
    else:
        raise ServiceError('mode must be "plan" or "execute"')
    role: str | None
    if "role" not in body:
        role = None
    else:
        role_value = body["role"]
        if role_value not in {"production", "candidate"}:
            raise ServiceError('role must be "production" or "candidate"')
        role = role_value
    assignment_key: str | None
    if "assignment_key" not in body:
        assignment_key = None
    else:
        key_value = body["assignment_key"]
        if not isinstance(key_value, str) or key_value == "":
            raise ServiceError("assignment_key must be a non-empty string")
        assignment_key = key_value
    return _EpisodeRequest(
        task_id=task_id,
        mode=chosen_mode,
        role=role,
        assignment_key=assignment_key,
    )


def _parse_candidate_admission(body: object) -> str:
    """Parse the request for candidate admission down to one artifact id.

    Candidate admission no longer accepts a caller-typed gate document:
    the request names the ``ValidationArtifact`` a real ``run_offline_gate``
    call already produced and persisted (``EpisodeStore.
    append_validation_artifact``), by its content-addressed id. Every
    other field of that artifact is read back from the store, never from
    this request body, so nothing here can substitute different evidence
    under a borrowed id.
    """

    if not isinstance(body, dict):
        raise ServiceError("body must be a JSON object")
    if not all(isinstance(key, str) for key in body):
        raise ServiceError("body keys must be strings")
    _reject_overrides(body)
    extra = set(body) - _CANDIDATE_ADMISSION_FIELDS
    if extra:
        raise ServiceError(
            "unexpected fields: " + ", ".join(sorted(extra))
        )
    missing = _CANDIDATE_ADMISSION_FIELDS - set(body)
    if missing:
        raise ServiceError(
            "missing fields: " + ", ".join(sorted(missing))
        )
    artifact_id = body["artifact_id"]
    if not isinstance(artifact_id, str) or _ARTIFACT_ID.fullmatch(artifact_id) is None:
        raise ServiceError("artifact_id must be a lowercase SHA-256 hex digest")
    return artifact_id


def _apply_mode(runtime: RuntimeDependencies, mode: ModeName) -> None:
    setter = getattr(runtime.agent, "set_mode", None)
    if callable(setter):
        setter(mode)


def _runtime_for(
    state: _ServiceState,
    config: RunConfiguration,
    mode: ModeName,
    role: str,
) -> RuntimeDependencies:
    """One fresh runtime for ``config`` in ``mode``.

    A ``RuntimeFactory`` builds the runtime for the mode (plan mode then
    bypasses the execute workflow wrapper); a plain callable keeps the
    legacy contract of one runtime per configuration with ``set_mode``.
    """

    factory = state.dependencies.runtime_factory
    if isinstance(factory, RuntimeFactory):
        return factory(config, mode=mode, role=role)
    runtime = factory(config)
    _apply_mode(runtime, mode)
    return runtime


def _persistence_callbacks(
    store: EpisodeStore,
    task_id: str,
) -> tuple[
    Callable[[EpisodeIdentity, RunIdentity], None],
    Callable[[ModelStep | ToolStep], None],
]:
    current: dict[str, str] = {}

    def on_start(identity: EpisodeIdentity, run: RunIdentity) -> None:
        store.start_episode(identity, run, task_id)
        current["episode_id"] = identity.episode_id

    def on_step(step: ModelStep | ToolStep) -> None:
        episode_id = current.get("episode_id")
        if episode_id is None:
            raise StorageError("append_step failed")
        store.append_step(episode_id, step)

    return on_start, on_step


def _feed_monitor(
    state: _ServiceState,
    episode: EpisodeResult,
) -> str:
    """Feed every configured signal from one execute-mode episode.

    Loops over every scalar series ``state.dependencies.monitor.signals``
    names (``task_success``, ``requirement_fraction``, ``tool_error_count``,
    ``invalid_tool_call_count``, ``trajectory_length``, and, when a caller
    schedules them, ``plan_quality_score``/``plan_kl_mean_nats``), not just
    the one signal a launcher happens to name first, so a monitor
    configured for several signals actually sees all of them from the same
    episode. ``tool_selection`` is always recorded as a typed distributional
    observation; ``task_mix`` is recorded whenever the caller's metadata
    names one. Both are routed to their configured ``DistributionalMonitor``
    when the deployment wires one, never coerced into a scalar signal.
    The caller holds ``deployment_lock``, so a rollback the delivered alerts
    trigger (``_apply_rollback_policy``) cannot interleave with routing.
    Every fed signal's metadata is persisted (``EpisodeStore.
    append_monitor_metadata``) beside the already-persisted episode, so a
    later reader can reconstruct the exact detector input this call used.
    Scalar values themselves come only from ``episode.evaluator_outcome``
    and ``episode.tool_steps``/``model_steps`` (``normalize_episode``),
    real execution and evaluation, never a launcher-supplied override.
    ``plan_quality_score`` and ``plan_kl_mean_nats`` are never derivable
    from the episode alone, so each is only observed when the deployment
    wires ``plan_quality_features_for``/``plan_kl_mean_nats_for`` and that
    callable returns a value for this episode; otherwise the signal is
    explicitly skipped (withheld), never forced through the scalar
    episode-normalization path that would reject it.
    """

    base_metadata = state.dependencies.metadata_for(episode)
    monitor = state.dependencies.monitor
    period_id = state.monitor_period_id
    store = state.dependencies.store
    if not state.incidents_restored:
        monitor.restore_open_incidents(store.load_alerts())
        state.incidents_restored = True
    episode_id = episode.episode.episode_id
    all_alerts: list[Alert] = []
    updated = False
    recorded = False
    withheld = False

    for signal in monitor.signals:
        metadata = replace(base_metadata, signal=signal)
        store.append_monitor_metadata(metadata.to_record(episode_id))
        if signal == PLAN_QUALITY_SIGNAL:
            features = (
                None
                if state.dependencies.plan_quality_features_for is None
                else state.dependencies.plan_quality_features_for(episode)
            )
            if features is None:
                withheld = True
                continue
            observation = plan_quality_observation_from_features(
                episode, features=features
            )
        elif signal == PLAN_KL_SIGNAL:
            mean_kl_nats = (
                None
                if state.dependencies.plan_kl_mean_nats_for is None
                else state.dependencies.plan_kl_mean_nats_for(episode)
            )
            if mean_kl_nats is None:
                withheld = True
                continue
            observation = plan_kl_observation(episode, mean_kl_nats=mean_kl_nats)
        else:
            try:
                alerts = monitor.update_from_episode(
                    episode,
                    task_metadata=metadata,
                    period_id=period_id,
                )
            except (MissingEvaluatorOutcome, UndefinedRequirementFraction):
                withheld = True
                continue
            updated = True
            all_alerts.extend(alerts)
            continue
        slice_name = resolved_slice_name(
            episode,
            metadata,
            use_slice_attribution=monitor.use_slice_attribution,
        )
        alerts = monitor.update(
            observation,
            completion_index=metadata.completion_index,
            period_id=period_id,
            slice_name=slice_name,
        )
        updated = True
        all_alerts.extend(alerts)

    tool_metadata = replace(base_metadata, signal="tool_selection")
    store.append_monitor_metadata(tool_metadata.to_record(episode_id))
    selection = tool_selection_observation_from_episode(
        episode,
        task_metadata=tool_metadata,
    )
    tool_monitor = state.dependencies.tool_selection_monitor
    if tool_monitor is not None:
        slice_name = resolved_slice_name(
            episode,
            tool_metadata,
            use_slice_attribution=monitor.use_slice_attribution,
        )
        all_alerts.extend(tool_monitor.update(selection, slice_name=slice_name))
        updated = True
    else:
        recorded = True

    if base_metadata.task_mix is not None:
        mix_metadata = replace(base_metadata, signal="task_mix")
        store.append_monitor_metadata(mix_metadata.to_record(episode_id))
        mix = task_mix_observation_from_episode(
            episode,
            task_metadata=mix_metadata,
        )
        mix_monitor = state.dependencies.task_mix_monitor
        if mix_monitor is not None:
            slice_name = resolved_slice_name(
                episode,
                mix_metadata,
                use_slice_attribution=monitor.use_slice_attribution,
            )
            all_alerts.extend(mix_monitor.update(mix, slice_name=slice_name))
            updated = True
        else:
            recorded = True

    if all_alerts:
        _apply_rollback_policy(state, state.alert_sink.deliver(all_alerts))
    if updated:
        return "updated"
    if recorded:
        return "recorded"
    if withheld:
        return "withheld"
    return "recorded"


def _public_episode_receipt(
    *,
    episode_id: str | None,
    pair_id: str | None,
    role: str,
    mode: str,
    configuration_hash: str,
    status: str,
    monitoring_status: str,
) -> dict[str, object]:
    document: dict[str, object] = {
        "role": role,
        "mode": mode,
        "configuration_hash": configuration_hash,
        "status": status,
        "monitoring_status": monitoring_status,
    }
    if pair_id is not None:
        document["pair_id"] = pair_id
    if episode_id is not None:
        document["episode_id"] = episode_id
    return document


def _snapshot_fields(snapshot: DeploymentSnapshot) -> dict[str, object]:
    return {
        "state": snapshot.state,
        "serving_configuration_hash": snapshot.serving_configuration_hash,
        "previous_production_configuration_hash": (
            snapshot.previous_production_configuration_hash
        ),
        "candidate_configuration_hash": snapshot.candidate_configuration_hash,
        "candidate_episodes_started": snapshot.candidate_episodes_started,
        "candidate_episodes_served": snapshot.candidate_episodes_served,
        "candidate_episodes_failed": snapshot.candidate_episodes_failed,
        "candidate_episodes_evaluator_unsuccessful": (
            snapshot.candidate_episodes_evaluator_unsuccessful
        ),
        "outstanding": snapshot.outstanding,
        "in_flight_at_rollback": snapshot.in_flight_at_rollback,
        "served_before_rollback": snapshot.served_before_rollback,
        "promoted_configuration_hash": snapshot.promoted_configuration_hash,
        "monitoring_reset_required": snapshot.monitoring_reset_required,
    }


def _deployment_document(state: _ServiceState) -> dict[str, object]:
    registry = state.dependencies.registry
    production_hash = run_configuration_hash(state.serving_configuration)
    candidate_hash = (
        None
        if registry.candidate is None
        else run_configuration_hash(registry.candidate)
    )
    document: dict[str, object] = {
        "admission": state.admission,
        "production_configuration_hash": production_hash,
        "candidate_configuration_hash": candidate_hash,
        "monitor_period_id": state.monitor_period_id,
        "post_promotion_rollback": state.dependencies.post_promotion_rollback,
    }
    controller = state.controller
    if controller is not None:
        document.update(_snapshot_fields(controller.snapshot()))
    elif state.recovered is not None:
        recovered = state.recovered
        document.update(
            {
                "state": recovered.state,
                "serving_configuration_hash": production_hash,
                "previous_production_configuration_hash": (
                    recovered.previous_production_configuration_hash
                ),
                "promoted_configuration_hash": recovered.promoted_configuration_hash,
                "rejected_candidate_configuration_hash": (
                    recovered.rejected_candidate_configuration_hash
                ),
                "recovered_from_store": True,
                "recovered_decision_method": recovered.method,
            }
        )
    rollback = state.production_rollback
    if rollback is not None:
        document.update(
            {
                "state": PRODUCTION_ROLLED_BACK,
                "serving_configuration_hash": production_hash,
                "previous_production_configuration_hash": (
                    rollback.restored_configuration_hash
                ),
                "promoted_configuration_hash": rollback.rolled_back_configuration_hash,
                "rolled_back_configuration_hash": rollback.rolled_back_configuration_hash,
                "production_rollback": _production_rollback_fields(rollback),
            }
        )
    return document


def _production_rollback_fields(rollback: _ProductionRollback) -> dict[str, object]:
    trigger = rollback.trigger
    return {
        "method": rollback.method,
        "source": rollback.source,
        "decided_at": rollback.decided_at.isoformat(),
        "rolled_back_configuration_hash": rollback.rolled_back_configuration_hash,
        "restored_configuration_hash": rollback.restored_configuration_hash,
        "rolled_back_period_id": rollback.rolled_back_period_id,
        "persisted": rollback.persisted,
        "recovered_from_store": rollback.recovered,
        "trigger": (
            None
            if trigger is None
            else {
                "signal": trigger.signal,
                "slice_name": trigger.slice_name,
                "method": trigger.method,
                "estimate": trigger.estimate,
                "boundary": trigger.boundary,
                "sample_size": trigger.sample_size,
                "period_id": trigger.period_id,
            }
        ),
    }


def _persist_lifecycle_decision(
    state: _ServiceState,
    *,
    decision: str,
    snapshot: DeploymentSnapshot,
    decided_at: datetime,
    method: str,
    evidence_artifact_id: str | None = None,
    evidence_source: str | None = None,
) -> None:
    state.dependencies.store.append_deployment_decision(
        DeploymentDecisionRecord(
            configuration_hash=snapshot.serving_configuration_hash,
            reference_configuration_hash=(
                snapshot.previous_production_configuration_hash
            ),
            signal="deployment",
            slice_name="canary",
            decision=decision,
            method=method,
            estimate=0.0,
            boundary=None,
            sample_size=snapshot.candidate_episodes_served,
            decided_at=decided_at,
            evidence_artifact_id=evidence_artifact_id,
            evidence_source=evidence_source,
        )
    )


def _start_monitoring_period(state: _ServiceState, serving_hash: str, reference_hash: str) -> None:
    """Monitor ``serving_hash`` against ``reference_hash`` in their own period.

    Detectors are rebuilt from the same frozen baselines, bound to
    ``reference_hash``. A monitor already in that period is left as it is.
    """

    monitor = state.dependencies.monitor
    period = monitoring_period_id(serving_hash, reference_hash)
    if monitor.period_id != period:
        current = monitor.reference
        monitor.reset_for_promotion(
            FrozenReference(
                configuration_hash=reference_hash,
                baselines=current.baselines,
                slice_baselines=current.slice_baselines,
                source=current.source,
            ),
            period_id=period,
        )
        for distributional in (
            state.dependencies.tool_selection_monitor,
            state.dependencies.task_mix_monitor,
        ):
            if distributional is not None:
                distributional.reset(
                    reference_configuration_hash=reference_hash,
                    period_id=period,
                )
    state.monitor_period_id = period
    state.incidents_restored = False


def _serve_promoted(
    state: _ServiceState,
    promoted: RunConfiguration,
    previous_hash: str,
) -> None:
    """Serve ``promoted`` and monitor it against ``previous_hash`` in its own period."""

    state.serving_configuration = promoted
    _start_monitoring_period(state, run_configuration_hash(promoted), previous_hash)


def _serve_known_good(state: _ServiceState) -> None:
    """Serve the registered production again, monitored against itself.

    Its period is the one the service started in before admission, never
    the rolled-back configuration's, so fallback traffic is not scored as
    if it belonged to the failed promotion.
    """

    production = state.dependencies.registry.production
    production_hash = run_configuration_hash(production)
    state.serving_configuration = production
    _start_monitoring_period(state, production_hash, production_hash)


def _promoted_lineage(state: _ServiceState) -> tuple[str, str] | None:
    """``(promoted hash, previous production hash)`` while a promotion serves."""

    if state.production_rollback is not None:
        return None
    controller = state.controller
    if controller is not None:
        snapshot = controller.snapshot()
        if snapshot.state != "PROMOTED":
            return None
        return (
            snapshot.serving_configuration_hash,
            snapshot.previous_production_configuration_hash,
        )
    recovered = state.recovered
    if (
        recovered is not None
        and recovered.state == "PROMOTED"
        and recovered.promoted_configuration_hash is not None
    ):
        return (
            recovered.promoted_configuration_hash,
            recovered.previous_production_configuration_hash,
        )
    return None


def _is_rollback_alert(alert: Alert, promoted_hash: str, previous_hash: str) -> bool:
    """Whether ``alert`` is the aggregate ``task_success`` incident of this promotion."""

    return (
        alert.signal == ROLLBACK_SIGNAL
        and alert.slice_name == ROLLBACK_SIGNAL
        and alert.configuration_hash == promoted_hash
        and alert.reference_configuration_hash == previous_hash
        and alert.period_id == monitoring_period_id(promoted_hash, previous_hash)
    )


def _stored_rollback_alert(
    state: _ServiceState,
    promoted_hash: str,
    previous_hash: str,
) -> Alert | None:
    for record in state.dependencies.store.load_alerts(signal=ROLLBACK_SIGNAL):
        alert = Alert.from_record(record)
        if _is_rollback_alert(alert, promoted_hash, previous_hash):
            return alert
    return None


def _persist_production_rollback(state: _ServiceState, rollback: _ProductionRollback) -> None:
    trigger = rollback.trigger
    state.dependencies.store.append_deployment_decision(
        DeploymentDecisionRecord(
            configuration_hash=rollback.rolled_back_configuration_hash,
            reference_configuration_hash=rollback.restored_configuration_hash,
            signal="deployment",
            slice_name="production",
            decision=_PRODUCTION_ROLLBACK,
            method=rollback.method,
            estimate=0.0 if trigger is None else trigger.estimate,
            boundary=None if trigger is None else trigger.boundary,
            sample_size=0 if trigger is None else trigger.sample_size,
            decided_at=rollback.decided_at,
            evidence_source=rollback.source,
            period_id=rollback.rolled_back_period_id,
            trigger_signal=None if trigger is None else trigger.signal,
        )
    )
    state.production_rollback = replace(rollback, persisted=True)


def _roll_back_production(
    state: _ServiceState,
    *,
    method: str,
    source: str,
    trigger: Alert | None,
) -> None:
    """Stop serving the promoted configuration and serve the previous production.

    The one post-promotion transition, shared by the monitor policy, the
    manual endpoint, and restart reconciliation; the caller holds
    ``deployment_lock`` or runs before the app serves. The canary controller
    is not touched: it stays PROMOTED as the record of Tier 2. The previous
    production must be the registered production and the promoted
    configuration the registered candidate. Routing switches before the
    decision is written, so a failed write still leaves the known-good
    configuration serving; the write is retried by the next call. A call
    after the rollback changes nothing.
    """

    existing = state.production_rollback
    if existing is not None:
        if not existing.persisted:
            _persist_production_rollback(state, existing)
        return
    lineage = _promoted_lineage(state)
    if lineage is None:
        raise _Conflict("no promoted deployment to roll back")
    promoted_hash, previous_hash = lineage
    registry = state.dependencies.registry
    if previous_hash != run_configuration_hash(registry.production):
        raise _Conflict("the promotion's previous production is not the registered production")
    if registry.candidate is None or promoted_hash != run_configuration_hash(registry.candidate):
        raise _Conflict("the promoted configuration is not the registered candidate")
    rollback = _ProductionRollback(
        rolled_back_configuration_hash=promoted_hash,
        restored_configuration_hash=previous_hash,
        rolled_back_period_id=monitoring_period_id(promoted_hash, previous_hash),
        method=method,
        source=source,
        trigger=trigger,
        decided_at=state.dependencies.clock(),
        persisted=False,
        recovered=False,
    )
    state.production_rollback = rollback
    state.admission = "rollback_requested"
    _serve_known_good(state)
    _persist_production_rollback(state, rollback)


def _apply_rollback_policy(state: _ServiceState, delivered: tuple[Alert, ...]) -> None:
    """Under ``rollback_on_alert``, roll a promotion back on its first eligible new alert.

    Only alerts ``LocalAlertSink`` newly delivered reach here, so a
    de-duplicated or restored incident never acts twice. Eligible means the
    aggregate ``task_success`` alert of the configuration being served
    against the frozen previous production, in the current monitoring
    period; other signals, slices, and periods are recorded only.
    """

    if state.dependencies.post_promotion_rollback != "rollback_on_alert":
        return
    lineage = _promoted_lineage(state)
    if lineage is None:
        return
    promoted_hash, previous_hash = lineage
    if state.dependencies.monitor.reference.configuration_hash != previous_hash:
        return
    for alert in delivered:
        if alert.period_id == state.monitor_period_id and _is_rollback_alert(
            alert, promoted_hash, previous_hash
        ):
            _roll_back_production(
                state, method="monitor_alert", source="production_monitor", trigger=alert
            )
            return


def _admitted_candidate_hash(
    state: _ServiceState,
    admit: DeploymentDecisionRecord,
) -> str | None:
    """The registered candidate's hash when ``admit``'s artifact authorized it, else ``None``.

    This identifies the candidate a stored admission let in; it is not an
    admission, so synthetic evidence is accepted here.
    """

    registry = state.dependencies.registry
    if registry.candidate is None:
        return None
    if admit.evidence_artifact_id is None:
        raise ServiceError("stored admission names no validation artifact")
    artifact = state.dependencies.store.load_validation_artifact(admit.evidence_artifact_id)
    if artifact is None:
        raise ServiceError("stored admission's validation artifact is not recorded")
    try:
        admission = authorize_test_gated_candidate(
            artifact,
            registry.production,
            registry.candidate,
            allowance=state.dependencies.task_selection_allowance,
        )
    except ProtocolError:
        return None
    return admission.served_candidate_configuration_hash


def _reconcile_deployment(state: _ServiceState) -> None:
    """Restore the last stored deployment decision at startup, or refuse to start.

    No decision keeps the registered production. A stored promotion must
    name the registered candidate as promoted over the registered
    production; it is served again against that production in the same
    monitoring period. A stored rollback keeps production and refuses the
    candidate that admission let in. An admission with no promotion or
    rollback after it is closed with a persisted rollback decision, so
    candidate traffic never resumes. A stored ``production_rollback`` after
    its promotion serves the previous production again and refuses the
    candidate (``_recover_production_rollback``). A promotion whose eligible
    alert was stored with no rollback after it is rolled back now under
    ``rollback_on_alert``. A decision whose hashes do not match the
    registered configurations raises ``ServiceError``.
    """

    dependencies = state.dependencies
    store = dependencies.store
    rows = [
        row
        for row in store.load_deployment_decisions()
        if row.signal == "deployment" and row.decision in _LIFECYCLE_DECISIONS
    ]
    if not rows:
        return
    registry = dependencies.registry
    production_hash = run_configuration_hash(registry.production)
    candidate_hash = (
        None if registry.candidate is None else run_configuration_hash(registry.candidate)
    )
    last = rows[-1]
    if last.decision == _PRODUCTION_ROLLBACK:
        _recover_production_rollback(state, rows)
        return
    if any(row.decision == _PRODUCTION_ROLLBACK for row in rows):
        raise ServiceError("a stored production rollback is followed by another lifecycle decision")
    if last.decision == "promote":
        if (
            registry.candidate is None
            or last.configuration_hash != candidate_hash
            or last.reference_configuration_hash != production_hash
        ):
            raise ServiceError(
                "stored promotion does not match the registered production and "
                "candidate configurations"
            )
        _serve_promoted(state, registry.candidate, production_hash)
        state.recovered = _RecoveredDeployment(
            state="PROMOTED",
            previous_production_configuration_hash=production_hash,
            promoted_configuration_hash=candidate_hash,
            rejected_candidate_configuration_hash=None,
            method=last.method,
        )
        if dependencies.post_promotion_rollback == "rollback_on_alert":
            alert = _stored_rollback_alert(state, candidate_hash, production_hash)
            if alert is not None:
                _roll_back_production(
                    state,
                    method="monitor_alert",
                    source="restart_reconciliation",
                    trigger=alert,
                )
        return
    if (
        last.configuration_hash != production_hash
        or last.reference_configuration_hash != production_hash
    ):
        raise ServiceError(
            f"stored {last.decision} decision does not match the registered "
            "production configuration"
        )
    admits = [row for row in rows if row.decision == "admit"]
    if not admits:
        raise ServiceError("stored rollback has no admission before it")
    rejected = _admitted_candidate_hash(state, admits[-1])
    method = last.method
    if last.decision == "admit":
        method = _UNFINISHED_AT_RESTART
        store.append_deployment_decision(
            DeploymentDecisionRecord(
                configuration_hash=production_hash,
                reference_configuration_hash=production_hash,
                signal="deployment",
                slice_name="canary",
                decision="rollback",
                method=method,
                estimate=0.0,
                boundary=None,
                sample_size=0,
                decided_at=dependencies.clock(),
                evidence_artifact_id=last.evidence_artifact_id,
                evidence_source=last.evidence_source,
            )
        )
    state.admission = "rollback_requested"
    state.recovered = _RecoveredDeployment(
        state="ROLLED_BACK",
        previous_production_configuration_hash=production_hash,
        promoted_configuration_hash=None,
        rejected_candidate_configuration_hash=rejected,
        method=method,
    )


def _recover_production_rollback(
    state: _ServiceState,
    rows: list[DeploymentDecisionRecord],
) -> None:
    """Serve the previous production after a stored post-promotion rollback.

    The stored lifecycle must end admit, promote, production_rollback, for
    the registered candidate over the registered production, in that
    promotion's monitoring period; a ``monitor_alert`` rollback must also
    find its alert in the store. Nothing is written: the promotion, alert,
    and rollback rows stay as they are, and no alert is delivered again.
    """

    registry = state.dependencies.registry
    production_hash = run_configuration_hash(registry.production)
    candidate_hash = (
        None if registry.candidate is None else run_configuration_hash(registry.candidate)
    )
    rollback = rows[-1]
    if (
        len(rows) < 3
        or rows[-2].decision != "promote"
        or rows[-3].decision != "admit"
        or sum(row.decision == _PRODUCTION_ROLLBACK for row in rows) != 1
    ):
        raise ServiceError(
            "stored production rollback does not follow an admission and a promotion"
        )
    promote = rows[-2]
    if (
        candidate_hash is None
        or promote.configuration_hash != candidate_hash
        or rollback.configuration_hash != candidate_hash
        or promote.reference_configuration_hash != production_hash
        or rollback.reference_configuration_hash != production_hash
    ):
        raise ServiceError(
            "stored production rollback does not match the registered production and "
            "candidate configurations"
        )
    period = monitoring_period_id(candidate_hash, production_hash)
    if rollback.period_id != period:
        raise ServiceError("stored production rollback names another monitoring period")
    if rollback.method not in _PRODUCTION_ROLLBACK_METHODS:
        raise ServiceError(f"stored production rollback method {rollback.method} is unknown")
    trigger = None
    if rollback.method == "monitor_alert":
        trigger = _stored_rollback_alert(state, candidate_hash, production_hash)
        if trigger is None or rollback.trigger_signal != trigger.signal:
            raise ServiceError("stored monitor-triggered rollback has no matching alert")
    state.recovered = _RecoveredDeployment(
        state="PROMOTED",
        previous_production_configuration_hash=production_hash,
        promoted_configuration_hash=candidate_hash,
        rejected_candidate_configuration_hash=candidate_hash,
        method=promote.method,
    )
    state.production_rollback = _ProductionRollback(
        rolled_back_configuration_hash=candidate_hash,
        restored_configuration_hash=production_hash,
        rolled_back_period_id=period,
        method=rollback.method,
        source=rollback.evidence_source or "unrecorded",
        trigger=trigger,
        decided_at=rollback.decided_at,
        persisted=True,
        recovered=True,
    )
    state.admission = "rollback_requested"
    _serve_known_good(state)


def _apply_canary_decision(
    state: _ServiceState,
    decision: CanaryDecision,
) -> None:
    if decision.action == "promote":
        registry = state.dependencies.registry
        if registry.candidate is None:
            raise _Conflict("candidate configuration is not registered")
        _serve_promoted(
            state,
            registry.candidate,
            decision.snapshot.previous_production_configuration_hash,
        )
        decided_at = decision.snapshot.promoted_at
        if decided_at is None:
            decided_at = state.dependencies.clock()
        _persist_lifecycle_decision(
            state,
            decision="promote",
            snapshot=decision.snapshot,
            decided_at=decided_at,
            method="stopping_rule",
        )
        return
    if decision.action == "rollback":
        state.admission = "rollback_requested"
        decided_at = decision.snapshot.rollback_at
        if decided_at is None:
            decided_at = state.dependencies.clock()
        reason = decision.snapshot.rollback_reason or "rollback"
        _persist_lifecycle_decision(
            state,
            decision="rollback",
            snapshot=decision.snapshot,
            decided_at=decided_at,
            method=reason,
        )


class _Conflict(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class _Unavailable(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class _TooMany(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def _controller_is_active(state: _ServiceState) -> bool:
    controller = state.controller
    if controller is None:
        return False
    return controller.snapshot().state in _ACTIVE_CANARY


def _should_run_canary(state: _ServiceState, request: _EpisodeRequest) -> bool:
    if request.role == "production":
        return False
    if request.role == "candidate":
        return True
    if state.admission == "rollback_requested":
        return False
    if not _controller_is_active(state):
        return False
    if state.dependencies.registry.candidate is None:
        return False
    if request.assignment_key is not None:
        key = request.assignment_key
    else:
        stamped = state.dependencies.clock()
        key = f"{request.task_id}:{stamped.isoformat()}"
    return assign_canary(
        key,
        fraction=float(state.dependencies.canary_settings.fraction),
        seed=state.dependencies.canary_assignment_seed,
    )


def _require_model_calls(state: _ServiceState, request: _EpisodeRequest) -> None:
    """Refuse an episode the model-call budget cannot cover at its turn limit.

    A canary arrival may run a pair, so while a canary is active the
    candidate's limit is added. The budget's own counter still stops a call
    that would exceed it inside an episode.
    """

    budget = state.dependencies.model_call_budget
    if budget is None:
        return
    with state.deployment_lock:
        serving = state.serving_configuration
        active = _controller_is_active(state)
    candidate = state.dependencies.registry.candidate
    if request.mode == "plan":
        needed = 1 + (1 if active else 0)
    else:
        needed = serving.agent.execute_turn_limit + (
            candidate.agent.execute_turn_limit if active and candidate is not None else 0
        )
    if budget.remaining < needed:
        raise _Unavailable("the release's model-call budget is spent")


def _run_production_episode(
    state: _ServiceState,
    request: _EpisodeRequest,
) -> dict[str, object]:
    """Serve one episode on the configuration serving when it was admitted.

    The configuration and monitoring period are read together under
    ``deployment_lock``. The receipt names the configuration that actually
    ran. An execute episode whose period ended while it ran (a promotion or
    a rollback in between) is persisted but not fed to the monitor, and its
    ``monitoring_status`` is ``superseded``.
    """

    with state.deployment_lock:
        config = state.serving_configuration
        period_id = state.monitor_period_id
    run = new_run_identity(config)
    runtime = _runtime_for(state, config, request.mode, "production")
    on_start, on_step = _persistence_callbacks(
        state.dependencies.store,
        request.task_id,
    )
    episode = run_episode(
        request.task_id,
        config,
        request.mode,
        run=run,
        runtime=runtime,
        on_start=on_start,
        on_step=on_step,
    )
    state.dependencies.store.finish_episode(episode)
    if request.mode == "execute":
        with state.deployment_lock:
            if state.monitor_period_id == period_id:
                monitoring_status = _feed_monitor(state, episode)
            else:
                monitoring_status = _SUPERSEDED
    else:
        monitoring_status = "skipped"
    return _public_episode_receipt(
        episode_id=episode.episode.episode_id,
        pair_id=None,
        role="production",
        mode=request.mode,
        configuration_hash=run_configuration_hash(config),
        status=episode.status,
        monitoring_status=monitoring_status,
    )


def _run_candidate_episode(
    state: _ServiceState,
    request: _EpisodeRequest,
) -> dict[str, object]:
    if state.admission == "rollback_requested":
        raise _Conflict("candidate serving is closed after rollback")
    registry = state.dependencies.registry
    if registry.candidate is None:
        raise _Conflict("candidate configuration is not registered")
    begun = False
    with state.deployment_lock:
        controller = state.controller
        if controller is None or controller.snapshot().state not in _ACTIVE_CANARY:
            raise _Conflict("no active canary controller")
        controller.begin_candidate_episode()
        begun = True
    reference = registry.production
    candidate = registry.candidate
    reference_run = new_run_identity(reference)
    candidate_run = new_run_identity(candidate)
    reference_runtime = _runtime_for(state, reference, request.mode, "reference")
    candidate_runtime = _runtime_for(state, candidate, request.mode, "candidate")
    on_start, on_step = _persistence_callbacks(
        state.dependencies.store,
        request.task_id,
    )
    try:
        pair = run_pair(
            request.task_id,
            reference,
            candidate,
            reference_run=reference_run,
            candidate_run=candidate_run,
            runtime=reference_runtime,
            candidate_runtime=candidate_runtime,
            mode=request.mode,
            on_start=on_start,
            on_step=on_step,
        )
    except Exception:
        if begun:
            with state.deployment_lock:
                if state.controller is not None:
                    phase = state.controller.snapshot().state
                    if phase in {"CANARY_ACTIVE", "ROLLED_BACK"}:
                        state.controller.abort_outstanding()
        raise
    state.dependencies.store.finish_episode(pair.reference)
    state.dependencies.store.finish_episode(pair.candidate)
    with state.deployment_lock:
        if state.controller is None:
            raise _Conflict("no active canary controller")
        state.dependencies.store.append_pair(pair)
        phase = state.controller.snapshot().state
        if phase == "ROLLED_BACK":
            state.controller.complete_outstanding(pair)
        elif phase == "CANARY_ACTIVE":
            decision = state.controller.observe(pair)
            _apply_canary_decision(state, decision)
        else:
            raise _Conflict("no active canary controller")
    pair_id = pair.reference.episode.pair_id
    return _public_episode_receipt(
        episode_id=pair.candidate.episode.episode_id,
        pair_id=pair_id,
        role="candidate",
        mode=request.mode,
        configuration_hash=run_configuration_hash(candidate),
        status=pair.candidate.status,
        monitoring_status="skipped",
    )


def _admit_candidate(state: _ServiceState, artifact_id: str) -> dict[str, object]:
    registry = state.dependencies.registry
    if registry.candidate is None:
        raise _Conflict("candidate configuration is not registered")
    if (
        registry.production.task.split == "test_normal"
        or registry.candidate.task.split == "test_normal"
    ):
        raise _Conflict(
            "final-test admission is not performed by this service"
        )
    if state.production_rollback is not None:
        raise _Conflict("the registered candidate was rolled back after promotion")
    recovered = state.recovered
    if recovered is not None and state.controller is None:
        candidate_hash = run_configuration_hash(registry.candidate)
        if recovered.promoted_configuration_hash == candidate_hash:
            raise _Conflict("the registered candidate is already promoted")
        if recovered.rejected_candidate_configuration_hash == candidate_hash:
            raise _Conflict("the registered candidate was rolled back")
    artifact = state.dependencies.store.load_validation_artifact(artifact_id)
    if artifact is None:
        raise _Conflict("validation artifact is not recorded")
    authorize = (
        authorize_test_gated_candidate
        if state.dependencies.admission_mode == "test"
        else authorize_gated_candidate
    )
    try:
        admission = authorize(
            artifact,
            registry.production,
            registry.candidate,
            allowance=state.dependencies.task_selection_allowance,
        )
    except ProtocolError as error:
        raise _Conflict(_message(error)) from error
    with state.deployment_lock:
        if state.production_rollback is not None:
            raise _Conflict("the registered candidate was rolled back after promotion")
        existing = state.controller
        if existing is not None and existing.snapshot().state not in _TERMINAL_CANARY:
            raise _Conflict("a non-terminal canary controller already exists")
        if existing is not None and existing.snapshot().state == "PROMOTED":
            raise _Conflict("the registered candidate is already promoted")
        controller = CanaryController(
            registry.production,
            registry.candidate,
            settings=state.dependencies.canary_settings,
            clock=state.dependencies.clock,
        )
        try:
            controller.start_from_admission(admission)
        except CanaryRejected as error:
            raise _Conflict(_message(error)) from error
        state.controller = controller
        state.admission = "open"
        _persist_lifecycle_decision(
            state,
            decision="admit",
            snapshot=controller.snapshot(),
            decided_at=state.dependencies.clock(),
            method="validation_artifact",
            evidence_artifact_id=admission.evidence_artifact_id,
            evidence_source=artifact.evidence_source,
        )
        return _deployment_document(state)


def _request_rollback(state: _ServiceState) -> dict[str, object]:
    """Roll back whatever the deployment is doing, at most once.

    An active canary is closed through ``CanaryController.rollback``. A
    promotion, live or recovered, takes the post-promotion transition
    (``_roll_back_production``) instead. After either, a repeated request
    changes nothing and writes no decision.
    """

    with state.deployment_lock:
        if state.production_rollback is not None or _promoted_lineage(state) is not None:
            _roll_back_production(
                state, method="manual_rollback", source="operator", trigger=None
            )
            return _deployment_document(state)
        state.admission = "rollback_requested"
        controller = state.controller
        if controller is not None:
            phase = controller.snapshot().state
            if phase in {"GATE_PASSED", "CANARY_ACTIVE"}:
                decision = controller.rollback("manual_rollback", manual=True)
                _persist_lifecycle_decision(
                    state,
                    decision="rollback",
                    snapshot=decision.snapshot,
                    decided_at=decision.snapshot.rollback_at
                    or state.dependencies.clock(),
                    method="manual_rollback",
                )
        return _deployment_document(state)


def create_app(dependencies: ServiceDependencies) -> FastAPI:
    """Build the FastAPI app bound to explicit service dependencies."""

    if not isinstance(dependencies, ServiceDependencies):
        raise ServiceError("dependencies must be ServiceDependencies")
    state = _ServiceState(dependencies)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        del app
        state.admission = "open"
        _reconcile_deployment(state)
        yield
        state.admission = "shutting_down"
        deadline = monotonic() + float(
            dependencies.shutdown_timeout_seconds
        )
        while state.in_flight() > 0 and monotonic() < deadline:
            await asyncio.sleep(0.01)
        state.close_store_once()

    app = FastAPI(lifespan=lifespan)
    app.state.service_state = state
    app.state.service = state

    @app.exception_handler(ServiceError)
    async def service_error_handler(
        request: Request,
        error: ServiceError,
    ) -> JSONResponse:
        del request
        return JSONResponse(status_code=422, content={"detail": _message(error)})

    @app.exception_handler(_Conflict)
    async def conflict_handler(
        request: Request,
        error: _Conflict,
    ) -> JSONResponse:
        del request
        return JSONResponse(status_code=409, content={"detail": error.message})

    @app.exception_handler(_Unavailable)
    async def unavailable_handler(
        request: Request,
        error: _Unavailable,
    ) -> JSONResponse:
        del request
        return JSONResponse(status_code=503, content={"detail": error.message})

    @app.exception_handler(_TooMany)
    async def too_many_handler(
        request: Request,
        error: _TooMany,
    ) -> JSONResponse:
        del request
        return JSONResponse(status_code=429, content={"detail": error.message})

    @app.exception_handler(EpisodeRejected)
    async def episode_rejected_handler(
        request: Request,
        error: EpisodeRejected,
    ) -> JSONResponse:
        del request
        return JSONResponse(status_code=400, content={"detail": _message(error)})

    @app.exception_handler(CanaryRejected)
    async def canary_rejected_handler(
        request: Request,
        error: CanaryRejected,
    ) -> JSONResponse:
        del request
        return JSONResponse(status_code=400, content={"detail": _message(error)})

    @app.exception_handler(ConfigError)
    async def config_error_handler(
        request: Request,
        error: ConfigError,
    ) -> JSONResponse:
        del request
        return JSONResponse(status_code=400, content={"detail": _message(error)})

    @app.exception_handler(StorageError)
    async def storage_error_handler(
        request: Request,
        error: StorageError,
    ) -> JSONResponse:
        del request
        return JSONResponse(status_code=400, content={"detail": _message(error)})

    @app.exception_handler(RuntimeUnavailable)
    async def runtime_unavailable_handler(
        request: Request,
        error: RuntimeUnavailable,
    ) -> JSONResponse:
        del request
        return JSONResponse(status_code=503, content={"detail": _message(error)})

    @app.exception_handler(MonitorRejected)
    async def monitor_rejected_handler(
        request: Request,
        error: MonitorRejected,
    ) -> JSONResponse:
        del request
        return JSONResponse(status_code=400, content={"detail": _message(error)})

    @app.exception_handler(ProtocolError)
    async def protocol_error_handler(
        request: Request,
        error: ProtocolError,
    ) -> JSONResponse:
        del request
        return JSONResponse(status_code=409, content={"detail": _message(error)})

    @app.get("/deployment")
    async def get_deployment() -> dict[str, object]:
        with state.deployment_lock:
            return _deployment_document(state)

    @app.post("/candidates")
    async def post_candidates(request: Request) -> dict[str, object]:
        if state.admission == "shutting_down":
            raise _Unavailable("service is shutting down")
        try:
            body = await request.json()
        except Exception as error:
            raise ServiceError("body must be JSON") from error
        artifact_id = _parse_candidate_admission(body)
        return _admit_candidate(state, artifact_id)

    @app.post("/deployment/rollback")
    async def post_rollback() -> dict[str, object]:
        if state.admission == "shutting_down":
            raise _Unavailable("service is shutting down")
        return _request_rollback(state)

    @app.post("/episodes")
    async def post_episodes(request: Request) -> dict[str, object]:
        if state.admission == "shutting_down":
            raise _Unavailable("service is shutting down")
        try:
            body = await request.json()
        except Exception as error:
            raise ServiceError("body must be JSON") from error
        episode_request = _parse_episode_request(body)
        _require_model_calls(state, episode_request)
        if not state.try_acquire_slot():
            raise _TooMany("max_in_flight episodes already running")
        try:
            if _should_run_canary(state, episode_request):
                return _run_candidate_episode(state, episode_request)
            return _run_production_episode(state, episode_request)
        finally:
            state.release_slot()

    return app
