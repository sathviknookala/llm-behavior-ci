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
_ACTIVE_CANARY = frozenset({"GATE_PASSED", "CANARY_ACTIVE"})


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
        except ValueError as error:
            raise ServiceError(error) from error


@dataclass(frozen=True)
class _EpisodeRequest:
    task_id: str
    mode: ModeName
    role: str | None
    assignment_key: str | None


class _ServiceState:
    def __init__(self, dependencies: ServiceDependencies) -> None:
        self.dependencies = dependencies
        self.admission = "open"
        self.controller: CanaryController | None = None
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
        state.alert_sink.deliver(all_alerts)
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
    }
    controller = state.controller
    if controller is not None:
        document.update(_snapshot_fields(controller.snapshot()))
    return document


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


def _apply_canary_decision(
    state: _ServiceState,
    decision: CanaryDecision,
) -> None:
    if decision.action == "promote":
        registry = state.dependencies.registry
        if registry.candidate is None:
            raise _Conflict("candidate configuration is not registered")
        state.serving_configuration = registry.candidate
        previous_hash = decision.snapshot.previous_production_configuration_hash
        current = state.dependencies.monitor.reference
        new_period = monitoring_period_id(
            run_configuration_hash(registry.candidate), previous_hash
        )
        state.dependencies.monitor.reset_for_promotion(
            FrozenReference(
                configuration_hash=previous_hash,
                baselines=current.baselines,
                slice_baselines=current.slice_baselines,
                source=current.source,
            ),
            period_id=new_period,
        )
        for distributional in (
            state.dependencies.tool_selection_monitor,
            state.dependencies.task_mix_monitor,
        ):
            if distributional is not None:
                distributional.reset(
                    reference_configuration_hash=previous_hash,
                    period_id=new_period,
                )
        state.monitor_period_id = new_period
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


def _run_production_episode(
    state: _ServiceState,
    request: _EpisodeRequest,
) -> dict[str, object]:
    config = state.serving_configuration
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
        monitoring_status = _feed_monitor(state, episode)
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
        existing = state.controller
        if existing is not None and existing.snapshot().state not in _TERMINAL_CANARY:
            raise _Conflict("a non-terminal canary controller already exists")
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
    with state.deployment_lock:
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
        yield
        state.admission = "shutting_down"
        deadline = monotonic() + float(
            dependencies.shutdown_timeout_seconds
        )
        while state.in_flight() > 0 and monotonic() < deadline:
            await asyncio.sleep(0.01)
        state.close_store_once()

    app = FastAPI(lifespan=lifespan)
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
        if not state.try_acquire_slot():
            raise _TooMany("max_in_flight episodes already running")
        try:
            if _should_run_canary(state, episode_request):
                return _run_candidate_episode(state, episode_request)
            return _run_production_episode(state, episode_request)
        finally:
            state.release_slot()

    return app
