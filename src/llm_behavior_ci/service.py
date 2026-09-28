"""HTTP service for plan/execute episodes, canary admission, and deployment status."""

from __future__ import annotations

import asyncio
import math
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from llm_behavior_ci.config import (
    CanarySettings,
    ConfigError,
    RunConfiguration,
    new_run_identity,
    run_configuration_hash,
)
from llm_behavior_ci.lifecycle.canary import (
    CanaryController,
    CanaryRejected,
    DeploymentSnapshot,
)
from llm_behavior_ci.lifecycle.monitoring import (
    MissingEvaluatorOutcome,
    MonitorRejected,
    ProductionMonitor,
    TaskMetadata,
    UndefinedRequirementFraction,
    observation_from_episode,
)
from llm_behavior_ci.records import EpisodeResult
from llm_behavior_ci.runtime.episode import (
    EpisodeRejected,
    RuntimeDependencies,
    RuntimeUnavailable,
    run_episode,
    run_pair,
)
from llm_behavior_ci.storage import EpisodeStore, StorageError

_EPISODE_FIELDS = frozenset({"task_id", "mode", "role"})
_GATE_FIELDS = frozenset(
    {
        "outcome",
        "reason_codes",
        "reference_configuration_hash",
        "candidate_configuration_hash",
        "task_set_hash",
        "reference_protocol_hash",
        "candidate_protocol_hash",
    }
)
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
        except ValueError as error:
            raise ServiceError(error) from error


@dataclass(frozen=True)
class _GateDocument:
    outcome: str
    reason_codes: tuple[str, ...]
    reference_configuration_hash: str
    candidate_configuration_hash: str
    task_set_hash: str
    reference_protocol_hash: str | None
    candidate_protocol_hash: str | None


@dataclass(frozen=True)
class _EpisodeRequest:
    task_id: str
    mode: ModeName
    role: str


class _ServiceState:
    def __init__(self, dependencies: ServiceDependencies) -> None:
        self.dependencies = dependencies
        self.admission = "open"
        self.controller: CanaryController | None = None
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
    missing = _EPISODE_FIELDS - set(body)
    if missing:
        raise ServiceError(
            "missing fields: " + ", ".join(sorted(missing))
        )
    task_id = body["task_id"]
    mode = body["mode"]
    role = body["role"]
    if not isinstance(task_id, str) or task_id == "":
        raise ServiceError("task_id must be a non-empty string")
    if mode == "plan" or mode == "execute":
        chosen_mode = mode
    else:
        raise ServiceError('mode must be "plan" or "execute"')
    if role not in {"production", "candidate"}:
        raise ServiceError('role must be "production" or "candidate"')
    return _EpisodeRequest(task_id=task_id, mode=chosen_mode, role=role)


def _parse_gate(body: object) -> _GateDocument:
    if not isinstance(body, dict):
        raise ServiceError("body must be a JSON object")
    if not all(isinstance(key, str) for key in body):
        raise ServiceError("body keys must be strings")
    _reject_overrides(body)
    extra = set(body) - _GATE_FIELDS
    if extra:
        raise ServiceError(
            "unexpected fields: " + ", ".join(sorted(extra))
        )
    missing = _GATE_FIELDS - set(body)
    if missing:
        raise ServiceError(
            "missing fields: " + ", ".join(sorted(missing))
        )
    outcome = body["outcome"]
    reason_codes = body["reason_codes"]
    if not isinstance(outcome, str) or outcome == "":
        raise ServiceError("outcome must be a non-empty string")
    if not isinstance(reason_codes, list) or not all(
        isinstance(item, str) for item in reason_codes
    ):
        raise ServiceError("reason_codes must be a list of strings")
    for name in (
        "reference_configuration_hash",
        "candidate_configuration_hash",
        "task_set_hash",
    ):
        value = body[name]
        if not isinstance(value, str) or value == "":
            raise ServiceError(f"{name} must be a non-empty string")
    for name in ("reference_protocol_hash", "candidate_protocol_hash"):
        value = body[name]
        if value is not None and (not isinstance(value, str) or value == ""):
            raise ServiceError(f"{name} must be a string or null")
    return _GateDocument(
        outcome=outcome,
        reason_codes=tuple(reason_codes),
        reference_configuration_hash=str(body["reference_configuration_hash"]),
        candidate_configuration_hash=str(body["candidate_configuration_hash"]),
        task_set_hash=str(body["task_set_hash"]),
        reference_protocol_hash=body["reference_protocol_hash"],
        candidate_protocol_hash=body["candidate_protocol_hash"],
    )


def _apply_mode(runtime: RuntimeDependencies, mode: ModeName) -> None:
    setter = getattr(runtime.agent, "set_mode", None)
    if callable(setter):
        setter(mode)


def _persist_episode(store: EpisodeStore, episode: EpisodeResult, task_id: str) -> None:
    store.start_episode(episode.episode, episode.run, task_id)
    ordered = sorted(
        (*episode.model_steps, *episode.tool_steps),
        key=lambda step: step.index,
    )
    for step in ordered:
        store.append_step(episode.episode.episode_id, step)
    store.finish_episode(episode)


def _feed_monitor(
    state: _ServiceState,
    episode: EpisodeResult,
) -> str:
    metadata = state.dependencies.metadata_for(episode)
    try:
        observation = observation_from_episode(
            episode,
            task_metadata=metadata,
        )
    except (MissingEvaluatorOutcome, UndefinedRequirementFraction):
        return "withheld"
    state.dependencies.monitor.update(observation)
    return "updated"


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
        "candidate_episodes_started": snapshot.candidate_episodes_started,
        "candidate_episodes_served": snapshot.candidate_episodes_served,
        "candidate_episodes_failed": snapshot.candidate_episodes_failed,
        "outstanding": snapshot.outstanding,
        "in_flight_at_rollback": snapshot.in_flight_at_rollback,
        "served_before_rollback": snapshot.served_before_rollback,
    }


def _deployment_document(state: _ServiceState) -> dict[str, object]:
    registry = state.dependencies.registry
    production_hash = run_configuration_hash(registry.production)
    candidate_hash = (
        None
        if registry.candidate is None
        else run_configuration_hash(registry.candidate)
    )
    document: dict[str, object] = {
        "admission": state.admission,
        "production_configuration_hash": production_hash,
        "candidate_configuration_hash": candidate_hash,
    }
    controller = state.controller
    if controller is not None:
        document.update(_snapshot_fields(controller.snapshot()))
    return document


def _resolve_config(state: _ServiceState, role: str) -> RunConfiguration:
    registry = state.dependencies.registry
    if role == "production":
        return registry.production
    if registry.candidate is None:
        raise _Conflict("candidate configuration is not registered")
    return registry.candidate


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


def _run_production_episode(
    state: _ServiceState,
    request: _EpisodeRequest,
) -> dict[str, object]:
    config = _resolve_config(state, "production")
    run = new_run_identity(config)
    runtime = state.dependencies.runtime_factory(config)
    _apply_mode(runtime, request.mode)
    episode = run_episode(
        request.task_id,
        config,
        request.mode,
        run=run,
        runtime=runtime,
    )
    _persist_episode(state.dependencies.store, episode, request.task_id)
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
    with state.deployment_lock:
        controller = state.controller
        if controller is None or controller.snapshot().state not in _ACTIVE_CANARY:
            raise _Conflict("no active canary controller")
        controller.begin_candidate_episode()
    reference = registry.production
    candidate = registry.candidate
    reference_run = new_run_identity(reference)
    candidate_run = new_run_identity(candidate)
    runtime = state.dependencies.runtime_factory(candidate)
    _apply_mode(runtime, request.mode)
    pair = run_pair(
        request.task_id,
        reference,
        candidate,
        reference_run=reference_run,
        candidate_run=candidate_run,
        runtime=runtime,
        mode=request.mode,
    )
    with state.deployment_lock:
        if state.controller is None:
            raise _Conflict("no active canary controller")
        state.dependencies.store.append_pair(pair)
        state.controller.observe(pair)
        if request.mode == "execute":
            monitoring_status = _feed_monitor(state, pair.candidate)
        else:
            monitoring_status = "skipped"
    pair_id = pair.reference.episode.pair_id
    return _public_episode_receipt(
        episode_id=None,
        pair_id=pair_id,
        role="candidate",
        mode=request.mode,
        configuration_hash=run_configuration_hash(candidate),
        status=pair.candidate.status,
        monitoring_status=monitoring_status,
    )


def _admit_candidate(state: _ServiceState, gate: _GateDocument) -> dict[str, object]:
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
    production_hash = run_configuration_hash(registry.production)
    candidate_hash = run_configuration_hash(registry.candidate)
    if gate.reference_configuration_hash != production_hash:
        raise _Conflict("gate reference configuration hash mismatch")
    if gate.candidate_configuration_hash != candidate_hash:
        raise _Conflict("gate candidate configuration hash mismatch")
    if gate.task_set_hash != registry.production.task.task_set_hash:
        raise _Conflict("gate task_set_hash mismatch")
    if gate.task_set_hash != registry.candidate.task.task_set_hash:
        raise _Conflict("gate task_set_hash mismatch")
    if gate.reference_protocol_hash != registry.production.protocol_hash:
        raise _Conflict("gate reference protocol hash mismatch")
    if gate.candidate_protocol_hash != registry.candidate.protocol_hash:
        raise _Conflict("gate candidate protocol hash mismatch")
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
        controller.start(gate)
        state.controller = controller
        return _deployment_document(state)


def _request_rollback(state: _ServiceState) -> dict[str, object]:
    with state.deployment_lock:
        state.admission = "rollback_requested"
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
        deadline = time.monotonic() + float(
            dependencies.shutdown_timeout_seconds
        )
        while state.in_flight() > 0 and time.monotonic() < deadline:
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
        gate = _parse_gate(body)
        return _admit_candidate(state, gate)

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
            if episode_request.role == "production":
                return _run_production_episode(state, episode_request)
            return _run_candidate_episode(state, episode_request)
        finally:
            state.release_slot()

    return app
