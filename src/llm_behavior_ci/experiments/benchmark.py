"""Lifecycle benchmark over the production gate, canary, and monitor.

Thresholds and stopping rules come from the protocol lock. Task ids, frozen
baselines, and checkpoints are caller-supplied and are not protocol fields.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from itertools import takewhile
from pathlib import Path
from typing import Any, Literal

from llm_behavior_ci.config import (
    CanarySettings,
    DistributionalMonitorSettings,
    GateSettings,
    MonitorSettings,
    RunConfiguration,
    StreamSettings,
    new_run_identity,
    run_configuration_hash,
)
from llm_behavior_ci.experiments.faults import (
    FaultError,
    FaultSpec,
    HarmLabel,
    apply_fault,
    live_fault_available,
)
from llm_behavior_ci.experiments.protocol import (
    ProtocolError,
    ProtocolLock,
    TaskSelectionAllowance,
    authorize_faulted_candidate,
    authorize_gated_candidate,
    authorize_test_gated_candidate,
    bind_protocol,
    task_selection_allowance_for,
)
from llm_behavior_ci.export import AggregateResults, ExportError, export_public_results
from llm_behavior_ci.lifecycle.canary import CanaryController, CanaryDecision, CanaryRejected
from llm_behavior_ci.lifecycle.monitoring import (
    PLAN_KL_SIGNAL,
    PLAN_QUALITY_SIGNAL,
    FrozenReference,
    ProductionMonitor,
    TaskMetadata,
    build_distributional_monitors,
    observation_from_episode,
    tool_selection_observation_from_episode,
)
from llm_behavior_ci.lifecycle.offline_gate import (
    GateDecision,
    GateExecutionError,
    PlanEvidenceInputs,
    plan_evidence_from_dict as _canonical_plan_evidence_from_dict,
    plan_evidence_to_dict,
    run_offline_gate,
)
from llm_behavior_ci.lifecycle.validation_artifact import ValidationArtifact
from llm_behavior_ci.records import (
    AggregateRecord,
    LifecycleDecision,
    MonitorObservation,
    PairedResult,
    StatisticalEvidence,
    ToolSelectionObservation,
)
from llm_behavior_ci.experiments.schedule import (
    BenchmarkSchedule,
    ScheduledArrival,
    ScheduleError,
    SimulatedClock,
    StreamCounters,
    plan_arrivals,
    run_scheduled_monitor,
    schedule_hash,
)
from llm_behavior_ci.runtime.clock import wall_now
from llm_behavior_ci.runtime.episode import (
    EpisodeRejected,
    PairExecution,
    RuntimeDependencies,
    pair_execution,
    restore_pair_execution,
    run_episode,
    run_pair,
)
from llm_behavior_ci.runtime.factory import RuntimeFactory, StaticRuntimeFactory
from llm_behavior_ci.tasks.selection import TaskSet
from llm_behavior_ci.tasks.streams import generate_stream


class BenchmarkError(ValueError):
    pass


@dataclass(frozen=True)
class BenchmarkProgress:
    """Resume cursor without task identity."""

    fault_version: str
    replicate_seed: int
    tier: str
    stream_index: int

    def __post_init__(self) -> None:
        if self.tier not in {"gate", "canary", "monitor"}:
            raise BenchmarkError("tier must be gate, canary, or monitor")
        if not isinstance(self.stream_index, int) or isinstance(self.stream_index, bool):
            raise BenchmarkError("stream_index must be an integer")
        if self.stream_index < 0:
            raise BenchmarkError("stream_index must be nonnegative")


@dataclass(frozen=True)
class GateTierOutcome:
    status: str
    reason: str | None
    outcome: str | None
    classification: str | None
    compute_seconds: float | None
    agent_execution_seconds: float | None
    detector_compute_seconds: float | None
    public_decision: LifecycleDecision | None
    statistics: tuple[StatisticalEvidence, ...]
    validation_provenance: str | None
    evidence_source: str | None = None


@dataclass(frozen=True)
class CanaryTierOutcome:
    status: str
    reason: str | None
    rollback_delay_episodes: int | None
    candidate_episodes_served: int | None
    candidate_episodes_failed: int | None
    served_before_rollback: int | None
    in_flight_at_rollback: int | None
    compute_seconds: float | None
    agent_execution_seconds: float | None
    detector_compute_seconds: float | None
    public_decision: LifecycleDecision | None


@dataclass(frozen=True)
class MonitorTierOutcome:
    status: str
    reason: str | None
    delay_episodes: int | None
    miss: bool
    false_alarm: bool
    compute_seconds: float | None
    agent_execution_seconds: float | None
    detector_compute_seconds: float | None


@dataclass(frozen=True)
class ReplicateOutcome:
    fault_version: str
    replicate_seed: int
    harmful: bool
    gate: GateTierOutcome
    canary: CanaryTierOutcome
    monitor: MonitorTierOutcome


@dataclass(frozen=True)
class BenchmarkResult:
    status: str
    replicates: tuple[ReplicateOutcome, ...]
    gate_catch_count: int
    gate_false_block_count: int
    mean_canary_rollback_delay: float | None
    candidate_episodes_served: int
    candidate_episodes_failed: int
    mean_monitor_delay: float | None
    monitor_miss_count: int
    monitor_false_alarm_count: int
    canary_not_reached_count: int
    monitor_not_reached_count: int
    compute_seconds: float
    agent_execution_seconds: float
    detector_compute_seconds: float
    wall_seconds: float
    gpu_memory_mib: float | None
    gpu_hours: float | None
    admission_mode: str


def _under_results(path: Path) -> bool:
    for parent in path.resolve().parents:
        if parent.name == "results":
            return True
    return False


def _refuse_results_path(path: Path, label: str) -> None:
    if _under_results(Path(path)):
        raise BenchmarkError(f"{label} must not be under results")


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise BenchmarkError(f"{name} must be an object")
    return value


def _seeds_from_lock(protocol: ProtocolLock) -> tuple[int, ...]:
    raw = protocol.payload.get("seeds")
    if isinstance(raw, bool) or not isinstance(raw, (list, tuple)) or not raw:
        raise BenchmarkError("protocol seeds must be a non-empty sequence of integers")
    seeds: list[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int):
            raise BenchmarkError("protocol seeds must be a non-empty sequence of integers")
        seeds.append(item)
    return tuple(seeds)


def _settings_from_lock(
    protocol: ProtocolLock,
) -> tuple[
    GateSettings,
    CanarySettings,
    MonitorSettings,
    StreamSettings,
    tuple[DistributionalMonitorSettings, ...],
]:
    try:
        gate = GateSettings.from_dict(protocol.payload["gate"])
        canary = CanarySettings.from_dict(protocol.payload["canary"])
        monitor = MonitorSettings.from_dict(protocol.payload["monitor"])
        stream = StreamSettings.from_dict(protocol.payload["stream"])
        distributional_raw = protocol.payload.get("distributional_monitors", [])
        distributional = tuple(
            DistributionalMonitorSettings.from_dict(item)
            for item in distributional_raw
        )
    except Exception as error:
        raise BenchmarkError("protocol lock settings are invalid") from error
    return gate, canary, monitor, stream, distributional


def _templates_by_split(
    protocol: ProtocolLock,
) -> tuple[RunConfiguration, RunConfiguration]:
    train: list[RunConfiguration] = []
    test_normal: list[RunConfiguration] = []
    for configuration in protocol.configurations:
        if configuration.task.split == "train":
            train.append(configuration)
        elif configuration.task.split == "test_normal":
            test_normal.append(configuration)
    if len(train) != 1:
        raise BenchmarkError("protocol lock requires exactly one train configuration")
    if len(test_normal) != 1:
        raise BenchmarkError(
            "protocol lock requires exactly one test_normal configuration"
        )
    return train[0], test_normal[0]


def _harm_label_for(protocol: ProtocolLock, fault_version: str) -> HarmLabel:
    raw = protocol.payload.get("harm_labels")
    if not isinstance(raw, list):
        raise BenchmarkError("protocol harm_labels must be a list")
    for item in raw:
        mapping = _require_mapping(item, "harm label")
        if mapping.get("fault_version") != fault_version:
            continue
        try:
            return HarmLabel(
                fault_version=str(mapping["fault_version"]),
                base_configuration_hash=str(mapping["base_configuration_hash"]),
                candidate_configuration_hash=str(
                    mapping["candidate_configuration_hash"]
                ),
                task_set_hash=str(mapping["task_set_hash"]),
                effect_estimate=float(mapping["effect_estimate"]),
                interval_low=float(mapping["interval_low"]),
                interval_high=float(mapping["interval_high"]),
                margin=float(mapping["margin"]),
                harmful=bool(mapping["harmful"]),
                split=str(mapping["split"]),
                confidence_level=float(mapping["confidence_level"]),
                resamples=int(mapping["resamples"]),
                seed=int(mapping["seed"]),
            )
        except (FaultError, KeyError, TypeError, ValueError) as error:
            raise BenchmarkError("harm label is invalid") from error
    raise BenchmarkError(f"missing harm label for {fault_version}")


def _stream_for_seed(base: StreamSettings, seed: int) -> StreamSettings:
    return StreamSettings(
        split=base.split,
        selection_rule=base.selection_rule,
        selection_seed=base.selection_seed,
        task_set_hash=base.task_set_hash,
        stream_seed=seed,
        arrival_rate_per_second=base.arrival_rate_per_second,
        concurrency=base.concurrency,
        with_replacement=base.with_replacement,
        task_mix_rule=base.task_mix_rule,
    )


def _validate_task_bindings(
    *,
    train_template: RunConfiguration,
    test_normal_template: RunConfiguration,
    train_tasks: TaskSet,
    test_normal_tasks: TaskSet,
    stream: StreamSettings,
) -> None:
    if train_template.task.task_set_hash != train_tasks.task_set_hash:
        raise BenchmarkError("train task_set_hash does not match train_tasks")
    if test_normal_template.task.task_set_hash != test_normal_tasks.task_set_hash:
        raise BenchmarkError(
            "test_normal task_set_hash does not match test_normal_tasks"
        )
    if stream.split != test_normal_tasks.split:
        raise BenchmarkError("stream split must match test_normal_tasks")
    if stream.selection_rule != test_normal_tasks.selection_rule:
        raise BenchmarkError("stream selection_rule must match test_normal_tasks")
    if stream.selection_seed != test_normal_tasks.selection_seed:
        raise BenchmarkError("stream selection_seed must match test_normal_tasks")
    if stream.task_set_hash != test_normal_tasks.task_set_hash:
        raise BenchmarkError("stream task_set_hash must match test_normal_tasks")


def _not_representable_gate(reason: str | None = None) -> GateTierOutcome:
    return GateTierOutcome(
        status="not_representable",
        reason=reason,
        outcome=None,
        classification=None,
        compute_seconds=None,
        agent_execution_seconds=None,
        detector_compute_seconds=None,
        public_decision=None,
        statistics=(),
        validation_provenance=None,
    )


def _not_representable_canary(reason: str | None = None) -> CanaryTierOutcome:
    return CanaryTierOutcome(
        status="not_representable",
        reason=reason,
        rollback_delay_episodes=None,
        candidate_episodes_served=None,
        candidate_episodes_failed=None,
        served_before_rollback=None,
        in_flight_at_rollback=None,
        compute_seconds=None,
        agent_execution_seconds=None,
        detector_compute_seconds=None,
        public_decision=None,
    )


def _not_representable_monitor(reason: str | None = None) -> MonitorTierOutcome:
    return MonitorTierOutcome(
        status="not_representable",
        reason=reason,
        delay_episodes=None,
        miss=False,
        false_alarm=False,
        compute_seconds=None,
        agent_execution_seconds=None,
        detector_compute_seconds=None,
    )


def _unavailable_gate(reason: str) -> GateTierOutcome:
    return GateTierOutcome(
        status="unavailable",
        reason=reason,
        outcome=None,
        classification=None,
        compute_seconds=None,
        agent_execution_seconds=None,
        detector_compute_seconds=None,
        public_decision=None,
        statistics=(),
        validation_provenance=None,
    )


def _unavailable_canary(reason: str) -> CanaryTierOutcome:
    return CanaryTierOutcome(
        status="unavailable",
        reason=reason,
        rollback_delay_episodes=None,
        candidate_episodes_served=None,
        candidate_episodes_failed=None,
        served_before_rollback=None,
        in_flight_at_rollback=None,
        compute_seconds=None,
        agent_execution_seconds=None,
        detector_compute_seconds=None,
        public_decision=None,
    )


def _unavailable_monitor(reason: str) -> MonitorTierOutcome:
    return MonitorTierOutcome(
        status="unavailable",
        reason=reason,
        delay_episodes=None,
        miss=False,
        false_alarm=False,
        compute_seconds=None,
        agent_execution_seconds=None,
        detector_compute_seconds=None,
    )


def _not_reached_canary(reason: str) -> CanaryTierOutcome:
    return CanaryTierOutcome(
        status="not_reached",
        reason=reason,
        rollback_delay_episodes=None,
        candidate_episodes_served=None,
        candidate_episodes_failed=None,
        served_before_rollback=None,
        in_flight_at_rollback=None,
        compute_seconds=None,
        agent_execution_seconds=None,
        detector_compute_seconds=None,
        public_decision=None,
    )


def _not_reached_monitor(reason: str) -> MonitorTierOutcome:
    return MonitorTierOutcome(
        status="not_reached",
        reason=reason,
        delay_episodes=None,
        miss=False,
        false_alarm=False,
        compute_seconds=None,
        agent_execution_seconds=None,
        detector_compute_seconds=None,
    )


def _gate_classification(outcome: str, harmful: bool) -> str | None:
    if outcome != "BLOCK":
        return None
    if harmful:
        return "catch"
    return "false_block"


def _reference_baselines_to_dict(reference: FrozenReference) -> dict[str, object]:
    return {
        "configuration_hash": reference.configuration_hash,
        "baselines": [list(item) for item in reference.baselines],
    }


def _plan_evidence_from_dict(payload: Mapping[str, object]) -> PlanEvidenceInputs:
    try:
        return _canonical_plan_evidence_from_dict(payload)
    except GateExecutionError as error:
        raise BenchmarkError("plan evidence is invalid") from error


def _gate_decision_to_dict(decision: GateDecision) -> dict[str, object]:
    return {
        "outcome": decision.outcome,
        "reason_codes": list(decision.reason_codes),
        "reference_configuration_hash": decision.reference_configuration_hash,
        "candidate_configuration_hash": decision.candidate_configuration_hash,
        "task_set_hash": decision.task_set_hash,
        "reference_protocol_hash": decision.reference_protocol_hash,
        "candidate_protocol_hash": decision.candidate_protocol_hash,
        "thresholds": decision.thresholds.to_dict(),
        "statistics": [item.to_dict() for item in decision.statistics],
        "public_decision": (
            decision.public_decision.to_dict()
            if decision.public_decision is not None
            else None
        ),
        "validation_provenance": decision.validation_provenance,
        "artifact": decision.artifact.to_dict(),
    }


def _gate_decision_from_dict(payload: Mapping[str, object]) -> GateDecision:
    statistics_raw = payload.get("statistics")
    if not isinstance(statistics_raw, list):
        raise BenchmarkError("checkpoint gate statistics are invalid")
    statistics = tuple(StatisticalEvidence.from_dict(item) for item in statistics_raw)
    public_raw = payload.get("public_decision")
    public_decision = (
        None if public_raw is None else LifecycleDecision.from_dict(public_raw)
    )
    reason_codes = payload.get("reason_codes")
    if not isinstance(reason_codes, list):
        raise BenchmarkError("checkpoint gate reason_codes are invalid")
    provenance = payload.get("validation_provenance")
    if not isinstance(provenance, str) or provenance == "":
        raise BenchmarkError("checkpoint gate validation_provenance is invalid")
    artifact_raw = payload.get("artifact")
    if not isinstance(artifact_raw, Mapping):
        raise BenchmarkError("checkpoint gate artifact is invalid")
    try:
        artifact = ValidationArtifact.from_dict(artifact_raw)
    except Exception as error:
        raise BenchmarkError("checkpoint gate artifact is invalid") from error
    return GateDecision(
        outcome=str(payload["outcome"]),
        reason_codes=tuple(str(item) for item in reason_codes),
        reference_configuration_hash=str(payload["reference_configuration_hash"]),
        candidate_configuration_hash=str(payload["candidate_configuration_hash"]),
        task_set_hash=str(payload["task_set_hash"]),
        reference_protocol_hash=(
            None
            if payload.get("reference_protocol_hash") is None
            else str(payload["reference_protocol_hash"])
        ),
        candidate_protocol_hash=(
            None
            if payload.get("candidate_protocol_hash") is None
            else str(payload["candidate_protocol_hash"])
        ),
        thresholds=GateSettings.from_dict(payload["thresholds"]),
        statistics=statistics,
        public_decision=public_decision,
        validation_provenance=provenance,
        artifact=artifact,
    )


def _pair_execution_to_dict(record: PairExecution) -> dict[str, object]:
    return {
        "pair_id": record.pair_id,
        "execution_order": list(record.execution_order),
        "reference_episode_id": record.reference_episode_id,
        "candidate_episode_id": record.candidate_episode_id,
        "reference_seed": record.reference_seed,
        "candidate_seed": record.candidate_seed,
        "reference_run_seed": record.reference_run_seed,
        "candidate_run_seed": record.candidate_run_seed,
        "initial_state_identity": record.initial_state_identity,
    }


def _pair_execution_from_dict(payload: Mapping[str, object]) -> PairExecution:
    order = payload.get("execution_order")
    if not isinstance(order, list) or len(order) != 2:
        raise BenchmarkError("checkpoint pair execution order is invalid")
    return PairExecution(
        pair_id=str(payload["pair_id"]),
        execution_order=(str(order[0]), str(order[1])),
        reference_episode_id=str(payload["reference_episode_id"]),
        candidate_episode_id=str(payload["candidate_episode_id"]),
        reference_seed=int(payload["reference_seed"]),
        candidate_seed=int(payload["candidate_seed"]),
        reference_run_seed=int(payload["reference_run_seed"]),
        candidate_run_seed=int(payload["candidate_run_seed"]),
        initial_state_identity=str(payload["initial_state_identity"]),
    )


def _lifecycle_decision_to_dict(
    decision: LifecycleDecision | None,
) -> dict[str, object] | None:
    if decision is None:
        return None
    return decision.to_dict()


def _lifecycle_decision_from_dict(payload: object) -> LifecycleDecision | None:
    if payload is None:
        return None
    return LifecycleDecision.from_dict(payload)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"replicates": {}}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BenchmarkError("checkpoint is not readable JSON") from error
    if not isinstance(document, dict):
        raise BenchmarkError("checkpoint must be an object")
    replicates = document.get("replicates")
    if replicates is None:
        document["replicates"] = {}
        return document
    if not isinstance(replicates, dict):
        raise BenchmarkError("checkpoint replicates must be an object")
    return document


def _write_checkpoint(path: Path, document: Mapping[str, object]) -> None:
    _refuse_results_path(path, "checkpoint_path")
    if not path.parent.is_dir():
        raise BenchmarkError("checkpoint parent directory does not exist")
    temporary = path.parent / (path.name + ".tmp")
    try:
        text = json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BenchmarkError:
        raise
    except Exception as error:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
        raise BenchmarkError("checkpoint write failed") from error


def _replicate_key(fault_version: str, seed: int) -> str:
    return f"{fault_version}|{seed}"


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _gate_outcome_from_decision(
    decision: GateDecision,
    *,
    harmful: bool,
    compute_seconds: float,
    agent_execution_seconds: float,
    detector_compute_seconds: float,
) -> GateTierOutcome:
    return GateTierOutcome(
        status="completed",
        reason=None,
        outcome=decision.outcome,
        classification=_gate_classification(decision.outcome, harmful),
        compute_seconds=compute_seconds,
        agent_execution_seconds=agent_execution_seconds,
        detector_compute_seconds=detector_compute_seconds,
        public_decision=decision.public_decision,
        statistics=decision.statistics,
        validation_provenance=decision.validation_provenance,
        evidence_source=decision.artifact.evidence_source,
    )


def _canary_outcome_from_decision(
    decision: CanaryDecision,
    *,
    compute_seconds: float,
    agent_execution_seconds: float,
    detector_compute_seconds: float,
) -> CanaryTierOutcome:
    snapshot = decision.snapshot
    if decision.action == "promote":
        return CanaryTierOutcome(
            status="promoted",
            reason=None,
            rollback_delay_episodes=None,
            candidate_episodes_served=snapshot.candidate_episodes_served,
            candidate_episodes_failed=snapshot.candidate_episodes_failed,
            served_before_rollback=None,
            in_flight_at_rollback=None,
            compute_seconds=compute_seconds,
            agent_execution_seconds=agent_execution_seconds,
            detector_compute_seconds=detector_compute_seconds,
            public_decision=decision.public_decision,
        )
    if decision.action == "rollback":
        return CanaryTierOutcome(
            status="completed",
            reason=None,
            rollback_delay_episodes=snapshot.candidate_episodes_served,
            candidate_episodes_served=snapshot.candidate_episodes_served,
            candidate_episodes_failed=snapshot.candidate_episodes_failed,
            served_before_rollback=snapshot.served_before_rollback,
            in_flight_at_rollback=snapshot.in_flight_at_rollback,
            compute_seconds=compute_seconds,
            agent_execution_seconds=agent_execution_seconds,
            detector_compute_seconds=detector_compute_seconds,
            public_decision=decision.public_decision,
        )
    raise BenchmarkError("canary finished without rollback or promote")


def _serialize_canary_outcome(outcome: CanaryTierOutcome) -> dict[str, object]:
    return {
        "status": outcome.status,
        "reason": outcome.reason,
        "rollback_delay_episodes": outcome.rollback_delay_episodes,
        "candidate_episodes_served": outcome.candidate_episodes_served,
        "candidate_episodes_failed": outcome.candidate_episodes_failed,
        "served_before_rollback": outcome.served_before_rollback,
        "in_flight_at_rollback": outcome.in_flight_at_rollback,
        "compute_seconds": outcome.compute_seconds,
        "agent_execution_seconds": outcome.agent_execution_seconds,
        "detector_compute_seconds": outcome.detector_compute_seconds,
        "public_decision": _lifecycle_decision_to_dict(outcome.public_decision),
    }


def _serialize_monitor_outcome(outcome: MonitorTierOutcome) -> dict[str, object]:
    return {
        "status": outcome.status,
        "reason": outcome.reason,
        "delay_episodes": outcome.delay_episodes,
        "miss": outcome.miss,
        "false_alarm": outcome.false_alarm,
        "compute_seconds": outcome.compute_seconds,
        "agent_execution_seconds": outcome.agent_execution_seconds,
        "detector_compute_seconds": outcome.detector_compute_seconds,
    }


def _serialize_replicate(outcome: ReplicateOutcome) -> dict[str, object]:
    return {
        "fault_version": outcome.fault_version,
        "replicate_seed": outcome.replicate_seed,
        "harmful": outcome.harmful,
        "gate": {
            "status": outcome.gate.status,
            "reason": outcome.gate.reason,
            "outcome": outcome.gate.outcome,
            "classification": outcome.gate.classification,
            "compute_seconds": outcome.gate.compute_seconds,
            "agent_execution_seconds": outcome.gate.agent_execution_seconds,
            "detector_compute_seconds": outcome.gate.detector_compute_seconds,
            "public_decision": _lifecycle_decision_to_dict(
                outcome.gate.public_decision
            ),
            "statistics": [item.to_dict() for item in outcome.gate.statistics],
            "validation_provenance": outcome.gate.validation_provenance,
            "evidence_source": outcome.gate.evidence_source,
        },
        "canary": _serialize_canary_outcome(outcome.canary),
        "monitor": _serialize_monitor_outcome(outcome.monitor),
    }


def _deserialize_replicate(payload: Mapping[str, object]) -> ReplicateOutcome:
    gate = _require_mapping(payload["gate"], "gate outcome")
    canary = _require_mapping(payload["canary"], "canary outcome")
    monitor = _require_mapping(payload["monitor"], "monitor outcome")
    statistics_raw = gate.get("statistics")
    if not isinstance(statistics_raw, list):
        statistics_raw = []
    return ReplicateOutcome(
        fault_version=str(payload["fault_version"]),
        replicate_seed=int(payload["replicate_seed"]),
        harmful=bool(payload["harmful"]),
        gate=GateTierOutcome(
            status=str(gate["status"]),
            reason=None if gate.get("reason") is None else str(gate["reason"]),
            outcome=None if gate.get("outcome") is None else str(gate["outcome"]),
            classification=(
                None
                if gate.get("classification") is None
                else str(gate["classification"])
            ),
            compute_seconds=_optional_float(gate.get("compute_seconds")),
            agent_execution_seconds=_optional_float(
                gate.get("agent_execution_seconds")
            ),
            detector_compute_seconds=_optional_float(
                gate.get("detector_compute_seconds")
            ),
            public_decision=_lifecycle_decision_from_dict(gate.get("public_decision")),
            statistics=tuple(
                StatisticalEvidence.from_dict(item) for item in statistics_raw
            ),
            validation_provenance=(
                None
                if gate.get("validation_provenance") is None
                else str(gate["validation_provenance"])
            ),
            evidence_source=(
                None
                if gate.get("evidence_source") is None
                else str(gate["evidence_source"])
            ),
        ),
        canary=CanaryTierOutcome(
            status=str(canary["status"]),
            reason=None if canary.get("reason") is None else str(canary["reason"]),
            rollback_delay_episodes=canary.get("rollback_delay_episodes"),
            candidate_episodes_served=canary.get("candidate_episodes_served"),
            candidate_episodes_failed=canary.get("candidate_episodes_failed"),
            served_before_rollback=canary.get("served_before_rollback"),
            in_flight_at_rollback=canary.get("in_flight_at_rollback"),
            compute_seconds=_optional_float(canary.get("compute_seconds")),
            agent_execution_seconds=_optional_float(
                canary.get("agent_execution_seconds")
            ),
            detector_compute_seconds=_optional_float(
                canary.get("detector_compute_seconds")
            ),
            public_decision=_lifecycle_decision_from_dict(
                canary.get("public_decision")
            ),
        ),
        monitor=MonitorTierOutcome(
            status=str(monitor["status"]),
            reason=None if monitor.get("reason") is None else str(monitor["reason"]),
            delay_episodes=monitor.get("delay_episodes"),
            miss=bool(monitor.get("miss", False)),
            false_alarm=bool(monitor.get("false_alarm", False)),
            compute_seconds=_optional_float(monitor.get("compute_seconds")),
            agent_execution_seconds=_optional_float(
                monitor.get("agent_execution_seconds")
            ),
            detector_compute_seconds=_optional_float(
                monitor.get("detector_compute_seconds")
            ),
        ),
    )


def _monitor_horizon(settings: MonitorSettings) -> int:
    return max(rule.horizon_episodes for rule in settings.stopping_rules)


def _is_complete_replicate(item: ReplicateOutcome) -> bool:
    if item.gate.status in {"not_representable", "unavailable"}:
        return True
    if item.gate.status != "completed":
        return False
    if item.canary.status == "not_reached":
        return item.monitor.status == "not_reached"
    if item.canary.status == "completed":
        return item.monitor.status == "not_reached"
    if item.canary.status == "promoted":
        return item.monitor.status == "completed"
    return False


def _aggregate(
    replicates: Sequence[ReplicateOutcome],
    *,
    status: str,
    wall_seconds: float,
    gpu_memory_mib: float | None,
    gpu_hours: float | None,
    admission_mode: str,
) -> BenchmarkResult:
    scored = [item for item in replicates if _is_complete_replicate(item)]
    gate_catch = 0
    gate_false_block = 0
    rollback_delays: list[int] = []
    served = 0
    failed = 0
    monitor_delays: list[int] = []
    misses = 0
    false_alarms = 0
    canary_not_reached = 0
    monitor_not_reached = 0
    compute = 0.0
    agent_execution = 0.0
    detector_compute = 0.0

    for item in scored:
        if item.gate.classification == "catch":
            gate_catch += 1
        elif item.gate.classification == "false_block":
            gate_false_block += 1
        if item.canary.status == "not_reached":
            canary_not_reached += 1
        if item.monitor.status == "not_reached":
            monitor_not_reached += 1
        if item.canary.status in {"completed", "promoted"}:
            if item.canary.candidate_episodes_served is not None:
                served += item.canary.candidate_episodes_served
            if item.canary.candidate_episodes_failed is not None:
                failed += item.canary.candidate_episodes_failed
        if (
            item.canary.status == "completed"
            and item.canary.rollback_delay_episodes is not None
        ):
            rollback_delays.append(item.canary.rollback_delay_episodes)
        if item.monitor.status == "completed":
            if item.monitor.delay_episodes is not None:
                monitor_delays.append(item.monitor.delay_episodes)
            if item.monitor.miss:
                misses += 1
            if item.monitor.false_alarm:
                false_alarms += 1
        for tier in (item.gate, item.canary, item.monitor):
            if tier.compute_seconds is not None:
                compute += tier.compute_seconds
            if tier.agent_execution_seconds is not None:
                agent_execution += tier.agent_execution_seconds
            if tier.detector_compute_seconds is not None:
                detector_compute += tier.detector_compute_seconds

    return BenchmarkResult(
        status=status,
        replicates=tuple(replicates),
        gate_catch_count=gate_catch,
        gate_false_block_count=gate_false_block,
        mean_canary_rollback_delay=(
            sum(rollback_delays) / len(rollback_delays) if rollback_delays else None
        ),
        candidate_episodes_served=served,
        candidate_episodes_failed=failed,
        mean_monitor_delay=(
            sum(monitor_delays) / len(monitor_delays) if monitor_delays else None
        ),
        monitor_miss_count=misses,
        monitor_false_alarm_count=false_alarms,
        canary_not_reached_count=canary_not_reached,
        monitor_not_reached_count=monitor_not_reached,
        compute_seconds=compute,
        agent_execution_seconds=agent_execution,
        detector_compute_seconds=detector_compute,
        wall_seconds=wall_seconds,
        gpu_memory_mib=gpu_memory_mib,
        gpu_hours=gpu_hours,
        admission_mode=admission_mode,
    )


def _export_aggregates(
    result: BenchmarkResult,
    *,
    train_bound: RunConfiguration,
    test_normal_bound: RunConfiguration,
    train_tasks: TaskSet,
    test_normal_tasks: TaskSet,
) -> AggregateResults:
    train_hash = run_configuration_hash(train_bound)
    test_hash = run_configuration_hash(test_normal_bound)
    aggregates: list[AggregateRecord] = []

    def add(
        *,
        split: str,
        configuration_hash: str,
        task_set_hash: str,
        metric: str,
        value: float,
        episode_count: int,
        task_count: int,
        scenario_count: int,
    ) -> None:
        if episode_count < 1:
            return
        aggregates.append(
            AggregateRecord(
                split=split,
                configuration_hash=configuration_hash,
                task_set_hash=task_set_hash,
                scenario_count=scenario_count,
                task_count=task_count,
                episode_count=episode_count,
                metric=metric,
                value=value,
            )
        )

    scored = [item for item in result.replicates if _is_complete_replicate(item)]
    completed_n = len(scored)
    if completed_n > 0:
        add(
            split="train",
            configuration_hash=train_hash,
            task_set_hash=train_tasks.task_set_hash,
            metric="gate_catch_count",
            value=float(result.gate_catch_count),
            episode_count=completed_n,
            task_count=train_tasks.task_count,
            scenario_count=train_tasks.scenario_count,
        )
        add(
            split="train",
            configuration_hash=train_hash,
            task_set_hash=train_tasks.task_set_hash,
            metric="gate_false_block_count",
            value=float(result.gate_false_block_count),
            episode_count=completed_n,
            task_count=train_tasks.task_count,
            scenario_count=train_tasks.scenario_count,
        )
    if result.canary_not_reached_count > 0:
        add(
            split="test_normal",
            configuration_hash=test_hash,
            task_set_hash=test_normal_tasks.task_set_hash,
            metric="canary_not_reached_count",
            value=float(result.canary_not_reached_count),
            episode_count=result.canary_not_reached_count,
            task_count=test_normal_tasks.task_count,
            scenario_count=test_normal_tasks.scenario_count,
        )
    if result.monitor_not_reached_count > 0:
        add(
            split="test_normal",
            configuration_hash=test_hash,
            task_set_hash=test_normal_tasks.task_set_hash,
            metric="monitor_not_reached_count",
            value=float(result.monitor_not_reached_count),
            episode_count=result.monitor_not_reached_count,
            task_count=test_normal_tasks.task_count,
            scenario_count=test_normal_tasks.scenario_count,
        )
    if result.candidate_episodes_served > 0:
        add(
            split="test_normal",
            configuration_hash=test_hash,
            task_set_hash=test_normal_tasks.task_set_hash,
            metric="candidate_episodes_served",
            value=float(result.candidate_episodes_served),
            episode_count=result.candidate_episodes_served,
            task_count=test_normal_tasks.task_count,
            scenario_count=test_normal_tasks.scenario_count,
        )
    if result.candidate_episodes_failed > 0:
        add(
            split="test_normal",
            configuration_hash=test_hash,
            task_set_hash=test_normal_tasks.task_set_hash,
            metric="candidate_episodes_failed",
            value=float(result.candidate_episodes_failed),
            episode_count=result.candidate_episodes_failed,
            task_count=test_normal_tasks.task_count,
            scenario_count=test_normal_tasks.scenario_count,
        )
    if result.mean_canary_rollback_delay is not None:
        rollback_n = sum(
            1
            for item in scored
            if item.canary.status == "completed"
            and item.canary.rollback_delay_episodes is not None
        )
        add(
            split="test_normal",
            configuration_hash=test_hash,
            task_set_hash=test_normal_tasks.task_set_hash,
            metric="canary_rollback_delay_episodes",
            value=float(result.mean_canary_rollback_delay),
            episode_count=rollback_n,
            task_count=test_normal_tasks.task_count,
            scenario_count=test_normal_tasks.scenario_count,
        )
    if result.mean_monitor_delay is not None:
        alert_n = sum(
            1
            for item in scored
            if item.monitor.status == "completed"
            and item.monitor.delay_episodes is not None
        )
        add(
            split="test_normal",
            configuration_hash=test_hash,
            task_set_hash=test_normal_tasks.task_set_hash,
            metric="monitor_delay_episodes",
            value=float(result.mean_monitor_delay),
            episode_count=alert_n,
            task_count=test_normal_tasks.task_count,
            scenario_count=test_normal_tasks.scenario_count,
        )
    monitor_done = sum(1 for item in scored if item.monitor.status == "completed")
    if monitor_done > 0:
        add(
            split="test_normal",
            configuration_hash=test_hash,
            task_set_hash=test_normal_tasks.task_set_hash,
            metric="monitor_miss_count",
            value=float(result.monitor_miss_count),
            episode_count=monitor_done,
            task_count=test_normal_tasks.task_count,
            scenario_count=test_normal_tasks.scenario_count,
        )
        add(
            split="test_normal",
            configuration_hash=test_hash,
            task_set_hash=test_normal_tasks.task_set_hash,
            metric="monitor_false_alarm_count",
            value=float(result.monitor_false_alarm_count),
            episode_count=monitor_done,
            task_count=test_normal_tasks.task_count,
            scenario_count=test_normal_tasks.scenario_count,
        )
    if result.compute_seconds > 0.0 and completed_n > 0:
        add(
            split="test_normal",
            configuration_hash=test_hash,
            task_set_hash=test_normal_tasks.task_set_hash,
            metric="compute_seconds",
            value=float(result.compute_seconds),
            episode_count=completed_n,
            task_count=test_normal_tasks.task_count,
            scenario_count=test_normal_tasks.scenario_count,
        )

    evidence: list[StatisticalEvidence] = []
    decisions: list[LifecycleDecision] = []
    for item in scored:
        if item.gate.public_decision is not None:
            decisions.append(item.gate.public_decision)
        for stat in item.gate.statistics:
            if stat.sample_size >= 1:
                evidence.append(stat)
        if item.canary.public_decision is not None:
            decisions.append(item.canary.public_decision)
            for stat in item.canary.public_decision.evidence:
                if stat.sample_size >= 1:
                    evidence.append(stat)

    return AggregateResults(
        aggregates=tuple(aggregates),
        evidence=tuple(evidence),
        decisions=tuple(decisions),
    )


def _task_selection_allowance(
    train_template: RunConfiguration,
) -> TaskSelectionAllowance:
    try:
        return task_selection_allowance_for(train_template)
    except ProtocolError as error:
        raise BenchmarkError(str(error)) from error


def _authorize_faulted_test_normal(
    protocol: ProtocolLock,
    template: RunConfiguration,
    task_set: TaskSet,
    fault: FaultSpec,
) -> tuple[RunConfiguration, RunConfiguration]:
    try:
        reference = bind_protocol(template, protocol)
        candidate = apply_fault(reference, fault)
        authorize_faulted_candidate(
            protocol,
            reference,
            candidate,
            fault,
            task_set,
        )
    except (ProtocolError, FaultError) as error:
        raise BenchmarkError(str(error)) from error
    return reference, candidate


def _checkpoint_identity(
    *,
    protocol: ProtocolLock,
    train_bound: RunConfiguration,
    test_normal_bound: RunConfiguration,
    faults: Sequence[FaultSpec],
    plan_evidence: PlanEvidenceInputs,
    reference_baselines: FrozenReference,
    schedule: BenchmarkSchedule | None = None,
) -> dict[str, object]:
    identity: dict[str, object] = {
        "protocol_digest": protocol.digest,
        "train_configuration_hash": run_configuration_hash(train_bound),
        "test_normal_configuration_hash": run_configuration_hash(test_normal_bound),
        "fault_versions": [fault.fault_version for fault in faults],
        "plan_evidence": plan_evidence_to_dict(plan_evidence),
        "reference_baselines": _reference_baselines_to_dict(reference_baselines),
    }
    if schedule is not None:
        identity["schedule_hash"] = schedule_hash(schedule)
    return identity


def _validate_checkpoint_identity(
    document: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    stored = document.get("identity")
    if stored is None:
        return
    if not isinstance(stored, Mapping):
        raise BenchmarkError("checkpoint identity is invalid")
    for key, value in expected.items():
        if stored.get(key) != value:
            raise BenchmarkError(
                "checkpoint identity does not match protocol, configuration, or fault"
            )


def _start_canary_controller(
    controller: CanaryController,
    *,
    gate_decision: GateDecision,
    reference: RunConfiguration,
    candidate: RunConfiguration,
    allowance: TaskSelectionAllowance,
    admission_mode: Literal["release", "test"],
) -> None:
    """Start the canary from this run's own, just-computed gate decision.

    ``admission_mode="test"`` calls ``authorize_test_gated_candidate``,
    which also accepts ``synthetic_fixture``-sourced evidence; that is
    correct for the CPU synthetic path, where ``gate_decision.artifact``
    was produced a few calls up this same stack from a
    ``synthetic_fixture``-provenance ``PlanEvidenceInputs``.
    ``admission_mode="release"`` calls the strict
    ``authorize_gated_candidate``, which refuses synthetic evidence, and is
    the mode a caller running this benchmark against a real, live-runtime
    gate decision must choose.
    """

    authorize = (
        authorize_test_gated_candidate
        if admission_mode == "test"
        else authorize_gated_candidate
    )
    try:
        admission = authorize(
            gate_decision.artifact,
            reference,
            candidate,
            allowance,
        )
    except ProtocolError as error:
        raise BenchmarkError(str(error)) from error
    starter = getattr(controller, "start_from_admission", None)
    if callable(starter):
        try:
            starter(admission)
        except CanaryRejected as error:
            raise BenchmarkError(str(error)) from error
        return
    served_reference = run_configuration_hash(reference)
    served_candidate = run_configuration_hash(candidate)
    if (
        served_reference == gate_decision.reference_configuration_hash
        and served_candidate == gate_decision.candidate_configuration_hash
    ):
        try:
            controller.start(gate_decision)
        except CanaryRejected as error:
            raise BenchmarkError(str(error)) from error
        return
    raise BenchmarkError(
        "CanaryController.start_from_admission is required when served "
        "configuration hashes differ from the gate"
    )


def run_lifecycle_benchmark(
    protocol: ProtocolLock,
    faults: Sequence[FaultSpec],
    *,
    runtime: RuntimeDependencies | None = None,
    train_tasks: TaskSet,
    test_normal_tasks: TaskSet,
    reference_baselines: FrozenReference,
    checkpoint_path: Path,
    plan_evidence: PlanEvidenceInputs,
    admission_mode: Literal["release", "test"],
    export_path: Path | None = None,
    should_interrupt: Callable[[BenchmarkProgress], bool] | None = None,
    candidate_runtime: RuntimeDependencies | None = None,
    gpu_memory_mib: float | None = None,
    gpu_hours: float | None = None,
    runtime_factory: RuntimeFactory | None = None,
    schedule: BenchmarkSchedule | None = None,
    difficulty_for: Callable[[str], int | None] | None = None,
) -> BenchmarkResult:
    """Run the three-tier lifecycle benchmark for each fault and lock seed.

    Runtimes come from ``runtime_factory`` (one per configuration, mode,
    and role) or, for injected CPU runtimes, from ``runtime`` and
    ``candidate_runtime``. With ``schedule``, the canary assigns the
    frozen fraction over the full arrival stream and the monitor serves a
    healthy prefix, then the faulted configuration from onset, on a
    simulated clock; the schedule hash is part of the checkpoint identity.
    Without it, the legacy path runs every arrival as a pair and serves the
    faulted configuration from the first arrival.
    """

    if not isinstance(protocol, ProtocolLock):
        raise BenchmarkError("run_lifecycle_benchmark requires a protocol lock")
    if (runtime is None) == (runtime_factory is None):
        raise BenchmarkError(
            "run_lifecycle_benchmark requires exactly one of runtime or runtime_factory"
        )
    if runtime is not None and not isinstance(runtime, RuntimeDependencies):
        raise BenchmarkError("run_lifecycle_benchmark requires runtime dependencies")
    if candidate_runtime is not None and not isinstance(
        candidate_runtime, RuntimeDependencies
    ):
        raise BenchmarkError("candidate_runtime must be RuntimeDependencies when set")
    if runtime_factory is not None and not isinstance(runtime_factory, RuntimeFactory):
        raise BenchmarkError("runtime_factory must be a RuntimeFactory")
    if schedule is not None and not isinstance(schedule, BenchmarkSchedule):
        raise BenchmarkError("schedule must be a BenchmarkSchedule")
    factory: RuntimeFactory = (
        runtime_factory
        if runtime_factory is not None
        else StaticRuntimeFactory(reference=runtime, candidate=candidate_runtime)
    )
    clock = wall_now if runtime is None else runtime.clock
    if not isinstance(plan_evidence, PlanEvidenceInputs):
        raise BenchmarkError("run_lifecycle_benchmark requires plan evidence inputs")
    if admission_mode not in {"release", "test"}:
        raise BenchmarkError("admission_mode must be release or test")
    if not isinstance(train_tasks, TaskSet) or not isinstance(
        test_normal_tasks, TaskSet
    ):
        raise BenchmarkError("task sets must be TaskSet instances")
    if not isinstance(reference_baselines, FrozenReference):
        raise BenchmarkError("reference_baselines must be FrozenReference")
    checkpoint = Path(checkpoint_path)
    _refuse_results_path(checkpoint, "checkpoint_path")
    if export_path is not None:
        _refuse_results_path(Path(export_path), "export_path")

    (
        gate_settings,
        canary_settings,
        monitor_settings,
        stream_settings,
        distributional_monitor_settings,
    ) = _settings_from_lock(protocol)
    if (
        reference_baselines.configuration_hash
        != monitor_settings.reference_configuration_hash
    ):
        raise BenchmarkError(
            "reference_baselines.configuration_hash must equal the lock monitor hash"
        )

    train_template, test_normal_template = _templates_by_split(protocol)
    _validate_task_bindings(
        train_template=train_template,
        test_normal_template=test_normal_template,
        train_tasks=train_tasks,
        test_normal_tasks=test_normal_tasks,
        stream=stream_settings,
    )

    seeds = _seeds_from_lock(protocol)
    document = _load_checkpoint(checkpoint)
    train_bound_for_export = bind_protocol(train_template, protocol)
    test_normal_bound_for_export = bind_protocol(test_normal_template, protocol)
    identity = _checkpoint_identity(
        protocol=protocol,
        train_bound=train_bound_for_export,
        test_normal_bound=test_normal_bound_for_export,
        faults=faults,
        plan_evidence=plan_evidence,
        reference_baselines=reference_baselines,
        schedule=schedule,
    )
    if schedule is not None:
        try:
            scheduled_arrivals = plan_arrivals(test_normal_tasks, schedule)
        except ScheduleError as error:
            raise BenchmarkError(str(error)) from error
    _validate_checkpoint_identity(document, identity)
    document["identity"] = identity
    replicates_doc: dict[str, Any] = document.setdefault("replicates", {})
    outcomes: list[ReplicateOutcome] = []
    interrupted = False
    wall_started = time.perf_counter()
    allowance = _task_selection_allowance(train_template)

    for fault in faults:
        if not isinstance(fault, FaultSpec):
            raise BenchmarkError("faults must contain FaultSpec")
        label = _harm_label_for(protocol, fault.fault_version)
        for seed in seeds:
            if interrupted:
                break
            key = _replicate_key(fault.fault_version, seed)
            stored = replicates_doc.get(key)
            if not isinstance(stored, dict):
                stored = {}
                replicates_doc[key] = stored

            finished = stored.get("outcome")
            if isinstance(finished, dict):
                outcomes.append(_deserialize_replicate(finished))
                continue

            if not fault.representable:
                reason = (
                    None
                    if fault.schema_request is None
                    else str(fault.schema_request)
                )
                outcome = ReplicateOutcome(
                    fault_version=fault.fault_version,
                    replicate_seed=seed,
                    harmful=label.harmful,
                    gate=_not_representable_gate(reason),
                    canary=_not_representable_canary(reason),
                    monitor=_not_representable_monitor(reason),
                )
                outcomes.append(outcome)
                stored["outcome"] = _serialize_replicate(outcome)
                _write_checkpoint(checkpoint, document)
                continue

            availability = live_fault_available(fault, train_bound_for_export)
            if not availability.available:
                reason = (
                    "live fault unavailable"
                    if availability.reason is None
                    else availability.reason
                )
                outcome = ReplicateOutcome(
                    fault_version=fault.fault_version,
                    replicate_seed=seed,
                    harmful=label.harmful,
                    gate=_unavailable_gate(reason),
                    canary=_unavailable_canary(reason),
                    monitor=_unavailable_monitor(reason),
                )
                outcomes.append(outcome)
                stored["outcome"] = _serialize_replicate(outcome)
                _write_checkpoint(checkpoint, document)
                continue

            progress = BenchmarkProgress(
                fault_version=fault.fault_version,
                replicate_seed=seed,
                tier="gate",
                stream_index=0,
            )
            if should_interrupt is not None and should_interrupt(progress):
                interrupted = True
                _write_checkpoint(checkpoint, document)
                break

            gate_outcome, gate_decision = _run_or_resume_gate(
                protocol=protocol,
                fault=fault,
                train_template=train_template,
                train_tasks=train_tasks,
                gate_settings=gate_settings,
                factory=factory,
                plan_evidence=plan_evidence,
                harmful=label.harmful,
                stored=stored,
                checkpoint=checkpoint,
                document=document,
            )
            _write_checkpoint(checkpoint, document)
            if should_interrupt is not None and should_interrupt(progress):
                interrupted = True
                break

            if gate_decision.outcome != "PASS":
                replicate = ReplicateOutcome(
                    fault_version=fault.fault_version,
                    replicate_seed=seed,
                    harmful=label.harmful,
                    gate=gate_outcome,
                    canary=_not_reached_canary("gate_block"),
                    monitor=_not_reached_monitor("gate_block"),
                )
                outcomes.append(replicate)
                stored["outcome"] = _serialize_replicate(replicate)
                _write_checkpoint(checkpoint, document)
                continue

            progress = BenchmarkProgress(
                fault_version=fault.fault_version,
                replicate_seed=seed,
                tier="canary",
                stream_index=0,
            )
            if should_interrupt is not None and should_interrupt(progress):
                interrupted = True
                _write_checkpoint(checkpoint, document)
                break

            canary_result = _run_or_resume_canary(
                protocol=protocol,
                fault=fault,
                test_normal_template=test_normal_template,
                test_normal_tasks=test_normal_tasks,
                canary_settings=canary_settings,
                stream_settings=stream_settings,
                factory=factory,
                clock=clock if schedule is None else SimulatedClock(schedule.clock_start),
                scheduled_arrivals=None if schedule is None else scheduled_arrivals,
                gate_decision=gate_decision,
                allowance=allowance,
                seed=seed,
                stored=stored,
                checkpoint=checkpoint,
                document=document,
                should_interrupt=should_interrupt,
                fault_version=fault.fault_version,
                admission_mode=admission_mode,
            )
            if canary_result["interrupted"]:
                interrupted = True
                outcomes.append(
                    ReplicateOutcome(
                        fault_version=fault.fault_version,
                        replicate_seed=seed,
                        harmful=label.harmful,
                        gate=gate_outcome,
                        canary=canary_result["outcome"],
                        monitor=MonitorTierOutcome(
                            status="interrupted",
                            reason=None,
                            delay_episodes=None,
                            miss=False,
                            false_alarm=False,
                            compute_seconds=None,
                            agent_execution_seconds=None,
                            detector_compute_seconds=None,
                        ),
                    )
                )
                break

            canary_outcome: CanaryTierOutcome = canary_result["outcome"]
            if canary_outcome.status != "promoted":
                replicate = ReplicateOutcome(
                    fault_version=fault.fault_version,
                    replicate_seed=seed,
                    harmful=label.harmful,
                    gate=gate_outcome,
                    canary=canary_outcome,
                    monitor=_not_reached_monitor(
                        "canary_horizon_exhausted"
                        if canary_outcome.status == "horizon_exhausted"
                        else "canary_rollback"
                    ),
                )
                outcomes.append(replicate)
                stored["outcome"] = _serialize_replicate(replicate)
                _write_checkpoint(checkpoint, document)
                continue

            progress = BenchmarkProgress(
                fault_version=fault.fault_version,
                replicate_seed=seed,
                tier="monitor",
                stream_index=0,
            )
            if should_interrupt is not None and should_interrupt(progress):
                interrupted = True
                _write_checkpoint(checkpoint, document)
                outcomes.append(
                    ReplicateOutcome(
                        fault_version=fault.fault_version,
                        replicate_seed=seed,
                        harmful=label.harmful,
                        gate=gate_outcome,
                        canary=canary_outcome,
                        monitor=MonitorTierOutcome(
                            status="interrupted",
                            reason=None,
                            delay_episodes=None,
                            miss=False,
                            false_alarm=False,
                            compute_seconds=None,
                            agent_execution_seconds=None,
                            detector_compute_seconds=None,
                        ),
                    )
                )
                break

            monitor_runner = (
                _run_or_resume_monitor
                if schedule is None
                else partial(
                    _run_or_resume_scheduled_monitor,
                    schedule=schedule,
                    arrivals=scheduled_arrivals,
                    difficulty_for=difficulty_for,
                )
            )
            monitor_result = monitor_runner(
                protocol=protocol,
                fault=fault,
                test_normal_template=test_normal_template,
                test_normal_tasks=test_normal_tasks,
                monitor_settings=monitor_settings,
                distributional_monitor_settings=distributional_monitor_settings,
                stream_settings=stream_settings,
                reference_baselines=reference_baselines,
                factory=factory,
                clock=clock,
                seed=seed,
                harmful=label.harmful,
                stored=stored,
                checkpoint=checkpoint,
                document=document,
                should_interrupt=should_interrupt,
                fault_version=fault.fault_version,
            )
            if monitor_result["interrupted"]:
                interrupted = True
                outcomes.append(
                    ReplicateOutcome(
                        fault_version=fault.fault_version,
                        replicate_seed=seed,
                        harmful=label.harmful,
                        gate=gate_outcome,
                        canary=canary_outcome,
                        monitor=monitor_result["outcome"],
                    )
                )
                break

            replicate = ReplicateOutcome(
                fault_version=fault.fault_version,
                replicate_seed=seed,
                harmful=label.harmful,
                gate=gate_outcome,
                canary=canary_outcome,
                monitor=monitor_result["outcome"],
            )
            outcomes.append(replicate)
            stored["outcome"] = _serialize_replicate(replicate)
            _write_checkpoint(checkpoint, document)

    status = "interrupted" if interrupted else "completed"
    wall_seconds = time.perf_counter() - wall_started
    result = _aggregate(
        outcomes,
        status=status,
        wall_seconds=wall_seconds,
        gpu_memory_mib=gpu_memory_mib,
        gpu_hours=gpu_hours,
        admission_mode=admission_mode,
    )
    if status == "completed" and export_path is not None:
        try:
            export_public_results(
                _export_aggregates(
                    result,
                    train_bound=train_bound_for_export,
                    test_normal_bound=test_normal_bound_for_export,
                    train_tasks=train_tasks,
                    test_normal_tasks=test_normal_tasks,
                ),
                output_path=Path(export_path),
            )
        except ExportError as error:
            raise BenchmarkError("public export failed") from error
    return result


def _role_runtimes(
    factory: RuntimeFactory,
    reference: RunConfiguration,
    candidate: RunConfiguration,
    *,
    mode: str,
) -> tuple[RuntimeDependencies, RuntimeDependencies]:
    try:
        factory.preflight({"reference": reference, "candidate": candidate})
        return (
            factory(reference, mode=mode, role="reference"),
            factory(candidate, mode=mode, role="candidate"),
        )
    except EpisodeRejected as error:
        raise BenchmarkError(str(error)) from error


def _run_or_resume_scheduled_monitor(
    *,
    schedule: BenchmarkSchedule,
    arrivals: Sequence[ScheduledArrival],
    difficulty_for: Callable[[str], int | None] | None,
    protocol: ProtocolLock,
    fault: FaultSpec,
    test_normal_template: RunConfiguration,
    test_normal_tasks: TaskSet,
    monitor_settings: MonitorSettings,
    distributional_monitor_settings: tuple[DistributionalMonitorSettings, ...],
    stream_settings: StreamSettings,
    reference_baselines: FrozenReference,
    factory: RuntimeFactory,
    clock: Callable[[], datetime],
    seed: int,
    harmful: bool,
    stored: dict[str, Any],
    checkpoint: Path,
    document: dict[str, Any],
    should_interrupt: Callable[[BenchmarkProgress], bool] | None,
    fault_version: str,
) -> dict[str, Any]:
    """The monitor tier on the explicit schedule: healthy prefix, then onset.

    Uses ``run_scheduled_monitor``, the same loop a dev rehearsal runs.
    ``delay_episodes`` counts arrivals from onset to the first post-onset
    alert; healthy-prefix alarms are stored apart and never count as
    detection. A miss is a harmful fault without a post-onset alert; a
    false alarm is a post-onset alert on a non-harmful fault.
    """

    del stream_settings
    healthy, faulted = _authorize_faulted_test_normal(
        protocol, test_normal_template, test_normal_tasks, fault
    )
    monitor_blob = stored.setdefault("monitor", {})
    if isinstance(monitor_blob.get("final"), dict):
        return {
            "interrupted": False,
            "outcome": _deserialize_monitor_outcome(monitor_blob["final"]),
        }
    simulated = SimulatedClock(schedule.clock_start)
    monitor = ProductionMonitor(
        monitor_settings,
        reference_baselines,
        clock=simulated,
        period_id=f"benchmark-{reference_baselines.configuration_hash[:16]}-{seed}",
    )
    distributional = build_distributional_monitors(
        distributional_monitor_settings,
        reference_configuration_hash=reference_baselines.configuration_hash,
        clock=simulated,
        dedup_seconds=0.0,
    )
    started = time.perf_counter()

    def runtime_for(configuration: RunConfiguration) -> RuntimeDependencies:
        role = "reference" if configuration == healthy else "candidate"
        return factory(configuration, mode="execute", role=role)

    def interrupt(index: int) -> bool:
        if should_interrupt is None:
            return False
        return should_interrupt(
            BenchmarkProgress(
                fault_version=fault_version,
                replicate_seed=seed,
                tier="monitor",
                stream_index=index,
            )
        )

    try:
        result = run_scheduled_monitor(
            schedule=schedule,
            arrivals=arrivals,
            healthy=healthy,
            faulted=faulted,
            runtime_for=runtime_for,
            monitor=monitor,
            distributional=distributional,
            clock=simulated,
            state=monitor_blob,
            persist=lambda: _write_checkpoint(checkpoint, document),
            difficulty_for=difficulty_for,
            should_interrupt=interrupt,
        )
    except (ScheduleError, EpisodeRejected) as error:
        raise BenchmarkError(str(error)) from error
    compute_seconds = float(monitor_blob.get("compute_seconds", 0.0)) + (
        time.perf_counter() - started
    )
    monitor_blob["compute_seconds"] = compute_seconds
    monitor_blob["counters"] = result.counters.to_dict()
    monitor_blob["healthy_prefix_alarms"] = result.healthy_prefix_alarms
    detected = result.detected()
    outcome = MonitorTierOutcome(
        status="interrupted" if result.status == "interrupted" else "completed",
        reason=None,
        delay_episodes=result.post_onset_delay_episodes,
        miss=bool(result.status == "completed" and harmful and not detected),
        false_alarm=bool(result.status == "completed" and not harmful and detected),
        compute_seconds=compute_seconds,
        agent_execution_seconds=None,
        detector_compute_seconds=None,
    )
    if result.status == "completed":
        monitor_blob["final"] = _serialize_monitor_outcome(outcome)
    _write_checkpoint(checkpoint, document)
    return {"interrupted": result.status == "interrupted", "outcome": outcome}


def _deserialize_monitor_outcome(final: Mapping[str, Any]) -> MonitorTierOutcome:
    return MonitorTierOutcome(
        status=str(final["status"]),
        reason=final.get("reason"),
        delay_episodes=final.get("delay_episodes"),
        miss=bool(final.get("miss", False)),
        false_alarm=bool(final.get("false_alarm", False)),
        compute_seconds=_optional_float(final.get("compute_seconds")),
        agent_execution_seconds=_optional_float(final.get("agent_execution_seconds")),
        detector_compute_seconds=_optional_float(final.get("detector_compute_seconds")),
    )


def _run_or_resume_gate(
    *,
    protocol: ProtocolLock,
    fault: FaultSpec,
    train_template: RunConfiguration,
    train_tasks: TaskSet,
    gate_settings: GateSettings,
    factory: RuntimeFactory,
    plan_evidence: PlanEvidenceInputs,
    harmful: bool,
    stored: dict[str, Any],
    checkpoint: Path,
    document: dict[str, Any],
) -> tuple[GateTierOutcome, GateDecision]:
    gate_blob = stored.get("gate")
    if isinstance(gate_blob, dict) and "decision" in gate_blob:
        decision = _gate_decision_from_dict(
            _require_mapping(gate_blob["decision"], "gate decision")
        )
        if decision.validation_provenance != plan_evidence.validation_provenance:
            raise BenchmarkError(
                "checkpoint gate validation_provenance does not match plan evidence"
            )
        compute_seconds = float(gate_blob.get("compute_seconds", 0.0))
        agent_seconds = float(
            gate_blob.get("agent_execution_seconds", compute_seconds)
        )
        detector_seconds = float(gate_blob.get("detector_compute_seconds", 0.0))
        return (
            _gate_outcome_from_decision(
                decision,
                harmful=harmful,
                compute_seconds=compute_seconds,
                agent_execution_seconds=agent_seconds,
                detector_compute_seconds=detector_seconds,
            ),
            decision,
        )

    try:
        reference = bind_protocol(train_template, protocol)
        candidate = apply_fault(reference, fault)
    except (ProtocolError, FaultError) as error:
        raise BenchmarkError(str(error)) from error

    started = time.perf_counter()
    try:
        reference_runtime, candidate_runtime = _role_runtimes(
            factory, reference, candidate, mode="plan"
        )
        decision = run_offline_gate(
            reference,
            candidate,
            train_tasks,
            settings=gate_settings,
            runtime=reference_runtime,
            plan_evidence=plan_evidence,
            candidate_runtime=candidate_runtime,
        )
    except GateExecutionError as error:
        raise BenchmarkError(str(error)) from error
    compute_seconds = time.perf_counter() - started
    stored["gate"] = {
        "decision": _gate_decision_to_dict(decision),
        "compute_seconds": compute_seconds,
        "agent_execution_seconds": compute_seconds,
        "detector_compute_seconds": 0.0,
        "fault_version": fault.fault_version,
        "reference_configuration_hash": run_configuration_hash(reference),
        "candidate_configuration_hash": run_configuration_hash(candidate),
    }
    _write_checkpoint(checkpoint, document)
    return (
        _gate_outcome_from_decision(
            decision,
            harmful=harmful,
            compute_seconds=compute_seconds,
            agent_execution_seconds=compute_seconds,
            detector_compute_seconds=0.0,
        ),
        decision,
    )


def _run_or_resume_canary(
    *,
    protocol: ProtocolLock,
    fault: FaultSpec,
    test_normal_template: RunConfiguration,
    test_normal_tasks: TaskSet,
    canary_settings: CanarySettings,
    stream_settings: StreamSettings,
    factory: RuntimeFactory,
    clock: Callable[[], datetime],
    scheduled_arrivals: Sequence[ScheduledArrival] | None,
    gate_decision: GateDecision,
    allowance: TaskSelectionAllowance,
    seed: int,
    stored: dict[str, Any],
    checkpoint: Path,
    document: dict[str, Any],
    should_interrupt: Callable[[BenchmarkProgress], bool] | None,
    fault_version: str,
    admission_mode: Literal["release", "test"],
) -> dict[str, Any]:
    reference, candidate = _authorize_faulted_test_normal(
        protocol, test_normal_template, test_normal_tasks, fault
    )

    canary_blob = stored.setdefault("canary", {})
    if not isinstance(canary_blob, dict):
        raise BenchmarkError("checkpoint canary section is invalid")
    items_raw = canary_blob.get("items")
    if not isinstance(items_raw, list):
        items_raw = []
        canary_blob["items"] = items_raw

    if isinstance(canary_blob.get("final"), dict):
        final = canary_blob["final"]
        outcome = CanaryTierOutcome(
            status=str(final["status"]),
            reason=final.get("reason"),
            rollback_delay_episodes=final.get("rollback_delay_episodes"),
            candidate_episodes_served=final.get("candidate_episodes_served"),
            candidate_episodes_failed=final.get("candidate_episodes_failed"),
            served_before_rollback=final.get("served_before_rollback"),
            in_flight_at_rollback=final.get("in_flight_at_rollback"),
            compute_seconds=_optional_float(final.get("compute_seconds")),
            agent_execution_seconds=_optional_float(
                final.get("agent_execution_seconds")
            ),
            detector_compute_seconds=_optional_float(
                final.get("detector_compute_seconds")
            ),
            public_decision=_lifecycle_decision_from_dict(
                final.get("public_decision")
            ),
        )
        return {"interrupted": False, "outcome": outcome}

    controller = CanaryController(
        reference,
        candidate,
        settings=canary_settings,
        clock=clock,
    )
    _start_canary_controller(
        controller,
        gate_decision=gate_decision,
        reference=reference,
        candidate=candidate,
        allowance=allowance,
        admission_mode=admission_mode,
    )

    started = time.perf_counter()
    compute_base = float(canary_blob.get("compute_seconds", 0.0))
    agent_base = float(canary_blob.get("agent_execution_seconds", 0.0))
    detector_base = float(canary_blob.get("detector_compute_seconds", 0.0))
    agent_delta = 0.0
    detector_delta = 0.0
    stored_indexes = {
        int(item["stream_index"])
        for item in items_raw
        if isinstance(item, dict) and "stream_index" in item
    }
    for item in sorted(items_raw, key=lambda row: int(row["stream_index"])):
        if not isinstance(item, dict):
            raise BenchmarkError("checkpoint canary item is invalid")
        pair = PairedResult.from_dict(item["pair"])
        execution = _pair_execution_from_dict(
            _require_mapping(item["pair_execution"], "pair_execution")
        )
        restore_pair_execution(execution)
        decided = canary_blob.get("decisions", {}).get(str(item["stream_index"]))
        if isinstance(decided, dict) and callable(getattr(clock, "advance_to", None)):
            clock.advance_to(datetime.fromisoformat(str(decided["simulated_at"])))
        controller.begin_candidate_episode()
        detector_started = time.perf_counter()
        decision = controller.observe(pair)
        detector_delta += time.perf_counter() - detector_started
        if decision.action in {"rollback", "promote"}:
            compute_seconds = compute_base + (time.perf_counter() - started)
            agent_seconds = agent_base + agent_delta
            detector_seconds = detector_base + detector_delta
            outcome = _canary_outcome_from_decision(
                decision,
                compute_seconds=compute_seconds,
                agent_execution_seconds=agent_seconds,
                detector_compute_seconds=detector_seconds,
            )
            canary_blob["final"] = _serialize_canary_outcome(outcome)
            canary_blob["compute_seconds"] = compute_seconds
            canary_blob["agent_execution_seconds"] = agent_seconds
            canary_blob["detector_compute_seconds"] = detector_seconds
            _write_checkpoint(checkpoint, document)
            return {"interrupted": False, "outcome": outcome}

    reference_run = new_run_identity(reference)
    candidate_run = new_run_identity(candidate)
    if "reference_run" in canary_blob and "candidate_run" in canary_blob:
        from llm_behavior_ci.config import RunIdentity

        reference_run = RunIdentity.from_dict(canary_blob["reference_run"])
        candidate_run = RunIdentity.from_dict(canary_blob["candidate_run"])
    else:
        canary_blob["reference_run"] = reference_run.to_dict()
        canary_blob["candidate_run"] = candidate_run.to_dict()
        _write_checkpoint(checkpoint, document)

    try:
        reference_runtime, candidate_runtime = _role_runtimes(
            factory, reference, candidate, mode="execute"
        )
    except EpisodeRejected as error:
        raise BenchmarkError(str(error)) from error
    counters = canary_blob.setdefault("counters", StreamCounters().to_dict())
    if scheduled_arrivals is None:
        stream = _stream_for_seed(stream_settings, seed)
        horizon = canary_settings.stopping_rule.horizon_episodes
        arrivals_iter: Any = takewhile(
            lambda arrival: arrival.index < horizon,
            generate_stream(test_normal_tasks, stream),
        )
    else:
        horizon = len(scheduled_arrivals)
        arrivals_iter = iter(scheduled_arrivals)

    for arrival in arrivals_iter:
        if arrival.index >= horizon:
            break
        if arrival.index in stored_indexes:
            continue
        if scheduled_arrivals is not None:
            decided = canary_blob.setdefault("decisions", {})
            decided[str(arrival.index)] = arrival.decision_dict()
            if str(arrival.index) not in canary_blob.setdefault("counted", []):
                canary_blob["counted"].append(str(arrival.index))
                counters["arrivals"] += 1
                if not arrival.canary_assigned:
                    counters["production_only_arrivals"] += 1
            if not arrival.canary_assigned:
                _write_checkpoint(checkpoint, document)
                continue
        progress = BenchmarkProgress(
            fault_version=fault_version,
            replicate_seed=seed,
            tier="canary",
            stream_index=arrival.index,
        )
        if should_interrupt is not None and should_interrupt(progress):
            compute_seconds = compute_base + (time.perf_counter() - started)
            canary_blob["compute_seconds"] = compute_seconds
            canary_blob["agent_execution_seconds"] = agent_base + agent_delta
            canary_blob["detector_compute_seconds"] = detector_base + detector_delta
            _write_checkpoint(checkpoint, document)
            snap = controller.snapshot()
            return {
                "interrupted": True,
                "outcome": CanaryTierOutcome(
                    status="interrupted",
                    reason=None,
                    rollback_delay_episodes=None,
                    candidate_episodes_served=snap.candidate_episodes_served,
                    candidate_episodes_failed=snap.candidate_episodes_failed,
                    served_before_rollback=None,
                    in_flight_at_rollback=None,
                    compute_seconds=compute_seconds,
                    agent_execution_seconds=agent_base + agent_delta,
                    detector_compute_seconds=detector_base + detector_delta,
                    public_decision=None,
                ),
            }

        if scheduled_arrivals is not None and callable(getattr(clock, "advance_to", None)):
            clock.advance_to(arrival.simulated_at)
        controller.begin_candidate_episode()
        agent_started = time.perf_counter()
        pair = run_pair(
            arrival.task_id,
            reference,
            candidate,
            reference_run=reference_run,
            candidate_run=candidate_run,
            runtime=reference_runtime,
            mode="execute",
            scenario_id=arrival.scenario_id,
            candidate_runtime=candidate_runtime,
        )
        agent_delta += time.perf_counter() - agent_started
        counters["candidate_exposures"] += 1
        counters["paired_outcomes"] += 1
        if (
            pair.reference.evaluator_outcome is not None
            and pair.candidate.evaluator_outcome is not None
        ):
            counters["completed_evaluator_outcomes"] += 1
        else:
            counters["missing_evaluator_outcomes"] += 1
        detector_started = time.perf_counter()
        decision = controller.observe(pair)
        detector_delta += time.perf_counter() - detector_started
        execution = pair_execution(pair)
        items_raw.append(
            {
                "stream_index": arrival.index,
                "pair": pair.to_dict(),
                "pair_execution": _pair_execution_to_dict(execution),
                "action": decision.action,
            }
        )
        compute_seconds = compute_base + (time.perf_counter() - started)
        canary_blob["compute_seconds"] = compute_seconds
        canary_blob["agent_execution_seconds"] = agent_base + agent_delta
        canary_blob["detector_compute_seconds"] = detector_base + detector_delta
        _write_checkpoint(checkpoint, document)
        if decision.action in {"rollback", "promote"}:
            outcome = _canary_outcome_from_decision(
                decision,
                compute_seconds=compute_seconds,
                agent_execution_seconds=agent_base + agent_delta,
                detector_compute_seconds=detector_base + detector_delta,
            )
            canary_blob["final"] = _serialize_canary_outcome(outcome)
            _write_checkpoint(checkpoint, document)
            return {"interrupted": False, "outcome": outcome}

    if scheduled_arrivals is not None:
        snap = controller.snapshot()
        outcome = CanaryTierOutcome(
            status="horizon_exhausted",
            reason="analysis horizon ended before the stopping rule decided",
            rollback_delay_episodes=None,
            candidate_episodes_served=snap.candidate_episodes_served,
            candidate_episodes_failed=snap.candidate_episodes_failed,
            served_before_rollback=None,
            in_flight_at_rollback=None,
            compute_seconds=compute_base + (time.perf_counter() - started),
            agent_execution_seconds=agent_base + agent_delta,
            detector_compute_seconds=detector_base + detector_delta,
            public_decision=None,
        )
        canary_blob["final"] = _serialize_canary_outcome(outcome)
        _write_checkpoint(checkpoint, document)
        return {"interrupted": False, "outcome": outcome}
    raise BenchmarkError("canary stream ended without rollback or promote")


def _run_or_resume_monitor(
    *,
    protocol: ProtocolLock,
    fault: FaultSpec,
    test_normal_template: RunConfiguration,
    test_normal_tasks: TaskSet,
    monitor_settings: MonitorSettings,
    distributional_monitor_settings: tuple[DistributionalMonitorSettings, ...] = (),
    stream_settings: StreamSettings,
    reference_baselines: FrozenReference,
    factory: RuntimeFactory,
    clock: Callable[[], datetime],
    seed: int,
    harmful: bool,
    stored: dict[str, Any],
    checkpoint: Path,
    document: dict[str, Any],
    should_interrupt: Callable[[BenchmarkProgress], bool] | None,
    fault_version: str,
) -> dict[str, Any]:
    _reference, candidate = _authorize_faulted_test_normal(
        protocol, test_normal_template, test_normal_tasks, fault
    )
    try:
        active_runtime = factory(candidate, mode="execute", role="candidate")
    except EpisodeRejected as error:
        raise BenchmarkError(str(error)) from error

    monitor_blob = stored.setdefault("monitor", {})
    if not isinstance(monitor_blob, dict):
        raise BenchmarkError("checkpoint monitor section is invalid")
    if isinstance(monitor_blob.get("final"), dict):
        final = monitor_blob["final"]
        return {
            "interrupted": False,
            "outcome": MonitorTierOutcome(
                status=str(final["status"]),
                reason=final.get("reason"),
                delay_episodes=final.get("delay_episodes"),
                miss=bool(final.get("miss", False)),
                false_alarm=bool(final.get("false_alarm", False)),
                compute_seconds=_optional_float(final.get("compute_seconds")),
                agent_execution_seconds=_optional_float(
                    final.get("agent_execution_seconds")
                ),
                detector_compute_seconds=_optional_float(
                    final.get("detector_compute_seconds")
                ),
            ),
        }

    items_raw = monitor_blob.get("items")
    if not isinstance(items_raw, list):
        items_raw = []
        monitor_blob["items"] = items_raw

    monitor = ProductionMonitor(
        monitor_settings,
        reference_baselines,
        clock=clock,
        period_id=f"benchmark-{reference_baselines.configuration_hash[:16]}-{seed}",
    )
    distributional_monitors = build_distributional_monitors(
        distributional_monitor_settings,
        reference_configuration_hash=reference_baselines.configuration_hash,
        clock=clock,
        dedup_seconds=0.0,
    )
    tool_monitor = distributional_monitors.get("tool_selection")
    started = time.perf_counter()
    compute_base = float(monitor_blob.get("compute_seconds", 0.0))
    agent_base = float(monitor_blob.get("agent_execution_seconds", 0.0))
    detector_base = float(monitor_blob.get("detector_compute_seconds", 0.0))
    agent_delta = 0.0
    detector_delta = 0.0
    first_alert_index: int | None = monitor_blob.get("first_alert_index")
    if first_alert_index is not None:
        first_alert_index = int(first_alert_index)
    alerted = first_alert_index is not None
    stored_indexes = {
        int(item["stream_index"])
        for item in items_raw
        if isinstance(item, dict) and "stream_index" in item
    }

    for item in sorted(items_raw, key=lambda row: int(row["stream_index"])):
        if not isinstance(item, dict):
            raise BenchmarkError("checkpoint monitor item is invalid")
        observations = item.get("observations")
        if not isinstance(observations, list):
            raise BenchmarkError("checkpoint monitor observations are invalid")
        for observation_payload in observations:
            observation = MonitorObservation.from_dict(observation_payload)
            detector_started = time.perf_counter()
            alerts = monitor.update(observation)
            detector_delta += time.perf_counter() - detector_started
            if alerts and first_alert_index is None:
                first_alert_index = int(item["stream_index"])
                alerted = True
        tool_selection_payload = item.get("tool_selection")
        if tool_monitor is not None and tool_selection_payload is not None:
            selection = ToolSelectionObservation.from_dict(tool_selection_payload)
            detector_started = time.perf_counter()
            alerts = tool_monitor.update(selection)
            detector_delta += time.perf_counter() - detector_started
            if alerts and first_alert_index is None:
                first_alert_index = int(item["stream_index"])
                alerted = True

    from llm_behavior_ci.config import RunIdentity

    if "run" in monitor_blob:
        run = RunIdentity.from_dict(monitor_blob["run"])
    else:
        run = new_run_identity(candidate)
        monitor_blob["run"] = run.to_dict()
        _write_checkpoint(checkpoint, document)

    stream = _stream_for_seed(stream_settings, seed)
    horizon = _monitor_horizon(monitor_settings)

    for arrival in generate_stream(test_normal_tasks, stream):
        if arrival.index >= horizon:
            break
        if arrival.index in stored_indexes:
            continue
        progress = BenchmarkProgress(
            fault_version=fault_version,
            replicate_seed=seed,
            tier="monitor",
            stream_index=arrival.index,
        )
        if should_interrupt is not None and should_interrupt(progress):
            compute_seconds = compute_base + (time.perf_counter() - started)
            monitor_blob["compute_seconds"] = compute_seconds
            monitor_blob["agent_execution_seconds"] = agent_base + agent_delta
            monitor_blob["detector_compute_seconds"] = detector_base + detector_delta
            if first_alert_index is not None:
                monitor_blob["first_alert_index"] = first_alert_index
            _write_checkpoint(checkpoint, document)
            return {
                "interrupted": True,
                "outcome": MonitorTierOutcome(
                    status="interrupted",
                    reason=None,
                    delay_episodes=(
                        None
                        if first_alert_index is None
                        else first_alert_index + 1
                    ),
                    miss=False,
                    false_alarm=False,
                    compute_seconds=compute_seconds,
                    agent_execution_seconds=agent_base + agent_delta,
                    detector_compute_seconds=detector_base + detector_delta,
                ),
            }

        agent_started = time.perf_counter()
        episode = run_episode(
            arrival.task_id,
            candidate,
            "execute",
            run=run,
            runtime=active_runtime,
            scenario_id=arrival.scenario_id,
        )
        agent_delta += time.perf_counter() - agent_started
        observation_payloads: list[dict[str, object]] = []
        new_alerts = False
        for signal in monitor_settings.signals:
            if signal in (PLAN_QUALITY_SIGNAL, PLAN_KL_SIGNAL):
                continue
            observation = observation_from_episode(
                episode,
                task_metadata=TaskMetadata(
                    signal=signal,
                    completion_index=arrival.index,
                ),
            )
            detector_started = time.perf_counter()
            alerts = monitor.update(observation)
            detector_delta += time.perf_counter() - detector_started
            observation_payloads.append(observation.to_dict())
            if alerts:
                new_alerts = True
        tool_selection_payload: dict[str, object] | None = None
        if tool_monitor is not None:
            selection = tool_selection_observation_from_episode(
                episode,
                task_metadata=TaskMetadata(
                    signal="tool_selection",
                    completion_index=arrival.index,
                ),
            )
            detector_started = time.perf_counter()
            alerts = tool_monitor.update(selection)
            detector_delta += time.perf_counter() - detector_started
            tool_selection_payload = selection.to_dict()
            if alerts:
                new_alerts = True
        if new_alerts and first_alert_index is None:
            first_alert_index = arrival.index
            alerted = True
        items_raw.append(
            {
                "stream_index": arrival.index,
                "observations": observation_payloads,
                "tool_selection": tool_selection_payload,
            }
        )
        if first_alert_index is not None:
            monitor_blob["first_alert_index"] = first_alert_index
        compute_seconds = compute_base + (time.perf_counter() - started)
        monitor_blob["compute_seconds"] = compute_seconds
        monitor_blob["agent_execution_seconds"] = agent_base + agent_delta
        monitor_blob["detector_compute_seconds"] = detector_base + detector_delta
        _write_checkpoint(checkpoint, document)

    compute_seconds = compute_base + (time.perf_counter() - started)
    agent_seconds = agent_base + agent_delta
    detector_seconds = detector_base + detector_delta
    delay = None if first_alert_index is None else first_alert_index + 1
    miss = bool(harmful and not alerted)
    false_alarm = bool((not harmful) and alerted)
    outcome = MonitorTierOutcome(
        status="completed",
        reason=None,
        delay_episodes=delay,
        miss=miss,
        false_alarm=false_alarm,
        compute_seconds=compute_seconds,
        agent_execution_seconds=agent_seconds,
        detector_compute_seconds=detector_seconds,
    )
    monitor_blob["final"] = _serialize_monitor_outcome(outcome)
    monitor_blob["compute_seconds"] = compute_seconds
    monitor_blob["agent_execution_seconds"] = agent_seconds
    monitor_blob["detector_compute_seconds"] = detector_seconds
    _write_checkpoint(checkpoint, document)
    return {"interrupted": False, "outcome": outcome}
