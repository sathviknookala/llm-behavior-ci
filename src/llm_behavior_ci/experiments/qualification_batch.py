"""Ledgered qualification batches and the evidence computed from them.

Two designs share one runner. ``dev_four_arm`` runs, for every task of a
dev set, a healthy reference (H1), an independent healthy repeat (H2), a
declared no-op candidate (C0), and a regression candidate (C1), each as
its own execute episode in a fresh world, in a seeded per-task arm order.
``plan_aa`` runs two healthy plan generations per train task as one plan
pair, exactly as the offline gate pairs reference and candidate.

Every episode is an ``AttemptLedger`` attempt in the batch checkpoint,
reserved and started before dispatch, keyed by ``(task, arm)`` so arms
with one configuration hash stay distinct. An attempt that failed or was
interrupted keeps its slot and its scope is never dispatched again. The
evidence functions read only what the checkpoint recorded: they call no
model, never replace a missing outcome, and reuse ``measure_harm``'s and
the gate's own scoring.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from llm_behavior_ci.config import (
    ConfigError,
    EpisodeIdentity,
    GateSettings,
    RunConfiguration,
    RunIdentity,
    new_run_identity,
    run_configuration_hash,
)
from llm_behavior_ci.experiments.attempts import (
    AttemptBudget,
    AttemptError,
    AttemptLedger,
    AttemptRequest,
    attempt_state,
)
from llm_behavior_ci.experiments.faults import (
    FaultError,
    FaultSpec,
    apply_fault,
    harm_label_from_outcomes,
)
from llm_behavior_ci.experiments.validation import (
    AAContext,
    AADependenceReport,
    MethodSpec,
    validate_method,
)
from llm_behavior_ci.lifecycle.offline_gate import (
    GateExecutionError,
    PlanEvidenceInputs,
    plan_quality_score,
    plan_representation,
    replay_plan_gate,
)
from llm_behavior_ci.records import EpisodeResult, ModelStep, MonitorObservation, ToolStep
from llm_behavior_ci.runtime.factory import RuntimeFactory
from llm_behavior_ci.tasks.selection import SelectionError, TaskSet, verify_task_set

DEV_FOUR_ARM = "dev_four_arm"
PLAN_AA = "plan_aa"
DEV_ARMS = (
    "H1_healthy_reference",
    "H2_healthy_repeat",
    "C0_noop_candidate",
    "C1_regression_candidate",
)
PLAN_ARMS = ("P0_plan_reference", "P1_plan_repeat")
EXECUTION_AA_METHODS = frozenset({"sequential_canary", "cusum"})
PLAN_AA_METHODS = {
    "clustered_paired_bootstrap": "plan_quality",
    "mmd_permutation_test": "mmd:",
}
_ARM_ROLES = {
    "H1_healthy_reference": "reference",
    "H2_healthy_repeat": "reference",
    "C0_noop_candidate": "candidate",
    "C1_regression_candidate": "candidate",
    "P0_plan_reference": "reference",
    "P1_plan_repeat": "candidate",
}


class QualificationBatchError(ValueError):
    pass


def seeded_arm_orders(task_count: int, seed: int) -> tuple[tuple[str, ...], ...]:
    """One shuffled ``DEV_ARMS`` order per task from ``random.Random(seed)``."""

    generator = random.Random(seed)
    orders = []
    for _index in range(task_count):
        arms = list(DEV_ARMS)
        generator.shuffle(arms)
        orders.append(tuple(arms))
    return tuple(orders)


@dataclass(frozen=True)
class BatchArm:
    name: str
    configuration: RunConfiguration
    fault: FaultSpec | None = None

    @property
    def configuration_hash(self) -> str:
        return run_configuration_hash(self.configuration)


@dataclass(frozen=True)
class BatchDesign:
    kind: str
    task_set: TaskSet
    arms: tuple[BatchArm, ...]
    arm_orders: tuple[tuple[str, ...], ...]

    @property
    def mode(self) -> str:
        return "plan" if self.kind == PLAN_AA else "execute"

    def arm(self, name: str) -> BatchArm:
        for arm in self.arms:
            if arm.name == name:
                return arm
        raise QualificationBatchError(f"unknown arm {name}")

    def attempt_count(self) -> int:
        return sum(len(order) for order in self.arm_orders)

    def identity(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "mode": self.mode,
            "task_set_hash": self.task_set.task_set_hash,
            "task_count": self.task_set.task_count,
            "arms": [
                {
                    "name": arm.name,
                    "configuration_hash": arm.configuration_hash,
                    "fault_version": None if arm.fault is None else arm.fault.fault_version,
                }
                for arm in self.arms
            ],
            "arm_orders": [list(order) for order in self.arm_orders],
        }

    def identity_hash(self) -> str:
        text = json.dumps(self.identity(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bind(configuration: RunConfiguration, task_set: TaskSet, label: str) -> None:
    if not isinstance(configuration, RunConfiguration):
        raise QualificationBatchError(f"{label} must be a run configuration")
    try:
        verify_task_set(configuration.task, task_set)
    except SelectionError as error:
        raise QualificationBatchError(f"{label}: {error}") from error


def dev_four_arm_design(
    task_set: TaskSet,
    *,
    healthy: RunConfiguration,
    noop: RunConfiguration,
    noop_fault: FaultSpec,
    regression: RunConfiguration,
    regression_fault: FaultSpec,
    arm_orders: Sequence[Sequence[str]],
) -> BatchDesign:
    """Bind the four arms to one dev task set and a per-task arm order.

    Each candidate must be exactly its declared fault applied to the
    healthy configuration; a no-op whose patch changes nothing is the
    healthy hash and is still its own arm.
    """

    if not isinstance(task_set, TaskSet) or task_set.split != "dev":
        raise QualificationBatchError("the four-arm batch runs on a dev task set")
    for configuration, label in (
        (healthy, "healthy"),
        (noop, "noop"),
        (regression, "regression"),
    ):
        _bind(configuration, task_set, label)
    for fault, candidate, label in (
        (noop_fault, noop, "noop"),
        (regression_fault, regression, "regression"),
    ):
        try:
            produced = apply_fault(healthy, fault)
        except FaultError as error:
            raise QualificationBatchError(f"{label}: {error}") from error
        if produced != candidate:
            raise QualificationBatchError(f"{label} is not its declared fault")
    orders = tuple(tuple(order) for order in arm_orders)
    if len(orders) != task_set.task_count:
        raise QualificationBatchError("one arm order per task is required")
    for order in orders:
        if sorted(order) != sorted(DEV_ARMS):
            raise QualificationBatchError("each arm order must list every arm once")
    return BatchDesign(
        kind=DEV_FOUR_ARM,
        task_set=task_set,
        arms=(
            BatchArm(DEV_ARMS[0], healthy),
            BatchArm(DEV_ARMS[1], healthy),
            BatchArm(DEV_ARMS[2], noop, noop_fault),
            BatchArm(DEV_ARMS[3], regression, regression_fault),
        ),
        arm_orders=orders,
    )


def plan_aa_design(task_set: TaskSet, *, healthy: RunConfiguration) -> BatchDesign:
    """Two healthy plan generations per train task, run as one gate-style pair."""

    if not isinstance(task_set, TaskSet) or task_set.split != "train":
        raise QualificationBatchError("the plan A/A batch runs on a train task set")
    _bind(healthy, task_set, "healthy")
    return BatchDesign(
        kind=PLAN_AA,
        task_set=task_set,
        arms=(BatchArm(PLAN_ARMS[0], healthy), BatchArm(PLAN_ARMS[1], healthy)),
        arm_orders=tuple(PLAN_ARMS for _task in task_set.task_ids),
    )


def scope_for(design: BatchDesign, task_index: int, arm: str | None = None) -> str:
    if design.kind == PLAN_AA:
        return f"{PLAN_AA}|{task_index}"
    return f"{DEV_FOUR_ARM}|{task_index}|{arm}"


def load_batch_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QualificationBatchError("checkpoint is not readable JSON") from error
    if not isinstance(document, dict):
        raise QualificationBatchError("checkpoint must be an object")
    return document


def write_batch_checkpoint(path: Path, document: Mapping[str, object]) -> None:
    _refuse_results(path)
    if not path.parent.is_dir():
        raise QualificationBatchError("checkpoint parent directory does not exist")
    temporary = path.parent / (path.name + ".tmp")
    text = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise QualificationBatchError("checkpoint write failed") from error


def _refuse_results(path: Path) -> None:
    if any(parent.name == "results" for parent in (path, *path.parents)):
        raise QualificationBatchError("batch output cannot be written under results")


def _budget_limit(design: BatchDesign, budget: AttemptBudget) -> None:
    planned = design.attempt_count()
    if design.mode == "plan":
        cap, other = budget.plan_generations, budget.executions
    else:
        cap, other = budget.executions, budget.plan_generations
    if cap > planned:
        raise QualificationBatchError(
            f"a {design.mode} cap of {cap} exceeds the design's {planned} attempts; "
            "replacement attempts are not bought"
        )
    if other != 0:
        raise QualificationBatchError(
            f"a {design.kind} batch spends no {'execute' if design.mode == 'plan' else 'plan'} attempts"
        )


def open_batch(
    design: BatchDesign,
    checkpoint_path: Path,
    *,
    budget: AttemptBudget | None,
) -> tuple[dict[str, Any], AttemptLedger]:
    """Load or create the checkpoint, bind it to ``design``, and reconcile.

    A new checkpoint needs caps; an existing one keeps the caps it was
    created with. Open attempts left by a crash become ``interrupted``.
    """

    _refuse_results(checkpoint_path)
    document = load_batch_checkpoint(checkpoint_path)
    identity = design.identity()
    if not document:
        if budget is None:
            raise QualificationBatchError("a new batch needs attempt caps")
        _budget_limit(design, budget)
        document = {
            "record": "qualification_batch_checkpoint",
            "identity": identity,
            "identity_hash": design.identity_hash(),
            "runs": {},
            "results": {},
        }
    elif document.get("identity") != identity:
        raise QualificationBatchError("checkpoint belongs to a different batch design")
    section = document.setdefault("attempts", {})
    if not isinstance(section, dict):
        raise QualificationBatchError("checkpoint attempts section is invalid")

    def persist() -> None:
        write_batch_checkpoint(checkpoint_path, document)

    try:
        ledger = AttemptLedger(section, budget=budget, persist=persist)
    except AttemptError as error:
        raise QualificationBatchError(str(error)) from error
    if ledger.caps is None:
        raise QualificationBatchError("checkpoint has no attempt caps")
    if not checkpoint_path.exists():
        persist()
    ledger.reconcile()
    return document, ledger


class _StoreWriter:
    """Persists steps as they happen; one episode at a time."""

    def __init__(self, store: Any | None, task_id: str) -> None:
        self._store = store
        self._task_id = task_id
        self._episode_id: str | None = None

    def on_start(self, identity: EpisodeIdentity, run: RunIdentity) -> None:
        self._episode_id = identity.episode_id
        if self._store is not None:
            self._store.start_episode(identity, run, self._task_id)

    def on_step(self, step: ModelStep | ToolStep) -> None:
        if self._store is not None and self._episode_id is not None:
            self._store.append_step(self._episode_id, step)

    def finish(self, episode: EpisodeResult) -> None:
        if self._store is not None:
            self._store.finish_episode(episode)


def _run_identity(document: dict[str, Any], arm: BatchArm) -> RunIdentity:
    runs = document.setdefault("runs", {})
    stored = runs.get(arm.name)
    if stored is None:
        identity = new_run_identity(arm.configuration)
        runs[arm.name] = identity.to_dict()
        return identity
    identity = RunIdentity.from_dict(stored)
    if identity.configuration_hash != arm.configuration_hash:
        raise QualificationBatchError("stored run identity does not match its arm")
    return identity


def _execute_summary(episode: EpisodeResult) -> dict[str, object]:
    outcome = episode.evaluator_outcome
    return {
        "episode": episode.episode.to_dict(),
        "run": episode.run.to_dict(),
        "task_id": episode.task.task_id,
        "status": episode.status,
        "termination_reason": episode.termination_reason,
        "ended_at": episode.ended_at.isoformat(),
        "success": None if outcome is None else bool(outcome.success),
        "passed_requirements": None if outcome is None else outcome.passed_requirements,
        "total_requirements": None if outcome is None else outcome.total_requirements,
    }


@dataclass(frozen=True)
class BatchRunSummary:
    dispatched: int
    skipped: int
    attempts: Mapping[str, Mapping[str, int | None]]


def run_qualification_batch(
    design: BatchDesign,
    *,
    runtime_factory: RuntimeFactory,
    checkpoint_path: Path,
    budget: AttemptBudget | None,
    store: Any | None = None,
) -> BatchRunSummary:
    """Run every scope the ledger has not seen, in task order then arm order.

    A scope with any attempt, in any state, is skipped. A raised error
    marks the started attempt failed, records the error, and stops the
    batch; a crash leaves the attempt open for ``reconcile``.
    """

    from llm_behavior_ci.runtime.episode import run_episode, run_pair

    document, ledger = open_batch(design, checkpoint_path, budget=budget)
    results = document.setdefault("results", {})

    def persist() -> None:
        write_batch_checkpoint(checkpoint_path, document)

    dispatched = skipped = 0
    for index, task_id in enumerate(design.task_set.task_ids):
        scenario_id = design.task_set.scenario_ids[index]
        if design.kind == PLAN_AA:
            scope = scope_for(design, index)
            if ledger.scope_attempts(scope):
                skipped += 1
                continue
            reference, candidate = design.arms
            attempts = ledger.reserve(
                scope,
                [
                    AttemptRequest(arm.name, "plan", arm.configuration_hash, task_id)
                    for arm in design.arms
                ],
            )
            reference_run = _run_identity(document, reference)
            candidate_run = _run_identity(document, candidate)
            ledger.start(attempts)
            writer = _StoreWriter(store, task_id)
            try:
                pair = run_pair(
                    task_id,
                    reference.configuration,
                    candidate.configuration,
                    reference_run=reference_run,
                    candidate_run=candidate_run,
                    runtime=runtime_factory(
                        reference.configuration, mode="plan", role=_ARM_ROLES[reference.name]
                    ),
                    candidate_runtime=runtime_factory(
                        candidate.configuration, mode="plan", role=_ARM_ROLES[candidate.name]
                    ),
                    mode="plan",
                    scenario_id=scenario_id,
                    on_start=writer.on_start,
                    on_step=writer.on_step,
                )
            except Exception as error:
                for attempt in attempts:
                    if attempt["state"] == "started":
                        ledger.finish(attempt, "failed")
                results[scope] = {"error": _error_text(error)}
                persist()
                raise QualificationBatchError(
                    f"plan pair {index} failed before a result: {_error_text(error)}"
                ) from error
            for attempt, episode in zip(attempts, (pair.reference, pair.candidate), strict=True):
                writer.finish(episode)
                ledger.finish(attempt, attempt_state(episode))
            results[scope] = {
                "reference": pair.reference.to_dict(),
                "candidate": pair.candidate.to_dict(),
            }
            persist()
            dispatched += 1
            continue
        for arm_name in design.arm_orders[index]:
            scope = scope_for(design, index, arm_name)
            if ledger.scope_attempts(scope):
                skipped += 1
                continue
            arm = design.arm(arm_name)
            (attempt,) = ledger.reserve(
                scope,
                [AttemptRequest(arm.name, "execute", arm.configuration_hash, task_id)],
            )
            run = _run_identity(document, arm)
            ledger.start([attempt])
            writer = _StoreWriter(store, task_id)
            try:
                episode = run_episode(
                    task_id,
                    arm.configuration,
                    "execute",
                    run=run,
                    runtime=runtime_factory(
                        arm.configuration, mode="execute", role=_ARM_ROLES[arm.name]
                    ),
                    scenario_id=scenario_id,
                    on_start=writer.on_start,
                    on_step=writer.on_step,
                )
            except Exception as error:
                ledger.finish(attempt, "failed")
                results[scope] = {"error": _error_text(error)}
                persist()
                raise QualificationBatchError(
                    f"{arm_name} on task {index} failed before a result: {_error_text(error)}"
                ) from error
            writer.finish(episode)
            ledger.finish(attempt, attempt_state(episode))
            results[scope] = _execute_summary(episode)
            persist()
            dispatched += 1
    return BatchRunSummary(dispatched=dispatched, skipped=skipped, attempts=ledger.summary())


def _error_text(error: BaseException) -> str:
    message = str(error)
    return type(error).__name__ if not message else f"{type(error).__name__}: {message}"


def _attempt_states(document: Mapping[str, Any]) -> dict[str, str]:
    section = document.get("attempts", {})
    records = section.get("records", []) if isinstance(section, Mapping) else []
    states: dict[str, str] = {}
    for record in records:
        states[str(record["scope"])] = str(record["state"])
    return states


def _require_design(design: BatchDesign, document: Mapping[str, Any], kind: str) -> None:
    if design.kind != kind:
        raise QualificationBatchError(f"evidence needs a {kind} design")
    if document.get("identity") != design.identity():
        raise QualificationBatchError("checkpoint belongs to a different batch design")


def dev_outcome_table(
    design: BatchDesign, document: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], ...]:
    """Per task, each arm's attempt state and evaluator success (``None`` if absent)."""

    _require_design(design, document, DEV_FOUR_ARM)
    states = _attempt_states(document)
    results = document.get("results", {})
    rows = []
    for index in range(design.task_set.task_count):
        row: dict[str, dict[str, Any]] = {}
        for arm in DEV_ARMS:
            scope = scope_for(design, index, arm)
            result = results.get(scope)
            success = result.get("success") if isinstance(result, Mapping) else None
            row[arm] = {
                "state": states.get(scope, "not_attempted"),
                "success": success if isinstance(success, bool) else None,
            }
        rows.append(row)
    return tuple(rows)


def execution_aa_inputs(
    design: BatchDesign, document: Mapping[str, Any]
) -> tuple[tuple[MonitorObservation, ...], AAContext]:
    """H1 as repetition 0 and H2 as repetition 1 ``task_success`` observations.

    Only scored episodes enter; a task whose arm has no evaluator outcome
    contributes nothing for that arm. ``C0`` never enters, although it
    shares the healthy hash.
    """

    _require_design(design, document, DEV_FOUR_ARM)
    results = document.get("results", {})
    observations: list[MonitorObservation] = []
    task_ids: list[str] = []
    scenario_ids: list[str | None] = []
    repetitions: list[int] = []
    for index, task_id in enumerate(design.task_set.task_ids):
        for repetition, arm in enumerate(DEV_ARMS[:2]):
            result = results.get(scope_for(design, index, arm))
            if not isinstance(result, Mapping) or not isinstance(result.get("success"), bool):
                continue
            observations.append(
                MonitorObservation(
                    episode=EpisodeIdentity.from_dict(result["episode"]),
                    run=RunIdentity.from_dict(result["run"]),
                    split=design.task_set.split,
                    signal="task_success",
                    value=1.0 if result["success"] else 0.0,
                    observed_at=datetime.fromisoformat(str(result["ended_at"])),
                )
            )
            task_ids.append(task_id)
            scenario_ids.append(design.task_set.scenario_ids[index])
            repetitions.append(repetition)
    context = AAContext(
        provenance="local_runtime",
        hardware_observed=False,
        scenario_ids=tuple(scenario_ids),
        task_ids=tuple(task_ids),
        repetitions=tuple(repetitions),
    )
    return tuple(observations), context


def cusum_scale(design: BatchDesign, document: Mapping[str, Any]) -> dict[str, Any]:
    """The proposal's scale rule on scored H1 and H2 outcomes.

    target = mean, sigma = sample standard deviation, slack = 0.5 sigma,
    threshold = 5 sigma. ``C0`` is excluded.
    """

    observations, context = execution_aa_inputs(design, document)
    values = [item.value for item in observations]
    scenarios = {item for item in context.scenario_ids or () if item is not None}
    if len(values) < 2:
        return {
            "status": "unavailable",
            "scored_outcomes": len(values),
            "scenarios": len(scenarios),
        }
    target = fmean(values)
    sigma = stdev(values)
    return {
        "status": "estimated",
        "scored_outcomes": len(values),
        "scenarios": len(scenarios),
        "target": target,
        "sigma": sigma,
        "slack": 0.5 * sigma,
        "threshold": 5.0 * sigma,
    }


def dev_harm_labels(
    design: BatchDesign,
    document: Mapping[str, Any],
    *,
    margin: float,
    confidence_level: float,
    resamples: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """No-op (H1, C0) and regression (H1, C1) labels with ``measure_harm``'s math.

    Both share the H1 column. A label with any missing outcome is
    ``unmeasurable`` and carries no estimate.
    """

    table = dev_outcome_table(design, document)
    healthy = design.arm(DEV_ARMS[0])
    reference = [row[DEV_ARMS[0]]["success"] for row in table]
    labels: dict[str, dict[str, Any]] = {}
    for name, arm_name in (("noop", DEV_ARMS[2]), ("regression", DEV_ARMS[3])):
        arm = design.arm(arm_name)
        assert arm.fault is not None
        candidate = [row[arm_name]["success"] for row in table]
        missing = sum(
            1
            for base, other in zip(reference, candidate, strict=True)
            if base is None or other is None
        )
        entry: dict[str, Any] = {
            "fault_version": arm.fault.fault_version,
            "base_configuration_hash": healthy.configuration_hash,
            "candidate_configuration_hash": arm.configuration_hash,
            "reference_arm": DEV_ARMS[0],
            "candidate_arm": arm_name,
            "tasks": design.task_set.task_count,
            "missing_pairs": missing,
        }
        if missing:
            entry["status"] = "unmeasurable"
            labels[name] = entry
            continue
        try:
            label = harm_label_from_outcomes(
                healthy.configuration,
                arm.configuration,
                design.task_set,
                fault=arm.fault,
                base_successes=reference,  # type: ignore[arg-type]
                candidate_successes=candidate,  # type: ignore[arg-type]
                margin=margin,
                confidence_level=confidence_level,
                resamples=resamples,
                seed=seed,
            )
        except FaultError as error:
            raise QualificationBatchError(str(error)) from error
        entry.update(
            {
                "status": "measured",
                "effect_estimate": label.effect_estimate,
                "interval_low": label.interval_low,
                "interval_high": label.interval_high,
                "harmful": label.harmful,
                "margin": label.margin,
                "confidence_level": label.confidence_level,
                "resamples": label.resamples,
                "seed": label.seed,
                "task_set_hash": label.task_set_hash,
            }
        )
        labels[name] = entry
    return labels


def plan_pairs(
    design: BatchDesign, document: Mapping[str, Any]
) -> tuple[tuple[EpisodeResult, EpisodeResult] | None, ...]:
    """The stored plan pair per task, or ``None`` where no result was recorded."""

    _require_design(design, document, PLAN_AA)
    results = document.get("results", {})
    pairs: list[tuple[EpisodeResult, EpisodeResult] | None] = []
    for index in range(design.task_set.task_count):
        result = results.get(scope_for(design, index))
        if not isinstance(result, Mapping) or "reference" not in result:
            pairs.append(None)
            continue
        pairs.append((_episode(result["reference"]), _episode(result["candidate"])))
    return tuple(pairs)


def _episode(payload: object) -> EpisodeResult:
    episode = EpisodeResult.from_dict(payload)
    if not isinstance(episode, EpisodeResult):
        raise QualificationBatchError("stored plan episode is invalid")
    return episode


def _fraction_features(features: Sequence[str], label: str) -> None:
    for feature in features:
        if not feature.endswith("_fraction"):
            raise QualificationBatchError(
                f"{label} feature {feature} is not a bounded fraction"
            )


def plan_quality_bounds(plan_evidence: PlanEvidenceInputs) -> tuple[float, float]:
    """The range of the gate's weighted score when every feature is a fraction."""

    _fraction_features(plan_evidence.plan_quality_features, "plan quality")
    low = sum(min(0.0, float(weight)) for weight in plan_evidence.plan_quality_weights)
    high = sum(max(0.0, float(weight)) for weight in plan_evidence.plan_quality_weights)
    if high <= low:
        raise QualificationBatchError("plan quality weights give an empty range")
    return low, high


def _plan_succeeded(episode: EpisodeResult) -> bool:
    return (
        episode.status == "completed"
        and episode.termination_reason == "plan_emitted"
        and not episode.tool_steps
    )


@dataclass(frozen=True)
class PlanAASeries:
    """One scalar A/A series: the gate's score, or one MMD coordinate.

    ``observations`` carry ``plan_quality_score`` values in [0, 1]: the
    gate's weighted score mapped affinely from ``bounds``, or the
    coordinate itself, which is already a fraction. ``raw`` keeps the
    gate's own units in observation order.
    """

    name: str
    observations: tuple[MonitorObservation, ...]
    context: AAContext
    raw: tuple[float, ...]
    bounds: tuple[float, float]


def plan_aa_series(
    design: BatchDesign,
    document: Mapping[str, Any],
    plan_evidence: PlanEvidenceInputs,
) -> tuple[PlanAASeries, ...]:
    """The scalar series each gate method's A/A check reads.

    ``plan_quality`` feeds ``clustered_paired_bootstrap``. ``mmd:<feature>``
    is one series per coordinate of the MMD representation, which feeds
    ``mmd_permutation_test``. Scores and vectors are the gate's own
    (``plan_quality_score``, ``plan_representation``). A plan that did not
    complete as a plan enters no series.
    """

    low, high = plan_quality_bounds(plan_evidence)
    _fraction_features(plan_evidence.mmd_features, "MMD")
    collected: dict[str, list[tuple[EpisodeResult, str, str | None, int, float]]] = {
        "plan_quality": []
    }
    for feature in plan_evidence.mmd_features:
        collected[f"mmd:{feature}"] = []
    pairs = plan_pairs(design, document)
    for index, pair in enumerate(pairs):
        if pair is None:
            continue
        task_id = design.task_set.task_ids[index]
        scenario_id = design.task_set.scenario_ids[index]
        for repetition, episode in enumerate(pair):
            if not _plan_succeeded(episode):
                continue
            try:
                score = plan_quality_score(episode, plan_evidence)
                vector = plan_representation(episode, plan_evidence)
            except GateExecutionError as error:
                raise QualificationBatchError(str(error)) from error
            collected["plan_quality"].append((episode, task_id, scenario_id, repetition, score))
            for feature, value in zip(plan_evidence.mmd_features, vector, strict=True):
                collected[f"mmd:{feature}"].append(
                    (episode, task_id, scenario_id, repetition, value)
                )
    series = []
    for name, items in collected.items():
        bounds = (low, high) if name == "plan_quality" else (0.0, 1.0)
        observations = tuple(
            MonitorObservation(
                episode=episode.episode,
                run=episode.run,
                split=design.task_set.split,
                signal="plan_quality_score",
                value=(value - bounds[0]) / (bounds[1] - bounds[0]),
                observed_at=episode.ended_at,
            )
            for episode, _task, _scenario, _repetition, value in items
        )
        series.append(
            PlanAASeries(
                name=name,
                observations=observations,
                context=AAContext(
                    provenance="local_runtime",
                    hardware_observed=False,
                    scenario_ids=tuple(item[2] for item in items),
                    task_ids=tuple(item[1] for item in items),
                    repetitions=tuple(item[3] for item in items),
                ),
                raw=tuple(item[4] for item in items),
                bounds=bounds,
            )
        )
    return tuple(series)


def plan_aa_gate_replay(
    design: BatchDesign,
    document: Mapping[str, Any],
    *,
    settings: GateSettings,
    plan_evidence: PlanEvidenceInputs,
) -> dict[str, object]:
    """The frozen gate's decision with repetition 0 as reference and 1 as candidate.

    ``incomplete`` when any task lacks a recorded pair; the gate is never
    run on a partial set.
    """

    pairs = plan_pairs(design, document)
    missing = sum(1 for pair in pairs if pair is None)
    if missing:
        return {"status": "incomplete", "missing_pairs": missing}
    healthy = design.arm(PLAN_ARMS[0]).configuration
    try:
        replay = replay_plan_gate(
            healthy,
            healthy,
            design.task_set,
            [pair for pair in pairs if pair is not None],
            settings=settings,
            plan_evidence=plan_evidence,
        )
    except GateExecutionError as error:
        return {"status": "execution_failed", "reason": str(error)}
    return {
        "status": "decided",
        "outcome": replay.outcome,
        "reason_codes": list(replay.reason_codes),
        "seed": replay.seed,
        "statistics": [item.to_dict() for item in replay.statistics],
    }


def load_configuration(path: Path) -> RunConfiguration:
    try:
        return RunConfiguration.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ConfigError) as error:
        raise QualificationBatchError(f"configuration {path.name} is not loadable") from error


def aa_dependence(
    spec: MethodSpec,
    observations: Sequence[MonitorObservation],
    context: AAContext,
) -> AADependenceReport:
    """``validate_method``'s A/A dependence result for one method and series."""

    return validate_method(
        spec,
        null_seeds=(),
        reference_cases=(),
        aa_observations=tuple(observations),
        aa_context=context,
    ).aa


def execution_aa_reports(
    design: BatchDesign,
    document: Mapping[str, Any],
    specs: Sequence[MethodSpec],
) -> dict[str, dict[str, object]]:
    observations, context = execution_aa_inputs(design, document)
    reports: dict[str, dict[str, object]] = {}
    for spec in specs:
        if spec.name not in EXECUTION_AA_METHODS:
            raise QualificationBatchError(f"{spec.name} does not read execution A/A")
        reports[spec.name] = asdict(aa_dependence(spec, observations, context))
    return reports


def plan_aa_reports(
    series: Sequence[PlanAASeries],
    specs: Sequence[MethodSpec],
) -> dict[str, list[dict[str, Any]]]:
    """One A/A result per series a gate method reads.

    The bootstrap reads the score; MMD reads every coordinate of its
    representation, so it has one result per coordinate.
    """

    reports: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        prefix = PLAN_AA_METHODS.get(spec.name)
        if prefix is None:
            raise QualificationBatchError(f"{spec.name} does not read plan A/A")
        chosen = [
            item
            for item in series
            if item.name == prefix or (prefix.endswith(":") and item.name.startswith(prefix))
        ]
        reports[spec.name] = [
            {
                "series": item.name,
                "bounds": list(item.bounds),
                "report": asdict(aa_dependence(spec, item.observations, item.context)),
            }
            for item in chosen
        ]
    return reports

