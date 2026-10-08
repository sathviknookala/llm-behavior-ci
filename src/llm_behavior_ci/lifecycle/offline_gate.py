"""Plan-only offline CI gate over paired AppWorld plan runs."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Mapping, Sequence

from llm_behavior_ci.config import (
    GateSettings,
    KL_FIDELITY_MODES,
    RunConfiguration,
    RunIdentity,
    hosted_provider,
    new_run_identity,
    run_configuration_hash,
)
from llm_behavior_ci.lifecycle.plan_features import (
    PLAN_FEATURE_SCHEMA_VERSION,
    PlanFeatureError,
    SEMANTIC_PLAN_FEATURES,
    STRUCTURAL_PLAN_FEATURES,
    TOOL_REFERENCE_QUALITY_FEATURES,
    require_semantic_coverage,
    semantic_plan_features,
    structural_plan_features,
)
from llm_behavior_ci.lifecycle.validation_artifact import (
    SYNTHETIC_FIXTURE_PROVENANCE,
    ValidationArtifact,
    build_validation_artifact,
)
from llm_behavior_ci.records import (
    EpisodeResult,
    LifecycleDecision,
    StatisticalEvidence,
    TokenLogprob,
)
from llm_behavior_ci.runtime.episode import (
    EpisodeRejected,
    RuntimeDependencies,
    RuntimeUnavailable,
    is_live_runtime,
    run_pair,
)
from llm_behavior_ci.runtime.scoring import (
    FidelityProofError,
    PositionAlignmentError,
    ScoredPosition,
    ScoringContract,
    ScoringError,
    SupportAlignmentError,
    fingerprint_messages,
    score_full,
    score_top_k,
    verify_scoring_contracts,
)
from llm_behavior_ci.stats.bootstrap import clustered_paired_bootstrap
from llm_behavior_ci.stats.mmd import MMDError, mmd_permutation_test, paired_permutation_resolution
from llm_behavior_ci.tasks.plan_specs import TaskPlanSpec
from llm_behavior_ci.tasks.selection import SelectionError, TaskSet, verify_task_set

_STATISTIC_NAMES = frozenset({"plan_quality", "kl", "mmd"})
_MODEL_STEP_FEATURE = "model_step_count"
_KL_APPROXIMATIONS = KL_FIDELITY_MODES
VALIDATED_PROVENANCE = "validated"
_VALIDATION_PROVENANCE_VALUES = frozenset(
    {SYNTHETIC_FIXTURE_PROVENANCE, VALIDATED_PROVENANCE}
)


class GateExecutionError(ValueError):
    """Raised when the gate cannot produce a PASS or BLOCK decision."""


class GateCapabilityError(GateExecutionError):
    """A required gate statistic cannot be computed for these configurations.

    Raised during preflight, before any world opens or any provider is
    called. It is an execution error, never a BLOCK.
    """


def require_gate_capabilities(
    reference: RunConfiguration,
    candidate: RunConfiguration,
    required_statistics: Sequence[str],
) -> None:
    """Refuse a required statistic that a configuration cannot supply.

    Teacher-forced KL needs prompt-token logprobs and a tokenizer, which
    only a self-hosted vLLM configuration has. A hosted gate must drop
    ``kl`` from ``required_statistics``; no zero, independent-generation,
    or empty KL is substituted.
    """

    if "kl" not in required_statistics:
        return
    for role, configuration in (("reference", reference), ("candidate", candidate)):
        provider = hosted_provider(configuration.model)
        if provider is not None:
            raise GateCapabilityError(
                f"teacher-forced plan KL is unsupported for the {role} "
                f"configuration (hosted provider {provider} exposes no "
                "prompt-token logprobs); remove kl from required_statistics"
            )


@dataclass(frozen=True)
class PlanEvidenceInputs:
    """Caller-supplied plan-evidence settings.

    ``task_plan_specs`` binds pre-execution ``TaskPlanSpec`` metadata by
    task id. It is optional and defaults to empty: a caller whose
    ``plan_quality_features``/``mmd_features`` are all structural (see
    ``lifecycle.plan_features.STRUCTURAL_PLAN_FEATURES``) never needs it.
    Requesting a semantic feature (``lifecycle.plan_features.
    SEMANTIC_PLAN_FEATURES``) for a task with no matching spec raises
    ``GateExecutionError`` at scoring time rather than degrading silently;
    that is a missing-metadata failure, not an empty-metadata one.

    ``kl_vocabulary_size`` is the verified tokenizer's true vocabulary
    size; it is a required companion to ``kl_approximation="full"`` and is
    otherwise unused. Supplying it is a necessary precondition for a
    "full" claim, not proof: ``runtime.scoring.verify_scoring_contracts``
    still checks, from the teacher-forced positions actually returned,
    that every one of them covers exactly that vocabulary and is
    normalized before the claim is honored.
    """

    plan_format_version: str
    plan_quality_features: tuple[str, ...]
    plan_quality_weights: tuple[float, ...]
    mmd_features: tuple[str, ...]
    kl_approximation: Literal["full", "top_k"]
    required_statistics: tuple[str, ...]
    validation_provenance: str
    task_plan_specs: tuple[TaskPlanSpec, ...] = ()
    kl_vocabulary_size: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.plan_format_version, str) or not self.plan_format_version:
            raise GateExecutionError("plan_format_version must be a non-empty string")
        if self.validation_provenance not in _VALIDATION_PROVENANCE_VALUES:
            raise GateExecutionError(
                "validation_provenance must be synthetic_fixture or validated"
            )
        if self.kl_approximation not in _KL_APPROXIMATIONS:
            raise GateExecutionError("kl_approximation must be full or top_k")
        if self.kl_vocabulary_size is not None and (
            not isinstance(self.kl_vocabulary_size, int)
            or isinstance(self.kl_vocabulary_size, bool)
            or self.kl_vocabulary_size <= 0
        ):
            raise GateExecutionError("kl_vocabulary_size must be a positive int")
        if self.kl_approximation == "full" and self.kl_vocabulary_size is None:
            raise GateExecutionError(
                "kl_approximation full requires kl_vocabulary_size"
            )
        if not self.required_statistics:
            raise GateExecutionError("required_statistics must be non-empty")
        for name in self.required_statistics:
            if name not in _STATISTIC_NAMES:
                raise GateExecutionError(
                    "required_statistics entries must be plan_quality, kl, or mmd"
                )
        if len(self.plan_quality_features) != len(self.plan_quality_weights):
            raise GateExecutionError(
                "plan_quality_features and plan_quality_weights length differ"
            )
        if "plan_quality" in self.required_statistics:
            if not self.plan_quality_features:
                raise GateExecutionError("plan_quality_features must be non-empty")
            for weight in self.plan_quality_weights:
                if not isinstance(weight, (int, float)) or isinstance(weight, bool):
                    raise GateExecutionError("plan_quality_weights must be numeric")
            weights = dict(zip(self.plan_quality_features, self.plan_quality_weights))
            if set(weights) & TOOL_REFERENCE_QUALITY_FEATURES and not (
                weights.get(REQUIRED_TOOL_COVERAGE_FEATURE, 0.0) > 0.0
            ):
                raise GateExecutionError(
                    "plan quality terms that read tool references need "
                    "required_tool_coverage_fraction with a positive weight"
                )
        if "mmd" in self.required_statistics and not self.mmd_features:
            raise GateExecutionError("mmd_features must be non-empty")
        if not isinstance(self.task_plan_specs, tuple):
            raise GateExecutionError("task_plan_specs must be a tuple")
        seen_spec_ids: set[str] = set()
        for spec in self.task_plan_specs:
            if not isinstance(spec, TaskPlanSpec):
                raise GateExecutionError(
                    "task_plan_specs entries must be TaskPlanSpec instances"
                )
            if spec.task_id in seen_spec_ids:
                raise GateExecutionError(
                    "task_plan_specs contains a duplicate task_id"
                )
            seen_spec_ids.add(spec.task_id)


def _task_plan_spec_to_dict(spec: TaskPlanSpec) -> dict[str, object]:
    return {
        "task_id": spec.task_id,
        "available_tools": list(spec.available_tools),
        "subgoal_keywords": [list(group) for group in spec.subgoal_keywords],
        "required_entities": list(spec.required_entities),
        "dependency_pairs": [list(pair) for pair in spec.dependency_pairs],
    }


def plan_evidence_to_dict(plan_evidence: PlanEvidenceInputs) -> dict[str, object]:
    """A complete, JSON-canonicalizable mapping of one ``PlanEvidenceInputs``.

    Every field round-trips through ``plan_evidence_from_dict``, including
    ``task_plan_specs`` and ``kl_vocabulary_size``: the caller-supplied
    inputs that decide which plan features a semantic gate statistic reads
    and whether teacher-forced KL claims full-vocabulary fidelity. A
    protocol lock or a benchmark checkpoint that binds only the scalar
    settings and drops these two would let either change silently between
    runs without changing the lock digest or failing a resume.
    """

    return {
        "plan_format_version": plan_evidence.plan_format_version,
        "plan_quality_features": list(plan_evidence.plan_quality_features),
        "plan_quality_weights": list(plan_evidence.plan_quality_weights),
        "mmd_features": list(plan_evidence.mmd_features),
        "kl_approximation": plan_evidence.kl_approximation,
        "kl_vocabulary_size": plan_evidence.kl_vocabulary_size,
        "required_statistics": list(plan_evidence.required_statistics),
        "validation_provenance": plan_evidence.validation_provenance,
        "task_plan_specs": [
            _task_plan_spec_to_dict(spec) for spec in plan_evidence.task_plan_specs
        ],
        "plan_feature_schema_version": PLAN_FEATURE_SCHEMA_VERSION,
    }


def _task_plan_spec_from_dict(payload: Mapping[str, object]) -> TaskPlanSpec:
    from llm_behavior_ci.tasks.plan_specs import task_plan_spec_from_mapping

    return task_plan_spec_from_mapping(
        {
            "task_id": payload["task_id"],
            "available_tools": list(payload["available_tools"]),
            "subgoal_keywords": [
                list(group) for group in payload["subgoal_keywords"]
            ],
            "required_entities": list(payload["required_entities"]),
            "dependency_pairs": [
                list(pair) for pair in payload["dependency_pairs"]
            ],
        }
    )


def plan_evidence_from_dict(payload: Mapping[str, object]) -> PlanEvidenceInputs:
    """Reconstruct a ``PlanEvidenceInputs`` from ``plan_evidence_to_dict``'s shape.

    Raises ``GateExecutionError`` (not a bare ``KeyError``/``TypeError``) on
    a malformed payload, matching every other loader in this package.
    """

    try:
        features = payload["plan_quality_features"]
        weights = payload["plan_quality_weights"]
        mmd_features = payload["mmd_features"]
        required = payload["required_statistics"]
        specs = payload.get("task_plan_specs", [])
        if not isinstance(features, list) or not isinstance(weights, list):
            raise GateExecutionError("plan evidence features are invalid")
        if not isinstance(mmd_features, list) or not isinstance(required, list):
            raise GateExecutionError("plan evidence feature lists are invalid")
        if not isinstance(specs, list):
            raise GateExecutionError("plan evidence task_plan_specs is invalid")
        kl_vocabulary_size = payload.get("kl_vocabulary_size")
        return PlanEvidenceInputs(
            plan_format_version=str(payload["plan_format_version"]),
            plan_quality_features=tuple(str(item) for item in features),
            plan_quality_weights=tuple(float(item) for item in weights),
            mmd_features=tuple(str(item) for item in mmd_features),
            kl_approximation=str(payload["kl_approximation"]),  # type: ignore[arg-type]
            required_statistics=tuple(str(item) for item in required),
            validation_provenance=str(payload["validation_provenance"]),
            task_plan_specs=tuple(
                _task_plan_spec_from_dict(item) for item in specs
            ),
            kl_vocabulary_size=(
                int(kl_vocabulary_size) if kl_vocabulary_size is not None else None
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise GateExecutionError("plan evidence is invalid") from error


@dataclass(frozen=True)
class GateDecision:
    """Public offline-gate outcome and the evidence that supports it."""

    outcome: str
    reason_codes: tuple[str, ...]
    reference_configuration_hash: str
    candidate_configuration_hash: str
    task_set_hash: str
    reference_protocol_hash: str | None
    candidate_protocol_hash: str | None
    thresholds: GateSettings
    statistics: tuple[StatisticalEvidence, ...]
    public_decision: LifecycleDecision | None
    validation_provenance: str
    artifact: ValidationArtifact


def _require_train_inputs(
    reference: RunConfiguration,
    candidate: RunConfiguration,
    task_set: TaskSet,
    settings: GateSettings,
    plan_evidence: PlanEvidenceInputs,
) -> None:
    if not isinstance(task_set, TaskSet):
        raise GateExecutionError("run_offline_gate requires a task set")
    if task_set.split != "train":
        raise GateExecutionError("offline gate requires a train task set")
    if not isinstance(reference, RunConfiguration) or not isinstance(
        candidate, RunConfiguration
    ):
        raise GateExecutionError("run_offline_gate requires run configurations")
    if not isinstance(settings, GateSettings):
        raise GateExecutionError("run_offline_gate requires gate settings")
    if not isinstance(plan_evidence, PlanEvidenceInputs):
        raise GateExecutionError("run_offline_gate requires plan evidence inputs")
    if settings.plan_format_version != plan_evidence.plan_format_version:
        raise GateExecutionError(
            "settings.plan_format_version must match plan_evidence.plan_format_version"
        )
    if plan_evidence.plan_format_version != "plan-v1":
        raise GateExecutionError("plan_format_version must be plan-v1")
    if reference.task.task_set_hash != candidate.task.task_set_hash:
        raise GateExecutionError("configuration task set hashes differ")
    try:
        verify_task_set(reference.task, task_set)
        verify_task_set(candidate.task, task_set)
    except SelectionError as error:
        raise GateExecutionError(str(error)) from error


def require_mmd_resolution(task_set: TaskSet, settings: GateSettings) -> None:
    """Refuse a paired-MMD design that cannot reject at ``mmd_alpha``.

    Clusters are the independent scenario labels the gate swaps
    (``_cluster_label``). The check needs only the task set and settings, so
    an insufficient design fails before any world opens or model is called.
    """

    clusters = {
        _cluster_label(task_id, scenario_id)
        for task_id, scenario_id in zip(task_set.task_ids, task_set.scenario_ids, strict=True)
    }
    resolution = paired_permutation_resolution(len(clusters), settings.mmd_permutations)
    if resolution > settings.mmd_alpha:
        raise GateExecutionError(
            f"paired MMD cannot reject at mmd_alpha {settings.mmd_alpha}: "
            f"{len(clusters)} independent scenario clusters and "
            f"{settings.mmd_permutations} permutations give an expected smallest "
            f"p-value of {resolution:.4f}"
        )


def _decision_seed(
    reference_hash: str,
    candidate_hash: str,
    task_set_hash: str,
    plan_format_version: str,
) -> int:
    material = (
        reference_hash + candidate_hash + task_set_hash + plan_format_version
    ).encode("utf-8")
    return int(hashlib.sha256(material).hexdigest()[:8], 16)


def _cluster_label(task_id: str, scenario_id: str | None) -> str:
    if isinstance(scenario_id, str) and scenario_id:
        return scenario_id
    return task_id


_EMPTY_PLAN_MESSAGE = "empty plan"
REQUIRED_TOOL_COVERAGE_FEATURE = "required_tool_coverage_fraction"
_INFRASTRUCTURE_TERMINATIONS = frozenset({"runtime_error", "timeout", "cancelled"})


def _raise_on_runtime_failure(episode: EpisodeResult, role: str) -> None:
    if episode.termination_reason not in _INFRASTRUCTURE_TERMINATIONS:
        return
    messages = [error.message for error in episode.episode_errors]
    if episode.termination_reason == "runtime_error" and messages == [_EMPTY_PLAN_MESSAGE]:
        return
    kinds = sorted(
        {call.error_kind for call in episode.provider_calls if call.error_kind}
    )
    detail = f" ({', '.join(kinds)})" if kinds else ""
    raise GateExecutionError(
        f"{role} plan episode ended in {episode.termination_reason}{detail}; "
        "this is an execution failure, not a gate decision"
    )


def _plan_succeeded(episode: EpisodeResult) -> bool:
    return (
        episode.status == "completed"
        and episode.termination_reason == "plan_emitted"
    )


def _task_plan_spec(
    task_plan_specs: Mapping[str, TaskPlanSpec],
    task_id: str,
) -> TaskPlanSpec:
    spec = task_plan_specs.get(task_id)
    if spec is None:
        raise GateExecutionError(
            f"missing task_plan_spec for task_id: {task_id}"
        )
    return spec


def _feature_value(
    episode: EpisodeResult,
    feature: str,
    plan_format_version: str,
    task_plan_specs: Mapping[str, TaskPlanSpec],
) -> float:
    if feature == _MODEL_STEP_FEATURE:
        return float(len(episode.model_steps))
    if plan_format_version != "plan-v1":
        raise GateExecutionError("plan_format_version must be plan-v1")
    if feature in STRUCTURAL_PLAN_FEATURES:
        if episode.plan_text is None:
            raise GateExecutionError("plan text is required for plan features")
        return structural_plan_features(episode.plan_text)[feature]
    if feature in SEMANTIC_PLAN_FEATURES:
        if episode.plan_text is None:
            raise GateExecutionError("plan text is required for plan features")
        spec = _task_plan_spec(task_plan_specs, episode.task.task_id)
        try:
            return semantic_plan_features(episode.plan_text, spec)[feature]
        except PlanFeatureError as error:
            raise GateExecutionError(str(error)) from error
    raise GateExecutionError(f"unknown plan feature: {feature}")


def _plan_quality_score(
    episode: EpisodeResult,
    plan_evidence: PlanEvidenceInputs,
    task_plan_specs: Mapping[str, TaskPlanSpec],
) -> float:
    total = 0.0
    for feature, weight in zip(
        plan_evidence.plan_quality_features,
        plan_evidence.plan_quality_weights,
        strict=True,
    ):
        total += float(weight) * _feature_value(
            episode,
            feature,
            plan_evidence.plan_format_version,
            task_plan_specs,
        )
    return total


def _plan_representation(
    episode: EpisodeResult,
    plan_evidence: PlanEvidenceInputs,
    task_plan_specs: Mapping[str, TaskPlanSpec],
) -> tuple[float, ...]:
    return tuple(
        _feature_value(
            episode,
            feature,
            plan_evidence.plan_format_version,
            task_plan_specs,
        )
        for feature in plan_evidence.mmd_features
    )


def _scored_positions(
    positions: Sequence[Sequence[TokenLogprob]],
) -> tuple[ScoredPosition, ...] | None:
    converted: list[ScoredPosition] = []
    for index, position in enumerate(positions):
        if not position:
            return None
        converted.append(
            ScoredPosition(
                position=index,
                support_token_ids=tuple(item.token_id for item in position),
                log_probabilities=tuple(item.logprob for item in position),
            )
        )
    if not converted:
        return None
    return tuple(converted)


def _matched_messages(agent: object, context: object, config: RunConfiguration) -> list[dict[str, str]]:
    begin = getattr(agent, "begin", None)
    if not callable(begin):
        raise GateExecutionError("agent begin is required for teacher-forced KL")
    begin(context, config)
    messages_fn = getattr(agent, "messages", None)
    if callable(messages_fn):
        built = messages_fn()
        if (
            isinstance(built, list)
            and built
            and all(isinstance(item, dict) for item in built)
        ):
            return [
                {"role": str(item["role"]), "content": str(item["content"])}
                for item in built
            ]
    raise GateExecutionError("matched teacher-force messages are unavailable")


def _teacher_force_pair(
    *,
    runtime: RuntimeDependencies,
    task_id: str,
    reference: RunConfiguration,
    candidate: RunConfiguration,
    frozen_plan_text: str,
    candidate_runtime: RuntimeDependencies | None = None,
) -> (
    tuple[
        tuple[tuple[TokenLogprob, ...], ...],
        tuple[tuple[TokenLogprob, ...], ...],
        str,
    ]
    | str
):
    candidate_side = candidate_runtime if candidate_runtime is not None else runtime
    teacher_force = getattr(runtime.agent, "teacher_force_plan", None)
    candidate_force = getattr(candidate_side.agent, "teacher_force_plan", None)
    if not callable(teacher_force) or not callable(candidate_force):
        return "teacher_force_unavailable"
    session = runtime.session_factory(task_id)
    try:
        context = session.context()
        messages = _matched_messages(runtime.agent, context, reference)
        prefix_fingerprint = fingerprint_messages(messages)
        runtime.agent.begin(context, reference)
        try:
            reference_positions = teacher_force(
                messages=messages,
                plan_text=frozen_plan_text,
            )
        except Exception as error:
            message = str(error).lower()
            if "unsupported" in message:
                return "teacher_force_unavailable"
            raise GateExecutionError("teacher-force failed under reference") from error
        candidate_side.agent.begin(context, candidate)
        try:
            candidate_positions = candidate_force(
                messages=messages,
                plan_text=frozen_plan_text,
            )
        except Exception as error:
            message = str(error).lower()
            if "unsupported" in message:
                return "teacher_force_unavailable"
            raise GateExecutionError("teacher-force failed under candidate") from error
    finally:
        session.close()
    if not isinstance(reference_positions, tuple) or not isinstance(
        candidate_positions, tuple
    ):
        return "kl_alignment_failed"
    return reference_positions, candidate_positions, prefix_fingerprint


def _evidence(
    *,
    method: str,
    estimate: float,
    unit: str,
    sample_size: int,
    reference_hash: str,
    candidate_hash: str,
    threshold: float | None,
    seed: int | None,
    confidence_low: float | None = None,
    confidence_high: float | None = None,
    confidence_level: float | None = None,
    p_value: float | None = None,
) -> StatisticalEvidence:
    return StatisticalEvidence(
        method=method,
        split="train",
        configuration_hash=candidate_hash,
        estimate=estimate,
        sample_size=sample_size,
        unit=unit,
        reference_configuration_hash=reference_hash,
        confidence_low=confidence_low,
        confidence_high=confidence_high,
        confidence_level=confidence_level,
        p_value=p_value,
        threshold=threshold,
        seed=seed,
    )


def run_offline_gate(
    reference: RunConfiguration,
    candidate: RunConfiguration,
    task_set: TaskSet,
    *,
    settings: GateSettings,
    runtime: RuntimeDependencies,
    plan_evidence: PlanEvidenceInputs,
    candidate_runtime: RuntimeDependencies | None = None,
    before_pair: Callable[[int, str], None] | None = None,
) -> GateDecision:
    """Score a plan-only offline gate on a train task set.

    Runs ``run_pair`` in plan mode for every task, then applies the caller-
    required plan-quality, teacher-forced KL, and plan-level MMD checks.
    Thresholds come only from ``settings``. Metric definitions come only from
    ``plan_evidence``. ``candidate_runtime``, when supplied, carries the
    candidate's own agent for both the plan episode and teacher forcing;
    worlds still come from ``runtime``. Preflight refuses, before any world
    opens, a ``kl`` requirement on a hosted configuration and semantic
    features whose task specs are missing or empty. Execution problems,
    including a provider or runtime failure inside a plan episode, raise
    ``GateExecutionError`` and are not BLOCK decisions. An empty plan is a
    model behavior and still blocks as ``plan_run_failed``.
    ``before_pair``, when supplied, is called with the task index and id
    before each plan pair opens its worlds; an exception from it stops the
    gate as an execution problem.
    """

    _require_train_inputs(reference, candidate, task_set, settings, plan_evidence)
    if not isinstance(runtime, RuntimeDependencies):
        raise GateExecutionError("run_offline_gate requires runtime dependencies")
    if candidate_runtime is not None and not isinstance(
        candidate_runtime, RuntimeDependencies
    ):
        raise GateExecutionError("candidate_runtime must be runtime dependencies")
    require_gate_capabilities(reference, candidate, plan_evidence.required_statistics)
    if "mmd" in plan_evidence.required_statistics:
        require_mmd_resolution(task_set, settings)
    requested_features: list[str] = []
    if "plan_quality" in plan_evidence.required_statistics:
        requested_features.extend(plan_evidence.plan_quality_features)
    if "mmd" in plan_evidence.required_statistics:
        requested_features.extend(plan_evidence.mmd_features)
    try:
        require_semantic_coverage(
            task_set.task_ids,
            {spec.task_id: spec for spec in plan_evidence.task_plan_specs},
            requested_features,
        )
    except PlanFeatureError as error:
        raise GateExecutionError(str(error)) from error

    if plan_evidence.validation_provenance == VALIDATED_PROVENANCE:
        if not is_live_runtime(runtime) or (
            candidate_runtime is not None and not is_live_runtime(candidate_runtime)
        ):
            raise GateExecutionError(
                "validated provenance requires a live AppWorld runtime and live agents"
            )
        evidence_source = "gate_run"
    else:
        evidence_source = "synthetic_fixture"

    reference_hash = run_configuration_hash(reference)
    candidate_hash = run_configuration_hash(candidate)
    seed = _decision_seed(
        reference_hash,
        candidate_hash,
        task_set.task_set_hash,
        settings.plan_format_version,
    )
    reference_run = new_run_identity(reference)
    candidate_run = new_run_identity(candidate)
    required = frozenset(plan_evidence.required_statistics)
    task_plan_specs = {
        spec.task_id: spec for spec in plan_evidence.task_plan_specs
    }

    pairs: list[tuple[EpisodeResult, EpisodeResult, str, str]] = []
    try:
        for index, task_id in enumerate(task_set.task_ids):
            scenario_id = task_set.scenario_ids[index]
            if before_pair is not None:
                before_pair(index, task_id)
            pair = run_pair(
                task_id,
                reference,
                candidate,
                reference_run=reference_run,
                candidate_run=candidate_run,
                runtime=runtime,
                mode="plan",
                scenario_id=scenario_id,
                candidate_runtime=candidate_runtime,
            )
            if pair.reference.mode != "plan" or pair.candidate.mode != "plan":
                raise GateExecutionError("paired episodes must be plan mode")
            _raise_on_runtime_failure(pair.reference, "reference")
            _raise_on_runtime_failure(pair.candidate, "candidate")
            if pair.reference.tool_steps or pair.candidate.tool_steps:
                return _blocked_without_statistics(
                    reference=reference,
                    candidate=candidate,
                    reference_hash=reference_hash,
                    candidate_hash=candidate_hash,
                    reference_run=reference_run,
                    candidate_run=candidate_run,
                    task_set=task_set,
                    settings=settings,
                    reason_codes=("tool_execution",),
                    validation_provenance=plan_evidence.validation_provenance,
                    evidence_source=evidence_source,
                    created_at=runtime.clock(),
                )
            if not _plan_succeeded(pair.reference) or not _plan_succeeded(
                pair.candidate
            ):
                return _blocked_without_statistics(
                    reference=reference,
                    candidate=candidate,
                    reference_hash=reference_hash,
                    candidate_hash=candidate_hash,
                    reference_run=reference_run,
                    candidate_run=candidate_run,
                    task_set=task_set,
                    settings=settings,
                    reason_codes=("plan_run_failed",),
                    validation_provenance=plan_evidence.validation_provenance,
                    evidence_source=evidence_source,
                    created_at=runtime.clock(),
                )
            pairs.append(
                (
                    pair.reference,
                    pair.candidate,
                    _cluster_label(task_id, scenario_id),
                    task_id,
                )
            )
    except GateExecutionError:
        raise
    except (EpisodeRejected, RuntimeUnavailable, SelectionError) as error:
        raise GateExecutionError(str(error)) from error
    except Exception as error:
        raise GateExecutionError("runtime failed before a gate decision") from error

    sample_size = len(pairs)
    clusters = tuple(label for _reference, _candidate, label, _task_id in pairs)
    statistics: list[StatisticalEvidence] = []
    reason_codes: list[str] = []
    scoring_contract_hashes: list[str] = []

    if "plan_quality" in required:
        reference_scores = tuple(
            _plan_quality_score(reference_episode, plan_evidence, task_plan_specs)
            for reference_episode, _candidate, _label, _task_id in pairs
        )
        candidate_scores = tuple(
            _plan_quality_score(candidate_episode, plan_evidence, task_plan_specs)
            for _reference, candidate_episode, _label, _task_id in pairs
        )
        bootstrap = clustered_paired_bootstrap(
            candidate_scores,
            reference_scores,
            clusters,
            confidence_level=settings.confidence_level,
            resamples=settings.bootstrap_resamples,
            seed=seed,
        )
        statistics.append(
            _evidence(
                method="plan_quality_bootstrap",
                estimate=bootstrap.mean_difference,
                unit="score_delta",
                sample_size=sample_size,
                reference_hash=reference_hash,
                candidate_hash=candidate_hash,
                threshold=settings.score_margin,
                seed=seed,
                confidence_low=bootstrap.confidence_low,
                confidence_high=bootstrap.confidence_high,
                confidence_level=bootstrap.confidence_level,
            )
        )
        if bootstrap.confidence_low < settings.score_margin:
            reason_codes.append("plan_quality_margin")

    if "kl" in required:
        tokenizers_match = (
            reference.model.tokenizer.repository
            == candidate.model.tokenizer.repository
            and reference.model.tokenizer.revision
            == candidate.model.tokenizer.revision
        )
        if not tokenizers_match:
            reason_codes.append("unsupported_tokenizer")
        else:
            production_positions: list[tuple[float, ...]] = []
            candidate_positions: list[tuple[float, ...]] = []
            contract_hashes: list[str] = []
            kl_reason: str | None = None
            for reference_episode, _candidate_episode, _label, task_id in pairs:
                if reference_episode.plan_text is None:
                    raise GateExecutionError("plan text is required for teacher-forced KL")
                forced = _teacher_force_pair(
                    runtime=runtime,
                    task_id=task_id,
                    reference=reference,
                    candidate=candidate,
                    frozen_plan_text=reference_episode.plan_text,
                    candidate_runtime=candidate_runtime,
                )
                if isinstance(forced, str):
                    kl_reason = forced
                    break
                reference_forced, candidate_forced, prefix_fingerprint = forced
                reference_scored = _scored_positions(reference_forced)
                candidate_scored = _scored_positions(candidate_forced)
                if reference_scored is None or candidate_scored is None:
                    kl_reason = "kl_alignment_failed"
                    break
                try:
                    reference_contract = ScoringContract(
                        model_repository=reference.model.model.repository,
                        model_revision=reference.model.model.revision,
                        tokenizer_repository=reference.model.tokenizer.repository,
                        tokenizer_revision=reference.model.tokenizer.revision,
                        input_prefix_fingerprint=prefix_fingerprint,
                        positions=reference_scored,
                        fidelity=plan_evidence.kl_approximation,
                        declared_vocabulary_size=plan_evidence.kl_vocabulary_size,
                    )
                    candidate_contract = ScoringContract(
                        model_repository=candidate.model.model.repository,
                        model_revision=candidate.model.model.revision,
                        tokenizer_repository=candidate.model.tokenizer.repository,
                        tokenizer_revision=candidate.model.tokenizer.revision,
                        input_prefix_fingerprint=prefix_fingerprint,
                        positions=candidate_scored,
                        fidelity=plan_evidence.kl_approximation,
                        declared_vocabulary_size=plan_evidence.kl_vocabulary_size,
                    )
                    verify_scoring_contracts(reference_contract, candidate_contract)
                    contract_hashes.append(reference_contract.content_hash())
                    contract_hashes.append(candidate_contract.content_hash())
                except (PositionAlignmentError, SupportAlignmentError):
                    kl_reason = "kl_alignment_failed"
                    break
                except FidelityProofError:
                    kl_reason = "kl_fidelity_unproven"
                    break
                except ScoringError as error:
                    raise GateExecutionError(str(error)) from error
                production_positions.extend(
                    position.log_probabilities for position in reference_scored
                )
                candidate_positions.extend(
                    position.log_probabilities for position in candidate_scored
                )
            if kl_reason is not None:
                reason_codes.append(kl_reason)
            else:
                scoring_contract_hashes.extend(contract_hashes)
                try:
                    if plan_evidence.kl_approximation == "full":
                        scored = score_full(production_positions, candidate_positions)
                        method = "plan_kl_full"
                    else:
                        scored = score_top_k(production_positions, candidate_positions)
                        method = "plan_kl_top_k"
                except ScoringError as error:
                    raise GateExecutionError(str(error)) from error
                if scored.kind != plan_evidence.kl_approximation:
                    raise GateExecutionError("KL approximation label mismatch")
                statistics.append(
                    _evidence(
                        method=method,
                        estimate=scored.mean_kl_nats,
                        unit="nats",
                        sample_size=sample_size,
                        reference_hash=reference_hash,
                        candidate_hash=candidate_hash,
                        threshold=settings.kl_limit_nats,
                        seed=None,
                    )
                )
                if scored.mean_kl_nats > settings.kl_limit_nats:
                    reason_codes.append("kl_limit")

    if "mmd" in required:
        reference_points = tuple(
            _plan_representation(reference_episode, plan_evidence, task_plan_specs)
            for reference_episode, _candidate, _label, _task_id in pairs
        )
        candidate_points = tuple(
            _plan_representation(candidate_episode, plan_evidence, task_plan_specs)
            for _reference, candidate_episode, _label, _task_id in pairs
        )
        try:
            mmd = mmd_permutation_test(
                production=reference_points,
                candidate=candidate_points,
                bandwidth=settings.mmd_bandwidth,
                permutations=settings.mmd_permutations,
                seed=seed,
                clusters=clusters,
            )
        except MMDError as error:
            raise GateExecutionError(str(error)) from error
        statistics.append(
            _evidence(
                method="plan_mmd",
                estimate=mmd.mmd_squared,
                unit="mmd_squared",
                sample_size=sample_size,
                reference_hash=reference_hash,
                candidate_hash=candidate_hash,
                threshold=settings.mmd_alpha,
                seed=seed,
                p_value=mmd.p_value,
            )
        )
        if mmd.p_value <= settings.mmd_alpha:
            reason_codes.append("mmd_rejected")

    decided_at = runtime.clock()
    outcome = "PASS" if not reason_codes else "BLOCK"
    if not statistics:
        public_decision = None
    else:
        public_decision = LifecycleDecision(
            tier="offline_gate",
            decision="allow_canary" if outcome == "PASS" else "block",
            split="train",
            candidate_configuration_hash=candidate_hash,
            reference_configuration_hash=reference_hash,
            evidence=tuple(statistics),
            decided_at=decided_at,
        )
    artifact = build_validation_artifact(
        outcome=outcome,
        reason_codes=reason_codes,
        reference=reference,
        candidate=candidate,
        reference_run=reference_run,
        candidate_run=candidate_run,
        task_set_hash=task_set.task_set_hash,
        task_split=task_set.split,
        statistics=statistics,
        evidence_source=evidence_source,
        scoring_contract_hashes=scoring_contract_hashes,
        created_at=decided_at,
    )
    return GateDecision(
        outcome=outcome,
        reason_codes=tuple(reason_codes),
        reference_configuration_hash=reference_hash,
        candidate_configuration_hash=candidate_hash,
        task_set_hash=task_set.task_set_hash,
        reference_protocol_hash=reference.protocol_hash,
        candidate_protocol_hash=candidate.protocol_hash,
        thresholds=settings,
        statistics=tuple(statistics),
        public_decision=public_decision,
        validation_provenance=plan_evidence.validation_provenance,
        artifact=artifact,
    )


def _blocked_without_statistics(
    *,
    reference: RunConfiguration,
    candidate: RunConfiguration,
    reference_hash: str,
    candidate_hash: str,
    reference_run: RunIdentity,
    candidate_run: RunIdentity,
    task_set: TaskSet,
    settings: GateSettings,
    reason_codes: Sequence[str],
    validation_provenance: str,
    evidence_source: str,
    created_at: datetime,
) -> GateDecision:
    artifact = build_validation_artifact(
        outcome="BLOCK",
        reason_codes=reason_codes,
        reference=reference,
        candidate=candidate,
        reference_run=reference_run,
        candidate_run=candidate_run,
        task_set_hash=task_set.task_set_hash,
        task_split=task_set.split,
        statistics=(),
        evidence_source=evidence_source,
        created_at=created_at,
    )
    return GateDecision(
        outcome="BLOCK",
        reason_codes=tuple(reason_codes),
        reference_configuration_hash=reference_hash,
        candidate_configuration_hash=candidate_hash,
        task_set_hash=task_set.task_set_hash,
        reference_protocol_hash=reference.protocol_hash,
        candidate_protocol_hash=candidate.protocol_hash,
        thresholds=settings,
        statistics=(),
        public_decision=None,
        validation_provenance=validation_provenance,
        artifact=artifact,
    )
