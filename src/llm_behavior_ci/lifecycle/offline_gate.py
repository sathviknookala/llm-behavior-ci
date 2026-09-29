"""Plan-only offline CI gate over paired AppWorld plan runs."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Literal, Sequence

from llm_behavior_ci.config import (
    GateSettings,
    RunConfiguration,
    new_run_identity,
    run_configuration_hash,
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
    run_pair,
)
from llm_behavior_ci.runtime.scoring import ScoringError, score_full, score_top_k
from llm_behavior_ci.stats.bootstrap import clustered_paired_bootstrap
from llm_behavior_ci.stats.mmd import MMDError, mmd_permutation_test
from llm_behavior_ci.tasks.selection import SelectionError, TaskSet, verify_task_set

_STATISTIC_NAMES = frozenset({"plan_quality", "kl", "mmd"})
_PLAN_V1_FEATURES = frozenset(
    {
        "char_count",
        "line_count",
        "numbered_step_count",
        "token_count",
        "empty_line_count",
        "mean_step_chars",
        "model_step_count",
    }
)
_NUMBERED_STEP = re.compile(r"^\s*\d+\.")
_KL_APPROXIMATIONS = frozenset({"full", "top_k"})


class GateExecutionError(ValueError):
    """Raised when the gate cannot produce a PASS or BLOCK decision."""


@dataclass(frozen=True)
class PlanEvidenceInputs:
    plan_format_version: str
    plan_quality_features: tuple[str, ...]
    plan_quality_weights: tuple[float, ...]
    mmd_features: tuple[str, ...]
    kl_approximation: Literal["full", "top_k"]
    required_statistics: tuple[str, ...]
    validation_provenance: str

    def __post_init__(self) -> None:
        if not isinstance(self.plan_format_version, str) or not self.plan_format_version:
            raise GateExecutionError("plan_format_version must be a non-empty string")
        if not isinstance(self.validation_provenance, str) or not self.validation_provenance:
            raise GateExecutionError("validation_provenance must be a non-empty string")
        if self.kl_approximation not in _KL_APPROXIMATIONS:
            raise GateExecutionError("kl_approximation must be full or top_k")
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
        if "mmd" in self.required_statistics and not self.mmd_features:
            raise GateExecutionError("mmd_features must be non-empty")


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


def _plan_succeeded(episode: EpisodeResult) -> bool:
    return (
        episode.status == "completed"
        and episode.termination_reason == "plan_emitted"
    )


def _plan_v1_structure(plan_text: str) -> dict[str, float]:
    lines = plan_text.splitlines()
    non_empty = [line for line in lines if line.strip()]
    numbered = [line for line in non_empty if _NUMBERED_STEP.match(line)]
    tokens = plan_text.split()
    mean_step_chars = (
        float(sum(len(line) for line in numbered) / len(numbered))
        if numbered
        else 0.0
    )
    return {
        "char_count": float(len(plan_text)),
        "line_count": float(len(non_empty)),
        "numbered_step_count": float(len(numbered)),
        "token_count": float(len(tokens)),
        "empty_line_count": float(len(lines) - len(non_empty)),
        "mean_step_chars": mean_step_chars,
    }


def _feature_value(
    episode: EpisodeResult,
    feature: str,
    plan_format_version: str,
) -> float:
    if feature not in _PLAN_V1_FEATURES:
        raise GateExecutionError(f"unknown plan feature: {feature}")
    if plan_format_version != "plan-v1":
        raise GateExecutionError("plan_format_version must be plan-v1")
    if feature == "model_step_count":
        return float(len(episode.model_steps))
    if episode.plan_text is None:
        raise GateExecutionError("plan text is required for plan features")
    return _plan_v1_structure(episode.plan_text)[feature]


def _plan_quality_score(
    episode: EpisodeResult,
    plan_evidence: PlanEvidenceInputs,
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
        )
    return total


def _plan_representation(
    episode: EpisodeResult,
    plan_evidence: PlanEvidenceInputs,
) -> tuple[float, ...]:
    return tuple(
        _feature_value(episode, feature, plan_evidence.plan_format_version)
        for feature in plan_evidence.mmd_features
    )


def _aligned_logprob_vectors(
    reference_positions: Sequence[Sequence[TokenLogprob]],
    candidate_positions: Sequence[Sequence[TokenLogprob]],
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]] | None:
    if len(reference_positions) != len(candidate_positions):
        return None
    production: list[tuple[float, ...]] = []
    comparison: list[tuple[float, ...]] = []
    for reference_position, candidate_position in zip(
        reference_positions,
        candidate_positions,
        strict=True,
    ):
        if not reference_position or not candidate_position:
            return None
        reference_ids = tuple(item.token_id for item in reference_position)
        candidate_ids = tuple(item.token_id for item in candidate_position)
        if reference_ids != candidate_ids:
            return None
        production.append(tuple(item.logprob for item in reference_position))
        comparison.append(tuple(item.logprob for item in candidate_position))
    if not production:
        return None
    return tuple(production), tuple(comparison)


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
) -> tuple[tuple[tuple[TokenLogprob, ...], ...], tuple[tuple[TokenLogprob, ...], ...]] | str:
    teacher_force = getattr(runtime.agent, "teacher_force_plan", None)
    if not callable(teacher_force):
        return "teacher_force_unavailable"
    session = runtime.session_factory(task_id)
    try:
        context = session.context()
        messages = _matched_messages(runtime.agent, context, reference)
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
        runtime.agent.begin(context, candidate)
        try:
            candidate_positions = teacher_force(
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
    return reference_positions, candidate_positions


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
) -> GateDecision:
    """Score a plan-only offline gate on a train task set.

    Runs ``run_pair`` in plan mode for every task, then applies the caller-
    required plan-quality, teacher-forced KL, and plan-level MMD checks.
    Thresholds come only from ``settings``. Metric definitions come only from
    ``plan_evidence``. Execution problems raise ``GateExecutionError`` and are
    not BLOCK decisions.
    """

    _require_train_inputs(reference, candidate, task_set, settings, plan_evidence)
    if not isinstance(runtime, RuntimeDependencies):
        raise GateExecutionError("run_offline_gate requires runtime dependencies")

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

    pairs: list[tuple[EpisodeResult, EpisodeResult, str, str]] = []
    try:
        for index, task_id in enumerate(task_set.task_ids):
            scenario_id = task_set.scenario_ids[index]
            pair = run_pair(
                task_id,
                reference,
                candidate,
                reference_run=reference_run,
                candidate_run=candidate_run,
                runtime=runtime,
                mode="plan",
                scenario_id=scenario_id,
            )
            if pair.reference.mode != "plan" or pair.candidate.mode != "plan":
                raise GateExecutionError("paired episodes must be plan mode")
            if pair.reference.tool_steps or pair.candidate.tool_steps:
                return _blocked_without_statistics(
                    reference=reference,
                    candidate=candidate,
                    reference_hash=reference_hash,
                    candidate_hash=candidate_hash,
                    task_set=task_set,
                    settings=settings,
                    reason_codes=("tool_execution",),
                    validation_provenance=plan_evidence.validation_provenance,
                )
            if not _plan_succeeded(pair.reference) or not _plan_succeeded(
                pair.candidate
            ):
                return _blocked_without_statistics(
                    reference=reference,
                    candidate=candidate,
                    reference_hash=reference_hash,
                    candidate_hash=candidate_hash,
                    task_set=task_set,
                    settings=settings,
                    reason_codes=("plan_run_failed",),
                    validation_provenance=plan_evidence.validation_provenance,
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

    if "plan_quality" in required:
        reference_scores = tuple(
            _plan_quality_score(reference_episode, plan_evidence)
            for reference_episode, _candidate, _label, _task_id in pairs
        )
        candidate_scores = tuple(
            _plan_quality_score(candidate_episode, plan_evidence)
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
                )
                if isinstance(forced, str):
                    kl_reason = forced
                    break
                reference_forced, candidate_forced = forced
                vectors = _aligned_logprob_vectors(reference_forced, candidate_forced)
                if vectors is None:
                    kl_reason = "kl_alignment_failed"
                    break
                production_chunk, candidate_chunk = vectors
                production_positions.extend(production_chunk)
                candidate_positions.extend(candidate_chunk)
            if kl_reason is not None:
                reason_codes.append(kl_reason)
            else:
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
            _plan_representation(reference_episode, plan_evidence)
            for reference_episode, _candidate, _label, _task_id in pairs
        )
        candidate_points = tuple(
            _plan_representation(candidate_episode, plan_evidence)
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
    )


def _blocked_without_statistics(
    *,
    reference: RunConfiguration,
    candidate: RunConfiguration,
    reference_hash: str,
    candidate_hash: str,
    task_set: TaskSet,
    settings: GateSettings,
    reason_codes: Sequence[str],
    validation_provenance: str,
) -> GateDecision:
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
    )
