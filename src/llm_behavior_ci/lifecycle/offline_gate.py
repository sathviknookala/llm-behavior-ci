"""Plan-only offline CI gate over paired AppWorld plan runs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

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
)
from llm_behavior_ci.runtime.episode import (
    EpisodeRejected,
    RuntimeDependencies,
    RuntimeUnavailable,
    run_pair,
)
from llm_behavior_ci.stats.bootstrap import clustered_paired_bootstrap
from llm_behavior_ci.stats.kl import TruncatedKLError, truncated_next_token_kl
from llm_behavior_ci.stats.mmd import MMDError, mmd_permutation_test
from llm_behavior_ci.tasks.selection import SelectionError, TaskSet, verify_task_set

_PLAN_FORMAT = "plan-v1"


class GateExecutionError(ValueError):
    """Raised when the gate cannot produce a PASS or BLOCK decision."""


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


def _require_train_inputs(
    reference: RunConfiguration,
    candidate: RunConfiguration,
    task_set: TaskSet,
    settings: GateSettings,
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
    if settings.plan_format_version != _PLAN_FORMAT:
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


def _aligned_top_k_vectors(
    reference: EpisodeResult,
    candidate: EpisodeResult,
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]] | None:
    if not reference.model_steps or not candidate.model_steps:
        return None
    reference_positions = reference.model_steps[0].top_k_logprobs
    candidate_positions = candidate.model_steps[0].top_k_logprobs
    if len(reference_positions) != len(candidate_positions):
        return None
    production: list[tuple[float, ...]] = []
    comparison: list[tuple[float, ...]] = []
    for reference_position, candidate_position in zip(
        reference_positions,
        candidate_positions,
        strict=True,
    ):
        reference_ids = tuple(item.token_id for item in reference_position)
        candidate_ids = tuple(item.token_id for item in candidate_position)
        if reference_ids != candidate_ids:
            return None
        production.append(tuple(item.logprob for item in reference_position))
        comparison.append(tuple(item.logprob for item in candidate_position))
    if not production:
        return None
    return tuple(production), tuple(comparison)


def _plan_point(episode: EpisodeResult) -> tuple[float, float]:
    plan_text = episode.plan_text if episode.plan_text is not None else ""
    return (float(len(plan_text)), float(len(episode.model_steps)))


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
) -> GateDecision:
    """Score a plan-only offline gate on a train task set.

    Runs ``run_pair`` in plan mode for every task, then applies clustered
    paired bootstrap on plan-quality scores, truncated top-k plan KL when
    tokenizers match and top-k tables align, and plan-level MMD. Thresholds
    come only from ``settings``. Execution problems raise
    ``GateExecutionError`` and are not BLOCK decisions.
    """

    _require_train_inputs(reference, candidate, task_set, settings)
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

    pairs: list[tuple[EpisodeResult, EpisodeResult, str]] = []
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
                )
            pairs.append(
                (
                    pair.reference,
                    pair.candidate,
                    _cluster_label(task_id, scenario_id),
                )
            )
    except GateExecutionError:
        raise
    except (EpisodeRejected, RuntimeUnavailable, SelectionError) as error:
        raise GateExecutionError(str(error)) from error
    except Exception as error:
        raise GateExecutionError("runtime failed before a gate decision") from error

    sample_size = len(pairs)
    clusters = tuple(label for _reference, _candidate, label in pairs)
    reference_scores = tuple(1.0 for _ in pairs)
    candidate_scores = tuple(1.0 for _ in pairs)
    bootstrap = clustered_paired_bootstrap(
        candidate_scores,
        reference_scores,
        clusters,
        confidence_level=settings.confidence_level,
        resamples=settings.bootstrap_resamples,
        seed=seed,
    )
    statistics: list[StatisticalEvidence] = [
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
    ]
    reason_codes: list[str] = []
    if bootstrap.confidence_low < settings.score_margin:
        reason_codes.append("plan_quality_margin")

    tokenizers_match = (
        reference.model.tokenizer.repository == candidate.model.tokenizer.repository
        and reference.model.tokenizer.revision == candidate.model.tokenizer.revision
    )
    if not tokenizers_match:
        reason_codes.append("unsupported_tokenizer")
    else:
        production_positions: list[tuple[float, ...]] = []
        candidate_positions: list[tuple[float, ...]] = []
        aligned = True
        for reference_episode, candidate_episode, _label in pairs:
            vectors = _aligned_top_k_vectors(reference_episode, candidate_episode)
            if vectors is None:
                aligned = False
                break
            production_chunk, candidate_chunk = vectors
            production_positions.extend(production_chunk)
            candidate_positions.extend(candidate_chunk)
        if not aligned:
            reason_codes.append("kl_alignment_failed")
        else:
            try:
                kl = truncated_next_token_kl(
                    production_positions,
                    candidate_positions,
                )
            except TruncatedKLError as error:
                raise GateExecutionError(str(error)) from error
            statistics.append(
                _evidence(
                    method="truncated_plan_kl",
                    estimate=kl.mean_kl_nats,
                    unit="nats",
                    sample_size=sample_size,
                    reference_hash=reference_hash,
                    candidate_hash=candidate_hash,
                    threshold=settings.kl_limit_nats,
                    seed=None,
                )
            )
            if kl.mean_kl_nats > settings.kl_limit_nats:
                reason_codes.append("kl_limit")

    reference_points = tuple(
        _plan_point(reference_episode) for reference_episode, _candidate, _ in pairs
    )
    candidate_points = tuple(
        _plan_point(candidate_episode) for _reference, candidate_episode, _ in pairs
    )
    try:
        mmd = mmd_permutation_test(
            production=reference_points,
            candidate=candidate_points,
            bandwidth=settings.mmd_bandwidth,
            permutations=settings.mmd_permutations,
            seed=seed,
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
    )
