"""Full-vocabulary teacher-forced plan scoring for truncation checks.

Scores one frozen plan through an injected scorer (tests inject fakes; a
Transformers forward stays behind that seam and is not imported until a
caller invokes a scorer that loads it). Top-k at 20 and at a larger caller
k are cuts of the same full distribution. Numeric truncation error is
delegated to ``compare_plan_kl``; this module does not choose full vs
top-k or write a protocol threshold.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from llm_behavior_ci.experiments.validation import (
    KLApproximationReport,
    KLPositionSample,
    compare_plan_kl,
    default_results_root,
)
from llm_behavior_ci.records import TokenLogprob
from llm_behavior_ci.runtime.scoring import (
    PositionAlignmentError,
    SupportAlignmentError,
    fingerprint_messages,
)

DEFAULT_TOP_K = 20


class FullVocabError(ValueError):
    pass


class FullVocabScorerResult(Protocol):
    """One scorer return: dense full-vocabulary log-probabilities.

    ``position_log_probabilities[i][token_id]`` is the log-probability of
    ``token_id`` at forced-plan position ``i``. Length of each row equals
    ``vocabulary_size``. ``forced_token_ids[i]`` is the token actually
    forced at that position (the plan token identity used for alignment
    with a vLLM ``teacher_force_plan`` table).
    """

    @property
    def vocabulary_size(self) -> int: ...

    @property
    def forced_token_ids(self) -> tuple[int, ...]: ...

    @property
    def position_log_probabilities(self) -> tuple[tuple[float, ...], ...]: ...


@dataclass(frozen=True)
class DenseFullVocabResult:
    """Concrete scorer payload used by fakes and local collectors."""

    vocabulary_size: int
    forced_token_ids: tuple[int, ...]
    position_log_probabilities: tuple[tuple[float, ...], ...]


FullVocabScorer = Callable[
    ...,
    FullVocabScorerResult,
]


@dataclass(frozen=True)
class FullVocabPlanScore:
    """Aligned full-vocabulary teacher-forced score for one frozen plan."""

    vocabulary_size: int
    input_prefix_fingerprint: str
    forced_token_ids: tuple[int, ...]
    position_log_probabilities: tuple[tuple[float, ...], ...]
    top_k_default: tuple[tuple[TokenLogprob, ...], ...]
    top_k_larger: tuple[tuple[TokenLogprob, ...], ...]
    larger_k: int

    @property
    def position_count(self) -> int:
        return len(self.forced_token_ids)


def score_frozen_plan(
    *,
    messages: Sequence[Mapping[str, str]],
    plan_text: str,
    scorer: FullVocabScorer,
    larger_k: int,
    default_k: int = DEFAULT_TOP_K,
) -> FullVocabPlanScore:
    """Teacher-force one frozen plan through an injected full-vocab scorer.

    ``messages`` and ``plan_text`` match the vLLM ``teacher_force_plan`` /
    ``teacher_force_payload`` contract: the conversation prefix without the
    plan, plus the frozen plan text. Does not call vLLM or load weights.
    """

    if plan_text == "":
        raise FullVocabError("plan_text is required for teacher forcing")
    if (
        not isinstance(default_k, int)
        or isinstance(default_k, bool)
        or default_k <= 0
    ):
        raise FullVocabError("default_k must be a positive int")
    if (
        not isinstance(larger_k, int)
        or isinstance(larger_k, bool)
        or larger_k <= default_k
    ):
        raise FullVocabError(f"larger_k must be an int greater than {default_k}")
    fingerprint = fingerprint_messages(messages)
    raw = scorer(messages=list(messages), plan_text=plan_text)
    vocabulary_size = _positive_int(raw.vocabulary_size, "vocabulary_size")
    forced = tuple(int(token_id) for token_id in raw.forced_token_ids)
    rows = tuple(
        tuple(float(value) for value in row)
        for row in raw.position_log_probabilities
    )
    if not forced or not rows:
        raise FullVocabError("scorer returned no forced-plan positions")
    if len(forced) != len(rows):
        raise FullVocabError(
            "forced_token_ids and position_log_probabilities length differ"
        )
    if larger_k > vocabulary_size or default_k > vocabulary_size:
        raise FullVocabError("requested top-k exceeds vocabulary_size")
    for index, (token_id, row) in enumerate(zip(forced, rows, strict=True)):
        if len(row) != vocabulary_size:
            raise FullVocabError(
                f"position {index} support length differs from vocabulary_size"
            )
        if token_id < 0 or token_id >= vocabulary_size:
            raise FullVocabError(
                f"forced token id out of vocabulary at position {index}"
            )
    return FullVocabPlanScore(
        vocabulary_size=vocabulary_size,
        input_prefix_fingerprint=fingerprint,
        forced_token_ids=forced,
        position_log_probabilities=rows,
        top_k_default=tuple(top_k_slice(row, default_k) for row in rows),
        top_k_larger=tuple(top_k_slice(row, larger_k) for row in rows),
        larger_k=larger_k,
    )


def top_k_slice(
    log_probabilities: Sequence[float],
    k: int,
) -> tuple[TokenLogprob, ...]:
    """The top-k cut of one full-vocabulary position, ranked by log-prob.

    Ties break toward the smaller token id, matching ``compare_plan_kl``.
    """

    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise FullVocabError("k must be a positive int")
    if k > len(log_probabilities):
        raise FullVocabError("k exceeds the supplied support")
    indexes = _top_indexes(log_probabilities, k)
    return tuple(
        TokenLogprob(
            token_id=token_id,
            logprob=float(log_probabilities[token_id]),
            rank=rank,
        )
        for rank, token_id in enumerate(indexes)
    )


def assert_aligned_with_vllm_topk(
    score: FullVocabPlanScore,
    vllm_positions: Sequence[Sequence[TokenLogprob]],
) -> None:
    """Fail closed when vLLM top-k positions do not match this forced plan.

    Checks position count and the forced token id at each position (vLLM
    rank-0). Does not pad, invent, or reorder log-probabilities.
    """

    if len(score.forced_token_ids) != len(vllm_positions):
        raise PositionAlignmentError(
            "full-vocabulary and vLLM scored a different number of positions"
        )
    for index, (forced_id, position) in enumerate(
        zip(score.forced_token_ids, vllm_positions, strict=True)
    ):
        if not position:
            raise SupportAlignmentError(
                f"vLLM support is empty at position {index}"
            )
        if int(position[0].token_id) != int(forced_id):
            raise SupportAlignmentError(
                f"forced token id differs at position {index}"
            )


def kl_position_sample(
    production: FullVocabPlanScore,
    candidate: FullVocabPlanScore,
) -> KLPositionSample:
    """Build the sample type ``compare_plan_kl`` already scores."""

    _pair_aligned(production, candidate)
    return KLPositionSample(
        production_log_probabilities=production.position_log_probabilities,
        candidate_log_probabilities=candidate.position_log_probabilities,
    )


def compare_plan_kl_sample_document(
    pairs: Sequence[tuple[FullVocabPlanScore, FullVocabPlanScore]],
) -> dict[str, object]:
    """JSON document accepted by ``compare_plan_kl`` / ``_load_kl_samples``."""

    samples: list[dict[str, object]] = []
    for production, candidate in pairs:
        sample = kl_position_sample(production, candidate)
        samples.append(
            {
                "production": [
                    list(row) for row in sample.production_log_probabilities
                ],
                "candidate": [
                    list(row) for row in sample.candidate_log_probabilities
                ],
            }
        )
    return {"samples": samples}


def score_truncation_error(
    production: FullVocabPlanScore,
    candidate: FullVocabPlanScore,
    *,
    top_k: int,
    provenance: str = "supplied_sample",
) -> KLApproximationReport:
    """Record truncation error only through ``compare_plan_kl``."""

    sample = kl_position_sample(production, candidate)
    return compare_plan_kl(
        (sample,),
        top_k=top_k,
        provenance=provenance,
        vocabulary_size=production.vocabulary_size,
    )


def public_full_vocab_aggregate(
    score: FullVocabPlanScore,
    *,
    report: KLApproximationReport | None = None,
) -> dict[str, object]:
    """Public aggregate without plan text or token strings."""

    document: dict[str, object] = {
        "vocabulary_size": score.vocabulary_size,
        "position_count": score.position_count,
        "larger_k": score.larger_k,
        "default_top_k": DEFAULT_TOP_K,
        "input_prefix_fingerprint": score.input_prefix_fingerprint,
    }
    if report is not None:
        document["mean_signed_error"] = report.mean_signed_error
        document["mean_absolute_error"] = report.mean_absolute_error
        document["max_absolute_error"] = report.max_absolute_error
        document["top_k"] = report.top_k
        document["status"] = report.status
        document["approximation"] = report.approximation
        document["gpu_floor_measured"] = report.gpu_floor_measured
    return document


def write_local_kl_sample(
    path: Path,
    pairs: Sequence[tuple[FullVocabPlanScore, FullVocabPlanScore]],
    *,
    results_root: Path | None = None,
) -> None:
    """Write a local sample file for ``compare_plan_kl``. Refuses ``results/``."""

    root = results_root if results_root is not None else default_results_root()
    resolved = path.resolve()
    base = root.resolve()
    if resolved == base or base in resolved.parents:
        raise FullVocabError("refusing to write under results/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            compare_plan_kl_sample_document(pairs),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _pair_aligned(
    production: FullVocabPlanScore,
    candidate: FullVocabPlanScore,
) -> None:
    if production.vocabulary_size != candidate.vocabulary_size:
        raise FullVocabError("production and candidate vocabulary_size differ")
    if production.input_prefix_fingerprint != candidate.input_prefix_fingerprint:
        raise PositionAlignmentError(
            "production and candidate were not teacher-forced over the same "
            "input prefix"
        )
    if len(production.forced_token_ids) != len(candidate.forced_token_ids):
        raise PositionAlignmentError(
            "production and candidate scored a different number of positions"
        )
    if production.forced_token_ids != candidate.forced_token_ids:
        raise SupportAlignmentError(
            "production and candidate forced token ids differ"
        )


def _top_indexes(log_probabilities: Sequence[float], top_k: int) -> tuple[int, ...]:
    ranked = sorted(
        range(len(log_probabilities)),
        key=lambda index: (-log_probabilities[index], index),
    )
    return tuple(ranked[:top_k])


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise FullVocabError(f"{name} must be a positive int")
    return value
