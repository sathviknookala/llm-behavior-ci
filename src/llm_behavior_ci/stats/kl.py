from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import fmean
from typing import Sequence


class NextTokenKLError(ValueError):
    pass


class TruncatedKLError(ValueError):
    pass


@dataclass(frozen=True)
class NextTokenKLResult:
    position_kl_nats: tuple[float, ...]
    mean_kl_nats: float


@dataclass(frozen=True)
class TruncatedKLResult:
    position_kl_nats: tuple[float, ...]
    mean_kl_nats: float
    approximation: str
    vocabulary_size: int | None


def _normalize(
    log_probabilities: Sequence[float],
    role: str,
    position: int,
) -> tuple[float, ...]:
    values = tuple(float(value) for value in log_probabilities)
    if not values:
        raise NextTokenKLError(f"{role} position {position} is empty")
    if any(math.isnan(value) or value == math.inf for value in values):
        raise NextTokenKLError(
            f"{role} position {position} is not a log-probability distribution"
        )
    finite_values = tuple(value for value in values if math.isfinite(value))
    if not finite_values:
        raise NextTokenKLError(f"{role} position {position} has no probability mass")
    maximum = max(finite_values)
    log_normalizer = maximum + math.log(
        math.fsum(math.exp(value - maximum) for value in finite_values)
    )
    return tuple(
        value - log_normalizer if math.isfinite(value) else -math.inf
        for value in values
    )


def next_token_kl(
    production_log_probabilities: Sequence[Sequence[float]],
    candidate_log_probabilities: Sequence[Sequence[float]],
) -> NextTokenKLResult:
    """Inputs must be aligned log-probabilities over the same vocabulary positions. A single selected-token log-probability is not full-vocabulary KL.

    Null hypothesis: the two distributions are equal, so KL is zero.
    Assumptions: each position is renormalized independently; -inf means no mass.
    Direction of harm: larger KL.
    Boundary: none.
    Reset: stateless.
    Evidence: mean_kl_nats.
    """
    production_positions = tuple(production_log_probabilities)
    candidate_positions = tuple(candidate_log_probabilities)
    if not production_positions:
        raise NextTokenKLError("production has no token positions")
    if len(production_positions) != len(candidate_positions):
        raise NextTokenKLError(
            "production and candidate must have equal position counts"
        )

    position_kl_nats = []
    for position, (production, candidate) in enumerate(
        zip(production_positions, candidate_positions, strict=True)
    ):
        normalized_production = _normalize(production, "production", position)
        normalized_candidate = _normalize(candidate, "candidate", position)
        if len(normalized_production) != len(normalized_candidate):
            raise NextTokenKLError(
                f"production and candidate vocabularies differ at position {position}"
            )

        terms = []
        for reference, comparison in zip(
            normalized_production, normalized_candidate, strict=True
        ):
            if not math.isfinite(reference):
                continue
            if not math.isfinite(comparison):
                raise NextTokenKLError(
                    f"candidate has no mass on production support at position {position}"
                )
            probability = math.exp(reference)
            if probability == 0.0:
                continue
            terms.append(probability * (reference - comparison))

        divergence = math.fsum(terms)
        if divergence < -1e-12:
            raise NextTokenKLError(
                f"KL is negative at position {position}: {divergence}"
            )
        position_kl_nats.append(max(divergence, 0.0))

    position_values = tuple(position_kl_nats)
    return NextTokenKLResult(
        position_kl_nats=position_values,
        mean_kl_nats=fmean(position_values),
    )


def truncated_next_token_kl(
    production_log_probabilities: Sequence[Sequence[float]],
    candidate_log_probabilities: Sequence[Sequence[float]],
    *,
    vocabulary_size: int | None = None,
) -> TruncatedKLResult:
    """Top-k truncated next-token KL on aligned log-probabilities.

    Null hypothesis: the two distributions are equal, so KL is zero.
    Assumptions: each position is renormalized independently; -inf means no mass.
    Direction of harm: larger KL.
    Boundary: none.
    Reset: stateless.
    Evidence: mean_kl_nats.
    """
    production_positions = tuple(production_log_probabilities)
    candidate_positions = tuple(candidate_log_probabilities)
    longest = 0
    for position in (*production_positions, *candidate_positions):
        longest = max(longest, len(position))
    if vocabulary_size is not None:
        if (
            not isinstance(vocabulary_size, int)
            or isinstance(vocabulary_size, bool)
            or vocabulary_size < longest
        ):
            raise TruncatedKLError(
                "vocabulary_size must be an int at least as large as the longest support"
            )
    try:
        result = next_token_kl(production_positions, candidate_positions)
    except NextTokenKLError as error:
        raise TruncatedKLError(*error.args) from error
    return TruncatedKLResult(
        position_kl_nats=result.position_kl_nats,
        mean_kl_nats=result.mean_kl_nats,
        approximation="top_k",
        vocabulary_size=vocabulary_size,
    )
