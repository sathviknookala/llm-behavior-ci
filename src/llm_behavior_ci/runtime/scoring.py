from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from llm_behavior_ci.stats.kl import next_token_kl


class ScoringError(ValueError):
    pass


@dataclass(frozen=True)
class DistributionScore:
    kind: Literal["full", "top_k"]
    position_kl_nats: tuple[float, ...]
    mean_kl_nats: float


def score_full(
    production: Sequence[Sequence[float]],
    candidate: Sequence[Sequence[float]],
) -> DistributionScore:
    """Score aligned full-vocabulary next-token log-probabilities.

    This is full-vocabulary KL only when the caller passes aligned
    full-vocabulary log-probabilities. A selected token log-probability is
    not a valid input.
    """

    result = next_token_kl(production, candidate)
    return DistributionScore(
        kind="full",
        position_kl_nats=result.position_kl_nats,
        mean_kl_nats=result.mean_kl_nats,
    )


def score_top_k(
    production: Sequence[Sequence[float]],
    candidate: Sequence[Sequence[float]],
) -> DistributionScore:
    """Score truncated top-k next-token log-probabilities.

    This is a truncated approximation, not exact KL.
    """

    from llm_behavior_ci.stats import kl as kl_module

    result = kl_module.truncated_next_token_kl(production, candidate)
    if result.approximation != "top_k":
        raise ScoringError("truncated score is not labeled top_k")
    return DistributionScore(
        kind="top_k",
        position_kl_nats=result.position_kl_nats,
        mean_kl_nats=result.mean_kl_nats,
    )
