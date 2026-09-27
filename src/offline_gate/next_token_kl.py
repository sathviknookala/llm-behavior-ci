from dataclasses import dataclass
import math
from statistics import fmean
from typing import Sequence


class NextTokenKLError(ValueError):
    pass


@dataclass(frozen=True)
class NextTokenKLResult:
    position_kl_nats: tuple[float, ...]
    mean_kl_nats: float


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
