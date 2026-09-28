from __future__ import annotations

from dataclasses import dataclass
import math
import random
from statistics import fmean
from typing import Sequence


class PairedBootstrapError(ValueError):
    pass


@dataclass(frozen=True)
class PairedBootstrapResult:
    candidate_mean: float
    production_mean: float
    mean_difference: float
    confidence_low: float
    confidence_high: float
    confidence_level: float
    resamples: int


def _finite_values(values: Sequence[float], name: str) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if not converted:
        raise PairedBootstrapError(f"{name} is empty")
    if not all(math.isfinite(value) for value in converted):
        raise PairedBootstrapError(f"{name} contains a non-finite value")
    return converted


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    return lower + (upper - lower) * (position - lower_index)


def _cluster_labels(clusters: Sequence[str]) -> tuple[str, ...]:
    labels = tuple(clusters)
    if not labels:
        raise PairedBootstrapError("clusters is empty")
    for label in labels:
        if not isinstance(label, str) or label == "":
            raise PairedBootstrapError("cluster labels must be non-empty")
    return labels


def _label_indexes(clusters: Sequence[str]) -> tuple[tuple[str, ...], dict[str, tuple[int, ...]]]:
    labels = _cluster_labels(clusters)
    unique: list[str] = []
    seen: set[str] = set()
    grouped: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        if label not in seen:
            unique.append(label)
            seen.add(label)
            grouped[label] = []
        grouped[label].append(index)
    return tuple(unique), {label: tuple(indexes) for label, indexes in grouped.items()}


def cluster_draw_indexes(
    clusters: Sequence[str],
    rng: random.Random,
) -> tuple[int, ...]:
    """Draw one clustered bootstrap replicate's expanded indexes.

    Null hypothesis: the paired mean difference is zero.
    Assumptions: pairs are independent across clusters and dependent within a cluster.
    Direction of harm: a negative mean difference means the candidate is lower.
    Boundary: the percentile interval.
    Reset: this function has no state.
    Evidence: the returned interval, not an alarm.
    """
    unique, grouped = _label_indexes(clusters)
    expanded: list[int] = []
    for _ in range(len(unique)):
        label = unique[rng.randrange(len(unique))]
        expanded.extend(grouped[label])
    return tuple(expanded)


def paired_bootstrap(
    candidate: Sequence[float],
    production: Sequence[float],
    *,
    confidence_level: float,
    resamples: int,
    seed: int,
) -> PairedBootstrapResult:
    """Percentile interval for a paired mean difference.

    Null hypothesis: the paired mean difference is zero.
    Assumptions: pairs are independent.
    Direction of harm: a negative mean difference means the candidate is lower.
    Boundary: the percentile interval.
    Reset: this function has no state.
    Evidence: the returned interval, not an alarm.
    """
    candidate_values = _finite_values(candidate, "candidate")
    production_values = _finite_values(production, "production")
    if len(candidate_values) != len(production_values):
        raise PairedBootstrapError("candidate and production must have equal length")
    if not 0.0 < confidence_level < 1.0:
        raise PairedBootstrapError("confidence_level must be between zero and one")
    if (
        not isinstance(resamples, int)
        or isinstance(resamples, bool)
        or resamples < 1
    ):
        raise PairedBootstrapError("resamples must be a positive integer")

    differences = tuple(
        candidate_value - production_value
        for candidate_value, production_value in zip(
            candidate_values, production_values, strict=True
        )
    )
    random_generator = random.Random(seed)
    sample_size = len(differences)
    bootstrap_differences = sorted(
        fmean(
            differences[random_generator.randrange(sample_size)]
            for _ in range(sample_size)
        )
        for _ in range(resamples)
    )
    tail_probability = (1.0 - confidence_level) / 2.0

    return PairedBootstrapResult(
        candidate_mean=fmean(candidate_values),
        production_mean=fmean(production_values),
        mean_difference=fmean(differences),
        confidence_low=_quantile(bootstrap_differences, tail_probability),
        confidence_high=_quantile(
            bootstrap_differences, 1.0 - tail_probability
        ),
        confidence_level=confidence_level,
        resamples=resamples,
    )


def clustered_paired_bootstrap(
    candidate: Sequence[float],
    production: Sequence[float],
    clusters: Sequence[str],
    *,
    confidence_level: float,
    resamples: int,
    seed: int,
) -> PairedBootstrapResult:
    """Percentile interval for a cluster-resampled paired mean difference.

    Null hypothesis: the paired mean difference is zero.
    Assumptions: pairs are independent across clusters and dependent within a cluster.
    Direction of harm: a negative mean difference means the candidate is lower.
    Boundary: the percentile interval.
    Reset: this function has no state.
    Evidence: the returned interval, not an alarm.
    """
    candidate_values = _finite_values(candidate, "candidate")
    production_values = _finite_values(production, "production")
    if len(candidate_values) != len(production_values):
        raise PairedBootstrapError("candidate and production must have equal length")
    labels = _cluster_labels(clusters)
    if len(labels) != len(candidate_values):
        raise PairedBootstrapError("clusters must have the same length as the pairs")
    if not 0.0 < confidence_level < 1.0:
        raise PairedBootstrapError("confidence_level must be between zero and one")
    if (
        not isinstance(resamples, int)
        or isinstance(resamples, bool)
        or resamples < 1
    ):
        raise PairedBootstrapError("resamples must be a positive integer")

    differences = tuple(
        candidate_value - production_value
        for candidate_value, production_value in zip(
            candidate_values, production_values, strict=True
        )
    )
    random_generator = random.Random(seed)
    bootstrap_differences = sorted(
        fmean(differences[index] for index in cluster_draw_indexes(labels, random_generator))
        for _ in range(resamples)
    )
    tail_probability = (1.0 - confidence_level) / 2.0

    return PairedBootstrapResult(
        candidate_mean=fmean(candidate_values),
        production_mean=fmean(production_values),
        mean_difference=fmean(differences),
        confidence_low=_quantile(bootstrap_differences, tail_probability),
        confidence_high=_quantile(
            bootstrap_differences, 1.0 - tail_probability
        ),
        confidence_level=confidence_level,
        resamples=resamples,
    )
