from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence


class MMDError(ValueError):
    pass


@dataclass(frozen=True)
class MMDResult:
    mmd_squared: float
    p_value: float
    bandwidth: float
    permutations: int
    production_size: int
    candidate_size: int


def _points(
    samples: Sequence[Sequence[float]], name: str
) -> tuple[tuple[float, ...], ...]:
    converted = tuple(tuple(float(value) for value in sample) for sample in samples)
    if len(converted) < 2:
        raise MMDError(f"{name} must contain at least two samples")
    dimension = len(converted[0])
    if dimension == 0:
        raise MMDError(f"{name} samples are empty")
    if any(len(sample) != dimension for sample in converted):
        raise MMDError(f"{name} samples have inconsistent dimensions")
    if not all(math.isfinite(value) for sample in converted for value in sample):
        raise MMDError(f"{name} contains a non-finite value")
    return converted


def _rbf(
    left: Sequence[float],
    right: Sequence[float],
    kernel_denominator: float,
) -> float:
    squared_distance = math.fsum(
        (left_value - right_value) ** 2
        for left_value, right_value in zip(left, right, strict=True)
    )
    return math.exp(-squared_distance / kernel_denominator)


def _mmd_squared(
    production: Sequence[Sequence[float]],
    candidate: Sequence[Sequence[float]],
    kernel_denominator: float,
) -> float:
    production_size = len(production)
    candidate_size = len(candidate)
    production_term = math.fsum(
        _rbf(left, right, kernel_denominator)
        for left in production
        for right in production
    ) / (production_size**2)
    candidate_term = math.fsum(
        _rbf(left, right, kernel_denominator)
        for left in candidate
        for right in candidate
    ) / (candidate_size**2)
    cross_term = math.fsum(
        _rbf(left, right, kernel_denominator)
        for left in production
        for right in candidate
    ) / (production_size * candidate_size)
    return max(production_term + candidate_term - 2.0 * cross_term, 0.0)


def cluster_swap_bits(
    clusters: Sequence[str],
    rng: random.Random,
) -> tuple[int, ...]:
    """One paired-swap bit per pair, constant inside a cluster id.

    Null hypothesis: the two samples are equal under paired exchangeability.
    Assumptions: cluster mode preserves within-cluster dependence by swapping whole clusters.
    Direction of harm: larger MMD.
    Boundary: permutation p-value (exceedances + 1) / (permutations + 1), unchanged.
    Reset: stateless.
    Evidence: p_value.
    """
    bits_by_label: dict[str, int] = {}
    labels: list[str] = []
    for label in clusters:
        if not isinstance(label, str) or label == "":
            raise MMDError("cluster labels must be non-empty")
        labels.append(label)
        if label not in bits_by_label:
            bits_by_label[label] = rng.getrandbits(1)
    return tuple(bits_by_label[label] for label in labels)


def mmd_permutation_test(
    production: Sequence[Sequence[float]],
    candidate: Sequence[Sequence[float]],
    *,
    bandwidth: float,
    permutations: int,
    seed: int,
    clusters: Sequence[str] | None = None,
) -> MMDResult:
    """Paired permutation test for maximum mean discrepancy.

    Null hypothesis: the two samples are equal under paired exchangeability.
    Assumptions: cluster mode preserves within-cluster dependence by swapping whole clusters.
    Direction of harm: larger MMD.
    Boundary: permutation p-value (exceedances + 1) / (permutations + 1), unchanged.
    Reset: stateless.
    Evidence: p_value.
    """
    production_points = _points(production, "production")
    candidate_points = _points(candidate, "candidate")
    if len(production_points[0]) != len(candidate_points[0]):
        raise MMDError("production and candidate dimensions differ")
    if len(production_points) != len(candidate_points):
        raise MMDError("production and candidate must contain aligned pairs")
    if not math.isfinite(bandwidth) or bandwidth <= 0.0:
        raise MMDError("bandwidth must be finite and positive")
    kernel_denominator = 2.0 * bandwidth * bandwidth
    if (
        not math.isfinite(kernel_denominator)
        or kernel_denominator <= 0.0
    ):
        raise MMDError("bandwidth is outside the representable range")
    if (
        not isinstance(permutations, int)
        or isinstance(permutations, bool)
        or permutations < 1
    ):
        raise MMDError("permutations must be a positive integer")
    if clusters is not None:
        if len(clusters) != len(production_points):
            raise MMDError("clusters must have one label per pair")
        for label in clusters:
            if not isinstance(label, str) or label == "":
                raise MMDError("cluster labels must be non-empty")

    observed = _mmd_squared(
        production_points, candidate_points, kernel_denominator
    )
    random_generator = random.Random(seed)
    exceedances = 0

    for _ in range(permutations):
        permuted_production = []
        permuted_candidate = []
        if clusters is None:
            for production_point, candidate_point in zip(
                production_points, candidate_points, strict=True
            ):
                if random_generator.getrandbits(1):
                    permuted_production.append(candidate_point)
                    permuted_candidate.append(production_point)
                else:
                    permuted_production.append(production_point)
                    permuted_candidate.append(candidate_point)
        else:
            bits = cluster_swap_bits(clusters, random_generator)
            for bit, production_point, candidate_point in zip(
                bits, production_points, candidate_points, strict=True
            ):
                if bit:
                    permuted_production.append(candidate_point)
                    permuted_candidate.append(production_point)
                else:
                    permuted_production.append(production_point)
                    permuted_candidate.append(candidate_point)
        permuted = _mmd_squared(
            permuted_production,
            permuted_candidate,
            kernel_denominator,
        )
        if permuted >= observed - 1e-15:
            exceedances += 1

    return MMDResult(
        mmd_squared=observed,
        p_value=(exceedances + 1) / (permutations + 1),
        bandwidth=bandwidth,
        permutations=permutations,
        production_size=len(production_points),
        candidate_size=len(candidate_points),
    )
