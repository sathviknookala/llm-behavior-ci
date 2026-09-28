from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence


class C2STError(ValueError):
    pass


@dataclass(frozen=True)
class C2STResult:
    accuracy: float
    p_value: float
    permutations: int


def _rows(samples: Sequence[Sequence[float]], name: str) -> tuple[tuple[float, ...], ...]:
    if len(samples) < 2:
        raise C2STError(f"{name} must contain at least two rows")
    converted: list[tuple[float, ...]] = []
    dimension: int | None = None
    for row in samples:
        values = tuple(float(value) for value in row)
        if dimension is None:
            dimension = len(values)
            if dimension == 0:
                raise C2STError(f"{name} rows are empty")
        elif len(values) != dimension:
            raise C2STError(f"{name} rows are ragged")
        if not all(math.isfinite(value) for value in values):
            raise C2STError(f"{name} contains a non-finite value")
        converted.append(values)
    return tuple(converted)


def _logistic(score: float) -> float:
    clipped = min(30.0, max(-30.0, score))
    return 1.0 / (1.0 + math.exp(-clipped))


def _dot(weights: Sequence[float], row: Sequence[float]) -> float:
    return math.fsum(
        weight * value for weight, value in zip(weights, row, strict=True)
    )


def _fit_accuracy(
    rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    steps: int,
    learning_rate: float,
    l2: float,
) -> float:
    weights = [0.0] * len(rows[0])
    bias = 0.0
    for _ in range(steps):
        for row, label in zip(rows, labels, strict=True):
            probability = _logistic(_dot(weights, row) + bias)
            residual = probability - label
            weights = [
                weight - learning_rate * (residual * value + l2 * weight)
                for weight, value in zip(weights, row, strict=True)
            ]
            bias -= learning_rate * residual
    correct = 0
    for row, label in zip(rows, labels, strict=True):
        probability = _logistic(_dot(weights, row) + bias)
        predicted = 1 if probability >= 0.5 else 0
        if predicted == label:
            correct += 1
    return correct / len(rows)


def classifier_two_sample_test(
    production: Sequence[Sequence[float]],
    candidate: Sequence[Sequence[float]],
    *,
    permutations: int,
    seed: int,
    steps: int = 200,
    learning_rate: float = 0.1,
    l2: float = 0.01,
) -> C2STResult:
    """In-sample logistic classifier two-sample test.

    Null hypothesis: the two samples are exchangeable.
    Assumptions: the statistic is in-sample accuracy of this fixed logistic procedure, not cross-validated accuracy.
    Direction of harm: accuracy above 0.5.
    Boundary: the permutation p-value.
    Reset: stateless.
    Evidence: p_value. This is not a power claim.
    """
    if (
        not isinstance(permutations, int)
        or isinstance(permutations, bool)
        or permutations < 1
    ):
        raise C2STError("permutations must be a positive integer")
    production_rows = _rows(production, "production")
    candidate_rows = _rows(candidate, "candidate")
    if len(production_rows[0]) != len(candidate_rows[0]):
        raise C2STError("production and candidate dimensions differ")
    rows = production_rows + candidate_rows
    labels = (0,) * len(production_rows) + (1,) * len(candidate_rows)
    observed = _fit_accuracy(
        rows,
        labels,
        steps=steps,
        learning_rate=learning_rate,
        l2=l2,
    )
    random_generator = random.Random(seed)
    exceedances = 0
    for _ in range(permutations):
        shuffled = list(labels)
        random_generator.shuffle(shuffled)
        accuracy = _fit_accuracy(
            rows,
            shuffled,
            steps=steps,
            learning_rate=learning_rate,
            l2=l2,
        )
        if accuracy >= observed - 1e-12:
            exceedances += 1
    return C2STResult(
        accuracy=observed,
        p_value=(exceedances + 1) / (permutations + 1),
        permutations=permutations,
    )
