from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


class ChiSquareError(ValueError):
    pass


@dataclass(frozen=True)
class ChiSquareResult:
    statistic: float
    p_value: float
    degrees_of_freedom: int


def _observed_counts(values: Sequence[int], name: str) -> tuple[int, ...]:
    counts: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ChiSquareError(f"{name} counts must be non-negative integers")
        counts.append(value)
    return tuple(counts)


def _expected_counts(values: Sequence[float]) -> tuple[float, ...]:
    converted: list[float] = []
    for value in values:
        if isinstance(value, bool):
            raise ChiSquareError("expected counts must be finite and positive")
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise ChiSquareError("expected counts must be finite and positive")
        converted.append(number)
    return tuple(converted)


def _pearson_statistic(
    observed: Sequence[float],
    expected: Sequence[float],
) -> float:
    return math.fsum(
        (observed_count - expected_count) ** 2 / expected_count
        for observed_count, expected_count in zip(observed, expected, strict=True)
    )


def _regularized_gamma_p_series(a: float, x: float) -> float:
    if x == 0.0:
        return 0.0
    term = 1.0 / a
    total = term
    ap = a
    for _ in range(200):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < 1e-14:
            break
    value = total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    return min(1.0, max(0.0, value))


def _regularized_gamma_q_cf(a: float, x: float) -> float:
    smallest = 1e-300
    b = x + 1.0 - a
    c = 1.0 / smallest
    d = 1.0 / b
    fraction = d
    for index in range(1, 201):
        numerator = -index * (index - a)
        b += 2.0
        d = numerator * d + b
        if abs(d) < smallest:
            d = smallest
        c = b + numerator / c
        if abs(c) < smallest:
            c = smallest
        d = 1.0 / d
        delta = d * c
        fraction *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    value = fraction * math.exp(-x + a * math.log(x) - math.lgamma(a))
    return min(1.0, max(0.0, value))


def chi_square_survival(statistic: float, degrees_of_freedom: int) -> float:
    """Upper-tail chi-square survival from the regularized gamma function.

    Null hypothesis: counts follow the supplied expected frequencies, or the two rows are homogeneous.
    Assumptions: the chi-square approximation, positive expected counts.
    Direction of harm: a larger statistic.
    Boundary: the survival function.
    Reset: stateless.
    Evidence: p_value.
    """
    if statistic == 0.0:
        return 1.0
    if (
        isinstance(statistic, bool)
        or isinstance(degrees_of_freedom, bool)
        or not isinstance(degrees_of_freedom, int)
        or statistic < 0.0
        or degrees_of_freedom < 1
    ):
        raise ChiSquareError("statistic must be non-negative and degrees_of_freedom at least 1")
    shape = degrees_of_freedom / 2.0
    scale = statistic / 2.0
    if scale < shape + 1.0:
        value = 1.0 - _regularized_gamma_p_series(shape, scale)
    else:
        value = _regularized_gamma_q_cf(shape, scale)
    return min(1.0, max(0.0, value))


def chi_square_goodness_of_fit(
    observed: Sequence[int],
    expected: Sequence[float],
) -> ChiSquareResult:
    """Pearson goodness-of-fit chi-square test.

    Null hypothesis: counts follow the supplied expected frequencies, or the two rows are homogeneous.
    Assumptions: the chi-square approximation, positive expected counts.
    Direction of harm: a larger statistic.
    Boundary: the survival function.
    Reset: stateless.
    Evidence: p_value.
    """
    observed_counts = _observed_counts(observed, "observed")
    expected_counts = _expected_counts(expected)
    if len(observed_counts) != len(expected_counts) or len(observed_counts) < 2:
        raise ChiSquareError("observed and expected must have the same length of at least 2")
    if sum(observed_counts) <= 0 or math.fsum(expected_counts) <= 0.0:
        raise ChiSquareError("observed and expected sums must be positive")
    statistic = _pearson_statistic(observed_counts, expected_counts)
    degrees_of_freedom = len(observed_counts) - 1
    return ChiSquareResult(
        statistic=statistic,
        p_value=chi_square_survival(statistic, degrees_of_freedom),
        degrees_of_freedom=degrees_of_freedom,
    )


def chi_square_homogeneity(
    left: Sequence[int],
    right: Sequence[int],
) -> ChiSquareResult:
    """Two-row chi-square test of homogeneity.

    Null hypothesis: counts follow the supplied expected frequencies, or the two rows are homogeneous.
    Assumptions: the chi-square approximation, positive expected counts.
    Direction of harm: a larger statistic.
    Boundary: the survival function.
    Reset: stateless.
    Evidence: p_value.
    """
    left_counts = _observed_counts(left, "left")
    right_counts = _observed_counts(right, "right")
    if len(left_counts) != len(right_counts) or len(left_counts) < 2:
        raise ChiSquareError("both rows must have the same length of at least 2")
    left_total = sum(left_counts)
    right_total = sum(right_counts)
    if left_total <= 0 or right_total <= 0:
        raise ChiSquareError("row totals must be positive")
    grand_total = left_total + right_total
    observed: list[float] = []
    expected: list[float] = []
    for left_count, right_count in zip(left_counts, right_counts, strict=True):
        column_total = left_count + right_count
        if column_total <= 0:
            raise ChiSquareError("column totals must be positive")
        observed.append(float(left_count))
        expected.append(left_total * column_total / grand_total)
        observed.append(float(right_count))
        expected.append(right_total * column_total / grand_total)
    statistic = _pearson_statistic(observed, expected)
    degrees_of_freedom = len(left_counts) - 1
    return ChiSquareResult(
        statistic=statistic,
        p_value=chi_square_survival(statistic, degrees_of_freedom),
        degrees_of_freedom=degrees_of_freedom,
    )
