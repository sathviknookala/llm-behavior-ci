from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


class KSError(ValueError):
    pass


@dataclass(frozen=True)
class KSResult:
    statistic: float
    p_value: float
    sample_size: int


def _finite_sample(values: Sequence[float], name: str) -> tuple[float, ...]:
    if len(values) < 1:
        raise KSError(f"{name} is empty")
    converted: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise KSError(f"{name} contains a non-float value")
        number = float(value)
        if not math.isfinite(number):
            raise KSError(f"{name} contains a non-finite value")
        converted.append(number)
    return tuple(converted)


def kolmogorov_survival(z: float) -> float:
    """Asymptotic Kolmogorov survival function.

    Null hypothesis: the two samples are i.i.d. from the same continuous distribution.
    Assumptions: the asymptotic Kolmogorov approximation, not an exact small-sample p-value.
    Direction of harm: a larger statistic.
    Boundary: the survival function above.
    Reset: stateless.
    Evidence: p_value. This unit test does not establish type I error.
    """
    if z <= 0.0:
        return 1.0
    if z >= 10.0:
        return 0.0
    total = math.fsum(
        ((-1) ** (k - 1)) * math.exp(-2.0 * k * k * z * z)
        for k in range(1, 101)
    )
    return min(1.0, max(0.0, 2.0 * total))


def ks_two_sample(left: Sequence[float], right: Sequence[float]) -> KSResult:
    """Two-sample Kolmogorov–Smirnov statistic with an asymptotic p-value.

    Null hypothesis: the two samples are i.i.d. from the same continuous distribution.
    Assumptions: the asymptotic Kolmogorov approximation, not an exact small-sample p-value.
    Direction of harm: a larger statistic.
    Boundary: the survival function above.
    Reset: stateless.
    Evidence: p_value. This unit test does not establish type I error.
    """
    left_values = _finite_sample(left, "left")
    right_values = _finite_sample(right, "right")
    points = sorted(set(left_values) | set(right_values))
    left_size = len(left_values)
    right_size = len(right_values)
    statistic = 0.0
    for point in points:
        left_cdf = sum(1 for value in left_values if value <= point) / left_size
        right_cdf = sum(1 for value in right_values if value <= point) / right_size
        statistic = max(statistic, abs(left_cdf - right_cdf))
    if statistic == 0.0:
        p_value = 1.0
    else:
        effective = left_size * right_size / (left_size + right_size)
        p_value = kolmogorov_survival(statistic * math.sqrt(effective))
    return KSResult(
        statistic=statistic,
        p_value=p_value,
        sample_size=left_size + right_size,
    )
