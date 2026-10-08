from __future__ import annotations

import math

from llm_behavior_ci.stats.evidence import Evidence, StatisticsError

_DIRECTIONS = frozenset({"above", "below"})


def _finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StatisticsError(f"{name} must be a finite float")
    number = float(value)
    if not math.isfinite(number):
        raise StatisticsError(f"{name} must be a finite float")
    return number


class BettingEDetector:
    """Predictable-lambda betting e-process for a bounded mean.

    Null hypothesis: the mean is at most null_mean for direction above, and at least null_mean for direction below.
    Assumptions: bounded iid observations and a predictable lambda, so the wealth is an e-process under that null. The bet is capped so every multiplier stays nonnegative on [lower, upper]: lambda <= 1/(null_mean - lower) above, and -lambda <= 1/(upper - null_mean) below (Waudby-Smith and Ramdas, 2024).
    Direction of harm: above or below, as configured.
    Boundary: wealth crossing 1/alpha.
    Reset: wealth returns to 1.
    Evidence: alarm and wealth. Not a simulated type I error check.
    """

    def __init__(
        self,
        *,
        null_mean: float,
        alpha: float,
        lower: float,
        upper: float,
        direction: str,
    ) -> None:
        if direction not in _DIRECTIONS:
            raise StatisticsError("direction must be above or below")
        lower_value = _finite_float(lower, "lower")
        upper_value = _finite_float(upper, "upper")
        null_value = _finite_float(null_mean, "null_mean")
        alpha_value = _finite_float(alpha, "alpha")
        if not 0.0 < alpha_value < 1.0:
            raise StatisticsError("alpha must be between zero and one")
        if not lower_value < null_value < upper_value:
            raise StatisticsError("null_mean must lie strictly inside (lower, upper)")
        self._null_mean = null_value
        self._alpha = alpha_value
        self._lower = lower_value
        self._upper = upper_value
        self._direction = direction
        self._wealth = 1.0
        self._sample_size = 0
        self._next_lambda = 0.0
        self._running_sum = 0.0

    def update(self, observation: float) -> Evidence:
        value = _finite_float(observation, "observation")
        if not self._lower <= value <= self._upper:
            raise StatisticsError("observation is outside [lower, upper]")
        multiplier = 1.0 + self._next_lambda * (value - self._null_mean)
        if multiplier < 0.0:
            raise StatisticsError("betting multiplier is negative")
        self._wealth *= multiplier
        self._sample_size += 1
        self._running_sum += value
        running_mean = self._running_sum / self._sample_size
        span = self._upper - self._lower
        if self._direction == "above":
            gap = running_mean - self._null_mean
            cap = 1.0 / (self._null_mean - self._lower)
            self._next_lambda = 0.0 if gap <= 0.0 else min(gap / span, cap)
        else:
            gap = self._null_mean - running_mean
            cap = 1.0 / (self._upper - self._null_mean)
            self._next_lambda = 0.0 if gap <= 0.0 else -min(gap / span, cap)
        return Evidence(
            method="betting_e_detector",
            estimate=running_mean,
            sample_size=self._sample_size,
            alarm=self._wealth >= 1.0 / self._alpha,
            boundary=1.0 / self._alpha,
            p_value=None,
            details=(("wealth", self._wealth), ("lambda", self._next_lambda)),
        )

    def reset(self) -> None:
        self._wealth = 1.0
        self._sample_size = 0
        self._next_lambda = 0.0
        self._running_sum = 0.0

    def snapshot(self) -> dict[str, object]:
        estimate = (
            0.0 if self._sample_size == 0 else self._running_sum / self._sample_size
        )
        return {
            "method": "betting_e_detector",
            "estimate": estimate,
            "sample_size": self._sample_size,
            "alarm": self._wealth >= 1.0 / self._alpha,
            "wealth": self._wealth,
        }
