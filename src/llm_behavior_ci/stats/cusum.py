from __future__ import annotations

import math

from llm_behavior_ci.stats.evidence import Evidence, StatisticsError

_DIRECTIONS = frozenset({"increase", "decrease", "two_sided"})


def _finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StatisticsError(f"{name} must be a finite float")
    number = float(value)
    if not math.isfinite(number):
        raise StatisticsError(f"{name} must be a finite float")
    return number


class CUSUM:
    """Page CUSUM recursion on a scalar stream.

    Null hypothesis: for increase, the mean is at most target.
    Assumptions: iid observations, Page's recursion, not an anytime-valid false-alarm guarantee.
    Direction of harm: increase, decrease, or either, as configured.
    Boundary: the caller-supplied threshold.
    Reset: clears S and the count, keeps parameters.
    Evidence: alarm when S crosses the threshold.
    """

    def __init__(
        self,
        *,
        target: float,
        slack: float,
        threshold: float,
        direction: str,
    ) -> None:
        if direction not in _DIRECTIONS:
            raise StatisticsError("direction must be increase, decrease, or two_sided")
        target_value = _finite_float(target, "target")
        slack_value = _finite_float(slack, "slack")
        threshold_value = _finite_float(threshold, "threshold")
        if slack_value < 0.0:
            raise StatisticsError("slack must be at least zero")
        if threshold_value <= 0.0:
            raise StatisticsError("threshold must be positive")
        self._target = target_value
        self._slack = slack_value
        self._threshold = threshold_value
        self._direction = direction
        self._s_increase = 0.0
        self._s_decrease = 0.0
        self._sample_size = 0

    def _estimate(self) -> float:
        if self._direction == "increase":
            return self._s_increase
        if self._direction == "decrease":
            return self._s_decrease
        return max(self._s_increase, self._s_decrease)

    def update(self, observation: float) -> Evidence:
        value = _finite_float(observation, "observation")
        self._s_increase = max(
            0.0, self._s_increase + value - self._target - self._slack
        )
        self._s_decrease = max(
            0.0, self._s_decrease + self._target - self._slack - value
        )
        self._sample_size += 1
        estimate = self._estimate()
        return Evidence(
            method="cusum",
            estimate=estimate,
            sample_size=self._sample_size,
            alarm=estimate >= self._threshold,
            boundary=self._threshold,
            p_value=None,
            details=(("target", self._target), ("slack", self._slack)),
        )

    def reset(self) -> None:
        self._s_increase = 0.0
        self._s_decrease = 0.0
        self._sample_size = 0

    def snapshot(self) -> dict[str, object]:
        estimate = self._estimate()
        return {
            "method": "cusum",
            "estimate": estimate,
            "sample_size": self._sample_size,
            "alarm": estimate >= self._threshold,
            "direction": self._direction,
        }
