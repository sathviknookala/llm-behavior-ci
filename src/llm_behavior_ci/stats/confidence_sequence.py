from __future__ import annotations

import math

from llm_behavior_ci.stats.evidence import Evidence, PairedSuccess, StatisticsError


def _finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StatisticsError(f"{name} must be a finite float")
    number = float(value)
    if not math.isfinite(number):
        raise StatisticsError(f"{name} must be a finite float")
    return number


def _alpha(value: float) -> float:
    number = _finite_float(value, "alpha")
    if not 0.0 < number < 1.0:
        raise StatisticsError("alpha must be between zero and one")
    return number


def stitched_radius(t: int, alpha: float, scale: float) -> float:
    """Stitched sub-Gaussian radius for a bounded mean.

    Null hypothesis: the mean, or the mean paired difference, equals null_mean when a null is set.
    Assumptions: observations lie in [lower, upper] and are iid; the radius is the stitched sub-Gaussian bound with proxy scale^2/4 and scale = upper - lower.
    Direction of harm: the interval excluding null_mean.
    Boundary: stitched_radius.
    Reset: clears the sum and count.
    Evidence: the interval endpoints in details. These tests check the formula. They do not establish coverage.
    """
    if not isinstance(t, int) or isinstance(t, bool) or t < 1:
        raise StatisticsError("t must be an integer of at least 1")
    alpha_value = _alpha(alpha)
    scale_value = _finite_float(scale, "scale")
    if scale_value <= 0.0:
        raise StatisticsError("scale must be positive")
    log_term = math.log(1.0 / alpha_value) + math.log(math.log(math.e * t))
    return math.sqrt((scale_value * scale_value / 4.0) * (2.0 * log_term) / t)


class BoundedMeanCS:
    """Stitched confidence sequence for a bounded mean.

    Null hypothesis: the mean, or the mean paired difference, equals null_mean when a null is set.
    Assumptions: observations lie in [lower, upper] and are iid; the radius is the stitched sub-Gaussian bound with proxy scale^2/4 and scale = upper - lower.
    Direction of harm: the interval excluding null_mean.
    Boundary: stitched_radius.
    Reset: clears the sum and count.
    Evidence: the interval endpoints in details. These tests check the formula. They do not establish coverage.
    """

    def __init__(
        self,
        *,
        alpha: float,
        lower: float,
        upper: float,
        null_mean: float | None = None,
    ) -> None:
        lower_value = _finite_float(lower, "lower")
        upper_value = _finite_float(upper, "upper")
        if upper_value <= lower_value:
            raise StatisticsError("upper must be greater than lower")
        self._alpha = _alpha(alpha)
        self._lower = lower_value
        self._upper = upper_value
        if null_mean is None:
            self._null_mean: float | None = None
        else:
            null_value = _finite_float(null_mean, "null_mean")
            if not lower_value <= null_value <= upper_value:
                raise StatisticsError("null_mean must lie inside [lower, upper]")
            self._null_mean = null_value
        self._total = 0.0
        self._sample_size = 0

    def _evidence(self, estimate: float, boundary: float) -> Evidence:
        alarm = False
        if self._null_mean is not None:
            alarm = (
                estimate - boundary > self._null_mean
                or estimate + boundary < self._null_mean
            )
        return Evidence(
            method="bounded_mean_cs",
            estimate=estimate,
            sample_size=self._sample_size,
            alarm=alarm,
            boundary=boundary,
            p_value=None,
            details=(
                ("lower_end", estimate - boundary),
                ("upper_end", estimate + boundary),
            ),
        )

    def update(self, observation: float) -> Evidence:
        value = _finite_float(observation, "observation")
        if not self._lower <= value <= self._upper:
            raise StatisticsError("observation is outside [lower, upper]")
        self._total += value
        self._sample_size += 1
        estimate = self._total / self._sample_size
        boundary = stitched_radius(
            self._sample_size, self._alpha, self._upper - self._lower
        )
        return self._evidence(estimate, boundary)

    def reset(self) -> None:
        self._total = 0.0
        self._sample_size = 0

    def snapshot(self) -> dict[str, object]:
        if self._sample_size == 0:
            estimate = 0.0
            alarm = False
        else:
            estimate = self._total / self._sample_size
            boundary = stitched_radius(
                self._sample_size, self._alpha, self._upper - self._lower
            )
            alarm = self._evidence(estimate, boundary).alarm
        return {
            "method": "bounded_mean_cs",
            "estimate": estimate,
            "sample_size": self._sample_size,
            "alarm": alarm,
        }


class PairedDifferenceCS:
    """Stitched confidence sequence for a paired difference.

    Null hypothesis: the mean, or the mean paired difference, equals null_mean when a null is set.
    Assumptions: observations lie in [lower, upper] and are iid; the radius is the stitched sub-Gaussian bound with proxy scale^2/4 and scale = upper - lower.
    Direction of harm: the interval excluding null_mean.
    Boundary: stitched_radius.
    Reset: clears the sum and count.
    Evidence: the interval endpoints in details. These tests check the formula. They do not establish coverage.
    """

    def __init__(
        self,
        *,
        alpha: float,
        lower: float,
        upper: float,
        null_mean: float,
    ) -> None:
        self._inner = BoundedMeanCS(
            alpha=alpha,
            lower=lower,
            upper=upper,
            null_mean=null_mean,
        )

    def update(self, observation: PairedSuccess) -> Evidence:
        difference = float(observation.candidate) - float(observation.reference)
        evidence = self._inner.update(difference)
        return Evidence(
            method="paired_difference_cs",
            estimate=evidence.estimate,
            sample_size=evidence.sample_size,
            alarm=evidence.alarm,
            boundary=evidence.boundary,
            p_value=evidence.p_value,
            details=evidence.details,
        )

    def reset(self) -> None:
        self._inner.reset()

    def snapshot(self) -> dict[str, object]:
        snapshot = dict(self._inner.snapshot())
        snapshot["method"] = "paired_difference_cs"
        return snapshot
