"""Canonical construction of production and replay detectors."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

from llm_behavior_ci.config import (
    DISTRIBUTIONAL_CORRECTIONS,
    DISTRIBUTIONAL_SIGNALS,
    MONITOR_SIGNALS,
    StoppingRule,
)
from llm_behavior_ci.stats.adwin import ADWIN
from llm_behavior_ci.stats.chi_square import ChiSquareError, chi_square_homogeneity
from llm_behavior_ci.stats.confidence_sequence import BoundedMeanCS
from llm_behavior_ci.stats.cusum import CUSUM
from llm_behavior_ci.stats.e_detector import BettingEDetector
from llm_behavior_ci.stats.evidence import Detector, Evidence, PairedSuccess, StatisticsError
from llm_behavior_ci.stats.harmful_shift import HarmfulShiftTest
from llm_behavior_ci.stats.ks import KSError, ks_two_sample

BOUNDED_SIGNALS = frozenset(
    {"task_success", "requirement_fraction", "plan_quality_score"}
)
MONITOR_RULE_NAMES = frozenset({"cusum", "adwin", "e_detector", "bounded_mean_cs"})
WINDOW_KINDS = frozenset({"ks", "chi_square"})
CORRECTIONS = frozenset({"none", "bonferroni"})


class DetectorConstructionError(ValueError):
    pass


def signal_bounds(signal: str, baseline: float) -> tuple[float, float]:
    if signal not in MONITOR_SIGNALS:
        raise DetectorConstructionError("signal must be a monitor signal")
    if signal in BOUNDED_SIGNALS:
        return (0.0, 1.0)
    return (0.0, max(baseline * 4.0, 1.0))


def build_detector(
    rule: StoppingRule,
    *,
    signal: str,
    baseline: float,
) -> Detector:
    if not isinstance(rule, StoppingRule):
        raise DetectorConstructionError("rule must be a StoppingRule")
    if signal not in MONITOR_SIGNALS:
        raise DetectorConstructionError(
            f"unsupported signal/rule pair: {signal!r} with {rule.name!r}"
        )
    if rule.name not in MONITOR_RULE_NAMES:
        raise DetectorConstructionError(
            f"unsupported signal/rule pair: {signal!r} with {rule.name!r}"
        )
    if isinstance(baseline, bool) or not isinstance(baseline, (int, float)):
        raise DetectorConstructionError("baseline must be a finite float")
    baseline_value = float(baseline)
    if not math.isfinite(baseline_value):
        raise DetectorConstructionError("baseline must be a finite float")
    if rule.name == "cusum":
        if rule.threshold is None:
            raise DetectorConstructionError("cusum requires a threshold")
        direction = "decrease" if signal in BOUNDED_SIGNALS else "increase"
        return CUSUM(
            target=baseline_value,
            slack=0.0,
            threshold=rule.threshold,
            direction=direction,
        )
    if rule.name == "adwin":
        return ADWIN(delta=rule.alpha)
    lower, upper = signal_bounds(signal, baseline_value)
    if rule.name == "e_detector":
        if not lower < baseline_value < upper:
            raise DetectorConstructionError(
                "e_detector null_mean must lie strictly inside (lower, upper)"
            )
        direction = "below" if signal in BOUNDED_SIGNALS else "above"
        return BettingEDetector(
            null_mean=baseline_value,
            alpha=rule.alpha,
            lower=lower,
            upper=upper,
            direction=direction,
        )
    if rule.name == "bounded_mean_cs":
        return BoundedMeanCS(
            alpha=rule.alpha,
            lower=lower,
            upper=upper,
            null_mean=baseline_value,
        )
    raise DetectorConstructionError(
        f"unsupported signal/rule pair: {signal!r} with {rule.name!r}"
    )


def build_detectors(
    *,
    signals: tuple[str, ...],
    stopping_rules: tuple[StoppingRule, ...],
    baselines: Mapping[str, float],
) -> dict[str, list[Detector]]:
    if not isinstance(signals, tuple) or not signals:
        raise DetectorConstructionError("signals must be a non-empty tuple")
    if not isinstance(stopping_rules, tuple) or not stopping_rules:
        raise DetectorConstructionError("stopping_rules must be a non-empty tuple")
    if not isinstance(baselines, Mapping):
        raise DetectorConstructionError("baselines must map each signal to a float")
    detectors: dict[str, list[Detector]] = {}
    for signal in signals:
        if signal not in baselines:
            raise DetectorConstructionError("baselines must include every signal")
        built: list[Detector] = []
        for rule in stopping_rules:
            built.append(
                build_detector(rule, signal=signal, baseline=float(baselines[signal]))
            )
        detectors[signal] = built
    return detectors


def _histogram(values: Sequence[float], baseline: float) -> tuple[int, int]:
    low = 0
    high = 0
    for value in values:
        if value <= baseline:
            low += 1
        else:
            high += 1
    return (low, high)


class HourlyWindowDetector:
    def __init__(
        self,
        *,
        method: str,
        alpha: float,
        window_episodes: int,
        reference_sample: Sequence[float],
        baseline: float,
        correction: str,
        kind: str,
    ) -> None:
        self._method = method
        self._alpha = alpha
        self._window_episodes = window_episodes
        self._reference = tuple(float(item) for item in reference_sample)
        self._baseline = baseline
        self._correction = correction
        self._kind = kind
        self._window: list[float] = []
        self._looks = 0
        self._sample_size = 0

    def _p_value(self) -> float | None:
        if not self._window:
            return None
        if self._kind == "ks":
            try:
                return ks_two_sample(self._reference, self._window).p_value
            except KSError:
                return None
        left = _histogram(self._reference, self._baseline)
        right = _histogram(self._window, self._baseline)
        try:
            return chi_square_homogeneity(left, right).p_value
        except ChiSquareError:
            return None

    def update(self, observation: float) -> Evidence:
        self._window.append(float(observation))
        self._sample_size += 1
        alarm = False
        p_value: float | None = None
        if len(self._window) >= self._window_episodes:
            self._looks += 1
            p_value = self._p_value()
            if p_value is not None:
                if self._correction == "bonferroni":
                    adjusted = min(1.0, p_value * self._looks)
                    alarm = adjusted <= self._alpha
                else:
                    alarm = p_value <= self._alpha
            self._window.clear()
        return Evidence(
            method=self._method,
            estimate=0.0 if p_value is None else p_value,
            sample_size=self._sample_size,
            alarm=alarm,
            boundary=self._alpha,
            p_value=p_value,
            details=(),
        )

    def reset(self) -> None:
        self._window.clear()
        self._looks = 0
        self._sample_size = 0

    def snapshot(self) -> Mapping[str, object]:
        return {
            "method": self._method,
            "sample_size": self._sample_size,
            "looks": self._looks,
            "window_size": len(self._window),
            "alarm": False,
        }


class HarmfulShiftFloatAdapter:
    def __init__(self, *, alpha: float, harm_margin: float) -> None:
        self._test = HarmfulShiftTest(alpha=alpha, harm_margin=harm_margin)

    def update(self, observation: float) -> Evidence:
        if isinstance(observation, bool) or not isinstance(observation, (int, float)):
            raise StatisticsError("paired difference must be a finite float in [-1, 1]")
        value = float(observation)
        if not math.isfinite(value) or not -1.0 <= value <= 1.0:
            raise StatisticsError("paired difference must be a finite float in [-1, 1]")
        paired = PairedSuccess(
            candidate=max(value, 0.0),
            reference=max(-value, 0.0),
        )
        return self._test.update(paired)

    def reset(self) -> None:
        self._test.reset()

    def snapshot(self) -> Mapping[str, object]:
        return self._test.snapshot()


def build_hourly_window_detector(
    *,
    kind: Literal["ks", "chi_square"],
    correction: Literal["none", "bonferroni"],
    alpha: float,
    window_episodes: int,
    reference_sample: Sequence[float],
    baseline: float,
    signal: str | None = None,
) -> Detector:
    if kind not in WINDOW_KINDS:
        raise DetectorConstructionError("kind must be ks or chi_square")
    if correction not in CORRECTIONS:
        raise DetectorConstructionError("correction must be none or bonferroni")
    if signal is not None and signal not in MONITOR_SIGNALS:
        raise DetectorConstructionError(
            f"unsupported signal/rule pair: {signal!r} with {kind!r}"
        )
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise DetectorConstructionError("alpha must be a finite float in (0, 1)")
    alpha_value = float(alpha)
    if not math.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
        raise DetectorConstructionError("alpha must be a finite float in (0, 1)")
    if isinstance(window_episodes, bool) or not isinstance(window_episodes, int):
        raise DetectorConstructionError("window_episodes must be an integer >= 1")
    if window_episodes < 1:
        raise DetectorConstructionError("window_episodes must be an integer >= 1")
    if isinstance(baseline, bool) or not isinstance(baseline, (int, float)):
        raise DetectorConstructionError("baseline must be a finite float")
    baseline_value = float(baseline)
    if not math.isfinite(baseline_value):
        raise DetectorConstructionError("baseline must be a finite float")
    if not isinstance(reference_sample, Sequence) or isinstance(
        reference_sample, (str, bytes)
    ):
        raise DetectorConstructionError("reference_sample must be a sequence of floats")
    sample: list[float] = []
    for item in reference_sample:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise DetectorConstructionError(
                "reference_sample must be a sequence of floats"
            )
        number = float(item)
        if not math.isfinite(number):
            raise DetectorConstructionError(
                "reference_sample must be a sequence of floats"
            )
        sample.append(number)
    method = "ks_hourly" if kind == "ks" else "chi_square_hourly"
    return HourlyWindowDetector(
        method=method,
        alpha=alpha_value,
        window_episodes=window_episodes,
        reference_sample=tuple(sample),
        baseline=baseline_value,
        correction=correction,
        kind=kind,
    )


class DistributionalWindowDetector:
    """Windowed chi-square drift test over a named-category distribution.

    Feeds ``tool_selection`` (the agent's own tool/API choices) or
    ``task_mix`` (the incoming task composition), never a scalar. Each
    ``update`` merges one observation's counts into the open window; once
    the window reaches ``window_episodes`` observations, it is compared
    against ``reference_counts`` with ``stats.chi_square.
    chi_square_homogeneity`` over the union of category names seen on
    either side (a name absent from one side counts as zero there, and a
    reference category otherwise expected to be empty is floored at one
    count so the test never divides by a zero expected total). The window
    then clears, matching ``HourlyWindowDetector``'s non-overlapping
    windows.
    """

    def __init__(
        self,
        *,
        method: str,
        reference_counts: Mapping[str, int],
        window_episodes: int,
        alpha: float,
        correction: str,
    ) -> None:
        self._method = method
        self._reference = dict(reference_counts)
        self._window_episodes = window_episodes
        self._alpha = alpha
        self._correction = correction
        self._window: dict[str, int] = {}
        self._window_size = 0
        self._looks = 0
        self._sample_size = 0

    def update(self, observation: Mapping[str, int]) -> Evidence:
        for name, count in observation.items():
            self._window[name] = self._window.get(name, 0) + int(count)
        self._window_size += 1
        self._sample_size += 1
        alarm = False
        p_value: float | None = None
        if self._window_size >= self._window_episodes:
            self._looks += 1
            categories = sorted(set(self._reference) | set(self._window))
            reference_vector = [
                max(1, self._reference.get(name, 0)) for name in categories
            ]
            window_vector = [self._window.get(name, 0) for name in categories]
            try:
                result = chi_square_homogeneity(window_vector, reference_vector)
                p_value = result.p_value
            except ChiSquareError:
                p_value = None
            if p_value is not None:
                if self._correction == "bonferroni":
                    adjusted = min(1.0, p_value * self._looks)
                    alarm = adjusted <= self._alpha
                else:
                    alarm = p_value <= self._alpha
            self._window = {}
            self._window_size = 0
        return Evidence(
            method=self._method,
            estimate=0.0 if p_value is None else p_value,
            sample_size=self._sample_size,
            alarm=alarm,
            boundary=self._alpha,
            p_value=p_value,
            details=(),
        )

    def reset(self) -> None:
        self._window = {}
        self._window_size = 0
        self._looks = 0
        self._sample_size = 0

    def snapshot(self) -> Mapping[str, object]:
        return {
            "method": self._method,
            "sample_size": self._sample_size,
            "looks": self._looks,
            "window_size": self._window_size,
            "alarm": False,
        }


def build_distributional_detector(
    *,
    signal: str,
    reference_counts: Mapping[str, int],
    window_episodes: int,
    alpha: float,
    correction: str,
) -> DistributionalWindowDetector:
    if signal not in DISTRIBUTIONAL_SIGNALS:
        raise DetectorConstructionError("signal must be tool_selection or task_mix")
    if correction not in DISTRIBUTIONAL_CORRECTIONS:
        raise DetectorConstructionError("correction must be none or bonferroni")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise DetectorConstructionError("alpha must be a finite float in (0, 1)")
    alpha_value = float(alpha)
    if not math.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
        raise DetectorConstructionError("alpha must be a finite float in (0, 1)")
    if isinstance(window_episodes, bool) or not isinstance(window_episodes, int):
        raise DetectorConstructionError("window_episodes must be an integer >= 1")
    if window_episodes < 1:
        raise DetectorConstructionError("window_episodes must be an integer >= 1")
    if not isinstance(reference_counts, Mapping) or not reference_counts:
        raise DetectorConstructionError(
            "reference_counts must be a non-empty mapping of name to count"
        )
    counts: dict[str, int] = {}
    for name, count in reference_counts.items():
        if not isinstance(name, str) or name == "":
            raise DetectorConstructionError("reference_counts names must be strings")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise DetectorConstructionError(
                "reference_counts counts must be non-negative integers"
            )
        counts[name] = count
    if sum(counts.values()) <= 0:
        raise DetectorConstructionError("reference_counts must have a positive total")
    method = (
        "chi_square_hourly" if correction == "none" else "chi_square_hourly_bonferroni"
    )
    return DistributionalWindowDetector(
        method=method,
        reference_counts=counts,
        window_episodes=window_episodes,
        alpha=alpha_value,
        correction=correction,
    )


def build_harmful_shift_adapter(
    *,
    alpha: float,
    harm_margin: float,
) -> Detector:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise DetectorConstructionError("alpha must be a finite float in (0, 1)")
    alpha_value = float(alpha)
    if not math.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
        raise DetectorConstructionError("alpha must be a finite float in (0, 1)")
    if isinstance(harm_margin, bool) or not isinstance(harm_margin, (int, float)):
        raise DetectorConstructionError("harm_margin must be a finite float >= 0")
    margin = float(harm_margin)
    if not math.isfinite(margin) or margin < 0.0:
        raise DetectorConstructionError("harm_margin must be a finite float >= 0")
    return HarmfulShiftFloatAdapter(alpha=alpha_value, harm_margin=margin)
