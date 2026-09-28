from __future__ import annotations

import math
from statistics import fmean

from llm_behavior_ci.stats.evidence import Evidence, StatisticsError


class ADWIN:
    """Exact-list ADWIN cut on a scalar window.

    Null hypothesis: one stationary Bernoulli or bounded mean.
    Assumptions: the ADWIN epsilon cut on the raw window, iid observations in [0, 1] are the intended use; the code accepts any finite float.
    Direction of harm: a mean change in either direction.
    Boundary: epsilon_cut.
    Reset: clears the window.
    Evidence: alarm on an update that drops a prefix. Not a coverage claim.
    """

    def __init__(self, *, delta: float) -> None:
        if isinstance(delta, bool) or not isinstance(delta, (int, float)):
            raise StatisticsError("delta must be a finite float in (0, 1)")
        value = float(delta)
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise StatisticsError("delta must be a finite float in (0, 1)")
        self._delta = value
        self._window: list[float] = []
        self._alarm = False

    def update(self, observation: float) -> Evidence:
        if isinstance(observation, bool) or not isinstance(observation, (int, float)):
            raise StatisticsError("observation must be a finite float")
        value = float(observation)
        if not math.isfinite(value):
            raise StatisticsError("observation must be a finite float")
        self._window.append(value)
        alarm = False
        dropped = 0
        boundary: float | None = None
        while len(self._window) >= 2:
            length = len(self._window)
            cut_index: int | None = None
            cut_epsilon = 0.0
            for cut in range(1, length):
                n0 = cut
                n1 = length - cut
                harmonic = 1.0 / (1.0 / n0 + 1.0 / n1)
                epsilon = math.sqrt(
                    (1.0 / (2.0 * harmonic)) * math.log(4.0 * length / self._delta)
                )
                gap = abs(fmean(self._window[:cut]) - fmean(self._window[cut:]))
                if gap >= epsilon:
                    cut_index = cut
                    cut_epsilon = epsilon
                    break
            if cut_index is None:
                break
            del self._window[:cut_index]
            alarm = True
            dropped += cut_index
            boundary = cut_epsilon
        self._alarm = alarm
        estimate = fmean(self._window) if self._window else 0.0
        return Evidence(
            method="adwin",
            estimate=estimate,
            sample_size=len(self._window),
            alarm=alarm,
            boundary=boundary,
            p_value=None,
            details=(("dropped", float(dropped)), ("delta", self._delta)),
        )

    def reset(self) -> None:
        self._window = []
        self._alarm = False

    def snapshot(self) -> dict[str, object]:
        if not self._window:
            estimate = 0.0
            alarm = False
        else:
            estimate = fmean(self._window)
            alarm = self._alarm
        return {
            "method": "adwin",
            "estimate": estimate,
            "sample_size": len(self._window),
            "alarm": alarm,
            "delta": self._delta,
        }
