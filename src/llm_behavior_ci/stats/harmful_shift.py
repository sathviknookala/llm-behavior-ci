from __future__ import annotations

import math

from llm_behavior_ci.stats.e_detector import BettingEDetector
from llm_behavior_ci.stats.evidence import Evidence, PairedSuccess, StatisticsError


class HarmfulShiftTest:
    """Betting test that the candidate falls below the reference by more than a margin.

    Null hypothesis: mean(candidate - reference) is at least -harm_margin.
    Assumptions: paired bounded outcomes and the betting e-process above.
    Direction of harm: the candidate falls below the reference by more than harm_margin.
    Boundary: wealth >= 1/alpha.
    Reset: clears the inner detector.
    Evidence: alarm means the e-process crossed 1/alpha. This is the project's harmful-shift test in the Podkopaev–Ramdas role. It is this betting procedure, not a line-by-line port. No rollback method.
    """

    def __init__(self, *, alpha: float, harm_margin: float) -> None:
        if isinstance(harm_margin, bool) or not isinstance(harm_margin, (int, float)):
            raise StatisticsError("harm_margin must be a finite float of at least zero")
        margin = float(harm_margin)
        if not math.isfinite(margin) or margin < 0.0:
            raise StatisticsError("harm_margin must be a finite float of at least zero")
        self._margin = margin
        self._detector = BettingEDetector(
            null_mean=-margin,
            alpha=alpha,
            lower=-1.0,
            upper=1.0,
            direction="below",
        )

    def update(self, observation: PairedSuccess) -> Evidence:
        candidate = observation.candidate
        reference = observation.reference
        if isinstance(candidate, bool) or isinstance(reference, bool):
            raise StatisticsError("candidate and reference must lie in [0, 1]")
        if not isinstance(candidate, (int, float)) or not isinstance(
            reference, (int, float)
        ):
            raise StatisticsError("candidate and reference must lie in [0, 1]")
        candidate_value = float(candidate)
        reference_value = float(reference)
        if not math.isfinite(candidate_value) or not math.isfinite(reference_value):
            raise StatisticsError("candidate and reference must lie in [0, 1]")
        if not 0.0 <= candidate_value <= 1.0 or not 0.0 <= reference_value <= 1.0:
            raise StatisticsError("candidate and reference must lie in [0, 1]")
        evidence = self._detector.update(candidate_value - reference_value)
        return Evidence(
            method="harmful_shift",
            estimate=evidence.estimate,
            sample_size=evidence.sample_size,
            alarm=evidence.alarm,
            boundary=evidence.boundary,
            p_value=evidence.p_value,
            details=evidence.details,
        )

    def reset(self) -> None:
        self._detector.reset()

    def snapshot(self) -> dict[str, object]:
        snapshot = dict(self._detector.snapshot())
        snapshot["method"] = "harmful_shift"
        return snapshot
