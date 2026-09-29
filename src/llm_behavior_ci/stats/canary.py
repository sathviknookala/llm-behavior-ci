from __future__ import annotations

import math

from llm_behavior_ci.stats.confidence_sequence import PairedDifferenceCS
from llm_behavior_ci.stats.evidence import Evidence, PairedSuccess, StatisticsError
from llm_behavior_ci.stats.harmful_shift import HarmfulShiftTest


class SequentialCanaryTest:
    """Horizon-bounded harmful-shift test for a paired canary.

    Null hypothesis: the candidate is not worse than the reference by harm_margin.
    Assumptions: the harmful-shift e-process, looked at once per paired episode, up to the horizon.
    Direction of harm: candidate success below reference success by more than harm_margin.
    Boundary: 1/alpha, and the horizon is a stopping time that does not raise an alarm by itself.
    Reset: clears wealth and the count.
    Evidence: alarm is the e-process crossing; stopped_at_horizon is 1 only when the horizon is reached without an alarm. No rollback, promotion, or alert method.
    """

    def __init__(
        self,
        *,
        alpha: float,
        harm_margin: float,
        horizon_episodes: int,
    ) -> None:
        if (
            not isinstance(horizon_episodes, int)
            or isinstance(horizon_episodes, bool)
            or horizon_episodes < 1
        ):
            raise StatisticsError("horizon_episodes must be an integer of at least 1")
        self._horizon = horizon_episodes
        self._test = HarmfulShiftTest(alpha=alpha, harm_margin=harm_margin)

    def update(self, observation: PairedSuccess) -> Evidence:
        evidence = self._test.update(observation)
        stopped = (not evidence.alarm) and evidence.sample_size >= self._horizon
        return Evidence(
            method="sequential_canary",
            estimate=evidence.estimate,
            sample_size=evidence.sample_size,
            alarm=evidence.alarm,
            boundary=evidence.boundary,
            p_value=evidence.p_value,
            details=evidence.details
            + (("stopped_at_horizon", 1.0 if stopped else 0.0),),
            direction=evidence.direction,
        )

    def reset(self) -> None:
        self._test.reset()

    def snapshot(self) -> dict[str, object]:
        snapshot = dict(self._test.snapshot())
        snapshot["method"] = "sequential_canary"
        snapshot["horizon_episodes"] = self._horizon
        return snapshot


class PairedDifferenceCanaryTest:
    """Horizon-bounded harm/benefit read of a paired-difference confidence sequence.

    Null hypothesis: none. This wraps ``PairedDifferenceCS`` pinned at
    ``null_mean=0.0`` — the only ``null_mean`` its own null check
    validates — and reads the resulting anytime-valid interval for
    ``mean(candidate - reference)`` against two fixed points on every
    look: zero, and ``-harm_margin``. A confidence sequence stays valid
    under repeated looks and under reading it against more than one fixed
    point, so this adds no further look-based error budget beyond the
    sequence's own ``alpha``.
    Assumptions: paired differences in [-1, 1], iid.
    Direction of harm: the whole interval lies below ``-harm_margin``
    (``alarm`` is only ever set in this direction). Direction of benefit:
    the whole interval lies above zero. Anything else, including an
    interval that excludes zero but not ``-harm_margin`` (not harmful
    enough to alarm, not yet confidently better), is insufficient
    evidence.
    Boundary: the confidence sequence's stitched radius, unchanged by
    which fixed points are read.
    Reset: clears the inner confidence sequence.
    Evidence: alarm is true only for the harmful direction, so it is safe
    for an automatic harmful-regression rollback to gate on ``alarm``
    alone; ``stopped_at_horizon`` is 1 only once the horizon is reached
    without a harmful alarm, mirroring ``sequential_canary``. No
    rollback, promotion, or alert method.
    """

    def __init__(
        self,
        *,
        alpha: float,
        harm_margin: float,
        horizon_episodes: int,
    ) -> None:
        if (
            not isinstance(horizon_episodes, int)
            or isinstance(horizon_episodes, bool)
            or horizon_episodes < 1
        ):
            raise StatisticsError("horizon_episodes must be an integer of at least 1")
        if isinstance(harm_margin, bool) or not isinstance(harm_margin, (int, float)):
            raise StatisticsError("harm_margin must be a finite float of at least zero")
        margin = float(harm_margin)
        if not math.isfinite(margin) or margin < 0.0:
            raise StatisticsError("harm_margin must be a finite float of at least zero")
        self._margin = margin
        self._horizon = horizon_episodes
        self._cs = PairedDifferenceCS(alpha=alpha, lower=-1.0, upper=1.0, null_mean=0.0)

    def update(self, observation: PairedSuccess) -> Evidence:
        evidence = self._cs.update(observation)
        details = dict(evidence.details)
        lower_end = details["lower_end"]
        upper_end = details["upper_end"]
        harmful = upper_end < -self._margin
        beneficial = lower_end > 0.0
        if harmful:
            direction = "harmful"
        elif beneficial:
            direction = "beneficial"
        else:
            direction = "insufficient"
        stopped = (not harmful) and evidence.sample_size >= self._horizon
        return Evidence(
            method="paired_difference_cs",
            estimate=evidence.estimate,
            sample_size=evidence.sample_size,
            alarm=harmful,
            boundary=evidence.boundary,
            p_value=evidence.p_value,
            details=evidence.details
            + (("stopped_at_horizon", 1.0 if stopped else 0.0),),
            direction=direction,
        )

    def reset(self) -> None:
        self._cs.reset()

    def snapshot(self) -> dict[str, object]:
        snapshot = dict(self._cs.snapshot())
        snapshot["method"] = "paired_difference_cs"
        snapshot["horizon_episodes"] = self._horizon
        return snapshot
