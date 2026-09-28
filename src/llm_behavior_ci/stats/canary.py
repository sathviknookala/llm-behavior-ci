from __future__ import annotations

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
        )

    def reset(self) -> None:
        self._test.reset()

    def snapshot(self) -> dict[str, object]:
        snapshot = dict(self._test.snapshot())
        snapshot["method"] = "sequential_canary"
        snapshot["horizon_episodes"] = self._horizon
        return snapshot
