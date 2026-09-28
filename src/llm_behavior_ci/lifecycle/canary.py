"""Paired canary controller: gate check, sequential stopping, rollback or promote."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

from llm_behavior_ci.config import CanarySettings, RunConfiguration, run_configuration_hash
from llm_behavior_ci.records import LifecycleDecision, PairedResult, StatisticalEvidence
from llm_behavior_ci.runtime.episode import pair_execution
from llm_behavior_ci.stats.canary import SequentialCanaryTest
from llm_behavior_ci.stats.confidence_sequence import PairedDifferenceCS
from llm_behavior_ci.stats.evidence import Evidence, PairedSuccess


class CanaryRejected(ValueError):
    pass


@dataclass(frozen=True)
class DeploymentSnapshot:
    state: str
    serving_configuration_hash: str
    previous_production_configuration_hash: str
    candidate_configuration_hash: str
    candidate_episodes_started: int
    candidate_episodes_served: int
    candidate_episodes_failed: int
    outstanding: int
    in_flight_at_rollback: int | None
    served_before_rollback: int | None
    rollback_at: datetime | None
    promoted_at: datetime | None


@dataclass(frozen=True)
class CanaryDecision:
    action: str
    state: str
    evidence: Evidence | None
    public_decision: LifecycleDecision | None
    snapshot: DeploymentSnapshot


@dataclass(frozen=True)
class _ControllerState:
    phase: str
    candidate_episodes_started: int
    candidate_episodes_served: int
    candidate_episodes_failed: int
    outstanding: int
    in_flight_at_rollback: int | None
    served_before_rollback: int | None
    rollback_at: datetime | None
    promoted_at: datetime | None
    seen_pair_ids: frozenset[str]
    held_pairs: tuple[PairedResult, ...]


class _FixedWindowCanary:
    """Horizon mean of paired success differences."""

    def __init__(self, *, harm_margin: float, horizon_episodes: int) -> None:
        self._harm_margin = harm_margin
        self._horizon = horizon_episodes
        self._total = 0.0
        self._count = 0

    def update(self, observation: PairedSuccess) -> Evidence:
        difference = float(observation.candidate) - float(observation.reference)
        self._total += difference
        self._count += 1
        estimate = self._total / self._count
        at_horizon = self._count >= self._horizon
        alarm = at_horizon and estimate < -self._harm_margin
        return Evidence(
            method="fixed_window",
            estimate=estimate,
            sample_size=self._count,
            alarm=alarm,
            boundary=-self._harm_margin,
            p_value=None,
            details=(("at_horizon", 1.0 if at_horizon else 0.0),),
        )


def _empty_state() -> _ControllerState:
    return _ControllerState(
        phase="CREATED",
        candidate_episodes_started=0,
        candidate_episodes_served=0,
        candidate_episodes_failed=0,
        outstanding=0,
        in_flight_at_rollback=None,
        served_before_rollback=None,
        rollback_at=None,
        promoted_at=None,
        seen_pair_ids=frozenset(),
        held_pairs=(),
    )


def _gate_attr(gate: object, name: str) -> object:
    if not hasattr(gate, name):
        raise CanaryRejected(f"gate is missing {name}")
    return getattr(gate, name)


class CanaryController:
    """Deploy a candidate after a PASS gate and stop on the configured rule.

    Detectors return Evidence. This controller owns continue, rollback, and
    promote. Thresholds and horizons come only from CanarySettings.
    """

    def __init__(
        self,
        reference: RunConfiguration,
        candidate: RunConfiguration,
        *,
        settings: CanarySettings,
        clock: Callable[[], datetime],
    ) -> None:
        if not isinstance(reference, RunConfiguration):
            raise CanaryRejected("reference must be a RunConfiguration")
        if not isinstance(candidate, RunConfiguration):
            raise CanaryRejected("candidate must be a RunConfiguration")
        if not isinstance(settings, CanarySettings):
            raise CanaryRejected("settings must be CanarySettings")
        if not callable(clock):
            raise CanaryRejected("clock must be callable")
        rule = settings.stopping_rule
        if rule.name == "sequential_canary":
            detector: object = SequentialCanaryTest(
                alpha=rule.alpha,
                harm_margin=settings.harm_margin,
                horizon_episodes=rule.horizon_episodes,
            )
        elif rule.name == "paired_difference_cs":
            detector = PairedDifferenceCS(
                alpha=rule.alpha,
                lower=-1.0,
                upper=1.0,
                null_mean=-settings.harm_margin,
            )
        elif rule.name == "fixed_window":
            detector = _FixedWindowCanary(
                harm_margin=settings.harm_margin,
                horizon_episodes=rule.horizon_episodes,
            )
        else:
            raise CanaryRejected(f"unsupported stopping rule: {rule.name}")
        self._reference = reference
        self._candidate = candidate
        self._settings = settings
        self._clock = clock
        self._reference_hash = run_configuration_hash(reference)
        self._candidate_hash = run_configuration_hash(candidate)
        self._detector = detector
        self._detector_updates = 0
        self._state = _empty_state()

    def start(self, gate: object) -> None:
        if self._state.phase != "CREATED":
            raise CanaryRejected("start requires CREATED")
        outcome = _gate_attr(gate, "outcome")
        reason_codes = _gate_attr(gate, "reason_codes")
        reference_configuration_hash = _gate_attr(
            gate, "reference_configuration_hash"
        )
        candidate_configuration_hash = _gate_attr(
            gate, "candidate_configuration_hash"
        )
        task_set_hash = _gate_attr(gate, "task_set_hash")
        reference_protocol_hash = _gate_attr(gate, "reference_protocol_hash")
        candidate_protocol_hash = _gate_attr(gate, "candidate_protocol_hash")
        if outcome != "PASS":
            raise CanaryRejected("gate outcome must be PASS")
        if reason_codes is None or len(reason_codes) != 0:
            raise CanaryRejected("gate reason_codes must be empty")
        if reference_configuration_hash != self._reference_hash:
            raise CanaryRejected("gate reference configuration hash mismatch")
        if candidate_configuration_hash != self._candidate_hash:
            raise CanaryRejected("gate candidate configuration hash mismatch")
        if task_set_hash != self._reference.task.task_set_hash:
            raise CanaryRejected("gate task_set_hash mismatch")
        if task_set_hash != self._candidate.task.task_set_hash:
            raise CanaryRejected("gate task_set_hash mismatch")
        if reference_protocol_hash != self._reference.protocol_hash:
            raise CanaryRejected("gate reference protocol hash mismatch")
        if candidate_protocol_hash != self._candidate.protocol_hash:
            raise CanaryRejected("gate candidate protocol hash mismatch")
        self._state = _ControllerState(
            phase="GATE_PASSED",
            candidate_episodes_started=0,
            candidate_episodes_served=0,
            candidate_episodes_failed=0,
            outstanding=0,
            in_flight_at_rollback=None,
            served_before_rollback=None,
            rollback_at=None,
            promoted_at=None,
            seen_pair_ids=frozenset(),
            held_pairs=(),
        )

    def begin_candidate_episode(self) -> None:
        state = self._state
        if state.phase == "GATE_PASSED":
            self._state = _ControllerState(
                phase="CANARY_ACTIVE",
                candidate_episodes_started=state.candidate_episodes_started + 1,
                candidate_episodes_served=state.candidate_episodes_served,
                candidate_episodes_failed=state.candidate_episodes_failed,
                outstanding=state.outstanding + 1,
                in_flight_at_rollback=state.in_flight_at_rollback,
                served_before_rollback=state.served_before_rollback,
                rollback_at=state.rollback_at,
                promoted_at=state.promoted_at,
                seen_pair_ids=state.seen_pair_ids,
                held_pairs=state.held_pairs,
            )
            return
        if state.phase == "CANARY_ACTIVE":
            self._state = _ControllerState(
                phase="CANARY_ACTIVE",
                candidate_episodes_started=state.candidate_episodes_started + 1,
                candidate_episodes_served=state.candidate_episodes_served,
                candidate_episodes_failed=state.candidate_episodes_failed,
                outstanding=state.outstanding + 1,
                in_flight_at_rollback=state.in_flight_at_rollback,
                served_before_rollback=state.served_before_rollback,
                rollback_at=state.rollback_at,
                promoted_at=state.promoted_at,
                seen_pair_ids=state.seen_pair_ids,
                held_pairs=state.held_pairs,
            )
            return
        raise CanaryRejected("begin_candidate_episode requires GATE_PASSED or CANARY_ACTIVE")

    def observe(self, pair: PairedResult) -> CanaryDecision:
        state = self._state
        if state.phase != "CANARY_ACTIVE":
            raise CanaryRejected("observe requires CANARY_ACTIVE")
        if state.outstanding < 1:
            raise CanaryRejected("observe requires outstanding work")
        try:
            pair_execution(pair)
        except Exception as exc:
            raise CanaryRejected("pair execution was not recorded") from exc
        if pair.reference.mode != "execute" or pair.candidate.mode != "execute":
            raise CanaryRejected("canary pairs must use execute mode")
        if pair.reference.run.configuration_hash != self._reference_hash:
            raise CanaryRejected("reference configuration hash mismatch")
        if pair.candidate.run.configuration_hash != self._candidate_hash:
            raise CanaryRejected("candidate configuration hash mismatch")
        pair_id = pair.reference.episode.pair_id
        if pair_id is None or pair_id in state.seen_pair_ids:
            raise CanaryRejected("duplicate or missing pair_id")

        failed = pair.candidate.status == "failed"
        next_state = _ControllerState(
            phase="CANARY_ACTIVE",
            candidate_episodes_started=state.candidate_episodes_started,
            candidate_episodes_served=state.candidate_episodes_served + 1,
            candidate_episodes_failed=state.candidate_episodes_failed
            + (1 if failed else 0),
            outstanding=state.outstanding - 1,
            in_flight_at_rollback=state.in_flight_at_rollback,
            served_before_rollback=state.served_before_rollback,
            rollback_at=state.rollback_at,
            promoted_at=state.promoted_at,
            seen_pair_ids=state.seen_pair_ids | {pair_id},
            held_pairs=state.held_pairs,
        )

        now = self._clock()
        delay = self._settings.outcome_delay_seconds
        ready: list[PairedResult] = []
        still_held: list[PairedResult] = []
        for held in next_state.held_pairs:
            if self._delay_elapsed(held, now, delay):
                ready.append(held)
            else:
                still_held.append(held)
        if self._has_both_outcomes(pair) and not self._delay_elapsed(pair, now, delay):
            still_held.append(pair)
        elif self._has_both_outcomes(pair):
            ready.append(pair)

        next_state = _ControllerState(
            phase=next_state.phase,
            candidate_episodes_started=next_state.candidate_episodes_started,
            candidate_episodes_served=next_state.candidate_episodes_served,
            candidate_episodes_failed=next_state.candidate_episodes_failed,
            outstanding=next_state.outstanding,
            in_flight_at_rollback=next_state.in_flight_at_rollback,
            served_before_rollback=next_state.served_before_rollback,
            rollback_at=next_state.rollback_at,
            promoted_at=next_state.promoted_at,
            seen_pair_ids=next_state.seen_pair_ids,
            held_pairs=tuple(still_held),
        )
        self._state = next_state

        last_evidence: Evidence | None = None
        last_action = "continue"
        for item in ready:
            evidence = self._apply_detector(item)
            if evidence is None:
                continue
            last_evidence = evidence
            action = self._action_from_evidence(evidence)
            last_action = action
            if action in ("rollback", "promote"):
                decided_at = self._clock()
                self._state = self._terminal_state(
                    self._state,
                    action=action,
                    decided_at=decided_at,
                )
                public = self._public_decision(action, evidence, decided_at)
                return CanaryDecision(
                    action=action,
                    state=self._state.phase,
                    evidence=evidence,
                    public_decision=public,
                    snapshot=self.snapshot(),
                )

        if last_evidence is None:
            return CanaryDecision(
                action="continue",
                state=self._state.phase,
                evidence=None,
                public_decision=None,
                snapshot=self.snapshot(),
            )
        decided_at = self._clock()
        public = self._public_decision("continue", last_evidence, decided_at)
        return CanaryDecision(
            action="continue",
            state=self._state.phase,
            evidence=last_evidence,
            public_decision=public,
            snapshot=self.snapshot(),
        )

    def snapshot(self) -> DeploymentSnapshot:
        state = self._state
        serving = self._reference_hash
        if state.phase == "PROMOTED":
            serving = self._candidate_hash
        return DeploymentSnapshot(
            state=state.phase,
            serving_configuration_hash=serving,
            previous_production_configuration_hash=self._reference_hash,
            candidate_configuration_hash=self._candidate_hash,
            candidate_episodes_started=state.candidate_episodes_started,
            candidate_episodes_served=state.candidate_episodes_served,
            candidate_episodes_failed=state.candidate_episodes_failed,
            outstanding=state.outstanding,
            in_flight_at_rollback=state.in_flight_at_rollback,
            served_before_rollback=state.served_before_rollback,
            rollback_at=state.rollback_at,
            promoted_at=state.promoted_at,
        )

    def _has_both_outcomes(self, pair: PairedResult) -> bool:
        return (
            pair.reference.evaluator_outcome is not None
            and pair.candidate.evaluator_outcome is not None
        )

    def _delay_elapsed(
        self,
        pair: PairedResult,
        now: datetime,
        delay: float,
    ) -> bool:
        if delay <= 0.0:
            return True
        return now >= pair.candidate.ended_at + timedelta(seconds=delay)

    def _apply_detector(self, pair: PairedResult) -> Evidence | None:
        reference_outcome = pair.reference.evaluator_outcome
        candidate_outcome = pair.candidate.evaluator_outcome
        if reference_outcome is None or candidate_outcome is None:
            return None
        observation = PairedSuccess(
            candidate=1.0 if candidate_outcome.success else 0.0,
            reference=1.0 if reference_outcome.success else 0.0,
        )
        evidence = self._detector.update(observation)
        self._detector_updates += 1
        return evidence

    def _action_from_evidence(self, evidence: Evidence) -> str:
        rule = self._settings.stopping_rule
        if evidence.alarm:
            return "rollback"
        if rule.name == "sequential_canary":
            details = dict(evidence.details)
            if details.get("stopped_at_horizon", 0.0) == 1.0:
                return "promote"
            return "continue"
        if rule.name == "paired_difference_cs":
            if self._detector_updates >= rule.horizon_episodes:
                return "promote"
            return "continue"
        if rule.name == "fixed_window":
            details = dict(evidence.details)
            if details.get("at_horizon", 0.0) == 1.0:
                return "promote"
            return "continue"
        raise CanaryRejected(f"unsupported stopping rule: {rule.name}")

    def _terminal_state(
        self,
        state: _ControllerState,
        *,
        action: str,
        decided_at: datetime,
    ) -> _ControllerState:
        if action == "rollback":
            in_flight = state.outstanding
            return _ControllerState(
                phase="ROLLED_BACK",
                candidate_episodes_started=state.candidate_episodes_started,
                candidate_episodes_served=state.candidate_episodes_served,
                candidate_episodes_failed=state.candidate_episodes_failed,
                outstanding=state.outstanding,
                in_flight_at_rollback=in_flight,
                served_before_rollback=state.candidate_episodes_served + in_flight,
                rollback_at=decided_at,
                promoted_at=None,
                seen_pair_ids=state.seen_pair_ids,
                held_pairs=state.held_pairs,
            )
        return _ControllerState(
            phase="PROMOTED",
            candidate_episodes_started=state.candidate_episodes_started,
            candidate_episodes_served=state.candidate_episodes_served,
            candidate_episodes_failed=state.candidate_episodes_failed,
            outstanding=state.outstanding,
            in_flight_at_rollback=state.in_flight_at_rollback,
            served_before_rollback=state.served_before_rollback,
            rollback_at=state.rollback_at,
            promoted_at=decided_at,
            seen_pair_ids=state.seen_pair_ids,
            held_pairs=state.held_pairs,
        )

    def _public_decision(
        self,
        action: str,
        evidence: Evidence,
        decided_at: datetime,
    ) -> LifecycleDecision:
        statistical = StatisticalEvidence(
            method=evidence.method,
            split=self._candidate.task.split,
            configuration_hash=self._candidate_hash,
            estimate=evidence.estimate,
            sample_size=evidence.sample_size,
            unit="success_delta",
            reference_configuration_hash=self._reference_hash,
            p_value=evidence.p_value,
            threshold=-self._settings.harm_margin,
        )
        return LifecycleDecision(
            tier="canary",
            decision=action,
            split=self._candidate.task.split,
            candidate_configuration_hash=self._candidate_hash,
            reference_configuration_hash=self._reference_hash,
            evidence=(statistical,),
            decided_at=decided_at,
        )
