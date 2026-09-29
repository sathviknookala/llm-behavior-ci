"""Paired canary controller: gate check, sequential stopping, rollback or promote."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Callable

from llm_behavior_ci.config import CanarySettings, RunConfiguration, run_configuration_hash
from llm_behavior_ci.records import LifecycleDecision, PairedResult, StatisticalEvidence
from llm_behavior_ci.runtime.episode import pair_execution
from llm_behavior_ci.stats.canary import SequentialCanaryTest
from llm_behavior_ci.stats.confidence_sequence import PairedDifferenceCS
from llm_behavior_ci.stats.evidence import Evidence, PairedSuccess

_SUPPORTED_STOPPING_RULES = frozenset(
    {"sequential_canary", "paired_difference_cs", "fixed_window"}
)


class CanaryRejected(ValueError):
    pass


def assign_canary(assignment_key: str, *, fraction: float, seed: int) -> bool:
    """Return whether ``assignment_key`` is assigned to the canary cohort.

    Membership is ``SHA-256(f"{seed}:{assignment_key}")`` interpreted as a
    big-endian integer from the first 8 digest bytes, scaled into ``[0, 1)``,
    then compared with ``fraction``. The same key, fraction, and seed always
    yield the same boolean; over many keys the rate tracks ``fraction``.
    """

    if not isinstance(assignment_key, str) or assignment_key == "":
        raise CanaryRejected("assignment_key must be a non-empty string")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise CanaryRejected("seed must be an integer")
    if isinstance(fraction, bool) or not isinstance(fraction, float):
        raise CanaryRejected("fraction must be a float")
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise CanaryRejected(
            "fraction must be greater than zero and at most one"
        )
    material = f"{seed}:{assignment_key}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    unit = int.from_bytes(digest[:8], "big") / 2**64
    return unit < fraction


@dataclass(frozen=True)
class DeploymentSnapshot:
    state: str
    serving_configuration_hash: str
    previous_production_configuration_hash: str
    candidate_configuration_hash: str
    candidate_episodes_started: int
    candidate_episodes_served: int
    candidate_episodes_failed: int
    candidate_episodes_evaluator_unsuccessful: int
    outstanding: int
    in_flight_at_rollback: int | None
    served_before_rollback: int | None
    rollback_at: datetime | None
    rollback_reason: str | None
    rollback_manual: bool | None
    promoted_at: datetime | None
    promoted_configuration_hash: str | None
    monitoring_reset_required: bool


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
    candidate_episodes_evaluator_unsuccessful: int
    outstanding: int
    in_flight_at_rollback: int | None
    served_before_rollback: int | None
    rollback_at: datetime | None
    rollback_reason: str | None
    rollback_manual: bool | None
    promoted_at: datetime | None
    promoted_configuration_hash: str | None
    monitoring_reset_required: bool
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
        candidate_episodes_evaluator_unsuccessful=0,
        outstanding=0,
        in_flight_at_rollback=None,
        served_before_rollback=None,
        rollback_at=None,
        rollback_reason=None,
        rollback_manual=None,
        promoted_at=None,
        promoted_configuration_hash=None,
        monitoring_reset_required=False,
        seen_pair_ids=frozenset(),
        held_pairs=(),
    )


def _gate_attr(gate: object, name: str) -> object:
    if not hasattr(gate, name):
        raise CanaryRejected(f"gate is missing {name}")
    return getattr(gate, name)


def _runtime_failed(candidate: object) -> bool:
    status = getattr(candidate, "status", None)
    termination_reason = getattr(candidate, "termination_reason", None)
    return status == "failed" or termination_reason == "runtime_error"


def _evaluator_unsuccessful(candidate: object) -> bool:
    status = getattr(candidate, "status", None)
    outcome = getattr(candidate, "evaluator_outcome", None)
    if status != "completed" or outcome is None:
        return False
    return not bool(outcome.success)


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
        if rule.name not in _SUPPORTED_STOPPING_RULES:
            raise CanaryRejected(f"unsupported stopping rule: {rule.name}")
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
        else:
            detector = _FixedWindowCanary(
                harm_margin=settings.harm_margin,
                horizon_episodes=rule.horizon_episodes,
            )
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
        self._state = replace(
            _empty_state(),
            phase="GATE_PASSED",
        )

    def start_from_admission(self, admission: object) -> None:
        if self._state.phase != "CREATED":
            raise CanaryRejected("start_from_admission requires CREATED")
        if getattr(type(admission), "__name__", None) != "GatedCandidateAdmission":
            raise CanaryRejected("admission must be GatedCandidateAdmission")
        outcome = _gate_attr(admission, "outcome")
        if outcome != "PASS":
            raise CanaryRejected("admission outcome must be PASS")
        served = _gate_attr(admission, "served_candidate_configuration_hash")
        if served != self._candidate_hash:
            raise CanaryRejected("admission served candidate hash mismatch")
        reference_protocol_hash = _gate_attr(admission, "reference_protocol_hash")
        candidate_protocol_hash = _gate_attr(admission, "candidate_protocol_hash")
        if reference_protocol_hash != self._reference.protocol_hash:
            raise CanaryRejected("admission reference protocol hash mismatch")
        if candidate_protocol_hash != self._candidate.protocol_hash:
            raise CanaryRejected("admission candidate protocol hash mismatch")
        allowed = _gate_attr(admission, "allowed_task_selection_leaves")
        try:
            leaves = frozenset(allowed)
        except TypeError as error:
            raise CanaryRejected(
                "admission allowed_task_selection_leaves invalid"
            ) from error
        gate_reference = _gate_attr(admission, "reference_configuration_hash")
        gate_candidate = _gate_attr(admission, "candidate_configuration_hash")
        train_task_set_hash = _gate_attr(admission, "train_task_set_hash")
        if not isinstance(train_task_set_hash, str):
            raise CanaryRejected("admission train_task_set_hash must be a string")
        if not leaves:
            if gate_reference != self._reference_hash:
                raise CanaryRejected(
                    "admission reference configuration hash mismatch"
                )
            if gate_candidate != self._candidate_hash:
                raise CanaryRejected(
                    "admission candidate configuration hash mismatch"
                )
            if train_task_set_hash != self._reference.task.task_set_hash:
                raise CanaryRejected("admission train_task_set_hash mismatch")
            if train_task_set_hash != self._candidate.task.task_set_hash:
                raise CanaryRejected("admission train_task_set_hash mismatch")
        self._state = replace(
            _empty_state(),
            phase="GATE_PASSED",
        )

    def begin_candidate_episode(self) -> None:
        state = self._state
        if state.phase == "GATE_PASSED":
            self._state = replace(
                state,
                phase="CANARY_ACTIVE",
                candidate_episodes_started=state.candidate_episodes_started + 1,
                outstanding=state.outstanding + 1,
            )
            return
        if state.phase == "CANARY_ACTIVE":
            self._state = replace(
                state,
                candidate_episodes_started=state.candidate_episodes_started + 1,
                outstanding=state.outstanding + 1,
            )
            return
        raise CanaryRejected(
            "begin_candidate_episode requires GATE_PASSED or CANARY_ACTIVE"
        )

    def observe(self, pair: PairedResult) -> CanaryDecision:
        state = self._state
        if state.phase != "CANARY_ACTIVE":
            raise CanaryRejected("observe requires CANARY_ACTIVE")
        if state.outstanding < 1:
            raise CanaryRejected("observe requires outstanding work")
        self._validate_pair(pair)
        pair_id = pair.reference.episode.pair_id
        if pair_id is None or pair_id in state.seen_pair_ids:
            raise CanaryRejected("duplicate or missing pair_id")

        failed = _runtime_failed(pair.candidate)
        evaluator_failed = _evaluator_unsuccessful(pair.candidate)
        next_state = replace(
            state,
            candidate_episodes_served=state.candidate_episodes_served + 1,
            candidate_episodes_failed=state.candidate_episodes_failed
            + (1 if failed else 0),
            candidate_episodes_evaluator_unsuccessful=(
                state.candidate_episodes_evaluator_unsuccessful
                + (1 if evaluator_failed else 0)
            ),
            outstanding=state.outstanding - 1,
            seen_pair_ids=state.seen_pair_ids | {pair_id},
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

        self._state = replace(next_state, held_pairs=tuple(still_held))

        last_evidence: Evidence | None = None
        for item in ready:
            evidence = self._apply_detector(item)
            if evidence is None:
                continue
            last_evidence = evidence
            action = self._action_from_evidence(evidence)
            if action == "rollback":
                decision = self.rollback("stopping_rule_alarm", manual=False)
                decided_at = decision.snapshot.rollback_at
                assert decided_at is not None
                public = self._public_decision("rollback", evidence, decided_at)
                return CanaryDecision(
                    action="rollback",
                    state=decision.state,
                    evidence=evidence,
                    public_decision=public,
                    snapshot=self.snapshot(),
                )
            if action == "promote":
                decision = self.promote()
                decided_at = decision.snapshot.promoted_at
                assert decided_at is not None
                public = self._public_decision("promote", evidence, decided_at)
                return CanaryDecision(
                    action="promote",
                    state=decision.state,
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

    def promote(self) -> CanaryDecision:
        state = self._state
        if state.phase != "CANARY_ACTIVE":
            raise CanaryRejected("promote requires CANARY_ACTIVE")
        decided_at = self._clock()
        self._state = replace(
            state,
            phase="PROMOTED",
            promoted_at=decided_at,
            promoted_configuration_hash=self._candidate_hash,
            monitoring_reset_required=True,
            held_pairs=(),
        )
        return CanaryDecision(
            action="promote",
            state=self._state.phase,
            evidence=None,
            public_decision=None,
            snapshot=self.snapshot(),
        )

    def rollback(self, reason: str, *, manual: bool) -> CanaryDecision:
        if not isinstance(reason, str) or reason == "":
            raise CanaryRejected("reason must be a non-empty string")
        if not isinstance(manual, bool):
            raise CanaryRejected("manual must be a boolean")
        state = self._state
        if state.phase not in ("GATE_PASSED", "CANARY_ACTIVE"):
            raise CanaryRejected("rollback requires GATE_PASSED or CANARY_ACTIVE")
        decided_at = self._clock()
        in_flight = state.outstanding
        self._state = replace(
            state,
            phase="ROLLED_BACK",
            in_flight_at_rollback=in_flight,
            served_before_rollback=state.candidate_episodes_served + in_flight,
            rollback_at=decided_at,
            rollback_reason=reason,
            rollback_manual=manual,
            promoted_at=None,
            held_pairs=(),
        )
        return CanaryDecision(
            action="rollback",
            state=self._state.phase,
            evidence=None,
            public_decision=None,
            snapshot=self.snapshot(),
        )

    def complete_outstanding(self, pair: PairedResult) -> DeploymentSnapshot:
        state = self._state
        if state.phase != "ROLLED_BACK":
            raise CanaryRejected("complete_outstanding requires ROLLED_BACK")
        if state.outstanding < 1:
            raise CanaryRejected("complete_outstanding requires outstanding work")
        self._validate_pair(pair)
        pair_id = pair.reference.episode.pair_id
        if pair_id is None or pair_id in state.seen_pair_ids:
            raise CanaryRejected("duplicate or missing pair_id")
        failed = _runtime_failed(pair.candidate)
        evaluator_failed = _evaluator_unsuccessful(pair.candidate)
        self._state = replace(
            state,
            candidate_episodes_served=state.candidate_episodes_served + 1,
            candidate_episodes_failed=state.candidate_episodes_failed
            + (1 if failed else 0),
            candidate_episodes_evaluator_unsuccessful=(
                state.candidate_episodes_evaluator_unsuccessful
                + (1 if evaluator_failed else 0)
            ),
            outstanding=state.outstanding - 1,
            seen_pair_ids=state.seen_pair_ids | {pair_id},
        )
        return self.snapshot()

    def abort_outstanding(self) -> DeploymentSnapshot:
        state = self._state
        if state.phase not in ("CANARY_ACTIVE", "ROLLED_BACK"):
            raise CanaryRejected(
                "abort_outstanding requires CANARY_ACTIVE or ROLLED_BACK"
            )
        if state.outstanding < 1:
            raise CanaryRejected("abort_outstanding requires outstanding work")
        self._state = replace(state, outstanding=state.outstanding - 1)
        return self.snapshot()

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
            candidate_episodes_evaluator_unsuccessful=(
                state.candidate_episodes_evaluator_unsuccessful
            ),
            outstanding=state.outstanding,
            in_flight_at_rollback=state.in_flight_at_rollback,
            served_before_rollback=state.served_before_rollback,
            rollback_at=state.rollback_at,
            rollback_reason=state.rollback_reason,
            rollback_manual=state.rollback_manual,
            promoted_at=state.promoted_at,
            promoted_configuration_hash=state.promoted_configuration_hash,
            monitoring_reset_required=state.monitoring_reset_required,
        )

    def _validate_pair(self, pair: PairedResult) -> None:
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
