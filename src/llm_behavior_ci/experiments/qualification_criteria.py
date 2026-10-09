"""The approved qualification criteria, evaluated from saved batch evidence.

``qualification-criteria-v1`` is the rule set recorded in
``docs/EVAL_PROTOCOL.md`` (Qualification criteria). Every function reads
outcomes the batch checkpoint already holds, calls no model, and never
replaces a missing outcome. Each criterion reports ``pass``, ``fail``,
``inconclusive``, or ``reported`` for the criteria that only fix how a
quantity is reported.

Execution A/A uses tasks with both H1 and H2 scored. Exchangeability is
the exact two-sided McNemar test on discordant pairs; zero discordant
pairs gives p = 1. A pass is no detected directional imbalance, not
equivalence. The harm classification is diagnostic and separate from the
measured label's eligibility. The CUSUM sensitivity analysis is a
binomial model: healthy outcomes are treated as independent Bernoulli
draws, although up to 38 of them come from 19 scenario clusters.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

from llm_behavior_ci.stats.canary import SequentialCanaryTest
from llm_behavior_ci.stats.cusum import CUSUM
from llm_behavior_ci.stats.evidence import PairedSuccess

CRITERIA_VERSION = "qualification-criteria-v1"
AA_EXEC_ALPHA = 0.05
AA_EXEC_MIN_COMPLETE = 17
CANARY_ALPHA = 0.05
CANARY_MARGIN = 0.2
CANARY_HORIZON = 12
CUSUM_MIN_OUTCOMES = 34
CUSUM_MIN_SCENARIOS = 17
CUSUM_HORIZON = 25
CUSUM_SIMULATIONS = 10_000
CUSUM_SIMULATION_SEED = 17
WILSON_Z = 1.959963984540054
HARM_PARAMETERS = {
    "margin": 0.2,
    "confidence_level": 0.95,
    "resamples": 1000,
    "seed": 17,
}
PLAN_PAIRS = 30
HEALTHY_ARMS = ("H1_healthy_reference", "H2_healthy_repeat")


def exact_mcnemar_p(first_only: int, second_only: int) -> float:
    """Two-sided exact binomial p-value (p = 0.5) on the discordant counts.

    With no discordant pair the p-value is 1.
    """

    total = first_only + second_only
    if total == 0:
        return 1.0
    smaller = min(first_only, second_only)
    tail = sum(math.comb(total, k) for k in range(smaller + 1)) / 2.0**total
    return min(1.0, 2.0 * tail)


def wilson_bounds(rate: float, trials: int) -> tuple[float, float]:
    """95% Wilson interval for ``rate`` observed on ``trials`` independent draws."""

    z2 = WILSON_Z * WILSON_Z
    denominator = 1.0 + z2 / trials
    center = (rate + z2 / (2.0 * trials)) / denominator
    half = (
        WILSON_Z
        * math.sqrt(rate * (1.0 - rate) / trials + z2 / (4.0 * trials * trials))
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def cusum_null_alarm_rate(
    scale: Mapping[str, Any], healthy_rate: float, *, rng: random.Random
) -> float:
    """Share of simulated healthy Bernoulli streams on which the decrease CUSUM alarms."""

    alarms = 0
    for _ in range(CUSUM_SIMULATIONS):
        detector = CUSUM(
            target=float(scale["target"]),
            slack=float(scale["slack"]),
            threshold=float(scale["threshold"]),
            direction="decrease",
        )
        for _ in range(CUSUM_HORIZON):
            if detector.update(1.0 if rng.random() < healthy_rate else 0.0).alarm:
                alarms += 1
                break
    return alarms / CUSUM_SIMULATIONS


def harm_classification(label: Mapping[str, Any]) -> str:
    """P-HARM-1's diagnostic class for one measured label.

    The interval is on candidate minus reference success.
    """

    if label.get("status") != "measured":
        return "not_classified"
    margin = float(label["margin"])
    if float(label["interval_high"]) <= -margin:
        return "confirmed_harmful"
    if float(label["interval_low"]) > -margin:
        return "confirmed_not_harmful"
    return "indeterminate"


def _both_scored(table: Sequence[Mapping[str, Mapping[str, Any]]]) -> list[tuple[bool, bool]]:
    pairs = []
    for row in table:
        first = row[HEALTHY_ARMS[0]]["success"]
        second = row[HEALTHY_ARMS[1]]["success"]
        if isinstance(first, bool) and isinstance(second, bool):
            pairs.append((first, second))
    return pairs


def execution_criteria(
    table: Sequence[Mapping[str, Mapping[str, Any]]],
    scale: Mapping[str, Any],
    harm_labels: Mapping[str, Mapping[str, Any]],
    *,
    harm_parameters: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """P-AA-EXEC-1..3, P-CUSUM-1..2, and P-HARM-1..2 on the dev batch."""

    pairs = _both_scored(table)
    complete = len(pairs) >= AA_EXEC_MIN_COMPLETE
    criteria: dict[str, dict[str, Any]] = {}
    criteria["P-AA-EXEC-3"] = {
        "status": "pass" if complete else "inconclusive",
        "tasks": len(table),
        "both_scored": len(pairs),
        "required": AA_EXEC_MIN_COMPLETE,
    }
    first_only = sum(1 for first, second in pairs if first and not second)
    second_only = sum(1 for first, second in pairs if second and not first)
    exchange: dict[str, Any] = {
        "discordant": first_only + second_only,
        "h1_only_success": first_only,
        "h2_only_success": second_only,
        "alpha": AA_EXEC_ALPHA,
    }
    if not complete:
        exchange["status"] = "inconclusive"
        exchange["reason"] = "P-AA-EXEC-3 not met"
    else:
        p_value = exact_mcnemar_p(first_only, second_only)
        exchange["p_value"] = p_value
        exchange["status"] = "pass" if p_value >= AA_EXEC_ALPHA else "fail"
        exchange["reading"] = (
            "no detected directional imbalance"
            if p_value >= AA_EXEC_ALPHA
            else "directional imbalance detected"
        )
    criteria["P-AA-EXEC-1"] = exchange
    canary: dict[str, Any] = {
        "alpha": CANARY_ALPHA,
        "margin": CANARY_MARGIN,
        "horizon": CANARY_HORIZON,
    }
    if len(pairs) < CANARY_HORIZON:
        canary.update({"status": "inconclusive", "pairs": len(pairs)})
    else:
        test = SequentialCanaryTest(
            alpha=CANARY_ALPHA, harm_margin=CANARY_MARGIN, horizon_episodes=CANARY_HORIZON
        )
        rollback_at = None
        for index, (first, second) in enumerate(pairs[:CANARY_HORIZON], start=1):
            evidence = test.update(
                PairedSuccess(candidate=float(second), reference=float(first))
            )
            if evidence.alarm:
                rollback_at = index
                break
        canary.update(
            {
                "status": "fail" if rollback_at is not None else "pass",
                "pairs": CANARY_HORIZON,
                "rollback_at": rollback_at,
            }
        )
    criteria["P-AA-EXEC-2"] = canary
    criteria["P-CUSUM-1"] = _cusum_sample(scale)
    criteria["P-CUSUM-2"] = _cusum_sensitivity(scale)
    criteria["P-HARM-1"] = {
        "status": "reported",
        "classifications": {
            name: harm_classification(label) for name, label in harm_labels.items()
        },
        "eligibility": "separate: a measured label is eligible whatever its class",
        "claims": "confirmed regression harm and detection sensitivity are claimed only for a confirmed class",
        "limitations": "scenario-clustered percentile bootstrap on 19 binary pairs is approximate; under the Gaussian simulation its coverage was 0.9351 against nominal 0.95, and with few clusters and discrete outcomes it can undercover",
    }
    supplied = {key: harm_parameters.get(key) for key in HARM_PARAMETERS}
    criteria["P-HARM-2"] = {
        "status": "pass" if supplied == HARM_PARAMETERS else "fail",
        "approved": dict(HARM_PARAMETERS),
        "supplied": supplied,
        "clusters": "scenario",
    }
    return criteria


def _cusum_sample(scale: Mapping[str, Any]) -> dict[str, Any]:
    outcomes = int(scale.get("scored_outcomes", 0))
    scenarios = int(scale.get("scenarios", 0))
    met = outcomes >= CUSUM_MIN_OUTCOMES and scenarios >= CUSUM_MIN_SCENARIOS
    return {
        "status": "pass" if met else "inconclusive",
        "scale": "final" if met else "provisional",
        "scored_outcomes": outcomes,
        "scenarios": scenarios,
        "required_outcomes": CUSUM_MIN_OUTCOMES,
        "required_scenarios": CUSUM_MIN_SCENARIOS,
    }


def _cusum_sensitivity(scale: Mapping[str, Any]) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "model": "binomial sensitivity analysis; healthy outcomes simulated as independent Bernoulli draws",
        "dependence": "the scored healthy outcomes (up to 38) come from at most 19 scenario clusters; these bounds are not robust to scenario dependence and carry no false-alarm guarantee",
        "horizon": CUSUM_HORIZON,
        "simulations": CUSUM_SIMULATIONS,
        "seed": CUSUM_SIMULATION_SEED,
    }
    if scale.get("status") != "estimated" or float(scale["sigma"]) <= 0.0:
        entry["status"] = "inconclusive"
        entry["reason"] = "no positive scale estimate"
        return entry
    rate = float(scale["target"])
    scenarios = int(scale["scenarios"])
    low, high = wilson_bounds(rate, scenarios)
    rng = random.Random(CUSUM_SIMULATION_SEED)
    entry.update(
        {
            "status": "reported",
            "estimate": rate,
            "wilson_trials": scenarios,
            "wilson_low": low,
            "wilson_high": high,
            "null_alarm_rate": {
                "at_estimate": cusum_null_alarm_rate(scale, rate, rng=rng),
                "at_wilson_low": cusum_null_alarm_rate(scale, low, rng=rng),
                "at_wilson_high": cusum_null_alarm_rate(scale, high, rng=rng),
            },
        }
    )
    return entry


def plan_criteria(
    pairs: Sequence[tuple[Any, Any] | None],
    gate_replay: Mapping[str, Any],
    *,
    plan_complete: Any,
) -> dict[str, dict[str, Any]]:
    """P-AA-PLAN-1..2 on the plan batch.

    ``plan_complete`` decides whether one stored plan episode is a
    completed, non-empty plan.
    """

    complete = sum(
        1
        for pair in pairs
        if pair is not None and all(plan_complete(episode) for episode in pair)
    )
    criteria: dict[str, dict[str, Any]] = {
        "P-AA-PLAN-2": {
            "status": "pass" if complete == PLAN_PAIRS == len(pairs) else "inconclusive",
            "complete_pairs": complete,
            "required": PLAN_PAIRS,
        }
    }
    status = gate_replay.get("status")
    if status != "decided":
        gate = {"status": "inconclusive", "replay": status}
    else:
        outcome = gate_replay.get("outcome")
        gate = {"status": "pass" if outcome == "PASS" else "fail", "outcome": outcome}
    criteria["P-AA-PLAN-1"] = gate
    return criteria


def failure_rule(criteria: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Any failed criterion keeps the protocol BLOCKED."""

    failed = sorted(name for name, entry in criteria.items() if entry.get("status") == "fail")
    return {"status": "blocked" if failed else "no_failure", "failed": failed}
