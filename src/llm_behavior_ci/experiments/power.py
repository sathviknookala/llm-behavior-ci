"""Empirical paired and clustered power from measured baseline outcomes.

``harm_detection_power`` in ``validation.py`` is a normal approximation
from a supplied variance and stays advisory. This module simulates the
paired evaluator-success test instead, from local ``BaselineOutcome``
rows. A simulated pair draws a scenario cluster with replacement, then a
task in that scenario, then two independent repetitions of that task's
production outcome; the second is the candidate. A harmful candidate
loses each success with probability ``harm_margin / p``, where ``p`` is
the pooled production success rate, so the expected paired drop equals
the margin. The test is one-sided on scenario-cluster mean differences.
The null rate (no injected harm) is reported beside the power, so
repetition nondeterminism shows up as false alarms rather than being
assumed away. Neither number is a protocol threshold.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from llm_behavior_ci.experiments.validation import (
    BaselineOutcome,
    ValidationError,
    _normal_quantile,
)

POWER_METHOD = "empirical_clustered_paired_v1"


@dataclass(frozen=True)
class EmpiricalPowerRow:
    sample_size: int
    power: float
    null_rejection_rate: float


@dataclass(frozen=True)
class EmpiricalPowerReport:
    method: str
    harm_margin: float
    alpha: float
    simulations: int
    seed: int
    scenario_count: int
    task_count: int
    repeated_task_count: int
    production_success_rate: float
    rows: tuple[EmpiricalPowerRow, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "record": "empirical_power",
            "method": self.method,
            "harm_margin": self.harm_margin,
            "alpha": self.alpha,
            "simulations": self.simulations,
            "seed": self.seed,
            "scenario_count": self.scenario_count,
            "task_count": self.task_count,
            "repeated_task_count": self.repeated_task_count,
            "production_success_rate": self.production_success_rate,
            "normal_approximation": "advisory",
            "rows": [
                {
                    "sample_size": row.sample_size,
                    "power": row.power,
                    "null_rejection_rate": row.null_rejection_rate,
                }
                for row in self.rows
            ],
        }


def _task_key(outcome: BaselineOutcome) -> str:
    key = outcome.pair_key if outcome.pair_key is not None else outcome.scenario_id
    return key.split("#r", 1)[0]


def _clusters(
    outcomes: Sequence[BaselineOutcome],
) -> dict[str, dict[str, list[int]]]:
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for outcome in outcomes:
        if not isinstance(outcome, BaselineOutcome):
            raise ValidationError("outcomes must be baseline outcomes")
        if outcome.role != "production" or outcome.success is None:
            continue
        grouped[outcome.scenario_id][_task_key(outcome)].append(int(outcome.success))
    return {scenario: dict(tasks) for scenario, tasks in grouped.items()}


def _rejects(differences: Sequence[Sequence[int]], z: float) -> bool:
    means = [sum(cluster) / len(cluster) for cluster in differences]
    count = len(means)
    if count < 2:
        return False
    mean = sum(means) / count
    variance = sum((value - mean) ** 2 for value in means) / (count - 1)
    if variance == 0.0:
        return mean < 0.0
    return mean / math.sqrt(variance / count) < -z


def _simulate(
    clusters: dict[str, dict[str, list[int]]],
    *,
    sample_size: int,
    flip: float,
    z: float,
    rng: random.Random,
) -> bool:
    scenarios = sorted(clusters)
    differences: dict[int, list[int]] = defaultdict(list)
    for _ in range(sample_size):
        draw = rng.randrange(len(scenarios))
        tasks = clusters[scenarios[draw]]
        values = tasks[sorted(tasks)[rng.randrange(len(tasks))]]
        reference = values[rng.randrange(len(values))]
        candidate = values[rng.randrange(len(values))]
        if candidate == 1 and flip > 0.0 and rng.random() < flip:
            candidate = 0
        differences[draw].append(candidate - reference)
    return _rejects(list(differences.values()), z)


def empirical_harm_power(
    outcomes: Sequence[BaselineOutcome],
    *,
    harm_margin: float,
    alpha: float,
    sample_sizes: Sequence[int],
    simulations: int,
    seed: int,
) -> EmpiricalPowerReport:
    """Simulate power and the null rejection rate at each sample size.

    Rows without an evaluator outcome and do-nothing rows are excluded.
    At least two scenarios with an outcome are required, and the margin
    must not exceed the pooled production success rate.
    """

    if not 0.0 < alpha < 1.0:
        raise ValidationError("alpha must be in (0, 1)")
    if not isinstance(simulations, int) or isinstance(simulations, bool) or simulations < 1:
        raise ValidationError("simulations must be a positive integer")
    if not sample_sizes or any(
        not isinstance(size, int) or isinstance(size, bool) or size < 2 for size in sample_sizes
    ):
        raise ValidationError("sample sizes must be integers of at least 2")
    clusters = _clusters(outcomes)
    if len(clusters) < 2:
        raise ValidationError("empirical power needs outcomes from at least two scenarios")
    values = [value for tasks in clusters.values() for runs in tasks.values() for value in runs]
    rate = sum(values) / len(values)
    if not 0.0 < harm_margin <= rate:
        raise ValidationError(
            "harm margin must be positive and at most the production success rate"
        )
    flip = harm_margin / rate
    z = _normal_quantile(1.0 - alpha)
    rows: list[EmpiricalPowerRow] = []
    for size in sample_sizes:
        harmful = random.Random(f"{seed}:{size}:harm")
        null = random.Random(f"{seed}:{size}:null")
        hits = sum(
            _simulate(clusters, sample_size=size, flip=flip, z=z, rng=harmful)
            for _ in range(simulations)
        )
        false = sum(
            _simulate(clusters, sample_size=size, flip=0.0, z=z, rng=null)
            for _ in range(simulations)
        )
        rows.append(
            EmpiricalPowerRow(
                sample_size=size,
                power=hits / simulations,
                null_rejection_rate=false / simulations,
            )
        )
    tasks = [runs for scenario in clusters.values() for runs in scenario.values()]
    return EmpiricalPowerReport(
        method=POWER_METHOD,
        harm_margin=float(harm_margin),
        alpha=float(alpha),
        simulations=simulations,
        seed=seed,
        scenario_count=len(clusters),
        task_count=len(tasks),
        repeated_task_count=sum(1 for runs in tasks if len(runs) > 1),
        production_success_rate=rate,
        rows=tuple(rows),
    )
