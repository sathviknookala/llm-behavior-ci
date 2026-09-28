"""Reproducible validation for the implemented statistics.

``validate_method`` runs caller-supplied seeds, reference cases, and A/A
inputs through the statistics already implemented in ``stats``. Thresholds,
horizons, and tolerances come from the method spec. This module does not
read the offline-gate demo constants, does not start AppWorld or vLLM, and
does not read ``nvidia-smi``.

A report can show that a method is implemented while ``benchmark_eligible``
stays false. Eligibility requires every minimum check to be requested and
to pass. Real A/A dependence is accepted only when the caller sets
provenance to ``local_runtime`` or to ``gpu`` with a hardware reading.
Synthetic rows and empty captures do not become a noise floor.

Reference values are supplied by the caller. SciPy, River, and confseq are
reported as present or absent and are not executed. ``score_top_k`` does
not accept a vocabulary size, so the truncation comparison calls
``truncated_next_token_kl`` with that setting and ``score_full`` for the
full distribution. ``MonitorObservation`` has no scenario or task id;
those labels come from ``AAContext`` or from an ``AAStudyRow``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from statistics import fmean
from typing import Callable

from llm_behavior_ci.config import SPLITS
from llm_behavior_ci.records import MonitorObservation, RecordError, assert_public_payload
from llm_behavior_ci.runtime.aa_capture import AACaptureResult
from llm_behavior_ci.runtime.scoring import score_full
from llm_behavior_ci.stats.adwin import ADWIN
from llm_behavior_ci.stats.bootstrap import (
    PairedBootstrapError,
    cluster_draw_indexes,
    clustered_paired_bootstrap,
    paired_bootstrap,
)
from llm_behavior_ci.stats.c2st import C2STError, classifier_two_sample_test
from llm_behavior_ci.stats.canary import SequentialCanaryTest
from llm_behavior_ci.stats.chi_square import (
    ChiSquareError,
    chi_square_goodness_of_fit,
    chi_square_homogeneity,
)
from llm_behavior_ci.stats.confidence_sequence import BoundedMeanCS, PairedDifferenceCS
from llm_behavior_ci.stats.cusum import CUSUM
from llm_behavior_ci.stats.e_detector import BettingEDetector
from llm_behavior_ci.stats.evidence import PairedSuccess, StatisticsError
from llm_behavior_ci.stats.harmful_shift import HarmfulShiftTest
from llm_behavior_ci.stats.kl import (
    NextTokenKLError,
    TruncatedKLError,
    next_token_kl,
    truncated_next_token_kl,
)
from llm_behavior_ci.stats.ks import KSError, ks_two_sample
from llm_behavior_ci.stats.mmd import MMDError, mmd_permutation_test

_STUDIES = frozenset({"cpu_fast", "simulation", "gpu"})
_PROVENANCE = frozenset(
    {"unavailable", "synthetic", "local_runtime", "gpu"}
)
_REFERENCE_SOURCES = frozenset(
    {"caller", "closed_form", "scipy", "river", "confseq"}
)
_HELD_OUT = frozenset({"test_normal", "test_challenge"})
_TOKEN = __import__("re").compile(r"^[a-z][a-z0-9_]{0,63}$")
_INT_PARAMETERS = frozenset(
    {
        "categories",
        "trials",
        "permutations",
        "steps",
        "dimension",
        "resamples",
        "cluster_size",
        "horizon_episodes",
    }
)
_STAT_ERRORS = (
    StatisticsError,
    ChiSquareError,
    KSError,
    MMDError,
    C2STError,
    PairedBootstrapError,
    NextTokenKLError,
    TruncatedKLError,
)
_PAIRED_DETECTORS = frozenset(
    {"paired_difference_cs", "harmful_shift", "sequential_canary"}
)
_BATCH = frozenset(
    {
        "ks_two_sample",
        "chi_square_homogeneity",
        "chi_square_goodness_of_fit",
        "mmd_permutation_test",
        "classifier_two_sample_test",
        "paired_bootstrap",
        "clustered_paired_bootstrap",
    }
)


class ValidationError(ValueError):
    pass


@dataclass(frozen=True)
class StudyBudget:
    """Runner cap for one study kind.

    These caps keep a fast CPU call from becoming a long simulation. They
    are not false-alarm levels, margins, or protocol thresholds.
    """

    max_seeds: int
    max_sample_size: int
    max_horizon: int
    max_permutations: int
    max_resamples: int
    max_work: int


_CPU_BUDGET = StudyBudget(40, 64, 64, 40, 200, 20_000)
STUDY_BUDGETS = {
    "cpu_fast": _CPU_BUDGET,
    "simulation": StudyBudget(2_000, 5_000, 5_000, 2_000, 20_000, 5_000_000),
    "gpu": _CPU_BUDGET,
}


@dataclass(frozen=True)
class _Contract:
    name: str
    kind: str
    null_claim: str
    null_draws: frozenset[str]
    parameters: frozenset[str]
    always_parameters: frozenset[str]
    draw_parameters: tuple[tuple[str, frozenset[str]], ...]
    checks: tuple[str, ...]
    statistics: frozenset[str]
    data_keys: frozenset[str]
    optional_data: frozenset[str]


def _contract(
    name: str,
    kind: str,
    null_claim: str,
    null_draws: frozenset[str],
    parameters: frozenset[str],
    always: frozenset[str],
    draws: tuple[tuple[str, frozenset[str]], ...],
    checks: tuple[str, ...],
    statistics: frozenset[str],
    data_keys: frozenset[str],
    optional_data: frozenset[str] = frozenset(),
) -> _Contract:
    return _Contract(
        name,
        kind,
        null_claim,
        null_draws,
        parameters,
        always,
        draws,
        checks,
        statistics,
        data_keys,
        optional_data,
    )


_NONE = frozenset()
_LEVEL_WINDOW = (
    "null_false_alarm",
    "repeated_look",
    "reference",
    "aa_dependence",
)
_BOOTSTRAP_CHECKS = (
    "null_false_alarm",
    "coverage",
    "repeated_look",
    "reference",
    "aa_dependence",
)
_BOUND_SEQUENCE = (
    "null_false_alarm",
    "stopping",
    "reference",
    "aa_dependence",
)
_COVERAGE_SEQUENCE = (
    "null_false_alarm",
    "coverage",
    "stopping",
    "reference",
    "aa_dependence",
)
_P_VALUE = frozenset({"p_value", "statistic"})
_DETECTOR_STATS = frozenset({"alarm", "ever_alarmed", "estimate"})

_CONTRACTS: dict[str, _Contract] = {
    item.name: item
    for item in (
        _contract(
            "ks_two_sample",
            "batch",
            "level",
            frozenset({"uniform"}),
            _NONE,
            _NONE,
            (),
            _LEVEL_WINDOW,
            _P_VALUE,
            frozenset({"left", "right"}),
        ),
        _contract(
            "chi_square_homogeneity",
            "batch",
            "level",
            frozenset({"multinomial"}),
            frozenset({"categories", "trials"}),
            frozenset({"categories", "trials"}),
            (),
            _LEVEL_WINDOW,
            _P_VALUE,
            frozenset({"left", "right"}),
        ),
        _contract(
            "chi_square_goodness_of_fit",
            "batch",
            "level",
            frozenset({"multinomial"}),
            frozenset({"categories", "trials"}),
            frozenset({"categories", "trials"}),
            (),
            _LEVEL_WINDOW,
            _P_VALUE,
            frozenset({"observed", "expected"}),
        ),
        _contract(
            "mmd_permutation_test",
            "batch",
            "level",
            frozenset({"gaussian"}),
            frozenset(
                {"bandwidth", "permutations", "dimension", "null_mean", "null_scale"}
            ),
            frozenset(
                {"bandwidth", "permutations", "dimension", "null_mean", "null_scale"}
            ),
            (),
            _LEVEL_WINDOW,
            frozenset({"p_value", "mmd_squared"}),
            frozenset({"production", "candidate", "seed", "clusters"}),
            frozenset({"clusters"}),
        ),
        _contract(
            "classifier_two_sample_test",
            "batch",
            "level",
            frozenset({"gaussian"}),
            frozenset(
                {
                    "permutations",
                    "steps",
                    "learning_rate",
                    "l2",
                    "dimension",
                    "null_mean",
                    "null_scale",
                }
            ),
            frozenset(
                {
                    "permutations",
                    "steps",
                    "learning_rate",
                    "l2",
                    "dimension",
                    "null_mean",
                    "null_scale",
                }
            ),
            (),
            _LEVEL_WINDOW,
            frozenset({"p_value", "accuracy"}),
            frozenset({"production", "candidate", "seed"}),
        ),
        _contract(
            "paired_bootstrap",
            "batch",
            "level",
            frozenset({"gaussian"}),
            frozenset({"confidence_level", "resamples", "null_mean", "null_scale"}),
            frozenset({"confidence_level", "resamples", "null_mean", "null_scale"}),
            (),
            _BOOTSTRAP_CHECKS,
            frozenset({"mean_difference", "confidence_low", "confidence_high"}),
            frozenset({"candidate", "production", "seed"}),
        ),
        _contract(
            "clustered_paired_bootstrap",
            "batch",
            "level",
            frozenset({"gaussian"}),
            frozenset(
                {
                    "confidence_level",
                    "resamples",
                    "null_mean",
                    "null_scale",
                    "cluster_size",
                }
            ),
            frozenset(
                {
                    "confidence_level",
                    "resamples",
                    "null_mean",
                    "null_scale",
                    "cluster_size",
                }
            ),
            (),
            _BOOTSTRAP_CHECKS,
            frozenset({"mean_difference", "confidence_low", "confidence_high"}),
            frozenset({"candidate", "production", "clusters", "seed"}),
        ),
        _contract(
            "cusum",
            "sequential",
            "record",
            frozenset({"constant", "gaussian"}),
            frozenset({"target", "slack", "threshold", "direction", "null_scale"}),
            frozenset({"target", "slack", "threshold", "direction"}),
            (("gaussian", frozenset({"null_scale"})),),
            ("null_false_alarm", "stopping", "reference", "aa_dependence"),
            _DETECTOR_STATS,
            frozenset({"observations"}),
        ),
        _contract(
            "adwin",
            "sequential",
            "record",
            frozenset({"constant", "uniform"}),
            frozenset({"delta", "constant_value"}),
            frozenset({"delta"}),
            (("constant", frozenset({"constant_value"})),),
            ("null_false_alarm", "stopping", "reference", "aa_dependence"),
            _DETECTOR_STATS,
            frozenset({"observations"}),
        ),
        _contract(
            "bounded_mean_cs",
            "sequential",
            "at_most",
            frozenset({"constant", "bernoulli"}),
            frozenset({"alpha", "lower", "upper", "null_mean", "null_probability"}),
            frozenset({"alpha", "lower", "upper", "null_mean"}),
            (("bernoulli", frozenset({"null_probability"})),),
            _COVERAGE_SEQUENCE,
            _DETECTOR_STATS,
            frozenset({"observations"}),
        ),
        _contract(
            "paired_difference_cs",
            "sequential",
            "at_most",
            frozenset({"constant", "bernoulli"}),
            frozenset(
                {
                    "alpha",
                    "lower",
                    "upper",
                    "null_mean",
                    "null_probability",
                    "pair_value",
                }
            ),
            frozenset({"alpha", "lower", "upper", "null_mean"}),
            (
                ("constant", frozenset({"pair_value"})),
                ("bernoulli", frozenset({"null_probability"})),
            ),
            _COVERAGE_SEQUENCE,
            _DETECTOR_STATS,
            frozenset({"pairs"}),
        ),
        _contract(
            "betting_e_detector",
            "sequential",
            "at_most",
            frozenset({"constant", "bernoulli"}),
            frozenset(
                {
                    "null_mean",
                    "alpha",
                    "lower",
                    "upper",
                    "direction",
                    "null_probability",
                }
            ),
            frozenset({"null_mean", "alpha", "lower", "upper", "direction"}),
            (("bernoulli", frozenset({"null_probability"})),),
            _BOUND_SEQUENCE,
            _DETECTOR_STATS,
            frozenset({"observations"}),
        ),
        _contract(
            "harmful_shift",
            "sequential",
            "at_most",
            frozenset({"constant", "bernoulli"}),
            frozenset({"alpha", "harm_margin", "pair_value", "null_probability"}),
            frozenset({"alpha", "harm_margin"}),
            (
                ("constant", frozenset({"pair_value"})),
                ("bernoulli", frozenset({"null_probability"})),
            ),
            _BOUND_SEQUENCE,
            _DETECTOR_STATS,
            frozenset({"pairs"}),
        ),
        _contract(
            "sequential_canary",
            "sequential",
            "at_most",
            frozenset({"constant", "bernoulli"}),
            frozenset(
                {
                    "alpha",
                    "harm_margin",
                    "horizon_episodes",
                    "pair_value",
                    "null_probability",
                }
            ),
            frozenset({"alpha", "harm_margin", "horizon_episodes"}),
            (
                ("constant", frozenset({"pair_value"})),
                ("bernoulli", frozenset({"null_probability"})),
            ),
            _BOUND_SEQUENCE,
            _DETECTOR_STATS | frozenset({"stopped_at_horizon"}),
            frozenset({"pairs"}),
        ),
        _contract(
            "next_token_kl",
            "score",
            "none",
            frozenset({"none"}),
            _NONE,
            _NONE,
            (),
            ("reference",),
            frozenset({"mean_kl_nats"}),
            frozenset({"production", "candidate"}),
        ),
        _contract(
            "truncated_next_token_kl",
            "score",
            "none",
            frozenset({"none"}),
            _NONE,
            _NONE,
            (),
            ("reference", "kl_approximation"),
            frozenset({"mean_kl_nats"}),
            frozenset({"production", "candidate", "vocabulary_size"}),
            frozenset({"vocabulary_size"}),
        ),
    )
}


@dataclass(frozen=True)
class MethodSpec:
    """Explicit inputs for one validation run.

    ``false_alarm_tolerance`` and ``coverage_tolerance`` are the caller's
    Monte Carlo allowances. They are not nominal α. ``null_claim`` is a
    property of the implementation: ``level`` methods are judged near α,
    ``at_most`` methods are judged at or below α, and ``record`` methods
    are measured without a nominal false-alarm guarantee.
    """

    name: str
    parameters: tuple[tuple[str, int | float | str], ...]
    required_checks: tuple[str, ...]
    study: str
    null_sample_size: int
    uncertainty_level: float
    null_draw: str
    alpha: float | None = None
    horizon: int | None = None
    false_alarm_tolerance: float | None = None
    coverage_tolerance: float | None = None
    repeated_look_stride: int | None = None

    def __post_init__(self) -> None:
        _validate_spec(self)

    def parameter_map(self) -> dict[str, int | float | str]:
        return dict(self.parameters)


@dataclass(frozen=True)
class ReferenceCase:
    """One supplied reference value.

    The expected number is an input. A missing library does not create one.
    """

    case_id: str
    statistic: str
    expected: float
    tolerance: float
    source: str
    payload_json: str

    def payload(self) -> dict[str, object]:
        parsed = json.loads(self.payload_json)
        if not isinstance(parsed, dict):
            raise ValidationError("reference payload must be an object")
        return parsed


@dataclass(frozen=True)
class AAContext:
    """Labels aligned with monitor observations.

    Task and scenario ids are hashed into the input digest and are not
    copied into the public summary. Provenance is not inferred.
    """

    provenance: str
    hardware_observed: bool
    scenario_ids: tuple[str | None, ...] | None = None
    task_ids: tuple[str, ...] | None = None
    repetitions: tuple[int, ...] | None = None
    memory_used_mib: int | None = None
    wall_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.provenance not in _PROVENANCE:
            raise ValidationError("provenance is unknown")
        if not isinstance(self.hardware_observed, bool):
            raise ValidationError("hardware_observed must be a boolean")
        if not self.hardware_observed and (
            self.memory_used_mib is not None or self.wall_seconds is not None
        ):
            raise ValidationError("hardware readings require hardware_observed")
        _optional_int(self.memory_used_mib, "memory_used_mib", minimum=0)
        _optional_finite(self.wall_seconds, "wall_seconds", minimum=0.0)


@dataclass(frozen=True)
class AAStudyRow:
    """One paired A/A row.

    ``disagreement`` follows the evaluator flags. A missing outcome is not
    an agreement and is not a success. Task ids stay out of public reports.
    """

    scenario_id: str | None
    task_id: str
    repetition: int
    reference_success: float | None
    candidate_success: float | None
    disagreement: bool | None
    reference_requirement_fraction: float | None
    candidate_requirement_fraction: float | None
    trajectory_diverged: bool
    length_difference: int
    concurrency: int

    def __post_init__(self) -> None:
        _short_text(self.task_id, "task_id")
        if self.scenario_id is not None:
            _short_text(self.scenario_id, "scenario_id")
        _nonnegative_int(self.repetition, "repetition")
        _positive_int(self.concurrency, "concurrency")
        if isinstance(self.length_difference, bool) or not isinstance(
            self.length_difference, int
        ):
            raise ValidationError("length_difference must be an int")
        if not isinstance(self.trajectory_diverged, bool):
            raise ValidationError("trajectory_diverged must be a boolean")
        _optional_success(self.reference_success, "reference_success")
        _optional_success(self.candidate_success, "candidate_success")
        _optional_fraction(
            self.reference_requirement_fraction,
            "reference_requirement_fraction",
        )
        _optional_fraction(
            self.candidate_requirement_fraction,
            "candidate_requirement_fraction",
        )
        _match_fraction(self.reference_success, self.reference_requirement_fraction)
        _match_fraction(self.candidate_success, self.candidate_requirement_fraction)
        if self.reference_success is None or self.candidate_success is None:
            if self.disagreement is not None:
                raise ValidationError(
                    "missing evaluator outcomes are not a disagreement"
                )
            return
        if not isinstance(self.disagreement, bool):
            raise ValidationError("disagreement must match the evaluator flags")
        expected = self.reference_success != self.candidate_success
        if self.disagreement != expected:
            raise ValidationError("disagreement must match the evaluator flags")


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    reason: str
    sample_count: int
    seeds: tuple[int, ...]
    estimate: float | None
    interval_low: float | None
    interval_high: float | None
    uncertainty_level: float | None
    details: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class CaseAgreement:
    case_id: str
    statistic: str
    expected: float
    observed: float
    absolute_error: float
    tolerance: float
    source: str
    agreed: bool


@dataclass(frozen=True)
class AADependenceReport:
    """Dependence measured from supplied A/A rows or monitor labels.

    ``evidence_accepted`` is false for synthetic, unavailable, and unread
    GPU provenance. Effect fields stay empty in that case.
    """

    status: str
    evidence_accepted: bool
    provenance: str
    hardware_observed: bool
    observation_count: int
    pair_count: int
    repeated_task_effect: float | None
    scenario_clustering_effect: float | None
    inference_variation: float | None
    inference_source: str | None
    inference_low: float | None
    inference_high: float | None
    trajectory_divergence_rate: float | None
    interval_width_ratio: float | None
    concurrency_effect: float | None
    concurrency_levels: tuple[int, ...]
    series_alarm: bool | None
    memory_used_mib: int | None
    wall_seconds: float | None
    reason: str


@dataclass(frozen=True)
class ValidationReport:
    """One method after a validation run.

    ``implemented`` means the name is in this package's catalog.
    ``validated`` and ``benchmark_eligible`` are true only when the minimum
    checks passed on accepted evidence. ``gpu_floor_measured`` stays false:
    this runner does not record a live full-vocabulary floor.
    """

    method: str
    implemented: bool
    validated: bool
    benchmark_eligible: bool
    calibration: str
    study: str
    null_claim: str
    null_draw: str
    input_hash: str
    seeds: tuple[int, ...]
    sample_count: int
    null_sample_size: int
    uncertainty_level: float
    alpha: float | None
    horizon: int | None
    false_alarm_tolerance: float | None
    coverage_tolerance: float | None
    parameters: tuple[tuple[str, str], ...]
    required_checks: tuple[str, ...]
    omitted_checks: tuple[str, ...]
    checks: tuple[CheckResult, ...]
    reference_agreements: tuple[CaseAgreement, ...]
    aa: AADependenceReport
    libraries: tuple[tuple[str, bool], ...]
    configuration_hashes: tuple[str, ...]
    gpu_evidence: bool
    gpu_floor_measured: bool
    split: str | None


@dataclass(frozen=True)
class ImplementedMethod:
    name: str
    null_claim: str
    applicable_checks: tuple[str, ...]
    minimum_checks: tuple[str, ...]
    validated: bool


@dataclass(frozen=True)
class MethodValidationStatus:
    name: str
    implemented: bool
    validated: bool
    benchmark_eligible: bool
    calibration: str
    gpu_floor_measured: bool
    gpu_evidence: bool
    aa_evidence_accepted: bool


@dataclass(frozen=True)
class KLPositionSample:
    """Aligned full-vocabulary log-probabilities for one trace."""

    production_log_probabilities: tuple[tuple[float, ...], ...]
    candidate_log_probabilities: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class KLApproximationReport:
    """Top-k KL against full-vocabulary KL on a supplied sample.

    ``gpu_floor_measured`` is false. A passed status is the supplied-sample
    error, not a vLLM truncation floor.
    """

    status: str
    approximation: str
    provenance: str
    top_k: int | None
    vocabulary_size: int | None
    sample_count: int
    position_count: int
    mean_signed_error: float | None
    mean_absolute_error: float | None
    max_absolute_error: float | None
    sample_absolute_errors: tuple[float, ...]
    input_hash: str
    gpu_floor_measured: bool
    reason: str


@dataclass(frozen=True)
class BaselineOutcome:
    """One evaluator outcome used before harm characterization.

    ``success`` is the evaluator flag. A plan and an agent completion claim
    are not accepted here. ``scenario_id`` stays out of the public summary.
    """

    role: str
    success: bool | None
    requirement_fraction: float | None
    scenario_id: str
    app: str | None
    difficulty: int | None
    pair_key: str | None = None

    def __post_init__(self) -> None:
        if self.role not in {"production", "do_nothing"}:
            raise ValidationError("role must be production or do_nothing")
        if self.success is not None and not isinstance(self.success, bool):
            raise ValidationError("success must be a boolean or null")
        _optional_fraction(self.requirement_fraction, "requirement_fraction")
        if self.success is not None and self.requirement_fraction is not None:
            if self.success != (self.requirement_fraction == 1.0):
                raise ValidationError(
                    "success does not match the requirement fraction"
                )
        _short_text(self.scenario_id, "scenario_id")
        if self.app is not None:
            _short_text(self.app, "app")
        if self.difficulty is not None and self.difficulty not in {1, 2, 3}:
            raise ValidationError("difficulty must be 1, 2, or 3")
        if self.pair_key is not None:
            _short_text(self.pair_key, "pair_key")


@dataclass(frozen=True)
class TaskMixSlice:
    app: str | None
    difficulty: int | None
    episode_count: int
    scenario_count: int
    success_rate: float | None
    confidence_low: float | None
    confidence_high: float | None


@dataclass(frozen=True)
class PowerAtSize:
    sample_size: int
    power: float
    meets_target: bool


@dataclass(frozen=True)
class HarmStudyReport:
    """Whether the supplied harm margin is detectable, and the sanity check.

    Power uses a one-sided normal approximation and the caller-supplied
    variance. Beating the do-nothing agent is an evaluator sanity check,
    not a monitoring result. ``feasible`` is null when that sanity check
    has no outcomes.
    """

    split: str
    study: str
    harm_margin: float
    alpha: float
    variance: float
    power_target: float
    input_hash: str
    powers: tuple[PowerAtSize, ...]
    production_success_rate: float | None
    do_nothing_success_rate: float | None
    production_low: float | None
    production_high: float | None
    do_nothing_low: float | None
    do_nothing_high: float | None
    success_difference: float | None
    difference_low: float | None
    difference_high: float | None
    production_beats_do_nothing: bool | None
    requirement_fraction_difference: float | None
    task_mix_effect: float | None
    slices: tuple[TaskMixSlice, ...]
    episode_count: int
    scenario_count: int
    missing_success_count: int
    feasible: bool | None
    feasibility_reason: str
    empirical_evidence: bool


@dataclass(frozen=True)
class _Replicate:
    false_alarm: bool | None
    covered: bool | None
    stopping_time: int | None
    censored: bool | None
    single_alarm: bool | None
    repeated_alarm: bool | None
    horizon_ok: bool | None
    excluded: bool


def minimum_checks(name: str) -> tuple[str, ...]:
    """Return the checks a method must pass before it is benchmark-eligible."""

    return _lookup(name).checks


def implemented_methods() -> tuple[ImplementedMethod, ...]:
    """Catalog every implemented method as not yet validated."""

    return tuple(
        ImplementedMethod(
            name=contract.name,
            null_claim=contract.null_claim,
            applicable_checks=contract.checks,
            minimum_checks=contract.checks,
            validated=False,
        )
        for contract in _CONTRACTS.values()
    )


def reference_library_status() -> tuple[tuple[str, bool], ...]:
    """Report whether optional reference libraries can be imported.

    Presence is not agreement. This function does not call them.
    """

    return tuple(
        (name, importlib.util.find_spec(name) is not None)
        for name in ("confseq", "river", "scipy")
    )


def method_spec(
    name: str,
    parameters: Mapping[str, object],
    *,
    required_checks: Sequence[str],
    study: str,
    null_sample_size: int,
    uncertainty_level: float,
    null_draw: str,
    alpha: float | None = None,
    horizon: int | None = None,
    false_alarm_tolerance: float | None = None,
    coverage_tolerance: float | None = None,
    repeated_look_stride: int | None = None,
) -> MethodSpec:
    """Build a method spec from explicit parameters."""

    return MethodSpec(
        name=name,
        parameters=_freeze_parameters(parameters),
        required_checks=_string_tuple(required_checks, "required_checks"),
        study=study,
        null_sample_size=null_sample_size,
        uncertainty_level=uncertainty_level,
        null_draw=null_draw,
        alpha=alpha,
        horizon=horizon,
        false_alarm_tolerance=false_alarm_tolerance,
        coverage_tolerance=coverage_tolerance,
        repeated_look_stride=repeated_look_stride,
    )


def method_spec_from_dict(payload: object) -> MethodSpec:
    """Load a method spec. Unknown fields are rejected."""

    if not isinstance(payload, dict):
        raise ValidationError("method spec must be an object")
    allowed = {
        "name",
        "parameters",
        "required_checks",
        "study",
        "null_sample_size",
        "uncertainty_level",
        "null_draw",
        "alpha",
        "horizon",
        "false_alarm_tolerance",
        "coverage_tolerance",
        "repeated_look_stride",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValidationError("method spec has an unknown field")
    missing = {
        "name",
        "parameters",
        "required_checks",
        "study",
        "null_sample_size",
        "uncertainty_level",
        "null_draw",
    } - set(payload)
    if missing:
        raise ValidationError("method spec is missing a field")
    parameters = payload["parameters"]
    if not isinstance(parameters, dict):
        raise ValidationError("parameters must be an object")
    return method_spec(
        payload["name"],
        parameters,
        required_checks=payload["required_checks"],
        study=payload["study"],
        null_sample_size=payload["null_sample_size"],
        uncertainty_level=payload["uncertainty_level"],
        null_draw=payload["null_draw"],
        alpha=payload.get("alpha"),
        horizon=payload.get("horizon"),
        false_alarm_tolerance=payload.get("false_alarm_tolerance"),
        coverage_tolerance=payload.get("coverage_tolerance"),
        repeated_look_stride=payload.get("repeated_look_stride"),
    )


def reference_case(
    case_id: str,
    statistic: str,
    expected: float,
    tolerance: float,
    source: str,
    payload: Mapping[str, object],
) -> ReferenceCase:
    """Build one reference case and canonicalize its payload."""

    if not isinstance(case_id, str) or _TOKEN.fullmatch(case_id) is None:
        raise ValidationError("case_id must be a token")
    if not isinstance(statistic, str) or _TOKEN.fullmatch(statistic) is None:
        raise ValidationError("statistic must be a token")
    if source not in _REFERENCE_SOURCES:
        raise ValidationError("reference source is unknown")
    _finite(expected, "expected")
    _nonnegative_finite(tolerance, "tolerance")
    return ReferenceCase(
        case_id=case_id,
        statistic=statistic,
        expected=float(expected),
        tolerance=float(tolerance),
        source=source,
        payload_json=_canonical_json(_json_ready(dict(payload))),
    )


def reference_case_from_dict(payload: object) -> ReferenceCase:
    if not isinstance(payload, dict):
        raise ValidationError("reference case must be an object")
    try:
        body = payload["payload"]
        if not isinstance(body, dict):
            raise ValidationError("reference payload must be an object")
        return reference_case(
            payload["case_id"],
            payload["statistic"],
            payload["expected"],
            payload["tolerance"],
            payload["source"],
            body,
        )
    except KeyError as error:
        raise ValidationError("reference case is missing a field") from error
    except TypeError as error:
        raise ValidationError("reference case has a bad field") from error


def aa_study_rows(capture: AACaptureResult) -> tuple[AAStudyRow, ...]:
    """Copy pair fields from an A/A capture.

    The capture's task ids remain available to the caller. Public validation
    summaries do not include them.
    """

    if not isinstance(capture, AACaptureResult):
        raise ValidationError("aa capture is required")
    rows: list[AAStudyRow] = []
    for record in capture.records:
        reference = record.pair.reference.evaluator_outcome
        candidate = record.pair.candidate.evaluator_outcome
        rows.append(
            AAStudyRow(
                scenario_id=record.schedule.scenario_id,
                task_id=record.schedule.task_id,
                repetition=record.schedule.repetition,
                reference_success=(
                    None if reference is None else float(bool(reference.success))
                ),
                candidate_success=(
                    None if candidate is None else float(bool(candidate.success))
                ),
                disagreement=record.evaluator_disagreement,
                reference_requirement_fraction=record.reference_requirement_fraction,
                candidate_requirement_fraction=record.candidate_requirement_fraction,
                trajectory_diverged=record.trajectory.first_divergent_step is not None,
                length_difference=record.trajectory.length_difference,
                concurrency=capture.concurrency,
            )
        )
    return tuple(rows)


def harm_detection_power(
    *,
    harm_margin: float,
    sample_size: int,
    alpha: float,
    variance: float,
) -> float:
    """One-sided normal power for a paired mean drop of ``harm_margin``.

    The null mean difference is 0 and the alternative mean is ``-harm_margin``.
    ``variance`` is the caller-supplied variance of one paired difference.
    The critical value is the standard normal quantile at ``1 - alpha``.
    This is a planning calculation, not an AppWorld measurement and not an
    estimate from a sample.
    """

    _open_probability(alpha, "alpha")
    _nonnegative_finite(harm_margin, "harm_margin")
    if not isinstance(variance, (int, float)) or isinstance(variance, bool):
        raise ValidationError("variance must be a positive float")
    if not math.isfinite(float(variance)) or float(variance) <= 0.0:
        raise ValidationError("variance must be a positive float")
    _positive_int(sample_size, "sample_size")
    z_alpha = _normal_quantile(1.0 - alpha)
    standard_error = math.sqrt(float(variance) / sample_size)
    return _normal_cdf(-z_alpha + harm_margin / standard_error)


def validate_method(
    method: MethodSpec,
    *,
    null_seeds: Sequence[int],
    reference_cases: Sequence[ReferenceCase],
    aa_observations: Sequence[MonitorObservation],
    split: str | None = None,
    aa_context: AAContext | None = None,
    aa_rows: Sequence[AAStudyRow] | None = None,
    aa_capture: AACaptureResult | None = None,
    aa_confidence_level: float | None = None,
    aa_resamples: int | None = None,
    aa_seed: int | None = None,
    kl_comparison: KLApproximationReport | None = None,
) -> ValidationReport:
    """Validate one method and return a reproducible report.

    Null false-alarm checks use ``null_seeds``. Coverage uses the same
    draws when the method has an interval. Stopping is recorded for
    sequential methods. Reference agreement compares this package's
    statistic with ``reference_cases``. A/A dependence uses either monitor
    observations with ``aa_context`` or paired rows from an A/A capture.

    Empty A/A inputs, synthetic provenance, and a GPU study without a
    hardware reading do not fill effect estimates. A method is
    benchmark-eligible only when every minimum check was requested and
    passed, including accepted A/A evidence for monitor methods.
    """

    if not isinstance(method, MethodSpec):
        raise ValidationError("method spec is required")
    contract = _lookup(method.name)
    seeds = _seeds(null_seeds)
    cases = _cases(reference_cases, contract)
    observations = _observations(aa_observations)
    resolved_split = _resolve_split(split, observations)
    rows, hashes, context = _resolve_aa(
        observations, aa_context, aa_rows, aa_capture
    )
    _reject_mixed_aa(observations, rows)
    _check_aa_bootstrap(aa_confidence_level, aa_resamples, aa_seed, rows, context)
    if (
        aa_resamples is not None
        and aa_resamples > STUDY_BUDGETS[method.study].max_resamples
    ):
        raise ValidationError(f"{method.study} resamples exceed the study budget")
    _check_kl_argument(contract, kl_comparison)
    _enforce_budget(method, len(seeds))
    if contract.kind == "score" and seeds:
        raise ValidationError("a score method does not take null seeds")
    replicates = (
        _simulate(method, seeds) if seeds and contract.kind != "score" else ()
    )
    included = tuple(item for item in replicates if not item.excluded)
    required = set(method.required_checks)
    if method.study == "gpu":
        required.add("gpu_evidence")
    checks: list[CheckResult] = []
    agreements: list[CaseAgreement] = []
    if "null_false_alarm" in contract.checks:
        checks.append(_null_check(method, contract, seeds, replicates, included, required))
    if "coverage" in contract.checks:
        checks.append(
            _coverage_check(method, contract, seeds, included, required)
        )
    if "stopping" in contract.checks:
        checks.append(_stopping_check(method, contract, seeds, included, required))
    if "repeated_look" in contract.checks:
        checks.append(_repeated_check(method, seeds, included, required))
    if "reference" in contract.checks:
        reference_check, agreements = _reference_check(method, cases, required)
        checks.append(reference_check)
    aa_report = _aa_report(
        method,
        contract,
        observations,
        context,
        rows,
        resolved_split,
        aa_confidence_level,
        aa_resamples,
        aa_seed,
        required,
    )
    if "aa_dependence" in contract.checks:
        checks.append(_aa_check(aa_report, method, required))
    if "kl_approximation" in contract.checks:
        checks.append(_kl_check(kl_comparison, required))
    if method.study == "gpu":
        checks.append(_gpu_check(context, rows, observations))
    omitted = tuple(
        name for name in contract.checks if name not in required
    )
    eligible = _eligible(omitted, required, checks)
    calibration = _calibration(contract, checks)
    report = ValidationReport(
        method=method.name,
        implemented=True,
        validated=eligible,
        benchmark_eligible=eligible,
        calibration=calibration,
        study=method.study,
        null_claim=contract.null_claim,
        null_draw=method.null_draw,
        input_hash="",
        seeds=seeds,
        sample_count=len(included),
        null_sample_size=method.null_sample_size,
        uncertainty_level=method.uncertainty_level,
        alpha=method.alpha,
        horizon=method.horizon,
        false_alarm_tolerance=method.false_alarm_tolerance,
        coverage_tolerance=method.coverage_tolerance,
        parameters=_canonical_parameters(method),
        required_checks=method.required_checks,
        omitted_checks=omitted,
        checks=tuple(checks),
        reference_agreements=tuple(agreements),
        aa=aa_report,
        libraries=reference_library_status(),
        configuration_hashes=hashes,
        gpu_evidence=_gpu_evidence(context),
        gpu_floor_measured=False,
        split=resolved_split,
    )
    digest = _hash_document(
        _validation_inputs(
            method,
            seeds,
            cases,
            observations,
            context,
            rows,
            resolved_split,
            aa_confidence_level,
            aa_resamples,
            aa_seed,
            kl_comparison,
        )
    )
    return ValidationReport(
        **{
            **{item.name: getattr(report, item.name) for item in fields(report)},
            "input_hash": digest,
        }
    )


def apply_validation_reports(
    reports: Sequence[ValidationReport],
) -> tuple[MethodValidationStatus, ...]:
    """Mark a catalog method validated only when a report is eligible."""

    grouped: dict[str, list[ValidationReport]] = defaultdict(list)
    for report in reports:
        if not isinstance(report, ValidationReport):
            raise ValidationError("validation report is required")
        if report.method not in _CONTRACTS:
            raise ValidationError("validation report names an unknown method")
        grouped[report.method].append(report)
    statuses: list[MethodValidationStatus] = []
    for contract in _CONTRACTS.values():
        items = grouped.get(contract.name, [])
        eligible = [item for item in items if item.benchmark_eligible]
        chosen = eligible[-1] if eligible else (items[-1] if items else None)
        statuses.append(
            MethodValidationStatus(
                name=contract.name,
                implemented=True,
                validated=bool(eligible),
                benchmark_eligible=bool(eligible),
                calibration="unavailable" if chosen is None else chosen.calibration,
                gpu_floor_measured=False,
                gpu_evidence=bool(chosen and chosen.gpu_evidence),
                aa_evidence_accepted=bool(
                    chosen and chosen.aa.evidence_accepted and chosen.aa.status == "passed"
                ),
            )
        )
    return tuple(statuses)


def compare_plan_kl(
    samples: Sequence[KLPositionSample],
    *,
    top_k: int,
    provenance: str,
    vocabulary_size: int | None = None,
    hardware_observed: bool = False,
) -> KLApproximationReport:
    """Compare top-k plan KL with full-vocabulary KL.

    At each position the retained indexes are the ``top_k`` largest
    production log-probabilities, ties breaking toward the smaller index.
    The candidate vector is restricted to those same indexes. Both
    restricted vectors are renormalized by ``truncated_next_token_kl``.
    The signed error is full mean KL minus truncated mean KL, in nats.

    Provenance ``gpu``, ``synthetic``, and ``unavailable`` do not score
    the arrays. This function does not call vLLM or Transformers.
    """

    _positive_int(top_k, "top_k")
    if provenance not in _PROVENANCE | frozenset({"supplied_sample"}):
        raise ValidationError("kl provenance is unknown")
    if not isinstance(hardware_observed, bool):
        raise ValidationError("hardware_observed must be a boolean")
    if vocabulary_size is not None:
        _positive_int(vocabulary_size, "vocabulary_size")
    prepared = _kl_samples(samples)
    digest = _hash_document(
        {
            "top_k": top_k,
            "provenance": provenance,
            "vocabulary_size": vocabulary_size,
            "hardware_observed": hardware_observed,
            "samples": [
                {
                    "production": sample.production_log_probabilities,
                    "candidate": sample.candidate_log_probabilities,
                }
                for sample in prepared
            ],
        }
    )
    empty = KLApproximationReport(
        status="unavailable",
        approximation="top_k",
        provenance=provenance,
        top_k=top_k,
        vocabulary_size=vocabulary_size,
        sample_count=len(prepared),
        position_count=sum(len(sample.production_log_probabilities) for sample in prepared),
        mean_signed_error=None,
        mean_absolute_error=None,
        max_absolute_error=None,
        sample_absolute_errors=(),
        input_hash=digest,
        gpu_floor_measured=False,
        reason="",
    )
    if provenance not in {"supplied_sample", "local_runtime"}:
        return KLApproximationReport(
            **{
                **{item.name: getattr(empty, item.name) for item in fields(empty)},
                "reason": "this runner does not measure a GPU truncation floor",
            }
        )
    if not prepared:
        return KLApproximationReport(
            **{
                **{item.name: getattr(empty, item.name) for item in fields(empty)},
                "reason": "no KL sample was supplied",
            }
        )
    signed: list[float] = []
    absolute: list[float] = []
    try:
        for sample in prepared:
            full, truncated = _score_sample(sample, top_k, vocabulary_size)
            error = full - truncated
            signed.append(error)
            absolute.append(abs(error))
    except _STAT_ERRORS as error:
        return KLApproximationReport(
            **{
                **{item.name: getattr(empty, item.name) for item in fields(empty)},
                "status": "failed",
                "reason": str(error),
            }
        )
    return KLApproximationReport(
        status="passed",
        approximation="top_k",
        provenance=provenance,
        top_k=top_k,
        vocabulary_size=vocabulary_size,
        sample_count=len(prepared),
        position_count=empty.position_count,
        mean_signed_error=fmean(signed),
        mean_absolute_error=fmean(absolute),
        max_absolute_error=max(absolute),
        sample_absolute_errors=tuple(absolute),
        input_hash=digest,
        gpu_floor_measured=False,
        reason="scored the supplied sample",
    )


def assess_harm_study(
    outcomes: Sequence[BaselineOutcome],
    *,
    split: str,
    study: str,
    harm_margin: float,
    sample_sizes: Sequence[int],
    alpha: float,
    variance: float,
    power_target: float,
    confidence_level: float | None = None,
    bootstrap_resamples: int | None = None,
    seed: int | None = None,
) -> HarmStudyReport:
    """Compare production with do-nothing and compute planning power.

    Task-mix slices use the supplied app and difficulty labels. A null app
    stays unlabeled. ``test_normal`` and ``test_challenge`` are rejected.
    Power is ``harm_detection_power`` at each supplied sample size. Feasible
    is true only when every size meets ``power_target`` and the clustered
    interval for production minus do-nothing lies above zero.
    """

    resolved = _development_split(split, required=True)
    if study not in STUDY_BUDGETS:
        raise ValidationError("study is unknown")
    _positive_finite(harm_margin, "harm_margin")
    _open_probability(alpha, "alpha")
    _open_probability(power_target, "power_target")
    if not isinstance(variance, (int, float)) or isinstance(variance, bool):
        raise ValidationError("variance must be a positive float")
    if float(variance) <= 0.0 or not math.isfinite(float(variance)):
        raise ValidationError("variance must be a positive float")
    sizes = _sample_sizes(sample_sizes)
    prepared = _baseline_outcomes(outcomes)
    if prepared and (
        confidence_level is None or bootstrap_resamples is None or seed is None
    ):
        raise ValidationError("outcome intervals need confidence_level, resamples, and seed")
    if bootstrap_resamples is not None:
        _positive_int(bootstrap_resamples, "bootstrap_resamples")
        if bootstrap_resamples > STUDY_BUDGETS[study].max_resamples:
            raise ValidationError(f"{study} resamples exceed the study budget")
    if confidence_level is not None:
        _open_probability(confidence_level, "confidence_level")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValidationError("seed must be an int")
    powers = tuple(
        PowerAtSize(
            sample_size=size,
            power=harm_detection_power(
                harm_margin=harm_margin,
                sample_size=size,
                alpha=alpha,
                variance=float(variance),
            ),
            meets_target=False,
        )
        for size in sizes
    )
    powers = tuple(
        PowerAtSize(item.sample_size, item.power, item.power + 1e-15 >= power_target)
        for item in powers
    )
    production = [item for item in prepared if item.role == "production"]
    do_nothing = [item for item in prepared if item.role == "do_nothing"]
    missing = sum(item.success is None for item in prepared)
    level = confidence_level if confidence_level is not None else 0.0
    resamples = bootstrap_resamples if bootstrap_resamples is not None else 0
    seed_value = seed if seed is not None else 0
    production_rate = _role_interval(production, level, resamples, seed_value)
    nothing_rate = _role_interval(do_nothing, level, resamples, seed_value + 1)
    difference = _role_difference(
        production, do_nothing, level, resamples, seed_value + 2
    )
    fraction_difference = _fraction_difference(production, do_nothing)
    slices = _task_mix(production, level, resamples, seed_value + 3)
    effect = _mix_effect(slices)
    empirical = (
        production_rate[0] is not None and nothing_rate[0] is not None and difference[0] is not None
    )
    beats = difference[1] > 0.0 if empirical and difference[1] is not None else None
    power_ok = all(item.meets_target for item in powers)
    if not power_ok:
        feasible: bool | None = False
        reason = "the harm margin is not detectable at every supplied sample size"
    elif not empirical:
        feasible = None
        if not prepared:
            reason = "evaluator outcomes were not supplied"
        elif production_rate[0] is None:
            reason = "production outcomes were not supplied"
        else:
            reason = "do-nothing outcomes were not supplied"
    elif not beats:
        feasible = False
        reason = "production does not beat the do-nothing agent"
    else:
        feasible = True
        reason = (
            "the supplied sample sizes detect the margin and production beats do-nothing"
        )
    scenarios = {item.scenario_id for item in prepared}
    report = HarmStudyReport(
        split=resolved if resolved is not None else split,
        study=study,
        harm_margin=float(harm_margin),
        alpha=alpha,
        variance=float(variance),
        power_target=power_target,
        input_hash="",
        powers=powers,
        production_success_rate=production_rate[0],
        do_nothing_success_rate=nothing_rate[0],
        production_low=production_rate[1],
        production_high=production_rate[2],
        do_nothing_low=nothing_rate[1],
        do_nothing_high=nothing_rate[2],
        success_difference=difference[0],
        difference_low=difference[1],
        difference_high=difference[2],
        production_beats_do_nothing=beats,
        requirement_fraction_difference=fraction_difference,
        task_mix_effect=effect,
        slices=slices,
        episode_count=len(prepared),
        scenario_count=len(scenarios),
        missing_success_count=missing,
        feasible=feasible,
        feasibility_reason=reason,
        empirical_evidence=empirical,
    )
    digest = _hash_document(
        {
            "split": report.split,
            "study": study,
            "harm_margin": harm_margin,
            "sample_sizes": list(sizes),
            "alpha": alpha,
            "variance": variance,
            "power_target": power_target,
            "confidence_level": confidence_level,
            "bootstrap_resamples": bootstrap_resamples,
            "seed": seed,
            "outcomes": [
                {
                    "role": item.role,
                    "success": item.success,
                    "requirement_fraction": item.requirement_fraction,
                    "scenario_id": item.scenario_id,
                    "app": item.app,
                    "difficulty": item.difficulty,
                    "pair_key": item.pair_key,
                }
                for item in prepared
            ],
        }
    )
    return HarmStudyReport(
        **{
            **{item.name: getattr(report, item.name) for item in fields(report)},
            "input_hash": digest,
        }
    )


def public_validation_summary(report: ValidationReport) -> dict[str, object]:
    """Return a public document with no task, scenario, or episode ids."""

    return _public_document(report)


def public_harm_summary(report: HarmStudyReport) -> dict[str, object]:
    return _public_document(report)


def public_kl_summary(report: KLApproximationReport) -> dict[str, object]:
    return _public_document(report)


def format_validation_summary(report: ValidationReport) -> str:
    lines = [
        f"method: {report.method}",
        f"implemented: {str(report.implemented).lower()}",
        f"validated: {str(report.validated).lower()}",
        f"benchmark_eligible: {str(report.benchmark_eligible).lower()}",
        f"calibration: {report.calibration}",
        f"study: {report.study}",
        f"input_hash: {report.input_hash}",
        f"sample_count: {report.sample_count}",
        f"gpu_floor_measured: {str(report.gpu_floor_measured).lower()}",
        f"gpu_evidence: {str(report.gpu_evidence).lower()}",
    ]
    if report.omitted_checks:
        lines.append("omitted_checks: " + ",".join(report.omitted_checks))
    for check in report.checks:
        lines.append(f"check.{check.name}: {check.status}")
    return "\n".join(lines) + "\n"


def format_harm_summary(report: HarmStudyReport) -> str:
    feasible = "undetermined" if report.feasible is None else str(report.feasible).lower()
    lines = [
        f"split: {report.split}",
        f"feasible: {feasible}",
        f"empirical_evidence: {str(report.empirical_evidence).lower()}",
        f"input_hash: {report.input_hash}",
        f"reason: {report.feasibility_reason}",
    ]
    for item in report.powers:
        lines.append(
            f"power.{item.sample_size}: {item.power:.6f} meets={str(item.meets_target).lower()}"
        )
    return "\n".join(lines) + "\n"


def format_kl_summary(report: KLApproximationReport) -> str:
    error = "unset" if report.mean_absolute_error is None else f"{report.mean_absolute_error:.6g}"
    return (
        f"status: {report.status}\n"
        f"approximation: {report.approximation}\n"
        f"top_k: {report.top_k}\n"
        f"vocabulary_size: {report.vocabulary_size}\n"
        f"sample_count: {report.sample_count}\n"
        f"mean_absolute_error: {error}\n"
        f"gpu_floor_measured: {str(report.gpu_floor_measured).lower()}\n"
        f"input_hash: {report.input_hash}\n"
    )


def format_method_status(statuses: Sequence[MethodValidationStatus]) -> str:
    lines = []
    for status in statuses:
        lines.append(
            f"{status.name}: implemented={str(status.implemented).lower()} "
            f"validated={str(status.validated).lower()}"
        )
    return "\n".join(lines) + "\n"


def default_results_root() -> Path:
    return Path(__file__).resolve().parents[3] / "results"


def write_public_report(
    document: Mapping[str, object],
    output_path: Path,
    *,
    results_root: Path,
) -> None:
    """Write a public summary. Refuses a path inside ``results_root``."""

    assert_public_payload(document)
    if _under(output_path, results_root):
        raise ValidationError("validation output cannot be written as a public result")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )


def main_validate(argv: Sequence[str] | None = None) -> int:
    """Validate one method from a JSON spec and write a public summary."""

    parser = argparse.ArgumentParser(description="Validate one statistical method.")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--observations")
    parser.add_argument("--aa-rows")
    parser.add_argument("--kl-sample")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--vocabulary-size", type=int)
    parser.add_argument("--kl-provenance", default="supplied_sample")
    parser.add_argument("--results-root", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        payload = _load_json(Path(args.spec))
        if not isinstance(payload, dict):
            raise ValidationError("method spec must be an object")
        seeds = payload.pop("seeds", None)
        raw_cases = payload.pop("reference_cases", [])
        split = payload.pop("split", None)
        if not isinstance(seeds, list):
            raise ValidationError("seeds must be a list")
        if not isinstance(raw_cases, list):
            raise ValidationError("reference_cases must be a list")
        method = method_spec_from_dict(payload)
        cases = tuple(reference_case_from_dict(item) for item in raw_cases)
        observations: tuple[MonitorObservation, ...] = ()
        if args.observations:
            observations = _load_observations(Path(args.observations))
        rows = None
        context = None
        if args.aa_rows:
            context, rows = _load_aa_rows(Path(args.aa_rows))
        comparison = None
        if args.kl_sample is not None:
            if args.top_k is None:
                raise ValidationError("top_k is required with a KL sample")
            comparison = compare_plan_kl(
                _load_kl_samples(Path(args.kl_sample)),
                top_k=args.top_k,
                provenance=args.kl_provenance,
                vocabulary_size=args.vocabulary_size,
            )
        report = validate_method(
            method,
            null_seeds=tuple(seeds),
            reference_cases=cases,
            aa_observations=observations,
            split=split if isinstance(split, str) else None,
            aa_context=context,
            aa_rows=rows,
            kl_comparison=comparison,
        )
        document = public_validation_summary(report)
        root = Path(args.results_root) if args.results_root else default_results_root()
        write_public_report(document, Path(args.output), results_root=root)
    except (ValidationError, RecordError) as error:
        print(str(error), file=sys.stderr, flush=True)
        return 1
    print(format_validation_summary(report), end="", flush=True)
    return 0


def main_compare_kl(argv: Sequence[str] | None = None) -> int:
    """Score a supplied KL sample and write a public summary."""

    parser = argparse.ArgumentParser(description="Compare top-k and full plan KL.")
    parser.add_argument("--sample", required=True)
    parser.add_argument("--top-k", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocabulary-size", type=int)
    parser.add_argument("--provenance", default="supplied_sample")
    parser.add_argument("--results-root", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        report = compare_plan_kl(
            _load_kl_samples(Path(args.sample)),
            top_k=args.top_k,
            provenance=args.provenance,
            vocabulary_size=args.vocabulary_size,
        )
        root = Path(args.results_root) if args.results_root else default_results_root()
        write_public_report(
            public_kl_summary(report),
            Path(args.output),
            results_root=root,
        )
    except ValidationError as error:
        print(str(error), file=sys.stderr, flush=True)
        return 1
    print(format_kl_summary(report), end="", flush=True)
    return 0


def main_harm(argv: Sequence[str] | None = None) -> int:
    """Assess harm-study feasibility from explicit inputs."""

    parser = argparse.ArgumentParser(description="Assess harm-study feasibility.")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--results-root", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        report = _harm_from_payload(_load_json(Path(args.spec)))
        root = Path(args.results_root) if args.results_root else default_results_root()
        write_public_report(
            public_harm_summary(report),
            Path(args.output),
            results_root=root,
        )
    except ValidationError as error:
        print(str(error), file=sys.stderr, flush=True)
        return 1
    print(format_harm_summary(report), end="", flush=True)
    return 0


def _lookup(name: object) -> _Contract:
    if not isinstance(name, str) or name not in _CONTRACTS:
        raise ValidationError("method is not implemented")
    return _CONTRACTS[name]


def _validate_spec(spec: MethodSpec) -> None:
    contract = _lookup(spec.name)
    if spec.study not in _STUDIES:
        raise ValidationError("study is unknown")
    if spec.null_draw not in contract.null_draws:
        raise ValidationError("null_draw is not valid for the method")
    _open_probability(spec.uncertainty_level, "uncertainty_level")
    if spec.alpha is not None:
        _open_probability(spec.alpha, "alpha")
    _optional_nonnegative(spec.false_alarm_tolerance, "false_alarm_tolerance")
    _optional_nonnegative(spec.coverage_tolerance, "coverage_tolerance")
    if spec.repeated_look_stride is not None:
        _positive_int(spec.repeated_look_stride, "repeated_look_stride")
    if contract.kind == "score":
        if spec.null_sample_size != 0:
            raise ValidationError("a score method has no null sample")
        if spec.horizon is not None or spec.alpha is not None:
            raise ValidationError("a score method has no horizon or alpha")
    elif contract.kind == "sequential":
        _positive_int(spec.null_sample_size, "null_sample_size")
        _positive_int(spec.horizon, "horizon")
        if spec.horizon != spec.null_sample_size:
            raise ValidationError("horizon must equal null_sample_size")
    else:
        _positive_int(spec.null_sample_size, "null_sample_size")
        if spec.horizon is not None:
            raise ValidationError("a batch method has no horizon")
    names = [item[0] for item in spec.parameters]
    if len(names) != len(set(names)):
        raise ValidationError("parameters repeat a name")
    present = spec.parameter_map()
    extra = [name for name in present if name not in contract.parameters]
    if extra:
        raise ValidationError("parameters include an unknown name")
    required = set(contract.always_parameters)
    required |= dict(contract.draw_parameters).get(spec.null_draw, frozenset())
    missing = sorted(required - set(present))
    if missing:
        raise ValidationError("parameters are missing " + missing[0])
    unused = set(present) - required
    if unused:
        raise ValidationError("parameters include " + sorted(unused)[0] + " for this draw")
    _check_parameter_values(spec, contract)
    checks = spec.required_checks
    if len(checks) != len(set(checks)):
        raise ValidationError("required_checks repeats a name")
    for name in checks:
        if name not in contract.checks:
            raise ValidationError("required check is not applicable")
    if contract.null_claim in {"level", "at_most"} and spec.alpha is None:
        raise ValidationError("alpha is required")
    if "alpha" in present and spec.alpha is not None:
        if abs(float(present["alpha"]) - spec.alpha) > 1e-12:
            raise ValidationError("alpha does not match the method parameter")
    if "confidence_level" in present:
        if spec.alpha is None or abs((1.0 - float(present["confidence_level"])) - spec.alpha) > 1e-12:
            raise ValidationError("alpha must equal one minus confidence_level")
    if "horizon_episodes" in present and present["horizon_episodes"] != spec.horizon:
        raise ValidationError("horizon_episodes must equal horizon")
    if spec.repeated_look_stride is not None and "repeated_look" not in contract.checks:
        raise ValidationError("repeated_look_stride is not applicable")
    if spec.repeated_look_stride is not None:
        _look_points(spec)


def _check_parameter_values(spec: MethodSpec, contract: _Contract) -> None:
    params = spec.parameter_map()
    if "categories" in params and int(params["categories"]) < 2:
        raise ValidationError("categories must be at least 2")
    if "trials" in params:
        _positive_int(params["trials"], "trials")
        if int(params["trials"]) != spec.null_sample_size:
            raise ValidationError("trials must equal null_sample_size")
    if "permutations" in params:
        _positive_int(params["permutations"], "permutations")
    if "steps" in params:
        _positive_int(params["steps"], "steps")
    if "dimension" in params:
        _positive_int(params["dimension"], "dimension")
    if "resamples" in params:
        _positive_int(params["resamples"], "resamples")
    if "cluster_size" in params:
        _positive_int(params["cluster_size"], "cluster_size")
        if spec.null_sample_size % int(params["cluster_size"]) != 0:
            raise ValidationError("null_sample_size must be a multiple of cluster_size")
    if "bandwidth" in params:
        _positive_finite(params["bandwidth"], "bandwidth")
    if "learning_rate" in params:
        _positive_finite(params["learning_rate"], "learning_rate")
    if "l2" in params:
        _nonnegative_finite(params["l2"], "l2")
    if "null_scale" in params:
        _nonnegative_finite(params["null_scale"], "null_scale")
    if "confidence_level" in params:
        _open_probability(params["confidence_level"], "confidence_level")
    if "harm_margin" in params:
        _nonnegative_finite(params["harm_margin"], "harm_margin")
    if "slack" in params:
        _nonnegative_finite(params["slack"], "slack")
    if "threshold" in params:
        _positive_finite(params["threshold"], "threshold")
    if "delta" in params:
        _open_probability(params["delta"], "delta")
    if "direction" in params:
        allowed = {"above", "below"} if contract.name == "betting_e_detector" else {
            "increase",
            "decrease",
            "two_sided",
        }
        if params["direction"] not in allowed:
            raise ValidationError("direction is unknown")
    if "pair_value" in params:
        _unit_float(params["pair_value"], "pair_value")
    if "null_probability" in params:
        _unit_float(params["null_probability"], "null_probability")
        if contract.name in {"bounded_mean_cs", "betting_e_detector"}:
            if abs(float(params["null_probability"]) - float(params["null_mean"])) > 1e-12:
                raise ValidationError("bernoulli null_probability must equal null_mean")
    if contract.name == "paired_difference_cs" and abs(float(params["null_mean"])) > 1e-12:
        raise ValidationError("paired difference null_mean must be zero")
    if contract.kind == "batch" and contract.name in _BATCH:
        minimum = 2 if contract.name in {
            "mmd_permutation_test",
            "classifier_two_sample_test",
            "paired_bootstrap",
            "clustered_paired_bootstrap",
        } else 1
        if spec.null_sample_size < minimum:
            raise ValidationError("null_sample_size is too small")
    if contract.kind == "sequential":
        try:
            _build_detector(spec)
        except _STAT_ERRORS as error:
            raise ValidationError(str(error)) from error
        _check_draw_bounds(spec)


def _check_draw_bounds(spec: MethodSpec) -> None:
    params = spec.parameter_map()
    if spec.name == "bounded_mean_cs" and spec.null_draw == "bernoulli":
        if float(params["lower"]) > 0.0 or float(params["upper"]) < 1.0:
            raise ValidationError("bernoulli draws must lie inside [lower, upper]")
    if spec.name == "paired_difference_cs" and spec.null_draw == "bernoulli":
        if float(params["lower"]) > -1.0 or float(params["upper"]) < 1.0:
            raise ValidationError("paired differences must lie inside [lower, upper]")
    if spec.name == "betting_e_detector" and spec.null_draw == "bernoulli":
        if float(params["lower"]) > 0.0 or float(params["upper"]) < 1.0:
            raise ValidationError("bernoulli draws must lie inside [lower, upper]")


def _enforce_budget(spec: MethodSpec, seed_count: int) -> None:
    budget = STUDY_BUDGETS[spec.study]
    if seed_count > budget.max_seeds:
        raise ValidationError(f"{spec.study} seed count exceeds the study budget")
    if spec.null_sample_size > budget.max_sample_size:
        raise ValidationError(f"{spec.study} sample size exceeds the study budget")
    if spec.horizon is not None and spec.horizon > budget.max_horizon:
        raise ValidationError(f"{spec.study} horizon exceeds the study budget")
    params = spec.parameter_map()
    permutations = int(params["permutations"]) if "permutations" in params else 1
    resamples = int(params["resamples"]) if "resamples" in params else 1
    if "permutations" in params and permutations > budget.max_permutations:
        raise ValidationError(f"{spec.study} permutations exceed the study budget")
    if "resamples" in params and resamples > budget.max_resamples:
        raise ValidationError(f"{spec.study} resamples exceed the study budget")
    looks = 1
    if spec.repeated_look_stride is not None:
        looks = len(_look_points(spec))
    length = spec.horizon if spec.horizon is not None else max(spec.null_sample_size, 1)
    work = seed_count * looks * max(permutations, resamples, 1) * length
    if work > budget.max_work:
        raise ValidationError(f"{spec.study} work exceeds the study budget")


def _look_points(spec: MethodSpec) -> tuple[int, ...]:
    stride = spec.repeated_look_stride
    if stride is None:
        raise ValidationError("repeated_look_stride is required")
    n = spec.null_sample_size
    minimum = _window_minimum(spec)
    if stride < 1 or stride >= n:
        raise ValidationError(
            "repeated_look_stride must be at least 1 and less than the sample"
        )
    points = list(range(max(stride, minimum), n + 1, stride))
    if n not in points and n >= minimum:
        points.append(n)
    unique = tuple(sorted(set(points)))
    if len(unique) < 2:
        raise ValidationError("repeated looks need two windows")
    return unique


def _window_minimum(spec: MethodSpec) -> int:
    if spec.name in {"mmd_permutation_test", "classifier_two_sample_test"}:
        return 2
    if spec.name == "clustered_paired_bootstrap":
        return int(spec.parameter_map()["cluster_size"])
    return 1


def _simulate(spec: MethodSpec, seeds: Sequence[int]) -> tuple[_Replicate, ...]:
    return tuple(_one_replicate(spec, seed) for seed in seeds)


def _one_replicate(spec: MethodSpec, seed: int) -> _Replicate:
    if spec.name == "ks_two_sample":
        return _simulate_ks(spec, seed)
    if spec.name in {"chi_square_homogeneity", "chi_square_goodness_of_fit"}:
        return _simulate_chi_square(spec, seed)
    if spec.name in {"mmd_permutation_test", "classifier_two_sample_test"}:
        return _simulate_embedding(spec, seed)
    if spec.name in {"paired_bootstrap", "clustered_paired_bootstrap"}:
        return _simulate_bootstrap(spec, seed)
    return _simulate_sequential(spec, seed)


def _simulate_ks(spec: MethodSpec, seed: int) -> _Replicate:
    rng = random.Random(seed)
    n = spec.null_sample_size
    left = [rng.random() for _ in range(n)]
    right = [rng.random() for _ in range(n)]
    single = ks_two_sample(left, right).p_value < _alpha(spec)
    repeated = _repeated_flag(
        spec,
        lambda size: ks_two_sample(left[:size], right[:size]).p_value < _alpha(spec),
    )
    return _batch_replicate(single, None, repeated)


def _simulate_chi_square(spec: MethodSpec, seed: int) -> _Replicate:
    rng = random.Random(seed)
    categories = int(spec.parameter_map()["categories"])
    trials = int(spec.parameter_map()["trials"])
    if spec.name == "chi_square_goodness_of_fit":
        draws = [rng.randrange(categories) for _ in range(trials)]

        def alarm_at(size: int) -> bool | None:
            counts = _count_categories(draws[:size], categories)
            expected = [size / categories] * categories
            try:
                result = chi_square_goodness_of_fit(counts, expected)
            except ChiSquareError as error:
                raise ValidationError(str(error)) from error
            return result.p_value < _alpha(spec)

        return _batch_replicate(bool(alarm_at(trials)), None, _repeated_flag(spec, alarm_at))
    accepted: tuple[list[int], list[int]] | None = None
    for _ in range(20):
        left = [rng.randrange(categories) for _ in range(trials)]
        right = [rng.randrange(categories) for _ in range(trials)]
        if _columns_positive(left, right, categories):
            accepted = (left, right)
            break
    if accepted is None:
        return _Replicate(None, None, None, None, None, None, None, True)
    left_draws, right_draws = accepted

    def alarm_at(size: int) -> bool | None:
        if not _columns_positive(left_draws[:size], right_draws[:size], categories):
            return None
        result = chi_square_homogeneity(
            _count_categories(left_draws[:size], categories),
            _count_categories(right_draws[:size], categories),
        )
        return result.p_value < _alpha(spec)

    full = alarm_at(trials)
    if full is None:
        return _Replicate(None, None, None, None, None, None, None, True)
    return _batch_replicate(full, None, _repeated_flag(spec, alarm_at))


def _simulate_embedding(spec: MethodSpec, seed: int) -> _Replicate:
    params = spec.parameter_map()
    rng = random.Random(seed)
    n = spec.null_sample_size
    dimension = int(params["dimension"])
    mean = float(params["null_mean"])
    scale = float(params["null_scale"])

    def point() -> tuple[float, ...]:
        return tuple(rng.gauss(mean, scale) for _ in range(dimension))

    production = [point() for _ in range(n)]
    candidate = [point() for _ in range(n)]

    def alarm_at(size: int) -> bool | None:
        if spec.name == "mmd_permutation_test":
            result = mmd_permutation_test(
                production[:size],
                candidate[:size],
                bandwidth=float(params["bandwidth"]),
                permutations=int(params["permutations"]),
                seed=seed,
            )
            return result.p_value < _alpha(spec)
        result = classifier_two_sample_test(
            production[:size],
            candidate[:size],
            permutations=int(params["permutations"]),
            seed=seed,
            steps=int(params["steps"]),
            learning_rate=float(params["learning_rate"]),
            l2=float(params["l2"]),
        )
        return result.p_value < _alpha(spec)

    single = bool(alarm_at(n))
    return _batch_replicate(single, None, _repeated_flag(spec, alarm_at))


def _simulate_bootstrap(spec: MethodSpec, seed: int) -> _Replicate:
    params = spec.parameter_map()
    rng = random.Random(seed)
    n = spec.null_sample_size
    mean = float(params["null_mean"])
    scale = float(params["null_scale"])
    candidate = [rng.gauss(mean, scale) for _ in range(n)]
    production = [rng.gauss(mean, scale) for _ in range(n)]
    labels = None
    if spec.name == "clustered_paired_bootstrap":
        cluster_size = int(params["cluster_size"])
        labels = tuple(str(index // cluster_size) for index in range(n))

    def alarm_at(size: int) -> bool | None:
        if spec.name == "clustered_paired_bootstrap":
            assert labels is not None
            result = clustered_paired_bootstrap(
                candidate[:size],
                production[:size],
                labels[:size],
                confidence_level=float(params["confidence_level"]),
                resamples=int(params["resamples"]),
                seed=seed,
            )
        else:
            result = paired_bootstrap(
                candidate[:size],
                production[:size],
                confidence_level=float(params["confidence_level"]),
                resamples=int(params["resamples"]),
                seed=seed,
            )
        return not (result.confidence_low <= 0.0 <= result.confidence_high)

    single = bool(alarm_at(n))
    covered = not single
    return _batch_replicate(single, covered, _repeated_flag(spec, alarm_at))


def _simulate_sequential(spec: MethodSpec, seed: int) -> _Replicate:
    rng = random.Random(seed)
    detector = _build_detector(spec)
    horizon = spec.horizon
    if horizon is None:
        raise ValidationError("horizon is required")
    any_alarm = False
    first: int | None = None
    first_evidence = None
    last = None
    for step in range(1, horizon + 1):
        last = detector.update(_null_observation(spec, rng))
        if last.alarm and not any_alarm:
            any_alarm = True
            first = step
            first_evidence = last
    if last is None:
        raise ValidationError("horizon is required")
    horizon_ok = None
    if spec.name == "sequential_canary":
        checked = first_evidence if first_evidence is not None else last
        flag = dict(checked.details)["stopped_at_horizon"]
        horizon_ok = flag == (0.0 if any_alarm else 1.0)
    covered = (not any_alarm) if "coverage" in _lookup(spec.name).checks else None
    return _Replicate(
        false_alarm=any_alarm,
        covered=covered,
        stopping_time=horizon if first is None else first,
        censored=not any_alarm,
        single_alarm=None,
        repeated_alarm=None,
        horizon_ok=horizon_ok,
        excluded=False,
    )


def _batch_replicate(
    single: bool,
    covered: bool | None,
    repeated: bool | None,
) -> _Replicate:
    return _Replicate(
        false_alarm=single,
        covered=covered,
        stopping_time=None,
        censored=None,
        single_alarm=single,
        repeated_alarm=repeated,
        horizon_ok=None,
        excluded=False,
    )


def _repeated_flag(
    spec: MethodSpec,
    alarm_at: Callable[[int], bool | None],
) -> bool | None:
    if spec.repeated_look_stride is None:
        return None
    repeated = False
    saw_full = False
    for size in _look_points(spec):
        alarm = alarm_at(size)
        if size == spec.null_sample_size:
            saw_full = alarm is not None
        if alarm:
            repeated = True
    if not saw_full:
        return None
    return repeated


def _null_observation(spec: MethodSpec, rng: random.Random) -> float | PairedSuccess:
    params = spec.parameter_map()
    if spec.name == "cusum":
        target = float(params["target"])
        if spec.null_draw == "constant":
            return target
        return rng.gauss(target, float(params["null_scale"]))
    if spec.name == "adwin":
        if spec.null_draw == "constant":
            return float(params["constant_value"])
        return rng.random()
    if spec.name == "bounded_mean_cs":
        if spec.null_draw == "constant":
            return float(params["null_mean"])
        return 1.0 if rng.random() < float(params["null_probability"]) else 0.0
    if spec.name == "betting_e_detector":
        if spec.null_draw == "constant":
            return float(params["null_mean"])
        return 1.0 if rng.random() < float(params["null_probability"]) else 0.0
    if spec.null_draw == "constant":
        value = float(params["pair_value"])
        return PairedSuccess(value, value)
    probability = float(params["null_probability"])
    return PairedSuccess(
        1.0 if rng.random() < probability else 0.0,
        1.0 if rng.random() < probability else 0.0,
    )


def _build_detector(spec: MethodSpec) -> object:
    params = spec.parameter_map()
    if spec.name == "cusum":
        return CUSUM(
            target=float(params["target"]),
            slack=float(params["slack"]),
            threshold=float(params["threshold"]),
            direction=str(params["direction"]),
        )
    if spec.name == "adwin":
        return ADWIN(delta=float(params["delta"]))
    if spec.name == "bounded_mean_cs":
        return BoundedMeanCS(
            alpha=float(params["alpha"]),
            lower=float(params["lower"]),
            upper=float(params["upper"]),
            null_mean=float(params["null_mean"]),
        )
    if spec.name == "paired_difference_cs":
        return PairedDifferenceCS(
            alpha=float(params["alpha"]),
            lower=float(params["lower"]),
            upper=float(params["upper"]),
            null_mean=float(params["null_mean"]),
        )
    if spec.name == "betting_e_detector":
        return BettingEDetector(
            null_mean=float(params["null_mean"]),
            alpha=float(params["alpha"]),
            lower=float(params["lower"]),
            upper=float(params["upper"]),
            direction=str(params["direction"]),
        )
    if spec.name == "harmful_shift":
        return HarmfulShiftTest(
            alpha=float(params["alpha"]),
            harm_margin=float(params["harm_margin"]),
        )
    if spec.name == "sequential_canary":
        return SequentialCanaryTest(
            alpha=float(params["alpha"]),
            harm_margin=float(params["harm_margin"]),
            horizon_episodes=int(params["horizon_episodes"]),
        )
    raise ValidationError("method has no detector")


def _null_check(
    spec: MethodSpec,
    contract: _Contract,
    seeds: Sequence[int],
    replicates: Sequence[_Replicate],
    included: Sequence[_Replicate],
    required: set[str],
) -> CheckResult:
    wanted = "null_false_alarm" in required
    if not seeds:
        return _empty_check(
            "null_false_alarm",
            wanted,
            "null seeds were not supplied",
            spec,
        )
    flags = [bool(item.false_alarm) for item in included if item.false_alarm is not None]
    excluded = sum(item.excluded for item in replicates)
    details = (("excluded", float(excluded)),)
    if contract.null_claim == "record":
        return _judged_rate(
            "null_false_alarm",
            flags,
            spec,
            seeds,
            details,
            nominal=None,
            rule="record",
        )
    if spec.false_alarm_tolerance is None or spec.alpha is None:
        return _empty_check(
            "null_false_alarm",
            wanted,
            "false_alarm_tolerance was not supplied",
            spec,
            details=details,
        )
    return _judged_rate(
        "null_false_alarm",
        flags,
        spec,
        seeds,
        details + (("nominal", spec.alpha),),
        nominal=spec.alpha,
        rule=contract.null_claim,
    )


def _coverage_check(
    spec: MethodSpec,
    contract: _Contract,
    seeds: Sequence[int],
    included: Sequence[_Replicate],
    required: set[str],
) -> CheckResult:
    wanted = "coverage" in required
    if not seeds:
        return _empty_check("coverage", wanted, "null seeds were not supplied", spec)
    if spec.coverage_tolerance is None:
        return _empty_check(
            "coverage",
            wanted,
            "coverage_tolerance was not supplied",
            spec,
        )
    nominal = _coverage_nominal(spec, contract)
    flags = [bool(item.covered) for item in included if item.covered is not None]
    return _judged_rate(
        "coverage",
        flags,
        spec,
        seeds,
        (("nominal", nominal),),
        nominal=nominal,
        rule="coverage",
    )


def _stopping_check(
    spec: MethodSpec,
    contract: _Contract,
    seeds: Sequence[int],
    included: Sequence[_Replicate],
    required: set[str],
) -> CheckResult:
    wanted = "stopping" in required
    if not seeds:
        return _empty_check("stopping", wanted, "null seeds were not supplied", spec)
    flags = [item.censored is False for item in included]
    times = [
        item.stopping_time
        for item in included
        if item.censored is False and item.stopping_time is not None
    ]
    details: list[tuple[str, float]] = []
    if times:
        details.append(("mean_stopping_time", fmean(times)))
    failures = sum(item.horizon_ok is False for item in included)
    details.append(("horizon_flag_failures", float(failures)))
    if contract.null_claim == "record":
        result = _judged_rate(
            "stopping",
            flags,
            spec,
            seeds,
            tuple(details),
            nominal=None,
            rule="record",
        )
        return result
    if spec.false_alarm_tolerance is None or spec.alpha is None:
        return _empty_check(
            "stopping",
            wanted,
            "false_alarm_tolerance was not supplied",
            spec,
        )
    result = _judged_rate(
        "stopping",
        flags,
        spec,
        seeds,
        tuple(details) + (("nominal", spec.alpha),),
        nominal=spec.alpha,
        rule="at_most",
    )
    if result.status == "passed" and failures:
        return CheckResult(
            name="stopping",
            status="failed",
            reason="horizon stopping flag did not match the alarm",
            sample_count=result.sample_count,
            seeds=result.seeds,
            estimate=result.estimate,
            interval_low=result.interval_low,
            interval_high=result.interval_high,
            uncertainty_level=result.uncertainty_level,
            details=result.details,
        )
    return result


def _repeated_check(
    spec: MethodSpec,
    seeds: Sequence[int],
    included: Sequence[_Replicate],
    required: set[str],
) -> CheckResult:
    wanted = "repeated_look" in required
    if spec.repeated_look_stride is None:
        return _empty_check(
            "repeated_look",
            wanted,
            "repeated_look_stride was not supplied",
            spec,
        )
    if not seeds:
        return _empty_check("repeated_look", wanted, "null seeds were not supplied", spec)
    single = [bool(item.single_alarm) for item in included if item.single_alarm is not None]
    repeated = [
        bool(item.repeated_alarm)
        for item in included
        if item.repeated_alarm is not None and item.single_alarm is not None
    ]
    if not single or len(repeated) != len(single):
        return _empty_check(
            "repeated_look",
            True,
            "repeated looks were not measured",
            spec,
        )
    single_rate = fmean(1.0 if item else 0.0 for item in single)
    repeated_rate = fmean(1.0 if item else 0.0 for item in repeated)
    low, high = _wilson(sum(repeated), len(repeated), spec.uncertainty_level)
    status = "passed" if repeated_rate + 1e-12 >= single_rate else "failed"
    reason = (
        "repeated-look rate is at least the single-look rate"
        if status == "passed"
        else "repeated-look rate fell below the single look"
    )
    return CheckResult(
        name="repeated_look",
        status=status,
        reason=reason,
        sample_count=len(repeated),
        seeds=tuple(seeds),
        estimate=repeated_rate,
        interval_low=low,
        interval_high=high,
        uncertainty_level=spec.uncertainty_level,
        details=(
            ("single_rate", single_rate),
            ("inflation", repeated_rate - single_rate),
            ("stride", float(spec.repeated_look_stride)),
        ),
    )


def _reference_check(
    spec: MethodSpec,
    cases: Sequence[ReferenceCase],
    required: set[str],
) -> tuple[CheckResult, list[CaseAgreement]]:
    wanted = "reference" in required
    if not cases:
        return (
            _empty_check("reference", wanted, "reference cases were not supplied", spec),
            [],
        )
    agreements: list[CaseAgreement] = []
    for case in cases:
        observed = _evaluate_reference(spec, case)
        error = abs(observed - case.expected)
        agreements.append(
            CaseAgreement(
                case_id=case.case_id,
                statistic=case.statistic,
                expected=case.expected,
                observed=observed,
                absolute_error=error,
                tolerance=case.tolerance,
                source=case.source,
                agreed=error <= case.tolerance + 1e-12,
            )
        )
    worst = max(item.absolute_error for item in agreements)
    agreed = all(item.agreed for item in agreements)
    return (
        CheckResult(
            name="reference",
            status="passed" if agreed else "failed",
            reason="reference values agreed" if agreed else "reference value disagreed",
            sample_count=len(agreements),
            seeds=(),
            estimate=worst,
            interval_low=None,
            interval_high=None,
            uncertainty_level=None,
            details=tuple(
                (item.case_id, item.absolute_error) for item in agreements
            ),
        ),
        agreements,
    )


def _evaluate_reference(spec: MethodSpec, case: ReferenceCase) -> float:
    contract = _lookup(spec.name)
    if case.statistic not in contract.statistics:
        raise ValidationError("statistic is not produced by the method")
    payload = case.payload()
    _check_payload_keys(spec, contract, payload)
    merged = spec.parameter_map()
    try:
        if spec.name == "ks_two_sample":
            result = ks_two_sample(_floats(payload, "left"), _floats(payload, "right"))
            return float(getattr(result, case.statistic))
        if spec.name == "chi_square_homogeneity":
            result = chi_square_homogeneity(_ints(payload, "left"), _ints(payload, "right"))
            return float(getattr(result, case.statistic))
        if spec.name == "chi_square_goodness_of_fit":
            result = chi_square_goodness_of_fit(
                _ints(payload, "observed"),
                _floats(payload, "expected"),
            )
            return float(getattr(result, case.statistic))
        if spec.name == "mmd_permutation_test":
            result = mmd_permutation_test(
                _matrix(payload, "production"),
                _matrix(payload, "candidate"),
                bandwidth=float(merged["bandwidth"]),
                permutations=int(merged["permutations"]),
                seed=_int_value(payload["seed"], "seed"),
                clusters=_optional_strings(payload, "clusters"),
            )
            return float(getattr(result, case.statistic))
        if spec.name == "classifier_two_sample_test":
            result = classifier_two_sample_test(
                _matrix(payload, "production"),
                _matrix(payload, "candidate"),
                permutations=int(merged["permutations"]),
                seed=_int_value(payload["seed"], "seed"),
                steps=int(merged["steps"]),
                learning_rate=float(merged["learning_rate"]),
                l2=float(merged["l2"]),
            )
            return float(getattr(result, case.statistic))
        if spec.name == "paired_bootstrap":
            result = paired_bootstrap(
                _floats(payload, "candidate"),
                _floats(payload, "production"),
                confidence_level=float(merged["confidence_level"]),
                resamples=int(merged["resamples"]),
                seed=_int_value(payload["seed"], "seed"),
            )
            return float(getattr(result, case.statistic))
        if spec.name == "clustered_paired_bootstrap":
            result = clustered_paired_bootstrap(
                _floats(payload, "candidate"),
                _floats(payload, "production"),
                _strings(payload, "clusters"),
                confidence_level=float(merged["confidence_level"]),
                resamples=int(merged["resamples"]),
                seed=_int_value(payload["seed"], "seed"),
            )
            return float(getattr(result, case.statistic))
        if spec.name in {"next_token_kl", "truncated_next_token_kl"}:
            production = _matrix(payload, "production")
            candidate = _matrix(payload, "candidate")
            if spec.name == "next_token_kl":
                scored = next_token_kl(production, candidate)
            else:
                vocabulary = payload.get("vocabulary_size")
                scored = truncated_next_token_kl(
                    production,
                    candidate,
                    vocabulary_size=(
                        None if vocabulary is None else _int_value(vocabulary, "vocabulary_size")
                    ),
                )
            return float(scored.mean_kl_nats)
        evidence = _reference_evidence(spec, payload)
        if case.statistic == "alarm":
            return 1.0 if evidence.alarm else 0.0
        if case.statistic == "ever_alarmed":
            return _ever_alarmed(spec, payload)
        if case.statistic == "estimate":
            return float(evidence.estimate)
        if case.statistic == "stopped_at_horizon":
            return float(dict(evidence.details)["stopped_at_horizon"])
    except _STAT_ERRORS as error:
        raise ValidationError(str(error)) from error
    raise ValidationError("statistic is not produced by the method")


def _reference_evidence(spec: MethodSpec, payload: Mapping[str, object]) -> object:
    detector = _build_detector(spec)
    if spec.name in _PAIRED_DETECTORS:
        observations = _pair_values(payload)
        evidence = None
        for observation in observations:
            evidence = detector.update(observation)
    else:
        evidence = None
        for value in _floats(payload, "observations"):
            evidence = detector.update(value)
    if evidence is None:
        raise ValidationError("reference observations are empty")
    return evidence


def _ever_alarmed(spec: MethodSpec, payload: Mapping[str, object]) -> float:
    detector = _build_detector(spec)
    alarmed = False
    if spec.name in _PAIRED_DETECTORS:
        series: Sequence[object] = _pair_values(payload)
    else:
        series = _floats(payload, "observations")
    for observation in series:
        evidence = detector.update(observation)
        alarmed = alarmed or bool(evidence.alarm)
    return 1.0 if alarmed else 0.0


def _aa_report(
    spec: MethodSpec,
    contract: _Contract,
    observations: Sequence[MonitorObservation],
    context: AAContext | None,
    rows: Sequence[AAStudyRow] | None,
    split: str | None,
    confidence_level: float | None,
    resamples: int | None,
    seed: int | None,
    required: set[str],
) -> AADependenceReport:
    del split, required
    provenance = "unavailable" if context is None else context.provenance
    hardware = bool(context and context.hardware_observed)
    memory = context.memory_used_mib if context and hardware else None
    wall = context.wall_seconds if context and hardware else None
    pair_rows = () if rows is None else tuple(rows)
    accepted = _evidence_accepted(provenance, hardware, memory) and (
        bool(pair_rows) or bool(observations)
    )
    if len({item.run.configuration_hash for item in observations}) > 1:
        accepted = False
        reason = "observations mix configuration hashes"
        return _blank_aa(
            provenance,
            hardware,
            len(observations),
            len(pair_rows),
            memory,
            wall,
            reason,
            accepted=False,
        )
    if not accepted:
        reason = _aa_reject_reason(provenance, hardware, memory, observations, pair_rows)
        return _blank_aa(
            provenance,
            hardware,
            len(observations),
            len(pair_rows),
            memory,
            wall,
            reason,
            accepted=False,
        )
    if pair_rows:
        return _aa_from_rows(
            spec,
            pair_rows,
            provenance,
            hardware,
            memory,
            wall,
            confidence_level,
            resamples,
            seed,
        )
    return _aa_from_observations(
        spec,
        observations,
        context,
        provenance,
        hardware,
        memory,
        wall,
    )


def _aa_from_rows(
    spec: MethodSpec,
    rows: Sequence[AAStudyRow],
    provenance: str,
    hardware: bool,
    memory: int | None,
    wall: float | None,
    confidence_level: float | None,
    resamples: int | None,
    seed: int | None,
) -> AADependenceReport:
    successes = [
        (row.task_id, row.scenario_id, row.reference_success)
        for row in rows
        if row.reference_success is not None
    ]
    repeated = _repetition_effect(
        [value for _task, _scenario, value in successes],
        [task for task, _scenario, _value in successes],
    )
    clustered_values = [
        (scenario, value)
        for _task, scenario, value in successes
        if scenario is not None
    ]
    clustering = _icc(
        [value for _scenario, value in clustered_values],
        [scenario for scenario, _value in clustered_values],
    )
    disagreements = [row.disagreement for row in rows if row.disagreement is not None]
    inference = fmean(1.0 if item else 0.0 for item in disagreements) if disagreements else None
    low = high = None
    if disagreements:
        low, high = _wilson(
            sum(1 for item in disagreements if item),
            len(disagreements),
            spec.uncertainty_level,
        )
    trajectory = fmean(1.0 if row.trajectory_diverged else 0.0 for row in rows)
    ratio = _width_ratio(rows, confidence_level, resamples, seed)
    concurrency_effect, levels = _concurrency_effect(rows)
    missing = []
    if repeated is None:
        missing.append("repeated tasks")
    if clustering is None:
        missing.append("scenario clustering")
    if inference is None:
        missing.append("evaluator disagreement")
    status = "passed" if not missing else "unavailable"
    reason = (
        "measured repeated tasks, scenario clustering, and evaluator disagreement"
        if status == "passed"
        else "missing " + ", ".join(missing)
    )
    return AADependenceReport(
        status=status,
        evidence_accepted=True,
        provenance=provenance,
        hardware_observed=hardware,
        observation_count=0,
        pair_count=len(rows),
        repeated_task_effect=repeated,
        scenario_clustering_effect=clustering,
        inference_variation=inference,
        inference_source="evaluator_disagreement" if inference is not None else None,
        inference_low=low,
        inference_high=high,
        trajectory_divergence_rate=trajectory,
        interval_width_ratio=ratio,
        concurrency_effect=concurrency_effect,
        concurrency_levels=levels,
        series_alarm=None,
        memory_used_mib=memory,
        wall_seconds=wall,
        reason=reason,
    )


def _aa_from_observations(
    spec: MethodSpec,
    observations: Sequence[MonitorObservation],
    context: AAContext | None,
    provenance: str,
    hardware: bool,
    memory: int | None,
    wall: float | None,
) -> AADependenceReport:
    if context is None or context.task_ids is None or context.scenario_ids is None:
        return _blank_aa(
            provenance,
            hardware,
            len(observations),
            0,
            memory,
            wall,
            "scenario and task labels were not supplied",
            accepted=True,
        )
    if context.repetitions is None:
        return _blank_aa(
            provenance,
            hardware,
            len(observations),
            0,
            memory,
            wall,
            "repetition indexes were not supplied",
            accepted=True,
        )
    ordered = sorted(
        zip(
            observations,
            context.task_ids,
            context.scenario_ids,
            context.repetitions,
            strict=True,
        ),
        key=lambda item: (item[0].observed_at, item[0].episode.episode_id),
    )
    values = [item[0].value for item in ordered]
    tasks = [item[1] for item in ordered]
    scenarios = [item[2] for item in ordered]
    repeated = _repetition_effect(values, tasks)
    present = [
        (scenario, value)
        for scenario, value in zip(scenarios, values, strict=True)
        if scenario is not None
    ]
    clustering = _icc(
        [value for _scenario, value in present],
        [str(scenario) for scenario, _value in present],
    )
    inference = _within_spread(values, tasks)
    series_alarm = _series_alarm(spec, [item[0] for item in ordered])
    missing = []
    if repeated is None:
        missing.append("repeated tasks")
    if clustering is None:
        missing.append("scenario clustering")
    if inference is None:
        missing.append("within-task repetition")
    status = "passed" if not missing else "unavailable"
    reason = (
        "measured repeated tasks, scenario clustering, and within-task repetition"
        if status == "passed"
        else "missing " + ", ".join(missing)
    )
    return AADependenceReport(
        status=status,
        evidence_accepted=True,
        provenance=provenance,
        hardware_observed=hardware,
        observation_count=len(observations),
        pair_count=0,
        repeated_task_effect=repeated,
        scenario_clustering_effect=clustering,
        inference_variation=inference,
        inference_source="within_task_repetition" if inference is not None else None,
        inference_low=None,
        inference_high=None,
        trajectory_divergence_rate=None,
        interval_width_ratio=None,
        concurrency_effect=None,
        concurrency_levels=(),
        series_alarm=series_alarm,
        memory_used_mib=memory,
        wall_seconds=wall,
        reason=reason,
    )


def _series_alarm(
    spec: MethodSpec,
    observations: Sequence[MonitorObservation],
) -> bool | None:
    if spec.name not in {
        "cusum",
        "adwin",
        "bounded_mean_cs",
        "betting_e_detector",
    }:
        return None
    detector = _build_detector(spec)
    alarmed = False
    try:
        for observation in observations:
            evidence = detector.update(observation.value)
            alarmed = alarmed or bool(evidence.alarm)
    except _STAT_ERRORS as error:
        raise ValidationError(str(error)) from error
    return alarmed


def _aa_check(
    report: AADependenceReport,
    spec: MethodSpec,
    required: set[str],
) -> CheckResult:
    wanted = "aa_dependence" in required
    if not wanted and not report.evidence_accepted:
        status = "not_applicable"
    elif wanted and report.status != "passed":
        status = "failed" if report.status == "failed" else "unavailable"
    else:
        status = report.status
    return CheckResult(
        name="aa_dependence",
        status=status,
        reason=report.reason,
        sample_count=report.pair_count or report.observation_count,
        seeds=(),
        estimate=report.inference_variation if report.evidence_accepted else None,
        interval_low=report.inference_low if report.evidence_accepted else None,
        interval_high=report.inference_high if report.evidence_accepted else None,
        uncertainty_level=spec.uncertainty_level if report.inference_low is not None else None,
        details=_aa_details(report) if report.evidence_accepted else (),
    )


def _aa_details(report: AADependenceReport) -> tuple[tuple[str, float], ...]:
    pairs = (
        ("repeated_task_effect", report.repeated_task_effect),
        ("scenario_clustering_effect", report.scenario_clustering_effect),
        ("trajectory_divergence_rate", report.trajectory_divergence_rate),
        ("interval_width_ratio", report.interval_width_ratio),
        ("concurrency_effect", report.concurrency_effect),
    )
    return tuple((name, value) for name, value in pairs if value is not None)


def _kl_check(
    comparison: KLApproximationReport | None,
    required: set[str],
) -> CheckResult:
    wanted = "kl_approximation" in required
    if comparison is None:
        return CheckResult(
            name="kl_approximation",
            status="unavailable" if wanted else "not_applicable",
            reason="KL approximation sample was not supplied",
            sample_count=0,
            seeds=(),
            estimate=None,
            interval_low=None,
            interval_high=None,
            uncertainty_level=None,
            details=(),
        )
    return CheckResult(
        name="kl_approximation",
        status=comparison.status,
        reason=comparison.reason,
        sample_count=comparison.sample_count,
        seeds=(),
        estimate=comparison.mean_absolute_error,
        interval_low=None,
        interval_high=None,
        uncertainty_level=None,
        details=(
            (("top_k", float(comparison.top_k)),)
            if comparison.top_k is not None
            else ()
        ),
    )


def _gpu_check(
    context: AAContext | None,
    rows: Sequence[AAStudyRow] | None,
    observations: Sequence[MonitorObservation],
) -> CheckResult:
    accepted = _gpu_evidence(context) and (bool(rows) or bool(observations))
    return CheckResult(
        name="gpu_evidence",
        status="passed" if accepted else "unavailable",
        reason=(
            "caller supplied a GPU hardware reading"
            if accepted
            else "gpu study has no hardware reading"
        ),
        sample_count=(0 if rows is None else len(rows)) + len(observations),
        seeds=(),
        estimate=None if context is None or not accepted else (
            None if context.memory_used_mib is None else float(context.memory_used_mib)
        ),
        interval_low=None,
        interval_high=None,
        uncertainty_level=None,
        details=(),
    )


def _eligible(
    omitted: Sequence[str],
    required: set[str],
    checks: Sequence[CheckResult],
) -> bool:
    if omitted:
        return False
    by_name = {check.name: check for check in checks}
    for name in required:
        check = by_name.get(name)
        if check is None or check.status != "passed":
            return False
    return all(check.status != "failed" for check in checks)


def _calibration(contract: _Contract, checks: Sequence[CheckResult]) -> str:
    by_name = {check.name: check for check in checks}
    if contract.null_claim == "none":
        reference = by_name.get("reference")
        if reference is not None and reference.status == "passed":
            return "measured"
        return "unavailable"
    null = by_name.get("null_false_alarm")
    if null is None or null.status != "passed":
        return "unavailable"
    if contract.null_claim == "record":
        return "measured"
    return "calibrated"


def _coverage_nominal(spec: MethodSpec, contract: _Contract) -> float:
    if "confidence_level" in spec.parameter_map():
        return float(spec.parameter_map()["confidence_level"])
    if spec.alpha is None:
        raise ValidationError("alpha is required")
    del contract
    return 1.0 - spec.alpha


def _judged_rate(
    name: str,
    flags: Sequence[bool],
    spec: MethodSpec,
    seeds: Sequence[int],
    details: tuple[tuple[str, float], ...],
    *,
    nominal: float | None,
    rule: str,
) -> CheckResult:
    if not flags:
        return CheckResult(
            name=name,
            status="unavailable",
            reason="no null replicate was usable",
            sample_count=0,
            seeds=tuple(seeds),
            estimate=None,
            interval_low=None,
            interval_high=None,
            uncertainty_level=spec.uncertainty_level,
            details=details,
        )
    rate, low, high = _rate_interval(flags, spec.uncertainty_level)
    if rule == "record":
        status = "passed"
        reason = "null behavior was measured"
    elif rule == "coverage":
        assert nominal is not None and spec.coverage_tolerance is not None
        status = (
            "passed"
            if rate + 1e-12 >= nominal - spec.coverage_tolerance
            else "failed"
        )
        reason = (
            "coverage met the supplied tolerance"
            if status == "passed"
            else "coverage missed the supplied tolerance"
        )
    elif rule == "at_most":
        assert nominal is not None and spec.false_alarm_tolerance is not None
        status = (
            "passed"
            if rate <= nominal + spec.false_alarm_tolerance + 1e-12
            else "failed"
        )
        reason = (
            "false-alarm rate was within the supplied bound"
            if status == "passed"
            else "false-alarm rate exceeded the supplied bound"
        )
    else:
        assert nominal is not None and spec.false_alarm_tolerance is not None
        status = (
            "passed"
            if abs(rate - nominal) <= spec.false_alarm_tolerance + 1e-12
            else "failed"
        )
        reason = (
            "false-alarm rate was within the supplied tolerance"
            if status == "passed"
            else "false-alarm rate missed the supplied tolerance"
        )
    return CheckResult(
        name=name,
        status=status,
        reason=reason,
        sample_count=len(flags),
        seeds=tuple(seeds),
        estimate=rate,
        interval_low=low,
        interval_high=high,
        uncertainty_level=spec.uncertainty_level,
        details=details,
    )


def _empty_check(
    name: str,
    wanted: bool,
    reason: str,
    spec: MethodSpec,
    details: tuple[tuple[str, float], ...] = (),
) -> CheckResult:
    return CheckResult(
        name=name,
        status="unavailable" if wanted else "not_applicable",
        reason=reason,
        sample_count=0,
        seeds=(),
        estimate=None,
        interval_low=None,
        interval_high=None,
        uncertainty_level=None,
        details=details,
    )


def _blank_aa(
    provenance: str,
    hardware: bool,
    observations: int,
    pairs: int,
    memory: int | None,
    wall: float | None,
    reason: str,
    *,
    accepted: bool,
) -> AADependenceReport:
    return AADependenceReport(
        status="unavailable",
        evidence_accepted=accepted,
        provenance=provenance,
        hardware_observed=hardware,
        observation_count=observations,
        pair_count=pairs,
        repeated_task_effect=None,
        scenario_clustering_effect=None,
        inference_variation=None,
        inference_source=None,
        inference_low=None,
        inference_high=None,
        trajectory_divergence_rate=None,
        interval_width_ratio=None,
        concurrency_effect=None,
        concurrency_levels=(),
        series_alarm=None,
        memory_used_mib=memory if accepted else None,
        wall_seconds=wall if accepted else None,
        reason=reason,
    )


def _evidence_accepted(provenance: str, hardware: bool, memory: int | None) -> bool:
    if provenance == "local_runtime":
        return True
    if provenance == "gpu":
        return hardware and memory is not None
    return False


def _gpu_evidence(context: AAContext | None) -> bool:
    if context is None:
        return False
    return _evidence_accepted(
        "gpu" if context.provenance == "gpu" else "unavailable",
        context.hardware_observed,
        context.memory_used_mib,
    )


def _aa_reject_reason(
    provenance: str,
    hardware: bool,
    memory: int | None,
    observations: Sequence[MonitorObservation],
    rows: Sequence[AAStudyRow],
) -> str:
    if not observations and not rows:
        return "A/A records were not supplied"
    if provenance in {"synthetic", "unavailable"}:
        return "synthetic or unavailable records are not a noise floor"
    if provenance == "gpu" and not hardware:
        return "gpu provenance has no hardware reading"
    if provenance == "gpu" and memory is None:
        return "gpu provenance has no memory reading"
    return "A/A evidence was not accepted"


def _repetition_effect(values: Sequence[float], task_ids: Sequence[str]) -> float | None:
    if len(values) != len(task_ids):
        raise ValidationError("task labels must align with values")
    groups: dict[str, list[float]] = defaultdict(list)
    for task_id, value in zip(task_ids, values, strict=True):
        groups[task_id].append(value)
    repeated = [group for group in groups.values() if len(group) >= 2]
    if not repeated:
        return None
    flat = [value for group in repeated for value in group]
    total = _population_variance(flat)
    within = fmean(_population_variance(group) for group in repeated)
    if total == 0.0:
        return 0.0
    return within / total


def _within_spread(values: Sequence[float], task_ids: Sequence[str]) -> float | None:
    groups: dict[str, list[float]] = defaultdict(list)
    for task_id, value in zip(task_ids, values, strict=True):
        groups[task_id].append(value)
    repeated = [group for group in groups.values() if len(group) >= 2]
    if not repeated:
        return None
    spreads = []
    for group in repeated:
        center = fmean(group)
        spreads.append(fmean(abs(value - center) for value in group))
    return fmean(spreads)


def _icc(values: Sequence[float], labels: Sequence[str]) -> float | None:
    """One-way random-effects ICC for caller-supplied cluster labels.

    Labels are scenario ids. The effect is ``(MSB - MSW) / (MSB + (n0 - 1) MSW)``.
    Fewer than two clusters, or one observation per cluster, is not an effect.
    A zero denominator on constant data is recorded as zero.
    """

    if len(values) != len(labels) or len(values) < 2:
        return None
    groups: dict[str, list[float]] = defaultdict(list)
    for label, value in zip(labels, values, strict=True):
        groups[label].append(value)
    if len(groups) < 2:
        return None
    n = len(values)
    k = len(groups)
    if n == k:
        return None
    overall = fmean(values)
    sizes = [len(group) for group in groups.values()]
    ssb = math.fsum(len(group) * (fmean(group) - overall) ** 2 for group in groups.values())
    ssw = math.fsum(
        math.fsum((value - fmean(group)) ** 2 for value in group)
        for group in groups.values()
    )
    msb = ssb / (k - 1)
    msw = ssw / (n - k)
    n0 = (n - math.fsum(size * size for size in sizes) / n) / (k - 1)
    if n0 <= 0.0:
        return None
    denominator = msb + (n0 - 1.0) * msw
    if denominator == 0.0:
        return 0.0
    return (msb - msw) / denominator


def _population_variance(values: Sequence[float]) -> float:
    center = fmean(values)
    return math.fsum((value - center) ** 2 for value in values) / len(values)


def _width_ratio(
    rows: Sequence[AAStudyRow],
    confidence_level: float | None,
    resamples: int | None,
    seed: int | None,
) -> float | None:
    if confidence_level is None or resamples is None or seed is None:
        return None
    usable = [
        row
        for row in rows
        if row.reference_success is not None
        and row.candidate_success is not None
        and row.scenario_id is not None
    ]
    if len(usable) < 2 or len({row.scenario_id for row in usable}) < 2:
        return None
    candidate = [float(row.candidate_success) for row in usable]
    reference = [float(row.reference_success) for row in usable]
    clusters = [str(row.scenario_id) for row in usable]
    iid = paired_bootstrap(
        candidate,
        reference,
        confidence_level=confidence_level,
        resamples=resamples,
        seed=seed,
    )
    clustered = clustered_paired_bootstrap(
        candidate,
        reference,
        clusters,
        confidence_level=confidence_level,
        resamples=resamples,
        seed=seed,
    )
    iid_width = iid.confidence_high - iid.confidence_low
    clustered_width = clustered.confidence_high - clustered.confidence_low
    if iid_width == 0.0 and clustered_width == 0.0:
        return 1.0
    if iid_width == 0.0:
        return None
    return clustered_width / iid_width


def _concurrency_effect(
    rows: Sequence[AAStudyRow],
) -> tuple[float | None, tuple[int, ...]]:
    levels = tuple(sorted({row.concurrency for row in rows}))
    grouped: dict[int, list[bool]] = defaultdict(list)
    for row in rows:
        if row.disagreement is not None:
            grouped[row.concurrency].append(row.disagreement)
    if len(grouped) < 2:
        return None, levels
    rates = [sum(flags) / len(flags) for flags in grouped.values() if flags]
    if len(rates) < 2:
        return None, levels
    return max(rates) - min(rates), levels


def _score_sample(
    sample: KLPositionSample,
    top_k: int,
    vocabulary_size: int | None,
) -> tuple[float, float]:
    restricted_production = []
    restricted_candidate = []
    for production, candidate in zip(
        sample.production_log_probabilities,
        sample.candidate_log_probabilities,
        strict=True,
    ):
        if len(production) < top_k:
            raise ValidationError("top_k exceeds the supplied support")
        if vocabulary_size is not None and len(production) > vocabulary_size:
            raise ValidationError("vocabulary_size is smaller than the supplied support")
        indexes = _top_indexes(production, top_k)
        restricted_production.append(tuple(production[index] for index in indexes))
        restricted_candidate.append(tuple(candidate[index] for index in indexes))
    full = score_full(
        sample.production_log_probabilities,
        sample.candidate_log_probabilities,
    )
    truncated = truncated_next_token_kl(
        restricted_production,
        restricted_candidate,
        vocabulary_size=vocabulary_size,
    )
    if truncated.approximation != "top_k":
        raise ValidationError("truncated score is not labeled top_k")
    return full.mean_kl_nats, truncated.mean_kl_nats


def _top_indexes(log_probabilities: Sequence[float], top_k: int) -> tuple[int, ...]:
    ranked = sorted(
        range(len(log_probabilities)),
        key=lambda index: (-log_probabilities[index], index),
    )
    return tuple(ranked[:top_k])


def _kl_samples(samples: Sequence[KLPositionSample]) -> tuple[KLPositionSample, ...]:
    prepared: list[KLPositionSample] = []
    for sample in samples:
        if not isinstance(sample, KLPositionSample):
            raise ValidationError("KL sample is required")
        production = tuple(tuple(row) for row in sample.production_log_probabilities)
        candidate = tuple(tuple(row) for row in sample.candidate_log_probabilities)
        if len(production) != len(candidate):
            raise ValidationError("production and candidate position counts differ")
        for left, right in zip(production, candidate, strict=True):
            if len(left) != len(right) or not left:
                raise ValidationError("log-probability positions must align")
        prepared.append(
            KLPositionSample(
                production_log_probabilities=production,
                candidate_log_probabilities=candidate,
            )
        )
    return tuple(prepared)


def _role_interval(
    outcomes: Sequence[BaselineOutcome],
    confidence_level: float,
    resamples: int,
    seed: int,
) -> tuple[float | None, float | None, float | None]:
    values = [1.0 if item.success else 0.0 for item in outcomes if item.success is not None]
    clusters = [item.scenario_id for item in outcomes if item.success is not None]
    if not values or resamples < 1:
        return (None, None, None) if not values else (fmean(values), None, None)
    if confidence_level <= 0.0:
        return fmean(values), None, None
    result = clustered_paired_bootstrap(
        values,
        [0.0] * len(values),
        clusters,
        confidence_level=confidence_level,
        resamples=resamples,
        seed=seed,
    )
    return result.mean_difference, result.confidence_low, result.confidence_high


def _role_difference(
    production: Sequence[BaselineOutcome],
    do_nothing: Sequence[BaselineOutcome],
    confidence_level: float,
    resamples: int,
    seed: int,
) -> tuple[float | None, float | None, float | None]:
    left = [item for item in production if item.success is not None]
    right = [item for item in do_nothing if item.success is not None]
    if not left or not right or resamples < 1 or confidence_level <= 0.0:
        return None, None, None
    paired = _aligned_pairs(left, right)
    if paired is not None:
        candidate, baseline, clusters = paired
        result = clustered_paired_bootstrap(
            candidate,
            baseline,
            clusters,
            confidence_level=confidence_level,
            resamples=resamples,
            seed=seed,
        )
        return result.mean_difference, result.confidence_low, result.confidence_high
    return _unpaired_difference(left, right, confidence_level, resamples, seed)


def _aligned_pairs(
    production: Sequence[BaselineOutcome],
    do_nothing: Sequence[BaselineOutcome],
) -> tuple[list[float], list[float], list[str]] | None:
    if any(item.pair_key is None for item in (*production, *do_nothing)):
        return None
    left = {item.pair_key: item for item in production}
    right = {item.pair_key: item for item in do_nothing}
    if set(left) != set(right) or len(left) != len(production) or len(right) != len(do_nothing):
        raise ValidationError("pair keys must match one to one")
    keys = sorted(str(key) for key in left)
    for key in keys:
        if left[key].scenario_id != right[key].scenario_id:
            raise ValidationError("a paired outcome must share a scenario")
    return (
        [1.0 if left[key].success else 0.0 for key in keys],
        [1.0 if right[key].success else 0.0 for key in keys],
        [left[key].scenario_id for key in keys],
    )


def _unpaired_difference(
    production: Sequence[BaselineOutcome],
    do_nothing: Sequence[BaselineOutcome],
    confidence_level: float,
    resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    """Cluster-resample the difference of two unpaired success means.

    Each arm is resampled by scenario with ``cluster_draw_indexes``. The
    percentile interval uses the same linear rank as ``paired_bootstrap``.
    """

    prod_values = [1.0 if item.success else 0.0 for item in production]
    none_values = [1.0 if item.success else 0.0 for item in do_nothing]
    prod_clusters = [item.scenario_id for item in production]
    none_clusters = [item.scenario_id for item in do_nothing]
    point = fmean(prod_values) - fmean(none_values)
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        prod_indexes = cluster_draw_indexes(prod_clusters, rng)
        none_indexes = cluster_draw_indexes(none_clusters, rng)
        draws.append(
            fmean(prod_values[index] for index in prod_indexes)
            - fmean(none_values[index] for index in none_indexes)
        )
    draws.sort()
    tail = (1.0 - confidence_level) / 2.0
    return point, _quantile(draws, tail), _quantile(draws, 1.0 - tail)


def _fraction_difference(
    production: Sequence[BaselineOutcome],
    do_nothing: Sequence[BaselineOutcome],
) -> float | None:
    left = [
        item.requirement_fraction
        for item in production
        if item.requirement_fraction is not None
    ]
    right = [
        item.requirement_fraction
        for item in do_nothing
        if item.requirement_fraction is not None
    ]
    if not left or not right:
        return None
    return fmean(left) - fmean(right)


def _task_mix(
    production: Sequence[BaselineOutcome],
    confidence_level: float,
    resamples: int,
    seed: int,
) -> tuple[TaskMixSlice, ...]:
    groups: dict[tuple[str | None, int | None], list[BaselineOutcome]] = defaultdict(list)
    for item in production:
        if item.success is None:
            continue
        groups[(item.app, item.difficulty)].append(item)
    slices: list[TaskMixSlice] = []
    for index, key in enumerate(sorted(groups, key=lambda item: (item[0] or "", item[1] or 0))):
        members = groups[key]
        rate, low, high = _role_interval(
            members,
            confidence_level,
            resamples,
            seed + index,
        )
        slices.append(
            TaskMixSlice(
                app=key[0],
                difficulty=key[1],
                episode_count=len(members),
                scenario_count=len({item.scenario_id for item in members}),
                success_rate=rate,
                confidence_low=low,
                confidence_high=high,
            )
        )
    return tuple(slices)


def _mix_effect(slices: Sequence[TaskMixSlice]) -> float | None:
    rates = [item.success_rate for item in slices if item.success_rate is not None]
    if len(rates) < 2:
        return None
    return max(rates) - min(rates)


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    return lower + (upper - lower) * (position - lower_index)


def _rate_interval(
    flags: Sequence[bool],
    level: float,
) -> tuple[float, float, float]:
    successes = sum(1 for item in flags if item)
    rate = successes / len(flags)
    low, high = _wilson(successes, len(flags), level)
    return rate, low, high


def _wilson(successes: int, trials: int, level: float) -> tuple[float, float]:
    """Wilson score interval for a binomial rate.

    ``level`` is the two-sided central probability. The normal quantile is
    computed from the error function. This is the uncertainty stored on a
    rate, not a protocol threshold.
    """

    if trials < 1:
        raise ValidationError("wilson interval needs at least one trial")
    z = _normal_quantile(0.5 + level / 2.0)
    proportion = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (proportion + z2 / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z2 / (4.0 * trials * trials))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _normal_quantile(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValidationError("probability must be between zero and one")
    low = -8.0
    high = 8.0
    while _normal_cdf(low) > probability:
        low *= 2.0
    while _normal_cdf(high) < probability:
        high *= 2.0
    for _ in range(80):
        mid = (low + high) / 2.0
        if _normal_cdf(mid) < probability:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def _seeds(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValidationError("null_seeds must be a sequence of ints")
    seeds: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValidationError("null_seeds must be a sequence of ints")
        seeds.append(value)
    if len(seeds) != len(set(seeds)):
        raise ValidationError("null_seeds repeats a seed")
    return tuple(sorted(seeds))


def _cases(
    values: Sequence[ReferenceCase],
    contract: _Contract,
) -> tuple[ReferenceCase, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValidationError("reference_cases must be a sequence")
    cases: list[ReferenceCase] = []
    seen: set[str] = set()
    for case in values:
        if not isinstance(case, ReferenceCase):
            raise ValidationError("reference case is required")
        if case.case_id in seen:
            raise ValidationError("reference cases repeat a case_id")
        seen.add(case.case_id)
        if case.statistic not in contract.statistics:
            raise ValidationError("statistic is not produced by the method")
        cases.append(case)
    return tuple(cases)


def _observations(values: Sequence[MonitorObservation]) -> tuple[MonitorObservation, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValidationError("aa_observations must be a sequence")
    observations: list[MonitorObservation] = []
    for item in values:
        if not isinstance(item, MonitorObservation):
            raise ValidationError("monitor observation is required")
        if item.split in _HELD_OUT:
            raise ValidationError("validation does not run test_normal or test_challenge")
        observations.append(item)
    return tuple(observations)


def _resolve_split(
    split: str | None,
    observations: Sequence[MonitorObservation],
) -> str | None:
    observed = {item.split for item in observations}
    if len(observed) > 1:
        raise ValidationError("observations must share one split")
    if observed:
        only = next(iter(observed))
        if split is not None and split != only:
            raise ValidationError("split does not match the observations")
        return _development_split(only, required=True)
    return _development_split(split, required=False)


def _development_split(split: str | None, *, required: bool) -> str | None:
    if split is None:
        if required:
            raise ValidationError("split must be train or dev")
        return None
    if not isinstance(split, str) or split not in SPLITS:
        raise ValidationError("split is unknown")
    if split in _HELD_OUT:
        raise ValidationError("validation does not run test_normal or test_challenge")
    return split


def _resolve_aa(
    observations: Sequence[MonitorObservation],
    context: AAContext | None,
    rows: Sequence[AAStudyRow] | None,
    capture: AACaptureResult | None,
) -> tuple[tuple[AAStudyRow, ...] | None, tuple[str, ...], AAContext | None]:
    if rows is not None and capture is not None:
        raise ValidationError("pass an A/A capture or study rows, not both")
    if capture is not None:
        if context is None:
            raise ValidationError("A/A provenance must be supplied")
        if context.hardware_observed != capture.cost.hardware_observed:
            raise ValidationError("hardware_observed does not match the capture")
        if capture.cost.hardware_observed and context.memory_used_mib not in (
            None,
            capture.cost.memory_used_mib,
        ):
            raise ValidationError("memory_used_mib does not match the capture")
        derived = aa_study_rows(capture)
        memory = (
            capture.cost.memory_used_mib
            if capture.cost.hardware_observed
            else None
        )
        wall = capture.cost.wall_seconds if capture.cost.hardware_observed else None
        aligned = AAContext(
            provenance=context.provenance,
            hardware_observed=context.hardware_observed,
            memory_used_mib=memory,
            wall_seconds=wall,
        )
        return derived, (capture.configuration_hash,), aligned
    if rows is not None:
        prepared = tuple(rows)
        for row in prepared:
            if not isinstance(row, AAStudyRow):
                raise ValidationError("A/A study row is required")
        return prepared, (), context
    if context is not None and observations:
        _align_context(context, len(observations))
    hashes = tuple(
        sorted({item.run.configuration_hash for item in observations})
    )
    return None, hashes, context


def _align_context(context: AAContext, count: int) -> None:
    for name in ("scenario_ids", "task_ids", "repetitions"):
        value = getattr(context, name)
        if value is not None and len(value) != count:
            raise ValidationError(f"{name} must align with the observations")
    if context.repetitions is not None:
        for repetition in context.repetitions:
            _nonnegative_int(repetition, "repetition")
    if context.task_ids is not None:
        for task_id in context.task_ids:
            _short_text(task_id, "task_id")
    if context.scenario_ids is not None:
        for scenario_id in context.scenario_ids:
            if scenario_id is not None:
                _short_text(scenario_id, "scenario_id")


def _reject_mixed_aa(
    observations: Sequence[MonitorObservation],
    rows: Sequence[AAStudyRow] | None,
) -> None:
    if observations and rows:
        raise ValidationError("pass monitor observations or A/A pair rows, not both")


def _check_aa_bootstrap(
    confidence_level: float | None,
    resamples: int | None,
    seed: int | None,
    rows: Sequence[AAStudyRow] | None,
    context: AAContext | None,
) -> None:
    del rows, context
    supplied = (confidence_level is not None, resamples is not None, seed is not None)
    if any(supplied) and not all(supplied):
        raise ValidationError("A/A bootstrap inputs must be supplied together")
    if confidence_level is not None:
        _open_probability(confidence_level, "aa_confidence_level")
    if resamples is not None:
        _positive_int(resamples, "aa_resamples")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValidationError("aa_seed must be an int")


def _check_kl_argument(
    contract: _Contract,
    comparison: KLApproximationReport | None,
) -> None:
    if comparison is None:
        return
    if not isinstance(comparison, KLApproximationReport):
        raise ValidationError("KL approximation report is required")
    if contract.name != "truncated_next_token_kl":
        raise ValidationError("KL approximation applies only to truncated KL")


def _check_payload_keys(
    spec: MethodSpec,
    contract: _Contract,
    payload: Mapping[str, object],
) -> None:
    allowed = contract.data_keys | contract.parameters
    unknown = [key for key in payload if key not in allowed]
    if unknown:
        raise ValidationError("reference payload has an unknown field")
    required = contract.data_keys - contract.optional_data
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValidationError("reference payload is missing a field")
    for key, value in payload.items():
        if key in spec.parameter_map() and spec.parameter_map()[key] != _coerce_match(
            key, value
        ):
            raise ValidationError("reference payload does not match the method")


def _coerce_match(key: str, value: object) -> object:
    if key in _INT_PARAMETERS:
        return _int_value(value, key)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValidationError(f"{key} has a bad value")
    if isinstance(value, float):
        return value
    if isinstance(value, int) and key != "direction":
        return float(value)
    return value


def _validation_inputs(
    spec: MethodSpec,
    seeds: Sequence[int],
    cases: Sequence[ReferenceCase],
    observations: Sequence[MonitorObservation],
    context: AAContext | None,
    rows: Sequence[AAStudyRow] | None,
    split: str | None,
    confidence_level: float | None,
    resamples: int | None,
    seed: int | None,
    comparison: KLApproximationReport | None,
) -> dict[str, object]:
    return {
        "method": spec.name,
        "parameters": dict(spec.parameters),
        "required_checks": list(spec.required_checks),
        "study": spec.study,
        "null_sample_size": spec.null_sample_size,
        "uncertainty_level": spec.uncertainty_level,
        "null_draw": spec.null_draw,
        "alpha": spec.alpha,
        "horizon": spec.horizon,
        "false_alarm_tolerance": spec.false_alarm_tolerance,
        "coverage_tolerance": spec.coverage_tolerance,
        "repeated_look_stride": spec.repeated_look_stride,
        "seeds": list(seeds),
        "split": split,
        "reference_cases": [
            {
                "case_id": case.case_id,
                "statistic": case.statistic,
                "expected": case.expected,
                "tolerance": case.tolerance,
                "source": case.source,
                "payload": case.payload(),
            }
            for case in cases
        ],
        "observations": [
            {
                "episode_id": item.episode.episode_id,
                "signal": item.signal,
                "value": item.value,
                "split": item.split,
                "observed_at": item.observed_at.isoformat(),
            }
            for item in observations
        ],
        "context": None
        if context is None
        else {
            "provenance": context.provenance,
            "hardware_observed": context.hardware_observed,
            "scenario_ids": context.scenario_ids,
            "task_ids": context.task_ids,
            "repetitions": context.repetitions,
            "memory_used_mib": context.memory_used_mib,
            "wall_seconds": context.wall_seconds,
        },
        "rows": None
        if rows is None
        else [
            {
                "scenario_id": row.scenario_id,
                "task_id": row.task_id,
                "repetition": row.repetition,
                "reference_success": row.reference_success,
                "candidate_success": row.candidate_success,
                "disagreement": row.disagreement,
                "concurrency": row.concurrency,
            }
            for row in rows
        ],
        "aa_confidence_level": confidence_level,
        "aa_resamples": resamples,
        "aa_seed": seed,
        "kl_input_hash": None if comparison is None else comparison.input_hash,
    }


def _canonical_parameters(spec: MethodSpec) -> tuple[tuple[str, str], ...]:
    return tuple(
        (name, _canonical_scalar(value))
        for name, value in sorted(spec.parameters)
    )


def _canonical_scalar(value: int | float | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        return format(value, ".16g")
    return str(value)


def _public_document(report: object) -> dict[str, object]:
    document = _document(report)
    if not isinstance(document, dict):
        raise ValidationError("report document must be an object")
    document["visibility"] = "public"
    assert_public_payload(document)
    return document


def _document(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _document(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, tuple):
        return [_document(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _document(item) for key, item in value.items()}
    return value


def _hash_document(document: object) -> str:
    payload = _canonical_json(_json_ready(document))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json(document: object) -> str:
    try:
        return json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValidationError("hash document is not finite JSON") from error


def _json_ready(value: object) -> object:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        if value == -math.inf:
            return "-inf"
        raise ValidationError("hash document is not finite JSON")
    return value


def _freeze_parameters(
    parameters: Mapping[str, object],
) -> tuple[tuple[str, int | float | str], ...]:
    frozen: list[tuple[str, int | float | str]] = []
    for name, value in parameters.items():
        if not isinstance(name, str) or _TOKEN.fullmatch(name) is None:
            raise ValidationError("parameter name must be a token")
        frozen.append((name, _freeze_parameter(name, value)))
    return tuple(frozen)


def _freeze_parameter(name: str, value: object) -> int | float | str:
    if isinstance(value, bool) or value is None:
        raise ValidationError(f"{name} has a bad value")
    if name == "direction":
        if not isinstance(value, str):
            raise ValidationError("direction must be a string")
        return value
    if name in _INT_PARAMETERS:
        return _int_value(value, name)
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError(f"{name} must be finite")
        return value
    raise ValidationError(f"{name} has a bad value")


def _string_tuple(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise ValidationError(f"{name} must be a list")
    items: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ValidationError(f"{name} must be a list of strings")
        items.append(value)
    return tuple(items)


def _alpha(spec: MethodSpec) -> float:
    if spec.alpha is None:
        raise ValidationError("alpha is required")
    return spec.alpha


def _count_categories(draws: Sequence[int], categories: int) -> list[int]:
    counts = [0] * categories
    for item in draws:
        counts[item] += 1
    return counts


def _columns_positive(
    left: Sequence[int],
    right: Sequence[int],
    categories: int,
) -> bool:
    left_counts = _count_categories(left, categories)
    right_counts = _count_categories(right, categories)
    return all(left_count + right_count > 0 for left_count, right_count in zip(
        left_counts, right_counts, strict=True
    ))


def _floats(payload: Mapping[str, object], key: str) -> list[float]:
    return [_finite(item, key) for item in _list(payload, key)]


def _ints(payload: Mapping[str, object], key: str) -> list[int]:
    return [_int_value(item, key) for item in _list(payload, key)]


def _strings(payload: Mapping[str, object], key: str) -> list[str]:
    values = []
    for item in _list(payload, key):
        if not isinstance(item, str) or item == "":
            raise ValidationError(f"{key} must be non-empty strings")
        values.append(item)
    return values


def _optional_strings(payload: Mapping[str, object], key: str) -> list[str] | None:
    if key not in payload or payload[key] is None:
        return None
    return _strings(payload, key)


def _matrix(payload: Mapping[str, object], key: str) -> list[tuple[float, ...]]:
    rows = []
    for row in _list(payload, key):
        if isinstance(row, str) or not isinstance(row, list):
            raise ValidationError(f"{key} must be a matrix")
        rows.append(tuple(_finite(item, key) for item in row))
    return rows


def _pair_values(payload: Mapping[str, object]) -> list[PairedSuccess]:
    pairs: list[PairedSuccess] = []
    for row in _list(payload, "pairs"):
        if isinstance(row, str) or not isinstance(row, list) or len(row) != 2:
            raise ValidationError("pairs must be [candidate, reference] rows")
        pairs.append(
            PairedSuccess(_finite(row[0], "candidate"), _finite(row[1], "reference"))
        )
    if not pairs:
        raise ValidationError("pairs are empty")
    return pairs


def _list(payload: Mapping[str, object], key: str) -> list[object]:
    if key not in payload:
        raise ValidationError("reference payload is missing a field")
    value = payload[key]
    if isinstance(value, str) or not isinstance(value, list):
        raise ValidationError(f"{key} must be a list")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{name} must be finite")
    return number


def _int_value(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{name} must be an int")
    return value


def _positive_int(value: object, name: str) -> int:
    number = _int_value(value, name)
    if number < 1:
        raise ValidationError(f"{name} must be a positive integer")
    return number


def _nonnegative_int(value: object, name: str) -> int:
    number = _int_value(value, name)
    if number < 0:
        raise ValidationError(f"{name} must be a nonnegative integer")
    return number


def _optional_int(value: object, name: str, *, minimum: int) -> None:
    if value is None:
        return
    number = _int_value(value, name)
    if number < minimum:
        raise ValidationError(f"{name} is out of range")


def _open_probability(value: object, name: str) -> float:
    number = _finite(value, name)
    if not 0.0 < number < 1.0:
        raise ValidationError(f"{name} must be between zero and one")
    return number


def _unit_float(value: object, name: str) -> float:
    number = _finite(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValidationError(f"{name} must lie in [0, 1]")
    return number


def _positive_finite(value: object, name: str) -> float:
    number = _finite(value, name)
    if number <= 0.0:
        raise ValidationError(f"{name} must be positive")
    return number


def _nonnegative_finite(value: object, name: str) -> float:
    number = _finite(value, name)
    if number < 0.0:
        raise ValidationError(f"{name} must be at least zero")
    return number


def _optional_nonnegative(value: object, name: str) -> None:
    if value is not None:
        _nonnegative_finite(value, name)


def _optional_finite(value: object, name: str, *, minimum: float) -> None:
    if value is None:
        return
    number = _finite(value, name)
    if number < minimum:
        raise ValidationError(f"{name} is out of range")


def _optional_success(value: object, name: str) -> None:
    if value is None:
        return
    number = _finite(value, name)
    if number not in (0.0, 1.0):
        raise ValidationError(f"{name} must be 0 or 1")


def _optional_fraction(value: object, name: str) -> None:
    if value is None:
        return
    _unit_float(value, name)


def _match_fraction(success: float | None, fraction: float | None) -> None:
    if success is None or fraction is None:
        return
    if (success == 1.0) != (fraction == 1.0):
        raise ValidationError("success does not match the requirement fraction")


def _short_text(value: object, name: str) -> None:
    if not isinstance(value, str) or value == "" or len(value) > 256 or "\n" in value:
        raise ValidationError(f"{name} must be a single line")


def _sample_sizes(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ValidationError("sample_sizes must be a non-empty list")
    sizes = [_positive_int(value, "sample_size") for value in values]
    if len(sizes) != len(set(sizes)):
        raise ValidationError("sample_sizes repeats a size")
    return tuple(sizes)


def _baseline_outcomes(values: Sequence[BaselineOutcome]) -> tuple[BaselineOutcome, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValidationError("outcomes must be a sequence")
    prepared = []
    for item in values:
        if not isinstance(item, BaselineOutcome):
            raise ValidationError("baseline outcome is required")
        prepared.append(item)
    keyed = [item for item in prepared if item.pair_key is not None]
    if keyed and len(keyed) != len(prepared):
        raise ValidationError("pair keys must be present on every outcome or on none")
    return tuple(prepared)


def _harm_from_payload(payload: object) -> HarmStudyReport:
    if not isinstance(payload, dict):
        raise ValidationError("harm spec must be an object")
    outcomes_payload = payload.get("outcomes", [])
    if not isinstance(outcomes_payload, list):
        raise ValidationError("outcomes must be a list")
    outcomes = tuple(_baseline_from_dict(item) for item in outcomes_payload)
    try:
        return assess_harm_study(
            outcomes,
            split=payload["split"],
            study=payload["study"],
            harm_margin=payload["harm_margin"],
            sample_sizes=payload["sample_sizes"],
            alpha=payload["alpha"],
            variance=payload["variance"],
            power_target=payload["power_target"],
            confidence_level=payload.get("confidence_level"),
            bootstrap_resamples=payload.get("bootstrap_resamples"),
            seed=payload.get("seed"),
        )
    except KeyError as error:
        raise ValidationError("harm spec is missing a field") from error


def _baseline_from_dict(payload: object) -> BaselineOutcome:
    if not isinstance(payload, dict):
        raise ValidationError("baseline outcome must be an object")
    try:
        return BaselineOutcome(
            role=payload["role"],
            success=payload["success"],
            requirement_fraction=payload.get("requirement_fraction"),
            scenario_id=payload["scenario_id"],
            app=payload.get("app"),
            difficulty=payload.get("difficulty"),
            pair_key=payload.get("pair_key"),
        )
    except KeyError as error:
        raise ValidationError("baseline outcome is missing a field") from error


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError("input is not readable JSON") from error


def _load_observations(path: Path) -> tuple[MonitorObservation, ...]:
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValidationError("observations must be a list")
    return tuple(MonitorObservation.from_dict(item) for item in payload)


def _load_aa_rows(path: Path) -> tuple[AAContext, tuple[AAStudyRow, ...]]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValidationError("A/A rows must be an object")
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise ValidationError("rows must be a list")
    try:
        context = AAContext(
            provenance=payload["provenance"],
            hardware_observed=payload["hardware_observed"],
            memory_used_mib=payload.get("memory_used_mib"),
            wall_seconds=payload.get("wall_seconds"),
        )
        rows = []
        for item in raw_rows:
            if not isinstance(item, dict):
                raise ValidationError("A/A study row is required")
            rows.append(
                AAStudyRow(
                    scenario_id=item.get("scenario_id"),
                    task_id=item["task_id"],
                    repetition=item["repetition"],
                    reference_success=item.get("reference_success"),
                    candidate_success=item.get("candidate_success"),
                    disagreement=item.get("disagreement"),
                    reference_requirement_fraction=item.get(
                        "reference_requirement_fraction"
                    ),
                    candidate_requirement_fraction=item.get(
                        "candidate_requirement_fraction"
                    ),
                    trajectory_diverged=item["trajectory_diverged"],
                    length_difference=item["length_difference"],
                    concurrency=item["concurrency"],
                )
            )
    except KeyError as error:
        raise ValidationError("A/A rows are missing a field") from error
    return context, tuple(rows)


def _load_kl_samples(path: Path) -> tuple[KLPositionSample, ...]:
    payload = _load_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValidationError("KL sample file must contain samples")
    samples = []
    try:
        for item in payload["samples"]:
            if not isinstance(item, dict):
                raise ValidationError("KL sample is required")
            samples.append(
                KLPositionSample(
                    production_log_probabilities=tuple(
                        tuple(row) for row in item["production"]
                    ),
                    candidate_log_probabilities=tuple(
                        tuple(row) for row in item["candidate"]
                    ),
                )
            )
    except KeyError as error:
        raise ValidationError("KL sample is missing a field") from error
    return tuple(samples)


def _under(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    base = root.resolve()
    return resolved == base or base in resolved.parents
