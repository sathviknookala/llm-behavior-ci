from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from llm_behavior_ci.config import EpisodeIdentity, RunIdentity
from llm_behavior_ci.experiments.validation import (
    AAContext,
    AAStudyRow,
    BaselineOutcome,
    KLPositionSample,
    ValidationError,
    aa_study_rows,
    apply_validation_reports,
    assess_harm_study,
    compare_plan_kl,
    harm_detection_power,
    implemented_methods,
    main_compare_kl,
    main_harm,
    main_validate,
    method_spec,
    minimum_checks,
    public_harm_summary,
    public_kl_summary,
    public_validation_summary,
    reference_case,
    reference_library_status,
    validate_method,
)
from llm_behavior_ci.records import (
    EvaluatorOutcome,
    LocalTaskRef,
    ModelStep,
    MonitorObservation,
    PairedResult,
    TokenLogprob,
    EpisodeResult,
)
from llm_behavior_ci.runtime.aa_capture import (
    AACaptureResult,
    AAPairRecord,
    ExecutionCost,
    ScheduledInput,
    TrajectoryDivergence,
)
from llm_behavior_ci.runtime.episode import PairExecution
from llm_behavior_ci.stats.chi_square import chi_square_homogeneity
from llm_behavior_ci.stats.ks import ks_two_sample

_HASH = "a" * 64
_TASK_HASH = "c" * 64
_GIT = "d" * 40
_START = datetime(2026, 9, 27, 15, 0, tzinfo=timezone.utc)


def _run() -> RunIdentity:
    return RunIdentity(
        run_id=f"{_HASH}.{'1' * 32}",
        configuration_hash=_HASH,
        task_set_hash=_TASK_HASH,
        protocol_hash=None,
        git_commit=_GIT,
    )


def _observation(
    token: str,
    value: float,
    minute: int,
    signal: str = "task_success",
) -> MonitorObservation:
    run = _run()
    return MonitorObservation(
        episode=EpisodeIdentity(
            episode_id=f"{run.run_id}.{token}",
            run_id=run.run_id,
        ),
        run=run,
        split="dev",
        signal=signal,
        value=value,
        observed_at=_START + timedelta(minutes=minute),
    )


def _canary_spec(**overrides: object):
    values = {
        "required_checks": (
            "null_false_alarm",
            "stopping",
            "reference",
            "aa_dependence",
        ),
        "study": "cpu_fast",
        "null_sample_size": 2,
        "uncertainty_level": 0.95,
        "null_draw": "constant",
        "alpha": 0.05,
        "horizon": 2,
        "false_alarm_tolerance": 0.0,
    }
    values.update(overrides)
    return method_spec(
        "sequential_canary",
        {
            "alpha": 0.05,
            "harm_margin": 0.0,
            "horizon_episodes": 2,
            "pair_value": 1.0,
        },
        **values,
    )


def _reference_cases():
    payload = {"pairs": [[1.0, 1.0], [1.0, 1.0]]}
    return (
        reference_case("no_alarm", "alarm", 0.0, 0.0, "closed_form", payload),
        reference_case(
            "horizon_stop",
            "stopped_at_horizon",
            1.0,
            0.0,
            "closed_form",
            payload,
        ),
    )


def _rows(task_one: str = "secret-task-alpha", task_two: str = "secret-task-beta"):
    scenario_one = "secret-scenario-one"
    scenario_two = "secret-scenario-two"
    return (
        AAStudyRow(
            scenario_one,
            task_one,
            0,
            1.0,
            0.0,
            True,
            1.0,
            0.0,
            False,
            0,
            1,
        ),
        AAStudyRow(
            scenario_one,
            task_one,
            1,
            1.0,
            1.0,
            False,
            1.0,
            1.0,
            False,
            0,
            1,
        ),
        AAStudyRow(
            scenario_two,
            task_two,
            0,
            0.0,
            0.0,
            False,
            0.0,
            0.0,
            False,
            0,
            1,
        ),
        AAStudyRow(
            scenario_two,
            task_two,
            1,
            1.0,
            1.0,
            False,
            1.0,
            1.0,
            True,
            1,
            1,
        ),
    )


def _validate_canary(rows=_rows(), study="cpu_fast", provenance="local_runtime", **context):
    return validate_method(
        _canary_spec(study=study),
        null_seeds=(1, 2, 3),
        reference_cases=_reference_cases(),
        aa_observations=(),
        split="dev",
        aa_context=AAContext(provenance=provenance, hardware_observed=False, **context),
        aa_rows=rows,
        aa_confidence_level=0.9,
        aa_resamples=20,
        aa_seed=4,
    )


def _check(report, name: str):
    return next(item for item in report.checks if item.name == name)


class CatalogTests(unittest.TestCase):
    def test_catalog_lists_implemented_methods_as_not_validated(self) -> None:
        methods = implemented_methods()
        names = [item.name for item in methods]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("sequential_canary", names)
        self.assertIn("next_token_kl", names)
        self.assertIn("truncated_next_token_kl", names)
        self.assertTrue(all(not item.validated for item in methods))
        self.assertIn("aa_dependence", minimum_checks("sequential_canary"))
        self.assertNotIn("aa_dependence", minimum_checks("next_token_kl"))
        self.assertEqual(
            {name for name, _present in reference_library_status()},
            {"confseq", "river", "scipy"},
        )

    def test_source_does_not_read_demo_constants_or_launch_a_runtime(self) -> None:
        source = Path(__file__).parents[2].joinpath(
            "src/llm_behavior_ci/experiments/validation.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("DEMO_", source)
        self.assertNotIn("offline_gate", source)
        self.assertNotIn("subprocess", source)


class NullAndReferenceTests(unittest.TestCase):
    def test_constant_canary_is_eligible_only_with_accepted_aa_evidence(self) -> None:
        report = _validate_canary()
        self.assertTrue(report.implemented)
        self.assertTrue(report.benchmark_eligible)
        self.assertTrue(report.validated)
        self.assertEqual(report.calibration, "calibrated")
        self.assertEqual(report.sample_count, 3)
        self.assertEqual(_check(report, "null_false_alarm").estimate, 0.0)
        self.assertEqual(_check(report, "stopping").status, "passed")
        self.assertEqual(_check(report, "reference").status, "passed")
        self.assertEqual(report.aa.status, "passed")
        self.assertEqual(report.aa.inference_source, "evaluator_disagreement")
        self.assertAlmostEqual(report.aa.inference_variation or 0.0, 0.25, places=6)
        self.assertIsNotNone(report.aa.repeated_task_effect)
        self.assertIsNotNone(report.aa.scenario_clustering_effect)
        self.assertIsNotNone(report.aa.interval_width_ratio)
        self.assertFalse(report.gpu_floor_measured)
        self.assertFalse(report.gpu_evidence)
        again = _validate_canary()
        self.assertEqual(report, again)
        summary = json.dumps(public_validation_summary(report))
        self.assertNotIn("secret-task-alpha", summary)
        self.assertNotIn("secret-scenario-one", summary)
        changed = _validate_canary(rows=_rows(task_one="secret-task-other"))
        self.assertNotEqual(report.input_hash, changed.input_hash)

    def test_missing_or_synthetic_aa_is_not_a_floor_and_not_eligible(self) -> None:
        missing = validate_method(
            _canary_spec(),
            null_seeds=(1, 2, 3),
            reference_cases=_reference_cases(),
            aa_observations=(),
            split="dev",
        )
        self.assertFalse(missing.benchmark_eligible)
        self.assertIsNone(missing.aa.inference_variation)
        self.assertIsNone(missing.aa.repeated_task_effect)
        self.assertEqual(missing.aa.pair_count, 0)
        self.assertFalse(missing.aa.evidence_accepted)
        synthetic = _validate_canary(provenance="synthetic")
        self.assertFalse(synthetic.benchmark_eligible)
        self.assertIsNone(synthetic.aa.inference_variation)
        self.assertIsNone(synthetic.aa.scenario_clustering_effect)
        self.assertEqual(synthetic.aa.pair_count, 4)
        self.assertFalse(synthetic.aa.evidence_accepted)
        self.assertIsNone(synthetic.aa.memory_used_mib)

    def test_wrong_reference_value_fails_the_method(self) -> None:
        bad = reference_case(
            "no_alarm",
            "alarm",
            1.0,
            0.0,
            "caller",
            {"pairs": [[1.0, 1.0], [1.0, 1.0]]},
        )
        report = validate_method(
            _canary_spec(required_checks=("reference",)),
            null_seeds=(),
            reference_cases=(bad,),
            aa_observations=(),
        )
        self.assertEqual(_check(report, "reference").status, "failed")
        self.assertFalse(report.benchmark_eligible)
        self.assertIn("aa_dependence", report.omitted_checks)

    def test_degenerate_bootstrap_covers_zero_and_does_not_fake_a_level(self) -> None:
        spec = method_spec(
            "paired_bootstrap",
            {
                "confidence_level": 0.9,
                "resamples": 20,
                "null_mean": 0.0,
                "null_scale": 0.0,
            },
            required_checks=("null_false_alarm", "coverage"),
            study="cpu_fast",
            null_sample_size=4,
            uncertainty_level=0.95,
            null_draw="gaussian",
            alpha=0.1,
            false_alarm_tolerance=0.0,
            coverage_tolerance=0.0,
        )
        report = validate_method(
            spec,
            null_seeds=(1, 2),
            reference_cases=(),
            aa_observations=(),
        )
        self.assertEqual(_check(report, "coverage").estimate, 1.0)
        self.assertEqual(_check(report, "coverage").status, "passed")
        self.assertEqual(_check(report, "null_false_alarm").estimate, 0.0)
        self.assertEqual(_check(report, "null_false_alarm").status, "failed")
        self.assertFalse(report.benchmark_eligible)

    def test_repeated_looks_are_at_least_the_single_look(self) -> None:
        spec = method_spec(
            "ks_two_sample",
            {},
            required_checks=("repeated_look",),
            study="cpu_fast",
            null_sample_size=16,
            uncertainty_level=0.95,
            null_draw="uniform",
            alpha=0.05,
            false_alarm_tolerance=1.0,
            repeated_look_stride=4,
        )
        report = validate_method(
            spec,
            null_seeds=(1, 2, 3, 4),
            reference_cases=(
                reference_case(
                    "identical",
                    "p_value",
                    1.0,
                    0.0,
                    "closed_form",
                    {"left": [0.0, 0.0], "right": [0.0, 0.0]},
                ),
            ),
            aa_observations=(),
        )
        repeated = _check(report, "repeated_look")
        single = dict(repeated.details)["single_rate"]
        self.assertGreaterEqual(repeated.estimate or 0.0, single)
        self.assertEqual(repeated.status, "passed")
        self.assertEqual(_check(report, "reference").status, "passed")
        self.assertEqual(
            validate_method(
                spec,
                null_seeds=(4, 1, 3, 2),
                reference_cases=report.reference_agreements and (
                    reference_case(
                        "identical",
                        "p_value",
                        1.0,
                        0.0,
                        "closed_form",
                        {"left": [0.0, 0.0], "right": [0.0, 0.0]},
                    ),
                ),
                aa_observations=(),
            ),
            report,
        )

    def test_ks_reference_matches_the_implementation_and_scipy_when_present(self) -> None:
        left = [0.1, 0.4, 0.2, 0.8]
        right = [0.3, 0.5, 0.15, 0.7]
        observed = ks_two_sample(left, right)
        spec = method_spec(
            "ks_two_sample",
            {},
            required_checks=("reference",),
            study="cpu_fast",
            null_sample_size=4,
            uncertainty_level=0.95,
            null_draw="uniform",
            alpha=0.05,
            false_alarm_tolerance=1.0,
        )
        report = validate_method(
            spec,
            null_seeds=(),
            reference_cases=(
                reference_case(
                    "shared",
                    "p_value",
                    observed.p_value,
                    0.0,
                    "caller",
                    {"left": left, "right": right},
                ),
            ),
            aa_observations=(),
        )
        self.assertEqual(report.reference_agreements[0].absolute_error, 0.0)
        if importlib.util.find_spec("scipy") is None:
            return
        from scipy.stats import chi2_contingency, ks_2samp

        scipy_ks = ks_2samp(left, right, method="asymp")
        self.assertAlmostEqual(observed.statistic, float(scipy_ks.statistic), places=6)
        table = chi_square_homogeneity([12, 8, 10], [9, 11, 10])
        _statistic, p_value, _dof, _expected = chi2_contingency(
            [[12, 8, 10], [9, 11, 10]],
            correction=False,
        )
        self.assertAlmostEqual(table.p_value, float(p_value), places=5)

    def test_held_out_splits_and_unknown_methods_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            method_spec(
                "not_a_method",
                {},
                required_checks=(),
                study="cpu_fast",
                null_sample_size=1,
                uncertainty_level=0.95,
                null_draw="uniform",
            )
        observation = _observation("2" * 32, 1.0, 0)
        held_out = MonitorObservation(
            episode=observation.episode,
            run=observation.run,
            split="test_normal",
            signal="task_success",
            value=1.0,
            observed_at=observation.observed_at,
        )
        with self.assertRaises(ValidationError):
            validate_method(
                _canary_spec(required_checks=("null_false_alarm",)),
                null_seeds=(1,),
                reference_cases=(),
                aa_observations=(held_out,),
            )


class KLAndBaselineTests(unittest.TestCase):
    def test_topk_error_is_recorded_and_absent_without_a_sample_or_gpu_floor(self) -> None:
        production = (math.log(0.7), math.log(0.2), math.log(0.1))
        candidate = (math.log(0.2), math.log(0.1), math.log(0.7))
        sample = KLPositionSample((production,), (candidate,))
        report = compare_plan_kl(
            (sample,),
            top_k=2,
            provenance="supplied_sample",
            vocabulary_size=8,
        )
        full = _kl((0.7, 0.2, 0.1), (0.2, 0.1, 0.7))
        truncated = _kl((0.7 / 0.9, 0.2 / 0.9), (0.2 / 0.3, 0.1 / 0.3))
        self.assertEqual(report.status, "passed")
        self.assertEqual(report.approximation, "top_k")
        self.assertEqual(report.top_k, 2)
        self.assertEqual(report.vocabulary_size, 8)
        self.assertAlmostEqual(report.mean_signed_error or 0.0, full - truncated, places=8)
        self.assertGreater(report.mean_absolute_error or 0.0, 0.5)
        self.assertFalse(report.gpu_floor_measured)
        self.assertNotIn("task_id", json.dumps(public_kl_summary(report)))
        exact = compare_plan_kl(
            (sample,),
            top_k=3,
            provenance="local_runtime",
            vocabulary_size=3,
        )
        self.assertIsNotNone(exact.mean_absolute_error)
        self.assertAlmostEqual(exact.mean_absolute_error, 0.0, places=8)
        empty = compare_plan_kl((), top_k=2, provenance="supplied_sample")
        self.assertEqual(empty.status, "unavailable")
        self.assertIsNone(empty.mean_absolute_error)
        gpu = compare_plan_kl((sample,), top_k=2, provenance="gpu", hardware_observed=True)
        self.assertEqual(gpu.status, "unavailable")
        self.assertIsNone(gpu.mean_absolute_error)
        self.assertFalse(gpu.gpu_floor_measured)

    def test_truncated_kl_needs_the_supplied_sample_before_it_is_validated(self) -> None:
        logs = [[0.0, -1.0]]
        spec = method_spec(
            "truncated_next_token_kl",
            {},
            required_checks=("reference", "kl_approximation"),
            study="cpu_fast",
            null_sample_size=0,
            uncertainty_level=0.95,
            null_draw="none",
        )
        case = reference_case(
            "identical",
            "mean_kl_nats",
            0.0,
            1e-12,
            "closed_form",
            {"production": logs, "candidate": logs},
        )
        without = validate_method(
            spec,
            null_seeds=(),
            reference_cases=(case,),
            aa_observations=(),
        )
        self.assertEqual(_check(without, "reference").status, "passed")
        self.assertEqual(_check(without, "kl_approximation").status, "unavailable")
        self.assertFalse(without.benchmark_eligible)
        comparison = compare_plan_kl(
            (
                KLPositionSample(
                    ((math.log(0.6), math.log(0.4)),),
                    ((math.log(0.3), math.log(0.7)),),
                ),
            ),
            top_k=1,
            provenance="supplied_sample",
        )
        with_sample = validate_method(
            spec,
            null_seeds=(),
            reference_cases=(case,),
            aa_observations=(),
            kl_comparison=comparison,
        )
        self.assertTrue(with_sample.benchmark_eligible)
        self.assertEqual(with_sample.calibration, "measured")
        self.assertFalse(with_sample.gpu_floor_measured)
        statuses = apply_validation_reports((without, with_sample))
        truncated = next(item for item in statuses if item.name == "truncated_next_token_kl")
        canary = next(item for item in statuses if item.name == "sequential_canary")
        self.assertTrue(truncated.validated)
        self.assertFalse(canary.validated)
        self.assertFalse(truncated.gpu_floor_measured)

    def test_full_kl_reference_is_eligible_without_aa_or_a_gpu_floor(self) -> None:
        spec = method_spec(
            "next_token_kl",
            {},
            required_checks=("reference",),
            study="cpu_fast",
            null_sample_size=0,
            uncertainty_level=0.95,
            null_draw="none",
        )
        report = validate_method(
            spec,
            null_seeds=(),
            reference_cases=(
                reference_case(
                    "identical",
                    "mean_kl_nats",
                    0.0,
                    1e-12,
                    "closed_form",
                    {"production": [[0.0, -1.0]], "candidate": [[0.0, -1.0]]},
                ),
            ),
            aa_observations=(),
        )
        self.assertTrue(report.benchmark_eligible)
        self.assertEqual(report.calibration, "measured")
        self.assertFalse(report.gpu_floor_measured)

    def test_power_and_harm_feasibility_use_only_supplied_evidence(self) -> None:
        z_alpha = 1.6448536269514722
        margin = 0.2
        variance = 0.25
        size = 100
        standard_error = math.sqrt(variance / size)
        expected = 0.5 * (
            1.0 + math.erf((-z_alpha + margin / standard_error) / math.sqrt(2.0))
        )
        self.assertAlmostEqual(
            harm_detection_power(
                harm_margin=margin,
                sample_size=size,
                alpha=0.05,
                variance=variance,
            ),
            expected,
            places=8,
        )
        self.assertAlmostEqual(
            _normal_quantile_check(0.975),
            1.959963984540054,
            places=6,
        )
        undetermined = assess_harm_study(
            (),
            split="dev",
            study="cpu_fast",
            harm_margin=0.2,
            sample_sizes=(100, 400),
            alpha=0.05,
            variance=0.25,
            power_target=0.8,
        )
        self.assertIsNone(undetermined.feasible)
        self.assertFalse(undetermined.empirical_evidence)
        self.assertIsNone(undetermined.production_success_rate)
        self.assertTrue(all(item.meets_target for item in undetermined.powers))
        weak = assess_harm_study(
            (),
            split="train",
            study="cpu_fast",
            harm_margin=0.2,
            sample_sizes=(4,),
            alpha=0.05,
            variance=0.25,
            power_target=0.8,
        )
        self.assertFalse(weak.feasible)
        self.assertLess(weak.powers[0].power, 0.8)
        feasible = assess_harm_study(
            _baseline_pairs(success=True),
            split="dev",
            study="cpu_fast",
            harm_margin=0.2,
            sample_sizes=(100,),
            alpha=0.05,
            variance=0.25,
            power_target=0.8,
            confidence_level=0.9,
            bootstrap_resamples=30,
            seed=5,
        )
        self.assertTrue(feasible.production_beats_do_nothing)
        self.assertTrue(feasible.feasible)
        self.assertEqual(feasible.requirement_fraction_difference, 1.0)
        mixed = assess_harm_study(
            _mixed_production(),
            split="dev",
            study="cpu_fast",
            harm_margin=0.2,
            sample_sizes=(100,),
            alpha=0.05,
            variance=0.25,
            power_target=0.8,
            confidence_level=0.9,
            bootstrap_resamples=20,
            seed=6,
        )
        self.assertEqual(mixed.task_mix_effect, 1.0)
        summary = json.dumps(public_harm_summary(mixed))
        self.assertNotIn("secret-scenario", summary)
        with self.assertRaises(ValidationError):
            assess_harm_study(
                (),
                split="test_normal",
                study="cpu_fast",
                harm_margin=0.2,
                sample_sizes=(100,),
                alpha=0.05,
                variance=0.25,
                power_target=0.8,
            )
        with self.assertRaises(ValidationError):
            BaselineOutcome(
                "production",
                True,
                0.4,
                "scenario",
                None,
                1,
            )

    def test_monitor_labels_measure_repetition_without_writing_task_ids(self) -> None:
        observations = (
            _observation("2" * 32, 1.0, 0),
            _observation("3" * 32, 0.0, 1),
            _observation("4" * 32, 1.0, 2),
            _observation("5" * 32, 1.0, 3),
        )
        context = AAContext(
            provenance="local_runtime",
            hardware_observed=False,
            scenario_ids=("secret-scenario-one", "secret-scenario-one", "secret-scenario-two", "secret-scenario-two"),
            task_ids=("secret-task-alpha", "secret-task-alpha", "secret-task-beta", "secret-task-beta"),
            repetitions=(0, 1, 0, 1),
        )
        spec = method_spec(
            "cusum",
            {"target": 0.0, "slack": 0.0, "threshold": 5.0, "direction": "increase"},
            required_checks=("null_false_alarm", "stopping", "aa_dependence"),
            study="cpu_fast",
            null_sample_size=4,
            uncertainty_level=0.95,
            null_draw="constant",
            horizon=4,
        )
        report = validate_method(
            spec,
            null_seeds=(1,),
            reference_cases=(),
            aa_observations=observations,
            aa_context=context,
        )
        self.assertEqual(report.aa.inference_source, "within_task_repetition")
        self.assertIsNotNone(report.aa.repeated_task_effect)
        self.assertIsNotNone(report.aa.scenario_clustering_effect)
        self.assertIsNotNone(report.aa.inference_variation)
        summary = json.dumps(public_validation_summary(report))
        self.assertNotIn("secret-task-alpha", summary)
        self.assertNotIn("2" * 32, summary)


class MethodSmokeTests(unittest.TestCase):
    def test_every_catalog_method_has_a_small_cpu_path(self) -> None:
        seen = set()
        for spec, seeds, cases in _cpu_paths():
            seen.add(spec.name)
            report = validate_method(
                spec,
                null_seeds=seeds,
                reference_cases=cases,
                aa_observations=(),
            )
            self.assertTrue(report.implemented)
            self.assertEqual(report.benchmark_eligible, spec.name == "next_token_kl")
            self.assertTrue(any(check.status == "passed" for check in report.checks))
            self.assertEqual(len(report.input_hash), 64)
        self.assertEqual(seen, {item.name for item in implemented_methods()})


class CaptureAdapterTests(unittest.TestCase):
    def test_capture_rows_copy_evaluator_fields_and_not_plan_text(self) -> None:
        run = _run()
        pair_id = "e" * 32
        reference = _episode(run, "2" * 32, pair_id, "reference", True, 2, 2)
        candidate = _episode(run, "3" * 32, pair_id, "candidate", False, 1, 2)
        capture = AACaptureResult(
            configuration_hash=_HASH,
            task_set_hash=_TASK_HASH,
            repetitions=1,
            concurrency=2,
            modes=("execute",),
            schedule=(
                ScheduledInput(0, 0, "task-local", "scenario-local", 3, 0.0),
            ),
            records=(
                AAPairRecord(
                    schedule=ScheduledInput(0, 0, "task-local", "scenario-local", 3, 0.0),
                    mode="execute",
                    pair=PairedResult(reference, candidate),
                    execution=PairExecution(
                        pair_id,
                        ("reference", "candidate"),
                        reference.episode.episode_id,
                        candidate.episode.episode_id,
                        7,
                        7,
                        1,
                        1,
                        "state",
                    ),
                    evaluator_disagreement=True,
                    reference_requirement_fraction=1.0,
                    candidate_requirement_fraction=0.5,
                    trajectory=TrajectoryDivergence(None, 0, (), ()),
                    plan_scoring_inputs=None,
                ),
            ),
            evaluated_pairs=1,
            disagreement_count=1,
            missing_outcome_count=0,
            tool_selection=None,
            cost=ExecutionCost(False, None, None, None, "unobserved"),
        )
        rows = aa_study_rows(capture)
        self.assertEqual(rows[0].task_id, "task-local")
        self.assertEqual(rows[0].scenario_id, "scenario-local")
        self.assertTrue(rows[0].disagreement)
        self.assertEqual(rows[0].concurrency, 2)
        self.assertEqual(rows[0].reference_success, 1.0)
        self.assertEqual(rows[0].candidate_success, 0.0)


class CommandTests(unittest.TestCase):
    def test_commands_write_public_summaries_and_refuse_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = root / "spec.json"
            output = root / "out.json"
            spec.write_text(
                json.dumps(
                    {
                        "name": "next_token_kl",
                        "parameters": {},
                        "required_checks": ["reference"],
                        "study": "cpu_fast",
                        "null_sample_size": 0,
                        "uncertainty_level": 0.95,
                        "null_draw": "none",
                        "seeds": [],
                        "split": "dev",
                        "reference_cases": [
                            {
                                "case_id": "identical",
                                "statistic": "mean_kl_nats",
                                "expected": 0.0,
                                "tolerance": 1e-12,
                                "source": "closed_form",
                                "payload": {
                                    "production": [[0.0, -1.0]],
                                    "candidate": [[0.0, -1.0]],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            code = main_validate(
                ["--spec", str(spec), "--output", str(output), "--results-root", str(root / "elsewhere")]
            )
            self.assertEqual(code, 0)
            document = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(document["benchmark_eligible"])
            self.assertEqual(document["visibility"], "public")
            refused = main_validate(
                ["--spec", str(spec), "--output", str(root / "blocked.json"), "--results-root", str(root)]
            )
            self.assertEqual(refused, 1)
            self.assertFalse((root / "blocked.json").exists())
            sample = root / "sample.json"
            sample.write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "production": [[math.log(0.5), math.log(0.5)]],
                                "candidate": [[math.log(0.25), math.log(0.75)]],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            kl_output = root / "kl.json"
            self.assertEqual(
                main_compare_kl(
                    [
                        "--sample",
                        str(sample),
                        "--top-k",
                        "1",
                        "--output",
                        str(kl_output),
                        "--results-root",
                        str(root / "elsewhere"),
                    ]
                ),
                0,
            )
            kl_document = json.loads(kl_output.read_text(encoding="utf-8"))
            self.assertEqual(kl_document["status"], "passed")
            self.assertFalse(kl_document["gpu_floor_measured"])
            harm_spec = root / "harm.json"
            harm_spec.write_text(
                json.dumps(
                    {
                        "split": "dev",
                        "study": "cpu_fast",
                        "harm_margin": 0.2,
                        "sample_sizes": [100],
                        "alpha": 0.05,
                        "variance": 0.25,
                        "power_target": 0.8,
                        "outcomes": [],
                    }
                ),
                encoding="utf-8",
            )
            harm_output = root / "harm-out.json"
            self.assertEqual(
                main_harm(
                    [
                        "--spec",
                        str(harm_spec),
                        "--output",
                        str(harm_output),
                        "--results-root",
                        str(root / "elsewhere"),
                    ]
                ),
                0,
            )
            harm_document = json.loads(harm_output.read_text(encoding="utf-8"))
            self.assertIsNone(harm_document["feasible"])
            self.assertIsNone(harm_document["production_success_rate"])


def _cpu_paths():
    level = dict(
        study="cpu_fast",
        uncertainty_level=0.95,
        false_alarm_tolerance=1.0,
        alpha=0.05,
        required_checks=("null_false_alarm",),
    )
    sequence = dict(
        study="cpu_fast",
        uncertainty_level=0.95,
        false_alarm_tolerance=0.0,
        alpha=0.05,
        horizon=4,
        null_sample_size=4,
        required_checks=("null_false_alarm",),
    )
    bootstrap = {
        "confidence_level": 0.9,
        "resamples": 10,
        "null_mean": 0.0,
        "null_scale": 1.0,
    }
    identical = reference_case(
        "identical",
        "mean_kl_nats",
        0.0,
        1e-12,
        "closed_form",
        {"production": [[0.0, -1.0]], "candidate": [[0.0, -1.0]]},
    )
    paths = [
        (
            method_spec(
                "ks_two_sample",
                {},
                null_sample_size=8,
                null_draw="uniform",
                **level,
            ),
            (1, 2),
            (),
        ),
        (
            method_spec(
                "chi_square_homogeneity",
                {"categories": 2, "trials": 8},
                null_sample_size=8,
                null_draw="multinomial",
                **level,
            ),
            (1, 2),
            (),
        ),
        (
            method_spec(
                "chi_square_goodness_of_fit",
                {"categories": 2, "trials": 8},
                null_sample_size=8,
                null_draw="multinomial",
                **level,
            ),
            (1, 2),
            (),
        ),
        (
            method_spec(
                "mmd_permutation_test",
                {
                    "bandwidth": 1.0,
                    "permutations": 5,
                    "dimension": 2,
                    "null_mean": 0.0,
                    "null_scale": 1.0,
                },
                null_sample_size=4,
                null_draw="gaussian",
                **level,
            ),
            (1,),
            (),
        ),
        (
            method_spec(
                "classifier_two_sample_test",
                {
                    "permutations": 5,
                    "steps": 10,
                    "learning_rate": 0.1,
                    "l2": 0.01,
                    "dimension": 2,
                    "null_mean": 0.0,
                    "null_scale": 1.0,
                },
                null_sample_size=4,
                null_draw="gaussian",
                **level,
            ),
            (1,),
            (),
        ),
        (
            method_spec(
                "paired_bootstrap",
                bootstrap,
                required_checks=("null_false_alarm",),
                study="cpu_fast",
                null_sample_size=4,
                uncertainty_level=0.95,
                null_draw="gaussian",
                alpha=0.1,
                false_alarm_tolerance=1.0,
            ),
            (1,),
            (),
        ),
        (
            method_spec(
                "clustered_paired_bootstrap",
                {**bootstrap, "cluster_size": 2},
                required_checks=("null_false_alarm",),
                study="cpu_fast",
                null_sample_size=4,
                uncertainty_level=0.95,
                null_draw="gaussian",
                alpha=0.1,
                false_alarm_tolerance=1.0,
            ),
            (1,),
            (),
        ),
        (
            method_spec(
                "cusum",
                {"target": 0.0, "slack": 0.0, "threshold": 2.0, "direction": "increase"},
                null_draw="constant",
                **sequence,
            ),
            (1,),
            (),
        ),
        (
            method_spec(
                "adwin",
                {"delta": 0.05, "constant_value": 0.0},
                null_draw="constant",
                study="cpu_fast",
                null_sample_size=4,
                uncertainty_level=0.95,
                horizon=4,
                required_checks=("null_false_alarm",),
            ),
            (1,),
            (),
        ),
        (
            method_spec(
                "bounded_mean_cs",
                {"alpha": 0.05, "lower": 0.0, "upper": 1.0, "null_mean": 0.5},
                null_draw="constant",
                **sequence,
            ),
            (1,),
            (),
        ),
        (
            method_spec(
                "paired_difference_cs",
                {
                    "alpha": 0.05,
                    "lower": -1.0,
                    "upper": 1.0,
                    "null_mean": 0.0,
                    "pair_value": 1.0,
                },
                null_draw="constant",
                **sequence,
            ),
            (1,),
            (),
        ),
        (
            method_spec(
                "betting_e_detector",
                {
                    "null_mean": 0.5,
                    "alpha": 0.05,
                    "lower": 0.0,
                    "upper": 1.0,
                    "direction": "above",
                },
                null_draw="constant",
                **sequence,
            ),
            (1,),
            (),
        ),
        (
            method_spec(
                "harmful_shift",
                {"alpha": 0.05, "harm_margin": 0.0, "pair_value": 1.0},
                null_draw="constant",
                **sequence,
            ),
            (1,),
            (),
        ),
        (
            method_spec(
                "sequential_canary",
                {
                    "alpha": 0.05,
                    "harm_margin": 0.0,
                    "horizon_episodes": 4,
                    "pair_value": 1.0,
                },
                null_draw="constant",
                **sequence,
            ),
            (1,),
            (),
        ),
        (
            method_spec(
                "next_token_kl",
                {},
                required_checks=("reference",),
                study="cpu_fast",
                null_sample_size=0,
                uncertainty_level=0.95,
                null_draw="none",
            ),
            (),
            (identical,),
        ),
        (
            method_spec(
                "truncated_next_token_kl",
                {},
                required_checks=("reference",),
                study="cpu_fast",
                null_sample_size=0,
                uncertainty_level=0.95,
                null_draw="none",
            ),
            (),
            (identical,),
        ),
    ]
    return paths


def _kl(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.fsum(item * math.log(item / other) for item, other in zip(left, right, strict=True))


def _normal_quantile_check(probability: float) -> float:
    from llm_behavior_ci.experiments.validation import _normal_quantile

    return _normal_quantile(probability)


def _baseline_pairs(*, success: bool) -> tuple[BaselineOutcome, ...]:
    rows = []
    for index, scenario in enumerate(("s1", "s2")):
        rows.append(
            BaselineOutcome(
                "production",
                success,
                1.0 if success else 0.0,
                scenario,
                "mail",
                1,
                f"p{index}",
            )
        )
        rows.append(
            BaselineOutcome(
                "do_nothing",
                False,
                0.0,
                scenario,
                "mail",
                1,
                f"p{index}",
            )
        )
    return tuple(rows)


def _mixed_production() -> tuple[BaselineOutcome, ...]:
    rows = []
    for index, (scenario, app, success) in enumerate(
        (
            ("secret-scenario-mail-a", "mail", True),
            ("secret-scenario-mail-b", "mail", True),
            ("secret-scenario-bank-a", "bank", False),
            ("secret-scenario-bank-b", "bank", False),
        )
    ):
        rows.append(
            BaselineOutcome(
                "production",
                success,
                1.0 if success else 0.0,
                scenario,
                app,
                1 if app == "mail" else 2,
                f"p{index}",
            )
        )
        rows.append(
            BaselineOutcome(
                "do_nothing",
                False,
                0.0,
                scenario,
                app,
                1 if app == "mail" else 2,
                f"p{index}",
            )
        )
    return tuple(rows)


def _episode(
    run: RunIdentity,
    token: str,
    pair_id: str,
    role: str,
    success: bool,
    passed: int,
    total: int,
) -> EpisodeResult:
    started = _START
    step_at = _START + timedelta(seconds=1)
    ended = _START + timedelta(seconds=2)
    return EpisodeResult(
        episode=EpisodeIdentity(
            episode_id=f"{run.run_id}.{token}",
            run_id=run.run_id,
            pair_id=pair_id,
        ),
        run=run,
        task=LocalTaskRef("task-local", "scenario-local", "dev"),
        mode="execute",
        execution_seed=7,
        status="completed",
        started_at=started,
        ended_at=ended,
        model_steps=(
            ModelStep(
                index=0,
                prompt_text="plan the task",
                output_text="open the app",
                top_k_logprobs=(
                    (
                        TokenLogprob(token_id=3, logprob=-0.5, rank=0),
                        TokenLogprob(token_id=4, logprob=-1.0, rank=1),
                    ),
                ),
                generated_token_count=1,
                latency_seconds=0.1,
                started_at=step_at,
            ),
        ),
        tool_steps=(),
        plan_text=None,
        evaluator_outcome=EvaluatorOutcome(success, passed, total, 1),
        termination_reason="agent_stopped",
        episode_errors=(),
        role=role,
    )


if __name__ == "__main__":
    unittest.main()
