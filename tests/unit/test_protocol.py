from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from llm_behavior_ci.config import (
    CanarySettings,
    DistributionalMonitorSettings,
    GateSettings,
    MonitorSettings,
    RunConfiguration,
    StoppingRule,
    StreamSettings,
    TaskConfiguration,
    new_run_identity,
    run_configuration_hash,
)
from llm_behavior_ci.experiments.faults import (
    FaultPatch,
    FaultSpec,
    HarmLabel,
    apply_fault,
    load_fault,
)
from llm_behavior_ci.experiments.protocol import (
    ProtocolError,
    ProtocolLock,
    ProtocolSettings,
    TaskSelectionAllowance,
    admit_test_normal,
    authorize_faulted_candidate,
    authorize_gated_candidate,
    authorize_test_gated_candidate,
    bind_protocol,
    lock_protocol,
    require_protocol_lock,
    verify_runtime_bindings,
)
from llm_behavior_ci.experiments.validation import (
    AAComponent,
    AADependenceReport,
    ValidationReport,
    assemble_validation_report,
    method_spec,
    reference_case,
)
from llm_behavior_ci.lifecycle.offline_gate import PlanEvidenceInputs
from llm_behavior_ci.lifecycle.validation_artifact import (
    ValidationArtifact,
    build_validation_artifact,
)
from llm_behavior_ci.records import StatisticalEvidence
from llm_behavior_ci.tasks.selection import (
    TaskSet,
    canonical_task_set_bytes,
    task_set_hash_from_bytes,
)

_REVISION = "0123456789abcdef0123456789abcdef01234567"
_TOKENIZER_REVISION = "fedcba9876543210fedcba9876543210fedcba98"
_GIT_COMMIT = "a" * 40
_HASH_A = "1" * 64
_HASH_B = "2" * 64
_REPO = Path(__file__).resolve().parents[2]
_CATALOG = _REPO / "configs" / "faults"


def _make_task_set(*, split: str, task_count: int = 2) -> TaskSet:
    tasks = tuple(
        (f"task-{index}", f"scenario-{(index % 2) + 1}")
        for index in range(task_count)
    )
    payload = canonical_task_set_bytes(
        appworld_version="0.1.3.post1",
        split=split,
        selection_rule="fixed-v1",
        selection_seed=20260926,
        tasks=tasks,
    )
    digest = task_set_hash_from_bytes(payload)
    return TaskSet(
        appworld_version="0.1.3.post1",
        split=split,
        selection_rule="fixed-v1",
        selection_seed=20260926,
        task_count=task_count,
        scenario_count=len({scenario for _, scenario in tasks}),
        task_ids=tuple(task_id for task_id, _ in tasks),
        scenario_ids=tuple(scenario for _, scenario in tasks),
        task_set_hash=digest,
    )


_TRAIN_TASKS = _make_task_set(split="train")
_DEV_TASKS = _make_task_set(split="dev")
_TEST_TASKS = _make_task_set(split="test_normal")
_TRAIN_TASK_SET_HASH = _TRAIN_TASKS.task_set_hash
_DEV_TASK_SET_HASH = _DEV_TASKS.task_set_hash
_TEST_TASK_SET_HASH = _TEST_TASKS.task_set_hash


def _payload(
    *,
    split: str,
    task_set_hash: str,
    selection_rule: str = "fixed-v1",
    selection_seed: int = 20260926,
    task_count: int = 2,
) -> dict[str, object]:
    return {
        "model": {
            "model": {
                "repository": "Qwen/Qwen3-4B",
                "revision": _REVISION,
            },
            "tokenizer": {
                "repository": "Qwen/Qwen3-4B",
                "revision": _TOKENIZER_REVISION,
            },
            "quantization": {"method": "none"},
            "vllm_version": "0.30.0",
            "serving": {
                "dtype": "bfloat16",
                "max_model_len": 8192,
                "gpu_memory_utilization": 0.9,
                "max_num_seqs": 16,
                "max_num_batched_tokens": 8192,
                "kv_cache_dtype": "bfloat16",
                "enable_prefix_caching": False,
                "enable_chunked_prefill": False,
                "enforce_eager": False,
                "tensor_parallel_size": 1,
                "max_logprobs": 20,
                "batch_invariant": False,
                "sampler_backend": "native",
            },
        },
        "agent": {
            "smolagents_version": "1.22.0",
            "action_interface": "code",
            "prompt": {
                "prompt_version": "prompt-v1",
                "plan_format_version": "plan-v1",
                "thinking_enabled": False,
            },
            "step_limit": 40,
            "sampling": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 20,
                "min_p": 0.0,
                "seed": 17,
                "max_tokens": 512,
            },
        },
        "task": {
            "appworld_version": "0.1.3.post1",
            "split": split,
            "selection_rule": selection_rule,
            "selection_seed": selection_seed,
            "task_count": task_count,
            "task_set_hash": task_set_hash,
        },
        "run_seed": 7,
        "git_commit": _GIT_COMMIT,
        "protocol_hash": None,
    }


def _dev_config() -> RunConfiguration:
    return RunConfiguration.from_dict(
        _payload(split="dev", task_set_hash=_DEV_TASK_SET_HASH)
    )


def _test_normal_config() -> RunConfiguration:
    return RunConfiguration.from_dict(
        _payload(split="test_normal", task_set_hash=_TEST_TASK_SET_HASH)
    )


def _train_config() -> RunConfiguration:
    return RunConfiguration.from_dict(
        _payload(split="train", task_set_hash=_TRAIN_TASK_SET_HASH)
    )


def _aa(*, reason: str = "unavailable") -> AADependenceReport:
    return AADependenceReport(
        status="unavailable",
        evidence_accepted=False,
        provenance="unavailable",
        hardware_observed=False,
        observation_count=0,
        pair_count=0,
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
        memory_used_mib=None,
        wall_seconds=None,
        reason=reason,
    )


def _report(*, method: str, validated: bool) -> ValidationReport:
    return ValidationReport(
        method=method,
        implemented=True,
        validated=validated,
        benchmark_eligible=validated,
        calibration="none",
        study="cpu_fast",
        null_claim="type_i",
        null_draw="gaussian",
        input_hash=_HASH_A,
        seeds=(1,),
        sample_count=1,
        null_sample_size=1,
        uncertainty_level=0.95,
        alpha=None,
        horizon=None,
        false_alarm_tolerance=None,
        coverage_tolerance=None,
        parameters=(),
        required_checks=(),
        omitted_checks=(),
        checks=(),
        reference_agreements=(),
        aa=_aa(),
        libraries=(),
        configuration_hashes=(),
        gpu_evidence=False,
        gpu_floor_measured=False,
        split=None,
    )


def _harm_label(task_set_hash: str = _DEV_TASK_SET_HASH) -> HarmLabel:
    return HarmLabel(
        fault_version="sampling_temperature_one:1",
        base_configuration_hash=_HASH_A,
        candidate_configuration_hash=_HASH_B,
        task_set_hash=task_set_hash,
        effect_estimate=-0.2,
        interval_low=-0.3,
        interval_high=-0.1,
        margin=0.1,
        harmful=True,
        split="dev",
        confidence_level=0.9,
        resamples=25,
        seed=3,
    )


def _fault() -> FaultSpec:
    return load_fault(_CATALOG / "sampling_temperature_one.v1.json")


def _plan_evidence() -> PlanEvidenceInputs:
    return PlanEvidenceInputs(
        plan_format_version="plan-v1",
        plan_quality_features=("char_count",),
        plan_quality_weights=(1.0,),
        mmd_features=(),
        kl_approximation="top_k",
        required_statistics=("plan_quality",),
        validation_provenance="synthetic_fixture",
    )


def _gate() -> GateSettings:
    return GateSettings(
        confidence_level=0.9,
        bootstrap_resamples=40,
        score_margin=-0.02,
        kl_limit_nats=0.05,
        mmd_bandwidth=1.0,
        mmd_permutations=19,
        mmd_alpha=0.05,
        plan_format_version="plan-v1",
    )


def _canary() -> CanarySettings:
    return CanarySettings(
        fraction=0.1,
        outcome_delay_seconds=0.0,
        harm_margin=0.1,
        stopping_rule=StoppingRule(
            name="sequential_canary",
            alpha=0.05,
            horizon_episodes=20,
        ),
        metric_orientation="higher_is_better",
        promotion_policy="horizon_reached_without_harm",
    )


def _monitor() -> MonitorSettings:
    return MonitorSettings(
        reference_configuration_hash=_HASH_A,
        outcome_delay_seconds=0.0,
        signals=("task_success",),
        stopping_rules=(
            StoppingRule(
                name="cusum",
                alpha=0.05,
                horizon_episodes=50,
                threshold=0.5,
            ),
        ),
    )


def _stream() -> StreamSettings:
    return StreamSettings(
        split="dev",
        selection_rule="fixed-v1",
        selection_seed=20260926,
        task_set_hash=_DEV_TASK_SET_HASH,
        stream_seed=3,
        arrival_rate_per_second=1.0,
        concurrency=1,
        with_replacement=False,
        task_mix_rule="uniform",
    )


def _settings(
    *,
    configurations: tuple[RunConfiguration, ...] | None = None,
    task_selections: tuple[TaskConfiguration, ...] | None = None,
    harm_labels: tuple[HarmLabel, ...] | None = None,
    validation_reports: tuple[ValidationReport, ...] | None = None,
    faults: tuple[FaultSpec, ...] | None = None,
    plan_evidence: PlanEvidenceInputs | None = None,
) -> ProtocolSettings:
    train = _train_config()
    dev = _dev_config()
    test_normal = _test_normal_config()
    chosen = (
        configurations
        if configurations is not None
        else (train, dev, test_normal)
    )
    selections = (
        task_selections
        if task_selections is not None
        else (train.task, dev.task, test_normal.task)
    )
    labels = harm_labels if harm_labels is not None else (_harm_label(),)
    reports = (
        validation_reports
        if validation_reports is not None
        else (_report(method="cusum", validated=True),)
    )
    chosen_faults = faults if faults is not None else (_fault(),)
    chosen_plan_evidence = (
        plan_evidence if plan_evidence is not None else _plan_evidence()
    )
    return ProtocolSettings(
        configurations=chosen,
        task_selections=selections,
        harm_labels=labels,
        validation_reports=reports,
        faults=chosen_faults,
        plan_evidence=chosen_plan_evidence,
        gate=_gate(),
        canary=_canary(),
        monitor=_monitor(),
        stream=_stream(),
        analysis_version="protocol-test-v1",
        seeds=(7,),
    )


def _write_lock_bytes(path: Path, digest: str, payload: dict[str, object]) -> None:
    document = json.dumps(
        {"digest": digest, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    path.write_bytes(document)


def _assembled_mmd_report() -> ValidationReport:
    passed = replace(
        _aa(reason="measured"),
        status="passed",
        evidence_accepted=True,
        provenance="local_runtime",
        observation_count=12,
    )
    series = tuple(f"mmd:feature_{index}_fraction" for index in range(5))
    origin = [0.0] * 5
    identical = {"production": [origin] * 6, "candidate": [origin] * 6, "seed": 17}
    return assemble_validation_report(
        method_spec(
            "mmd_permutation_test",
            {"bandwidth": 1.0, "permutations": 19, "dimension": 5, "null_mean": 0.0, "null_scale": 1.0},
            required_checks=("null_false_alarm", "repeated_look", "reference", "aa_dependence"),
            study="simulation",
            null_sample_size=6,
            uncertainty_level=0.95,
            null_draw="gaussian",
            alpha=0.05,
            false_alarm_tolerance=0.5,
            repeated_look_stride=3,
        ),
        null_seed_blocks=((1, 2, 3), (4, 5, 6)),
        reference_cases=(
            reference_case("identical_mmd", "mmd_squared", 0.0, 1e-12, "closed_form", identical),
        ),
        aa_series=series,
        aa_components=tuple(
            AAComponent(
                series=name,
                input_hash=_HASH_B,
                configuration_hashes=(_HASH_A,),
                split="train",
                aa=passed,
            )
            for name in series
        ),
        configuration_hash=_HASH_A,
    )


class ProtocolLockTests(unittest.TestCase):
    def test_an_assembled_report_locks_with_its_components(self) -> None:
        report = _assembled_mmd_report()
        self.assertTrue(report.validated)
        settings = _settings(validation_reports=(report,))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "assembled.lock.json"
            lock_protocol(settings, path)
            required = require_protocol_lock(path)
        (stored,) = required.payload["validation_reports"]
        self.assertEqual(required.method_names, ("mmd_permutation_test",))
        self.assertEqual(stored["null_seed_blocks"], [3, 3])
        self.assertEqual(
            [item["series"] for item in stored["aa_components"]],
            [f"mmd:feature_{index}_fraction" for index in range(5)],
        )
        self.assertEqual(stored["aa"]["status"], "passed")

    def test_deterministic_digest_and_bytes(self) -> None:
        settings = _settings()
        with tempfile.TemporaryDirectory() as temporary:
            first_path = Path(temporary) / "first.lock.json"
            second_path = Path(temporary) / "second.lock.json"
            first = lock_protocol(settings, first_path)
            second = lock_protocol(settings, second_path)
            self.assertEqual(first.digest, second.digest)
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
            self.assertEqual(first.method_names, ("cusum",))
            self.assertEqual(first.validated_flags, (True,))
            self.assertEqual(
                set(first.task_set_hashes),
                {
                    _TRAIN_TASK_SET_HASH,
                    _DEV_TASK_SET_HASH,
                    _TEST_TASK_SET_HASH,
                },
            )

    def test_mutation_detection(self) -> None:
        settings = _settings()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock.json"
            lock = lock_protocol(settings, path)
            raw = bytearray(path.read_bytes())
            index = raw.find(b"protocol-test-v1")
            self.assertNotEqual(index, -1)
            raw[index] = ord("Q") if raw[index] != ord("Q") else ord("Z")
            path.write_bytes(bytes(raw))
            with self.assertRaises(ProtocolError):
                require_protocol_lock(path)

            payload = copy.deepcopy(dict(lock.payload))
            payload["analysis_version"] = "mutated-version"
            _write_lock_bytes(path, lock.digest, payload)
            with self.assertRaises(ProtocolError):
                require_protocol_lock(path)

    def test_non_circular_binding(self) -> None:
        settings = _settings()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock.json"
            lock = lock_protocol(settings, path)
            template = _test_normal_config()
            self.assertIsNone(template.protocol_hash)
            bound = bind_protocol(template, lock)
            self.assertEqual(bound.protocol_hash, lock.digest)
            self.assertNotEqual(
                run_configuration_hash(bound),
                run_configuration_hash(template),
            )
            restored = bound.to_dict()
            restored["protocol_hash"] = None
            self.assertEqual(restored, template.to_dict())
            payload_digest = hashlib.sha256(
                json.dumps(
                    lock.payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            self.assertEqual(lock.digest, payload_digest)
            for configuration in lock.configurations:
                self.assertIsNone(configuration.protocol_hash)

    def test_missing_prerequisites_write_no_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock.json"
            with self.assertRaises(ProtocolError):
                ProtocolSettings(
                    configurations=(_dev_config(),),
                    task_selections=(_dev_config().task,),
                    harm_labels=(),
                    validation_reports=(_report(method="cusum", validated=True),),
                    faults=(_fault(),),
                    plan_evidence=_plan_evidence(),
                    gate=_gate(),
                    canary=_canary(),
                    monitor=_monitor(),
                    stream=_stream(),
                    analysis_version="protocol-test-v1",
                    seeds=(7,),
                )
            self.assertFalse(path.exists())

            with self.assertRaises(ProtocolError):
                ProtocolSettings(
                    configurations=(_dev_config(),),
                    task_selections=(_dev_config().task,),
                    harm_labels=(_harm_label(),),
                    validation_reports=(),
                    faults=(_fault(),),
                    plan_evidence=_plan_evidence(),
                    gate=_gate(),
                    canary=_canary(),
                    monitor=_monitor(),
                    stream=_stream(),
                    analysis_version="protocol-test-v1",
                    seeds=(7,),
                )
            self.assertFalse(path.exists())

            already_bound = RunConfiguration.from_dict(
                {
                    **_payload(split="dev", task_set_hash=_DEV_TASK_SET_HASH),
                    "protocol_hash": "e" * 64,
                }
            )
            with self.assertRaises(ProtocolError):
                ProtocolSettings(
                    configurations=(already_bound,),
                    task_selections=(already_bound.task,),
                    harm_labels=(_harm_label(),),
                    validation_reports=(_report(method="cusum", validated=True),),
                    faults=(_fault(),),
                    plan_evidence=_plan_evidence(),
                    gate=_gate(),
                    canary=_canary(),
                    monitor=_monitor(),
                    stream=_stream(),
                    analysis_version="protocol-test-v1",
                    seeds=(7,),
                )
            self.assertFalse(path.exists())

            with self.assertRaises(ProtocolError):
                ProtocolSettings(
                    configurations=(_dev_config(),),
                    task_selections=(_dev_config().task,),
                    harm_labels=(_harm_label(task_set_hash="f" * 64),),
                    validation_reports=(_report(method="cusum", validated=True),),
                    faults=(_fault(),),
                    plan_evidence=_plan_evidence(),
                    gate=_gate(),
                    canary=_canary(),
                    monitor=_monitor(),
                    stream=_stream(),
                    analysis_version="protocol-test-v1",
                    seeds=(7,),
                )
            self.assertFalse(path.exists())

            with self.assertRaises(ProtocolError):
                lock_protocol(
                    _settings(
                        validation_reports=(
                            _report(method="cusum", validated=False),
                        )
                    ),
                    path,
                )
            self.assertFalse(path.exists())

    def test_final_test_rejection(self) -> None:
        settings = _settings()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock.json"
            missing = Path(temporary) / "missing.lock.json"
            with self.assertRaises(ProtocolError):
                require_protocol_lock(missing)

            lock = lock_protocol(settings, path)
            bound = bind_protocol(_test_normal_config(), lock)

            mutated = bytearray(path.read_bytes())
            mutated[0] = mutated[0] ^ 0x01
            path.write_bytes(bytes(mutated))
            with self.assertRaises(ProtocolError):
                require_protocol_lock(path)

            lock = lock_protocol(settings, path)
            bound = bind_protocol(_test_normal_config(), lock)
            changed = bound.to_dict()
            changed["agent"]["sampling"]["temperature"] = 1.0
            mismatched = RunConfiguration.from_dict(changed)
            with self.assertRaises(ProtocolError):
                admit_test_normal(
                    lock,
                    mismatched,
                    task_set_hash=_TEST_TASK_SET_HASH,
                )

            with self.assertRaises(ProtocolError):
                admit_test_normal(
                    lock,
                    bound,
                    task_set_hash=_DEV_TASK_SET_HASH,
                )

            payload = copy.deepcopy(dict(lock.payload))
            payload["validation_reports"][0]["validated"] = False
            payload["validation_reports"][0]["benchmark_eligible"] = False
            digest = hashlib.sha256(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            bad_path = Path(temporary) / "unvalidated.lock.json"
            _write_lock_bytes(bad_path, digest, payload)
            with self.assertRaises(ProtocolError):
                require_protocol_lock(bad_path)
            unvalidated_lock = ProtocolLock(digest=digest, payload=payload)
            bound_unvalidated = bind_protocol(
                _test_normal_config(),
                unvalidated_lock,
            )
            with self.assertRaises(ProtocolError):
                admit_test_normal(
                    unvalidated_lock,
                    bound_unvalidated,
                    task_set_hash=_TEST_TASK_SET_HASH,
                )

            dev_bound = bind_protocol(_dev_config(), lock)
            with self.assertRaises(ProtocolError):
                admit_test_normal(
                    lock,
                    dev_bound,
                    task_set_hash=_DEV_TASK_SET_HASH,
                )

    def test_admit_test_normal_success(self) -> None:
        settings = _settings()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock.json"
            lock = lock_protocol(settings, path)
            loaded = require_protocol_lock(path)
            bound = bind_protocol(_test_normal_config(), loaded)
            self.assertIsNone(
                admit_test_normal(
                    loaded,
                    bound,
                    task_set_hash=_TEST_TASK_SET_HASH,
                )
            )


def _gate_statistics(reference: RunConfiguration, candidate: RunConfiguration) -> tuple[StatisticalEvidence, ...]:
    return (
        StatisticalEvidence(
            method="plan_quality_bootstrap",
            split=reference.task.split,
            configuration_hash=run_configuration_hash(candidate),
            estimate=0.05,
            sample_size=2,
            unit="score_delta",
            reference_configuration_hash=run_configuration_hash(reference),
        ),
    )


def _artifact(
    *,
    reference: RunConfiguration,
    candidate: RunConfiguration,
    task_set_hash: str,
    outcome: str = "PASS",
    reason_codes: tuple[str, ...] = (),
    validation_provenance: str = "validated",
) -> ValidationArtifact:
    return build_validation_artifact(
        outcome=outcome,
        reason_codes=reason_codes,
        reference=reference,
        candidate=candidate,
        reference_run=new_run_identity(reference),
        candidate_run=new_run_identity(candidate),
        task_set_hash=task_set_hash,
        task_split=reference.task.split,
        statistics=(
            _gate_statistics(reference, candidate) if outcome == "PASS" else ()
        ),
        evidence_source=(
            "synthetic_fixture"
            if validation_provenance == "synthetic_fixture"
            else "gate_run"
        ),
        created_at=datetime.now(timezone.utc),
    )


def _task_selection_allowance() -> TaskSelectionAllowance:
    return TaskSelectionAllowance(
        allowed_leaves=frozenset(
            {
                "task.split",
                "task.selection_rule",
                "task.selection_seed",
                "task.task_count",
                "task.task_set_hash",
            }
        ),
        train_task_set_hash=_TRAIN_TASK_SET_HASH,
        train_values={
            "task.split": "train",
            "task.selection_rule": "fixed-v1",
            "task.selection_seed": 20260926,
            "task.task_count": 2,
            "task.task_set_hash": _TRAIN_TASK_SET_HASH,
        },
    )


class AuthorizeGatedCandidateTests(unittest.TestCase):
    def test_legal_task_selection_difference_accepted(self) -> None:
        train_reference = _train_config()
        train_candidate = RunConfiguration.from_dict(
            {
                **_payload(split="train", task_set_hash=_TRAIN_TASK_SET_HASH),
                "agent": {
                    **_payload(split="train", task_set_hash=_TRAIN_TASK_SET_HASH)[
                        "agent"
                    ],
                    "step_limit": 4,
                },
            }
        )
        artifact = _artifact(
            reference=train_reference,
            candidate=train_candidate,
            task_set_hash=_TRAIN_TASK_SET_HASH,
        )
        served_reference = _test_normal_config()
        served_candidate = RunConfiguration.from_dict(
            {
                **_payload(
                    split="test_normal",
                    task_set_hash=_TEST_TASK_SET_HASH,
                ),
                "agent": {
                    **_payload(
                        split="test_normal",
                        task_set_hash=_TEST_TASK_SET_HASH,
                    )["agent"],
                    "step_limit": 4,
                },
            }
        )
        admission = authorize_gated_candidate(
            artifact,
            served_reference,
            served_candidate,
            allowance=_task_selection_allowance(),
        )
        self.assertEqual(admission.outcome, "PASS")
        self.assertEqual(
            admission.reference_configuration_hash,
            run_configuration_hash(train_reference),
        )
        self.assertEqual(
            admission.candidate_configuration_hash,
            run_configuration_hash(train_candidate),
        )
        self.assertEqual(admission.train_task_set_hash, _TRAIN_TASK_SET_HASH)
        self.assertEqual(
            admission.served_candidate_configuration_hash,
            run_configuration_hash(served_candidate),
        )
        self.assertNotEqual(
            admission.served_candidate_configuration_hash,
            admission.candidate_configuration_hash,
        )
        self.assertEqual(admission.evidence_artifact_id, artifact.artifact_id)

    def test_sampling_change_rejected_with_legal_task_diff(self) -> None:
        train_reference = _train_config()
        train_candidate = train_reference
        artifact = _artifact(
            reference=train_reference,
            candidate=train_candidate,
            task_set_hash=_TRAIN_TASK_SET_HASH,
        )
        served_reference = _test_normal_config()
        changed = _test_normal_config().to_dict()
        changed["agent"]["sampling"]["temperature"] = 1.0
        served_candidate = RunConfiguration.from_dict(changed)
        with self.assertRaises(ProtocolError) as ctx:
            authorize_gated_candidate(
                artifact,
                served_reference,
                served_candidate,
                allowance=_task_selection_allowance(),
            )
        self.assertIn("does not match the gate", str(ctx.exception))

    def test_prompt_change_rejected_with_legal_task_diff(self) -> None:
        train_reference = _train_config()
        artifact = _artifact(
            reference=train_reference,
            candidate=train_reference,
            task_set_hash=_TRAIN_TASK_SET_HASH,
        )
        served_reference = _test_normal_config()
        changed = _test_normal_config().to_dict()
        changed["agent"]["prompt"]["prompt_version"] = "prompt-no-api-guidance"
        served_candidate = RunConfiguration.from_dict(changed)
        with self.assertRaises(ProtocolError):
            authorize_gated_candidate(
                artifact,
                served_reference,
                served_candidate,
                allowance=_task_selection_allowance(),
            )

    def test_manufactured_pass_with_swapped_hashes_rejected(self) -> None:
        train_reference = _train_config()
        train_candidate = RunConfiguration.from_dict(
            {
                **_payload(split="train", task_set_hash=_TRAIN_TASK_SET_HASH),
                "agent": {
                    **_payload(split="train", task_set_hash=_TRAIN_TASK_SET_HASH)[
                        "agent"
                    ],
                    "step_limit": 4,
                },
            }
        )
        artifact = _artifact(
            reference=train_candidate,
            candidate=train_reference,
            task_set_hash=_TRAIN_TASK_SET_HASH,
        )
        with self.assertRaises(ProtocolError):
            authorize_gated_candidate(
                artifact,
                train_reference,
                train_candidate,
            )

    def test_free_form_gate_document_rejected(self) -> None:
        train_reference = _train_config()

        class _GateView:
            outcome = "PASS"
            reason_codes: tuple[str, ...] = ()
            reference_configuration_hash = run_configuration_hash(train_reference)
            candidate_configuration_hash = run_configuration_hash(train_reference)
            task_set_hash = _TRAIN_TASK_SET_HASH
            reference_protocol_hash = None
            candidate_protocol_hash = None

        with self.assertRaises(ProtocolError) as ctx:
            authorize_gated_candidate(_GateView(), train_reference, train_reference)
        self.assertIn("ValidationArtifact", str(ctx.exception))

    def test_failed_gate_rejected(self) -> None:
        train_reference = _train_config()
        artifact = _artifact(
            reference=train_reference,
            candidate=train_reference,
            task_set_hash=_TRAIN_TASK_SET_HASH,
            outcome="BLOCK",
            reason_codes=("plan_quality_margin",),
        )
        with self.assertRaises(ProtocolError) as ctx:
            authorize_gated_candidate(artifact, train_reference, train_reference)
        self.assertIn("PASS", str(ctx.exception))

    def test_synthetic_fixture_evidence_rejected_by_real_admission(self) -> None:
        train_reference = _train_config()
        synthetic = _artifact(
            reference=train_reference,
            candidate=train_reference,
            task_set_hash=_TRAIN_TASK_SET_HASH,
            validation_provenance="synthetic_fixture",
        )
        self.assertEqual(synthetic.evidence_source, "synthetic_fixture")
        with self.assertRaises(ProtocolError) as ctx:
            authorize_gated_candidate(synthetic, train_reference, train_reference)
        self.assertIn("synthetic_fixture", str(ctx.exception))

        admission = authorize_test_gated_candidate(
            synthetic,
            train_reference,
            train_reference,
        )
        self.assertEqual(admission.outcome, "PASS")
        self.assertEqual(admission.evidence_artifact_id, synthetic.artifact_id)

    def test_malformed_artifact_rejected(self) -> None:
        train_reference = _train_config()
        artifact = _artifact(
            reference=train_reference,
            candidate=train_reference,
            task_set_hash=_TRAIN_TASK_SET_HASH,
        )
        malformed_payload = dict(artifact.to_dict())
        malformed_payload["candidate_configuration_hash"] = "not-a-hash"
        with self.assertRaises(Exception):
            ValidationArtifact.from_dict(malformed_payload)

    def test_missing_artifact_rejected(self) -> None:
        train_reference = _train_config()
        with self.assertRaises(ProtocolError):
            authorize_gated_candidate(None, train_reference, train_reference)


class AuthorizeFaultedCandidateTests(unittest.TestCase):
    def test_declared_fault_admitted(self) -> None:
        settings = _settings()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock.json"
            lock = lock_protocol(settings, path)
            reference = bind_protocol(_test_normal_config(), lock)
            fault = load_fault(_CATALOG / "sampling_temperature_one.v1.json")
            candidate = apply_fault(reference, fault)
            admission = authorize_faulted_candidate(
                lock,
                reference,
                candidate,
                fault,
                _TEST_TASKS,
            )
            self.assertEqual(admission.fault_version, fault.fault_version)
            self.assertEqual(
                admission.candidate_configuration_hash,
                run_configuration_hash(candidate),
            )
            self.assertEqual(admission.task_set_hash, _TEST_TASK_SET_HASH)

    def test_undeclared_patch_candidate_rejected(self) -> None:
        settings = _settings()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock.json"
            lock = lock_protocol(settings, path)
            reference = bind_protocol(_test_normal_config(), lock)
            fault = load_fault(_CATALOG / "sampling_temperature_one.v1.json")
            other = apply_fault(
                reference,
                load_fault(_CATALOG / "step_limit_reduced.v1.json"),
            )
            with self.assertRaises(ProtocolError) as ctx:
                authorize_faulted_candidate(
                    lock,
                    reference,
                    other,
                    fault,
                    _TEST_TASKS,
                )
            self.assertIn("declared fault", str(ctx.exception))

    def test_unnamed_fault_rejected(self) -> None:
        settings = _settings()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock.json"
            lock = lock_protocol(settings, path)
            reference = bind_protocol(_test_normal_config(), lock)
            fault = load_fault(_CATALOG / "step_limit_reduced.v1.json")
            candidate = apply_fault(reference, fault)
            with self.assertRaises(ProtocolError) as ctx:
                authorize_faulted_candidate(
                    lock,
                    reference,
                    candidate,
                    fault,
                    _TEST_TASKS,
                )
            self.assertIn("not named", str(ctx.exception))

    def test_locked_fault_list_rejects_kind_mismatch(self) -> None:
        settings = _settings()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock.json"
            lock = lock_protocol(settings, path)
            payload = copy.deepcopy(dict(lock.payload))
            payload["faults"] = [
                {
                    "fault_id": "sampling_temperature_one",
                    "version": "1",
                    "kind": "sampling",
                    "patches": [
                        {
                            "path": "agent.sampling.temperature",
                            "value": 1.0,
                        }
                    ],
                }
            ]
            digest = hashlib.sha256(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            listed = ProtocolLock(digest=digest, payload=payload)
            reference = bind_protocol(_test_normal_config(), listed)
            fault = FaultSpec(
                fault_id="sampling_temperature_one",
                version="1",
                kind="token_limit",
                patches=(FaultPatch("agent.sampling.max_tokens", 16),),
            )
            with self.assertRaises(ProtocolError) as ctx:
                authorize_faulted_candidate(
                    listed,
                    reference,
                    apply_fault(reference, fault),
                    fault,
                    _TEST_TASKS,
                )
            self.assertIn("kind", str(ctx.exception))

    def test_absent_faults_key_no_longer_bypasses_the_lock(self) -> None:
        """A lock built without ``ProtocolSettings`` can no longer omit ``faults``.

        Guards the fix for the bypass this task closes: before it,
        ``_authorize_fault_against_lock`` returned early on a missing
        ``faults`` key, so any fault named only in ``harm_labels`` was
        admitted with no check that its patches, kind, or control matched
        anything declared. A hand-built lock without the key must now be
        refused outright, the same way a malformed lock is.
        """

        settings = _settings()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock.json"
            lock = lock_protocol(settings, path)
            payload = copy.deepcopy(dict(lock.payload))
            del payload["faults"]
            digest = hashlib.sha256(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            bare = ProtocolLock(digest=digest, payload=payload)
            reference = bind_protocol(_test_normal_config(), bare)
            fault = _fault()
            candidate = apply_fault(reference, fault)
            with self.assertRaises(ProtocolError) as ctx:
                authorize_faulted_candidate(
                    bare,
                    reference,
                    candidate,
                    fault,
                    _TEST_TASKS,
                )
            self.assertIn("faults", str(ctx.exception))


class ProtocolSettingsFaultBindingTests(unittest.TestCase):
    def test_faults_must_be_a_non_empty_tuple(self) -> None:
        with self.assertRaises(ProtocolError) as ctx:
            _settings(faults=())
        self.assertIn("faults", str(ctx.exception))

    def test_duplicate_fault_version_rejected(self) -> None:
        with self.assertRaises(ProtocolError) as ctx:
            _settings(faults=(_fault(), _fault()))
        self.assertIn("duplicate fault_version", str(ctx.exception))

    def test_harm_label_must_match_a_declared_fault(self) -> None:
        other_fault = load_fault(_CATALOG / "step_limit_reduced.v1.json")
        with self.assertRaises(ProtocolError) as ctx:
            _settings(faults=(other_fault,))
        self.assertIn("matches no declared fault", str(ctx.exception))

    def test_locked_faults_round_trip_through_the_lock(self) -> None:
        settings = _settings()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock.json"
            lock = lock_protocol(settings, path)
            stored = lock.payload["faults"]
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0]["fault_id"], "sampling_temperature_one")

    def test_distributional_monitors_default_empty_and_round_trip_when_set(self) -> None:
        settings = _settings()
        self.assertEqual(settings.distributional_monitors, ())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock.json"
            lock = lock_protocol(settings, path)
            self.assertEqual(lock.payload["distributional_monitors"], [])

        entry = DistributionalMonitorSettings(
            signal="tool_selection",
            reference_counts=(("calendar.lookup", 3), ("venmo.pay", 1)),
            window_episodes=10,
            alpha=0.05,
            correction="bonferroni",
        )
        with_entry = _settings()
        with_entry = ProtocolSettings(
            configurations=with_entry.configurations,
            task_selections=with_entry.task_selections,
            harm_labels=with_entry.harm_labels,
            validation_reports=with_entry.validation_reports,
            gate=with_entry.gate,
            canary=with_entry.canary,
            monitor=with_entry.monitor,
            stream=with_entry.stream,
            analysis_version=with_entry.analysis_version,
            seeds=with_entry.seeds,
            faults=with_entry.faults,
            plan_evidence=with_entry.plan_evidence,
            distributional_monitors=(entry,),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock.json"
            lock = lock_protocol(with_entry, path)
            stored = lock.payload["distributional_monitors"]
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0]["signal"], "tool_selection")
            self.assertEqual(stored[0]["window_episodes"], 10)

    def test_duplicate_distributional_monitor_signal_rejected(self) -> None:
        entry = DistributionalMonitorSettings(
            signal="task_mix",
            reference_counts=(("normal", 1),),
            window_episodes=5,
            alpha=0.05,
            correction="none",
        )
        base = _settings()
        with self.assertRaises(ProtocolError) as ctx:
            ProtocolSettings(
                configurations=base.configurations,
                task_selections=base.task_selections,
                harm_labels=base.harm_labels,
                validation_reports=base.validation_reports,
                gate=base.gate,
                canary=base.canary,
                monitor=base.monitor,
                stream=base.stream,
                analysis_version=base.analysis_version,
                seeds=base.seeds,
                faults=base.faults,
                plan_evidence=base.plan_evidence,
                distributional_monitors=(entry, entry),
            )
        self.assertIn("duplicate distributional monitor signal", str(ctx.exception))


class RuntimeBindingDriftTests(unittest.TestCase):
    def _locked(self) -> tuple[ProtocolLock, Path]:
        settings = _settings()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "lock.json"
        lock = lock_protocol(settings, path)
        return lock, path

    def _relocked_with(self, lock: ProtocolLock, **overrides: object) -> ProtocolLock:
        payload = copy.deepcopy(dict(lock.payload))
        payload.update(overrides)
        digest = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        return ProtocolLock(digest=digest, payload=payload)

    def test_stale_plan_feature_schema_version_fails_closed(self) -> None:
        lock, _path = self._locked()
        stale = self._relocked_with(
            lock,
            plan_feature_schema_version="plan-features-v0",
        )
        with self.assertRaises(ProtocolError) as ctx:
            verify_runtime_bindings(stale)
        self.assertIn("plan_feature_schema_version", str(ctx.exception))

    def test_stale_gate_validation_artifact_schema_version_fails_closed(self) -> None:
        lock, _path = self._locked()
        stale = self._relocked_with(
            lock,
            gate_validation_artifact_schema_version="gate-validation-artifact-v0",
        )
        with self.assertRaises(ProtocolError) as ctx:
            verify_runtime_bindings(stale)
        self.assertIn("gate_validation_artifact_schema_version", str(ctx.exception))

    def test_edited_prompt_template_content_fails_closed(self) -> None:
        lock, _path = self._locked()
        edited_hashes = [dict(item) for item in lock.payload["prompt_template_hashes"]]
        self.assertTrue(edited_hashes)
        edited_hashes[0]["content_hash"] = "0" * 64
        stale = self._relocked_with(lock, prompt_template_hashes=edited_hashes)
        with self.assertRaises(ProtocolError) as ctx:
            verify_runtime_bindings(stale)
        self.assertIn("prompt template content changed", str(ctx.exception))

    def test_require_protocol_lock_fails_closed_on_drift(self) -> None:
        lock, path = self._locked()
        stale = self._relocked_with(
            lock,
            plan_feature_schema_version="plan-features-v0",
        )
        document = json.dumps(
            {"digest": stale.digest, "payload": stale.payload},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        path.write_bytes(document)
        with self.assertRaises(ProtocolError) as ctx:
            require_protocol_lock(path)
        self.assertIn("plan_feature_schema_version", str(ctx.exception))

    def test_admit_test_normal_fails_closed_on_drift(self) -> None:
        lock, _path = self._locked()
        reference = bind_protocol(_test_normal_config(), lock)
        stale = self._relocked_with(
            lock,
            plan_feature_schema_version="plan-features-v0",
        )
        with self.assertRaises(ProtocolError) as ctx:
            admit_test_normal(
                stale,
                reference,
                task_set_hash=_TEST_TASK_SET_HASH,
            )
        self.assertIn("plan_feature_schema_version", str(ctx.exception))

    def test_unregistered_prompt_version_refuses_to_lock(self) -> None:
        mutated = replace(
            _train_config(),
            agent=replace(
                _train_config().agent,
                prompt=replace(
                    _train_config().agent.prompt,
                    prompt_version="prompt-does-not-exist",
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock.json"
            with self.assertRaises(ProtocolError):
                lock_protocol(
                    _settings(configurations=(mutated, _dev_config(), _test_normal_config())),
                    path,
                )
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
