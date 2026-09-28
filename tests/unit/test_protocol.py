from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from llm_behavior_ci.config import (
    CanarySettings,
    GateSettings,
    MonitorSettings,
    RunConfiguration,
    StoppingRule,
    StreamSettings,
    TaskConfiguration,
    run_configuration_hash,
)
from llm_behavior_ci.experiments.faults import HarmLabel
from llm_behavior_ci.experiments.protocol import (
    ProtocolError,
    ProtocolLock,
    ProtocolSettings,
    admit_test_normal,
    bind_protocol,
    lock_protocol,
    require_protocol_lock,
)
from llm_behavior_ci.experiments.validation import (
    AADependenceReport,
    ValidationReport,
)

_REVISION = "0123456789abcdef0123456789abcdef01234567"
_TOKENIZER_REVISION = "fedcba9876543210fedcba9876543210fedcba98"
_GIT_COMMIT = "a" * 40
_DEV_TASK_SET_HASH = "c" * 64
_TEST_TASK_SET_HASH = "d" * 64
_HASH_A = "1" * 64
_HASH_B = "2" * 64


def _payload(*, split: str, task_set_hash: str) -> dict[str, object]:
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
            "selection_rule": "fixed-v1",
            "selection_seed": 20260926,
            "task_count": 50,
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
) -> ProtocolSettings:
    dev = _dev_config()
    test_normal = _test_normal_config()
    chosen = configurations if configurations is not None else (dev, test_normal)
    selections = (
        task_selections
        if task_selections is not None
        else (dev.task, test_normal.task)
    )
    labels = harm_labels if harm_labels is not None else (_harm_label(),)
    reports = (
        validation_reports
        if validation_reports is not None
        else (_report(method="cusum", validated=True),)
    )
    return ProtocolSettings(
        configurations=chosen,
        task_selections=selections,
        harm_labels=labels,
        validation_reports=reports,
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


class ProtocolLockTests(unittest.TestCase):
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
                {_DEV_TASK_SET_HASH, _TEST_TASK_SET_HASH},
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


if __name__ == "__main__":
    unittest.main()
