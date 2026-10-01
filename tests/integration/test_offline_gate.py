from __future__ import annotations

import copy
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

from llm_behavior_ci.config import GateSettings, RunConfiguration
from llm_behavior_ci.lifecycle.offline_gate import (
    PlanEvidenceInputs,
    plan_evidence_to_dict,
    run_offline_gate,
)
from llm_behavior_ci.records import TokenLogprob, assert_public_payload
from llm_behavior_ci.runtime.agent import AgentTurn
from llm_behavior_ci.runtime.appworld import EvaluationResult, TaskContext, ToolResult
from llm_behavior_ci.runtime.episode import RuntimeDependencies
from llm_behavior_ci.tasks.selection import (
    TaskSet,
    canonical_task_set_bytes,
    task_set_hash_from_bytes,
)


def _load_cli_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_offline_gate.py"
    spec = importlib.util.spec_from_file_location("run_offline_gate_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_cli_main():
    return _load_cli_module().main

_START = datetime(2026, 9, 27, 15, 0, tzinfo=timezone.utc)
_PLAN = "1. open the calendar"
_LOGPROBS = (
    (
        TokenLogprob(token_id=7, logprob=-0.2, rank=0),
        TokenLogprob(token_id=9, logprob=-1.5, rank=1),
    ),
)


def _payload() -> dict[str, object]:
    return {
        "model": {
            "model": {
                "repository": "Qwen/Qwen3-4B",
                "revision": "0123456789abcdef0123456789abcdef01234567",
            },
            "tokenizer": {
                "repository": "Qwen/Qwen3-4B",
                "revision": "fedcba9876543210fedcba9876543210fedcba98",
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
            "split": "train",
            "selection_rule": "deterministic_sample",
            "selection_seed": 20260926,
            "task_count": 50,
            "task_set_hash": "c" * 64,
        },
        "run_seed": 7,
        "git_commit": "a" * 40,
        "protocol_hash": "e" * 64,
    }


def _task_set() -> TaskSet:
    tasks = (("task-a", "scenario-1"), ("task-b", None))
    payload = canonical_task_set_bytes(
        appworld_version="0.1.3.post1",
        split="train",
        selection_rule="deterministic_sample",
        selection_seed=20260926,
        tasks=tasks,
    )
    return TaskSet(
        appworld_version="0.1.3.post1",
        split="train",
        selection_rule="deterministic_sample",
        selection_seed=20260926,
        task_count=2,
        scenario_count=2,
        task_ids=("task-a", "task-b"),
        scenario_ids=("scenario-1", None),
        task_set_hash=task_set_hash_from_bytes(payload),
    )


def _config(task_set: TaskSet, *, run_seed: int = 7) -> RunConfiguration:
    payload = copy.deepcopy(_payload())
    payload["run_seed"] = run_seed
    payload["task"] = {
        "appworld_version": task_set.appworld_version,
        "split": task_set.split,
        "selection_rule": task_set.selection_rule,
        "selection_seed": task_set.selection_seed,
        "task_count": task_set.task_count,
        "task_set_hash": task_set.task_set_hash,
    }
    return RunConfiguration.from_dict(payload)


def _settings(**overrides: object) -> GateSettings:
    values: dict[str, object] = {
        "confidence_level": 0.9,
        "bootstrap_resamples": 40,
        "score_margin": -0.02,
        "kl_limit_nats": 0.05,
        "mmd_bandwidth": 1.0,
        "mmd_permutations": 19,
        "mmd_alpha": 0.05,
        "plan_format_version": "plan-v1",
    }
    values.update(overrides)
    return GateSettings(**values)


def _evidence(**overrides: object) -> PlanEvidenceInputs:
    values: dict[str, object] = {
        "plan_format_version": "plan-v1",
        "plan_quality_features": ("numbered_step_count", "token_count"),
        "plan_quality_weights": (1.0, 0.1),
        "mmd_features": ("char_count", "numbered_step_count", "model_step_count"),
        "kl_approximation": "top_k",
        "required_statistics": ("plan_quality", "kl", "mmd"),
        "validation_provenance": "synthetic_fixture",
    }
    values.update(overrides)
    return PlanEvidenceInputs(**values)


class FakeSession:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id

    def context(self) -> TaskContext:
        return TaskContext(
            task_id=self.task_id,
            instruction="solve the task",
            api_documentation="calendar docs",
        )

    def execute(self, action: str) -> ToolResult:
        del action
        raise AssertionError("plan mode must not execute tools")

    def evaluate(self) -> EvaluationResult:
        raise AssertionError("plan mode must not evaluate")

    def close(self) -> None:
        return None


class PlanAgent:
    def __init__(self, clock) -> None:
        self._clock = clock
        self._context = None
        self.config = None

    def begin(self, context: TaskContext, config: RunConfiguration) -> None:
        self._context = context
        self.config = config

    def messages(self, tool_output: str | None = None) -> list[dict[str, str]]:
        del tool_output
        assert self._context is not None
        return [
            {"role": "system", "content": "Emit a plan."},
            {
                "role": "user",
                "content": (
                    f"{self._context.instruction}\n"
                    f"{self._context.api_documentation}"
                ),
            },
        ]

    def next_turn(self, *, tool_output: str | None) -> AgentTurn:
        del tool_output
        return AgentTurn(
            prompt_text="plan the next action",
            output_text=_PLAN,
            top_k_logprobs=_LOGPROBS,
            generated_token_count=len(_LOGPROBS),
            latency_seconds=0.1,
            started_at=self._clock(),
            action=None,
            app_name=None,
            api_name=None,
        )

    def teacher_force_plan(
        self,
        *,
        messages: list[dict[str, str]],
        plan_text: str,
    ) -> tuple[tuple[TokenLogprob, ...], ...]:
        del messages, plan_text
        return _LOGPROBS


def _clock():
    current = _START

    def tick() -> datetime:
        nonlocal current
        value = current
        current = current + timedelta(seconds=1)
        return value

    return tick


def _runtime() -> RuntimeDependencies:
    clock = _clock()
    return RuntimeDependencies(
        session_factory=lambda task_id: FakeSession(task_id),
        agent=PlanAgent(clock),
        clock=clock,
    )


def _task_set_dict(task_set: TaskSet) -> dict[str, object]:
    return {
        "appworld_version": task_set.appworld_version,
        "split": task_set.split,
        "selection_rule": task_set.selection_rule,
        "selection_seed": task_set.selection_seed,
        "task_count": task_set.task_count,
        "scenario_count": task_set.scenario_count,
        "task_ids": list(task_set.task_ids),
        "scenario_ids": list(task_set.scenario_ids),
        "task_set_hash": task_set.task_set_hash,
    }


def _public_document(decision) -> dict[str, object]:
    from llm_behavior_ci.records import public_record_dict

    document: dict[str, object] = {
        "outcome": decision.outcome,
        "reason_codes": list(decision.reason_codes),
        "reference_configuration_hash": decision.reference_configuration_hash,
        "candidate_configuration_hash": decision.candidate_configuration_hash,
        "task_set_hash": decision.task_set_hash,
        "reference_protocol_hash": decision.reference_protocol_hash,
        "candidate_protocol_hash": decision.candidate_protocol_hash,
        "thresholds": decision.thresholds.to_dict(),
        "statistics": [
            public_record_dict(item) for item in decision.statistics
        ],
        "public_decision": (
            public_record_dict(decision.public_decision)
            if decision.public_decision is not None
            else None
        ),
    }
    assert_public_payload(document)
    return document


class OfflineGateIntegrationTests(unittest.TestCase):
    def test_cli_plan_evidence_loader_round_trips_vocabulary_size(self) -> None:
        module = _load_cli_module()
        evidence = _evidence(kl_approximation="full", kl_vocabulary_size=32000)
        payload = plan_evidence_to_dict(evidence)
        restored = module._load_plan_evidence(payload)
        self.assertEqual(restored.kl_vocabulary_size, 32000)

    def test_library_pass_and_block_public_payload(self) -> None:
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(task_set, run_seed=8)
        decision = run_offline_gate(
            reference,
            candidate,
            task_set,
            settings=_settings(),
            runtime=_runtime(),
            plan_evidence=_evidence(),
        )
        self.assertEqual(decision.outcome, "PASS")
        document = _public_document(decision)
        self.assertEqual(document["outcome"], "PASS")
        self.assertEqual(document["reason_codes"], [])
        self.assertEqual(len(document["statistics"]), 3)
        self.assertEqual(decision.validation_provenance, "synthetic_fixture")

        blocked = run_offline_gate(
            reference,
            candidate,
            task_set,
            settings=_settings(score_margin=0.01),
            runtime=_runtime(),
            plan_evidence=_evidence(),
        )
        self.assertEqual(blocked.outcome, "BLOCK")
        block_document = _public_document(blocked)
        self.assertIn("plan_quality_margin", block_document["reason_codes"])

    def test_cli_requires_plan_evidence_argument(self) -> None:
        main = _load_cli_main()
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(task_set, run_seed=8)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_path = root / "reference.json"
            candidate_path = root / "candidate.json"
            task_set_path = root / "task_set.json"
            settings_path = root / "settings.json"
            reference_path.write_text(
                json.dumps(reference.to_dict()),
                encoding="utf-8",
            )
            candidate_path.write_text(
                json.dumps(candidate.to_dict()),
                encoding="utf-8",
            )
            task_set_path.write_text(
                json.dumps(_task_set_dict(task_set)),
                encoding="utf-8",
            )
            settings_path.write_text(
                json.dumps(_settings().to_dict()),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(
                    [
                        "--reference",
                        str(reference_path),
                        "--candidate",
                        str(candidate_path),
                        "--task-set",
                        str(task_set_path),
                        "--settings",
                        str(settings_path),
                    ],
                    runtime=_runtime(),
                )
            self.assertEqual(code, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertNotEqual(stderr.getvalue().strip(), "")
            self.assertNotIn('"outcome": "PASS"', stderr.getvalue())
            self.assertNotIn('"outcome": "BLOCK"', stderr.getvalue())

        code = main([])
        self.assertEqual(code, 2)

    def test_cli_execution_error_has_no_pass_block_document(self) -> None:
        main = _load_cli_main()
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(task_set, run_seed=8)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_path = root / "reference.json"
            candidate_path = root / "candidate.json"
            task_set_path = root / "task_set.json"
            settings_path = root / "settings.json"
            reference_path.write_text(
                json.dumps(reference.to_dict()),
                encoding="utf-8",
            )
            candidate_path.write_text(
                json.dumps(candidate.to_dict()),
                encoding="utf-8",
            )
            broken = _task_set_dict(task_set)
            broken["task_set_hash"] = "f" * 64
            task_set_path.write_text(json.dumps(broken), encoding="utf-8")
            settings_path.write_text(
                json.dumps(_settings().to_dict()),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(
                    [
                        "--reference",
                        str(reference_path),
                        "--candidate",
                        str(candidate_path),
                        "--task-set",
                        str(task_set_path),
                        "--settings",
                        str(settings_path),
                    ],
                    runtime=_runtime(),
                )
            self.assertEqual(code, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertNotEqual(stderr.getvalue().strip(), "")
            self.assertNotIn('"outcome": "PASS"', stderr.getvalue())
            self.assertNotIn('"outcome": "BLOCK"', stderr.getvalue())

    def test_gate_library_pass_is_stable(self) -> None:
        task_set = _task_set()
        reference = _config(task_set, run_seed=7)
        candidate = _config(task_set, run_seed=8)
        decision = run_offline_gate(
            reference,
            candidate,
            task_set,
            settings=_settings(),
            runtime=_runtime(),
            plan_evidence=_evidence(),
        )
        self.assertEqual(decision.outcome, "PASS")
        again = run_offline_gate(
            reference,
            candidate,
            task_set,
            settings=_settings(),
            runtime=_runtime(),
            plan_evidence=_evidence(),
        )
        self.assertEqual(
            decision.reference_configuration_hash,
            again.reference_configuration_hash,
        )
        self.assertEqual(
            decision.candidate_configuration_hash,
            again.candidate_configuration_hash,
        )


if __name__ == "__main__":
    unittest.main()
