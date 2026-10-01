import copy
import io
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from llm_behavior_ci.config import (
    RunConfiguration,
    hashed_values,
    new_run_identity,
    run_configuration_hash,
)
from llm_behavior_ci.experiments.faults import (
    FaultError,
    FaultPatch,
    FaultSpec,
    HarmLabel,
    apply_fault,
    default_results_root,
    freeze_harm_label,
    live_fault_available,
    load_fault,
    load_fault_catalog,
    main,
    measure_harm,
)
from llm_behavior_ci.records import TokenLogprob
from llm_behavior_ci.runtime.agent import AgentTurn
from llm_behavior_ci.runtime.appworld import EvaluationResult, TaskContext, ToolResult
from llm_behavior_ci.runtime.episode import RuntimeDependencies, run_episode
from llm_behavior_ci.stats.bootstrap import clustered_paired_bootstrap
from llm_behavior_ci.tasks.selection import (
    TaskSet,
    canonical_task_set_bytes,
    task_set_hash_from_bytes,
)

_REPO = Path(__file__).resolve().parents[2]
_CATALOG = _REPO / "configs" / "faults"
_START = datetime(2026, 9, 27, 15, 0, tzinfo=timezone.utc)
_LOGPROBS = ((TokenLogprob(token_id=7, logprob=-0.5, rank=0),),)
_EXPECTED_FAULT_IDS = (
    "api_documentation_one_app",
    "benign_batch_invariant",
    "benign_identical",
    "benign_logging_refactor",
    "benign_noop_redeploy",
    "fp8_weights",
    "lora_off_distribution",
    "model_downgrade_qwen3_1_7b",
    "nvfp4_weights",
    "prompt_remove_api_guidance",
    "sampling_temperature_one",
    "step_limit_reduced",
    "template_thinking_enabled",
    "token_limit_truncation",
)


def _task_set(*, split: str = "dev") -> TaskSet:
    tasks = (("task-a", "scenario-1"), ("task-b", "scenario-1"))
    payload = canonical_task_set_bytes(
        appworld_version="0.1.3.post1",
        split=split,
        selection_rule="deterministic_sample",
        selection_seed=20260926,
        tasks=tasks,
    )
    return TaskSet(
        appworld_version="0.1.3.post1",
        split=split,
        selection_rule="deterministic_sample",
        selection_seed=20260926,
        task_count=2,
        scenario_count=1,
        task_ids=("task-a", "task-b"),
        scenario_ids=("scenario-1", "scenario-1"),
        task_set_hash=task_set_hash_from_bytes(payload),
    )


def _payload(task_set: TaskSet | None = None) -> dict[str, object]:
    selected = task_set if task_set is not None else _task_set()
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
            "appworld_version": selected.appworld_version,
            "split": selected.split,
            "selection_rule": selected.selection_rule,
            "selection_seed": selected.selection_seed,
            "task_count": selected.task_count,
            "task_set_hash": selected.task_set_hash,
        },
        "run_seed": 7,
        "git_commit": "a" * 40,
        "protocol_hash": "e" * 64,
    }


def _base(task_set: TaskSet | None = None) -> RunConfiguration:
    return RunConfiguration.from_dict(_payload(task_set))


class Clock:
    def __init__(self) -> None:
        self.current = _START

    def __call__(self) -> datetime:
        value = self.current
        self.current += timedelta(seconds=1)
        return value


class World:
    def __init__(self, task_id: str, *, success: bool) -> None:
        self.task_id = task_id
        self.success = success
        self.close_count = 0

    def initial_state_identity(self) -> str:
        return self.task_id

    def context(self) -> TaskContext:
        return TaskContext(
            task_id=self.task_id,
            instruction="solve the task",
            api_documentation="calendar docs",
        )

    def evaluate(self) -> EvaluationResult:
        passed = 1 if self.success else 0
        return EvaluationResult(
            success=self.success,
            passed_requirements=passed,
            total_requirements=1,
            difficulty=1,
        )

    def execute(self, action: str) -> ToolResult:
        del action
        return ToolResult(
            output_text="ok",
            error_message=None,
            recoverable=True,
            app_name=None,
            api_name=None,
        )

    def close(self) -> None:
        self.close_count += 1


class ActingAgent:
    def __init__(self, clock: Clock) -> None:
        self._clock = clock

    def begin(self, context: TaskContext, config: RunConfiguration) -> None:
        del context, config

    def next_turn(self, *, tool_output: str | None) -> AgentTurn:
        del tool_output
        return AgentTurn(
            prompt_text="plan the next action",
            output_text="call",
            top_k_logprobs=_LOGPROBS,
            generated_token_count=len(_LOGPROBS),
            latency_seconds=0.1,
            started_at=self._clock(),
            action="calendar.lookup()",
            app_name=None,
            api_name=None,
        )


class StopAgent:
    def __init__(self, clock: Clock, *, hash_in_text: bool = False) -> None:
        self._clock = clock
        self._hash_in_text = hash_in_text
        self._config: RunConfiguration | None = None

    def begin(self, context: TaskContext, config: RunConfiguration) -> None:
        del context
        self._config = config

    def next_turn(self, *, tool_output: str | None) -> AgentTurn:
        del tool_output
        assert self._config is not None
        if self._hash_in_text:
            text = f"done:{run_configuration_hash(self._config)}"
        else:
            text = "done"
        return AgentTurn(
            prompt_text="plan the next action",
            output_text=text,
            top_k_logprobs=_LOGPROBS,
            generated_token_count=len(_LOGPROBS),
            latency_seconds=0.1,
            started_at=self._clock(),
            action=None,
            app_name=None,
            api_name=None,
        )


def _runtime(
    *,
    candidate_fails: bool = False,
    hash_in_text: bool = False,
    opened: list[str] | None = None,
) -> RuntimeDependencies:
    clock = Clock()
    calls = {"n": 0}

    def factory(task_id: str) -> World:
        if opened is not None:
            opened.append(task_id)
        is_candidate = calls["n"] % 2 == 1
        calls["n"] += 1
        success = not (candidate_fails and is_candidate)
        return World(task_id, success=success)

    return RuntimeDependencies(
        session_factory=factory,
        agent=StopAgent(clock, hash_in_text=hash_in_text),
        clock=clock,
    )


def _fp8_fault() -> FaultSpec:
    return load_fault(_CATALOG / "fp8_weights.v1.json")


def _measure(
    base: RunConfiguration,
    *,
    candidate_fails: bool,
    hash_in_text: bool = False,
    margin: float = 0.05,
    confidence_level: float = 0.9,
    resamples: int = 25,
    seed: int = 3,
    task_set: TaskSet | None = None,
    fault: FaultSpec | None = None,
    runtime: RuntimeDependencies | None = None,
) -> HarmLabel:
    selected = task_set if task_set is not None else _task_set()
    chosen = fault if fault is not None else _fp8_fault()
    candidate = apply_fault(base, chosen)
    deps = runtime if runtime is not None else _runtime(
        candidate_fails=candidate_fails,
        hash_in_text=hash_in_text,
    )
    return measure_harm(
        base,
        candidate,
        selected,
        margin=margin,
        runtime=deps,
        fault=chosen,
        confidence_level=confidence_level,
        resamples=resamples,
        seed=seed,
    )


class FaultPatchValidationTests(unittest.TestCase):
    def test_unknown_path_rejected(self) -> None:
        with self.assertRaises(FaultError) as ctx:
            FaultSpec(
                fault_id="bad_path",
                version="1",
                kind="quantization",
                patches=(FaultPatch("model.quantization.missing", "fp8"),),
            )
        self.assertIn("unknown path", str(ctx.exception))

    def test_invalid_values_rejected(self) -> None:
        base = _base()
        cases = (
            FaultSpec(
                fault_id="bad_quant",
                version="1",
                kind="quantization",
                patches=(FaultPatch("model.quantization.method", "int8"),),
            ),
            FaultSpec(
                fault_id="bad_temp",
                version="1",
                kind="sampling",
                patches=(FaultPatch("agent.sampling.temperature", -1.0),),
            ),
            FaultSpec(
                fault_id="bad_tokens",
                version="1",
                kind="token_limit",
                patches=(FaultPatch("agent.sampling.max_tokens", 0),),
            ),
            FaultSpec(
                fault_id="int_temp",
                version="1",
                kind="sampling",
                patches=(FaultPatch("agent.sampling.temperature", 1),),
            ),
        )
        for fault in cases:
            with self.subTest(fault_id=fault.fault_id):
                with self.assertRaises(FaultError) as ctx:
                    apply_fault(base, fault)
                self.assertIn("invalid value", str(ctx.exception))

    def test_path_outside_declared_kind_rejected(self) -> None:
        with self.assertRaises(FaultError) as ctx:
            FaultSpec(
                fault_id="seed_escape",
                version="1",
                kind="sampling",
                patches=(FaultPatch("agent.sampling.seed", 99),),
            )
        self.assertIn("outside the declared fault", str(ctx.exception))
        with self.assertRaises(FaultError) as ctx:
            FaultSpec(
                fault_id="step_on_quant",
                version="1",
                kind="quantization",
                patches=(FaultPatch("agent.step_limit", 4),),
            )
        self.assertIn("outside the declared fault", str(ctx.exception))

    def test_noop_patch_rejected(self) -> None:
        base = _base()
        fault = FaultSpec(
            fault_id="noop_quant",
            version="1",
            kind="quantization",
            patches=(FaultPatch("model.quantization.method", "none"),),
        )
        with self.assertRaises(FaultError) as ctx:
            apply_fault(base, fault)
        self.assertIn("outside the declared fault", str(ctx.exception))

    def test_lora_and_api_documentation_are_now_representable(self) -> None:
        base = _base()
        before = copy.deepcopy(base)
        model_id = id(base.model)
        agent_id = id(base.agent)
        task_id = id(base.task)
        lora_fault = load_fault(_CATALOG / "lora_off_distribution.v1.json")
        docs_fault = load_fault(_CATALOG / "api_documentation_one_app.v1.json")
        self.assertTrue(lora_fault.representable)
        self.assertTrue(docs_fault.representable)

        lora_candidate = apply_fault(base, lora_fault)
        self.assertEqual(
            lora_candidate.model.lora.repository,
            "Qwen/Qwen3-4B-lora-off-distribution",
        )
        self.assertIsNone(base.model.lora)

        docs_candidate = apply_fault(base, docs_fault)
        self.assertEqual(
            docs_candidate.agent.api_docs_version,
            "api-docs-corrupt-v1",
        )
        self.assertEqual(docs_candidate.agent.api_docs_app, "supervisor")
        self.assertIsNone(base.agent.api_docs_version)

        self.assertEqual(base, before)
        self.assertEqual(id(base.model), model_id)
        self.assertEqual(id(base.agent), agent_id)
        self.assertEqual(id(base.task), task_id)


class FaultReproducibilityTests(unittest.TestCase):
    def test_apply_fault_twice_equal_and_same_hash(self) -> None:
        base = _base()
        fault = _fp8_fault()
        first = apply_fault(base, fault)
        second = apply_fault(base, fault)
        self.assertEqual(first, second)
        self.assertEqual(
            run_configuration_hash(first),
            run_configuration_hash(second),
        )

    def test_measure_harm_twice_equal(self) -> None:
        base = _base()
        first = _measure(base, candidate_fails=True)
        second = _measure(base, candidate_fails=True)
        self.assertEqual(first, second)


class FaultBaselineTests(unittest.TestCase):
    def test_apply_fault_does_not_mutate_base(self) -> None:
        base = _base()
        before = copy.deepcopy(base)
        model_id = id(base.model)
        agent_id = id(base.agent)
        task_id = id(base.task)
        candidate = apply_fault(base, _fp8_fault())
        self.assertEqual(base, before)
        self.assertEqual(id(base.model), model_id)
        self.assertEqual(id(base.agent), agent_id)
        self.assertEqual(id(base.task), task_id)
        self.assertNotEqual(candidate, base)
        self.assertIsNot(candidate, base)


class FaultLabelProvenanceTests(unittest.TestCase):
    def test_harm_label_provenance_candidate_fails(self) -> None:
        task_set = _task_set()
        base = _base(task_set)
        fault = _fp8_fault()
        candidate = apply_fault(base, fault)
        label = measure_harm(
            base,
            candidate,
            task_set,
            margin=0.05,
            runtime=_runtime(candidate_fails=True),
            fault=fault,
            confidence_level=0.9,
            resamples=25,
            seed=3,
        )
        expected = clustered_paired_bootstrap(
            (0.0, 0.0),
            (1.0, 1.0),
            ("scenario:scenario-1", "scenario:scenario-1"),
            confidence_level=0.9,
            resamples=25,
            seed=3,
        )
        self.assertEqual(label.fault_version, fault.fault_version)
        self.assertEqual(
            label.base_configuration_hash,
            run_configuration_hash(base),
        )
        self.assertEqual(
            label.candidate_configuration_hash,
            run_configuration_hash(candidate),
        )
        self.assertEqual(label.task_set_hash, task_set.task_set_hash)
        self.assertEqual(label.margin, 0.05)
        self.assertEqual(label.effect_estimate, -1.0)
        self.assertTrue(label.harmful)
        self.assertEqual(label.interval_low, expected.confidence_low)
        self.assertEqual(label.interval_high, expected.confidence_high)
        self.assertEqual(label.split, "dev")

    def test_harm_label_ignores_output_text(self) -> None:
        task_set = _task_set()
        base = _base(task_set)
        label = _measure(
            base,
            candidate_fails=False,
            hash_in_text=True,
            task_set=task_set,
        )
        self.assertEqual(label.effect_estimate, 0.0)
        self.assertFalse(label.harmful)
        self.assertEqual(label.split, "dev")

    def test_missing_evaluator_outcome_is_not_a_label(self) -> None:
        task_set = _task_set()
        base = _base(task_set)
        fault = _fp8_fault()
        candidate = apply_fault(base, fault)
        clock = Clock()

        class CrashingWorld(World):
            def execute(self, action: str) -> ToolResult:
                del action
                raise RuntimeError("execute failed")

        runtime = RuntimeDependencies(
            session_factory=lambda task_id: CrashingWorld(task_id, success=False),
            agent=ActingAgent(clock),
            clock=clock,
        )
        with self.assertRaises(FaultError) as ctx:
            measure_harm(
                base,
                candidate,
                task_set,
                margin=0.05,
                runtime=runtime,
                fault=fault,
                confidence_level=0.9,
                resamples=25,
                seed=3,
            )
        self.assertIn("missing evaluator outcome", str(ctx.exception))

    def test_undeclared_candidate_opens_no_world(self) -> None:
        task_set = _task_set()
        base = _base(task_set)
        fp8 = apply_fault(base, _fp8_fault())
        sampling = load_fault(_CATALOG / "sampling_temperature_one.v1.json")
        opened: list[str] = []
        with self.assertRaises(FaultError) as ctx:
            measure_harm(
                base,
                fp8,
                task_set,
                margin=0.05,
                runtime=_runtime(opened=opened),
                fault=sampling,
                confidence_level=0.9,
                resamples=25,
                seed=3,
            )
        self.assertIn("declared fault", str(ctx.exception))
        self.assertEqual(opened, [])


class FaultFreezeTests(unittest.TestCase):
    def test_freeze_idempotent_and_rejects_other_content(self) -> None:
        base = _base()
        label = _measure(base, candidate_fails=True)
        with TemporaryDirectory() as tmp:
            destination = Path(tmp) / "labels" / "fp8.json"
            freeze_harm_label(label, destination, results_root=Path(tmp) / "results")
            first = destination.read_bytes()
            freeze_harm_label(label, destination, results_root=Path(tmp) / "results")
            self.assertEqual(destination.read_bytes(), first)
            other = replace(label, margin=0.99)
            with self.assertRaises(FaultError) as ctx:
                freeze_harm_label(
                    other,
                    destination,
                    results_root=Path(tmp) / "results",
                )
            self.assertIn("already frozen", str(ctx.exception))
            self.assertEqual(destination.read_bytes(), first)

    def test_freeze_rejects_destination_under_results_root(self) -> None:
        base = _base()
        label = _measure(base, candidate_fails=True)
        with TemporaryDirectory() as tmp:
            results_root = Path(tmp) / "results"
            destination = results_root / "nested" / "label.json"
            with self.assertRaises(FaultError) as ctx:
                freeze_harm_label(
                    label,
                    destination,
                    results_root=results_root,
                )
            self.assertIn("results", str(ctx.exception))
            self.assertFalse(destination.exists())

    def test_default_results_root_is_repo_results(self) -> None:
        self.assertEqual(default_results_root(), _REPO / "results")

    def test_main_catalog_lists_faults(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(["--catalog", str(_CATALOG)])
        self.assertEqual(code, 0)
        text = buffer.getvalue()
        self.assertIn("fp8_weights:1", text)
        self.assertIn("live_unavailable", text)
        self.assertNotIn("schema_gap", text)


class FaultSplitRejectionTests(unittest.TestCase):
    def test_test_normal_rejected_before_world_opens(self) -> None:
        task_set = _task_set(split="test_normal")
        base = _base(_task_set())
        fault = _fp8_fault()
        candidate = apply_fault(base, fault)
        opened: list[str] = []
        runtime = _runtime(opened=opened)
        with self.assertRaises(FaultError) as ctx:
            measure_harm(
                base,
                candidate,
                task_set,
                margin=0.05,
                runtime=runtime,
                fault=fault,
                confidence_level=0.9,
                resamples=25,
                seed=3,
            )
        self.assertIn("dev", str(ctx.exception))
        self.assertEqual(opened, [])

    def test_train_split_rejected(self) -> None:
        task_set = _task_set(split="train")
        base = _base(_task_set())
        fault = _fp8_fault()
        candidate = apply_fault(base, fault)
        opened: list[str] = []
        runtime = _runtime(opened=opened)
        with self.assertRaises(FaultError) as ctx:
            measure_harm(
                base,
                candidate,
                task_set,
                margin=0.05,
                runtime=runtime,
                fault=fault,
                confidence_level=0.9,
                resamples=25,
                seed=3,
            )
        self.assertIn("dev", str(ctx.exception))
        self.assertEqual(opened, [])


class FaultCatalogTests(unittest.TestCase):
    def test_catalog_load_covers_all_fault_ids(self) -> None:
        catalog = load_fault_catalog(_CATALOG)
        ids = tuple(item.fault_id for item in catalog)
        self.assertEqual(sorted(ids), sorted(_EXPECTED_FAULT_IDS))
        self.assertEqual(len(ids), 14)
        base = _base()
        sampling = load_fault(_CATALOG / "sampling_temperature_one.v1.json")
        self.assertEqual(len(sampling.patches), 1)
        self.assertIsInstance(sampling.patches[0].value, float)
        self.assertEqual(sampling.patches[0].value, 1.0)
        for fault in catalog:
            with self.subTest(fault_id=fault.fault_id):
                self.assertTrue(fault.schema_supported)
                self.assertTrue(fault.representable)
                candidate = apply_fault(base, fault)
                self.assertIsNot(candidate, base)
                if fault.fault_id == "benign_identical":
                    self.assertEqual(candidate, base)
                elif fault.fault_id == "benign_noop_redeploy":
                    self.assertEqual(candidate.git_commit, "b" * 40)
                elif fault.fault_id == "benign_batch_invariant":
                    self.assertTrue(candidate.model.serving.batch_invariant)
                elif fault.fault_id == "benign_logging_refactor":
                    self.assertEqual(candidate.git_commit, "c" * 40)
                elif fault.fault_id == "fp8_weights":
                    self.assertEqual(
                        candidate.model.quantization.method,
                        "fp8",
                    )
                elif fault.fault_id == "nvfp4_weights":
                    self.assertEqual(
                        candidate.model.quantization.method,
                        "nvfp4",
                    )
                elif fault.fault_id == "model_downgrade_qwen3_1_7b":
                    self.assertEqual(
                        candidate.model.model.repository,
                        "Qwen/Qwen3-1.7B",
                    )
                    self.assertEqual(
                        candidate.model.tokenizer.repository,
                        "Qwen/Qwen3-1.7B",
                    )
                    self.assertEqual(
                        candidate.model.model.revision,
                        base.model.model.revision,
                    )
                    self.assertEqual(
                        candidate.model.tokenizer.revision,
                        base.model.tokenizer.revision,
                    )
                elif fault.fault_id == "prompt_remove_api_guidance":
                    self.assertEqual(
                        candidate.agent.prompt.prompt_version,
                        "prompt-no-api-guidance",
                    )
                elif fault.fault_id == "template_thinking_enabled":
                    self.assertTrue(candidate.agent.prompt.thinking_enabled)
                elif fault.fault_id == "sampling_temperature_one":
                    self.assertEqual(candidate.agent.sampling.temperature, 1.0)
                    self.assertIsInstance(
                        candidate.agent.sampling.temperature,
                        float,
                    )
                elif fault.fault_id == "token_limit_truncation":
                    self.assertEqual(candidate.agent.sampling.max_tokens, 16)
                elif fault.fault_id == "step_limit_reduced":
                    self.assertEqual(candidate.agent.step_limit, 4)
                elif fault.fault_id == "lora_off_distribution":
                    self.assertEqual(
                        candidate.model.lora.repository,
                        "Qwen/Qwen3-4B-lora-off-distribution",
                    )
                    self.assertEqual(
                        candidate.model.lora.revision,
                        "d" * 40,
                    )
                    self.assertIsNone(base.model.lora)
                elif fault.fault_id == "api_documentation_one_app":
                    self.assertEqual(
                        candidate.agent.api_docs_version,
                        "api-docs-corrupt-v1",
                    )
                    self.assertEqual(candidate.agent.api_docs_app, "supervisor")
                    self.assertIsNone(base.agent.api_docs_version)

    def test_live_unavailable_faults_reported_not_absent(self) -> None:
        catalog = load_fault_catalog(_CATALOG)
        by_id = {fault.fault_id: fault for fault in catalog}
        self.assertEqual(set(by_id), set(_EXPECTED_FAULT_IDS))
        unavailable = {
            "lora_off_distribution",
            "fp8_weights",
            "nvfp4_weights",
            "benign_batch_invariant",
        }
        for fault_id in unavailable:
            with self.subTest(fault_id=fault_id):
                availability = live_fault_available(by_id[fault_id])
                self.assertFalse(availability.available)
                self.assertIsNotNone(availability.reason)
        live = live_fault_available(by_id["sampling_temperature_one"])
        self.assertTrue(live.available)
        self.assertIsNone(live.reason)
        docs_live = live_fault_available(by_id["api_documentation_one_app"])
        self.assertTrue(docs_live.available)
        self.assertIsNone(docs_live.reason)
        prompt_live = live_fault_available(by_id["prompt_remove_api_guidance"])
        self.assertTrue(prompt_live.available)
        self.assertIsNone(prompt_live.reason)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(["--catalog", str(_CATALOG)])
        self.assertEqual(code, 0)
        text = buffer.getvalue()
        self.assertIn("fp8_weights:1\tquantization\tlive_unavailable", text)
        self.assertIn("lora_off_distribution:1\tlora\tlive_unavailable", text)
        self.assertIn("api_documentation_one_app:1\tapi_documentation\tlive", text)
        self.assertIn("sampling_temperature_one:1\tsampling\tlive", text)

    def test_unregistered_prompt_version_fault_is_unavailable(self) -> None:
        fault = FaultSpec(
            fault_id="prompt_unknown_version",
            version="1",
            kind="prompt",
            patches=(
                FaultPatch(
                    path="agent.prompt.prompt_version",
                    value="prompt-does-not-exist",
                ),
            ),
        )
        availability = live_fault_available(fault)
        self.assertFalse(availability.available)
        self.assertIsNotNone(availability.reason)

    def test_prompt_remove_api_guidance_changes_rendered_system_text(self) -> None:
        from llm_behavior_ci.runtime.prompts import render_system_text

        baseline = render_system_text(
            prompt_version="prompt-v1",
            plan_format_version="plan-v1",
            thinking_enabled=False,
            action_interface="code",
            mode="execute",
        )
        stripped = render_system_text(
            prompt_version="prompt-no-api-guidance",
            plan_format_version="plan-v1",
            thinking_enabled=False,
            action_interface="code",
            mode="execute",
        )
        self.assertNotEqual(baseline, stripped)
        self.assertIn("API documentation", baseline)
        self.assertNotIn("API documentation", stripped)


def _changed_leaves(
    base: RunConfiguration,
    candidate: RunConfiguration,
) -> set[str]:
    before = hashed_values(base)
    after = hashed_values(candidate)
    return {path for path in before if before[path] != after[path]}


def _run_execute(config: RunConfiguration):
    clock = Clock()

    def factory(task_id: str) -> World:
        return World(task_id, success=False)

    return run_episode(
        "task-a",
        config,
        "execute",
        run=new_run_identity(config),
        runtime=RuntimeDependencies(
            session_factory=factory,
            agent=ActingAgent(clock),
            clock=clock,
        ),
    )


class StepLimitFaultTests(unittest.TestCase):
    def test_legacy_step_limit_is_the_horizon_run_episode_enforces(self) -> None:
        fault = load_fault(_CATALOG / "step_limit_reduced.v1.json")
        base = _base()
        candidate = apply_fault(base, fault)
        self.assertEqual(base.agent.execute_turn_limit, 40)
        self.assertEqual(candidate.agent.step_limit, 4)
        self.assertIsNone(candidate.agent.execute_max_model_turns)
        self.assertEqual(candidate.agent.execute_turn_limit, 4)
        self.assertEqual(_changed_leaves(base, candidate), {"agent.step_limit"})
        self.assertNotEqual(
            run_configuration_hash(base),
            run_configuration_hash(candidate),
        )
        self.assertEqual(len(_run_execute(base).model_steps), 40)
        faulted = _run_execute(candidate)
        self.assertEqual(faulted.termination_reason, "step_limit")
        self.assertEqual(len(faulted.model_steps), 4)

    def test_execute_override_is_the_horizon_the_fault_reduces(self) -> None:
        fault = load_fault(_CATALOG / "step_limit_reduced.v1.json")
        payload = _payload()
        payload["agent"]["execute_max_model_turns"] = 20
        base = RunConfiguration.from_dict(payload)
        candidate = apply_fault(base, fault)
        self.assertEqual(base.agent.step_limit, 40)
        self.assertEqual(base.agent.execute_turn_limit, 20)
        self.assertEqual(candidate.agent.step_limit, 40)
        self.assertEqual(candidate.agent.execute_max_model_turns, 4)
        self.assertEqual(candidate.agent.execute_turn_limit, 4)
        self.assertEqual(
            _changed_leaves(base, candidate),
            {"agent.execute_max_model_turns"},
        )
        self.assertNotEqual(
            run_configuration_hash(base),
            run_configuration_hash(candidate),
        )
        self.assertEqual(len(_run_execute(base).model_steps), 20)
        faulted = _run_execute(candidate)
        self.assertEqual(faulted.termination_reason, "step_limit")
        self.assertEqual(len(faulted.model_steps), 4)


if __name__ == "__main__":
    unittest.main()
