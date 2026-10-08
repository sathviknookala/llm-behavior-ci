from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from llm_behavior_ci.config import GateSettings, RunConfiguration, run_configuration_hash
from llm_behavior_ci.experiments.attempts import (
    AttemptBudget,
    AttemptCapExceeded,
    AttemptLedger,
)
from llm_behavior_ci.experiments.faults import (
    FaultPatch,
    FaultSpec,
    apply_fault,
    harm_label_from_outcomes,
    measure_harm,
)
from llm_behavior_ci.experiments.qualification_batch import (
    DEV_ARMS,
    PLAN_ARMS,
    QualificationBatchError,
    cusum_scale,
    dev_four_arm_design,
    dev_harm_labels,
    dev_outcome_table,
    execution_aa_inputs,
    execution_aa_reports,
    load_batch_checkpoint,
    open_batch,
    plan_aa_design,
    plan_aa_gate_replay,
    plan_aa_reports,
    plan_aa_series,
    plan_pairs,
    run_qualification_batch,
    scope_for,
    seeded_arm_orders,
)
from llm_behavior_ci.experiments.benchmark import reconcile_checkpoint
from llm_behavior_ci.experiments.validation import method_spec
from llm_behavior_ci.lifecycle import offline_gate as gate_module
from llm_behavior_ci.lifecycle.offline_gate import (
    PlanEvidenceInputs,
    plan_quality_score,
    plan_representation,
    replay_plan_gate,
    run_offline_gate,
)
from llm_behavior_ci.records import TokenLogprob
from llm_behavior_ci.runtime.agent import AgentTurn
from llm_behavior_ci.runtime.appworld import EvaluationResult, TaskContext, ToolResult
from llm_behavior_ci.runtime.episode import RuntimeDependencies
from llm_behavior_ci.runtime.factory import RuntimeFactory
from llm_behavior_ci.storage import EpisodeStore
from llm_behavior_ci.tasks.plan_specs import TaskPlanSpec
from llm_behavior_ci.tasks.selection import (
    TaskSet,
    canonical_task_set_bytes,
    task_set_hash_from_bytes,
)

_START = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
_LOGPROBS = ((TokenLogprob(token_id=7, logprob=-0.2, rank=0),),)
_GOOD_PLAN = (
    "1. apis.calendar.open_calendar() to open the calendar\n"
    "2. apis.calendar.create_event() to create an event"
)
_WEAK_PLAN = "1. open the calendar\n2. apis.calendar.create_event()"
_BAD_PLAN = "1. apis.mail.send_mail() to add an event"


def _task_set(split: str, count: int) -> TaskSet:
    tasks = tuple((f"{split}_task_{index}", f"{split}_scenario_{index}") for index in range(count))
    payload = canonical_task_set_bytes(
        appworld_version="0.1.3.post1",
        split=split,
        selection_rule="one_per_scenario_v1",
        selection_seed=17,
        tasks=tasks,
    )
    return TaskSet(
        appworld_version="0.1.3.post1",
        split=split,
        selection_rule="one_per_scenario_v1",
        selection_seed=17,
        task_count=count,
        scenario_count=count,
        task_ids=tuple(item[0] for item in tasks),
        scenario_ids=tuple(item[1] for item in tasks),
        task_set_hash=task_set_hash_from_bytes(payload),
    )


def _configuration(task_set: TaskSet) -> RunConfiguration:
    return RunConfiguration.from_dict(
        {
            "model": {
                "model": {"repository": "Qwen/Qwen3-4B", "revision": "0" * 40},
                "tokenizer": {"repository": "Qwen/Qwen3-4B", "revision": "f" * 40},
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
                "appworld_version": task_set.appworld_version,
                "split": task_set.split,
                "selection_rule": task_set.selection_rule,
                "selection_seed": task_set.selection_seed,
                "task_count": task_set.task_count,
                "task_set_hash": task_set.task_set_hash,
            },
            "run_seed": 17,
            "git_commit": "a" * 40,
            "protocol_hash": None,
        }
    )


def _noop_fault() -> FaultSpec:
    return FaultSpec(
        fault_id="temperature_noop",
        version="1",
        kind="benign_control",
        control="declared_noop",
        patches=(FaultPatch(path="agent.sampling.temperature", value=0.0),),
    )


def _regression_fault() -> FaultSpec:
    return FaultSpec(
        fault_id="sampling_temperature_one",
        version="1",
        kind="sampling",
        patches=(FaultPatch(path="agent.sampling.temperature", value=1.0),),
    )


class Clock:
    def __init__(self) -> None:
        self.current = _START

    def __call__(self) -> datetime:
        value = self.current
        self.current += timedelta(seconds=1)
        return value


class World:
    def __init__(self, task_id: str, success: bool | None) -> None:
        self.task_id = task_id
        self.success = success

    def initial_state_identity(self) -> str:
        return "same-state"

    def prepare(self) -> None:
        if self.success is None:
            raise RuntimeError("environment setup failed")

    def context(self) -> TaskContext:
        return TaskContext(
            task_id=self.task_id,
            instruction="open the calendar and create an event",
            api_documentation="calendar docs",
        )

    def execute(self, action: str) -> ToolResult:
        del action
        return ToolResult(
            output_text="ok", error_message=None, recoverable=False, app_name=None, api_name=None
        )

    def evaluate(self) -> EvaluationResult:
        return EvaluationResult(
            success=bool(self.success),
            passed_requirements=1 if self.success else 0,
            total_requirements=1,
            difficulty=1,
        )

    def close(self) -> None:
        return None


class Crash(BaseException):
    pass


class Agent:
    def __init__(self, clock: Clock, plans: list[str], *, crash: bool = False) -> None:
        self._clock = clock
        self._plans = plans
        self._crash = crash
        self._context: TaskContext | None = None

    def begin(self, context: TaskContext, config: RunConfiguration) -> None:
        del config
        self._context = context

    def messages(self, tool_output: str | None = None) -> list[dict[str, str]]:
        del tool_output
        return [{"role": "user", "content": "plan"}]

    def next_turn(self, *, tool_output: str | None) -> AgentTurn:
        del tool_output
        if self._crash:
            raise Crash()
        return AgentTurn(
            prompt_text="plan the next action",
            output_text=self._plans.pop(0) if self._plans else _GOOD_PLAN,
            top_k_logprobs=_LOGPROBS,
            generated_token_count=1,
            latency_seconds=0.1,
            started_at=self._clock(),
            action=None,
            app_name=None,
            api_name=None,
        )

    def teacher_force_plan(self, *, messages, plan_text):
        del messages, plan_text
        return _LOGPROBS


class RecordingFactory(RuntimeFactory):
    """Fresh agent and world per call; outcomes come from ``outcome(call)``."""

    def __init__(self, outcome, *, plans=None, crash_on_call: int | None = None) -> None:
        self.clock = Clock()
        self.calls: list[tuple[str, str, str]] = []
        self.worlds: list[str] = []
        self._outcome = outcome
        self._plans = plans
        self._crash_on_call = crash_on_call

    def __call__(self, configuration, *, mode, role="reference"):
        self.calls.append((run_configuration_hash(configuration), mode, role))
        index = len(self.calls)
        plans = [] if self._plans is None else [self._plans(index)]

        def session(task_id: str) -> World:
            self.worlds.append(task_id)
            return World(task_id, self._outcome(index, configuration, task_id))

        return RuntimeDependencies(
            session_factory=session,
            agent=Agent(self.clock, plans, crash=index == self._crash_on_call),
            clock=self.clock,
        )


def _always(success: bool | None):
    return lambda _index, _configuration, _task_id: success


class DevBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.tasks = _task_set("dev", 4)
        self.healthy = _configuration(self.tasks)
        self.noop = apply_fault(self.healthy, _noop_fault())
        self.regression = apply_fault(self.healthy, _regression_fault())
        self.orders = seeded_arm_orders(self.tasks.task_count, 17)
        self.design = dev_four_arm_design(
            self.tasks,
            healthy=self.healthy,
            noop=self.noop,
            noop_fault=_noop_fault(),
            regression=self.regression,
            regression_fault=_regression_fault(),
            arm_orders=self.orders,
        )
        self.checkpoint = self.root / "checkpoint.json"

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _run(
        self,
        factory,
        budget: AttemptBudget | None = AttemptBudget(plan_generations=0, executions=16),
        store=None,
    ):
        return run_qualification_batch(
            self.design,
            runtime_factory=factory,
            checkpoint_path=self.checkpoint,
            budget=budget,
            store=store,
        )

    def test_arm_orders_reproduce_the_predeclared_seeded_shuffle(self) -> None:
        orders = seeded_arm_orders(19, 17)
        self.assertEqual(
            orders[0],
            ("H1_healthy_reference", "C0_noop_candidate", "H2_healthy_repeat", "C1_regression_candidate"),
        )
        self.assertEqual(
            orders[18],
            ("H2_healthy_repeat", "C1_regression_candidate", "H1_healthy_reference", "C0_noop_candidate"),
        )

    def test_four_arms_run_in_seeded_order_as_distinct_attempts(self) -> None:
        self.assertEqual(
            run_configuration_hash(self.noop), run_configuration_hash(self.healthy)
        )
        factory = RecordingFactory(_always(True))
        store = EpisodeStore(self.root / "episodes.sqlite")
        try:
            summary = self._run(factory, store=store)
        finally:
            store.close()
        self.assertEqual(summary.dispatched, 16)
        healthy = run_configuration_hash(self.healthy)
        regression = run_configuration_hash(self.regression)
        expected = [
            regression if arm == DEV_ARMS[3] else healthy
            for order in self.orders
            for arm in order
        ]
        self.assertEqual([call[0] for call in factory.calls], expected)
        self.assertEqual({call[1] for call in factory.calls}, {"execute"})
        self.assertEqual(len(factory.worlds), 16)
        document = load_batch_checkpoint(self.checkpoint)
        records = document["attempts"]["records"]
        self.assertEqual(len({record["scope"] for record in records}), 16)
        same_hash = [record for record in records if record["configuration_hash"] == healthy]
        self.assertEqual(len(same_hash), 12)
        self.assertEqual(
            {record["role"] for record in same_hash}, set(DEV_ARMS[:3])
        )
        runs = {
            arm: document["runs"][arm]["run_id"] for arm in DEV_ARMS
        }
        self.assertEqual(len(set(runs.values())), 4)
        self.assertEqual(summary.attempts["execute"]["completed"], 16)
        self.assertEqual(summary.attempts["plan"]["consumed"], 0)
        episode_ids = {
            document["results"][scope_for(self.design, index, arm)]["episode"]["episode_id"]
            for index in range(4)
            for arm in DEV_ARMS
        }
        self.assertEqual(len(episode_ids), 16)

    def test_shared_reference_feeds_both_labels_and_the_aa(self) -> None:
        def outcome(index, configuration, task_id):
            del index
            if run_configuration_hash(configuration) == run_configuration_hash(self.regression):
                return False
            return not task_id.endswith("_3")

        self._run(RecordingFactory(outcome))
        document = load_batch_checkpoint(self.checkpoint)
        labels = dev_harm_labels(
            self.design, document, margin=0.2, confidence_level=0.95, resamples=200, seed=17
        )
        self.assertEqual(labels["noop"]["reference_arm"], DEV_ARMS[0])
        self.assertEqual(labels["regression"]["reference_arm"], DEV_ARMS[0])
        self.assertEqual(labels["noop"]["status"], "measured")
        self.assertAlmostEqual(labels["noop"]["effect_estimate"], 0.0)
        self.assertFalse(labels["noop"]["harmful"])
        self.assertAlmostEqual(labels["regression"]["effect_estimate"], -0.75)
        self.assertTrue(labels["regression"]["harmful"])
        table = dev_outcome_table(self.design, document)
        expected = harm_label_from_outcomes(
            self.healthy,
            self.regression,
            self.tasks,
            fault=_regression_fault(),
            base_successes=[row[DEV_ARMS[0]]["success"] for row in table],
            candidate_successes=[row[DEV_ARMS[3]]["success"] for row in table],
            margin=0.2,
            confidence_level=0.95,
            resamples=200,
            seed=17,
        )
        self.assertEqual(labels["regression"]["interval_low"], expected.interval_low)
        observations, context = execution_aa_inputs(self.design, document)
        self.assertEqual(len(observations), 8)
        self.assertEqual(context.repetitions, (0, 1) * 4)
        scale = cusum_scale(self.design, document)
        self.assertEqual(scale["scored_outcomes"], 8)
        self.assertAlmostEqual(scale["target"], 0.75)
        self.assertAlmostEqual(scale["slack"], 0.5 * scale["sigma"])
        self.assertAlmostEqual(scale["threshold"], 5.0 * scale["sigma"])

    def test_execution_aa_reaches_the_monitor_and_canary_contracts(self) -> None:
        def outcome(index, configuration, task_id):
            del configuration
            return task_id.endswith(("_0", "_1")) or index % 2 == 0

        self._run(RecordingFactory(outcome))
        document = load_batch_checkpoint(self.checkpoint)
        specs = [
            method_spec(
                "cusum",
                {"target": 0.5833, "slack": 0.2575, "threshold": 2.5745, "direction": "decrease"},
                required_checks=("aa_dependence",),
                study="simulation",
                null_sample_size=25,
                uncertainty_level=0.95,
                null_draw="constant",
                alpha=0.05,
                horizon=25,
            ),
            method_spec(
                "sequential_canary",
                {"alpha": 0.05, "harm_margin": 0.2, "horizon_episodes": 12, "null_probability": 0.5833},
                required_checks=("aa_dependence",),
                study="simulation",
                null_sample_size=12,
                uncertainty_level=0.95,
                null_draw="bernoulli",
                alpha=0.05,
                horizon=12,
            ),
        ]
        reports = execution_aa_reports(self.design, document, specs)
        for name in ("cusum", "sequential_canary"):
            self.assertEqual(reports[name]["status"], "passed", reports[name]["reason"])
            self.assertEqual(reports[name]["observation_count"], 8)
            self.assertEqual(reports[name]["provenance"], "local_runtime")

    def test_missing_outcomes_stay_missing_and_leave_a_label_unmeasurable(self) -> None:
        def outcome(index, configuration, task_id):
            del index
            if (
                run_configuration_hash(configuration) == run_configuration_hash(self.regression)
                and task_id.endswith("_2")
            ):
                return None
            return True

        self._run(RecordingFactory(outcome))
        document = load_batch_checkpoint(self.checkpoint)
        table = dev_outcome_table(self.design, document)
        self.assertEqual(table[2][DEV_ARMS[3]], {"state": "failed", "success": None})
        labels = dev_harm_labels(
            self.design, document, margin=0.2, confidence_level=0.95, resamples=200, seed=17
        )
        self.assertEqual(labels["regression"]["status"], "unmeasurable")
        self.assertEqual(labels["regression"]["missing_pairs"], 1)
        self.assertNotIn("effect_estimate", labels["regression"])
        self.assertEqual(labels["noop"]["status"], "measured")
        summary = AttemptLedger(document["attempts"], budget=None, persist=lambda: None).summary()
        self.assertEqual(summary["execute"]["failed"], 1)
        self.assertEqual(summary["execute"]["consumed"], 16)

    def test_caps_cannot_exceed_the_design_or_spend_plan_generations(self) -> None:
        with self.assertRaisesRegex(QualificationBatchError, "replacement"):
            self._run(RecordingFactory(_always(True)), AttemptBudget(plan_generations=0, executions=17))
        with self.assertRaisesRegex(QualificationBatchError, "no plan"):
            self._run(RecordingFactory(_always(True)), AttemptBudget(plan_generations=1, executions=16))
        self.assertFalse(self.checkpoint.exists())

    def test_a_lower_execution_cap_stops_dispatch_and_stays_fixed(self) -> None:
        factory = RecordingFactory(_always(True))
        with self.assertRaises(AttemptCapExceeded):
            self._run(factory, AttemptBudget(plan_generations=0, executions=5))
        self.assertEqual(len(factory.calls), 5)
        with self.assertRaisesRegex(QualificationBatchError, "original caps"):
            self._run(factory, AttemptBudget(plan_generations=0, executions=16))
        with self.assertRaises(AttemptCapExceeded):
            self._run(factory, None)
        self.assertEqual(len(factory.calls), 5)

    def test_crash_recovery_never_reruns_an_interrupted_or_completed_arm(self) -> None:
        crashing = RecordingFactory(_always(True), crash_on_call=6)
        with self.assertRaises(Crash):
            self._run(crashing)
        self.assertEqual(len(crashing.calls), 6)
        document = load_batch_checkpoint(self.checkpoint)
        open_scopes = [r["scope"] for r in document["attempts"]["records"] if r["state"] == "started"]
        self.assertEqual(len(open_scopes), 1)
        resumed = RecordingFactory(_always(True))
        summary = self._run(resumed, None)
        self.assertEqual(summary.dispatched, 10)
        self.assertEqual(summary.skipped, 6)
        self.assertEqual(len(resumed.calls), 10)
        self.assertEqual(summary.attempts["execute"]["interrupted"], 1)
        self.assertEqual(summary.attempts["execute"]["completed"], 15)
        self.assertEqual(summary.attempts["execute"]["consumed"], 16)
        self.assertEqual(summary.attempts["execute"]["remaining"], 0)
        again = RecordingFactory(_always(True))
        self.assertEqual(self._run(again, None).dispatched, 0)
        self.assertEqual(again.calls, [])
        document = load_batch_checkpoint(self.checkpoint)
        self.assertNotIn(open_scopes[0], document["results"])
        table = dev_outcome_table(self.design, document)
        index, arm = int(open_scopes[0].split("|")[1]), open_scopes[0].split("|")[2]
        self.assertEqual(table[index][arm], {"state": "interrupted", "success": None})

    def test_a_crash_between_reserve_and_start_is_interrupted_before_start(self) -> None:
        original = AttemptLedger.start
        calls = {"count": 0}

        def start(self, attempts):
            calls["count"] += 1
            if calls["count"] == 3:
                raise Crash()
            return original(self, attempts)

        with mock.patch.object(AttemptLedger, "start", start), self.assertRaises(Crash):
            self._run(RecordingFactory(_always(True)))
        summary = reconcile_checkpoint(self.checkpoint)
        self.assertEqual(summary.reconciled, 1)
        self.assertEqual(summary.attempts["execute"]["interrupted_before_start"], 1)
        resumed = RecordingFactory(_always(True))
        result = self._run(resumed, None)
        self.assertEqual(result.dispatched, 13)
        self.assertEqual(result.attempts["execute"]["known_starts"], 15)

    def test_a_raised_error_fails_its_started_attempt_and_stops(self) -> None:
        class Broken(RecordingFactory):
            def __call__(self, configuration, *, mode, role="reference"):
                if len(self.calls) == 2:
                    self.calls.append(("broken", mode, role))
                    raise ValueError("runtime unavailable")
                return super().__call__(configuration, mode=mode, role=role)

        with self.assertRaisesRegex(QualificationBatchError, "runtime unavailable"):
            self._run(Broken(_always(True)))
        document = load_batch_checkpoint(self.checkpoint)
        states = [record["state"] for record in document["attempts"]["records"]]
        self.assertEqual(states, ["completed", "completed", "failed"])
        failed = document["attempts"]["records"][2]
        self.assertIsNotNone(failed["started_at"])
        self.assertIn("runtime unavailable", document["results"][failed["scope"]]["error"])
        resumed = RecordingFactory(_always(True))
        self.assertEqual(self._run(resumed, None).dispatched, 13)

    def test_resume_refuses_a_different_design(self) -> None:
        with self.assertRaises(AttemptCapExceeded):
            self._run(RecordingFactory(_always(True)), AttemptBudget(plan_generations=0, executions=4))
        other = dev_four_arm_design(
            self.tasks,
            healthy=self.healthy,
            noop=self.noop,
            noop_fault=_noop_fault(),
            regression=self.regression,
            regression_fault=_regression_fault(),
            arm_orders=seeded_arm_orders(4, 18),
        )
        with self.assertRaisesRegex(QualificationBatchError, "different batch design"):
            open_batch(other, self.checkpoint, budget=None)

    def test_a_candidate_must_be_its_declared_fault(self) -> None:
        with self.assertRaisesRegex(QualificationBatchError, "declared fault"):
            dev_four_arm_design(
                self.tasks,
                healthy=self.healthy,
                noop=self.regression,
                noop_fault=_noop_fault(),
                regression=self.regression,
                regression_fault=_regression_fault(),
                arm_orders=self.orders,
            )


def _plan_evidence(task_set: TaskSet) -> PlanEvidenceInputs:
    return PlanEvidenceInputs(
        plan_format_version="plan-v1",
        plan_quality_features=(
            "requirement_coverage_fraction",
            "entity_coverage_fraction",
            "dependency_consistency_fraction",
            "required_tool_coverage_fraction",
            "invalid_tool_reference_fraction",
        ),
        plan_quality_weights=(0.25, 0.25, 0.25, 0.25, -1.0),
        mmd_features=(
            "requirement_coverage_fraction",
            "entity_coverage_fraction",
            "dependency_consistency_fraction",
            "tool_reference_fraction",
            "required_tool_coverage_fraction",
        ),
        kl_approximation="top_k",
        required_statistics=("plan_quality", "mmd"),
        validation_provenance="synthetic_fixture",
        task_plan_specs=tuple(
            TaskPlanSpec(
                task_id=task_id,
                available_tools=("calendar.open_calendar", "calendar.create_event"),
                subgoal_keywords=(("open the calendar",), ("create an event", "add an event")),
                required_entities=("calendar",),
                dependency_pairs=(("calendar.open_calendar", "calendar.create_event"),),
            )
            for task_id in task_set.task_ids
        ),
    )


_SETTINGS = GateSettings(
    confidence_level=0.95,
    bootstrap_resamples=200,
    score_margin=-0.1,
    kl_limit_nats=0.0,
    mmd_bandwidth=1.0,
    mmd_permutations=99,
    mmd_alpha=0.05,
    plan_format_version="plan-v1",
)


def _plan_text(index: int) -> str:
    return (_GOOD_PLAN, _WEAK_PLAN, _BAD_PLAN)[(index * 7) % 3]


class PlanAATests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.tasks = _task_set("train", 7)
        self.healthy = _configuration(self.tasks)
        self.design = plan_aa_design(self.tasks, healthy=self.healthy)
        self.checkpoint = self.root / "plan.json"
        self.evidence = _plan_evidence(self.tasks)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _run(
        self, factory, budget: AttemptBudget | None = AttemptBudget(plan_generations=14, executions=0)
    ):
        return run_qualification_batch(
            self.design, runtime_factory=factory, checkpoint_path=self.checkpoint, budget=budget
        )

    def test_two_plan_generations_per_task_and_no_executions(self) -> None:
        factory = RecordingFactory(_always(True), plans=_plan_text)
        summary = self._run(factory)
        self.assertEqual(summary.dispatched, 7)
        self.assertEqual(len(factory.calls), 14)
        self.assertEqual({call[1] for call in factory.calls}, {"plan"})
        self.assertEqual(
            [call[2] for call in factory.calls], ["reference", "candidate"] * 7
        )
        self.assertEqual(summary.attempts["plan"]["completed"], 14)
        self.assertEqual(summary.attempts["execute"]["consumed"], 0)
        document = load_batch_checkpoint(self.checkpoint)
        roles = [record["role"] for record in document["attempts"]["records"]]
        self.assertEqual(roles, list(PLAN_ARMS) * 7)

    def test_plan_caps_cannot_exceed_two_per_task_or_spend_executions(self) -> None:
        with self.assertRaisesRegex(QualificationBatchError, "replacement"):
            self._run(RecordingFactory(_always(True)), AttemptBudget(plan_generations=15, executions=0))
        with self.assertRaisesRegex(QualificationBatchError, "no execute"):
            self._run(RecordingFactory(_always(True)), AttemptBudget(plan_generations=14, executions=1))

    def test_a_crashed_plan_pair_is_never_regenerated(self) -> None:
        crashing = RecordingFactory(_always(True), plans=_plan_text, crash_on_call=6)
        with self.assertRaises(Crash):
            self._run(crashing)
        resumed = RecordingFactory(_always(True), plans=_plan_text)
        summary = self._run(resumed, None)
        self.assertEqual(summary.skipped, 3)
        self.assertEqual(len(resumed.calls), 8)
        self.assertEqual(summary.attempts["plan"]["interrupted"], 2)
        self.assertEqual(summary.attempts["plan"]["consumed"], 14)
        document = load_batch_checkpoint(self.checkpoint)
        self.assertIsNone(plan_pairs(self.design, document)[2])
        replay = plan_aa_gate_replay(
            self.design, document, settings=_SETTINGS, plan_evidence=self.evidence
        )
        self.assertEqual(replay, {"status": "incomplete", "missing_pairs": 1})

    def test_scores_and_vectors_are_the_gates_own(self) -> None:
        self._run(RecordingFactory(_always(True), plans=_plan_text))
        document = load_batch_checkpoint(self.checkpoint)
        pairs = plan_pairs(self.design, document)
        series = {item.name: item for item in plan_aa_series(self.design, document, self.evidence)}
        episodes = [episode for pair in pairs if pair is not None for episode in pair]
        quality = series["plan_quality"]
        self.assertEqual(
            list(quality.raw), [plan_quality_score(episode, self.evidence) for episode in episodes]
        )
        self.assertEqual(quality.bounds, (-1.0, 1.0))
        for observation, raw in zip(quality.observations, quality.raw, strict=True):
            self.assertAlmostEqual(observation.value, (raw + 1.0) / 2.0)
        self.assertGreater(len(set(quality.raw)), 1)
        for position, feature in enumerate(self.evidence.mmd_features):
            self.assertEqual(
                list(series[f"mmd:{feature}"].raw),
                [plan_representation(episode, self.evidence)[position] for episode in episodes],
            )
        self.assertEqual(quality.context.repetitions, (0, 1) * 7)

    def test_replay_matches_a_live_gate_on_the_same_plans(self) -> None:
        recorded = []
        original = gate_module.run_pair

        def recording(*args, **kwargs):
            pair = original(*args, **kwargs)
            recorded.append((pair.reference, pair.candidate))
            return pair

        factory = RecordingFactory(_always(True), plans=_plan_text)
        runtime = factory(self.healthy, mode="plan")
        texts = iter(_plan_text(index) for index in range(1, 40))

        class Rotating(Agent):
            def next_turn(self, *, tool_output):
                self._plans = [next(texts)]
                return super().next_turn(tool_output=tool_output)

        runtime = RuntimeDependencies(
            runtime.session_factory, Rotating(factory.clock, []), factory.clock
        )
        with mock.patch.object(gate_module, "run_pair", recording):
            decision = run_offline_gate(
                self.healthy,
                self.healthy,
                self.tasks,
                settings=_SETTINGS,
                runtime=runtime,
                plan_evidence=self.evidence,
            )
        replay = replay_plan_gate(
            self.healthy,
            self.healthy,
            self.tasks,
            recorded,
            settings=_SETTINGS,
            plan_evidence=self.evidence,
        )
        self.assertEqual(replay.outcome, decision.outcome)
        self.assertEqual(replay.reason_codes, decision.reason_codes)
        self.assertEqual(replay.statistics, decision.statistics)

    def test_both_gate_methods_receive_measurable_aa_evidence(self) -> None:
        self._run(RecordingFactory(_always(True), plans=_plan_text))
        document = load_batch_checkpoint(self.checkpoint)
        specs = [
            method_spec(
                "clustered_paired_bootstrap",
                {"confidence_level": 0.95, "resamples": 1000, "null_mean": 0.0, "null_scale": 1.0, "cluster_size": 1},
                required_checks=("aa_dependence",),
                study="simulation",
                null_sample_size=30,
                uncertainty_level=0.95,
                null_draw="gaussian",
                alpha=0.05,
            ),
            method_spec(
                "mmd_permutation_test",
                {"bandwidth": 1.0, "permutations": 199, "dimension": 5, "null_mean": 0.0, "null_scale": 1.0},
                required_checks=("aa_dependence",),
                study="simulation",
                null_sample_size=30,
                uncertainty_level=0.95,
                null_draw="gaussian",
                alpha=0.05,
            ),
        ]
        reports = plan_aa_reports(plan_aa_series(self.design, document, self.evidence), specs)
        self.assertEqual([item["series"] for item in reports["clustered_paired_bootstrap"]], ["plan_quality"])
        self.assertEqual(len(reports["mmd_permutation_test"]), 5)
        bootstrap = reports["clustered_paired_bootstrap"][0]["report"]
        self.assertEqual(bootstrap["status"], "passed", bootstrap["reason"])
        self.assertEqual(bootstrap["observation_count"], 14)
        for item in reports["mmd_permutation_test"]:
            self.assertEqual(item["report"]["observation_count"], 14)
            self.assertTrue(item["report"]["evidence_accepted"])
        replay = plan_aa_gate_replay(self.design, document, settings=_SETTINGS, plan_evidence=self.evidence)
        self.assertEqual(replay["status"], "decided")
        self.assertIn(replay["outcome"], {"PASS", "BLOCK"})

    def test_a_score_range_needs_fraction_features(self) -> None:
        self._run(RecordingFactory(_always(True), plans=_plan_text))
        document = load_batch_checkpoint(self.checkpoint)
        evidence = PlanEvidenceInputs(
            plan_format_version="plan-v1",
            plan_quality_features=("numbered_step_count",),
            plan_quality_weights=(1.0,),
            mmd_features=("char_count",),
            kl_approximation="top_k",
            required_statistics=("plan_quality", "mmd"),
            validation_provenance="synthetic_fixture",
        )
        with self.assertRaisesRegex(QualificationBatchError, "bounded fraction"):
            plan_aa_series(self.design, document, evidence)


class HarmLabelReuseTests(unittest.TestCase):
    def test_outcome_label_equals_measure_harm_on_the_same_outcomes(self) -> None:
        tasks = _task_set("dev", 6)
        healthy = _configuration(tasks)
        candidate = apply_fault(healthy, _regression_fault())
        pattern = [True, True, False, True, True, True, False, False, True, False, True, True]
        worlds = iter(pattern)
        clock = Clock()
        runtime = RuntimeDependencies(
            session_factory=lambda task_id: World(task_id, next(worlds)),
            agent=Agent(clock, []),
            clock=clock,
        )
        measured = measure_harm(
            healthy,
            candidate,
            tasks,
            margin=0.2,
            runtime=runtime,
            fault=_regression_fault(),
            confidence_level=0.9,
            resamples=300,
            seed=5,
        )
        recomputed = harm_label_from_outcomes(
            healthy,
            candidate,
            tasks,
            fault=_regression_fault(),
            base_successes=pattern[0::2],
            candidate_successes=pattern[1::2],
            margin=0.2,
            confidence_level=0.9,
            resamples=300,
            seed=5,
        )
        self.assertEqual(recomputed, measured)


if __name__ == "__main__":
    unittest.main()
