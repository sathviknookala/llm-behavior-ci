import json
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from llm_behavior_ci.config import (
    MonitorSettings,
    RunConfiguration,
    StoppingRule,
    StreamSettings,
    run_configuration_hash,
)
from llm_behavior_ci.experiments.schedule import (
    BenchmarkSchedule,
    ScheduleError,
    SimulatedClock,
    exposure_for,
    plan_arrivals,
    run_scheduled_monitor,
    schedule_hash,
)
from llm_behavior_ci.lifecycle.monitoring import FrozenReference, ProductionMonitor
from llm_behavior_ci.records import TokenLogprob
from llm_behavior_ci.runtime.agent import AgentTurn
from llm_behavior_ci.runtime.appworld import EvaluationResult, TaskContext, ToolResult
from llm_behavior_ci.runtime.episode import RuntimeDependencies
from llm_behavior_ci.tasks.selection import (
    TaskSet,
    canonical_task_set_bytes,
    task_set_hash_from_bytes,
)

_START = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
_LOGPROBS = ((TokenLogprob(token_id=7, logprob=-0.4, rank=0),),)
_TASKS = (("t-a1", "s-a"), ("t-a2", "s-a"), ("t-b1", "s-b"), ("t-b2", "s-b"))


def _task_set() -> TaskSet:
    digest = task_set_hash_from_bytes(
        canonical_task_set_bytes(
            appworld_version="0.1.3.post1",
            split="dev",
            selection_rule="deterministic_sample",
            selection_seed=11,
            tasks=_TASKS,
        )
    )
    return TaskSet(
        appworld_version="0.1.3.post1",
        split="dev",
        selection_rule="deterministic_sample",
        selection_seed=11,
        task_count=len(_TASKS),
        scenario_count=2,
        task_ids=tuple(task for task, _ in _TASKS),
        scenario_ids=tuple(scenario for _, scenario in _TASKS),
        task_set_hash=digest,
    )


def _configs(task_set: TaskSet) -> tuple[RunConfiguration, RunConfiguration]:
    payload = {
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
            "appworld_version": task_set.appworld_version,
            "split": task_set.split,
            "selection_rule": task_set.selection_rule,
            "selection_seed": task_set.selection_seed,
            "task_count": task_set.task_count,
            "task_set_hash": task_set.task_set_hash,
        },
        "run_seed": 7,
        "git_commit": "a" * 40,
    }
    healthy = RunConfiguration.from_dict(payload)
    faulted = replace(
        healthy,
        agent=replace(
            healthy.agent,
            prompt=replace(healthy.agent.prompt, prompt_version="prompt-v2"),
        ),
    )
    return healthy, faulted


def _schedule(
    task_set: TaskSet,
    *,
    prefix: int = 6,
    horizon: int = 16,
    onset_mode: str = "abrupt",
    ramp: int = 0,
    with_replacement: bool = True,
) -> BenchmarkSchedule:
    return BenchmarkSchedule(
        stream=StreamSettings(
            split=task_set.split,
            selection_rule=task_set.selection_rule,
            selection_seed=task_set.selection_seed,
            task_set_hash=task_set.task_set_hash,
            stream_seed=5,
            arrival_rate_per_second=0.5,
            concurrency=1,
            with_replacement=with_replacement,
            task_mix_rule="uniform",
        ),
        healthy_prefix_episodes=prefix,
        onset_mode=onset_mode,
        ramp_episodes=ramp,
        analysis_horizon_episodes=horizon,
        canary_fraction=0.5,
        canary_assignment_seed=3,
        clock_start=_START,
    )


class _Clock:
    def __init__(self) -> None:
        self.now = _START

    def __call__(self) -> datetime:
        value = self.now
        self.now += timedelta(milliseconds=10)
        return value


class _World:
    def __init__(self, task_id: str, success: bool) -> None:
        self.task_id = task_id
        self.success = success

    def context(self) -> TaskContext:
        return TaskContext(task_id=self.task_id, instruction="do it", api_documentation="docs")

    def execute(self, action: str) -> ToolResult:
        del action
        return ToolResult(
            output_text="ok", error_message=None, recoverable=False, app_name=None, api_name=None
        )

    def evaluate(self) -> EvaluationResult:
        return EvaluationResult(
            success=self.success,
            passed_requirements=1 if self.success else 0,
            total_requirements=1,
            difficulty=1 if self.task_id.startswith("t-a") else 2,
        )

    def close(self) -> None:
        pass


class _Agent:
    def __init__(self, clock: _Clock) -> None:
        self._clock = clock
        self._step = 0

    def begin(self, context: TaskContext, config: RunConfiguration) -> None:
        del context, config
        self._step = 0

    def next_turn(self, *, tool_output: str | None) -> AgentTurn:
        del tool_output
        self._step += 1
        done = self._step > 1
        return AgentTurn(
            prompt_text="p",
            output_text="STOP" if done else "lookup()",
            top_k_logprobs=_LOGPROBS,
            generated_token_count=1,
            latency_seconds=0.1,
            started_at=self._clock(),
            action=None if done else "lookup()",
            app_name=None,
            api_name=None if done else "lookup",
        )


def _monitor(healthy: RunConfiguration, clock: SimulatedClock) -> ProductionMonitor:
    digest = run_configuration_hash(healthy)
    return ProductionMonitor(
        MonitorSettings(
            reference_configuration_hash=digest,
            outcome_delay_seconds=0.0,
            signals=("task_success",),
            stopping_rules=(
                StoppingRule(name="cusum", alpha=0.1, horizon_episodes=50, threshold=1.5),
            ),
        ),
        FrozenReference(configuration_hash=digest, baselines=(("task_success", 0.9),)),
        clock=clock,
        dedup_seconds=0.0,
    )


def _run(schedule, task_set, healthy, faulted, state, *, healthy_success=True, interrupt=None):
    episode_clock = _Clock()
    healthy_hash = run_configuration_hash(healthy)

    def runtime_for(configuration):
        success = healthy_success if run_configuration_hash(configuration) == healthy_hash else False
        return RuntimeDependencies(
            session_factory=lambda task_id: _World(task_id, success),
            agent=_Agent(episode_clock),
            clock=episode_clock,
        )

    clock = SimulatedClock(schedule.clock_start)
    return run_scheduled_monitor(
        schedule=schedule,
        arrivals=plan_arrivals(task_set, schedule),
        healthy=healthy,
        faulted=faulted,
        runtime_for=runtime_for,
        monitor=_monitor(healthy, clock),
        distributional={},
        clock=clock,
        state=state,
        persist=lambda: None,
        difficulty_for=lambda task_id: 1 if task_id.startswith("t-a") else 2,
        should_interrupt=interrupt,
    )


class ScheduleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task_set = _task_set()
        self.healthy, self.faulted = _configs(self.task_set)

    def test_plan_is_deterministic_and_hash_binds_inputs(self) -> None:
        schedule = _schedule(self.task_set)
        self.assertEqual(plan_arrivals(self.task_set, schedule), plan_arrivals(self.task_set, schedule))
        self.assertEqual(schedule_hash(schedule), schedule_hash(BenchmarkSchedule.from_dict(schedule.to_dict())))
        self.assertNotEqual(schedule_hash(schedule), schedule_hash(_schedule(self.task_set, prefix=5)))
        arrivals = plan_arrivals(self.task_set, schedule)
        self.assertEqual(len(arrivals), 16)
        self.assertTrue(all(row.exposure == "healthy" for row in arrivals[:6]))
        self.assertTrue(all(row.exposure == "faulted" for row in arrivals[6:]))
        self.assertTrue(all(row.phase == "healthy_prefix" for row in arrivals[:6]))
        self.assertEqual(arrivals[3].simulated_at, _START + timedelta(seconds=6))
        self.assertNotIn("task_id", arrivals[0].decision_dict())

    def test_ramp_mixes_then_saturates(self) -> None:
        schedule = _schedule(self.task_set, onset_mode="ramp", ramp=40, horizon=60)
        exposures = [exposure_for(schedule, index) for index in range(60)]
        self.assertTrue(all(value == "healthy" for value in exposures[:6]))
        self.assertTrue(all(value == "faulted" for value in exposures[46:]))
        ramp = exposures[6:46]
        self.assertIn("healthy", ramp)
        self.assertIn("faulted", ramp)
        self.assertLess(ramp[:20].count("faulted"), ramp[20:].count("faulted"))
        with self.assertRaises(ScheduleError):
            _schedule(self.task_set, onset_mode="ramp", ramp=0)
        with self.assertRaises(ScheduleError):
            _schedule(self.task_set, onset_mode="abrupt", ramp=3)

    def test_stream_shorter_than_horizon_is_refused(self) -> None:
        with self.assertRaises(ScheduleError):
            plan_arrivals(self.task_set, _schedule(self.task_set, prefix=2, horizon=5, with_replacement=False))

    def test_resumed_run_equals_uninterrupted(self) -> None:
        schedule = _schedule(self.task_set)
        whole = _run(schedule, self.task_set, self.healthy, self.faulted, {})
        state: dict = {}
        partial = _run(
            schedule, self.task_set, self.healthy, self.faulted, state, interrupt=lambda index: index == 9
        )
        self.assertEqual(partial.status, "interrupted")
        restored = json.loads(json.dumps(state))
        resumed = _run(schedule, self.task_set, self.healthy, self.faulted, restored)
        self.assertEqual(resumed.status, "completed")
        self.assertEqual(resumed.alerts, whole.alerts)
        self.assertEqual(resumed.counters, whole.counters)
        self.assertEqual(resumed.post_onset_delay_episodes, whole.post_onset_delay_episodes)
        self.assertEqual(whole.healthy_prefix_alarms, 0)
        self.assertTrue(whole.detected())
        self.assertEqual(whole.counters.arrivals, 16)
        self.assertEqual(whole.counters.healthy_exposures, 6)
        self.assertEqual(whole.counters.faulted_exposures, 10)
        self.assertEqual(whole.counters.completed_evaluator_outcomes, 16)
        self.assertEqual(whole.counters.candidate_exposures, 0)
        stored = restored["items"][0]
        self.assertNotIn("task_id", stored)

    def test_checkpoint_under_another_schedule_is_refused(self) -> None:
        state: dict = {}
        _run(_schedule(self.task_set), self.task_set, self.healthy, self.faulted, state)
        with self.assertRaises(ScheduleError):
            _run(_schedule(self.task_set, prefix=5), self.task_set, self.healthy, self.faulted, state)

    def test_healthy_prefix_alarms_are_reported_apart(self) -> None:
        result = _run(
            _schedule(self.task_set), self.task_set, self.healthy, self.faulted, {}, healthy_success=False
        )
        self.assertGreater(result.healthy_prefix_alarms, 0)
        onset_alerts = [alert for alert in result.alerts if alert["index"] < 6]
        self.assertEqual(len(onset_alerts), result.healthy_prefix_alarms)


if __name__ == "__main__":
    unittest.main()
