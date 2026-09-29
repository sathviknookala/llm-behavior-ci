import hashlib
import io
import json
import os
import subprocess
import sys
import threading
import time
import unittest
import unittest.mock
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from llm_behavior_ci.config import RunConfiguration, StreamSettings, new_run_identity
from llm_behavior_ci.records import RecordError, assert_public_payload
from llm_behavior_ci.records import TokenLogprob
from llm_behavior_ci.runtime.aa_capture import (
    AllowedGpuProcess,
    GpuBusy,
    GpuProcess,
    GpuSnapshot,
    assert_gpu_processes_allowed,
    capture_aa,
    format_summary,
    main,
    parse_gpu_snapshot,
    repeated_schedule,
    write_local_capture,
)
from llm_behavior_ci.runtime.agent import AgentTurn, SmolagentsVLLMAgent
from llm_behavior_ci.runtime.appworld import EvaluationResult, TaskContext, ToolResult
from llm_behavior_ci.runtime.episode import (
    EpisodeRejected,
    RuntimeDependencies,
    RuntimeUnavailable,
    build_runtime,
    evaluator_difference,
    pair_execution,
    run_pair,
)
from llm_behavior_ci.tasks.catalog import CatalogEntry, TaskCatalog
from llm_behavior_ci.tasks.selection import select_task_set
from llm_behavior_ci.tasks.streams import generate_stream

_START = datetime(2026, 9, 27, 15, 0, tzinfo=timezone.utc)
_ROOT = Path(__file__).resolve().parents[2]
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
            "step_limit": 4,
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


def _config() -> RunConfiguration:
    return RunConfiguration.from_dict(_payload())


class Clock:
    def __init__(self) -> None:
        self.current = _START

    def __call__(self) -> datetime:
        value = self.current
        self.current += timedelta(seconds=1)
        return value


def _turn(
    output_text: str,
    action: str | None,
    api_name: str | None,
    started_at: datetime,
) -> AgentTurn:
    return AgentTurn(
        prompt_text="plan the next action",
        output_text=output_text,
        top_k_logprobs=_LOGPROBS,
        latency_seconds=0.1,
        started_at=started_at,
        action=action,
        app_name=None,
        api_name=api_name,
    )


class World:
    def __init__(
        self,
        task_id: str,
        token: str | None,
        *,
        fail: bool = False,
        zero_total: bool = False,
    ) -> None:
        self.task_id = task_id
        self.token = token
        self.fail = fail
        self.zero_total = zero_total
        self.seen: list[str] = []
        self.close_count = 0
        self.execute_count = 0
        self.evaluate_count = 0

    def initial_state_identity(self) -> str:
        if self.token is None:
            raise AssertionError("fallback should not ask for an identity")
        return self.token

    def context(self) -> TaskContext:
        return TaskContext(
            task_id=self.task_id,
            instruction="solve the task",
            api_documentation="calendar docs",
        )

    def execute(self, action: str) -> ToolResult:
        self.execute_count += 1
        self.seen.append(action)
        if self.fail:
            raise RuntimeError("candidate failed")
        return ToolResult(
            output_text=f"ok:{action}",
            error_message=None,
            recoverable=False,
            app_name=None,
            api_name=None,
        )

    def evaluate(self) -> EvaluationResult:
        self.evaluate_count += 1
        if self.zero_total:
            return EvaluationResult(
                success=True,
                passed_requirements=0,
                total_requirements=0,
                difficulty=1,
            )
        if self.seen and self.seen[-1] == "candidate.lookup()":
            return EvaluationResult(
                success=False,
                passed_requirements=0,
                total_requirements=1,
                difficulty=1,
            )
        return EvaluationResult(
            success=True,
            passed_requirements=1,
            total_requirements=1,
            difficulty=1,
        )

    def close(self) -> None:
        self.close_count += 1


class PlainWorld:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.close_count = 0

    def context(self) -> TaskContext:
        return TaskContext(
            task_id=self.task_id,
            instruction="solve the task",
            api_documentation="calendar docs",
        )

    def execute(self, action: str) -> ToolResult:
        del action
        return ToolResult(
            output_text="ok",
            error_message=None,
            recoverable=False,
            app_name=None,
            api_name=None,
        )

    def evaluate(self) -> EvaluationResult:
        return EvaluationResult(
            success=True,
            passed_requirements=1,
            total_requirements=1,
            difficulty=1,
        )

    def close(self) -> None:
        self.close_count += 1


class PairAgent:
    def __init__(
        self,
        clock: Clock,
        *,
        mode: str,
        diverge: bool,
        endpoint: str | None = None,
        force_candidate: bool | None = None,
    ) -> None:
        self._clock = clock
        self._mode = mode
        self._diverge = diverge
        self.endpoint = endpoint
        self._force_candidate = force_candidate
        self._begins = 0
        self._step = 0
        self._reference = True
        self.begins = 0
        self.began_endpoints: list[str | None] = []
        self.tool_outputs: list[list[str | None]] = []

    def begin(self, context: TaskContext, config: RunConfiguration) -> None:
        del context, config
        self.begins += 1
        self._begins += 1
        if self._force_candidate is None:
            self._reference = self._begins % 2 == 1
        else:
            self._reference = not self._force_candidate
        self._step = 0
        self.began_endpoints.append(self.endpoint)
        self.tool_outputs.append([])

    def next_turn(self, *, tool_output: str | None) -> AgentTurn:
        self.tool_outputs[-1].append(tool_output)
        self._step += 1
        started_at = self._clock()
        if self._mode == "plan":
            if self._reference or not self._diverge:
                text = "1. reference step"
            else:
                text = "1. candidate step"
            return _turn(text, None, None, started_at)
        if self._step > 1:
            return _turn("STOP", None, None, started_at)
        if self._reference or not self._diverge:
            return _turn(
                "reference.lookup()",
                "reference.lookup()",
                "reference_lookup",
                started_at,
            )
        return _turn(
            "candidate.lookup()",
            "candidate.lookup()",
            "candidate_lookup",
            started_at,
        )


def _runtime(
    worlds: list[World],
    agent: PairAgent,
    clock: Clock,
    *,
    token: str = "same-state",
    fail_second: bool = False,
    zero_total: bool = False,
) -> RuntimeDependencies:
    def factory(task_id: str) -> World:
        fail = fail_second and len(worlds) == 1
        world = World(task_id, token, fail=fail, zero_total=zero_total)
        worlds.append(world)
        return world

    return RuntimeDependencies(session_factory=factory, agent=agent, clock=clock)


def _catalog() -> TaskCatalog:
    return TaskCatalog(
        appworld_version="0.1.3.post1",
        entries=(
            CatalogEntry("task-a", "scenario-1", "train", 1),
            CatalogEntry("task-b", "scenario-1", "train", 2),
            CatalogEntry("task-c", "scenario-2", "dev", 1),
        ),
    )


def _task_set():
    return select_task_set(
        _catalog(),
        split="train",
        selection_rule="deterministic_sample",
        seed=7,
        count=2,
    )


def _stream_settings(task_set) -> StreamSettings:
    return StreamSettings(
        split=task_set.split,
        selection_rule=task_set.selection_rule,
        selection_seed=task_set.selection_seed,
        task_set_hash=task_set.task_set_hash,
        stream_seed=3,
        arrival_rate_per_second=2.0,
        concurrency=4,
        with_replacement=False,
        task_mix_rule="uniform",
    )


def _stream_config(task_set) -> RunConfiguration:
    payload = _payload()
    payload["task"] = {
        "appworld_version": task_set.appworld_version,
        "split": task_set.split,
        "selection_rule": task_set.selection_rule,
        "selection_seed": task_set.selection_seed,
        "task_count": task_set.task_count,
        "task_set_hash": task_set.task_set_hash,
    }
    return RunConfiguration.from_dict(payload)


class PairedExecutionTests(unittest.TestCase):
    def test_shared_pair_identity(self) -> None:
        config = _config()
        reference_run = new_run_identity(config)
        candidate_run = new_run_identity(config)
        clock = Clock()
        worlds: list[World] = []
        agent = PairAgent(clock, mode="execute", diverge=False)
        pair = run_pair(
            "task-1",
            config,
            config,
            reference_run=reference_run,
            candidate_run=candidate_run,
            runtime=_runtime(worlds, agent, clock),
            mode="execute",
            scenario_id="scenario-1",
        )
        record = pair_execution(pair)
        self.assertEqual(pair.reference.episode.pair_id, pair.candidate.episode.pair_id)
        self.assertIsNotNone(pair.reference.episode.pair_id)
        self.assertNotEqual(
            pair.reference.episode.episode_id,
            pair.candidate.episode.episode_id,
        )
        self.assertEqual(pair.reference.role, "reference")
        self.assertEqual(pair.candidate.role, "candidate")
        self.assertEqual(pair.reference.task, pair.candidate.task)
        self.assertEqual(pair.reference.task.scenario_id, "scenario-1")
        self.assertEqual(record.pair_id, pair.reference.episode.pair_id)
        self.assertEqual(record.execution_order, ("reference", "candidate"))
        self.assertLess(pair.reference.started_at, pair.candidate.started_at)
        self.assertEqual(record.reference_seed, 17)
        self.assertEqual(record.candidate_seed, 17)
        self.assertEqual(record.reference_run_seed, 7)
        self.assertEqual(record.candidate_run_seed, 7)
        self.assertEqual(record.initial_state_identity, "same-state")
        self.assertEqual(len(worlds), 2)
        self.assertIsNot(worlds[0], worlds[1])

    def test_separate_worlds_and_independent_mutations(self) -> None:
        config = _config()
        clock = Clock()
        worlds: list[World] = []
        agent = PairAgent(clock, mode="execute", diverge=True)
        pair = run_pair(
            "task-1",
            config,
            config,
            reference_run=new_run_identity(config),
            candidate_run=new_run_identity(config),
            runtime=_runtime(worlds, agent, clock),
            mode="execute",
        )
        self.assertEqual(worlds[0].seen, ["reference.lookup()"])
        self.assertEqual(worlds[1].seen, ["candidate.lookup()"])
        self.assertEqual(worlds[0].execute_count, 1)
        self.assertEqual(worlds[1].execute_count, 1)
        self.assertEqual(worlds[0].close_count, 1)
        self.assertEqual(worlds[1].close_count, 1)
        self.assertEqual(agent.tool_outputs[0][0], None)
        self.assertEqual(agent.tool_outputs[1][0], None)
        self.assertEqual(agent.tool_outputs[0][1], "ok:reference.lookup()")
        self.assertEqual(agent.tool_outputs[1][1], "ok:candidate.lookup()")
        difference = evaluator_difference(pair)
        self.assertIsNotNone(difference)
        assert difference is not None
        self.assertEqual(difference.success_difference, -1)
        self.assertEqual(difference.requirement_fraction_difference, -1.0)
        self.assertEqual(pair.candidate.status, "completed")
        self.assertIsNotNone(pair.candidate.evaluator_outcome)

    def test_candidate_failure_is_preserved(self) -> None:
        config = _config()
        clock = Clock()
        worlds: list[World] = []
        agent = PairAgent(clock, mode="execute", diverge=False)
        pair = run_pair(
            "task-1",
            config,
            config,
            reference_run=new_run_identity(config),
            candidate_run=new_run_identity(config),
            runtime=_runtime(worlds, agent, clock, fail_second=True),
            mode="execute",
        )
        self.assertEqual(pair.reference.status, "completed")
        self.assertIsNotNone(pair.reference.evaluator_outcome)
        self.assertEqual(pair.candidate.status, "failed")
        self.assertEqual(pair.candidate.termination_reason, "runtime_error")
        self.assertIsNone(pair.candidate.evaluator_outcome)
        self.assertEqual(pair.candidate.episode_errors[0].message, "candidate failed")
        self.assertIsNone(evaluator_difference(pair))
        self.assertEqual(worlds[0].seen, ["reference.lookup()"])
        self.assertEqual(worlds[1].seen, ["reference.lookup()"])
        self.assertEqual(worlds[0].evaluate_count, 1)
        self.assertEqual(worlds[1].evaluate_count, 0)
        self.assertEqual(worlds[0].close_count, 1)
        self.assertEqual(worlds[1].close_count, 1)

    def test_missing_fraction_is_not_zero(self) -> None:
        config = _config()
        clock = Clock()
        worlds: list[World] = []
        agent = PairAgent(clock, mode="execute", diverge=False)
        pair = run_pair(
            "task-1",
            config,
            config,
            reference_run=new_run_identity(config),
            candidate_run=new_run_identity(config),
            runtime=_runtime(worlds, agent, clock, zero_total=True),
            mode="execute",
        )
        difference = evaluator_difference(pair)
        self.assertIsNotNone(difference)
        assert difference is not None
        self.assertEqual(difference.success_difference, 0)
        self.assertIsNone(difference.requirement_fraction_difference)
        self.assertIsNotNone(pair.reference.evaluator_outcome)
        assert pair.reference.evaluator_outcome is not None
        self.assertIsNone(pair.reference.evaluator_outcome.requirement_fraction)

    def test_different_run_identities(self) -> None:
        reference_config = _config()
        candidate_config = replace(reference_config, run_seed=8)
        reference_run = new_run_identity(reference_config)
        candidate_run = new_run_identity(candidate_config)
        clock = Clock()
        worlds: list[World] = []
        opened = {"count": 0}

        def factory(task_id: str) -> World:
            opened["count"] += 1
            world = World(task_id, "same-state")
            worlds.append(world)
            return world

        pair = run_pair(
            "task-1",
            reference_config,
            candidate_config,
            reference_run=reference_run,
            candidate_run=candidate_run,
            runtime=RuntimeDependencies(
                session_factory=factory,
                agent=PairAgent(clock, mode="execute", diverge=False),
                clock=clock,
            ),
            mode="execute",
        )
        self.assertIs(pair.reference.run, reference_run)
        self.assertIs(pair.candidate.run, candidate_run)
        self.assertNotEqual(reference_run.run_id, candidate_run.run_id)
        self.assertNotEqual(
            reference_run.configuration_hash,
            candidate_run.configuration_hash,
        )
        self.assertNotEqual(
            pair.reference.episode.episode_id,
            pair.candidate.episode.episode_id,
        )
        self.assertEqual(pair.reference.execution_seed, pair.candidate.execution_seed)
        same = new_run_identity(reference_config)
        with self.assertRaises(EpisodeRejected):
            run_pair(
                "task-1",
                reference_config,
                reference_config,
                reference_run=same,
                candidate_run=same,
                runtime=RuntimeDependencies(
                    session_factory=factory,
                    agent=PairAgent(Clock(), mode="execute", diverge=False),
                    clock=Clock(),
                ),
                mode="execute",
            )
        self.assertEqual(opened["count"], 2)

    def test_incompatible_task_set_opens_no_world(self) -> None:
        reference = _config()
        candidate = replace(
            reference,
            task=replace(reference.task, task_set_hash="d" * 64),
        )
        opened: list[str] = []
        with self.assertRaisesRegex(EpisodeRejected, "task sets are not compatible"):
            run_pair(
                "task-1",
                reference,
                candidate,
                reference_run=new_run_identity(reference),
                candidate_run=new_run_identity(candidate),
                runtime=RuntimeDependencies(
                    session_factory=lambda task_id: opened.append(task_id),
                    agent=PairAgent(Clock(), mode="execute", diverge=False),
                    clock=Clock(),
                ),
                mode="execute",
            )
        self.assertEqual(opened, [])
        split_candidate = replace(reference, task=replace(reference.task, split="dev"))
        with self.assertRaisesRegex(EpisodeRejected, "task sets are not compatible"):
            run_pair(
                "task-1",
                reference,
                split_candidate,
                reference_run=new_run_identity(reference),
                candidate_run=new_run_identity(split_candidate),
                runtime=RuntimeDependencies(
                    session_factory=lambda task_id: opened.append(task_id),
                    agent=PairAgent(Clock(), mode="execute", diverge=False),
                    clock=Clock(),
                ),
                mode="execute",
            )
        self.assertEqual(opened, [])
        seeded = replace(
            reference,
            agent=replace(
                reference.agent,
                sampling=replace(reference.agent.sampling, seed=18),
            ),
        )
        with self.assertRaisesRegex(EpisodeRejected, "execution seed"):
            run_pair(
                "task-1",
                reference,
                seeded,
                reference_run=new_run_identity(reference),
                candidate_run=new_run_identity(seeded),
                runtime=RuntimeDependencies(
                    session_factory=lambda task_id: opened.append(task_id),
                    agent=PairAgent(Clock(), mode="execute", diverge=False),
                    clock=Clock(),
                ),
                mode="execute",
            )
        self.assertEqual(opened, [])

    def test_shared_world_and_mismatched_state_are_rejected(self) -> None:
        config = _config()
        shared = World("task-1", "same-state")
        began = {"count": 0}

        class MarkingAgent(PairAgent):
            def begin(self, context: TaskContext, config: RunConfiguration) -> None:
                began["count"] += 1
                super().begin(context, config)

        with self.assertRaisesRegex(EpisodeRejected, "separate"):
            run_pair(
                "task-1",
                config,
                config,
                reference_run=new_run_identity(config),
                candidate_run=new_run_identity(config),
                runtime=RuntimeDependencies(
                    session_factory=lambda task_id: shared,
                    agent=MarkingAgent(Clock(), mode="execute", diverge=False),
                    clock=Clock(),
                ),
                mode="execute",
            )
        self.assertEqual(shared.close_count, 1)
        self.assertEqual(began["count"], 0)
        tokens = iter(("state-a", "state-b"))
        closed: list[World] = []

        def factory(task_id: str) -> World:
            world = World(task_id, next(tokens))
            closed.append(world)
            return world

        with self.assertRaisesRegex(EpisodeRejected, "initial state"):
            run_pair(
                "task-1",
                config,
                config,
                reference_run=new_run_identity(config),
                candidate_run=new_run_identity(config),
                runtime=RuntimeDependencies(
                    session_factory=factory,
                    agent=PairAgent(Clock(), mode="execute", diverge=False),
                    clock=Clock(),
                ),
                mode="execute",
            )
        self.assertEqual([world.close_count for world in closed], [1, 1])

    def test_context_fallback_records_one_initial_state(self) -> None:
        config = _config()
        worlds: list[PlainWorld] = []

        def factory(task_id: str) -> PlainWorld:
            world = PlainWorld(task_id)
            worlds.append(world)
            return world

        clock = Clock()
        pair = run_pair(
            "task-1",
            config,
            config,
            reference_run=new_run_identity(config),
            candidate_run=new_run_identity(config),
            runtime=RuntimeDependencies(
                session_factory=factory,
                agent=PairAgent(clock, mode="plan", diverge=False),
                clock=clock,
            ),
            mode="plan",
        )
        expected = hashlib.sha256(
            b"task-1\nsolve the task\ncalendar docs"
        ).hexdigest()
        self.assertEqual(pair_execution(pair).initial_state_identity, expected)
        self.assertEqual(len(worlds), 2)
        self.assertEqual([world.close_count for world in worlds], [1, 1])

    def test_distinct_runtimes_keep_reference_off_candidate_endpoint(self) -> None:
        config = _config()
        clock = Clock()
        worlds: list[World] = []
        reference_agent = PairAgent(
            clock,
            mode="execute",
            diverge=False,
            endpoint="http://127.0.0.1:8000",
            force_candidate=False,
        )
        candidate_agent = PairAgent(
            clock,
            mode="execute",
            diverge=True,
            endpoint="http://127.0.0.1:8001",
            force_candidate=True,
        )

        def factory(task_id: str) -> World:
            world = World(task_id, "same-state")
            worlds.append(world)
            return world

        pair = run_pair(
            "task-1",
            config,
            config,
            reference_run=new_run_identity(config),
            candidate_run=new_run_identity(config),
            runtime=RuntimeDependencies(
                session_factory=factory,
                agent=reference_agent,
                clock=clock,
            ),
            candidate_runtime=RuntimeDependencies(
                session_factory=factory,
                agent=candidate_agent,
                clock=clock,
            ),
            mode="execute",
        )
        self.assertEqual(reference_agent.began_endpoints, ["http://127.0.0.1:8000"])
        self.assertEqual(candidate_agent.began_endpoints, ["http://127.0.0.1:8001"])
        self.assertEqual(reference_agent.begins, 1)
        self.assertEqual(candidate_agent.begins, 1)
        self.assertEqual(worlds[0].seen, ["reference.lookup()"])
        self.assertEqual(worlds[1].seen, ["candidate.lookup()"])
        self.assertEqual(pair.reference.role, "reference")
        self.assertEqual(pair.candidate.role, "candidate")
        built = build_runtime(config, "http://127.0.0.1:8000")
        self.assertIsInstance(built.agent, SmolagentsVLLMAgent)
        self.assertEqual(built.agent.base_url, "http://127.0.0.1:8000")


class AACaptureTests(unittest.TestCase):
    def _factory(
        self,
        *,
        diverge: bool,
        fail_second: bool = False,
        slow_task: str | None = None,
    ):
        lock = threading.Lock()
        calls = {"n": 0}

        def factory(mode: str) -> RuntimeDependencies:
            clock = Clock()
            agent = PairAgent(clock, mode=mode, diverge=diverge)

            def session_factory(task_id: str) -> World:
                if slow_task is not None and task_id == slow_task:
                    time.sleep(0.05)
                with lock:
                    calls["n"] += 1
                    fail = fail_second and calls["n"] % 2 == 0
                return World(task_id, f"state:{task_id}", fail=fail)

            return RuntimeDependencies(
                session_factory=session_factory,
                agent=agent,
                clock=clock,
            )

        return factory

    def test_reproducible_input_schedules(self) -> None:
        task_set = _task_set()
        settings = _stream_settings(task_set)
        arrivals = tuple(generate_stream(task_set, settings))
        self.assertEqual(
            repeated_schedule(arrivals, repetitions=2),
            repeated_schedule(arrivals, repetitions=2),
        )
        self.assertEqual(
            arrivals,
            tuple(generate_stream(task_set, replace(settings, concurrency=1))),
        )
        replay = tuple(generate_stream(task_set, settings))
        self.assertEqual(arrivals, replay)
        config = _stream_config(task_set)
        slow = arrivals[0].task_id
        first = capture_aa(
            config,
            arrivals,
            task_set_hash=task_set.task_set_hash,
            repetitions=2,
            concurrency=1,
            modes=("execute",),
            runtime_factory=self._factory(diverge=False, slow_task=slow),
            observe_hardware=False,
        )
        second = capture_aa(
            config,
            arrivals,
            task_set_hash=task_set.task_set_hash,
            repetitions=2,
            concurrency=2,
            modes=("execute",),
            runtime_factory=self._factory(diverge=False, slow_task=slow),
            observe_hardware=False,
        )
        self.assertEqual(first.schedule, second.schedule)
        self.assertEqual(
            [item.task_id for item in first.schedule],
            [arrival.task_id for arrival in arrivals] * 2,
        )
        self.assertEqual(
            [record.schedule.task_id for record in second.records],
            [item.task_id for item in second.schedule],
        )
        self.assertEqual(first.concurrency, 1)
        self.assertEqual(second.concurrency, 2)
        self.assertTrue(
            all(item.stream_seed == settings.stream_seed for item in first.schedule)
        )
        self.assertEqual(first.disagreement_count, 0)
        self.assertEqual(first.missing_outcome_count, 0)
        self.assertEqual(first.evaluated_pairs, len(first.records))
        self.assertFalse(first.cost.hardware_observed)
        self.assertIsNone(first.cost.memory_used_mib)
        self.assertIsNone(first.cost.wall_seconds)
        summary = format_summary(first)
        for task_id in task_set.task_ids:
            self.assertNotIn(task_id, summary)

    def test_capture_records_outcomes_divergence_and_plan_inputs(self) -> None:
        task_set = _task_set()
        arrivals = tuple(generate_stream(task_set, _stream_settings(task_set)))[:1]
        config = _stream_config(task_set)
        result = capture_aa(
            config,
            arrivals,
            task_set_hash=task_set.task_set_hash,
            repetitions=1,
            concurrency=1,
            modes=("plan", "execute"),
            runtime_factory=self._factory(diverge=True),
            observe_hardware=False,
        )
        self.assertEqual(len(result.records), 2)
        plan, execute = result.records
        self.assertEqual(plan.mode, "plan")
        self.assertEqual(execute.mode, "execute")
        self.assertIsNone(plan.evaluator_disagreement)
        self.assertIsNotNone(plan.plan_scoring_inputs)
        assert plan.plan_scoring_inputs is not None
        self.assertEqual(plan.plan_scoring_inputs.reference_plan_text, "1. reference step")
        self.assertEqual(plan.plan_scoring_inputs.candidate_plan_text, "1. candidate step")
        self.assertEqual(
            plan.plan_scoring_inputs.reference_steps[0][0][0],
            (7, -0.2),
        )
        self.assertIsNotNone(plan.teacher_forced_plan_kl)
        assert plan.teacher_forced_plan_kl is not None
        self.assertIsNone(plan.teacher_forced_plan_kl.mean_kl_nats)
        self.assertIsNone(plan.teacher_forced_plan_kl.frozen_plan_text)
        self.assertIsNone(execute.plan_scoring_inputs)
        self.assertIsNone(execute.teacher_forced_plan_kl)
        self.assertTrue(execute.evaluator_disagreement)
        self.assertEqual(execute.reference_requirement_fraction, 1.0)
        self.assertEqual(execute.candidate_requirement_fraction, 0.0)
        self.assertEqual(execute.trajectory.first_divergent_step, 0)
        self.assertEqual(execute.trajectory.length_difference, 0)
        self.assertEqual(
            execute.trajectory.reference_tool_counts,
            (("reference_lookup", 1),),
        )
        self.assertEqual(
            execute.trajectory.candidate_tool_counts,
            (("candidate_lookup", 1),),
        )
        self.assertEqual(plan.trajectory.first_divergent_step, 0)
        self.assertIsNotNone(result.tool_selection)
        assert result.tool_selection is not None
        self.assertGreater(result.tool_selection.statistic, 0.0)
        self.assertEqual(result.evaluated_pairs, 1)
        self.assertEqual(result.disagreement_count, 1)
        self.assertEqual(result.missing_outcome_count, 0)
        self.assertEqual(
            result.records[0].pair.reference.run.configuration_hash,
            result.records[0].pair.candidate.run.configuration_hash,
        )
        self.assertNotEqual(
            result.records[0].pair.reference.run.run_id,
            result.records[0].pair.candidate.run.run_id,
        )
        failed = capture_aa(
            config,
            arrivals,
            task_set_hash=task_set.task_set_hash,
            repetitions=1,
            concurrency=1,
            modes=("execute",),
            runtime_factory=self._factory(diverge=False, fail_second=True),
            observe_hardware=False,
        )
        self.assertEqual(len(failed.records), 1)
        self.assertEqual(failed.records[0].pair.candidate.status, "failed")
        self.assertIsNone(failed.records[0].evaluator_disagreement)
        self.assertIsNone(failed.records[0].candidate_requirement_fraction)
        self.assertEqual(failed.records[0].reference_requirement_fraction, 1.0)
        self.assertEqual(failed.disagreement_count, 0)
        self.assertEqual(failed.missing_outcome_count, 1)
        self.assertEqual(failed.evaluated_pairs, 0)

    def test_synthetic_capture_is_not_hardware_evidence(self) -> None:
        task_set = _task_set()
        arrivals = tuple(generate_stream(task_set, _stream_settings(task_set)))[:1]
        config = _stream_config(task_set)

        def probe() -> GpuSnapshot:
            raise AssertionError("synthetic capture must not probe the GPU")

        result = capture_aa(
            config,
            arrivals,
            task_set_hash=task_set.task_set_hash,
            repetitions=1,
            concurrency=1,
            modes=("execute",),
            runtime_factory=self._factory(diverge=False),
            observe_hardware=False,
            gpu_probe=probe,
        )
        self.assertFalse(result.cost.hardware_observed)
        self.assertIsNone(result.cost.memory_used_mib)
        self.assertIsNone(result.cost.memory_total_mib)
        self.assertIsNone(result.cost.wall_seconds)
        self.assertEqual(result.cost.source, "not_observed")
        calls = {"n": 0}

        def counting_probe() -> GpuSnapshot:
            calls["n"] += 1
            return GpuSnapshot(
                memory_used_mib=1000 + calls["n"],
                memory_total_mib=24576,
                process_count=0,
            )

        probed = capture_aa(
            config,
            arrivals,
            task_set_hash=task_set.task_set_hash,
            repetitions=1,
            concurrency=1,
            modes=("execute",),
            runtime_factory=self._factory(diverge=False),
            observe_hardware=True,
            gpu_probe=counting_probe,
        )
        self.assertFalse(probed.cost.hardware_observed)
        self.assertEqual(probed.cost.source, "probe")
        self.assertEqual(probed.cost.memory_used_mib, 1002)
        self.assertEqual(probed.cost.memory_total_mib, 24576)
        self.assertIsNotNone(probed.cost.wall_seconds)
        self.assertNotIn("memory_used_mib", format_summary(probed))

    def test_gpu_snapshot_text_is_not_a_zero_default(self) -> None:
        idle = parse_gpu_snapshot("100, 24576\n", "\n")
        self.assertEqual(idle.memory_used_mib, 100)
        self.assertEqual(idle.memory_total_mib, 24576)
        self.assertEqual(idle.process_count, 0)
        self.assertEqual(idle.processes, ())
        busy = parse_gpu_snapshot("100, 24576\n200, 24576\n", "42\n99\n")
        self.assertEqual(busy.memory_used_mib, 100)
        self.assertEqual(busy.process_count, 2)
        self.assertEqual(
            busy.processes,
            (GpuProcess(pid=42), GpuProcess(pid=99)),
        )
        named = parse_gpu_snapshot(
            "20000, 24576\n",
            "3897887, VLLM::EngineCore\n",
        )
        self.assertEqual(
            named.processes,
            (GpuProcess(pid=3897887, name="VLLM::EngineCore"),),
        )
        with self.assertRaises(RuntimeUnavailable):
            parse_gpu_snapshot("", "")
        with self.assertRaises(RuntimeUnavailable):
            parse_gpu_snapshot("N/A, 24576\n", "")

    def test_busy_gpu_does_not_start(self) -> None:
        task_set = _task_set()
        arrivals = tuple(generate_stream(task_set, _stream_settings(task_set)))[:1]
        config = _stream_config(task_set)
        started = {"n": 0}

        def factory(mode: str) -> RuntimeDependencies:
            started["n"] += 1
            return self._factory(diverge=False)(mode)

        def probe() -> GpuSnapshot:
            return GpuSnapshot(
                memory_used_mib=20000,
                memory_total_mib=24576,
                process_count=1,
                processes=(GpuProcess(pid=42, name="other"),),
            )

        with self.assertRaisesRegex(RuntimeError, "compute processes"):
            capture_aa(
                config,
                arrivals,
                task_set_hash=task_set.task_set_hash,
                repetitions=1,
                concurrency=1,
                modes=("execute",),
                runtime_factory=factory,
                observe_hardware=True,
                gpu_probe=probe,
            )
        self.assertEqual(started["n"], 0)
        with self.assertRaisesRegex(EpisodeRejected, "test_normal"):
            capture_aa(
                replace(config, task=replace(config.task, split="test_normal")),
                arrivals,
                task_set_hash=config.task.task_set_hash,
                repetitions=1,
                concurrency=1,
                modes=("execute",),
                runtime_factory=factory,
                observe_hardware=False,
            )
        self.assertEqual(started["n"], 0)

    def test_expected_gpu_process_allowance(self) -> None:
        task_set = _task_set()
        arrivals = tuple(generate_stream(task_set, _stream_settings(task_set)))[:1]
        config = _stream_config(task_set)
        baseline = GpuProcess(pid=3897887, name="VLLM::EngineCore")
        snapshots = [
            GpuSnapshot(
                memory_used_mib=19922,
                memory_total_mib=24576,
                process_count=1,
                processes=(baseline,),
            ),
            GpuSnapshot(
                memory_used_mib=19950,
                memory_total_mib=24576,
                process_count=1,
                processes=(baseline,),
            ),
        ]
        calls = {"n": 0}

        def probe() -> GpuSnapshot:
            index = min(calls["n"], len(snapshots) - 1)
            calls["n"] += 1
            return snapshots[index]

        by_pid = capture_aa(
            config,
            arrivals,
            task_set_hash=task_set.task_set_hash,
            repetitions=1,
            concurrency=1,
            modes=("execute",),
            runtime_factory=self._factory(diverge=False),
            observe_hardware=True,
            gpu_probe=probe,
            allowed_gpu_processes=(AllowedGpuProcess(pid=3897887),),
        )
        self.assertFalse(by_pid.cost.hardware_observed)
        self.assertEqual(by_pid.cost.source, "probe")
        self.assertEqual(by_pid.cost.initial, snapshots[0])
        self.assertEqual(by_pid.cost.final, snapshots[1])
        self.assertEqual(by_pid.cost.initial.process_count, 1)
        self.assertEqual(by_pid.cost.allowed_processes, (baseline,))
        self.assertNotEqual(by_pid.cost.initial.process_count, 0)

        calls["n"] = 0
        by_name = capture_aa(
            config,
            arrivals,
            task_set_hash=task_set.task_set_hash,
            repetitions=1,
            concurrency=1,
            modes=("execute",),
            runtime_factory=self._factory(diverge=False),
            observe_hardware=True,
            gpu_probe=probe,
            allowed_gpu_processes=(AllowedGpuProcess(name="VLLM::EngineCore"),),
        )
        self.assertEqual(by_name.cost.allowed_processes, (baseline,))
        self.assertEqual(by_name.cost.initial.process_count, 1)

        with self.assertRaises(GpuBusy):
            assert_gpu_processes_allowed(
                GpuSnapshot(
                    memory_used_mib=19922,
                    memory_total_mib=24576,
                    process_count=2,
                    processes=(
                        baseline,
                        GpuProcess(pid=99, name="unexpected"),
                    ),
                ),
                (AllowedGpuProcess(pid=3897887),),
            )
        with self.assertRaises(GpuBusy):
            assert_gpu_processes_allowed(
                GpuSnapshot(
                    memory_used_mib=19922,
                    memory_total_mib=24576,
                    process_count=1,
                    processes=(baseline,),
                ),
                (),
            )
        self.assertEqual(
            assert_gpu_processes_allowed(
                GpuSnapshot(
                    memory_used_mib=100,
                    memory_total_mib=24576,
                    process_count=0,
                ),
                (),
            ),
            (),
        )
        with self.assertRaises(GpuBusy):
            assert_gpu_processes_allowed(
                GpuSnapshot(
                    memory_used_mib=19922,
                    memory_total_mib=24576,
                    process_count=1,
                    processes=(GpuProcess(pid=1, name="python"),),
                ),
                (AllowedGpuProcess(name="VLLM::EngineCore"),),
            )

    def test_hardware_observed_only_for_nvidia_smi_source(self) -> None:
        task_set = _task_set()
        arrivals = tuple(generate_stream(task_set, _stream_settings(task_set)))[:1]
        config = _stream_config(task_set)
        idle = GpuSnapshot(memory_used_mib=100, memory_total_mib=24576, process_count=0)
        calls = {"n": 0}

        def probe() -> GpuSnapshot:
            calls["n"] += 1
            return idle

        probed = capture_aa(
            config,
            arrivals,
            task_set_hash=task_set.task_set_hash,
            repetitions=1,
            concurrency=1,
            modes=("execute",),
            runtime_factory=self._factory(diverge=False),
            observe_hardware=True,
            gpu_probe=probe,
        )
        self.assertFalse(probed.cost.hardware_observed)
        self.assertEqual(probed.cost.source, "probe")
        self.assertIsNotNone(probed.cost.initial)
        self.assertIsNotNone(probed.cost.final)

        from llm_behavior_ci.runtime import aa_capture as module

        with unittest.mock.patch.object(
            module,
            "read_nvidia_smi_snapshot",
            side_effect=[idle, idle],
        ):
            observed = capture_aa(
                config,
                arrivals,
                task_set_hash=task_set.task_set_hash,
                repetitions=1,
                concurrency=1,
                modes=("execute",),
                runtime_factory=self._factory(diverge=False),
                observe_hardware=True,
            )
        self.assertTrue(observed.cost.hardware_observed)
        self.assertEqual(observed.cost.source, "nvidia-smi")
        self.assertEqual(observed.cost.initial, idle)
        self.assertEqual(observed.cost.final, idle)

    def test_teacher_forced_plan_kl_uses_shared_frozen_plan(self) -> None:
        task_set = _task_set()
        arrivals = tuple(generate_stream(task_set, _stream_settings(task_set)))[:1]
        config = _stream_config(task_set)
        forced_plans: list[str] = []

        class TeacherForceAgent(PairAgent):
            def messages(self, tool_output: str | None = None) -> list[dict[str, str]]:
                del tool_output
                return [{"role": "user", "content": "emit a plan"}]

            def teacher_force_plan(
                self,
                *,
                messages: list[dict[str, str]],
                plan_text: str,
            ) -> tuple[tuple[TokenLogprob, ...], ...]:
                del messages
                forced_plans.append(plan_text)
                if len(forced_plans) == 1:
                    return (
                        (
                            TokenLogprob(token_id=1, logprob=-0.1, rank=0),
                            TokenLogprob(token_id=2, logprob=-2.3, rank=1),
                        ),
                    )
                return (
                    (
                        TokenLogprob(token_id=1, logprob=-0.4, rank=0),
                        TokenLogprob(token_id=2, logprob=-1.1, rank=1),
                    ),
                )

        def factory(mode: str) -> RuntimeDependencies:
            clock = Clock()
            return RuntimeDependencies(
                session_factory=lambda task_id: World(task_id, f"state:{task_id}"),
                agent=TeacherForceAgent(clock, mode=mode, diverge=True),
                clock=clock,
            )

        result = capture_aa(
            config,
            arrivals,
            task_set_hash=task_set.task_set_hash,
            repetitions=1,
            concurrency=1,
            modes=("plan",),
            runtime_factory=factory,
            observe_hardware=False,
        )
        record = result.records[0]
        self.assertIsNotNone(record.plan_scoring_inputs)
        assert record.plan_scoring_inputs is not None
        self.assertEqual(
            record.plan_scoring_inputs.reference_plan_text,
            "1. reference step",
        )
        self.assertEqual(
            record.plan_scoring_inputs.candidate_plan_text,
            "1. candidate step",
        )
        self.assertNotEqual(
            record.plan_scoring_inputs.reference_plan_text,
            record.plan_scoring_inputs.candidate_plan_text,
        )
        self.assertEqual(forced_plans, ["1. reference step", "1. reference step"])
        self.assertIsNotNone(record.teacher_forced_plan_kl)
        assert record.teacher_forced_plan_kl is not None
        self.assertEqual(
            record.teacher_forced_plan_kl.frozen_plan_text,
            "1. reference step",
        )
        self.assertIsNotNone(record.teacher_forced_plan_kl.mean_kl_nats)
        self.assertIsNotNone(record.teacher_forced_plan_kl.position_kl_nats)
        self.assertIsNotNone(record.teacher_forced_plan_kl.reference_top_k)
        self.assertIsNotNone(record.teacher_forced_plan_kl.candidate_top_k)
        summary = format_summary(result)
        self.assertNotIn("1. reference step", summary)
        self.assertNotIn("frozen_plan_text", summary)

    def test_teacher_forced_plan_kl_stays_empty_when_token_ids_differ(self) -> None:
        task_set = _task_set()
        arrivals = tuple(generate_stream(task_set, _stream_settings(task_set)))[:1]
        config = _stream_config(task_set)

        class MismatchedSupportAgent(PairAgent):
            def messages(self, tool_output: str | None = None) -> list[dict[str, str]]:
                del tool_output
                return [{"role": "user", "content": "emit a plan"}]

            def teacher_force_plan(
                self,
                *,
                messages: list[dict[str, str]],
                plan_text: str,
            ) -> tuple[tuple[TokenLogprob, ...], ...]:
                del messages, plan_text
                if not hasattr(self, "_forced"):
                    self._forced = True
                    return (
                        (
                            TokenLogprob(token_id=1, logprob=-0.1, rank=0),
                            TokenLogprob(token_id=2, logprob=-2.3, rank=1),
                        ),
                    )
                return (
                    (
                        TokenLogprob(token_id=1, logprob=-0.4, rank=0),
                        TokenLogprob(token_id=9, logprob=-1.1, rank=1),
                    ),
                )

        def factory(mode: str) -> RuntimeDependencies:
            clock = Clock()
            return RuntimeDependencies(
                session_factory=lambda task_id: World(task_id, f"state:{task_id}"),
                agent=MismatchedSupportAgent(clock, mode=mode, diverge=True),
                clock=clock,
            )

        result = capture_aa(
            config,
            arrivals,
            task_set_hash=task_set.task_set_hash,
            repetitions=1,
            concurrency=1,
            modes=("plan",),
            runtime_factory=factory,
            observe_hardware=False,
        )
        record = result.records[0]
        assert record.teacher_forced_plan_kl is not None
        self.assertEqual(
            record.teacher_forced_plan_kl.frozen_plan_text,
            "1. reference step",
        )
        self.assertIsNone(record.teacher_forced_plan_kl.mean_kl_nats)
        self.assertIsNone(record.teacher_forced_plan_kl.position_kl_nats)
        self.assertEqual(
            record.teacher_forced_plan_kl.reference_top_k,
            (((1, -0.1), (2, -2.3)),),
        )
        self.assertEqual(
            record.teacher_forced_plan_kl.candidate_top_k,
            (((1, -0.4), (9, -1.1)),),
        )

    def test_teacher_forced_plan_kl_missing_without_teacher_force(self) -> None:
        task_set = _task_set()
        arrivals = tuple(generate_stream(task_set, _stream_settings(task_set)))[:1]
        config = _stream_config(task_set)
        result = capture_aa(
            config,
            arrivals,
            task_set_hash=task_set.task_set_hash,
            repetitions=1,
            concurrency=1,
            modes=("plan",),
            runtime_factory=self._factory(diverge=True),
            observe_hardware=False,
        )
        record = result.records[0]
        self.assertIsNotNone(record.teacher_forced_plan_kl)
        assert record.teacher_forced_plan_kl is not None
        self.assertIsNone(record.teacher_forced_plan_kl.mean_kl_nats)
        self.assertIsNone(record.teacher_forced_plan_kl.frozen_plan_text)
        self.assertIsNone(record.teacher_forced_plan_kl.reference_top_k)

    def test_local_capture_is_not_a_public_result(self) -> None:
        task_set = _task_set()
        arrivals = tuple(generate_stream(task_set, _stream_settings(task_set)))[:1]
        config = _stream_config(task_set)
        result = capture_aa(
            config,
            arrivals,
            task_set_hash=task_set.task_set_hash,
            repetitions=1,
            concurrency=1,
            modes=("execute",),
            runtime_factory=self._factory(diverge=False),
            observe_hardware=False,
        )
        with self.subTest("writer"):
            import tempfile

            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "results"
                output = root / "aa-capture.json"
                with self.assertRaisesRegex(EpisodeRejected, "public result"):
                    write_local_capture(result, output, results_root=root)
                self.assertFalse(output.exists())
        with self.subTest("payload"):
            import tempfile

            with tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "capture.json"
                write_local_capture(
                    result,
                    output,
                    results_root=Path(temporary) / "results",
                )
                payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["visibility"], "local")
            with self.assertRaises(RecordError):
                assert_public_payload(payload)

    def test_command_requires_arguments_and_refuses_results(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/evaluation/capture_aa.py"],
            cwd=_ROOT,
            env={**os.environ, "PYTHONPATH": "src"},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        task_set = _task_set()
        config = _stream_config(task_set)
        settings = _stream_settings(task_set)
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "configuration.json"
            task_path = root / "task-set.json"
            stream_path = root / "stream.json"
            results_root = root / "results"
            output = results_root / "capture.json"
            config_path.write_text(
                json.dumps(config.to_dict()),
                encoding="utf-8",
            )
            task_path.write_text(
                json.dumps(
                    {
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
                ),
                encoding="utf-8",
            )
            stream_path.write_text(
                json.dumps(settings.to_dict()),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(
                    [
                        "--configuration",
                        str(config_path),
                        "--task-set",
                        str(task_path),
                        "--stream-settings",
                        str(stream_path),
                        "--repetitions",
                        "1",
                        "--concurrency",
                        "1",
                        "--modes",
                        "execute",
                        "--output",
                        str(output),
                        "--vllm-base-url",
                        "http://127.0.0.1:9",
                        "--results-root",
                        str(results_root),
                    ]
                )
            self.assertEqual(code, 1)
            self.assertIn("public result", stderr.getvalue())
            self.assertFalse(output.exists())
            rendered = stdout.getvalue()
            for task_id in task_set.task_ids:
                self.assertNotIn(task_id, rendered)
                self.assertNotIn(task_id, stderr.getvalue())
