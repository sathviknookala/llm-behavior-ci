import hashlib
import io
import json
import os
import subprocess
import sys
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from llm_behavior_ci.config import RunConfiguration, StreamSettings, new_run_identity
from llm_behavior_ci.records import RecordError, assert_public_payload
from llm_behavior_ci.records import TokenLogprob
from llm_behavior_ci.runtime.agent import AgentTurn
from llm_behavior_ci.runtime.appworld import EvaluationResult, TaskContext, ToolResult
from llm_behavior_ci.runtime.episode import (
    EpisodeRejected,
    RuntimeDependencies,
    RuntimeUnavailable,
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
    def __init__(self, clock: Clock, *, mode: str, diverge: bool) -> None:
        self._clock = clock
        self._mode = mode
        self._diverge = diverge
        self._begins = 0
        self._step = 0
        self._reference = True
        self.begins = 0
        self.tool_outputs: list[list[str | None]] = []

    def begin(self, context: TaskContext, config: RunConfiguration) -> None:
        del context, config
        self.begins += 1
        self._begins += 1
        self._reference = self._begins % 2 == 1
        self._step = 0
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
