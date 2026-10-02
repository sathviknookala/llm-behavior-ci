import copy
import importlib.util
import json
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from llm_behavior_ci.config import (
    RunConfiguration,
    WorkflowSettings,
    new_run_identity,
)
from llm_behavior_ci.records import ModelStep, TokenLogprob, ToolStep
from llm_behavior_ci.runtime.actions import ActionRejected
from llm_behavior_ci.runtime.agent import AgentTurn
from llm_behavior_ci.runtime.appworld import (
    EvaluationResult,
    TaskContext,
    ToolResult,
)
from llm_behavior_ci.runtime.agent import SmolagentsVLLMAgent, action_execution_backend
from llm_behavior_ci.runtime.episode import (
    EpisodeRejected,
    RuntimeDependencies,
    RuntimeUnavailable,
    build_runtime,
    is_live_runtime,
    run_episode,
)
from llm_behavior_ci.runtime.workflow import WorkflowControlledAgent

_START = datetime(2026, 9, 27, 15, 0, tzinfo=timezone.utc)
_LOGPROBS = (
    (TokenLogprob(token_id=7, logprob=-0.5, rank=0),),
)
_PROMPT = "plan the next action"
_PLAN = "1. open the calendar"
_ACTION = "calendar.lookup()"


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


def _config() -> RunConfiguration:
    return RunConfiguration.from_dict(_payload())


def _clock() -> callable:
    current = _START

    def tick() -> datetime:
        nonlocal current
        value = current
        current = current + timedelta(minutes=1)
        return value

    return tick


def _turn(
    output_text: str,
    *,
    action: str | None,
    app_name: str | None = None,
    api_name: str | None = None,
    prompt_text: str = _PROMPT,
    started_at: datetime = _START,
    rejection: str | None = None,
    feedback: str | None = None,
) -> AgentTurn:
    return AgentTurn(
        prompt_text=prompt_text,
        output_text=output_text,
        top_k_logprobs=_LOGPROBS,
        generated_token_count=len(_LOGPROBS),
        latency_seconds=0.1,
        started_at=started_at,
        action=action,
        app_name=app_name,
        api_name=api_name,
        rejection=rejection,
        feedback=feedback,
    )


class FakeSession:
    def __init__(self, task_id: str = "task-1") -> None:
        self.task_id = task_id
        self.execute_count = 0
        self.evaluate_count = 0
        self.close_count = 0
        self.context_count = 0
        self.actions: list[str] = []
        self.execute_error: BaseException | None = None
        self.tool_results: list[ToolResult] = []
        self.evaluation = EvaluationResult(
            success=True,
            passed_requirements=1,
            total_requirements=1,
            difficulty=1,
        )

    def context(self) -> TaskContext:
        self.context_count += 1
        return TaskContext(
            task_id=self.task_id,
            instruction="solve the task",
            api_documentation="calendar docs",
        )

    def execute(self, action: str) -> ToolResult:
        self.execute_count += 1
        self.actions.append(action)
        if self.execute_error is not None:
            raise self.execute_error
        if self.tool_results:
            return self.tool_results.pop(0)
        return ToolResult(
            output_text="ok",
            error_message=None,
            recoverable=False,
            app_name=None,
            api_name=None,
        )

    def evaluate(self) -> EvaluationResult:
        self.evaluate_count += 1
        return self.evaluation

    def close(self) -> None:
        self.close_count += 1


class FakeAgent:
    def __init__(self, turns: list[AgentTurn] | None = None, *, clock=None) -> None:
        self._turns = list(turns or [])
        self._clock = clock
        self._index = 0
        self.config: RunConfiguration | None = None
        self.context: TaskContext | None = None
        self.tool_outputs: list[str | None] = []

    def begin(self, context: TaskContext, config: RunConfiguration) -> None:
        self.context = context
        self.config = config

    def next_turn(self, *, tool_output: str | None) -> AgentTurn:
        self.tool_outputs.append(tool_output)
        if self._index >= len(self._turns):
            source = _turn("STOP", action=None)
        else:
            source = self._turns[self._index]
            self._index += 1
        return _turn(
            source.output_text,
            action=source.action,
            app_name=source.app_name,
            api_name=source.api_name,
            prompt_text=source.prompt_text,
            started_at=self._clock(),
            rejection=source.rejection,
            feedback=source.feedback,
        )


class AlwaysActionAgent:
    def __init__(self, clock) -> None:
        self._clock = clock
        self.config: RunConfiguration | None = None

    def begin(self, context: TaskContext, config: RunConfiguration) -> None:
        del context
        self.config = config

    def next_turn(self, *, tool_output: str | None) -> AgentTurn:
        del tool_output
        return _turn(_ACTION, action=_ACTION, started_at=self._clock())


def _runtime(
    session: FakeSession,
    agent,
    clock,
) -> RuntimeDependencies:
    return RuntimeDependencies(
        session_factory=lambda task_id: session,
        agent=agent,
        clock=clock,
    )


class EpisodeRunnerTests(unittest.TestCase):
    def test_plan_mode_does_not_execute_or_evaluate(self) -> None:
        config = _config()
        clock = _clock()
        session = FakeSession()
        agent = FakeAgent([_turn(_PLAN, action=None)], clock=clock)
        result = run_episode(
            "task-1",
            config,
            "plan",
            run=new_run_identity(config),
            runtime=_runtime(session, agent, clock),
        )
        self.assertEqual(session.execute_count, 0)
        self.assertEqual(session.evaluate_count, 0)
        self.assertEqual(session.close_count, 1)
        self.assertEqual(result.mode, "plan")
        self.assertEqual(result.termination_reason, "plan_emitted")
        self.assertEqual(result.tool_steps, ())
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.plan_text, _PLAN)
        self.assertIsNone(result.evaluator_outcome)

    def test_execute_mode_evaluates_and_closes(self) -> None:
        config = _config()
        clock = _clock()
        session = FakeSession()
        agent = FakeAgent(
            [
                _turn(_ACTION, action=_ACTION),
                _turn("STOP", action=None),
            ],
            clock=clock,
        )
        result = run_episode(
            "task-1",
            config,
            "execute",
            run=new_run_identity(config),
            runtime=_runtime(session, agent, clock),
        )
        self.assertEqual(session.execute_count, 1)
        self.assertEqual(session.evaluate_count, 1)
        self.assertEqual(session.close_count, 1)
        self.assertIsNotNone(result.evaluator_outcome)
        self.assertEqual(
            result.evaluator_outcome.success,
            session.evaluation.passed_requirements
            == session.evaluation.total_requirements,
        )
        self.assertEqual(result.termination_reason, "agent_stopped")
        self.assertEqual(result.status, "completed")

    def test_configuration_is_passed_to_the_agent(self) -> None:
        config = _config()
        clock = _clock()
        session = FakeSession()
        agent = FakeAgent([_turn(_PLAN, action=None)], clock=clock)
        run_episode(
            "task-1",
            config,
            "plan",
            run=new_run_identity(config),
            runtime=_runtime(session, agent, clock),
        )
        self.assertIs(agent.config, config)
        self.assertEqual(agent.config.agent.sampling.seed, 17)
        self.assertEqual(agent.config.agent.prompt.prompt_version, "prompt-v1")

    def test_mismatched_run_identity_is_rejected(self) -> None:
        config_a = _config()
        other = copy.deepcopy(_payload())
        other["run_seed"] = 8
        config_b = RunConfiguration.from_dict(other)
        called: list[str] = []

        def factory(task_id: str) -> FakeSession:
            called.append(task_id)
            return FakeSession(task_id)

        clock = _clock()
        with self.assertRaises(EpisodeRejected):
            run_episode(
                "task-1",
                config_b,
                "plan",
                run=new_run_identity(config_a),
                runtime=RuntimeDependencies(
                    session_factory=factory,
                    agent=FakeAgent(clock=clock),
                    clock=clock,
                ),
            )
        self.assertEqual(called, [])

    def test_close_runs_when_execute_raises(self) -> None:
        config = _config()
        clock = _clock()
        session = FakeSession()
        session.execute_error = RuntimeError("execute failed")
        agent = FakeAgent(
            [_turn(_ACTION, action=_ACTION)],
            clock=clock,
        )
        result = run_episode(
            "task-1",
            config,
            "execute",
            run=new_run_identity(config),
            runtime=_runtime(session, agent, clock),
        )
        self.assertEqual(session.close_count, 1)
        self.assertEqual(result.termination_reason, "runtime_error")
        self.assertEqual(result.status, "failed")
        self.assertEqual(len(result.model_steps), 1)
        self.assertEqual(result.episode_errors[0].source, "runtime")
        self.assertEqual(result.episode_errors[0].message, "execute failed")
        self.assertEqual(session.evaluate_count, 0)
        self.assertIsNone(result.evaluator_outcome)

    def test_step_limit_evaluates_without_completing(self) -> None:
        config = replace(_config(), agent=replace(_config().agent, step_limit=1))
        clock = _clock()
        session = FakeSession()
        session.evaluation = EvaluationResult(
            success=True,
            passed_requirements=1,
            total_requirements=1,
            difficulty=1,
        )
        result = run_episode(
            "task-1",
            config,
            "execute",
            run=new_run_identity(config),
            runtime=_runtime(session, AlwaysActionAgent(clock), clock),
        )
        self.assertEqual(len(result.model_steps), 1)
        self.assertEqual(session.evaluate_count, 1)
        self.assertEqual(session.close_count, 1)
        self.assertEqual(result.termination_reason, "step_limit")
        self.assertEqual(result.status, "failed")
        self.assertIsNotNone(result.evaluator_outcome)
        assert result.evaluator_outcome is not None
        self.assertTrue(result.evaluator_outcome.success)
        self.assertEqual(result.evaluator_outcome.passed_requirements, 1)
        self.assertEqual(result.evaluator_outcome.total_requirements, 1)
        self.assertEqual(result.evaluator_outcome.difficulty, 1)
        self.assertEqual(result.episode_errors[0].source, "step_limit")
        self.assertEqual(result.episode_errors[0].message, "step limit reached")
        self.assertFalse(result.episode_errors[0].recoverable)

    def test_model_and_tool_records_are_present(self) -> None:
        config = _config()
        clock = _clock()
        session = FakeSession()
        agent = FakeAgent(
            [
                _turn(_ACTION, action=_ACTION, prompt_text=_PROMPT),
                _turn("STOP", action=None),
            ],
            clock=clock,
        )
        result = run_episode(
            "task-1",
            config,
            "execute",
            run=new_run_identity(config),
            runtime=_runtime(session, agent, clock),
        )
        self.assertIsInstance(result.model_steps[0], ModelStep)
        self.assertIsInstance(result.tool_steps[0], ToolStep)
        self.assertEqual(result.model_steps[0].index, 0)
        self.assertEqual(result.tool_steps[0].index, 1)
        self.assertEqual(result.model_steps[0].prompt_text, _PROMPT)
        self.assertEqual(result.tool_steps[0].action, _ACTION)

    def test_callback_failure_propagates_and_closes(self) -> None:
        config = _config()
        clock = _clock()
        session = FakeSession()
        agent = FakeAgent([_turn(_PLAN, action=None)], clock=clock)

        def on_step(step: ModelStep | ToolStep) -> None:
            del step
            raise RuntimeError("callback failed")

        with self.assertRaisesRegex(RuntimeError, "callback failed"):
            run_episode(
                "task-1",
                config,
                "plan",
                run=new_run_identity(config),
                runtime=_runtime(session, agent, clock),
                on_step=on_step,
            )
        self.assertEqual(session.close_count, 1)

    def test_recoverable_tool_error_can_still_succeed(self) -> None:
        config = _config()
        clock = _clock()
        session = FakeSession()
        session.tool_results = [
            ToolResult(
                output_text=None,
                error_message="missing",
                recoverable=True,
                app_name=None,
                api_name=None,
            )
        ]
        session.evaluation = EvaluationResult(
            success=True,
            passed_requirements=1,
            total_requirements=1,
            difficulty=1,
        )
        agent = FakeAgent(
            [
                _turn(_ACTION, action=_ACTION),
                _turn("STOP", action=None),
            ],
            clock=clock,
        )
        result = run_episode(
            "task-1",
            config,
            "execute",
            run=new_run_identity(config),
            runtime=_runtime(session, agent, clock),
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.termination_reason, "agent_stopped")
        self.assertEqual(len(result.tool_steps), 1)
        self.assertIsNotNone(result.tool_steps[0].error)
        self.assertTrue(result.tool_steps[0].error.recoverable)
        self.assertEqual(result.tool_steps[0].error.message, "missing")
        self.assertEqual(agent.tool_outputs[1], "missing")
        self.assertTrue(result.evaluator_outcome.success)

    def test_execution_failed_traceback_feeds_back_the_last_line(self) -> None:
        config = _config()
        clock = _clock()
        session = FakeSession()
        traceback = (
            "Execution failed. Traceback:\n"
            "  File \"<python-input>\", line 1, in <module>\n"
            "    apis.calendar.show()\n"
            "{'message': 'missing token'}\n"
        )
        session.tool_results = [
            ToolResult(
                output_text=None,
                error_message=traceback,
                recoverable=True,
                app_name="calendar",
                api_name="show",
            )
        ]
        agent = FakeAgent(
            [
                _turn("apis.calendar.show()", action="apis.calendar.show()", app_name="calendar", api_name="show"),
                _turn("STOP", action=None),
            ],
            clock=clock,
        )
        result = run_episode(
            "task-1",
            config,
            "execute",
            run=new_run_identity(config),
            runtime=_runtime(session, agent, clock),
        )
        self.assertEqual(result.tool_steps[0].error.message, traceback)
        self.assertEqual(agent.tool_outputs[1], "{'message': 'missing token'}")

    def test_execute_mode_sends_actions_through_session_execute(self) -> None:
        config = _config()
        clock = _clock()
        session = FakeSession()
        agent = FakeAgent(
            [
                _turn(_ACTION, action=_ACTION),
                _turn("STOP", action=None),
            ],
            clock=clock,
        )
        run_episode(
            "task-1",
            config,
            "execute",
            run=new_run_identity(config),
            runtime=_runtime(session, agent, clock),
        )
        self.assertEqual(session.actions, [_ACTION])
        self.assertEqual(action_execution_backend(), "appworld_session.execute")

    def test_tool_calling_action_interface_still_executes_through_appworld(
        self,
    ) -> None:
        config = replace(
            _config(),
            agent=replace(_config().agent, action_interface="tool_calling"),
        )
        clock = _clock()
        session = FakeSession()
        agent = FakeAgent(
            [
                _turn(_ACTION, action=_ACTION),
                _turn("STOP", action=None),
            ],
            clock=clock,
        )
        run_episode(
            "task-1",
            config,
            "execute",
            run=new_run_identity(config),
            runtime=_runtime(session, agent, clock),
        )
        self.assertEqual(session.actions, [_ACTION])
        self.assertEqual(action_execution_backend(), "appworld_session.execute")

    def test_build_runtime_uses_endpoint_and_rejects_unknown_prompt(self) -> None:
        config = _config()
        runtime = build_runtime(config, "http://127.0.0.1:9", mode="plan")
        self.assertIsInstance(runtime.agent, SmolagentsVLLMAgent)
        self.assertEqual(runtime.agent.base_url, "http://127.0.0.1:9")
        bad = replace(
            config,
            agent=replace(
                config.agent,
                prompt=replace(config.agent.prompt, prompt_version="prompt-missing"),
            ),
        )
        with self.assertRaisesRegex(RuntimeUnavailable, "unknown prompt_version"):
            build_runtime(bad, "http://127.0.0.1:9")

    def test_is_live_runtime_distinguishes_real_agents_from_fakes(self) -> None:
        config = _config()
        live = build_runtime(config, "http://127.0.0.1:9", mode="plan")
        self.assertTrue(is_live_runtime(live))
        session = FakeSession()
        agent = FakeAgent([], clock=lambda: _START)
        fake = _runtime(session, agent, lambda: _START)
        self.assertFalse(is_live_runtime(fake))
        real_agent_fake_session = RuntimeDependencies(
            session_factory=lambda task_id: session,
            agent=SmolagentsVLLMAgent("http://127.0.0.1:9"),
            clock=lambda: _START,
        )
        self.assertFalse(is_live_runtime(real_agent_fake_session))

    def test_plan_mode_keeps_embedded_api_call_unparsed(self) -> None:
        plan_with_call = (
            "1. inspect the calendar\n"
            'apis.calendar.show_calendar(date="2026-01-01")\n'
            "2. finish"
        )
        agent = SmolagentsVLLMAgent("http://127.0.0.1:9")
        agent.set_mode("plan")
        agent.begin(
            TaskContext(
                task_id="task-1",
                instruction="solve the task",
                api_documentation="calendar docs",
            ),
            _config(),
        )
        body = {
            "choices": [
                {
                    "message": {"content": plan_with_call},
                    "token_ids": [7],
                    "logprobs": {
                        "content": [
                            {
                                "token": "token_id:7",
                                "bytes": [55],
                                "logprob": -0.5,
                                "top_logprobs": [],
                            }
                        ]
                    },
                }
            ]
        }

        class _FakeResponse:
            def read(self) -> bytes:
                return json.dumps(body).encode("utf-8")

            def __enter__(self) -> "_FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

        with patch(
            "urllib.request.urlopen",
            side_effect=lambda request, timeout=None: _FakeResponse(),
        ):
            turn = agent.next_turn(tool_output=None)
        self.assertIsNone(turn.action)
        self.assertIsNone(turn.app_name)
        self.assertIsNone(turn.api_name)
        self.assertEqual(turn.output_text, plan_with_call)

        clock = _clock()
        session = FakeSession()
        fake_agent = FakeAgent(
            [_turn(plan_with_call, action=None)],
            clock=clock,
        )
        result = run_episode(
            "task-1",
            _config(),
            "plan",
            run=new_run_identity(_config()),
            runtime=_runtime(session, fake_agent, clock),
        )
        self.assertEqual(session.execute_count, 0)
        self.assertEqual(session.evaluate_count, 0)
        self.assertEqual(result.termination_reason, "plan_emitted")
        self.assertEqual(result.plan_text, plan_with_call)

    def test_complete_task_executes_then_evaluates(self) -> None:
        complete = "apis.supervisor.complete_task()"
        config = _config()
        clock = _clock()
        session = FakeSession()
        agent = FakeAgent(
            [
                _turn(
                    complete,
                    action=complete,
                    app_name="supervisor",
                    api_name="complete_task",
                )
            ],
            clock=clock,
        )
        result = run_episode(
            "task-1",
            config,
            "execute",
            run=new_run_identity(config),
            runtime=_runtime(session, agent, clock),
        )
        self.assertEqual(session.execute_count, 1)
        self.assertEqual(session.actions, [complete])
        self.assertEqual(session.evaluate_count, 1)
        self.assertEqual(result.termination_reason, "appworld_completed")
        self.assertEqual(result.status, "completed")
        self.assertIsNotNone(result.evaluator_outcome)
        self.assertEqual(len(result.tool_steps), 1)

    def test_invalid_action_fails_without_evaluate(self) -> None:
        class RejectingAgent:
            def __init__(self, clock) -> None:
                self._clock = clock

            def begin(self, context: TaskContext, config: RunConfiguration) -> None:
                del context, config

            def next_turn(self, *, tool_output: str | None) -> AgentTurn:
                del tool_output
                raise ActionRejected("action is not valid Python")

        config = _config()
        clock = _clock()
        session = FakeSession()
        result = run_episode(
            "task-1",
            config,
            "execute",
            run=new_run_identity(config),
            runtime=_runtime(session, RejectingAgent(clock), clock),
        )
        self.assertEqual(result.termination_reason, "invalid_action")
        self.assertEqual(result.status, "failed")
        self.assertEqual(session.execute_count, 0)
        self.assertEqual(session.evaluate_count, 0)
        self.assertEqual(result.episode_errors[0].source, "runtime")
        self.assertIsNone(result.evaluator_outcome)

    def test_rejected_output_is_not_executed_and_the_episode_continues(self) -> None:
        config = _config()
        clock = _clock()
        session = FakeSession()
        agent = FakeAgent(
            [
                _turn("not a call", action=None, rejection="action is not valid Python"),
                _turn("STOP", action=None),
            ],
            clock=clock,
        )
        result = run_episode(
            "task-1",
            config,
            "execute",
            run=new_run_identity(config),
            runtime=_runtime(session, agent, clock),
        )
        self.assertEqual(session.execute_count, 0)
        self.assertEqual(session.evaluate_count, 1)
        self.assertEqual(result.termination_reason, "agent_stopped")
        self.assertEqual(len(result.model_steps), 2)
        self.assertEqual(agent.tool_outputs[1], (
            "That output was not one apis.<app>.<api>(...) call. "
            "Emit exactly one call, with keyword arguments, and no other text."
        ))


def _workflow_settings() -> WorkflowSettings:
    return WorkflowSettings.from_dict(
        {
            "policy": "plan_progress_v1",
            "repeat_action_limit": 2,
            "no_progress_turns": 3,
            "completion_gate": True,
            "max_plan_steps": 5,
        }
    )


def _with_workflow(config: RunConfiguration, **agent_overrides: object) -> RunConfiguration:
    return replace(
        config,
        agent=replace(
            config.agent,
            workflow=_workflow_settings(),
            **agent_overrides,
        ),
    )


class ObservingAgent(FakeAgent):
    def __init__(self, turns: list[AgentTurn] | None = None, *, clock=None) -> None:
        super().__init__(turns, clock=clock)
        self.observed: list[tuple[str, object]] = []

    def observe_tool_result(self, action: str, result: object) -> None:
        self.observed.append((action, result))


class _ScriptedBase:
    def __init__(self, outputs: list[str], clock) -> None:
        self._outputs = list(outputs)
        self._clock = clock
        self.generations = 0
        self.tool_outputs: list[str | None] = []

    def begin(self, context: TaskContext, config: RunConfiguration) -> None:
        del context, config

    def generate_turn(
        self,
        *,
        tool_output: str | None,
        extra_instruction: str | None = None,
        parse_action: bool = True,
    ) -> AgentTurn:
        del extra_instruction, parse_action
        self.generations += 1
        self.tool_outputs.append(tool_output)
        return AgentTurn(
            prompt_text=_PROMPT,
            output_text=self._outputs.pop(0),
            top_k_logprobs=_LOGPROBS,
            generated_token_count=len(_LOGPROBS),
            latency_seconds=0.1,
            started_at=self._clock(),
            action=None,
            app_name=None,
            api_name=None,
        )


def _envelope(
    action: str,
    *,
    plan: list[str] | None = None,
    completed: list[int] | None = None,
    unfinished: list[int] | None = None,
    active: int | None = 1,
    ready: bool = False,
) -> str:
    payload: dict[str, object] = {
        "completed_steps": list(completed or []),
        "active_step": active,
        "ready_to_complete": ready,
        "unfinished_steps": list(unfinished or []),
        "action": action,
    }
    if plan is not None:
        payload = {"plan": plan, **payload}
    return json.dumps(payload)


class WorkflowEpisodeTests(unittest.TestCase):
    def test_rejection_feedback_replaces_the_generic_parser_message(self) -> None:
        config = _config()
        clock = _clock()
        session = FakeSession()
        agent = FakeAgent(
            [
                _turn(
                    "blocked",
                    action=None,
                    rejection="workflow_completion_blocked",
                    feedback="controller feedback",
                ),
                _turn("STOP", action=None),
            ],
            clock=clock,
        )
        result = run_episode(
            "task-1",
            config,
            "execute",
            run=new_run_identity(config),
            runtime=_runtime(session, agent, clock),
        )
        self.assertEqual(len(result.model_steps), 2)
        self.assertEqual(result.tool_steps, ())
        self.assertEqual(session.execute_count, 0)
        self.assertEqual(agent.tool_outputs[1], "controller feedback")
        self.assertNotIn("apis.<app>", agent.tool_outputs[1])

    def test_observer_sees_success_and_recoverable_errors_only(self) -> None:
        config = _config()
        clock = _clock()
        session = FakeSession()
        agent = ObservingAgent(
            [
                _turn(_ACTION, action=_ACTION),
                _turn("STOP", action=None),
            ],
            clock=clock,
        )
        run_episode(
            "task-1",
            config,
            "execute",
            run=new_run_identity(config),
            runtime=_runtime(session, agent, clock),
        )
        self.assertEqual(len(agent.observed), 1)
        self.assertIsInstance(agent.observed[0][1], ToolResult)
        self.assertIsNone(agent.observed[0][1].error_message)
        self.assertEqual(session.evaluate_count, 1)
        self.assertNotIsInstance(agent.observed[0][1], EvaluationResult)

        clock = _clock()
        session = FakeSession()
        session.tool_results = [
            ToolResult(
                output_text=None,
                error_message="missing",
                recoverable=True,
                app_name=None,
                api_name=None,
            )
        ]
        agent = ObservingAgent(
            [
                _turn(_ACTION, action=_ACTION),
                _turn("STOP", action=None),
            ],
            clock=clock,
        )
        run_episode(
            "task-1",
            config,
            "execute",
            run=new_run_identity(config),
            runtime=_runtime(session, agent, clock),
        )
        self.assertEqual(len(agent.observed), 1)
        observed = agent.observed[0][1]
        self.assertIsInstance(observed, ToolResult)
        self.assertEqual(observed.error_message, "missing")
        self.assertNotIsInstance(observed, EvaluationResult)

        clock = _clock()
        session = FakeSession()
        agent = ObservingAgent([_turn("STOP", action=None)], clock=clock)
        run_episode(
            "task-1",
            config,
            "execute",
            run=new_run_identity(config),
            runtime=_runtime(session, agent, clock),
        )
        self.assertEqual(agent.observed, [])
        self.assertEqual(session.evaluate_count, 1)

    def test_execute_workflow_wraps_the_live_agent_and_plan_mode_does_not(self) -> None:
        config = _with_workflow(_config())
        execute_runtime = build_runtime(config, "http://127.0.0.1:9", mode="execute")
        self.assertIsInstance(execute_runtime.agent, WorkflowControlledAgent)
        self.assertTrue(is_live_runtime(execute_runtime))
        self.assertEqual(
            execute_runtime.agent.underlying_agents(),
            (execute_runtime.agent._base_agent,),
        )
        plan_runtime = build_runtime(config, "http://127.0.0.1:9", mode="plan")
        self.assertIsInstance(plan_runtime.agent, SmolagentsVLLMAgent)
        self.assertNotIsInstance(plan_runtime.agent, WorkflowControlledAgent)

    def test_premature_completion_is_blocked_and_the_next_action_runs(self) -> None:
        read = "apis.calendar.show_calendar()"
        advance = 'apis.calendar.create_event(title="meetup")'
        done = "apis.supervisor.complete_task()"
        plan = ["Read the calendar", "Create the event"]
        clock = _clock()
        base = _ScriptedBase(
            [
                _envelope(
                    read,
                    plan=plan,
                    completed=[],
                    unfinished=[1, 2],
                    active=1,
                    ready=False,
                ),
                _envelope(
                    done,
                    completed=[1],
                    unfinished=[2],
                    active=2,
                    ready=False,
                ),
                _envelope(
                    advance,
                    completed=[1],
                    unfinished=[2],
                    active=2,
                    ready=False,
                ),
                _envelope(
                    done,
                    completed=[1, 2],
                    unfinished=[],
                    active=None,
                    ready=True,
                ),
            ],
            clock,
        )
        config = _with_workflow(_config())
        controller = WorkflowControlledAgent(base, config.agent.workflow)
        session = FakeSession()
        result = run_episode(
            "task-1",
            config,
            "execute",
            run=new_run_identity(config),
            runtime=_runtime(session, controller, clock),
        )
        self.assertEqual(
            [step.action for step in result.tool_steps],
            [read, advance, done],
        )
        self.assertEqual(
            base.tool_outputs[2],
            (
                "Completion was blocked because your workflow ledger "
                "still has unfinished steps. Continue with one action "
                "that advances an unfinished step. Do not call "
                "complete_task until every declared plan step is complete."
            ),
        )
        self.assertEqual(result.termination_reason, "appworld_completed")
        self.assertEqual(len(result.model_steps), 4)
        self.assertEqual(base.generations, 4)

    def test_third_exact_repeat_is_not_executed(self) -> None:
        action = "apis.calendar.show_calendar()"
        clock = _clock()
        later = _envelope(
            action,
            completed=[],
            unfinished=[1, 2],
            active=1,
            ready=False,
        )
        base = _ScriptedBase(
            [
                _envelope(
                    action,
                    plan=["Read the calendar", "Create the event"],
                    completed=[],
                    unfinished=[1, 2],
                    active=1,
                    ready=False,
                ),
                later,
                later,
                later,
            ],
            clock,
        )
        config = _with_workflow(_config(), execute_max_model_turns=4)
        controller = WorkflowControlledAgent(base, config.agent.workflow)
        session = FakeSession()
        result = run_episode(
            "task-1",
            config,
            "execute",
            run=new_run_identity(config),
            runtime=_runtime(session, controller, clock),
        )
        self.assertEqual([step.action for step in result.tool_steps], [action, action])
        self.assertEqual(
            base.tool_outputs[3],
            (
                "The exact same action has already repeated without "
                "sufficient progress. Choose a different documented "
                "action that advances an unfinished plan step."
            ),
        )
        self.assertEqual(result.termination_reason, "step_limit")
        self.assertEqual(len(result.model_steps), 4)
        self.assertEqual(base.generations, 4)

    def test_completed_ledger_runs_complete_task(self) -> None:
        read = "apis.calendar.show_calendar()"
        advance = 'apis.calendar.create_event(title="meetup")'
        done = "apis.supervisor.complete_task()"
        clock = _clock()
        base = _ScriptedBase(
            [
                _envelope(
                    read,
                    plan=["Read the calendar", "Create the event"],
                    completed=[],
                    unfinished=[1, 2],
                    active=1,
                    ready=False,
                ),
                _envelope(
                    advance,
                    completed=[1],
                    unfinished=[2],
                    active=2,
                    ready=False,
                ),
                _envelope(
                    done,
                    completed=[1, 2],
                    unfinished=[],
                    active=None,
                    ready=True,
                ),
            ],
            clock,
        )
        config = _with_workflow(_config())
        controller = WorkflowControlledAgent(base, config.agent.workflow)
        seen: list[object] = []
        original = controller.observe_tool_result

        def _spy(action: str, result: ToolResult) -> None:
            seen.append(result)
            original(action, result)

        controller.observe_tool_result = _spy
        session = FakeSession()
        result = run_episode(
            "task-1",
            config,
            "execute",
            run=new_run_identity(config),
            runtime=_runtime(session, controller, clock),
        )
        self.assertEqual(result.termination_reason, "appworld_completed")
        self.assertEqual(
            [step.action for step in result.tool_steps],
            [read, advance, done],
        )
        self.assertEqual(session.evaluate_count, 1)
        self.assertEqual(len(seen), 3)
        self.assertTrue(all(isinstance(item, ToolResult) for item in seen))
        self.assertTrue(all(not isinstance(item, EvaluationResult) for item in seen))

    def test_parser_accounting_follows_the_workflow_policy(self) -> None:
        path = (
            Path(__file__).resolve().parents[2]
            / "scripts/evaluation/run_capability_pilot.py"
        )
        spec = importlib.util.spec_from_file_location(
            "run_capability_pilot_workflow_test",
            path,
        )
        self.assertIsNotNone(spec)
        assert spec is not None and spec.loader is not None
        pilot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pilot)

        plain = _config()
        clock = _clock()
        stopped = run_episode(
            "task-1",
            plain,
            "execute",
            run=new_run_identity(plain),
            runtime=_runtime(
                FakeSession(),
                FakeAgent([_turn("STOP", action=None)], clock=clock),
                clock,
            ),
        )
        self.assertEqual(pilot._parser_errors(plain, stopped), 0)
        clock = _clock()
        prose = run_episode(
            "task-1",
            plain,
            "execute",
            run=new_run_identity(plain),
            runtime=_runtime(
                FakeSession(),
                FakeAgent(
                    [
                        _turn(
                            "not a call",
                            action=None,
                            rejection="action is not valid Python",
                        ),
                        _turn("STOP", action=None),
                    ],
                    clock=clock,
                ),
                clock,
            ),
        )
        self.assertEqual(pilot._parser_errors(plain, prose), 1)

        workflow = _with_workflow(_config(), execute_max_model_turns=1)
        clock = _clock()
        loose = json.dumps({"action": "apis.calendar.show_calendar()"})
        base = _ScriptedBase([loose], clock)
        controller = WorkflowControlledAgent(base, workflow.agent.workflow)
        loose_episode = run_episode(
            "task-1",
            workflow,
            "execute",
            run=new_run_identity(workflow),
            runtime=_runtime(FakeSession(), controller, clock),
        )
        self.assertEqual(loose_episode.termination_reason, "step_limit")
        self.assertEqual(pilot._parser_errors(workflow, loose_episode), 0)
        self.assertEqual(pilot._parser_errors(plain, loose_episode), 1)


if __name__ == "__main__":
    unittest.main()
