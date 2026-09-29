import copy
import json
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from llm_behavior_ci.config import (
    RunConfiguration,
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
) -> AgentTurn:
    return AgentTurn(
        prompt_text=prompt_text,
        output_text=output_text,
        top_k_logprobs=_LOGPROBS,
        latency_seconds=0.1,
        started_at=started_at,
        action=action,
        app_name=app_name,
        api_name=api_name,
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

    def test_step_limit_stops_execute_mode(self) -> None:
        config = replace(_config(), agent=replace(_config().agent, step_limit=1))
        clock = _clock()
        session = FakeSession()
        result = run_episode(
            "task-1",
            config,
            "execute",
            run=new_run_identity(config),
            runtime=_runtime(session, AlwaysActionAgent(clock), clock),
        )
        self.assertEqual(len(result.model_steps), 1)
        self.assertEqual(result.termination_reason, "step_limit")
        self.assertEqual(result.status, "failed")
        self.assertEqual(session.evaluate_count, 0)
        self.assertEqual(session.close_count, 1)

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


if __name__ == "__main__":
    unittest.main()
