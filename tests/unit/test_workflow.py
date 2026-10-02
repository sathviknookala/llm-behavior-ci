import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from llm_behavior_ci.config import RunConfiguration, WorkflowSettings
from llm_behavior_ci.runtime.agent import AgentTurn, SmolagentsVLLMAgent
from llm_behavior_ci.runtime.appworld import TaskContext, ToolResult
from llm_behavior_ci.runtime.workflow import (
    WorkflowControlledAgent,
    WorkflowEnvelopeError,
    canonical_action,
    parse_workflow_envelope,
    workflow_instruction,
    workflow_output_has_parseable_action,
)
from llm_behavior_ci.runtime import workflow as workflow_module

_START = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)
_PLAN = ["Read the calendar", "Create the event"]
_SHOW = "apis.calendar.show_calendar()"
_SHOW_WIDE = 'apis.calendar.create_event( title = "a" )'
_CREATE = 'apis.calendar.create_event(title="a")'
_CREATE_OTHER = 'apis.calendar.create_event(title="b")'
_DONE = "apis.supervisor.complete_task()"
_CONTEXT = TaskContext(
    task_id="task-1",
    instruction="update the calendar",
    api_documentation="calendar docs",
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
    }


def _settings(**overrides: object) -> WorkflowSettings:
    payload: dict[str, object] = {
        "policy": "plan_progress_v1",
        "repeat_action_limit": 2,
        "no_progress_turns": 3,
        "completion_gate": True,
        "max_plan_steps": 5,
    }
    payload.update(overrides)
    return WorkflowSettings.from_dict(payload)


def _run(settings: WorkflowSettings) -> RunConfiguration:
    config = RunConfiguration.from_dict(_payload())
    agent = config.agent
    return RunConfiguration(
        model=config.model,
        agent=type(agent)(
            smolagents_version=agent.smolagents_version,
            action_interface=agent.action_interface,
            prompt=agent.prompt,
            step_limit=agent.step_limit,
            sampling=agent.sampling,
            workflow=settings,
        ),
        task=config.task,
        run_seed=config.run_seed,
        git_commit=config.git_commit,
        protocol_hash=config.protocol_hash,
    )


def _first(action: str, plan: list[str] | None = None, active: int = 1) -> str:
    steps = list(plan or _PLAN)
    return json.dumps(
        {
            "plan": steps,
            "completed_steps": [],
            "active_step": active,
            "ready_to_complete": False,
            "unfinished_steps": list(range(1, len(steps) + 1)),
            "action": action,
        }
    )


def _later(
    action: str,
    completed: list[int] | None = None,
    *,
    step_count: int = 2,
    active: int | None = None,
    ready: bool | None = None,
    unfinished: list[int] | None = None,
    plan: list[str] | None = None,
) -> str:
    done = list(completed or [])
    remaining = (
        list(unfinished)
        if unfinished is not None
        else [number for number in range(1, step_count + 1) if number not in done]
    )
    is_ready = (len(remaining) == 0) if ready is None else ready
    if active is None and not is_ready and remaining:
        chosen: int | None = remaining[0]
    else:
        chosen = active
    payload: dict[str, object] = {
        "completed_steps": done,
        "active_step": None if is_ready and active is None else chosen,
        "ready_to_complete": is_ready,
        "unfinished_steps": remaining,
        "action": action,
    }
    if plan is not None:
        payload["plan"] = plan
    return json.dumps(payload)


def _ok() -> ToolResult:
    return ToolResult(
        output_text="ok",
        error_message=None,
        recoverable=False,
        app_name=None,
        api_name=None,
    )


def _err() -> ToolResult:
    return ToolResult(
        output_text=None,
        error_message="Execution failed. missing",
        recoverable=True,
        app_name=None,
        api_name=None,
    )


class _Base:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.generations = 0
        self.begin_calls = 0
        self.evaluate_calls = 0
        self.calls: list[dict[str, object]] = []
        self.forced: tuple[object, object] | None = None

    def begin(self, context: TaskContext, config: RunConfiguration) -> None:
        self.begin_calls += 1
        self.context = context
        self.config = config

    def generate_turn(
        self,
        *,
        tool_output: str | None,
        extra_instruction: str | None = None,
        parse_action: bool = True,
    ) -> AgentTurn:
        self.generations += 1
        self.calls.append(
            {
                "tool_output": tool_output,
                "extra_instruction": extra_instruction,
                "parse_action": parse_action,
            }
        )
        return AgentTurn(
            prompt_text=extra_instruction or "prompt",
            output_text=self.outputs.pop(0),
            top_k_logprobs=(),
            generated_token_count=1,
            latency_seconds=0.1,
            started_at=_START,
            action=None,
            app_name=None,
            api_name=None,
        )

    def teacher_force_plan(
        self,
        *,
        messages: list[dict[str, str]],
        plan_text: str,
    ) -> tuple[()]:
        self.forced = (messages, plan_text)
        return ()

    def evaluate(self) -> None:
        self.evaluate_calls += 1
        raise AssertionError("controller queried an evaluator")


def _boot(
    outputs: list[str],
    settings: WorkflowSettings | None = None,
) -> tuple[WorkflowControlledAgent, _Base]:
    chosen = settings or _settings()
    base = _Base(outputs)
    controller = WorkflowControlledAgent(base, chosen)
    controller.begin(_CONTEXT, _run(chosen))
    return controller, base


def _snapshot(controller: WorkflowControlledAgent) -> tuple[object, ...]:
    state = controller._state()
    return (
        state.plan,
        state.completed_steps,
        state.active_step,
        state.no_progress_turns,
        state.executed_tool_actions,
        state.last_action_key,
        state.consecutive_same_action_count,
        state.stall_reason,
    )


class WorkflowControllerTests(unittest.TestCase):
    def test_first_valid_envelope_creates_a_plan(self) -> None:
        controller, base = _boot([_first(_SHOW, active=2)])
        self.assertEqual(base.generations, 0)
        turn = controller.next_turn(tool_output=None)
        state = controller._state()
        self.assertEqual(state.plan, tuple(_PLAN))
        self.assertEqual(state.active_step, 2)
        self.assertEqual(state.completed_steps, frozenset())
        self.assertEqual(turn.action, _SHOW)
        self.assertEqual(turn.app_name, "calendar")
        self.assertEqual(turn.api_name, "show_calendar")
        self.assertIsNone(turn.rejection)
        self.assertEqual(base.generations, 1)
        self.assertIs(base.calls[0]["parse_action"], False)
        instruction = base.calls[0]["extra_instruction"]
        self.assertIsInstance(instruction, str)
        assert isinstance(instruction, str)
        self.assertIn("WORKFLOW CONTROLLER: plan_progress_v1", instruction)
        self.assertIn("2 to 5", instruction)
        self.assertNotIn("Authoritative plan:", instruction)
        self.assertNotIn("evaluator", instruction.lower())
        self.assertNotIn("ground_truth", instruction)
        self.assertNotIn("ground truth", instruction.lower())

    def test_initial_plan_rejections_leave_state_unset(self) -> None:
        long_step = "x" * 121
        cases = {
            "one step": _first(_SHOW, plan=["Only this step"]),
            "six steps": _first(
                _SHOW,
                plan=[f"Step {index}" for index in range(1, 7)],
            ),
            "duplicate steps": _first(_SHOW, plan=["Read the calendar", "Read the calendar"]),
            "multiline step": _first(_SHOW, plan=["Read the calendar", "Create\nthe event"]),
            "too long": _first(_SHOW, plan=["Read the calendar", long_step]),
            "completed claim": _later(
                _SHOW,
                [1],
                active=2,
                ready=False,
                plan=_PLAN,
            ),
            "partial unfinished": json.dumps(
                {
                    "plan": _PLAN,
                    "completed_steps": [],
                    "active_step": 1,
                    "ready_to_complete": False,
                    "unfinished_steps": [1],
                    "action": _SHOW,
                }
            ),
            "ready on first turn": json.dumps(
                {
                    "plan": _PLAN,
                    "completed_steps": [],
                    "active_step": 1,
                    "ready_to_complete": True,
                    "unfinished_steps": [1, 2],
                    "action": _SHOW,
                }
            ),
        }
        for label, text in cases.items():
            with self.subTest(case=label):
                controller, base = _boot([text])
                before = _snapshot(controller)
                turn = controller.next_turn(tool_output=None)
                self.assertEqual(turn.rejection, "workflow_envelope_invalid")
                self.assertIsNone(turn.action)
                self.assertEqual(_snapshot(controller), before)
                self.assertIsNone(controller._state().plan)
                self.assertEqual(base.generations, 1)

    def test_instruction_limit_follows_settings(self) -> None:
        settings = _settings(max_plan_steps=8)
        controller, base = _boot([_first(_SHOW)], settings)
        controller.next_turn(tool_output=None)
        instruction = base.calls[0]["extra_instruction"]
        assert isinstance(instruction, str)
        self.assertIn("2 to 8", instruction)

    def test_ledger_grows_and_rejects_inconsistent_updates(self) -> None:
        controller, _base = _boot(
            [
                _first(_SHOW),
                _later(_CREATE, [1], active=2),
                _later(_CREATE_OTHER, [1, 2], ready=True),
            ]
        )
        controller.next_turn(tool_output=None)
        controller.observe_tool_result(_SHOW, _ok())
        second = controller.next_turn(tool_output="ok")
        self.assertIsNone(second.rejection)
        self.assertEqual(controller._state().completed_steps, frozenset({1}))
        third = controller.next_turn(tool_output="ok")
        self.assertIsNone(third.rejection)
        self.assertEqual(controller._state().completed_steps, frozenset({1, 2}))

        regressed, _base = _boot(
            [
                _first(_SHOW),
                _later(_CREATE, [1], active=2),
                _later(_SHOW, []),
            ]
        )
        regressed.next_turn(tool_output=None)
        regressed.next_turn(tool_output="ok")
        before_regress = _snapshot(regressed)
        regress_turn = regressed.next_turn(tool_output="ok")
        self.assertEqual(regress_turn.rejection, "workflow_envelope_invalid")
        self.assertEqual(_snapshot(regressed), before_regress)
        self.assertEqual(regressed._state().completed_steps, frozenset({1}))

        opened = {
            "duplicate ids": _later(_SHOW, [1, 1]),
            "unfinished is not the complement": _later(
                _SHOW,
                [1],
                unfinished=[1, 2],
                active=2,
                ready=False,
            ),
            "active step is finished": _later(_SHOW, [1], unfinished=[2], active=1),
            "ready flag disagrees": _later(
                _SHOW,
                [1],
                unfinished=[2],
                active=2,
                ready=True,
            ),
            "active step set while ready": _later(
                _SHOW,
                [1, 2],
                unfinished=[],
                active=1,
                ready=True,
            ),
            "plan redefined": _later(_SHOW, [], plan=_PLAN),
        }
        for label, text in opened.items():
            with self.subTest(case=label):
                fresh, _base = _boot([_first(_SHOW), text])
                fresh.next_turn(tool_output=None)
                before = _snapshot(fresh)
                turn = fresh.next_turn(tool_output="ok")
                self.assertEqual(turn.rejection, "workflow_envelope_invalid")
                self.assertEqual(_snapshot(fresh), before)
                self.assertEqual(fresh._state().plan, tuple(_PLAN))

    def test_malformed_markdown_and_action_failures(self) -> None:
        valid = _first(_SHOW)
        cases = {
            "malformed json": "{",
            "markdown fence": "```json\n" + valid + "\n```",
            "missing action": json.dumps(
                {
                    "plan": _PLAN,
                    "completed_steps": [],
                    "active_step": 1,
                    "ready_to_complete": False,
                    "unfinished_steps": [1, 2],
                }
            ),
        }
        for label, text in cases.items():
            with self.subTest(case=label):
                controller, _base = _boot([text])
                turn = controller.next_turn(tool_output=None)
                self.assertEqual(turn.rejection, "workflow_envelope_invalid")
                self.assertIsNone(controller._state().plan)
                self.assertIn("workflow JSON", turn.feedback or "")
        prose, _base = _boot(
            [
                json.dumps(
                    {
                        "plan": _PLAN,
                        "completed_steps": [],
                        "active_step": 1,
                        "ready_to_complete": False,
                        "unfinished_steps": [1, 2],
                        "action": "look up the calendar",
                    }
                )
            ]
        )
        prose_turn = prose.next_turn(tool_output=None)
        self.assertIsNone(prose_turn.action)
        self.assertNotEqual(prose_turn.rejection, "workflow_envelope_invalid")
        self.assertIn("exactly one documented", prose_turn.feedback or "")
        self.assertEqual(prose._state().plan, tuple(_PLAN))
        stopped, _base = _boot(
            [
                json.dumps(
                    {
                        "plan": _PLAN,
                        "completed_steps": [],
                        "active_step": 1,
                        "ready_to_complete": False,
                        "unfinished_steps": [1, 2],
                        "action": "STOP",
                    }
                )
            ]
        )
        stop_turn = stopped.next_turn(tool_output=None)
        self.assertIsNone(stop_turn.action)
        self.assertIn("STOP", stop_turn.rejection or "")
        self.assertEqual(stopped._state().plan, tuple(_PLAN))
        accepted, _base = _boot([_first(_SHOW)])
        accepted_turn = accepted.next_turn(tool_output=None)
        self.assertEqual(accepted_turn.action, _SHOW)
        self.assertIsNone(accepted_turn.rejection)
        self.assertIsNone(accepted_turn.feedback)

    def test_begin_and_each_turn_use_one_generation(self) -> None:
        controller, base = _boot(["{", _first(_SHOW), "{"])
        self.assertEqual(base.begin_calls, 1)
        self.assertEqual(base.generations, 0)
        controller.next_turn(tool_output=None)
        self.assertEqual(base.generations, 1)
        controller.next_turn(tool_output="feedback")
        controller.next_turn(tool_output="ok")
        self.assertEqual(base.generations, 3)
        self.assertTrue(all(call["parse_action"] is False for call in base.calls))
        self.assertEqual(base.evaluate_calls, 0)

    def test_exact_repeats_block_the_third_proposal(self) -> None:
        self.assertEqual(canonical_action(_CREATE), canonical_action(_SHOW_WIDE))
        self.assertNotEqual(canonical_action(_CREATE), canonical_action(_CREATE_OTHER))
        controller, _base = _boot(
            [
                _first(_CREATE),
                _later(_CREATE),
                _later(_CREATE),
            ]
        )
        first = controller.next_turn(tool_output=None)
        self.assertEqual(first.action, _CREATE)
        controller.observe_tool_result(first.action or "", _ok())
        second = controller.next_turn(tool_output="ok")
        self.assertEqual(second.action, _CREATE)
        controller.observe_tool_result(second.action or "", _ok())
        self.assertEqual(controller._state().executed_tool_actions, 2)
        self.assertEqual(controller._state().consecutive_same_action_count, 2)
        third = controller.next_turn(tool_output="ok")
        self.assertEqual(third.rejection, "workflow_repeated_action_blocked")
        self.assertIsNone(third.action)
        self.assertIn("exact same action", third.feedback or "")
        self.assertEqual(controller._state().executed_tool_actions, 2)
        self.assertEqual(controller._state().consecutive_same_action_count, 2)

    def test_whitespace_matches_and_different_arguments_do_not(self) -> None:
        controller, _base = _boot(
            [
                _first(_CREATE),
                _later(_SHOW_WIDE),
                _later(_CREATE),
            ]
        )
        first = controller.next_turn(tool_output=None)
        controller.observe_tool_result(first.action or "", _ok())
        second = controller.next_turn(tool_output="ok")
        self.assertEqual(second.action, _SHOW_WIDE.strip())
        self.assertIsNone(second.rejection)
        controller.observe_tool_result(second.action or "", _ok())
        third = controller.next_turn(tool_output="ok")
        self.assertEqual(third.rejection, "workflow_repeated_action_blocked")
        other, _base = _boot([_first(_CREATE), _later(_CREATE_OTHER)])
        other.next_turn(tool_output=None)
        other.observe_tool_result(_CREATE, _ok())
        changed = other.next_turn(tool_output="ok")
        self.assertEqual(changed.action, _CREATE_OTHER)
        self.assertIsNone(changed.rejection)
        other.observe_tool_result(changed.action or "", _ok())
        self.assertEqual(other._state().consecutive_same_action_count, 1)
        self.assertEqual(other._state().executed_tool_actions, 2)

    def test_progress_counter_waits_for_executed_actions_and_resets(self) -> None:
        settings = _settings(no_progress_turns=3)
        controller, _base = _boot(
            [
                _first(_SHOW),
                _later(_CREATE),
                _later(_CREATE_OTHER),
                _later("apis.calendar.delete_event(event_id=1)", [1], active=2),
            ],
            settings,
        )
        controller.next_turn(tool_output=None)
        self.assertEqual(controller._state().no_progress_turns, 0)
        controller.next_turn(tool_output=None)
        self.assertEqual(controller._state().executed_tool_actions, 0)
        self.assertEqual(controller._state().no_progress_turns, 0)
        controller.observe_tool_result(_SHOW, _ok())
        controller.next_turn(tool_output="ok")
        self.assertEqual(controller._state().no_progress_turns, 1)
        controller.observe_tool_result(_CREATE, _ok())
        progressed = controller.next_turn(tool_output="ok")
        self.assertIsNone(progressed.rejection)
        self.assertEqual(controller._state().no_progress_turns, 0)
        self.assertEqual(controller._state().completed_steps, frozenset({1}))

    def test_no_progress_threshold_adds_stall_text(self) -> None:
        settings = _settings(no_progress_turns=3)
        actions = [
            _SHOW,
            _CREATE,
            _CREATE_OTHER,
            "apis.calendar.delete_event(event_id=1)",
            "apis.calendar.delete_event(event_id=2)",
        ]
        outputs = [_first(actions[0])]
        outputs.extend(_later(action) for action in actions[1:])
        controller, base = _boot(outputs, settings)
        controller.next_turn(tool_output=None)
        controller.observe_tool_result(actions[0], _ok())
        for action in actions[1:3]:
            controller.next_turn(tool_output="ok")
            controller.observe_tool_result(action, _ok())
        controller.next_turn(tool_output="ok")
        state = controller._state()
        self.assertEqual(state.no_progress_turns, 3)
        self.assertEqual(
            state.stall_reason,
            "no declared plan progress for 3 action turns",
        )
        preview = workflow_instruction(state, settings)
        self.assertIn("STALL DETECTED: no declared plan progress for 3 action turns", preview)
        self.assertNotIn("evaluator", preview.lower())
        controller.next_turn(tool_output="ok")
        instruction = base.calls[-1]["extra_instruction"]
        assert isinstance(instruction, str)
        self.assertIn("STALL DETECTED:", instruction)
        self.assertIn("no declared plan progress for 3 action turns", instruction)

    def test_completion_gate_blocks_until_the_ledger_is_complete(self) -> None:
        blocked, base = _boot(
            [
                _first(_SHOW),
                _later(_DONE, [1], active=2),
            ]
        )
        blocked.next_turn(tool_output=None)
        blocked.observe_tool_result(_SHOW, _ok())
        turn = blocked.next_turn(tool_output="ok")
        self.assertEqual(turn.rejection, "workflow_completion_blocked")
        self.assertIsNone(turn.action)
        self.assertIn("unfinished steps", turn.feedback or "")
        self.assertEqual(base.evaluate_calls, 0)
        allowed, allowed_base = _boot(
            [
                _first(_SHOW),
                _later(_DONE, [1, 2], ready=True),
            ]
        )
        allowed.next_turn(tool_output=None)
        done = allowed.next_turn(tool_output="ok")
        self.assertIsNone(done.rejection)
        self.assertEqual(done.action, _DONE)
        self.assertEqual(done.app_name, "supervisor")
        self.assertEqual(done.api_name, "complete_task")
        self.assertEqual(allowed_base.evaluate_calls, 0)
        self.assertEqual(allowed_base.generations, 2)
        finished = allowed._state()
        instruction = workflow_instruction(finished, _settings())
        self.assertIn("apis.supervisor.complete_task(...)", instruction)
        self.assertNotIn("evaluator", instruction.lower())

    def test_observer_records_execution_and_clears_repeated_stalls(self) -> None:
        controller, _base = _boot(
            [
                _first(_CREATE),
                _later(_CREATE),
                _later(_CREATE_OTHER),
            ]
        )
        first = controller.next_turn(tool_output=None)
        controller.observe_tool_result(first.action or "", _ok())
        self.assertEqual(controller._state().executed_tool_actions, 1)
        self.assertIs(controller._state().last_action_had_error, False)
        failed, _base = _boot([_first(_CREATE), _later(_CREATE), _later(_SHOW)])
        failed.next_turn(tool_output=None)
        failed.observe_tool_result(_CREATE, _err())
        self.assertIs(failed._state().last_action_had_error, True)
        failed.next_turn(tool_output="err")
        failed.observe_tool_result(_CREATE, _err())
        self.assertEqual(failed._state().stall_reason, "repeated failed action")
        cleared = failed.next_turn(tool_output="err")
        self.assertIsNone(cleared.rejection)
        failed.observe_tool_result(cleared.action or "", _ok())
        self.assertIsNone(failed._state().stall_reason)
        repeated, _base = _boot([_first(_CREATE), _later(_CREATE), _later(_SHOW)])
        repeated.next_turn(tool_output=None)
        repeated.observe_tool_result(_CREATE, _ok())
        repeated.next_turn(tool_output="ok")
        repeated.observe_tool_result(_CREATE, _ok())
        self.assertEqual(
            repeated._state().stall_reason,
            "repeated successful action without declared workflow progress",
        )
        moved = repeated.next_turn(tool_output="ok")
        repeated.observe_tool_result(moved.action or "", _ok())
        self.assertIsNone(repeated._state().stall_reason)

    def test_underlying_agent_and_delegation(self) -> None:
        settings = _settings()
        base = _Base([])
        controller = WorkflowControlledAgent(base, settings)
        self.assertEqual(controller.underlying_agents(), (base,))
        controller.begin(_CONTEXT, _run(settings))
        controller.teacher_force_plan(
            messages=[{"role": "user", "content": "prompt"}],
            plan_text="1. read",
        )
        self.assertEqual(
            base.forced,
            ([{"role": "user", "content": "prompt"}], "1. read"),
        )
        other = _settings(no_progress_turns=4)
        with self.assertRaisesRegex(ValueError, "do not match"):
            controller.begin(_CONTEXT, _run(other))
        object.__setattr__(settings, "policy", "other_policy")
        with self.assertRaisesRegex(ValueError, "plan_progress_v1"):
            WorkflowControlledAgent(base, settings)

    def test_direct_parser_rejects_non_objects(self) -> None:
        state_settings = _settings()
        with self.assertRaises(WorkflowEnvelopeError):
            parse_workflow_envelope(
                "[]",
                state=workflow_module.WorkflowState(),
                settings=state_settings,
            )

    def test_parseable_action_helper_is_looser_than_the_envelope(self) -> None:
        self.assertTrue(
            workflow_output_has_parseable_action(
                json.dumps({"action": _SHOW})
            )
        )
        self.assertFalse(workflow_output_has_parseable_action("{"))
        self.assertFalse(workflow_output_has_parseable_action("```json\n{}\n```"))
        self.assertFalse(workflow_output_has_parseable_action(json.dumps({"action": "STOP"})))
        self.assertFalse(
            workflow_output_has_parseable_action(
                json.dumps({"action": "look up the calendar"})
            )
        )
        self.assertFalse(workflow_output_has_parseable_action("[]"))
        self.assertFalse(workflow_output_has_parseable_action(json.dumps({"plan": _PLAN})))

    def test_controller_source_does_not_read_ground_truth(self) -> None:
        source = Path(workflow_module.__file__).read_text(encoding="utf-8")
        self.assertNotIn("ground_truth", source)
        self.assertNotIn(".evaluate(", source)

    def test_extra_instruction_is_request_local(self) -> None:
        agent = SmolagentsVLLMAgent("http://127.0.0.1:9")
        agent.set_mode("execute")
        agent.begin(_CONTEXT, RunConfiguration.from_dict(_payload()))
        body = {
            "choices": [
                {
                    "message": {"content": "not an action"},
                    "token_ids": [7],
                }
            ]
        }
        captured: dict[str, object] = {}

        class _Response:
            def read(self) -> bytes:
                return json.dumps(body).encode("utf-8")

            def __enter__(self) -> "_Response":
                return self

            def __exit__(self, *args: object) -> None:
                return None

        def _open(request: object, timeout: object = None) -> _Response:
            del timeout
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _Response()

        with patch("urllib.request.urlopen", side_effect=_open):
            turn = agent.generate_turn(
                tool_output="tool said ok",
                extra_instruction="WORKFLOW CONTROLLER",
                parse_action=False,
            )
        self.assertIsNone(turn.action)
        self.assertIsNone(turn.rejection)
        messages = captured["body"]["messages"]
        self.assertEqual(messages[-1]["content"], "WORKFLOW CONTROLLER")
        self.assertEqual(messages[-2]["content"], "tool said ok")
        history = [item["content"] for item in agent._state().history]
        self.assertEqual(history, ["tool said ok", "not an action"])
        self.assertEqual(agent._state().history[0]["role"], "user")
        self.assertEqual(agent._state().history[1]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
