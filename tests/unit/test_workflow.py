import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from llm_behavior_ci.config import RunConfiguration, WorkflowSettings
from llm_behavior_ci.runtime.agent import AgentTurn, SmolagentsVLLMAgent
from llm_behavior_ci.runtime.appworld import TaskContext, ToolResult
from llm_behavior_ci.runtime.workflow import (
    FORMAT_RETRY_LIMIT,
    WorkflowControlledAgent,
    WorkflowEnvelopeError,
    account_workflow_trace,
    canonical_action,
    parse_workflow_envelope,
    parse_workflow_envelope_v2,
    ready_to_complete,
    unfinished_step_ids,
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


def _settings_v2(**overrides: object) -> WorkflowSettings:
    return _settings(policy="plan_progress_v2", **overrides)


def _first_v2(action: str, plan: list[str] | None = None, active: int = 1) -> str:
    return json.dumps(
        {
            "plan": list(plan or _PLAN),
            "active_step": active,
            "action": action,
        }
    )


def _later_v2(
    action: str,
    mark: list[int] | None = None,
    active: int | None = 1,
) -> str:
    return json.dumps(
        {
            "mark_completed": list(mark or []),
            "active_step": active,
            "action": action,
        }
    )


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
        lowered = source.lower()
        self.assertNotIn("ground_truth", source)
        self.assertNotIn("ground truth", lowered)
        self.assertNotIn(".evaluate(", source)
        self.assertNotIn("evaluator", lowered)
        self.assertNotIn("requirement", lowered)

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


class WorkflowV2Tests(unittest.TestCase):
    def test_first_turn_accepts_a_bounded_plan_and_one_action(self) -> None:
        for count in (2, 5):
            with self.subTest(steps=count):
                plan = [f"Step {index}" for index in range(1, count + 1)]
                controller, base = _boot(
                    [_first_v2(_SHOW, plan=plan, active=count)],
                    _settings_v2(),
                )
                turn = controller.next_turn(tool_output=None)
                state = controller._state()
                self.assertIsNone(turn.rejection)
                self.assertEqual(turn.action, _SHOW)
                self.assertEqual(state.plan, tuple(plan))
                self.assertEqual(state.completed_steps, frozenset())
                self.assertEqual(state.active_step, count)
                self.assertEqual(unfinished_step_ids(state), tuple(range(1, count + 1)))
                self.assertFalse(ready_to_complete(state))
                self.assertEqual(base.generations, 1)
                instruction = base.calls[0]["extra_instruction"]
                assert isinstance(instruction, str)
                self.assertIn("WORKFLOW CONTROLLER: plan_progress_v2", instruction)
                self.assertIn("2 to 5", instruction)
                self.assertNotIn("Authoritative plan:", instruction)
                self.assertNotIn("evaluator", instruction.lower())
                self.assertNotIn("ground truth", instruction.lower())

    def test_first_turn_rejections_do_not_claim_progress(self) -> None:
        long_step = "x" * 121
        cases = {
            "one step": _first_v2(_SHOW, plan=["Only this step"]),
            "six steps": _first_v2(_SHOW, plan=[f"Step {index}" for index in range(6)]),
            "duplicate steps": _first_v2(
                _SHOW,
                plan=["Read the calendar", "Read the calendar"],
            ),
            "multiline step": _first_v2(_SHOW, plan=["Read the calendar", "Create\nthe event"]),
            "too long": _first_v2(_SHOW, plan=["Read the calendar", long_step]),
            "blank step": _first_v2(_SHOW, plan=["Read the calendar", " "]),
            "untrimmed step": _first_v2(_SHOW, plan=["Read the calendar", " Create the event"]),
            "active step zero": _first_v2(_SHOW, active=0),
            "active step past the plan": _first_v2(_SHOW, active=3),
            "null active step": json.dumps(
                {"plan": _PLAN, "active_step": None, "action": _SHOW}
            ),
            "boolean active step": json.dumps(
                {"plan": _PLAN, "active_step": True, "action": _SHOW}
            ),
            "missing action": json.dumps({"plan": _PLAN, "active_step": 1}),
            "missing plan": json.dumps({"active_step": 1, "action": _SHOW}),
            "extra completed steps": json.dumps(
                {
                    "plan": _PLAN,
                    "active_step": 1,
                    "completed_steps": [],
                    "action": _SHOW,
                }
            ),
            "extra ledger fields": json.dumps(
                {
                    "plan": _PLAN,
                    "active_step": 1,
                    "ready_to_complete": False,
                    "unfinished_steps": [1, 2],
                    "action": _SHOW,
                }
            ),
            "premature mark": json.dumps(
                {
                    "plan": _PLAN,
                    "mark_completed": [1],
                    "active_step": 2,
                    "action": _SHOW,
                }
            ),
            "malformed json": "{",
            "markdown fence": "```json\n" + _first_v2(_SHOW) + "\n```",
            "json array": "[]",
        }
        for label, text in cases.items():
            with self.subTest(case=label):
                controller, _base = _boot([text], _settings_v2())
                turn = controller.next_turn(tool_output=None)
                self.assertEqual(turn.rejection, "workflow_envelope_invalid")
                self.assertIsNone(turn.action)
                self.assertFalse(turn.consumes_execute_turn)
                self.assertIsNone(controller._state().plan)
                self.assertEqual(controller._state().completed_steps, frozenset())
        prose, _base = _boot(
            [
                json.dumps(
                    {
                        "plan": _PLAN,
                        "active_step": 1,
                        "action": "look up the calendar",
                    }
                )
            ],
            _settings_v2(),
        )
        prose_turn = prose.next_turn(tool_output=None)
        self.assertIsNone(prose_turn.action)
        self.assertNotEqual(prose_turn.rejection, "workflow_envelope_invalid")
        self.assertTrue(prose_turn.consumes_execute_turn)
        self.assertEqual(prose._state().plan, tuple(_PLAN))
        self.assertEqual(prose._state().completed_steps, frozenset())
        stopped, _base = _boot(
            [
                json.dumps(
                    {"plan": _PLAN, "active_step": 1, "action": "STOP"}
                )
            ],
            _settings_v2(),
        )
        stop_turn = stopped.next_turn(tool_output=None)
        self.assertIn("STOP", stop_turn.rejection or "")
        self.assertTrue(stop_turn.consumes_execute_turn)
        self.assertEqual(stopped._state().completed_steps, frozenset())

    def test_incremental_updates_are_monotonic_and_derived(self) -> None:
        waiting, _base = _boot(
            [_first_v2(_SHOW), _later_v2(_CREATE, mark=[])],
            _settings_v2(),
        )
        waiting.next_turn(tool_output=None)
        waiting.next_turn(tool_output=None)
        self.assertEqual(waiting._state().executed_tool_actions, 0)
        self.assertEqual(waiting._state().no_progress_turns, 0)
        self.assertEqual(waiting._state().completed_steps, frozenset())
        delete = "apis.calendar.delete_event(event_id=1)"
        controller, base = _boot(
            [
                _first_v2(_SHOW, active=1),
                _later_v2(_CREATE, mark=[], active=1),
                _later_v2(_CREATE_OTHER, mark=[1], active=2),
                _later_v2(delete, mark=[1], active=2),
                _later_v2("apis.calendar.delete_event(event_id=2)", mark=[1, 2], active=None),
            ],
            _settings_v2(),
        )
        first = controller.next_turn(tool_output=None)
        controller.observe_tool_result(first.action or "", _ok())
        empty = controller.next_turn(tool_output="ok")
        self.assertIsNone(empty.rejection)
        self.assertEqual(controller._state().completed_steps, frozenset())
        self.assertEqual(unfinished_step_ids(controller._state()), (1, 2))
        self.assertFalse(ready_to_complete(controller._state()))
        self.assertEqual(controller._state().no_progress_turns, 1)
        controller.observe_tool_result(empty.action or "", _ok())
        one = controller.next_turn(tool_output="ok")
        self.assertIsNone(one.rejection)
        self.assertEqual(controller._state().completed_steps, frozenset({1}))
        self.assertEqual(unfinished_step_ids(controller._state()), (2,))
        self.assertFalse(ready_to_complete(controller._state()))
        self.assertEqual(controller._state().no_progress_turns, 0)
        self.assertEqual(controller._state().active_step, 2)
        later_instruction = base.calls[1]["extra_instruction"]
        assert isinstance(later_instruction, str)
        self.assertIn('"mark_completed"', later_instruction)
        self.assertNotIn("evaluator", later_instruction.lower())
        controller.observe_tool_result(one.action or "", _ok())
        again = controller.next_turn(tool_output="ok")
        self.assertIsNone(again.rejection)
        self.assertEqual(controller._state().completed_steps, frozenset({1}))
        self.assertEqual(controller._state().no_progress_turns, 1)
        controller.observe_tool_result(again.action or "", _ok())
        both = controller.next_turn(tool_output="ok")
        self.assertIsNone(both.rejection)
        self.assertEqual(controller._state().completed_steps, frozenset({1, 2}))
        self.assertEqual(unfinished_step_ids(controller._state()), ())
        self.assertTrue(ready_to_complete(controller._state()))
        self.assertIsNone(controller._state().active_step)
        finished = workflow_instruction(controller._state(), _settings_v2())
        self.assertIn("ready_to_complete: true", finished)
        self.assertIn("complete_task(...) is now permitted", finished)
        controller.observe_tool_result(both.action or "", _err())
        self.assertEqual(controller._state().completed_steps, frozenset({1, 2}))

        invalid = {
            "unknown step": _later_v2(_SHOW, mark=[9], active=1),
            "duplicate mark": _later_v2(_SHOW, mark=[1, 1], active=2),
            "boolean mark": json.dumps(
                {"mark_completed": [True], "active_step": 1, "action": _SHOW}
            ),
            "completed step stays active": _later_v2(_SHOW, mark=[1], active=1),
            "null while unfinished": _later_v2(_SHOW, mark=[], active=None),
            "number while complete": _later_v2(_SHOW, mark=[1, 2], active=1),
            "reintroduced plan": json.dumps(
                {
                    "plan": _PLAN,
                    "mark_completed": [],
                    "active_step": 1,
                    "action": _SHOW,
                }
            ),
            "echoed ledger": json.dumps(
                {
                    "mark_completed": [],
                    "completed_steps": [1],
                    "unfinished_steps": [2],
                    "ready_to_complete": False,
                    "active_step": 2,
                    "action": _SHOW,
                }
            ),
        }
        for label, text in invalid.items():
            with self.subTest(case=label):
                fresh, _base = _boot(
                    [_first_v2(_SHOW), text],
                    _settings_v2(),
                )
                fresh.next_turn(tool_output=None)
                fresh.observe_tool_result(_SHOW, _ok())
                before = _snapshot(fresh)
                turn = fresh.next_turn(tool_output="ok")
                self.assertEqual(turn.rejection, "workflow_envelope_invalid")
                self.assertIsNone(turn.action)
                self.assertEqual(_snapshot(fresh), before)

    def test_completion_uses_the_derived_ledger_only(self) -> None:
        blocked, base = _boot(
            [
                _first_v2(_SHOW),
                _later_v2(_DONE, mark=[1], active=2),
            ],
            _settings_v2(),
        )
        blocked.next_turn(tool_output=None)
        blocked.observe_tool_result(_SHOW, _ok())
        turn = blocked.next_turn(tool_output="ok")
        self.assertEqual(turn.rejection, "workflow_completion_blocked")
        self.assertIsNone(turn.action)
        self.assertTrue(turn.consumes_execute_turn)
        self.assertIn("unfinished declared", turn.feedback or "")
        self.assertNotIn("evaluator", (turn.feedback or "").lower())
        self.assertEqual(base.evaluate_calls, 0)
        self.assertEqual(blocked._state().completed_steps, frozenset({1}))
        allowed, allowed_base = _boot(
            [
                _first_v2(_SHOW),
                _later_v2(_DONE, mark=[1, 2], active=None),
            ],
            _settings_v2(),
        )
        allowed.next_turn(tool_output=None)
        done = allowed.next_turn(tool_output="ok")
        self.assertIsNone(done.rejection)
        self.assertEqual(done.action, _DONE)
        self.assertEqual(done.api_name, "complete_task")
        self.assertEqual(allowed_base.evaluate_calls, 0)
        self.assertEqual(allowed_base.generations, 2)
        instruction = workflow_instruction(allowed._state(), _settings_v2())
        self.assertIn("apis.supervisor.complete_task(...) is now permitted", instruction)
        self.assertNotIn("evaluator", instruction.lower())
        self.assertNotIn("ready_to_complete", _later_v2(_DONE, mark=[1, 2], active=None))

    def test_repeat_gate_blocks_the_third_canonical_action(self) -> None:
        self.assertEqual(canonical_action(_CREATE), canonical_action(_SHOW_WIDE))
        controller, _base = _boot(
            [
                _first_v2(_CREATE),
                _later_v2(_CREATE, mark=[]),
                _later_v2(_CREATE, mark=[]),
            ],
            _settings_v2(),
        )
        first = controller.next_turn(tool_output=None)
        controller.observe_tool_result(first.action or "", _ok())
        second = controller.next_turn(tool_output="ok")
        self.assertIsNone(second.rejection)
        self.assertEqual(second.action, _CREATE)
        controller.observe_tool_result(second.action or "", _ok())
        third = controller.next_turn(tool_output="ok")
        self.assertEqual(third.rejection, "workflow_repeated_action_blocked")
        self.assertIsNone(third.action)
        self.assertTrue(third.consumes_execute_turn)
        self.assertEqual(controller._state().executed_tool_actions, 2)
        spaced, _base = _boot(
            [
                _first_v2(_CREATE),
                _later_v2(_SHOW_WIDE, mark=[]),
                _later_v2(_CREATE, mark=[]),
            ],
            _settings_v2(),
        )
        spaced.next_turn(tool_output=None)
        spaced.observe_tool_result(_CREATE, _ok())
        wide = spaced.next_turn(tool_output="ok")
        self.assertIsNone(wide.rejection)
        self.assertEqual(wide.action, _SHOW_WIDE.strip())
        spaced.observe_tool_result(wide.action or "", _ok())
        blocked = spaced.next_turn(tool_output="ok")
        self.assertEqual(blocked.rejection, "workflow_repeated_action_blocked")
        changed, _base = _boot(
            [_first_v2(_CREATE), _later_v2(_CREATE_OTHER, mark=[])],
            _settings_v2(),
        )
        changed.next_turn(tool_output=None)
        changed.observe_tool_result(_CREATE, _ok())
        other = changed.next_turn(tool_output="ok")
        self.assertIsNone(other.rejection)
        self.assertEqual(other.action, _CREATE_OTHER)
        changed.observe_tool_result(other.action or "", _ok())
        self.assertEqual(changed._state().consecutive_same_action_count, 1)

    def test_stalls_follow_executed_actions_and_stay_distinct(self) -> None:
        actions = [
            _SHOW,
            _CREATE,
            _CREATE_OTHER,
            "apis.calendar.delete_event(event_id=1)",
        ]
        outputs = [_first_v2(actions[0])]
        outputs.extend(_later_v2(action, mark=[]) for action in actions[1:])
        controller, base = _boot(outputs, _settings_v2())
        controller.next_turn(tool_output=None)
        controller.observe_tool_result(actions[0], _ok())
        controller.next_turn(tool_output="ok")
        controller.observe_tool_result(actions[1], _ok())
        self.assertIsNone(controller._state().stall_reason)
        self.assertEqual(controller._state().no_progress_turns, 1)
        controller.next_turn(tool_output="ok")
        controller.observe_tool_result(actions[2], _ok())
        self.assertIsNone(controller._state().stall_reason)
        self.assertEqual(controller._state().no_progress_turns, 2)
        controller.next_turn(tool_output="ok")
        state = controller._state()
        self.assertEqual(state.no_progress_turns, 3)
        self.assertEqual(state.stall_reason, "no declared plan progress for 3 action turns")
        self.assertEqual(state.stall_event_count, 1)
        preview = workflow_instruction(state, _settings_v2())
        self.assertIn("STALL DETECTED: no declared plan progress for 3 action turns", preview)
        self.assertIn("Reconsider how you are advancing the unfinished plan", preview)
        self.assertNotIn("spotify", preview.lower())
        self.assertNotIn("evaluator", preview.lower())
        cleared, cleared_base = _boot(
            [
                _first_v2(_SHOW),
                _later_v2(_CREATE, mark=[]),
                _later_v2(_CREATE_OTHER, mark=[]),
                _later_v2("apis.calendar.delete_event(event_id=1)", mark=[]),
                _later_v2("apis.calendar.delete_event(event_id=2)", mark=[1], active=2),
            ],
            _settings_v2(),
        )
        cleared.next_turn(tool_output=None)
        cleared.observe_tool_result(_SHOW, _ok())
        for action in (_CREATE, _CREATE_OTHER):
            cleared.next_turn(tool_output="ok")
            cleared.observe_tool_result(action, _ok())
        cleared.next_turn(tool_output="ok")
        self.assertIsNotNone(cleared._state().stall_reason)
        progressed = cleared.next_turn(tool_output="ok")
        self.assertIsNone(progressed.rejection)
        self.assertIsNone(cleared._state().stall_reason)
        self.assertEqual(cleared._state().completed_steps, frozenset({1}))
        follow = workflow_instruction(cleared._state(), _settings_v2())
        self.assertNotIn("STALL DETECTED", follow)
        self.assertEqual(cleared_base.evaluate_calls, 0)
        repeated, _base = _boot(
            [
                _first_v2(_CREATE),
                _later_v2(_CREATE, mark=[]),
                _later_v2(_SHOW, mark=[1], active=2),
            ],
            _settings_v2(),
        )
        repeated.next_turn(tool_output=None)
        repeated.observe_tool_result(_CREATE, _ok())
        repeated.next_turn(tool_output="ok")
        repeated.observe_tool_result(_CREATE, _ok())
        self.assertEqual(
            repeated._state().stall_reason,
            "repeated successful action without declared workflow progress",
        )
        repeated.next_turn(tool_output="ok")
        self.assertEqual(
            repeated._state().stall_reason,
            "repeated successful action without declared workflow progress",
        )
        self.assertEqual(repeated._state().completed_steps, frozenset({1}))
        failed, _base = _boot(
            [_first_v2(_CREATE), _later_v2(_CREATE, mark=[])],
            _settings_v2(),
        )
        failed.next_turn(tool_output=None)
        failed.observe_tool_result(_CREATE, _err())
        failed.next_turn(tool_output="err")
        failed.observe_tool_result(_CREATE, _err())
        self.assertEqual(failed._state().stall_reason, "repeated failed action")
        self.assertNotEqual(
            failed._state().stall_reason,
            repeated._state().stall_reason,
        )

    def test_format_retries_are_bounded_and_do_not_loop(self) -> None:
        outputs = ["{", "{", "{", "{", _first_v2(_SHOW)]
        controller, base = _boot(outputs, _settings_v2())
        first = controller.next_turn(tool_output=None)
        second = controller.next_turn(tool_output="retry")
        third = controller.next_turn(tool_output="retry")
        fourth = controller.next_turn(tool_output="retry")
        self.assertFalse(first.consumes_execute_turn)
        self.assertFalse(second.consumes_execute_turn)
        self.assertTrue(third.consumes_execute_turn)
        self.assertFalse(fourth.consumes_execute_turn)
        self.assertEqual(controller._state().plan, None)
        self.assertEqual(base.generations, 4)
        accepted = controller.next_turn(tool_output="retry")
        self.assertIsNone(accepted.rejection)
        self.assertTrue(accepted.consumes_execute_turn)
        self.assertEqual(base.generations, 5)
        self.assertEqual(FORMAT_RETRY_LIMIT, 2)

    def test_trace_accounting_separates_format_policy_and_execution(self) -> None:
        show = _first_v2(_SHOW)
        blocked = _later_v2(_DONE, mark=[1], active=2)
        repeated = _later_v2(_SHOW, mark=[])
        progressed = _later_v2(_CREATE, mark=[1], active=2)
        generations = [
            (0, "{"),
            (1, "{"),
            (2, "{"),
            (3, show),
            (5, repeated),
            (7, repeated),
            (8, blocked),
            (9, progressed),
        ]
        tools = [(4, _SHOW, False), (6, _SHOW, False), (10, _CREATE, False)]
        accounted = account_workflow_trace(generations, tools, _settings_v2())
        self.assertEqual(accounted.model_generation_count, 8)
        self.assertEqual(accounted.workflow_envelope_rejection_count, 3)
        self.assertEqual(accounted.workflow_format_retry_count, 2)
        self.assertEqual(accounted.completion_gate_block_count, 1)
        self.assertEqual(accounted.repeat_gate_block_count, 1)
        self.assertEqual(accounted.executed_tool_action_count, 3)
        self.assertEqual(accounted.productive_model_turns, 6)
        self.assertTrue(
            workflow_output_has_parseable_action(blocked)
        )
        self.assertFalse(workflow_output_has_parseable_action("{"))
        self.assertGreaterEqual(accounted.stall_event_count, 1)
        with self.assertRaises(WorkflowEnvelopeError):
            parse_workflow_envelope_v2(
                "{",
                state=workflow_module.WorkflowState(),
                settings=_settings_v2(),
            )


if __name__ == "__main__":
    unittest.main()
