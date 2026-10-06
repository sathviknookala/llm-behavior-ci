from __future__ import annotations

import ast
import json
import threading
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from llm_behavior_ci.config import RunConfiguration, WorkflowSettings
from llm_behavior_ci.runtime.actions import ActionRejected, parse_model_output
from llm_behavior_ci.runtime.agent import (
    AgentTurn,
    SmolagentsAnthropicAgent,
    SmolagentsVLLMAgent,
)
from llm_behavior_ci.runtime.appworld import TaskContext, ToolResult
from llm_behavior_ci.runtime.prompts import PROMPT_RUNTIME_AUTH_V2

_MIN_PLAN_STEPS = 2
_MAX_PLAN_STEP_CHARS = 120
_POLICY_V1 = "plan_progress_v1"
_POLICY_V2 = "plan_progress_v2"
FORMAT_RETRY_LIMIT = 2
_FIRST_KEYS = frozenset(
    {
        "plan",
        "completed_steps",
        "active_step",
        "ready_to_complete",
        "unfinished_steps",
        "action",
    }
)
_LATER_KEYS = frozenset(
    {
        "completed_steps",
        "active_step",
        "ready_to_complete",
        "unfinished_steps",
        "action",
    }
)
_V2_FIRST_KEYS = frozenset({"plan", "active_step", "action"})
_V2_LATER_KEYS = frozenset({"mark_completed", "active_step", "action"})
_ENVELOPE_FEEDBACK = (
    "Your workflow response was invalid. "
    "Return exactly the required workflow JSON object. "
    "Preserve all previously completed steps and propose "
    "exactly one documented apis.<app>.<api>(...) action."
)
_ACTION_FEEDBACK = (
    "The action field must contain exactly one documented "
    "apis.<app>.<api>(...) call."
)
_COMPLETION_FEEDBACK = (
    "Completion was blocked because your workflow ledger "
    "still has unfinished steps. Continue with one action "
    "that advances an unfinished step. Do not call "
    "complete_task until every declared plan step is complete."
)
_REPEAT_FEEDBACK = (
    "The exact same action has already repeated without "
    "sufficient progress. Choose a different documented "
    "action that advances an unfinished plan step."
)
_ENVELOPE_FEEDBACK_V2 = (
    "Your workflow response was invalid. "
    "Return exactly the required workflow JSON object "
    "and exactly one documented apis.<app>.<api>(...) action."
)
_COMPLETION_FEEDBACK_V2 = (
    "Completion was blocked because unfinished declared "
    "plan steps remain. Continue with one action that "
    "advances an unfinished step. Do not call complete_task "
    "until every declared plan step is complete."
)
_API_ERROR_RECOVERY = (
    "The previous API call was rejected.\n"
    "\n"
    "Read the returned error message and correct the offending API name "
    "or argument.\n"
    "Do not blindly repeat the same invalid call unchanged."
)


@dataclass(frozen=True)
class WorkflowEnvelope:
    plan: tuple[str, ...] | None
    completed_steps: tuple[int, ...]
    active_step: int | None
    ready_to_complete: bool
    unfinished_steps: tuple[int, ...]
    action: str


@dataclass(frozen=True)
class WorkflowEnvelopeV2:
    plan: tuple[str, ...] | None
    mark_completed: tuple[int, ...]
    active_step: int | None
    action: str


@dataclass
class WorkflowState:
    plan: tuple[str, ...] | None = None
    completed_steps: frozenset[int] = frozenset()
    active_step: int | None = None
    no_progress_turns: int = 0
    executed_tool_actions: int = 0
    last_action_key: str | None = None
    consecutive_same_action_count: int = 0
    last_action_had_error: bool | None = None
    stall_reason: str | None = None
    stall_event_count: int = 0
    consecutive_format_rejections: int = 0
    recoverable_api_error_pending: bool = False


class WorkflowEnvelopeError(ValueError):
    pass


def canonical_action(action: str) -> str:
    tree = ast.parse(action.strip(), mode="exec")
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _reject_keys(payload: dict[str, object], allowed: frozenset[str]) -> None:
    if set(payload) != allowed:
        raise WorkflowEnvelopeError("workflow response keys are not the required set")


def _parse_plan(value: object, settings: WorkflowSettings) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise WorkflowEnvelopeError("plan must be a list")
    if not _MIN_PLAN_STEPS <= len(value) <= settings.max_plan_steps:
        raise WorkflowEnvelopeError("plan length is outside the allowed range")
    steps: list[str] = []
    for item in value:
        if not isinstance(item, str) or item == "" or item != item.strip():
            raise WorkflowEnvelopeError("plan step must be a non-empty trimmed string")
        if "\n" in item or "\r" in item:
            raise WorkflowEnvelopeError("plan step must be one line")
        if len(item) > _MAX_PLAN_STEP_CHARS:
            raise WorkflowEnvelopeError("plan step exceeds 120 characters")
        steps.append(item)
    if len(set(steps)) != len(steps):
        raise WorkflowEnvelopeError("plan steps must be unique")
    return tuple(steps)


def _parse_completed(value: object, step_count: int) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise WorkflowEnvelopeError("completed_steps must be a list")
    numbers: list[int] = []
    for item in value:
        if not _is_int(item):
            raise WorkflowEnvelopeError("completed_steps must contain integers")
        if not 1 <= item <= step_count:
            raise WorkflowEnvelopeError("completed_steps must stay inside the plan")
        numbers.append(item)
    if any(
        numbers[index] >= numbers[index + 1] for index in range(len(numbers) - 1)
    ):
        raise WorkflowEnvelopeError("completed_steps must be strictly ascending")
    return tuple(numbers)


def _parse_unfinished(
    value: object,
    completed: tuple[int, ...],
    step_count: int,
) -> tuple[int, ...]:
    if not isinstance(value, list) or not all(_is_int(item) for item in value):
        raise WorkflowEnvelopeError("unfinished_steps must be a list of integers")
    expected = tuple(
        sorted(set(range(1, step_count + 1)) - set(completed))
    )
    if tuple(value) != expected:
        raise WorkflowEnvelopeError(
            "unfinished_steps must be the complement of completed_steps"
        )
    return expected


def _parse_ready(value: object, unfinished: tuple[int, ...]) -> bool:
    if not isinstance(value, bool):
        raise WorkflowEnvelopeError("ready_to_complete must be a boolean")
    if value != (len(unfinished) == 0):
        raise WorkflowEnvelopeError("ready_to_complete must match an empty unfinished list")
    return value


def _parse_active(value: object, *, ready: bool, unfinished: tuple[int, ...]) -> int | None:
    if ready:
        if value is not None:
            raise WorkflowEnvelopeError("active_step must be null when the plan is complete")
        return None
    if not _is_int(value) or value not in unfinished:
        raise WorkflowEnvelopeError("active_step must be an unfinished plan step")
    return value


def _parse_mark_completed(value: object, step_count: int) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise WorkflowEnvelopeError("mark_completed must be a list")
    numbers: list[int] = []
    seen: set[int] = set()
    for item in value:
        if not _is_int(item):
            raise WorkflowEnvelopeError("mark_completed must contain integers")
        if not 1 <= item <= step_count:
            raise WorkflowEnvelopeError("mark_completed must stay inside the plan")
        if item in seen:
            raise WorkflowEnvelopeError("mark_completed must not repeat a step")
        seen.add(item)
        numbers.append(item)
    return tuple(numbers)


def _parse_action_text(value: object) -> str:
    if not isinstance(value, str) or value == "" or value != value.strip():
        raise WorkflowEnvelopeError("action must be a non-empty trimmed string")
    return value


def parse_workflow_envelope(
    text: str,
    *,
    state: WorkflowState,
    settings: WorkflowSettings,
) -> WorkflowEnvelope:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise WorkflowEnvelopeError("workflow response is not JSON") from error
    if not isinstance(payload, dict):
        raise WorkflowEnvelopeError("workflow response must be a JSON object")
    if state.plan is None:
        _reject_keys(payload, _FIRST_KEYS)
        plan = _parse_plan(payload["plan"], settings)
        step_count = len(plan)
    else:
        _reject_keys(payload, _LATER_KEYS)
        plan = None
        step_count = len(state.plan)
    completed = _parse_completed(payload["completed_steps"], step_count)
    if not state.completed_steps <= frozenset(completed):
        raise WorkflowEnvelopeError("completed_steps must keep previously completed steps")
    unfinished = _parse_unfinished(payload["unfinished_steps"], completed, step_count)
    ready = _parse_ready(payload["ready_to_complete"], unfinished)
    active = _parse_active(payload["active_step"], ready=ready, unfinished=unfinished)
    action = _parse_action_text(payload["action"])
    if state.plan is None:
        if completed != ():
            raise WorkflowEnvelopeError("the first ledger must have no completed steps")
        if ready:
            raise WorkflowEnvelopeError("the first ledger must not be ready to complete")
        if unfinished != tuple(range(1, step_count + 1)):
            raise WorkflowEnvelopeError("the first ledger must leave every step unfinished")
    return WorkflowEnvelope(
        plan=plan,
        completed_steps=completed,
        active_step=active,
        ready_to_complete=ready,
        unfinished_steps=unfinished,
        action=action,
    )


def parse_workflow_envelope_v2(
    text: str,
    *,
    state: WorkflowState,
    settings: WorkflowSettings,
) -> WorkflowEnvelopeV2:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise WorkflowEnvelopeError("workflow response is not JSON") from error
    if not isinstance(payload, dict):
        raise WorkflowEnvelopeError("workflow response must be a JSON object")
    if state.plan is None:
        _reject_keys(payload, _V2_FIRST_KEYS)
        plan = _parse_plan(payload["plan"], settings)
        step_count = len(plan)
        completed: frozenset[int] = frozenset()
        mark: tuple[int, ...] = ()
    else:
        _reject_keys(payload, _V2_LATER_KEYS)
        plan = None
        step_count = len(state.plan)
        mark = _parse_mark_completed(payload["mark_completed"], step_count)
        completed = state.completed_steps | frozenset(mark)
    unfinished = tuple(sorted(set(range(1, step_count + 1)) - completed))
    active = _parse_active(
        payload["active_step"],
        ready=len(unfinished) == 0,
        unfinished=unfinished,
    )
    action = _parse_action_text(payload["action"])
    return WorkflowEnvelopeV2(
        plan=plan,
        mark_completed=mark,
        active_step=active,
        action=action,
    )


def unfinished_step_ids(state: WorkflowState) -> tuple[int, ...]:
    if state.plan is None:
        return ()
    return tuple(
        sorted(set(range(1, len(state.plan) + 1)) - state.completed_steps)
    )


def ready_to_complete(state: WorkflowState) -> bool:
    return state.plan is not None and not unfinished_step_ids(state)


def _stall_category(reason: str | None) -> str | None:
    if reason is None:
        return None
    if reason == "repeated failed action":
        return "repeated_failed"
    if reason.startswith("repeated successful action"):
        return "repeated_success"
    if reason.startswith("no declared plan progress"):
        return "no_progress"
    return reason


def _assign_stall(state: WorkflowState, reason: str | None) -> None:
    new_category = _stall_category(reason)
    old_category = _stall_category(state.stall_reason)
    if new_category is not None and new_category != old_category:
        state.stall_event_count += 1
    state.stall_reason = reason


def _number_list(values: list[int]) -> str:
    return json.dumps(values)


def workflow_instruction(state: WorkflowState, settings: WorkflowSettings) -> str:
    if settings.policy == _POLICY_V2:
        return _instruction_v2(state, settings)
    return _instruction_v1(state, settings)


def _instruction_v1(state: WorkflowState, settings: WorkflowSettings) -> str:
    if state.plan is None:
        return (
            "WORKFLOW CONTROLLER: plan_progress_v1\n"
            "\n"
            "Before the first tool execution, create a concise semantic plan "
            "for completing the user's task.\n"
            "\n"
            "Return exactly one JSON object and no prose or Markdown.\n"
            "\n"
            "Required JSON fields:\n"
            f'- "plan": 2 to {settings.max_plan_steps} concise semantic task subgoals.\n'
            '- "completed_steps": [] on this first turn.\n'
            '- "active_step": the numbered plan step you are currently advancing.\n'
            '- "ready_to_complete": false on this first turn.\n'
            '- "unfinished_steps": every numbered plan step.\n'
            '- "action": exactly one apis.<app>.<api>(...) call as a JSON string.\n'
            "\n"
            "Plan rules:\n"
            "- Describe task outcomes/subgoals, not authentication or runtime setup.\n"
            "- Do not include complete_task as a plan step.\n"
            "- Do not invent facts or claim work is already complete.\n"
            "- Each plan step must be one line and at most 120 characters.\n"
            "\n"
            "Action rules:\n"
            "- Use only APIs present in the supplied documentation.\n"
            "- Advance one unfinished plan step.\n"
            "- Do not include reasoning outside the JSON object.\n"
        )
    completed = sorted(state.completed_steps)
    unfinished = sorted(set(range(1, len(state.plan) + 1)) - state.completed_steps)
    active = "null" if state.active_step is None else str(state.active_step)
    lines = [
        "WORKFLOW CONTROLLER: plan_progress_v1",
        "",
        "Authoritative plan:",
    ]
    lines.extend(
        f"{number}. {step}" for number, step in enumerate(state.plan, start=1)
    )
    lines.extend(
        [
            "",
            "Controller state:",
            f"completed_steps: {_number_list(completed)}",
            f"unfinished_steps: {_number_list(unfinished)}",
            f"active_step: {active}",
            "",
            "The latest tool result is already present immediately before "
            "this workflow instruction.",
            "",
            "Update completed_steps only when the observed tool results support "
            "that the subgoal has been achieved.",
            "",
            "Previously completed steps may never become incomplete.",
            "",
            "Return exactly one JSON object and no prose or Markdown.",
            "",
            "Required fields:",
            '- "completed_steps"',
            '- "active_step"',
            '- "ready_to_complete"',
            '- "unfinished_steps"',
            '- "action"',
            "",
            'Do not include "plan" again.',
            "",
            "The action must be exactly one documented apis.<app>.<api>(...) "
            "call represented as a JSON string.",
        ]
    )
    if state.stall_reason is not None:
        lines.extend(
            [
                "",
                f"STALL DETECTED: {state.stall_reason}",
                "",
                "Reconsider how to advance an unfinished plan step. "
                "Do not blindly repeat a non-progressing action.",
            ]
        )
    if state.completed_steps == frozenset(range(1, len(state.plan) + 1)):
        lines.extend(
            [
                "",
                "All declared plan steps are complete. If that remains supported "
                "by the observed tool results, the final action should be "
                "apis.supervisor.complete_task(...).",
            ]
        )
    return "\n".join(lines)


def _instruction_v2(state: WorkflowState, settings: WorkflowSettings) -> str:
    if state.plan is None:
        return (
            "WORKFLOW CONTROLLER: plan_progress_v2\n"
            "\n"
            "Before the first tool execution, create a concise semantic plan "
            "for completing the user's task.\n"
            "\n"
            "Return exactly one JSON object and no prose or Markdown.\n"
            "\n"
            "Required JSON fields:\n"
            f'- "plan": 2 to {settings.max_plan_steps} concise semantic task subgoals.\n'
            '- "active_step": the numbered plan step this action advances.\n'
            '- "action": exactly one apis.<app>.<api>(...) call as a JSON string.\n'
            "\n"
            "Do not include completed_steps, unfinished_steps, ready_to_complete, "
            "or mark_completed. No plan step is complete before a tool result "
            "is observed. The controller owns those derived fields.\n"
            "\n"
            "Plan rules:\n"
            "- Describe task outcomes/subgoals, not authentication or runtime setup.\n"
            "- Do not include complete_task as a plan step.\n"
            "- Do not invent facts or claim work is already complete.\n"
            "- Each plan step must be one line and at most 120 characters.\n"
            "\n"
            "Action rules:\n"
            "- Use only APIs present in the supplied documentation.\n"
            "- Advance one unfinished plan step.\n"
            "- Do not include reasoning outside the JSON object.\n"
        )
    unfinished = list(unfinished_step_ids(state))
    active = "null" if state.active_step is None else str(state.active_step)
    lines = [
        "WORKFLOW CONTROLLER: plan_progress_v2",
        "",
        "Authoritative plan:",
    ]
    lines.extend(
        f"{number}. {step}" for number, step in enumerate(state.plan, start=1)
    )
    lines.extend(
        [
            "",
            "Controller state:",
            f"completed_steps: {_number_list(sorted(state.completed_steps))}",
            f"unfinished_steps: {_number_list(unfinished)}",
            f"ready_to_complete: {str(ready_to_complete(state)).lower()}",
            f"active_step: {active}",
            "",
            "The controller owns completed_steps, unfinished_steps, and "
            "ready_to_complete. Do not repeat them.",
            "",
            "The latest tool result is already present immediately before "
            "this workflow instruction.",
            "",
            "mark_completed lists plan steps that the observed tool results "
            "newly show are complete. Use an empty list when none are newly "
            "complete. A step that is already complete stays complete.",
            "",
            "Return exactly one JSON object and no prose or Markdown.",
            "",
            "Required fields:",
            '- "mark_completed"',
            '- "active_step"',
            '- "action"',
            "",
            'Do not include "plan", "completed_steps", "unfinished_steps", '
            'or "ready_to_complete".',
            "",
            "active_step must be an unfinished plan step. Use null only when "
            "every plan step is complete.",
            "",
            "The action must be exactly one documented apis.<app>.<api>(...) "
            "call represented as a JSON string.",
        ]
    )
    if state.stall_reason is not None:
        lines.extend(
            [
                "",
                f"STALL DETECTED: {state.stall_reason}",
                "",
                "Reconsider how you are advancing the unfinished plan. "
                "Do not repeat an action that is not completing a declared "
                "plan step.",
            ]
        )
    if ready_to_complete(state):
        lines.extend(
            [
                "",
                "All declared plan steps are complete. "
                "apis.supervisor.complete_task(...) is now permitted.",
            ]
        )
    return "\n".join(lines)


def _repeated_stall(reason: str | None) -> bool:
    return reason is not None and reason.startswith("repeated ")


def _invalid(
    raw: AgentTurn,
    rejection: str,
    feedback: str,
    *,
    consumes_execute_turn: bool = True,
) -> AgentTurn:
    return replace(
        raw,
        action=None,
        app_name=None,
        api_name=None,
        rejection=rejection,
        feedback=feedback,
        consumes_execute_turn=consumes_execute_turn,
    )


def _completion_allowed(state: WorkflowState, envelope: WorkflowEnvelope) -> bool:
    plan = state.plan
    return (
        envelope.ready_to_complete is True
        and not envelope.unfinished_steps
        and plan is not None
        and state.completed_steps == frozenset(range(1, len(plan) + 1))
        and envelope.active_step is None
    )


class WorkflowControlledAgent:
    def __init__(
        self,
        base_agent: SmolagentsVLLMAgent | SmolagentsAnthropicAgent,
        settings: WorkflowSettings,
    ) -> None:
        if settings.policy not in {_POLICY_V1, _POLICY_V2}:
            raise ValueError(
                "workflow policy must be plan_progress_v1 or plan_progress_v2"
            )
        self._base_agent = base_agent
        self._settings = settings
        self._local = threading.local()

    def underlying_agents(
        self,
    ) -> tuple[SmolagentsVLLMAgent | SmolagentsAnthropicAgent, ...]:
        return (self._base_agent,)

    def _state(self) -> WorkflowState:
        state = getattr(self._local, "state", None)
        if state is None:
            raise RuntimeError("agent begin was not called")
        return state

    def begin(self, context: TaskContext, config: RunConfiguration) -> None:
        if config.agent.workflow != self._settings:
            raise ValueError("workflow settings do not match runtime controller")
        self._local.state = WorkflowState()
        self._local.prompt_version = config.agent.prompt.prompt_version
        self._base_agent.begin(context, config)

    def teacher_force_plan(
        self,
        *,
        messages: list[dict[str, str]],
        plan_text: str,
    ):
        return self._base_agent.teacher_force_plan(
            messages=messages,
            plan_text=plan_text,
        )

    def next_turn(self, *, tool_output: str | None) -> AgentTurn:
        state = self._state()
        instruction = workflow_instruction(state, self._settings)
        if state.recoverable_api_error_pending:
            if getattr(self._local, "prompt_version", None) == PROMPT_RUNTIME_AUTH_V2:
                instruction = f"{instruction}\n\n{_API_ERROR_RECOVERY}"
            state.recoverable_api_error_pending = False
        raw = self._base_agent.generate_turn(
            tool_output=tool_output,
            extra_instruction=instruction,
            parse_action=False,
        )
        if self._settings.policy == _POLICY_V2:
            return self._decide_v2(raw, state)
        try:
            envelope = parse_workflow_envelope(
                raw.output_text,
                state=state,
                settings=self._settings,
            )
        except WorkflowEnvelopeError:
            return _invalid(raw, "workflow_envelope_invalid", _ENVELOPE_FEEDBACK)
        if state.plan is None:
            if envelope.plan is None:
                return _invalid(raw, "workflow_envelope_invalid", _ENVELOPE_FEEDBACK)
            state.plan = envelope.plan
        new_completed = frozenset(envelope.completed_steps)
        made_progress = bool(new_completed - state.completed_steps)
        if state.executed_tool_actions > 0:
            if made_progress:
                state.no_progress_turns = 0
            else:
                state.no_progress_turns += 1
        state.completed_steps = new_completed
        state.active_step = envelope.active_step
        if made_progress and (
            state.stall_reason is not None
            and state.stall_reason.startswith("no declared plan progress")
        ):
            _assign_stall(state, None)
        if (
            state.no_progress_turns >= self._settings.no_progress_turns
            and not _repeated_stall(state.stall_reason)
        ):
            _assign_stall(
                state,
                f"no declared plan progress for {state.no_progress_turns} action turns",
            )
        try:
            action, app_name, api_name = parse_model_output(envelope.action)
            if action is None:
                raise ActionRejected("STOP is not a workflow action")
        except ActionRejected as error:
            return _invalid(raw, str(error), _ACTION_FEEDBACK)
        completion = app_name == "supervisor" and api_name == "complete_task"
        if completion and self._settings.completion_gate:
            if not _completion_allowed(state, envelope):
                return _invalid(
                    raw,
                    "workflow_completion_blocked",
                    _COMPLETION_FEEDBACK,
                )
        else:
            key = canonical_action(action)
            if (
                key == state.last_action_key
                and state.consecutive_same_action_count
                >= self._settings.repeat_action_limit
            ):
                return _invalid(
                    raw,
                    "workflow_repeated_action_blocked",
                    _REPEAT_FEEDBACK,
                )
        return replace(
            raw,
            action=action,
            app_name=app_name,
            api_name=api_name,
            rejection=None,
            feedback=None,
        )

    def _decide_v2(self, raw: AgentTurn, state: WorkflowState) -> AgentTurn:
        settings = self._settings
        try:
            envelope = parse_workflow_envelope_v2(
                raw.output_text,
                state=state,
                settings=settings,
            )
        except WorkflowEnvelopeError:
            state.consecutive_format_rejections += 1
            consumes = state.consecutive_format_rejections > FORMAT_RETRY_LIMIT
            if consumes:
                state.consecutive_format_rejections = 0
            return _invalid(
                raw,
                "workflow_envelope_invalid",
                _ENVELOPE_FEEDBACK_V2,
                consumes_execute_turn=consumes,
            )
        state.consecutive_format_rejections = 0
        if state.plan is None:
            if envelope.plan is None:
                return _invalid(raw, "workflow_envelope_invalid", _ENVELOPE_FEEDBACK_V2)
            state.plan = envelope.plan
            state.completed_steps = frozenset()
            state.active_step = envelope.active_step
        else:
            new_steps = frozenset(envelope.mark_completed) - state.completed_steps
            made_progress = bool(new_steps)
            if state.executed_tool_actions > 0:
                if made_progress:
                    state.no_progress_turns = 0
                else:
                    state.no_progress_turns += 1
            state.completed_steps = state.completed_steps | frozenset(
                envelope.mark_completed
            )
            state.active_step = envelope.active_step
            if made_progress and (
                state.stall_reason is not None
                and state.stall_reason.startswith("no declared plan progress")
            ):
                _assign_stall(state, None)
            if (
                state.no_progress_turns >= settings.no_progress_turns
                and not _repeated_stall(state.stall_reason)
            ):
                _assign_stall(
                    state,
                    "no declared plan progress for "
                    f"{state.no_progress_turns} action turns",
                )
        try:
            action, app_name, api_name = parse_model_output(envelope.action)
            if action is None:
                raise ActionRejected("STOP is not a workflow action")
        except ActionRejected as error:
            return _invalid(raw, str(error), _ACTION_FEEDBACK)
        completion = app_name == "supervisor" and api_name == "complete_task"
        if completion and settings.completion_gate:
            plan = state.plan
            allowed = (
                plan is not None
                and state.completed_steps == frozenset(range(1, len(plan) + 1))
                and state.active_step is None
            )
            if not allowed:
                return _invalid(
                    raw,
                    "workflow_completion_blocked",
                    _COMPLETION_FEEDBACK_V2,
                )
        else:
            key = canonical_action(action)
            if (
                key == state.last_action_key
                and state.consecutive_same_action_count >= settings.repeat_action_limit
            ):
                return _invalid(
                    raw,
                    "workflow_repeated_action_blocked",
                    _REPEAT_FEEDBACK,
                )
        return replace(
            raw,
            action=action,
            app_name=app_name,
            api_name=api_name,
            rejection=None,
            feedback=None,
        )

    def observe_tool_result(self, action: str, result: ToolResult) -> None:
        state = self._state()
        state.executed_tool_actions += 1
        key = canonical_action(action)
        same = key == state.last_action_key
        if same:
            state.consecutive_same_action_count += 1
        else:
            state.last_action_key = key
            state.consecutive_same_action_count = 1
        state.last_action_had_error = result.error_message is not None
        state.recoverable_api_error_pending = (
            result.error_message is not None and result.recoverable
        )
        if state.consecutive_same_action_count >= self._settings.repeat_action_limit:
            if result.error_message is not None:
                _assign_stall(state, "repeated failed action")
            else:
                _assign_stall(
                    state,
                    "repeated successful action without declared workflow progress",
                )
        elif (
            not same
            and result.error_message is None
            and _repeated_stall(state.stall_reason)
        ):
            _assign_stall(state, None)


def workflow_output_has_parseable_action(text: str) -> bool:
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(payload, dict):
        return False
    action = payload.get("action")
    if not isinstance(action, str):
        return False
    try:
        parsed_action, _, _ = parse_model_output(action)
    except ActionRejected:
        return False
    return parsed_action is not None


@dataclass(frozen=True)
class WorkflowTraceAccounting:
    """Controller counts replayed from one episode's stored generations.

    ``workflow_format_retry_count`` is the number of malformed envelopes
    that did not spend an execute turn. A further consecutive malformed
    envelope is still an envelope rejection, and it does spend a turn.
    ``productive_model_turns`` counts generations that spend a turn,
    including policy blocks and that capped format failure.
    """

    model_generation_count: int
    workflow_envelope_rejection_count: int
    repeat_gate_block_count: int
    completion_gate_block_count: int
    stall_event_count: int
    executed_tool_action_count: int
    productive_model_turns: int
    workflow_format_retry_count: int


class _TraceModel:
    def __init__(self, outputs: Sequence[str]) -> None:
        self._outputs = list(outputs)

    def generate_turn(
        self,
        *,
        tool_output: str | None,
        extra_instruction: str | None = None,
        parse_action: bool = True,
    ) -> AgentTurn:
        del tool_output, extra_instruction, parse_action
        return AgentTurn(
            prompt_text="workflow",
            output_text=self._outputs.pop(0),
            top_k_logprobs=(),
            generated_token_count=1,
            latency_seconds=0.0,
            started_at=datetime(2026, 10, 3, tzinfo=timezone.utc),
            action=None,
            app_name=None,
            api_name=None,
        )


def account_workflow_trace(
    generations: Sequence[tuple[int, str]],
    tools: Sequence[tuple[int, str, bool]],
    settings: WorkflowSettings,
) -> WorkflowTraceAccounting:
    """Replay stored generations through the controller that produced them.

    ``generations`` are ``(step_index, output_text)`` in execution order.
    ``tools`` are ``(step_index, action, had_error)`` for executed calls.
    A generation that the controller accepts must be followed by the tool
    step at the next index. Policy and format rejections have no tool step.
    """

    model = _TraceModel([output for _index, output in generations])
    controller = WorkflowControlledAgent(model, settings)
    controller._local.state = WorkflowState()
    remaining = {index: had_error for index, _action, had_error in tools}
    envelope_rejections = 0
    repeat_blocks = 0
    completion_blocks = 0
    productive_turns = 0
    format_retries = 0
    executed = 0
    for index, _output in generations:
        turn = controller.next_turn(tool_output=None)
        if turn.rejection == "workflow_envelope_invalid":
            envelope_rejections += 1
        elif turn.rejection == "workflow_repeated_action_blocked":
            repeat_blocks += 1
        elif turn.rejection == "workflow_completion_blocked":
            completion_blocks += 1
        if turn.consumes_execute_turn:
            productive_turns += 1
        else:
            format_retries += 1
        if turn.rejection is None and turn.action is not None:
            try:
                had_error = remaining.pop(index + 1)
            except KeyError as error:
                raise ValueError(
                    "workflow trace is missing an executed tool step"
                ) from error
            executed += 1
            controller.observe_tool_result(
                turn.action,
                ToolResult(
                    output_text=None,
                    error_message="tool error" if had_error else None,
                    recoverable=True,
                    app_name=None,
                    api_name=None,
                ),
            )
    if remaining:
        raise ValueError("workflow trace left tool steps unconsumed")
    return WorkflowTraceAccounting(
        model_generation_count=len(generations),
        workflow_envelope_rejection_count=envelope_rejections,
        repeat_gate_block_count=repeat_blocks,
        completion_gate_block_count=completion_blocks,
        stall_event_count=controller._state().stall_event_count,
        executed_tool_action_count=executed,
        productive_model_turns=productive_turns,
        workflow_format_retry_count=format_retries,
    )
