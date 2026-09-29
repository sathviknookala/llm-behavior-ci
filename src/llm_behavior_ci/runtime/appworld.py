from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass(frozen=True)
class TaskContext:
    task_id: str
    instruction: str
    api_documentation: str


@dataclass(frozen=True)
class ToolResult:
    output_text: str | None
    error_message: str | None
    recoverable: bool
    app_name: str | None
    api_name: str | None


@dataclass(frozen=True)
class EvaluationResult:
    success: bool
    passed_requirements: int
    total_requirements: int
    difficulty: int | None


class AppWorldSession(Protocol):
    def context(self) -> TaskContext: ...

    def execute(self, action: str) -> ToolResult: ...

    def evaluate(self) -> EvaluationResult: ...

    def close(self) -> None: ...


def _open_appworld(task_id: str) -> object:
    try:
        from appworld import AppWorld
    except ImportError as error:
        from llm_behavior_ci.runtime.episode import RuntimeUnavailable

        raise RuntimeUnavailable("AppWorld is not installed") from error
    return AppWorld(task_id=task_id)


def _is_sequence(value: object) -> bool:
    return isinstance(value, (list, tuple))


def _parameter_text(parameters: object) -> str:
    parts: list[str] = []
    if isinstance(parameters, Mapping):
        required = parameters.get("required", [])
        optional = parameters.get("optional", [])
        if isinstance(required, list):
            parts.extend(
                name for name in required if isinstance(name, str) and name != ""
            )
        if isinstance(optional, list):
            parts.extend(
                f"{name}?"
                for name in optional
                if isinstance(name, str) and name != ""
            )
        return ", ".join(parts)
    if not isinstance(parameters, list):
        return ""
    for item in parameters:
        if not isinstance(item, Mapping):
            continue
        name = item.get("name")
        if not isinstance(name, str) or name == "":
            continue
        type_name = item.get("type")
        piece = name
        if isinstance(type_name, str) and type_name != "":
            piece = f"{name}:{type_name}"
        if not item.get("required"):
            piece += "?"
        parts.append(piece)
    return ", ".join(parts)


def _api_line(app_name: str, api_name: str, doc: object) -> str:
    description = ""
    parameters: object = []
    if isinstance(doc, Mapping):
        raw_description = doc.get("description", "")
        if isinstance(raw_description, str):
            description = " ".join(raw_description.split())
        parameters = doc.get("parameters", [])
    parameter_text = _parameter_text(parameters)
    if parameter_text:
        return f"{app_name}.{api_name}: {description} | {parameter_text}"
    return f"{app_name}.{api_name}: {description}"


def render_api_documentation(documentation: object) -> str:
    """Turn AppWorld's API-doc collection into line-oriented prompt text.

    A string is kept unchanged so fakes and already-rendered text stay
    stable. A mapping of app to API docs becomes one ``app.api:`` line per
    API, apps and APIs sorted, with the description and parameter name and
    type. That is the form ``api-docs-corrupt-v1`` can redact, and it drops
    response schemas. Anything else falls back to ``str``.
    """

    if documentation is None:
        return ""
    if isinstance(documentation, str):
        return documentation
    if not isinstance(documentation, Mapping):
        return str(documentation)
    if not documentation:
        return ""
    lines: list[str] = []
    for app_name in sorted(documentation, key=str):
        apis = documentation[app_name]
        if not isinstance(apis, Mapping):
            return str(documentation)
        for api_name in sorted(apis, key=str):
            lines.append(_api_line(str(app_name), str(api_name), apis[api_name]))
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


_EXECUTION_FAILED = "Execution failed."
_NO_CODE = "No code available to execute."


def _code_for_execute(action: str) -> str:
    """Print a single call so AppWorld's stdout capture keeps the return value.

    AppWorld records stdout and substitutes ``Execution successful.`` when
    that stream is empty. A bare ``apis.<app>.<api>(...)`` expression
    therefore drops the response body, including tokens and records the
    next turn needs. ``print`` of one dict or list is AppWorld's own
    JSON printer. Anything that is not a single call is left unchanged.
    """

    stripped = action.strip()
    try:
        tree = ast.parse(stripped, mode="exec")
    except SyntaxError:
        return action
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        return action
    value = tree.body[0].value
    if not isinstance(value, ast.Call):
        return action
    func = value.func
    if isinstance(func, ast.Name) and func.id == "print":
        return action
    return f"print({stripped})"


def _execute_output_is_error(value: str) -> bool:
    stripped = value.lstrip()
    return stripped.startswith(_EXECUTION_FAILED) or stripped.startswith(_NO_CODE)


class LiveAppWorldSession:
    """AppWorld session adapter that imports AppWorld only when a world is opened.

    ``context`` reads the task instruction and renders API docs without
    executing. ``execute`` prints a single call expression before handing
    it to AppWorld, so the return value is on stdout instead of being
    replaced by ``Execution successful.``. It treats AppWorld's returned
    ``Execution failed.`` text as a recoverable tool error; a raised
    exception is the same.
    ``evaluate`` reads ``pass_count`` and ``num_tests`` from the
    ``TestTracker``. ``close`` is idempotent. The live world has no
    ``initial_state_identity`` method.
    """

    def __init__(
        self,
        task_id: str,
        *,
        opener: Callable[[str], object] | None = None,
    ) -> None:
        try:
            self._world = (opener or _open_appworld)(task_id)
        except ImportError as error:
            from llm_behavior_ci.runtime.episode import RuntimeUnavailable

            raise RuntimeUnavailable("AppWorld is not installed") from error
        self._closed = False
        self._task_id = task_id

    def context(self) -> TaskContext:
        task = self._world.task
        return TaskContext(
            task_id=self._task_id,
            instruction=task.instruction,
            api_documentation=render_api_documentation(getattr(task, "api_docs", "")),
        )

    def execute(self, action: str) -> ToolResult:
        try:
            value = self._world.execute(_code_for_execute(action))
        except Exception as error:
            return ToolResult(
                output_text=None,
                error_message=str(error),
                recoverable=True,
                app_name=None,
                api_name=None,
            )
        text = value if isinstance(value, str) else str(value)
        if _execute_output_is_error(text):
            return ToolResult(
                output_text=None,
                error_message=text,
                recoverable=True,
                app_name=None,
                api_name=None,
            )
        return ToolResult(
            output_text=text,
            error_message=None,
            recoverable=False,
            app_name=None,
            api_name=None,
        )

    def evaluate(self) -> EvaluationResult:
        from llm_behavior_ci.runtime.episode import RuntimeUnavailable

        raw = self._world.evaluate()
        success = bool(raw.success)
        if hasattr(raw, "pass_count") and hasattr(raw, "num_tests"):
            passed = int(raw.pass_count)
            total = int(raw.num_tests)
        else:
            passes = getattr(raw, "passes", None)
            failures = getattr(raw, "failures", None)
            if _is_sequence(passes) and _is_sequence(failures):
                passed = len(passes)
                total = passed + len(failures)
            else:
                raise RuntimeUnavailable("evaluation result is missing counts")
        difficulty = getattr(raw, "difficulty", None)
        return EvaluationResult(
            success=success,
            passed_requirements=passed,
            total_requirements=total,
            difficulty=difficulty,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        closer = getattr(self._world, "close", None)
        if callable(closer):
            closer()
