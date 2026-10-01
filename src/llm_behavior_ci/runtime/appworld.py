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


_SPOTIFY_CAPABILITY_PROFILE = "spotify_capability_v1"

_SPOTIFY_SUPERVISOR_APIS = frozenset(
    {
        "show_profile",
        "show_account_passwords",
        "complete_task",
    }
)

_SPOTIFY_API_DOC_APIS = frozenset(
    {
        "show_api_descriptions",
        "show_api_doc",
    }
)


def _call_target(action: str) -> tuple[str, str]:
    """Return ``(app, api)`` for one ``apis.<app>.<api>(...)`` call."""

    stripped = action.strip()
    tree = ast.parse(stripped, mode="exec")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        raise ValueError("action must be one call")
    call = tree.body[0].value
    if not isinstance(call, ast.Call):
        raise ValueError("action must be one call")
    func = call.func
    if not isinstance(func, ast.Attribute):
        raise ValueError("action must target apis.<app>.<api>")
    api_name = func.attr
    app_node = func.value
    if (
        not isinstance(app_node, ast.Attribute)
        or not isinstance(app_node.value, ast.Name)
        or app_node.value.id != "apis"
    ):
        raise ValueError("action must target apis.<app>.<api>")
    return app_node.attr, api_name


def _keyword_string(action: str, name: str) -> str | None:
    """Return one string keyword from a call, or None when it is absent."""

    tree = ast.parse(action.strip(), mode="exec")
    call = tree.body[0].value
    if not isinstance(call, ast.Call):
        return None
    for keyword in call.keywords:
        if keyword.arg != name:
            continue
        if isinstance(keyword.value, ast.Constant) and isinstance(
            keyword.value.value, str
        ):
            return keyword.value.value
    return None


def _spotify_capability_allows(action: str) -> bool:
    """Whether one action stays inside the Spotify capability profile.

    Spotify APIs are allowed. Supervisor is limited to profile, account
    passwords, and task completion. ApiDocs is limited to the two lookup
    helpers, and only when the requested app is Spotify, so those helpers
    cannot browse an unrelated app.
    """

    try:
        app_name, api_name = _call_target(action)
    except (SyntaxError, ValueError):
        return False
    if app_name == "spotify":
        return True
    if app_name == "supervisor":
        return api_name in _SPOTIFY_SUPERVISOR_APIS
    if app_name == "api_docs":
        if api_name not in _SPOTIFY_API_DOC_APIS:
            return False
        return _keyword_string(action, "app_name") == "spotify"
    return False


def _spotify_capability_docs(documentation: object) -> object:
    """Keep Spotify and the approved supervisor and api_docs helpers."""

    if not isinstance(documentation, Mapping):
        return documentation
    selected: dict[str, object] = {}
    if "spotify" in documentation:
        selected["spotify"] = documentation["spotify"]
    supervisor = documentation.get("supervisor")
    if isinstance(supervisor, Mapping):
        selected["supervisor"] = {
            name: doc
            for name, doc in supervisor.items()
            if name in _SPOTIFY_SUPERVISOR_APIS
        }
    api_docs = documentation.get("api_docs")
    if isinstance(api_docs, Mapping):
        selected["api_docs"] = {
            name: doc
            for name, doc in api_docs.items()
            if name in _SPOTIFY_API_DOC_APIS
        }
    return selected


class LiveAppWorldSession:
    """AppWorld session adapter that imports AppWorld only when a world is opened.

    A second session for the same task does not open its world while
    another live world is still open. AppWorld starts nested time
    freezers that cannot be stopped safely in that state. The deferred
    session serves ``context`` from the open world, then opens its own
    world after that one has closed.

    ``context`` reads the task instruction and renders API docs without
    executing. ``execute`` prints a single call expression before handing
    it to AppWorld, so the return value is on stdout instead of being
    replaced by ``Execution successful.``. It treats AppWorld's returned
    ``Execution failed.`` text as a recoverable tool error; a raised
    exception is the same.
    ``evaluate`` reads ``pass_count`` and ``num_tests`` from the
    ``TestTracker``. ``close`` is idempotent. The live world has no
    ``initial_state_identity`` method.

    ``tool_access_profile`` ``spotify_capability_v1`` renders only the
    Spotify app plus the approved supervisor and api_docs helpers, and
    ``execute`` rejects every other API before AppWorld runs it. Any
    other value, including unset, leaves the existing surface unchanged.
    """

    _open_stack: list[LiveAppWorldSession] = []

    def __init__(
        self,
        task_id: str,
        *,
        opener: Callable[[str], object] | None = None,
        tool_access_profile: str | None = None,
    ) -> None:
        self._closed = False
        self._task_id = task_id
        self._opener = opener or _open_appworld
        self._tool_access_profile = tool_access_profile
        self._world: object | None = None
        if any(
            session._world is not None and not session._closed
            for session in type(self)._open_stack
        ):
            return
        self._open_world()

    def _open_world(self) -> None:
        try:
            self._world = self._opener(self._task_id)
        except ImportError as error:
            from llm_behavior_ci.runtime.episode import RuntimeUnavailable

            raise RuntimeUnavailable("AppWorld is not installed") from error
        type(self)._open_stack.append(self)

    def required_apps(self) -> tuple[str, ...]:
        if self._world is None:
            self._open_world()
        ground_truth = getattr(self._world.task, "ground_truth", None)
        if ground_truth is None:
            return ()
        apps = getattr(ground_truth, "required_apps", ())
        return tuple(str(app) for app in apps)

    def complete_without_work(self) -> None:
        from llm_behavior_ci.runtime.episode import RuntimeUnavailable

        result = self.execute("apis.supervisor.complete_task()")
        if result.error_message is not None:
            raise RuntimeUnavailable("do-nothing completion failed")

    def context(self) -> TaskContext:
        if self._world is None:
            for session in reversed(type(self)._open_stack):
                if (
                    session._task_id == self._task_id
                    and session._world is not None
                    and not session._closed
                ):
                    return session.context()
            self._open_world()
        task = self._world.task
        raw_docs = getattr(task, "api_docs", "")
        if self._tool_access_profile == _SPOTIFY_CAPABILITY_PROFILE:
            raw_docs = _spotify_capability_docs(raw_docs)
        return TaskContext(
            task_id=self._task_id,
            instruction=task.instruction,
            api_documentation=render_api_documentation(raw_docs),
        )

    def execute(self, action: str) -> ToolResult:
        if self._tool_access_profile == _SPOTIFY_CAPABILITY_PROFILE:
            if not _spotify_capability_allows(action):
                return ToolResult(
                    output_text=None,
                    error_message="API is outside the configured capability profile",
                    recoverable=True,
                    app_name=None,
                    api_name=None,
                )
        if self._world is None:
            self._open_world()
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

        if self._world is None:
            self._open_world()
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
        if self._world is None:
            return
        stack = type(self)._open_stack
        if self in stack:
            stack.remove(self)
        closer = getattr(self._world, "close", None)
        if callable(closer):
            closer()
