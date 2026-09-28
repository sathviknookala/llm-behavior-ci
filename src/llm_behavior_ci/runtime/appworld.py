from __future__ import annotations

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


class LiveAppWorldSession:
    """AppWorld session adapter that imports AppWorld only when a world is opened.

    ``context`` reads the task instruction and API docs without executing.
    ``execute`` maps a returned string or a raised error onto ``ToolResult``.
    ``evaluate`` reads the raw evaluator counts and lets ``RuntimeUnavailable``
    propagate. ``close`` is idempotent.
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
        documentation = getattr(task, "api_docs", "")
        if documentation is None:
            documentation = ""
        elif not isinstance(documentation, str):
            documentation = str(documentation)
        return TaskContext(
            task_id=self._task_id,
            instruction=task.instruction,
            api_documentation=documentation,
        )

    def execute(self, action: str) -> ToolResult:
        try:
            value = self._world.execute(action)
        except Exception as error:
            return ToolResult(
                output_text=None,
                error_message=str(error),
                recoverable=True,
                app_name=None,
                api_name=None,
            )
        if isinstance(value, str):
            return ToolResult(
                output_text=value,
                error_message=None,
                recoverable=False,
                app_name=None,
                api_name=None,
            )
        return ToolResult(
            output_text=str(value),
            error_message=None,
            recoverable=False,
            app_name=None,
            api_name=None,
        )

    def evaluate(self) -> EvaluationResult:
        from llm_behavior_ci.runtime.episode import RuntimeUnavailable

        raw = self._world.evaluate()
        success = bool(raw.success)
        passes = getattr(raw, "passes", None)
        fails = getattr(raw, "fails", None)
        if _is_sequence(passes) and _is_sequence(fails):
            passed = len(passes)
            total = passed + len(fails)
        elif hasattr(raw, "pass_count") and hasattr(raw, "fail_count"):
            passed = int(raw.pass_count)
            total = passed + int(raw.fail_count)
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
