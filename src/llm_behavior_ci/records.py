"""Shared episode, evidence, and decision records.

Local records are the episode log: task identity, observable plan text,
model and tool steps, evaluator outcome, and errors. Public records are
aggregates: statistical evidence, lifecycle decisions, and counts. A
public payload does not carry task identity, plan text, prompts, tool
output, or an evaluator report.

``EvaluatorOutcome.success`` is the evaluator's task-success flag and is
present only when execution reaches evaluation. ``agent_stopped`` records
that the agent ended its loop. A recoverable tool error is a tool-step
error with ``recoverable`` true; the episode status stays ``completed``
when the loop finishes. An episode that terminates unsuccessfully has
status ``failed`` and a failed termination reason.

``requirement_fraction`` is ``passed_requirements / total_requirements``
when ``total_requirements`` is positive. When ``total_requirements`` is
zero the fraction is undefined and the value is ``None``. Success follows
the evaluator rule ``passed_requirements == total_requirements``, including
the empty count, and the undefined fraction stays ``None``.
"""

from __future__ import annotations

import math
import re
from dataclasses import MISSING, dataclass, fields, is_dataclass
from datetime import datetime
from types import UnionType
from typing import Any, ClassVar, Mapping, Union, get_args, get_origin, get_type_hints

from llm_behavior_ci.config import (
    MONITOR_SIGNALS,
    SPLITS,
    EpisodeIdentity,
    RunIdentity,
)

LOCAL = "local"
PUBLIC = "public"
MODES = frozenset({"plan", "execute"})
EPISODE_STATUSES = frozenset({"completed", "failed"})
PAIR_ROLES = frozenset({"reference", "candidate"})
COMPLETED_TERMINATIONS = frozenset({"plan_emitted", "agent_stopped"})
FAILED_TERMINATIONS = frozenset(
    {
        "step_limit",
        "unrecoverable_tool_error",
        "runtime_error",
        "timeout",
        "cancelled",
    }
)
TERMINATION_REASONS = COMPLETED_TERMINATIONS | FAILED_TERMINATIONS
PLAN_TERMINATIONS = frozenset(
    {
        "plan_emitted",
        "step_limit",
        "runtime_error",
        "timeout",
        "cancelled",
    }
)
EXECUTE_TERMINATIONS = frozenset(
    {
        "agent_stopped",
        "step_limit",
        "unrecoverable_tool_error",
        "runtime_error",
        "timeout",
        "cancelled",
    }
)
ERROR_SOURCES = frozenset(
    {"tool", "runtime", "timeout", "cancelled", "step_limit"}
)
DIFFICULTIES = frozenset({1, 2, 3})
TIER_ACTIONS = {
    "offline_gate": frozenset({"allow_canary", "block"}),
    "canary": frozenset({"continue", "rollback", "promote"}),
    "production_monitor": frozenset({"alert", "no_alert"}),
}
TIERS = frozenset(TIER_ACTIONS)
PROTECTED_FIELDS = frozenset(
    {
        "task_id",
        "scenario_id",
        "plan_text",
        "prompt_text",
        "output_text",
        "action",
        "message",
        "instruction",
        "requirement",
        "passes",
        "failures",
        "episode_id",
        "pair_id",
    }
)
_TERMINATING_ERROR = {
    "runtime_error": "runtime",
    "timeout": "timeout",
    "cancelled": "cancelled",
    "step_limit": "step_limit",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_MAX_LINE = 256


class RecordError(ValueError):
    pass


class Record:
    visibility: ClassVar[str] = ""
    record_name: ClassVar[str] = "record"

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        for field in fields(self):
            payload[field.name] = _to_plain(getattr(self, field.name))
        payload["visibility"] = self.visibility
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: object,
        name: str | None = None,
    ) -> Record:
        label = name or cls.record_name
        return _from_mapping(cls, _prepare(cls, payload, label), label)


def _to_plain(value: object) -> object:
    if isinstance(value, Record):
        return value.to_dict()
    if is_dataclass(value):
        exporter = getattr(type(value), "to_dict", None)
        if callable(exporter):
            return value.to_dict()
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RecordError(f"{name} must be an object")
    return value


def _prepare(cls: type[Record], payload: object, name: str) -> dict[str, object]:
    mapping = dict(_object(payload, name))
    if not all(isinstance(key, str) for key in mapping):
        raise RecordError(f"{name} has a non-string field name")
    if "visibility" in mapping and mapping.pop("visibility") != cls.visibility:
        raise RecordError(f"{name} visibility must be {cls.visibility}")
    return mapping


def _union_args(annotation: object) -> tuple[object, ...] | None:
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        return get_args(annotation)
    return None


def _decode(annotation: object, value: object, name: str) -> object:
    union_args = _union_args(annotation)
    if union_args is not None:
        if value is None and type(None) in union_args:
            return None
        remaining = [arg for arg in union_args if arg is not type(None)]
        if len(remaining) == 1:
            return _decode(remaining[0], value, name)
        raise RecordError(f"{name} has an unsupported type")
    origin = get_origin(annotation)
    if origin is tuple:
        args = get_args(annotation)
        if isinstance(value, str) or not isinstance(value, (list, tuple)):
            raise RecordError(f"{name} must be a list")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode(args[0], item, name) for item in value)
        raise RecordError(f"{name} has an unsupported list shape")
    if annotation is datetime:
        return _timestamp(value, name)
    if isinstance(annotation, type) and is_dataclass(annotation):
        if not isinstance(value, Mapping):
            raise RecordError(f"{name} must be an object")
        return annotation.from_dict(value, name=name)
    if annotation is bool:
        if not isinstance(value, bool):
            raise RecordError(f"{name} must be a boolean")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise RecordError(f"{name} must be an integer")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, float):
            raise RecordError(f"{name} must be a float")
        return value
    if annotation is str:
        if not isinstance(value, str):
            raise RecordError(f"{name} must be a string")
        return value
    raise RecordError(f"{name} has an unsupported type")


def _from_mapping(cls: type, mapping: Mapping[str, object], name: str) -> Any:
    known = {field.name for field in fields(cls)}
    unknown = sorted(set(mapping) - known)
    if unknown:
        joined = ", ".join(unknown)
        raise RecordError(f"{name} contains unknown fields: {joined}")
    hints = get_type_hints(cls)
    arguments: dict[str, object] = {}
    missing: list[str] = []
    for field in fields(cls):
        if field.name not in mapping:
            if field.default is MISSING and field.default_factory is MISSING:
                missing.append(field.name)
            continue
        arguments[field.name] = _decode(
            hints[field.name],
            mapping[field.name],
            f"{name}.{field.name}",
        )
    if missing:
        joined = ", ".join(sorted(missing))
        raise RecordError(f"{name} is missing fields: {joined}")
    return cls(**arguments)


def _timestamp(value: object, name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as error:
            raise RecordError(f"{name} must be an ISO-8601 timestamp") from error
    else:
        raise RecordError(f"{name} must be an ISO-8601 timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RecordError(f"{name} must be timezone-aware")
    return parsed


def _flag(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise RecordError(f"{name} must be a boolean")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecordError(f"{name} must be an integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    number = _integer(value, name)
    if number < 0:
        raise RecordError(f"{name} must be zero or greater")
    return number


def _positive_int(value: object, name: str) -> int:
    number = _integer(value, name)
    if number < 1:
        raise RecordError(f"{name} must be a positive integer")
    return number


def _float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, float):
        raise RecordError(f"{name} must be a float")
    if not math.isfinite(value):
        raise RecordError(f"{name} must be finite")
    return value


def _nonnegative_float(value: object, name: str) -> float:
    number = _float(value, name)
    if number < 0.0:
        raise RecordError(f"{name} must be zero or greater")
    return number


def _closed_unit(value: object, name: str) -> float:
    number = _float(value, name)
    if not 0.0 <= number <= 1.0:
        raise RecordError(f"{name} must be between zero and one")
    return number


def _open_probability(value: object, name: str) -> float:
    number = _float(value, name)
    if not 0.0 < number < 1.0:
        raise RecordError(
            f"{name} must be greater than zero and less than one"
        )
    return number


def _line(value: object, name: str) -> str:
    if not isinstance(value, str) or value == "" or value != value.strip():
        raise RecordError(f"{name} must be a non-empty string")
    if len(value) > _MAX_LINE or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise RecordError(
            f"{name} must be a single line of at most {_MAX_LINE} characters"
        )
    return value


def _body(value: object, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise RecordError(f"{name} must be a string")
    if "\x00" in value or any(
        ord(character) < 32 and character not in "\n\r\t" for character in value
    ):
        raise RecordError(f"{name} contains a control character")
    if not allow_empty and value.strip() == "":
        raise RecordError(f"{name} must be non-empty")
    return value


def _token(value: object, name: str) -> str:
    if not isinstance(value, str) or _TOKEN.fullmatch(value) is None:
        raise RecordError(f"{name} must be a lowercase identifier")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RecordError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _choice(value: object, allowed: frozenset[str], name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise RecordError(f"{name} must be one of: {choices}")
    return value


def _kind(value: object, cls: type, name: str) -> None:
    if not isinstance(value, cls):
        raise RecordError(f"{name} must be a {cls.__name__}")


def _instances(value: object, cls: type, name: str) -> tuple[Any, ...]:
    if not isinstance(value, tuple):
        raise RecordError(f"{name} must be a tuple")
    for item in value:
        _kind(item, cls, name)
    return value


def _logprob(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, float):
        raise RecordError(f"{name} must be a float")
    if math.isnan(value) or value == math.inf:
        raise RecordError(f"{name} must be a log-probability")
    return value


def _optional_confidence(
    low: object,
    high: object,
    level: object,
    name: str,
) -> None:
    present = (low is not None, high is not None, level is not None)
    if any(present) and not all(present):
        raise RecordError(f"{name} confidence bounds are incomplete")
    if not all(present):
        return
    _float(low, f"{name} confidence_low")
    _float(high, f"{name} confidence_high")
    _open_probability(level, f"{name} confidence_level")
    if low > high:
        raise RecordError(f"{name} confidence_low exceeds confidence_high")


def _check_fraction(stored: object, derived: float | None, name: str) -> None:
    if stored is None:
        if derived is not None:
            raise RecordError(
                f"{name} requirement_fraction does not match the counts"
            )
        return
    if isinstance(stored, bool) or not isinstance(stored, float):
        raise RecordError(f"{name} requirement_fraction must be a float or null")
    if not math.isfinite(stored) or stored != derived:
        raise RecordError(
            f"{name} requirement_fraction does not match the counts"
        )


@dataclass(frozen=True)
class TokenLogprob(Record):
    token_id: int
    logprob: float
    rank: int

    def __post_init__(self) -> None:
        _nonnegative_int(self.token_id, "token_id")
        _logprob(self.logprob, "logprob")
        _nonnegative_int(self.rank, "rank")


@dataclass(frozen=True)
class ModelStep(Record):
    index: int
    prompt_text: str
    output_text: str
    top_k_logprobs: tuple[tuple[TokenLogprob, ...], ...]
    latency_seconds: float
    started_at: datetime

    def __post_init__(self) -> None:
        _nonnegative_int(self.index, "index")
        _body(self.prompt_text, "prompt_text")
        _body(self.output_text, "output_text", allow_empty=True)
        _nonnegative_float(self.latency_seconds, "latency_seconds")
        _timestamp(self.started_at, "started_at")
        if not isinstance(self.top_k_logprobs, tuple):
            raise RecordError("top_k_logprobs must be a tuple")
        for position in self.top_k_logprobs:
            if not isinstance(position, tuple) or not position:
                raise RecordError("top_k_logprobs positions must be non-empty")
            ranks: list[int] = []
            token_ids: list[int] = []
            for alternative in position:
                _kind(alternative, TokenLogprob, "top_k_logprobs")
                ranks.append(alternative.rank)
                token_ids.append(alternative.token_id)
            if len(set(ranks)) != len(ranks) or len(set(token_ids)) != len(token_ids):
                raise RecordError("top_k_logprobs repeats a rank or token")


@dataclass(frozen=True)
class RecordedError(Record):
    """An error recorded on a tool step or on the episode.

    ``recoverable`` is true only for a tool error that leaves the episode
    running. Runtime, timeout, cancellation, and step-limit errors are
    unrecoverable and belong to a failed termination.
    """

    source: str
    recoverable: bool
    message: str
    step_index: int | None

    def __post_init__(self) -> None:
        _choice(self.source, ERROR_SOURCES, "source")
        _flag(self.recoverable, "recoverable")
        _body(self.message, "message")
        if self.source == "tool":
            _nonnegative_int(self.step_index, "step_index")
            return
        if self.recoverable:
            raise RecordError("only a tool error can be recoverable")
        if self.step_index is not None:
            raise RecordError("episode errors have no step index")


@dataclass(frozen=True)
class ToolStep(Record):
    index: int
    action: str
    app_name: str | None
    api_name: str | None
    output_text: str | None
    error: RecordedError | None
    latency_seconds: float
    started_at: datetime

    def __post_init__(self) -> None:
        _nonnegative_int(self.index, "index")
        _body(self.action, "action")
        if self.app_name is not None:
            _line(self.app_name, "app_name")
        if self.api_name is not None:
            _line(self.api_name, "api_name")
        if self.output_text is not None:
            _body(self.output_text, "output_text", allow_empty=True)
        _nonnegative_float(self.latency_seconds, "latency_seconds")
        _timestamp(self.started_at, "started_at")
        if self.error is None:
            return
        _kind(self.error, RecordedError, "error")
        if self.error.source != "tool" or self.error.step_index != self.index:
            raise RecordError("tool error must point at this step")


@dataclass(frozen=True)
class EvaluatorOutcome(Record):
    """AppWorld evaluator counts for one episode.

    ``success`` is true when ``passed_requirements == total_requirements``.
    ``requirement_fraction`` divides those counts. A zero total leaves the
    fraction undefined.
    """

    success: bool
    passed_requirements: int
    total_requirements: int
    difficulty: int | None

    def __post_init__(self) -> None:
        _flag(self.success, "success")
        _nonnegative_int(self.passed_requirements, "passed_requirements")
        _nonnegative_int(self.total_requirements, "total_requirements")
        if self.passed_requirements > self.total_requirements:
            raise RecordError("passed_requirements exceeds total_requirements")
        if self.success != (self.passed_requirements == self.total_requirements):
            raise RecordError("success does not match the requirement counts")
        if self.difficulty is not None and self.difficulty not in DIFFICULTIES:
            raise RecordError("difficulty must be 1, 2, or 3")

    @property
    def requirement_fraction(self) -> float | None:
        """Passed requirements divided by total requirements.

        The fraction is ``None`` when ``total_requirements`` is zero.
        Callers leave that case out of numeric series.
        """

        if self.total_requirements == 0:
            return None
        return self.passed_requirements / self.total_requirements

    def to_dict(self) -> dict[str, object]:
        payload = Record.to_dict(self)
        payload["requirement_fraction"] = self.requirement_fraction
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: object,
        name: str | None = None,
    ) -> EvaluatorOutcome:
        label = name or cls.record_name
        mapping = _prepare(cls, payload, label)
        if "requirement_fraction" not in mapping:
            raise RecordError(f"{label} is missing fields: requirement_fraction")
        stored = mapping.pop("requirement_fraction")
        outcome = _from_mapping(cls, mapping, label)
        _check_fraction(stored, outcome.requirement_fraction, label)
        return outcome


@dataclass(frozen=True)
class LocalTaskRef(Record):
    """Task identity that stays in the local log."""

    task_id: str
    scenario_id: str | None
    split: str

    def __post_init__(self) -> None:
        _line(self.task_id, "task_id")
        if self.scenario_id is not None:
            _line(self.scenario_id, "scenario_id")
        _choice(self.split, SPLITS, "split")


@dataclass(frozen=True)
class EpisodeResult(Record):
    """One local episode.

    ``run`` is the explicit run identity. ``plan_text`` is the observable
    plan when the episode has one. ``evaluator_outcome`` is present when
    execution reaches evaluation. ``recorded_errors`` collects tool-step
    errors and episode-level errors.
    """

    episode: EpisodeIdentity
    run: RunIdentity
    task: LocalTaskRef
    mode: str
    execution_seed: int
    status: str
    started_at: datetime
    ended_at: datetime
    model_steps: tuple[ModelStep, ...]
    tool_steps: tuple[ToolStep, ...]
    plan_text: str | None
    evaluator_outcome: EvaluatorOutcome | None
    termination_reason: str
    episode_errors: tuple[RecordedError, ...]
    role: str | None

    def __post_init__(self) -> None:
        _validate_episode(self)

    @property
    def recorded_errors(self) -> tuple[RecordedError, ...]:
        tool_errors = tuple(
            step.error
            for step in sorted(self.tool_steps, key=lambda step: step.index)
            if step.error is not None
        )
        return tool_errors + self.episode_errors

    @property
    def reached_evaluation(self) -> bool:
        return self.evaluator_outcome is not None


def _validate_episode(episode: EpisodeResult) -> None:
    _kind(episode.episode, EpisodeIdentity, "episode")
    _kind(episode.run, RunIdentity, "run")
    _kind(episode.task, LocalTaskRef, "task")
    if episode.episode.run_id != episode.run.run_id:
        raise RecordError("episode run_id must match the run identity")
    _choice(episode.mode, MODES, "mode")
    _integer(episode.execution_seed, "execution_seed")
    if episode.termination_reason not in TERMINATION_REASONS:
        raise RecordError("termination_reason is unknown")
    expected_status = (
        "completed"
        if episode.termination_reason in COMPLETED_TERMINATIONS
        else "failed"
    )
    if episode.status != expected_status:
        raise RecordError("status does not match the termination reason")
    allowed = (
        PLAN_TERMINATIONS if episode.mode == "plan" else EXECUTE_TERMINATIONS
    )
    if episode.termination_reason not in allowed:
        raise RecordError("termination_reason is not valid for the episode mode")
    if episode.role is not None:
        _choice(episode.role, PAIR_ROLES, "role")
    if episode.plan_text is not None:
        _body(episode.plan_text, "plan_text")
    _timestamp(episode.started_at, "started_at")
    _timestamp(episode.ended_at, "ended_at")
    if episode.ended_at < episode.started_at:
        raise RecordError("ended_at precedes started_at")
    model_steps = _instances(episode.model_steps, ModelStep, "model_steps")
    tool_steps = _instances(episode.tool_steps, ToolStep, "tool_steps")
    episode_errors = _instances(
        episode.episode_errors,
        RecordedError,
        "episode_errors",
    )
    if episode.mode == "plan":
        if tool_steps:
            raise RecordError("plan mode records no tool steps")
        if episode.evaluator_outcome is not None:
            raise RecordError("plan mode records no evaluator outcome")
    elif episode.evaluator_outcome is not None:
        _kind(episode.evaluator_outcome, EvaluatorOutcome, "evaluator_outcome")
    if episode.termination_reason == "plan_emitted":
        if episode.plan_text is None:
            raise RecordError("plan_emitted requires plan text")
        if not model_steps:
            raise RecordError("plan_emitted requires a model step")
    _validate_steps(episode, model_steps, tool_steps)
    _validate_errors(episode, tool_steps, episode_errors)


def _validate_steps(
    episode: EpisodeResult,
    model_steps: tuple[ModelStep, ...],
    tool_steps: tuple[ToolStep, ...],
) -> None:
    combined = model_steps + tool_steps
    indexes = [step.index for step in combined]
    if len(indexes) != len(set(indexes)):
        raise RecordError("step indexes must be unique")
    if set(indexes) != set(range(len(indexes))):
        raise RecordError("step indexes must be contiguous from zero")
    previous: datetime | None = None
    for step in sorted(combined, key=lambda item: item.index):
        if step.started_at < episode.started_at or step.started_at > episode.ended_at:
            raise RecordError("step time falls outside the episode")
        if previous is not None and step.started_at < previous:
            raise RecordError("step times move backward")
        previous = step.started_at


def _validate_errors(
    episode: EpisodeResult,
    tool_steps: tuple[ToolStep, ...],
    episode_errors: tuple[RecordedError, ...],
) -> None:
    unrecoverable = [
        step
        for step in tool_steps
        if step.error is not None and not step.error.recoverable
    ]
    if episode.termination_reason == "unrecoverable_tool_error":
        if len(unrecoverable) != 1:
            raise RecordError(
                "unrecoverable_tool_error requires one unrecoverable tool error"
            )
        last_index = max(step.index for step in (*episode.model_steps, *tool_steps))
        if unrecoverable[0].index != last_index:
            raise RecordError("unrecoverable tool error must be the last step")
    elif unrecoverable:
        raise RecordError(
            "an unrecoverable tool error terminates the episode"
        )
    expected_source = _TERMINATING_ERROR.get(episode.termination_reason)
    if expected_source is None:
        if episode_errors:
            raise RecordError("episode errors belong to a failed termination")
        return
    if not episode_errors:
        raise RecordError("failed termination requires a recorded error")
    for error in episode_errors:
        if error.source != expected_source:
            raise RecordError(
                "recorded error does not match the termination reason"
            )


@dataclass(frozen=True)
class PairedResult(Record):
    """Reference and candidate episodes of one pair.

    The episodes share a task, a pair id, a mode, and an execution seed.
    Their episode ids differ. ``reference`` and ``candidate`` are the roles.
    """

    reference: EpisodeResult
    candidate: EpisodeResult

    def __post_init__(self) -> None:
        _kind(self.reference, EpisodeResult, "reference")
        _kind(self.candidate, EpisodeResult, "candidate")
        if self.reference.role != "reference" or self.candidate.role != "candidate":
            raise RecordError("pair roles must be reference and candidate")
        if self.reference.episode.episode_id == self.candidate.episode.episode_id:
            raise RecordError("pair episodes must have distinct episode ids")
        reference_pair = self.reference.episode.pair_id
        candidate_pair = self.candidate.episode.pair_id
        if reference_pair is None or reference_pair != candidate_pair:
            raise RecordError("pair episodes must share a pair id")
        if self.reference.task != self.candidate.task:
            raise RecordError("pair episodes must share a task")
        if self.reference.mode != self.candidate.mode:
            raise RecordError("pair episodes must share a mode")
        if self.reference.execution_seed != self.candidate.execution_seed:
            raise RecordError("pair episodes must share an execution seed")


@dataclass(frozen=True)
class MonitorObservation(Record):
    """One local monitor input.

    The series stays in the episode log. ``requirement_fraction`` is
    recorded only when the evaluator total is positive. ``task_success``
    is the evaluator flag as 0 or 1.
    """

    episode: EpisodeIdentity
    run: RunIdentity
    split: str
    signal: str
    value: float
    observed_at: datetime

    def __post_init__(self) -> None:
        _kind(self.episode, EpisodeIdentity, "episode")
        _kind(self.run, RunIdentity, "run")
        if self.episode.run_id != self.run.run_id:
            raise RecordError("episode run_id must match the run identity")
        _choice(self.split, SPLITS, "split")
        _choice(self.signal, MONITOR_SIGNALS, "signal")
        _timestamp(self.observed_at, "observed_at")
        _validate_signal_value(self.signal, self.value)


def _validate_signal_value(signal: str, value: object) -> None:
    if signal == "task_success":
        if (
            isinstance(value, bool)
            or not isinstance(value, float)
            or value not in (0.0, 1.0)
        ):
            raise RecordError("task_success must be 0 or 1")
        return
    if signal == "requirement_fraction":
        _closed_unit(value, "requirement_fraction")
        return
    number = _nonnegative_float(value, signal)
    if not number.is_integer():
        raise RecordError(f"{signal} must be an integer count")


@dataclass(frozen=True)
class NamedCount(Record):
    """One named non-negative count for a distributional observation."""

    name: str
    count: int

    def __post_init__(self) -> None:
        _line(self.name, "name")
        _nonnegative_int(self.count, "count")


@dataclass(frozen=True)
class ToolSelectionObservation(Record):
    """Local tool-selection counts for one episode.

    Not a ``MONITOR_SIGNALS`` scalar. Counts stay typed and are never
    coerced into ``MonitorObservation.value``.
    """

    episode: EpisodeIdentity
    run: RunIdentity
    split: str
    counts: tuple[NamedCount, ...]
    observed_at: datetime
    completion_index: int | None = None

    def __post_init__(self) -> None:
        _kind(self.episode, EpisodeIdentity, "episode")
        _kind(self.run, RunIdentity, "run")
        if self.episode.run_id != self.run.run_id:
            raise RecordError("episode run_id must match the run identity")
        _choice(self.split, SPLITS, "split")
        items = _instances(self.counts, NamedCount, "counts")
        names = [item.name for item in items]
        if len(names) != len(set(names)):
            raise RecordError("counts repeats a name")
        if names != sorted(names):
            raise RecordError("counts must be sorted by name")
        _timestamp(self.observed_at, "observed_at")
        if self.completion_index is not None:
            _nonnegative_int(self.completion_index, "completion_index")


@dataclass(frozen=True)
class TaskMixObservation(Record):
    """Local task-mix label for one episode.

    Not a ``MONITOR_SIGNALS`` scalar. The label is caller-supplied and is
    never coerced into ``MonitorObservation.value``.
    """

    episode: EpisodeIdentity
    run: RunIdentity
    split: str
    label: str
    observed_at: datetime
    completion_index: int | None = None

    def __post_init__(self) -> None:
        _kind(self.episode, EpisodeIdentity, "episode")
        _kind(self.run, RunIdentity, "run")
        if self.episode.run_id != self.run.run_id:
            raise RecordError("episode run_id must match the run identity")
        _choice(self.split, SPLITS, "split")
        _line(self.label, "label")
        _timestamp(self.observed_at, "observed_at")
        if self.completion_index is not None:
            _nonnegative_int(self.completion_index, "completion_index")


@dataclass(frozen=True)
class StatisticalEvidence(Record):
    """A public aggregate from one statistical method.

    Interval fields are omitted together when the method reports no
    interval. ``threshold`` is the caller-supplied decision threshold
    when the method used one. No level or threshold is filled in.
    """

    method: str
    split: str
    configuration_hash: str
    estimate: float
    sample_size: int
    unit: str
    reference_configuration_hash: str | None = None
    confidence_low: float | None = None
    confidence_high: float | None = None
    confidence_level: float | None = None
    p_value: float | None = None
    threshold: float | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        _token(self.method, "method")
        _choice(self.split, SPLITS, "split")
        _sha256(self.configuration_hash, "configuration_hash")
        _float(self.estimate, "estimate")
        _positive_int(self.sample_size, "sample_size")
        _token(self.unit, "unit")
        if self.reference_configuration_hash is not None:
            _sha256(
                self.reference_configuration_hash,
                "reference_configuration_hash",
            )
        _optional_confidence(
            self.confidence_low,
            self.confidence_high,
            self.confidence_level,
            "statistical evidence",
        )
        if self.p_value is not None:
            _closed_unit(self.p_value, "p_value")
        if self.threshold is not None:
            _float(self.threshold, "threshold")
        if self.seed is not None:
            _integer(self.seed, "seed")


@dataclass(frozen=True)
class LifecycleDecision(Record):
    """A public lifecycle action and the evidence that supports it."""

    tier: str
    decision: str
    split: str
    candidate_configuration_hash: str
    reference_configuration_hash: str
    evidence: tuple[StatisticalEvidence, ...]
    decided_at: datetime

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise RecordError("tier is unknown")
        allowed = TIER_ACTIONS[self.tier]
        if self.decision not in allowed:
            choices = ", ".join(sorted(allowed))
            raise RecordError(f"decision must be one of: {choices}")
        _choice(self.split, SPLITS, "split")
        _sha256(self.candidate_configuration_hash, "candidate_configuration_hash")
        _sha256(self.reference_configuration_hash, "reference_configuration_hash")
        items = _instances(self.evidence, StatisticalEvidence, "evidence")
        if not items:
            raise RecordError("evidence must contain a statistical result")
        allowed_hashes = {
            self.candidate_configuration_hash,
            self.reference_configuration_hash,
        }
        for item in items:
            if item.split != self.split:
                raise RecordError("evidence split must match the decision")
            if item.configuration_hash not in allowed_hashes:
                raise RecordError("evidence configuration hash is outside the decision")
            if (
                item.reference_configuration_hash is not None
                and item.reference_configuration_hash
                != self.reference_configuration_hash
            ):
                raise RecordError("evidence reference hash must match the decision")
        _timestamp(self.decided_at, "decided_at")


@dataclass(frozen=True)
class AggregateRecord(Record):
    """Public counts and one aggregate metric.

    An omitted confidence interval means this aggregate does not report
    one. The interval is not filled in.
    """

    split: str
    configuration_hash: str
    task_set_hash: str
    scenario_count: int
    task_count: int
    episode_count: int
    metric: str
    value: float
    confidence_low: float | None = None
    confidence_high: float | None = None
    confidence_level: float | None = None

    def __post_init__(self) -> None:
        _choice(self.split, SPLITS, "split")
        _sha256(self.configuration_hash, "configuration_hash")
        _sha256(self.task_set_hash, "task_set_hash")
        _nonnegative_int(self.scenario_count, "scenario_count")
        _nonnegative_int(self.task_count, "task_count")
        _nonnegative_int(self.episode_count, "episode_count")
        _token(self.metric, "metric")
        _float(self.value, "value")
        _optional_confidence(
            self.confidence_low,
            self.confidence_high,
            self.confidence_level,
            "aggregate record",
        )


def assert_public_payload(payload: object) -> None:
    """Reject task content and local records in a public payload."""

    if isinstance(payload, Mapping):
        if payload.get("visibility") == LOCAL:
            raise RecordError("local record payload is not a public aggregate")
        for key, value in payload.items():
            if not isinstance(key, str):
                raise RecordError("public payload has a non-string field name")
            if key in PROTECTED_FIELDS:
                raise RecordError(f"public payload contains {key}")
            assert_public_payload(value)
        return
    if isinstance(payload, (list, tuple)):
        for item in payload:
            assert_public_payload(item)


def public_record_dict(record: Record) -> dict[str, object]:
    """Return the public dictionary for an aggregate record."""

    if not isinstance(record, Record) or record.visibility != PUBLIC:
        raise RecordError("local records are not public aggregates")
    payload = record.to_dict()
    assert_public_payload(payload)
    return payload


def _mark(cls: type[Record], visibility: str, record_name: str) -> None:
    cls.visibility = visibility
    cls.record_name = record_name


_mark(TokenLogprob, LOCAL, "token logprob")
_mark(ModelStep, LOCAL, "model step")
_mark(RecordedError, LOCAL, "recorded error")
_mark(ToolStep, LOCAL, "tool step")
_mark(EvaluatorOutcome, LOCAL, "evaluator outcome")
_mark(LocalTaskRef, LOCAL, "local task")
_mark(EpisodeResult, LOCAL, "episode result")
_mark(PairedResult, LOCAL, "paired result")
_mark(MonitorObservation, LOCAL, "monitor observation")
_mark(NamedCount, LOCAL, "named count")
_mark(ToolSelectionObservation, LOCAL, "tool selection observation")
_mark(TaskMixObservation, LOCAL, "task mix observation")
_mark(StatisticalEvidence, PUBLIC, "statistical evidence")
_mark(LifecycleDecision, PUBLIC, "lifecycle decision")
_mark(AggregateRecord, PUBLIC, "aggregate record")
