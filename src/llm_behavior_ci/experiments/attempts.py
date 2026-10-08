"""Crash-safe accounting for paid plan generations and execute episodes.

The ledger is a section of the benchmark checkpoint, so an attempt's
terminal state is written by the same atomic replace that stores its
result. Every attempt is reserved, then started, each persisted before the
call that can reach a provider. ``reconcile`` turns any attempt left
``reserved`` or ``started`` by a crash into ``interrupted``; that slot
stays consumed and its scope is never dispatched again.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from llm_behavior_ci.runtime.clock import monotonic, wall_now

MODES = ("plan", "execute")
OPEN_STATES = frozenset({"reserved", "started"})
TERMINAL_STATES = frozenset({"completed", "failed", "interrupted"})


class AttemptError(ValueError):
    pass


class AttemptCapExceeded(AttemptError):
    pass


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AttemptError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class AttemptBudget:
    plan_generations: int
    executions: int

    def __post_init__(self) -> None:
        _nonnegative_int(self.plan_generations, "plan_generations")
        _nonnegative_int(self.executions, "executions")

    def to_dict(self) -> dict[str, int]:
        return {"plan": self.plan_generations, "execute": self.executions}


@dataclass(frozen=True)
class AttemptRequest:
    role: str
    mode: str
    configuration_hash: str
    task_id: str


def attempt_state(episode: object) -> str:
    """``failed`` for a recorded runtime failure, otherwise ``completed``."""

    if getattr(episode, "termination_reason", None) == "runtime_error":
        return "failed"
    return "completed"


class AttemptLedger:
    def __init__(
        self,
        section: dict[str, Any],
        *,
        budget: AttemptBudget | None,
        persist: Callable[[], None],
        clock: Callable[[], datetime] = wall_now,
        clock_monotonic: Callable[[], float] = monotonic,
    ) -> None:
        records = section.setdefault("records", [])
        if not isinstance(records, list):
            raise AttemptError("attempt records must be a list")
        stored = section.get("caps")
        supplied = None if budget is None else budget.to_dict()
        if "caps" not in section:
            section["caps"] = supplied
        elif supplied is not None and stored != supplied:
            raise AttemptError("attempt caps differ from the checkpoint's original caps")
        caps = section["caps"]
        if caps is not None and (
            not isinstance(caps, Mapping) or set(caps) != set(MODES)
        ):
            raise AttemptError("checkpoint attempt caps are invalid")
        self._section = section
        self._records: list[dict[str, Any]] = records
        self._persist = persist
        self._clock = clock
        self._monotonic = clock_monotonic
        self._started: dict[int, float] = {}

    @property
    def caps(self) -> Mapping[str, int] | None:
        return self._section["caps"]

    def reconcile(self) -> int:
        """Mark every open attempt interrupted; return how many changed."""

        changed = 0
        for record in self._records:
            if record["state"] in OPEN_STATES:
                record["state"] = "interrupted"
                record["finished_at"] = self._clock().isoformat()
                record["duration_seconds"] = None
                changed += 1
        if changed:
            self._persist()
        return changed

    def consumed(self, mode: str) -> int:
        return sum(1 for record in self._records if record["mode"] == mode)

    def remaining(self, mode: str) -> int | None:
        if mode not in MODES:
            raise AttemptError("mode must be plan or execute")
        if self.caps is None:
            return None
        return int(self.caps[mode]) - self.consumed(mode)

    def require(self, mode: str, count: int) -> None:
        remaining = self.remaining(mode)
        if remaining is not None and remaining < count:
            raise AttemptCapExceeded(
                f"{mode} cap reached: {count} more needed, {remaining} remaining"
            )

    def scope_attempts(self, scope: str) -> list[dict[str, Any]]:
        return [record for record in self._records if record["scope"] == scope]

    def scopes(self, prefix: str) -> set[str]:
        return {
            record["scope"]
            for record in self._records
            if record["scope"].startswith(prefix)
        }

    def reserve(
        self, scope: str, requests: Sequence[AttemptRequest]
    ) -> list[dict[str, Any]]:
        if not requests:
            raise AttemptError("reserve needs at least one attempt")
        if self.scope_attempts(scope):
            raise AttemptError(f"scope {scope} already holds attempts")
        for mode in MODES:
            self.require(mode, sum(1 for request in requests if request.mode == mode))
        reserved_at = self._clock().isoformat()
        attempts = []
        for request in requests:
            if request.mode not in MODES:
                raise AttemptError("mode must be plan or execute")
            attempts.append(
                {
                    "attempt_id": len(self._records) + len(attempts),
                    "scope": scope,
                    "role": request.role,
                    "mode": request.mode,
                    "configuration_hash": request.configuration_hash,
                    "task_id": request.task_id,
                    "state": "reserved",
                    "reserved_at": reserved_at,
                    "started_at": None,
                    "finished_at": None,
                    "duration_seconds": None,
                }
            )
        self._records.extend(attempts)
        self._persist()
        return attempts

    def start(self, attempts: Sequence[dict[str, Any]]) -> None:
        started_at = self._clock().isoformat()
        for attempt in attempts:
            if attempt["state"] != "reserved":
                raise AttemptError("only a reserved attempt can start")
            attempt["state"] = "started"
            attempt["started_at"] = started_at
            self._started[attempt["attempt_id"]] = self._monotonic()
        self._persist()

    def finish(self, attempt: dict[str, Any], state: str) -> None:
        """Set a terminal state; the caller persists it with the result."""

        if state not in {"completed", "failed"}:
            raise AttemptError("finish state must be completed or failed")
        if attempt["state"] != "started":
            raise AttemptError("only a started attempt can finish")
        began = self._started.pop(attempt["attempt_id"], None)
        attempt["state"] = state
        attempt["finished_at"] = self._clock().isoformat()
        attempt["duration_seconds"] = (
            None if began is None else self._monotonic() - began
        )

    def finish_scopes(self, prefix: str, state: str) -> None:
        for record in self._records:
            if record["scope"].startswith(prefix) and record["state"] == "started":
                self.finish(record, state)

    def summary(self) -> dict[str, dict[str, int | None]]:
        modes: dict[str, dict[str, int | None]] = {}
        for mode in MODES:
            records = [record for record in self._records if record["mode"] == mode]
            counts = {
                state: sum(1 for record in records if record["state"] == state)
                for state in ("reserved", "started", "completed", "failed", "interrupted")
            }
            modes[mode] = {
                "cap": None if self.caps is None else int(self.caps[mode]),
                "consumed": len(records),
                "remaining": self.remaining(mode),
                "known_starts": sum(
                    1 for record in records if record["started_at"] is not None
                ),
                "interrupted_before_start": sum(
                    1
                    for record in records
                    if record["state"] == "interrupted" and record["started_at"] is None
                ),
                **counts,
            }
        return modes
