"""A hard cap on model calls, shared by every runtime one process builds.

A release bounds spend twice: statically, by the number of arrivals and
gate tasks it may send, and here, by counting each generation
(``next_turn``) and each teacher-forced scoring call
(``teacher_force_plan``) before it reaches the model. Once the cap is
spent the next call raises ``ModelCallBudgetExhausted``, a
``RuntimeUnavailable``, so the episode ends as a runtime failure instead
of calling the provider. ``BudgetedAgent`` exposes the wrapped agent
through ``underlying_agents()``, so ``is_live_runtime`` still sees the
real live agent classes.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from llm_behavior_ci.config import RunConfiguration
from llm_behavior_ci.runtime.episode import RuntimeDependencies, RuntimeUnavailable
from llm_behavior_ci.runtime.factory import RuntimeFactory


class ModelCallBudgetExhausted(RuntimeUnavailable):
    """The release's model-call budget is spent; no call was made."""


@dataclass
class ModelCallBudget:
    limit: int
    spent: int = 0
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for label, value in (("limit", self.limit), ("spent", self.spent)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"model-call budget {label} must be a non-negative int")
        if self.limit < 1:
            raise ValueError("model-call budget limit must be positive")

    @property
    def remaining(self) -> int:
        with self._lock:
            return max(self.limit - self.spent, 0)

    def spend(self) -> None:
        with self._lock:
            if self.spent >= self.limit:
                raise ModelCallBudgetExhausted(
                    f"model-call budget of {self.limit} calls is spent"
                )
            self.spent += 1


class BudgetedAgent:
    """Spends one budget unit before each generation or teacher-forced call.

    ``teacher_force_plan`` is exposed only when the wrapped agent has it,
    so capability probes by ``getattr`` see the wrapped agent's answer.
    """

    def __init__(self, agent: Any, budget: ModelCallBudget) -> None:
        if not isinstance(budget, ModelCallBudget):
            raise TypeError("budget must be a ModelCallBudget")
        self._agent = agent
        self._budget = budget

    def underlying_agents(self) -> tuple[object, ...]:
        underlying = getattr(self._agent, "underlying_agents", None)
        return tuple(underlying()) if callable(underlying) else (self._agent,)

    def begin(self, context, config) -> None:
        self._agent.begin(context, config)

    def next_turn(self, *, tool_output: str | None):
        self._budget.spend()
        return self._agent.next_turn(tool_output=tool_output)

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._agent, name)
        if name != "teacher_force_plan" or not callable(attribute):
            return attribute

        def teacher_force_plan(*args: Any, **kwargs: Any) -> Any:
            self._budget.spend()
            return attribute(*args, **kwargs)

        return teacher_force_plan


def budgeted_runtime(runtime: RuntimeDependencies, budget: ModelCallBudget) -> RuntimeDependencies:
    if isinstance(runtime.agent, BudgetedAgent):
        return runtime
    return replace(runtime, agent=BudgetedAgent(runtime.agent, budget))


@dataclass(frozen=True)
class BudgetedRuntimeFactory(RuntimeFactory):
    """Wraps every runtime another factory builds in one shared budget.

    Works for the gate's role-keyed ``RuntimeFactory`` and for the
    service's legacy ``Callable[[RunConfiguration], RuntimeDependencies]``,
    to which it applies the requested mode as the service would.
    """

    inner: Any
    budget: ModelCallBudget

    def __post_init__(self) -> None:
        if not callable(self.inner):
            raise TypeError("inner runtime factory must be callable")
        if not isinstance(self.budget, ModelCallBudget):
            raise TypeError("budget must be a ModelCallBudget")

    def __call__(
        self,
        configuration: RunConfiguration,
        *,
        mode: str | None = None,
        role: str | None = None,
    ) -> RuntimeDependencies:
        if isinstance(self.inner, RuntimeFactory):
            keywords: dict[str, str] = {}
            if mode is not None:
                keywords["mode"] = mode
            if role is not None:
                keywords["role"] = role
            runtime = self.inner(configuration, **keywords)
        else:
            runtime = self.inner(configuration)
            setter = getattr(runtime.agent, "set_mode", None)
            if mode is not None and callable(setter):
                setter(mode)
        return budgeted_runtime(runtime, self.budget)

    def preflight(
        self,
        roles: Mapping[str, RunConfiguration],
        *,
        require_distinct_endpoints: bool = True,
    ) -> None:
        preflight = getattr(self.inner, "preflight", None)
        if callable(preflight):
            preflight(roles, require_distinct_endpoints=require_distinct_endpoints)
