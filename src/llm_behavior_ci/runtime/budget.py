"""A hard, durable cap on model calls, shared by every runtime one store backs.

A release bounds spend twice: statically, by the number of arrivals and
gate tasks it may send, and here, by reserving each generation
(``next_turn``) and each teacher-forced scoring call
(``teacher_force_plan``) in the release store before it reaches the
model (``storage.ModelCallLedger``). The reservation is committed before
the call and stays charged whether the call returns, raises, or never
returns because the process died, so a failed or interrupted provider
attempt is never refunded and a restart reads back every attempt. Once
the limit is reserved the next call raises ``ModelCallBudgetExhausted``,
a ``RuntimeUnavailable``, so the episode ends as a runtime failure
instead of calling the provider. A ledger that cannot be written raises
``ModelCallBudgetError`` and the model is not called either.

The gate and the service keep separate scopes in one store, so neither
spends the other's limit. The service scope also charges model steps
already persisted in the store (see ``ModelCallLedger``).
``BudgetedAgent`` exposes the wrapped agent through
``underlying_agents()``, so ``is_live_runtime`` still sees the real live
agent classes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from llm_behavior_ci.config import RunConfiguration
from llm_behavior_ci.runtime.episode import RuntimeDependencies, RuntimeUnavailable
from llm_behavior_ci.runtime.factory import RuntimeFactory
from llm_behavior_ci.storage import ModelCallLedger, StorageError

BUDGET_SCOPES = ("gate", "service")


class ModelCallBudgetError(RuntimeUnavailable):
    """No reservation could be recorded; no call was made."""


class ModelCallBudgetExhausted(ModelCallBudgetError):
    """The model-call budget is spent; no call was made."""


@dataclass
class ModelCallBudget:
    limit: int
    store_path: Path
    scope: str
    _ledger: ModelCallLedger = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.limit, bool) or not isinstance(self.limit, int) or self.limit < 1:
            raise ValueError("model-call budget limit must be a positive int")
        if self.scope not in BUDGET_SCOPES:
            raise ValueError("model-call budget scope must be one of: " + ", ".join(BUDGET_SCOPES))
        self.store_path = Path(self.store_path)
        self._ledger = ModelCallLedger(self.store_path)

    @property
    def _include_model_steps(self) -> bool:
        return self.scope == "service"

    @property
    def spent(self) -> int:
        try:
            return self._ledger.spent(self.scope, include_model_steps=self._include_model_steps)
        except StorageError as error:
            raise ModelCallBudgetError(str(error)) from error

    @property
    def remaining(self) -> int:
        return max(self.limit - self.spent, 0)

    def reserve(self, kind: str) -> int:
        try:
            call_id = self._ledger.reserve(
                self.scope,
                kind,
                limit=self.limit,
                include_model_steps=self._include_model_steps,
            )
        except StorageError as error:
            raise ModelCallBudgetError(str(error)) from error
        if call_id is None:
            raise ModelCallBudgetExhausted(f"model-call budget of {self.limit} calls is spent")
        return call_id

    def settle(self, call_id: int, outcome: str) -> None:
        try:
            self._ledger.settle(call_id, outcome)
        except StorageError as error:
            raise ModelCallBudgetError(str(error)) from error

    def call(self, kind: str, invoke: Callable[[], Any]) -> Any:
        """Reserve one unit, then call ``invoke``; nothing is called unless the reservation committed."""

        call_id = self.reserve(kind)
        try:
            result = invoke()
        except BaseException:
            try:
                self.settle(call_id, "failed")
            except ModelCallBudgetError:
                pass
            raise
        self.settle(call_id, "completed")
        return result


class BudgetedAgent:
    """Reserves one budget unit before each generation or teacher-forced call.

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
        return self._budget.call(
            "generation", lambda: self._agent.next_turn(tool_output=tool_output)
        )

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._agent, name)
        if name != "teacher_force_plan" or not callable(attribute):
            return attribute

        def teacher_force_plan(*args: Any, **kwargs: Any) -> Any:
            return self._budget.call("teacher_force", lambda: attribute(*args, **kwargs))

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
