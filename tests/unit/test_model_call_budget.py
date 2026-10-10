from __future__ import annotations

import threading
import unittest

from llm_behavior_ci.runtime.budget import (
    BudgetedAgent,
    BudgetedRuntimeFactory,
    ModelCallBudget,
    ModelCallBudgetExhausted,
)
from llm_behavior_ci.runtime.episode import RuntimeDependencies, RuntimeUnavailable
from llm_behavior_ci.runtime.factory import RuntimeFactory


class CountingAgent:
    def __init__(self) -> None:
        self.calls = 0
        self.mode = None

    def begin(self, context, config) -> None:
        del context, config

    def next_turn(self, *, tool_output):
        del tool_output
        self.calls += 1
        return "turn"

    def set_mode(self, mode) -> None:
        self.mode = mode


class ScoringAgent(CountingAgent):
    def teacher_force_plan(self, *args, **kwargs):
        del args, kwargs
        self.calls += 1
        return "scored"


class WrapperAgent(CountingAgent):
    def __init__(self, inner) -> None:
        super().__init__()
        self.inner = inner

    def underlying_agents(self):
        return (self.inner,)


def _runtime(agent) -> RuntimeDependencies:
    return RuntimeDependencies(session_factory=lambda task_id: None, agent=agent, clock=lambda: None)


class ModelCallBudgetTests(unittest.TestCase):
    def test_the_call_past_the_limit_is_refused_before_the_model(self) -> None:
        inner = CountingAgent()
        agent = BudgetedAgent(inner, ModelCallBudget(limit=2))
        agent.next_turn(tool_output=None)
        agent.next_turn(tool_output=None)
        with self.assertRaises(ModelCallBudgetExhausted) as caught:
            agent.next_turn(tool_output=None)
        self.assertIsInstance(caught.exception, RuntimeUnavailable)
        self.assertEqual(inner.calls, 2)

    def test_teacher_forcing_spends_the_budget_and_is_exposed_only_when_supported(self) -> None:
        budget = ModelCallBudget(limit=1)
        self.assertFalse(hasattr(BudgetedAgent(CountingAgent(), budget), "teacher_force_plan"))
        scoring = ScoringAgent()
        agent = BudgetedAgent(scoring, budget)
        self.assertEqual(agent.teacher_force_plan("plan"), "scored")
        with self.assertRaises(ModelCallBudgetExhausted):
            agent.teacher_force_plan("plan")
        self.assertEqual(scoring.calls, 1)

    def test_the_wrapped_agents_stay_visible_to_the_live_runtime_check(self) -> None:
        inner = CountingAgent()
        self.assertEqual(BudgetedAgent(inner, ModelCallBudget(limit=1)).underlying_agents(), (inner,))
        wrapped = WrapperAgent(inner)
        self.assertEqual(
            BudgetedAgent(wrapped, ModelCallBudget(limit=1)).underlying_agents(), (inner,)
        )

    def test_one_budget_is_shared_by_every_runtime_and_thread(self) -> None:
        budget = ModelCallBudget(limit=50)
        agents = [CountingAgent() for _ in range(4)]
        factory = BudgetedRuntimeFactory(lambda configuration: _runtime(agents.pop()), budget)
        runtimes = [factory(None, mode="execute") for _ in range(4)]

        def spend(runtime):
            for _ in range(30):
                try:
                    runtime.agent.next_turn(tool_output=None)
                except ModelCallBudgetExhausted:
                    return

        threads = [threading.Thread(target=spend, args=(runtime,)) for runtime in runtimes]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(budget.spent, 50)
        self.assertEqual(budget.remaining, 0)
        self.assertEqual(runtimes[0].agent.mode, "execute")

    def test_a_role_keyed_factory_receives_mode_and_role(self) -> None:
        seen = []

        class Factory(RuntimeFactory):
            def __call__(self, configuration, *, mode, role="reference"):
                seen.append((mode, role))
                return _runtime(CountingAgent())

            def preflight(self, roles, *, require_distinct_endpoints=True):
                seen.append(("preflight", require_distinct_endpoints))

        factory = BudgetedRuntimeFactory(Factory(), ModelCallBudget(limit=1))
        runtime = factory(None, mode="plan", role="candidate")
        factory.preflight({}, require_distinct_endpoints=False)
        self.assertIsInstance(runtime.agent, BudgetedAgent)
        self.assertEqual(seen, [("plan", "candidate"), ("preflight", False)])

    def test_invalid_budgets_are_refused(self) -> None:
        for kwargs in ({"limit": 0}, {"limit": -1}, {"limit": True}, {"limit": 1, "spent": -1}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ModelCallBudget(**kwargs)
        self.assertEqual(ModelCallBudget(limit=3, spent=5).remaining, 0)


if __name__ == "__main__":
    unittest.main()
