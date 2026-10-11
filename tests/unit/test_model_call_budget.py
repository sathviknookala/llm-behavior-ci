from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest
from pathlib import Path

from llm_behavior_ci.runtime.budget import (
    BudgetedAgent,
    BudgetedRuntimeFactory,
    ModelCallBudget,
    ModelCallBudgetError,
    ModelCallBudgetExhausted,
)
from llm_behavior_ci.runtime.episode import RuntimeDependencies, RuntimeUnavailable
from llm_behavior_ci.runtime.factory import RuntimeFactory
from llm_behavior_ci.storage import EpisodeStore, ModelCallLedger

_SRC = Path(__file__).resolve().parents[2] / "src"


class ProviderError(RuntimeError):
    pass


class CountingAgent:
    def __init__(self, observe=None) -> None:
        self.calls = 0
        self.mode = None
        self.observe = observe

    def begin(self, context, config) -> None:
        del context, config

    def next_turn(self, *, tool_output):
        del tool_output
        self.calls += 1
        if self.observe is not None:
            self.observe()
        return "turn"

    def set_mode(self, mode) -> None:
        self.mode = mode


class FailingAgent(CountingAgent):
    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.error = error

    def next_turn(self, *, tool_output):
        del tool_output
        self.calls += 1
        raise self.error


class ScoringAgent(CountingAgent):
    def __init__(self, fail: bool = False) -> None:
        super().__init__()
        self.fail = fail

    def teacher_force_plan(self, *args, **kwargs):
        del args, kwargs
        self.calls += 1
        if self.fail:
            raise ProviderError("scoring endpoint failed")
        return "scored"


class WrapperAgent(CountingAgent):
    def __init__(self, inner) -> None:
        super().__init__()
        self.inner = inner

    def underlying_agents(self):
        return (self.inner,)


def _runtime(agent) -> RuntimeDependencies:
    return RuntimeDependencies(session_factory=lambda task_id: None, agent=agent, clock=lambda: None)


class BudgetTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.store = Path(self._tmpdir.name) / "episodes.sqlite"

    def budget(self, limit: int, scope: str = "service") -> ModelCallBudget:
        return ModelCallBudget(limit=limit, store_path=self.store, scope=scope)

    def states(self, scope: str = "service") -> dict[str, int]:
        return ModelCallLedger(self.store).states(scope)


class ModelCallBudgetTests(BudgetTestCase):
    def test_a_successful_call_is_reserved_before_the_model_and_settled_after(self) -> None:
        seen = []
        inner = CountingAgent(observe=lambda: seen.append(self.states()))
        agent = BudgetedAgent(inner, self.budget(2))
        self.assertEqual(agent.next_turn(tool_output=None), "turn")
        self.assertEqual(seen, [{"reserved": 1}])
        self.assertEqual(self.states(), {"completed": 1})
        self.assertEqual(self.budget(2).spent, 1)

    def test_the_call_past_the_limit_is_refused_before_the_model(self) -> None:
        inner = CountingAgent()
        agent = BudgetedAgent(inner, self.budget(2))
        agent.next_turn(tool_output=None)
        agent.next_turn(tool_output=None)
        with self.assertRaises(ModelCallBudgetExhausted) as caught:
            agent.next_turn(tool_output=None)
        self.assertIsInstance(caught.exception, RuntimeUnavailable)
        self.assertEqual(inner.calls, 2)
        self.assertEqual(self.states(), {"completed": 2})

    def test_a_failed_provider_call_without_a_model_step_is_charged(self) -> None:
        inner = FailingAgent(ProviderError("provider returned 500"))
        agent = BudgetedAgent(inner, self.budget(3))
        with self.assertRaises(ProviderError):
            agent.next_turn(tool_output=None)
        self.assertEqual(inner.calls, 1)
        self.assertEqual(self.states(), {"failed": 1})
        reader = EpisodeStore(self.store)
        try:
            self.assertEqual(reader.count_model_steps(), 0)
        finally:
            reader.close()
        self.assertEqual(self.budget(3).spent, 1)
        self.assertEqual(self.budget(3).remaining, 2)

    def test_a_restart_after_a_failed_call_does_not_refund_it(self) -> None:
        with self.assertRaises(ProviderError):
            BudgetedAgent(FailingAgent(ProviderError("timeout")), self.budget(2)).next_turn(
                tool_output=None
            )
        BudgetedAgent(CountingAgent(), self.budget(2)).next_turn(tool_output=None)
        restarted = CountingAgent()
        agent = BudgetedAgent(restarted, self.budget(2))
        self.assertEqual(self.budget(2).remaining, 0)
        with self.assertRaises(ModelCallBudgetExhausted):
            agent.next_turn(tool_output=None)
        self.assertEqual(restarted.calls, 0)
        self.assertEqual(self.states(), {"failed": 1, "completed": 1})

    def test_concurrent_attempts_near_the_limit_never_exceed_it(self) -> None:
        BudgetedAgent(CountingAgent(), self.budget(7)).next_turn(tool_output=None)
        BudgetedAgent(CountingAgent(), self.budget(7)).next_turn(tool_output=None)
        workers = 16
        barrier = threading.Barrier(workers)
        agents = [CountingAgent() for _ in range(workers)]
        refused = []

        def attempt(inner: CountingAgent) -> None:
            agent = BudgetedAgent(inner, self.budget(7))
            barrier.wait()
            for _ in range(3):
                try:
                    agent.next_turn(tool_output=None)
                except ModelCallBudgetExhausted:
                    refused.append(1)

        threads = [threading.Thread(target=attempt, args=(inner,)) for inner in agents]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sum(inner.calls for inner in agents), 5)
        self.assertEqual(len(refused), workers * 3 - 5)
        self.assertEqual(self.states(), {"completed": 7})

    def test_concurrent_processes_near_the_limit_never_exceed_it(self) -> None:
        script = textwrap.dedent(
            """
            import sys
            from pathlib import Path
            from llm_behavior_ci.runtime.budget import ModelCallBudget, ModelCallBudgetExhausted
            budget = ModelCallBudget(limit=12, store_path=Path(sys.argv[1]), scope="gate")
            made = 0
            for _ in range(10):
                try:
                    budget.call("generation", lambda: None)
                    made += 1
                except ModelCallBudgetExhausted:
                    pass
            print(made)
            """
        )
        EpisodeStore(self.store).close()
        environment = {**os.environ, "PYTHONPATH": str(_SRC)}
        processes = [
            subprocess.Popen(
                [sys.executable, "-c", script, str(self.store)],
                stdout=subprocess.PIPE,
                text=True,
                env=environment,
            )
            for _ in range(4)
        ]
        made = []
        for process in processes:
            output, _ = process.communicate(timeout=120)
            self.assertEqual(process.returncode, 0)
            made.append(int(output))
        self.assertEqual(sum(made), 12)
        self.assertEqual(self.states("gate"), {"completed": 12})

    def test_teacher_forcing_spends_the_budget_and_is_exposed_only_when_supported(self) -> None:
        budget = self.budget(2, scope="gate")
        self.assertFalse(hasattr(BudgetedAgent(CountingAgent(), budget), "teacher_force_plan"))
        failing = ScoringAgent(fail=True)
        with self.assertRaises(ProviderError):
            BudgetedAgent(failing, budget).teacher_force_plan("plan")
        scoring = ScoringAgent()
        agent = BudgetedAgent(scoring, budget)
        self.assertEqual(agent.teacher_force_plan("plan"), "scored")
        with self.assertRaises(ModelCallBudgetExhausted):
            agent.teacher_force_plan("plan")
        with self.assertRaises(ModelCallBudgetExhausted):
            agent.next_turn(tool_output=None)
        self.assertEqual((failing.calls, scoring.calls), (1, 1))
        self.assertEqual(self.states("gate"), {"failed": 1, "completed": 1})

    def test_an_interrupted_call_stays_charged(self) -> None:
        inner = FailingAgent(KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            BudgetedAgent(inner, self.budget(1)).next_turn(tool_output=None)
        self.assertEqual(self.states(), {"failed": 1})
        self.assertEqual(self.budget(1).remaining, 0)

    def test_a_process_killed_after_reservation_before_response_is_charged_after_restart(
        self,
    ) -> None:
        script = textwrap.dedent(
            """
            import os
            import sys
            from pathlib import Path
            from llm_behavior_ci.runtime.budget import BudgetedAgent, ModelCallBudget

            class Provider:
                def begin(self, context, config):
                    pass

                def next_turn(self, *, tool_output):
                    os._exit(9)

            budget = ModelCallBudget(limit=2, store_path=Path(sys.argv[1]), scope="service")
            BudgetedAgent(Provider(), budget).next_turn(tool_output=None)
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script, str(self.store)],
            env={**os.environ, "PYTHONPATH": str(_SRC)},
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        self.assertEqual(completed.returncode, 9)
        self.assertEqual(self.states(), {"reserved": 1})
        restarted = self.budget(2)
        self.assertEqual(restarted.spent, 1)
        inner = CountingAgent()
        agent = BudgetedAgent(inner, restarted)
        agent.next_turn(tool_output=None)
        with self.assertRaises(ModelCallBudgetExhausted):
            agent.next_turn(tool_output=None)
        self.assertEqual(inner.calls, 1)

    def test_no_model_is_called_when_the_reservation_cannot_be_written(self) -> None:
        inner = CountingAgent()
        missing = Path(self._tmpdir.name) / "missing" / "episodes.sqlite"
        agent = BudgetedAgent(inner, ModelCallBudget(limit=5, store_path=missing, scope="service"))
        with self.assertRaises(ModelCallBudgetError) as caught:
            agent.next_turn(tool_output=None)
        self.assertIsInstance(caught.exception, RuntimeUnavailable)
        self.assertNotIsInstance(caught.exception, ModelCallBudgetExhausted)
        self.assertEqual(inner.calls, 0)

    def test_scopes_and_stores_are_isolated(self) -> None:
        BudgetedAgent(CountingAgent(), self.budget(1, scope="gate")).next_turn(tool_output=None)
        self.assertEqual(self.budget(1, scope="service").remaining, 1)
        other = Path(self._tmpdir.name) / "other.sqlite"
        self.assertEqual(ModelCallBudget(limit=1, store_path=other, scope="gate").remaining, 1)
        self.assertEqual(self.budget(1, scope="gate").remaining, 0)

    def test_the_service_scope_charges_model_steps_written_before_reservations(self) -> None:
        legacy = sqlite3.connect(self.store)
        legacy.execute(
            "CREATE TABLE steps (episode_id TEXT NOT NULL, step_index INTEGER NOT NULL, "
            "kind TEXT NOT NULL, step_json TEXT NOT NULL, PRIMARY KEY (episode_id, step_index))"
        )
        legacy.executemany(
            "INSERT INTO steps VALUES ('episode', ?, ?, '{}')",
            [(0, "model"), (1, "tool"), (2, "model"), (3, "model")],
        )
        legacy.commit()
        legacy.close()
        self.assertEqual(self.budget(10).spent, 3)
        self.assertEqual(self.budget(10, scope="gate").spent, 0)
        BudgetedAgent(CountingAgent(), self.budget(10)).next_turn(tool_output=None)
        self.assertEqual(self.budget(10).spent, 3)

    def test_the_wrapped_agents_stay_visible_to_the_live_runtime_check(self) -> None:
        inner = CountingAgent()
        self.assertEqual(BudgetedAgent(inner, self.budget(1)).underlying_agents(), (inner,))
        wrapped = WrapperAgent(inner)
        self.assertEqual(BudgetedAgent(wrapped, self.budget(1)).underlying_agents(), (inner,))

    def test_one_budget_is_shared_by_every_runtime_and_thread(self) -> None:
        budget = self.budget(50)
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

        factory = BudgetedRuntimeFactory(Factory(), self.budget(1, scope="gate"))
        runtime = factory(None, mode="plan", role="candidate")
        factory.preflight({}, require_distinct_endpoints=False)
        self.assertIsInstance(runtime.agent, BudgetedAgent)
        self.assertEqual(seen, [("plan", "candidate"), ("preflight", False)])

    def test_invalid_budgets_are_refused(self) -> None:
        for kwargs in (
            {"limit": 0, "scope": "service"},
            {"limit": -1, "scope": "service"},
            {"limit": True, "scope": "service"},
            {"limit": 1, "scope": "other"},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ModelCallBudget(store_path=self.store, **kwargs)


if __name__ == "__main__":
    unittest.main()
