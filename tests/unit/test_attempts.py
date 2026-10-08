import unittest
from typing import Any
from datetime import datetime, timedelta, timezone

from llm_behavior_ci.experiments.attempts import (
    AttemptBudget,
    AttemptCapExceeded,
    AttemptError,
    AttemptLedger,
    AttemptRequest,
)

_START = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)


def _request(role: str = "reference", mode: str = "execute") -> AttemptRequest:
    return AttemptRequest(role=role, mode=mode, configuration_hash="a" * 64, task_id="task-a")


class _Harness:
    def __init__(self, budget: AttemptBudget | None) -> None:
        self.section: dict[str, Any] = {}
        self.writes = 0
        self.ticks = 0
        self.ledger = AttemptLedger(
            self.section, budget=budget, persist=self.persist, clock=self.clock,
            clock_monotonic=lambda: float(self.ticks),
        )

    def persist(self) -> None:
        self.writes += 1

    def clock(self) -> datetime:
        self.ticks += 1
        return _START + timedelta(seconds=self.ticks)


class AttemptLedgerTests(unittest.TestCase):
    def test_reserve_and_start_each_persist_before_dispatch(self) -> None:
        harness = _Harness(AttemptBudget(plan_generations=0, executions=2))
        attempts = harness.ledger.reserve("s|0", (_request("reference"), _request("candidate")))
        self.assertEqual(harness.writes, 1)
        self.assertEqual({attempt["state"] for attempt in attempts}, {"reserved"})
        harness.ledger.start(attempts)
        self.assertEqual(harness.writes, 2)
        harness.ledger.finish(attempts[0], "completed")
        harness.ledger.finish(attempts[1], "failed")
        self.assertEqual(harness.writes, 2)
        summary = harness.ledger.summary()["execute"]
        self.assertEqual((summary["completed"], summary["failed"], summary["remaining"]), (1, 1, 0))
        self.assertGreater(attempts[0]["duration_seconds"], 0.0)

    def test_a_reservation_beyond_the_cap_records_nothing(self) -> None:
        harness = _Harness(AttemptBudget(plan_generations=1, executions=1))
        with self.assertRaises(AttemptCapExceeded):
            harness.ledger.reserve("s|0", (_request("reference"), _request("candidate")))
        with self.assertRaises(AttemptCapExceeded):
            harness.ledger.reserve("g|0", (_request(mode="plan"), _request("candidate", "plan")))
        self.assertEqual(harness.section["records"], [])
        self.assertEqual(harness.writes, 0)

    def test_a_scope_cannot_be_reserved_twice(self) -> None:
        harness = _Harness(None)
        harness.ledger.reserve("s|0", (_request(),))
        harness.ledger.reconcile()
        with self.assertRaises(AttemptError):
            harness.ledger.reserve("s|0", (_request(),))
        self.assertEqual(harness.ledger.consumed("execute"), 1)

    def test_reconcile_interrupts_open_attempts_once_and_keeps_their_slots(self) -> None:
        harness = _Harness(AttemptBudget(plan_generations=0, executions=3))
        reserved = harness.ledger.reserve("s|0", (_request(),))
        started = harness.ledger.reserve("s|1", (_request(),))
        harness.ledger.start(started)
        self.assertEqual(harness.ledger.reconcile(), 2)
        self.assertEqual(harness.ledger.reconcile(), 0)
        summary = harness.ledger.summary()["execute"]
        self.assertEqual(summary["interrupted"], 2)
        self.assertEqual(summary["known_starts"], 1)
        self.assertEqual(summary["interrupted_before_start"], 1)
        self.assertEqual(summary["remaining"], 1)
        with self.assertRaises(AttemptError):
            harness.ledger.finish(reserved[0], "completed")

    def test_resume_keeps_the_first_caps(self) -> None:
        harness = _Harness(AttemptBudget(plan_generations=120, executions=98))
        AttemptLedger(harness.section, budget=None, persist=harness.persist)
        with self.assertRaises(AttemptError):
            AttemptLedger(
                harness.section,
                budget=AttemptBudget(plan_generations=120, executions=99),
                persist=harness.persist,
            )
        self.assertEqual(harness.section["caps"], {"plan": 120, "execute": 98})


if __name__ == "__main__":
    unittest.main()
