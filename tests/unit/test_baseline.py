import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from llm_behavior_ci.config import RunConfiguration
from llm_behavior_ci.records import TokenLogprob
from llm_behavior_ci.runtime.agent import AgentTurn
from llm_behavior_ci.runtime.appworld import EvaluationResult, TaskContext, ToolResult
from llm_behavior_ci.runtime.baseline import (
    app_label,
    collect_baseline_outcomes,
    write_local_baselines,
)
from llm_behavior_ci.runtime.episode import EpisodeRejected, RuntimeDependencies
from llm_behavior_ci.tasks.streams import TaskArrival

_START = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)


def _config(split: str = "dev") -> RunConfiguration:
    root = Path(__file__).resolve().parents[2]
    document = json.loads(
        (root / "configs/models/qwen3_4b_production.json").read_text(encoding="utf-8")
    )
    document["task"] = {
        "appworld_version": "0.1.3.post1",
        "split": split,
        "selection_rule": "deterministic_sample",
        "selection_seed": 17,
        "task_count": 1,
        "task_set_hash": "c" * 64,
    }
    document["run_seed"] = 17
    document["git_commit"] = "a" * 40
    document["protocol_hash"] = None
    return RunConfiguration.from_dict(document)


def _clock():
    current = _START

    def tick() -> datetime:
        nonlocal current
        value = current
        current = current + timedelta(seconds=1)
        return value

    return tick


class Session:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.complete_count = 0
        self.evaluate_count = 0

    def context(self) -> TaskContext:
        return TaskContext(
            task_id=self.task_id,
            instruction="instruction",
            api_documentation="docs",
        )

    def required_apps(self) -> tuple[str, ...]:
        return ("gmail", "supervisor", "admin")

    def execute(self, action: str) -> ToolResult:
        raise AssertionError(action)

    def complete_without_work(self) -> None:
        self.complete_count += 1

    def evaluate(self) -> EvaluationResult:
        self.evaluate_count += 1
        return EvaluationResult(
            success=False,
            passed_requirements=0,
            total_requirements=4,
            difficulty=2,
        )

    def close(self) -> None:
        return None


class StopAgent:
    def begin(self, context: TaskContext, config: RunConfiguration) -> None:
        del context, config

    def next_turn(self, *, tool_output: str | None) -> AgentTurn:
        del tool_output
        return AgentTurn(
            prompt_text="prompt",
            output_text="stop",
            top_k_logprobs=((TokenLogprob(token_id=1, logprob=-0.1, rank=0),),),
            latency_seconds=0.0,
            started_at=_clock_holder[0](),
            action=None,
            app_name=None,
            api_name=None,
        )


_clock_holder: list = []


class BaselineRunnerTests(unittest.TestCase):
    def test_app_label_drops_infrastructure_apps(self) -> None:
        self.assertEqual(app_label(("supervisor", "gmail", "admin")), "gmail")
        self.assertEqual(app_label(("venmo", "gmail")), "gmail+venmo")
        self.assertIsNone(app_label(("admin", "api_docs")))
        self.assertIsNone(app_label(None))

    def test_collects_aligned_production_and_do_nothing_rows(self) -> None:
        config = _config()
        clock = _clock()
        _clock_holder.clear()
        _clock_holder.append(clock)
        sessions: list[Session] = []

        def factory(task_id: str) -> Session:
            session = Session(task_id)
            sessions.append(session)
            return session

        arrivals = (
            TaskArrival(
                index=0,
                task_id="task-dev",
                scenario_id="scenario-1",
                scheduled_offset_seconds=0.0,
                stream_seed=17,
            ),
        )
        runtime = RuntimeDependencies(
            session_factory=factory,
            agent=StopAgent(),
            clock=clock,
        )
        rows = collect_baseline_outcomes(
            arrivals,
            config,
            production_runtime=runtime,
            do_nothing_runtime=RuntimeDependencies(
                session_factory=factory,
                agent=StopAgent(),
                clock=clock,
            ),
        )
        self.assertEqual([item.role for item in rows], ["production", "do_nothing"])
        self.assertEqual({item.pair_key for item in rows}, {"task-dev"})
        self.assertEqual({item.scenario_id for item in rows}, {"scenario-1"})
        self.assertEqual({item.app for item in rows}, {"gmail"})
        self.assertEqual({item.difficulty for item in rows}, {2})
        self.assertTrue(all(item.success is False for item in rows))
        self.assertTrue(all(item.requirement_fraction == 0.0 for item in rows))
        self.assertEqual(sessions[1].complete_count, 1)
        self.assertEqual(sessions[0].complete_count, 0)
        self.assertEqual(sessions[0].evaluate_count, 1)
        self.assertEqual(sessions[1].evaluate_count, 1)

    def test_closed_splits_and_results_paths_are_refused(self) -> None:
        config = _config("test_normal")
        with self.assertRaisesRegex(EpisodeRejected, "closed"):
            collect_baseline_outcomes(
                (
                    TaskArrival(
                        index=0,
                        task_id="task-dev",
                        scenario_id="scenario-1",
                        scheduled_offset_seconds=0.0,
                        stream_seed=17,
                    ),
                ),
                config,
                production_runtime=RuntimeDependencies(
                    session_factory=Session,
                    agent=StopAgent(),
                    clock=_clock(),
                ),
                do_nothing_runtime=RuntimeDependencies(
                    session_factory=Session,
                    agent=StopAgent(),
                    clock=_clock(),
                ),
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "results"
            with self.assertRaisesRegex(EpisodeRejected, "public result"):
                write_local_baselines(
                    root / "baselines.json",
                    (),
                    results_root=root,
                )
