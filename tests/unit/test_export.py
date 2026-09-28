import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from llm_behavior_ci.config import EpisodeIdentity, RunIdentity
from llm_behavior_ci.export import AggregateResults, ExportError, export_public_results
from llm_behavior_ci.records import (
    PROTECTED_FIELDS,
    AggregateRecord,
    EpisodeResult,
    EvaluatorOutcome,
    LocalTaskRef,
    ModelStep,
    StatisticalEvidence,
    TokenLogprob,
)

_HASH = "a" * 64
_TASK_HASH = "c" * 64
_GIT = "d" * 40
_START = datetime(2026, 9, 27, 15, 0, tzinfo=timezone.utc)
_MID = datetime(2026, 9, 27, 15, 1, tzinfo=timezone.utc)
_LATER = datetime(2026, 9, 27, 15, 2, tzinfo=timezone.utc)


def _keys(payload: object) -> set[str]:
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            found.add(key)
            found.update(_keys(value))
    elif isinstance(payload, list):
        for item in payload:
            found.update(_keys(item))
    return found


def _aggregate() -> AggregateRecord:
    return AggregateRecord(
        split="dev",
        configuration_hash=_HASH,
        task_set_hash=_TASK_HASH,
        scenario_count=1,
        task_count=1,
        episode_count=1,
        metric="task_success",
        value=0.0,
    )


def _evidence() -> StatisticalEvidence:
    return StatisticalEvidence(
        method="paired_bootstrap",
        split="dev",
        configuration_hash=_HASH,
        estimate=0.0,
        sample_size=10,
        unit="task_success",
    )


def _results() -> AggregateResults:
    return AggregateResults(
        aggregates=(_aggregate(),),
        evidence=(_evidence(),),
        decisions=(),
    )


def _episode() -> EpisodeResult:
    run = RunIdentity(
        run_id=f"{_HASH}.{'1' * 32}",
        configuration_hash=_HASH,
        task_set_hash=_TASK_HASH,
        protocol_hash=None,
        git_commit=_GIT,
    )
    return EpisodeResult(
        episode=EpisodeIdentity(
            episode_id=f"{run.run_id}.{'2' * 32}",
            run_id=run.run_id,
        ),
        run=run,
        task=LocalTaskRef(task_id="task-a", scenario_id="scenario-1", split="dev"),
        mode="execute",
        execution_seed=7,
        status="completed",
        started_at=_START,
        ended_at=_LATER,
        model_steps=(
            ModelStep(
                index=0,
                prompt_text="plan the task",
                output_text="open the app",
                top_k_logprobs=(
                    (
                        TokenLogprob(token_id=3, logprob=-0.5, rank=0),
                        TokenLogprob(token_id=4, logprob=float("-inf"), rank=1),
                    ),
                ),
                latency_seconds=0.1,
                started_at=_MID,
            ),
        ),
        tool_steps=(),
        plan_text=None,
        evaluator_outcome=EvaluatorOutcome(
            success=True,
            passed_requirements=1,
            total_requirements=1,
            difficulty=1,
        ),
        termination_reason="agent_stopped",
        episode_errors=(),
        role=None,
    )


class ExportTests(unittest.TestCase):
    def test_exports_valid_aggregates(self) -> None:
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "public.json"
            export_public_results(_results(), output_path=destination)
            self.assertTrue(destination.is_file())
            decoded = json.loads(destination.read_text(encoding="utf-8"))
            self.assertTrue(PROTECTED_FIELDS.isdisjoint(_keys(decoded)))

    def test_replace_is_atomic(self) -> None:
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "public.json"
            output_path.write_bytes(b"keep\n")
            missing = Path(directory) / "absent" / "public.json"
            with self.assertRaises(ExportError):
                export_public_results(_results(), output_path=missing)
            self.assertEqual(output_path.read_bytes(), b"keep\n")
            export_public_results(_results(), output_path=output_path)
            self.assertFalse((Path(directory) / "public.json.tmp").exists())

    def test_preexisting_file_survives_a_failed_replace(self) -> None:
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "public.json"
            output_path.mkdir()
            with self.assertRaises(ExportError):
                export_public_results(_results(), output_path=output_path)
            self.assertTrue(output_path.is_dir())

    def test_rejects_an_episode_result(self) -> None:
        episode = _episode()
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "public.json"
            with self.assertRaises(ExportError):
                export_public_results(episode, output_path=destination)
        with self.assertRaises(ExportError):
            AggregateResults(aggregates=(episode,), evidence=(), decisions=())

    def test_rejects_a_dict_payload(self) -> None:
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "public.json"
            with self.assertRaises(ExportError):
                export_public_results(
                    {"task_id": "task-a"},
                    output_path=destination,
                )
