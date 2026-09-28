import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from llm_behavior_ci.config import EpisodeIdentity, RunIdentity
from llm_behavior_ci.records import (
    EpisodeResult,
    EvaluatorOutcome,
    LocalTaskRef,
    ModelStep,
    PairedResult,
    TokenLogprob,
    ToolStep,
)
from llm_behavior_ci.storage import EpisodeStore, StorageError

_HASH_A = "a" * 64
_HASH_B = "b" * 64
_TASK_HASH = "c" * 64
_GIT = "d" * 40
_PAIR = "e" * 32
_START = datetime(2026, 9, 27, 15, 0, tzinfo=timezone.utc)
_MID = datetime(2026, 9, 27, 15, 1, tzinfo=timezone.utc)
_LATER = datetime(2026, 9, 27, 15, 2, tzinfo=timezone.utc)


def _run(config: str = _HASH_A, token: str = "1" * 32) -> RunIdentity:
    return RunIdentity(
        run_id=f"{config}.{token}",
        configuration_hash=config,
        task_set_hash=_TASK_HASH,
        protocol_hash=None,
        git_commit=_GIT,
    )


def _identity(
    run: RunIdentity,
    token: str = "2" * 32,
    pair_id: str | None = None,
) -> EpisodeIdentity:
    return EpisodeIdentity(
        episode_id=f"{run.run_id}.{token}",
        run_id=run.run_id,
        pair_id=pair_id,
    )


def _task() -> LocalTaskRef:
    return LocalTaskRef(task_id="task-a", scenario_id="scenario-1", split="dev")


def _model_step(index: int = 0, started_at: datetime = _MID) -> ModelStep:
    return ModelStep(
        index=index,
        prompt_text="plan the task",
        output_text="open the app",
        top_k_logprobs=(
            (
                TokenLogprob(token_id=3, logprob=-0.5, rank=0),
                TokenLogprob(token_id=4, logprob=float("-inf"), rank=1),
            ),
        ),
        latency_seconds=0.1,
        started_at=started_at,
    )


def _tool_step(index: int, started_at: datetime = _LATER) -> ToolStep:
    return ToolStep(
        index=index,
        action="app.lookup()",
        app_name="calendar",
        api_name="lookup",
        output_text="",
        error=None,
        latency_seconds=0.2,
        started_at=started_at,
    )


def _episode(
    run: RunIdentity,
    identity: EpisodeIdentity,
    *,
    model_steps: tuple[ModelStep, ...] = (),
    tool_steps: tuple[ToolStep, ...] = (),
    mode: str = "execute",
    termination_reason: str = "agent_stopped",
    status: str = "completed",
    plan_text: str | None = None,
    evaluator_outcome: EvaluatorOutcome | None = None,
    role: str | None = None,
    execution_seed: int = 7,
) -> EpisodeResult:
    if model_steps == ():
        model_steps = (_model_step(),)
    if mode == "execute" and evaluator_outcome is None and termination_reason == "agent_stopped":
        evaluator_outcome = EvaluatorOutcome(
            success=True,
            passed_requirements=1,
            total_requirements=1,
            difficulty=1,
        )
    return EpisodeResult(
        episode=identity,
        run=run,
        task=_task(),
        mode=mode,
        execution_seed=execution_seed,
        status=status,
        started_at=_START,
        ended_at=_LATER,
        model_steps=model_steps,
        tool_steps=tool_steps,
        plan_text=plan_text,
        evaluator_outcome=evaluator_outcome,
        termination_reason=termination_reason,
        episode_errors=(),
        role=role,
    )


def _plan_episode(
    run: RunIdentity,
    token: str,
    role: str,
) -> EpisodeResult:
    return _episode(
        run,
        _identity(run, token, pair_id=_PAIR),
        mode="plan",
        termination_reason="plan_emitted",
        status="completed",
        plan_text="1. open the calendar",
        evaluator_outcome=None,
        tool_steps=(),
        role=role,
        execution_seed=11,
    )


class EpisodeStoreTests(unittest.TestCase):
    def test_round_trip_finished_episode(self) -> None:
        run = _run()
        identity = _identity(run)
        step = _model_step()
        episode = _episode(run, identity, model_steps=(step,), tool_steps=())
        with TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "episodes.sqlite")
            store.start_episode(identity, run, "task-a")
            store.append_step(identity.episode_id, step)
            store.finish_episode(episode)
            loaded = store.load_episode(identity.episode_id)
            self.assertEqual(loaded.to_dict(), episode.to_dict())
            store.close()

    def test_partial_recovery_after_reopen(self) -> None:
        run = _run()
        identity = _identity(run)
        first = _model_step(0, _START)
        second = _tool_step(1, _MID)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.sqlite"
            store = EpisodeStore(path)
            store.start_episode(identity, run, "task-a")
            store.append_step(identity.episode_id, first)
            store.append_step(identity.episode_id, second)
            store.close()
            reopened = EpisodeStore(path)
            opened = reopened.load_open_episode(identity.episode_id)
            self.assertEqual(opened.identity, identity)
            self.assertEqual(opened.run, run)
            self.assertEqual(opened.task_id, "task-a")
            self.assertEqual(opened.steps, (first, second))
            with self.assertRaises(StorageError):
                reopened.load_episode(identity.episode_id)
            reopened.close()

    def test_step_index_must_be_contiguous(self) -> None:
        run = _run()
        identity = _identity(run)
        with TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "episodes.sqlite")
            store.start_episode(identity, run, "task-a")
            with self.assertRaises(StorageError):
                store.append_step(identity.episode_id, _model_step(1, _MID))
            store.append_step(identity.episode_id, _model_step(0, _START))
            store.append_step(identity.episode_id, _tool_step(1, _MID))
            store.close()

    def test_second_finish_is_rejected(self) -> None:
        run = _run()
        identity = _identity(run)
        step = _model_step()
        episode = _episode(run, identity, model_steps=(step,), tool_steps=())
        with TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "episodes.sqlite")
            store.start_episode(identity, run, "task-a")
            store.append_step(identity.episode_id, step)
            store.finish_episode(episode)
            with self.assertRaises(StorageError):
                store.finish_episode(episode)
            store.close()

    def test_second_start_is_rejected(self) -> None:
        run = _run()
        identity = _identity(run)
        with TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "episodes.sqlite")
            store.start_episode(identity, run, "task-a")
            with self.assertRaises(StorageError):
                store.start_episode(identity, run, "task-a")
            store.close()

    def test_pair_round_trip(self) -> None:
        reference_run = _run(_HASH_A, "1" * 32)
        candidate_run = _run(_HASH_B, "3" * 32)
        pair = PairedResult(
            reference=_plan_episode(reference_run, "4" * 32, "reference"),
            candidate=_plan_episode(candidate_run, "5" * 32, "candidate"),
        )
        with TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "episodes.sqlite")
            store.append_pair(pair)
            loaded = store.load_pair(_PAIR)
            self.assertEqual(loaded.to_dict(), pair.to_dict())
            store.close()

    def test_duplicate_pair_is_rejected(self) -> None:
        reference_run = _run(_HASH_A, "1" * 32)
        candidate_run = _run(_HASH_B, "3" * 32)
        pair = PairedResult(
            reference=_plan_episode(reference_run, "4" * 32, "reference"),
            candidate=_plan_episode(candidate_run, "5" * 32, "candidate"),
        )
        with TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "episodes.sqlite")
            store.append_pair(pair)
            with self.assertRaises(StorageError):
                store.append_pair(pair)
            store.close()

    def test_write_after_close_raises(self) -> None:
        run = _run()
        identity = _identity(run)
        with TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "episodes.sqlite")
            store.start_episode(identity, run, "task-a")
            store.close()
            with self.assertRaises(StorageError):
                store.append_step(identity.episode_id, _model_step())

    def test_missing_parent_directory_raises(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "missing" / "episodes.sqlite"
            with self.assertRaises(StorageError):
                EpisodeStore(path)
