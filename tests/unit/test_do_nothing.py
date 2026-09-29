import unittest
from datetime import datetime, timedelta, timezone

from llm_behavior_ci.config import (
    EpisodeIdentity,
    RunConfiguration,
    RunIdentity,
    new_run_identity,
)
from llm_behavior_ci.records import (
    COMPLETED_TERMINATIONS,
    EXECUTE_TERMINATIONS,
    EpisodeResult,
    EvaluatorOutcome,
    LocalTaskRef,
    ModelStep,
    RecordError,
    TokenLogprob,
)
from llm_behavior_ci.runtime.appworld import (
    EvaluationResult,
    TaskContext,
    ToolResult,
)
from llm_behavior_ci.runtime.episode import (
    RuntimeDependencies,
    run_do_nothing_episode,
)


_START = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)
_END = datetime(2026, 9, 29, 12, 5, tzinfo=timezone.utc)
_HASH = "a" * 64
_TASK_HASH = "c" * 64
_GIT = "d" * 40


def _run() -> RunIdentity:
    return RunIdentity(
        run_id=f"{_HASH}.{'1' * 32}",
        configuration_hash=_HASH,
        task_set_hash=_TASK_HASH,
        protocol_hash=None,
        git_commit=_GIT,
    )


def _model_step() -> ModelStep:
    return ModelStep(
        index=0,
        prompt_text="plan the task",
        output_text="open the app",
        top_k_logprobs=((TokenLogprob(token_id=3, logprob=-0.5, rank=0),),),
        latency_seconds=0.1,
        started_at=_START,
    )


def _do_nothing_episode(**overrides: object) -> EpisodeResult:
    run = _run()
    values: dict[str, object] = {
        "episode": EpisodeIdentity(
            episode_id=f"{run.run_id}.{'2' * 32}",
            run_id=run.run_id,
            pair_id=None,
        ),
        "run": run,
        "task": LocalTaskRef(
            task_id="local-task",
            scenario_id="scenario-1",
            split="dev",
        ),
        "mode": "execute",
        "execution_seed": 7,
        "status": "completed",
        "started_at": _START,
        "ended_at": _END,
        "model_steps": (),
        "tool_steps": (),
        "plan_text": None,
        "evaluator_outcome": EvaluatorOutcome(
            success=False,
            passed_requirements=0,
            total_requirements=4,
            difficulty=1,
        ),
        "termination_reason": "do_nothing",
        "episode_errors": (),
        "role": None,
    }
    values.update(overrides)
    return EpisodeResult(**values)


def _payload() -> dict[str, object]:
    return {
        "model": {
            "model": {
                "repository": "Qwen/Qwen3-4B",
                "revision": "0123456789abcdef0123456789abcdef01234567",
            },
            "tokenizer": {
                "repository": "Qwen/Qwen3-4B",
                "revision": "fedcba9876543210fedcba9876543210fedcba98",
            },
            "quantization": {"method": "none"},
            "vllm_version": "0.30.0",
            "serving": {
                "dtype": "bfloat16",
                "max_model_len": 8192,
                "gpu_memory_utilization": 0.9,
                "max_num_seqs": 16,
                "max_num_batched_tokens": 8192,
                "kv_cache_dtype": "bfloat16",
                "enable_prefix_caching": False,
                "enable_chunked_prefill": False,
                "enforce_eager": False,
                "tensor_parallel_size": 1,
                "max_logprobs": 20,
                "batch_invariant": False,
                "sampler_backend": "native",
            },
        },
        "agent": {
            "smolagents_version": "1.22.0",
            "action_interface": "code",
            "prompt": {
                "prompt_version": "prompt-v1",
                "plan_format_version": "plan-v1",
                "thinking_enabled": False,
            },
            "step_limit": 40,
            "sampling": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 20,
                "min_p": 0.0,
                "seed": 17,
                "max_tokens": 512,
            },
        },
        "task": {
            "appworld_version": "0.1.3.post1",
            "split": "train",
            "selection_rule": "deterministic_sample",
            "selection_seed": 20260926,
            "task_count": 50,
            "task_set_hash": "c" * 64,
        },
        "run_seed": 7,
        "git_commit": "a" * 40,
        "protocol_hash": "e" * 64,
    }


def _config() -> RunConfiguration:
    return RunConfiguration.from_dict(_payload())


def _clock() -> callable:
    current = _START

    def tick() -> datetime:
        nonlocal current
        value = current
        current = current + timedelta(minutes=1)
        return value

    return tick


class FakeSession:
    def __init__(self, task_id: str = "task-1") -> None:
        self.task_id = task_id
        self.execute_count = 0
        self.evaluate_count = 0
        self.close_count = 0
        self.context_count = 0
        self.evaluation = EvaluationResult(
            success=False,
            passed_requirements=0,
            total_requirements=4,
            difficulty=1,
        )
        self.evaluate_error: BaseException | None = None

    def context(self) -> TaskContext:
        self.context_count += 1
        return TaskContext(
            task_id=self.task_id,
            instruction="solve the task",
            api_documentation="calendar docs",
        )

    def execute(self, action: str) -> ToolResult:
        self.execute_count += 1
        raise AssertionError(f"do-nothing must not execute: {action}")

    def evaluate(self) -> EvaluationResult:
        self.evaluate_count += 1
        if self.evaluate_error is not None:
            raise self.evaluate_error
        return self.evaluation

    def close(self) -> None:
        self.close_count += 1


class ForbiddenAgent:
    def begin(self, context: TaskContext, config: RunConfiguration) -> None:
        del context, config
        raise AssertionError("do-nothing must not begin the agent")

    def next_turn(self, *, tool_output: str | None):
        del tool_output
        raise AssertionError("do-nothing must not call the model")


class DoNothingEpisodeTests(unittest.TestCase):
    def test_evaluates_untouched_world_without_model_or_tools(self) -> None:
        config = _config()
        run = new_run_identity(config)
        session = FakeSession("task-42")
        sessions: list[FakeSession] = []

        def factory(task_id: str) -> FakeSession:
            self.assertEqual(task_id, "task-42")
            opened = FakeSession(task_id)
            opened.evaluation = session.evaluation
            sessions.append(opened)
            return opened

        result = run_do_nothing_episode(
            "task-42",
            config,
            run=run,
            runtime=RuntimeDependencies(
                session_factory=factory,
                agent=ForbiddenAgent(),
                clock=_clock(),
            ),
            scenario_id="scenario-7",
        )

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].execute_count, 0)
        self.assertEqual(sessions[0].evaluate_count, 1)
        self.assertEqual(sessions[0].close_count, 1)
        self.assertEqual(sessions[0].context_count, 0)
        self.assertEqual(result.mode, "execute")
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.termination_reason, "do_nothing")
        self.assertEqual(result.model_steps, ())
        self.assertEqual(result.tool_steps, ())
        self.assertEqual(result.episode_errors, ())
        self.assertEqual(result.task.task_id, "task-42")
        self.assertEqual(result.task.scenario_id, "scenario-7")
        self.assertEqual(result.task.split, "train")
        self.assertIsNotNone(result.evaluator_outcome)
        self.assertFalse(result.evaluator_outcome.success)
        self.assertEqual(result.evaluator_outcome.passed_requirements, 0)
        self.assertEqual(result.evaluator_outcome.total_requirements, 4)
        self.assertEqual(result.evaluator_outcome.requirement_fraction, 0.0)
        self.assertIn("do_nothing", COMPLETED_TERMINATIONS)
        self.assertIn("do_nothing", EXECUTE_TERMINATIONS)

    def test_evaluate_failure_stays_unevaluated(self) -> None:
        config = _config()
        run = new_run_identity(config)
        session = FakeSession()
        session.evaluate_error = RuntimeError("evaluator unavailable")

        with self.assertRaises(RuntimeError):
            run_do_nothing_episode(
                "task-1",
                config,
                run=run,
                runtime=RuntimeDependencies(
                    session_factory=lambda task_id: session,
                    agent=ForbiddenAgent(),
                    clock=_clock(),
                ),
            )
        self.assertEqual(session.evaluate_count, 1)
        self.assertEqual(session.execute_count, 0)
        self.assertEqual(session.close_count, 1)


class DoNothingRecordTests(unittest.TestCase):
    def test_do_nothing_record_is_accepted(self) -> None:
        episode = _do_nothing_episode()
        self.assertEqual(episode.termination_reason, "do_nothing")
        self.assertEqual(episode.status, "completed")
        self.assertTrue(episode.reached_evaluation)
        self.assertEqual(episode.model_steps, ())
        self.assertEqual(episode.tool_steps, ())

    def test_do_nothing_without_outcome_is_rejected(self) -> None:
        with self.assertRaises(RecordError):
            _do_nothing_episode(evaluator_outcome=None)

    def test_do_nothing_with_steps_is_rejected(self) -> None:
        with self.assertRaises(RecordError):
            _do_nothing_episode(model_steps=(_model_step(),))


if __name__ == "__main__":
    unittest.main()
