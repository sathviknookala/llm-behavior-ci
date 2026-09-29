import json
import math
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from llm_behavior_ci.config import (
    RunConfiguration,
    StreamSettings,
    TaskConfiguration,
    new_run_identity,
)
from llm_behavior_ci.export import AggregateResults, ExportError, export_public_results
from llm_behavior_ci.records import (
    PROTECTED_FIELDS,
    AggregateRecord,
    EpisodeResult,
    StatisticalEvidence,
    TokenLogprob,
)
from llm_behavior_ci.runtime.agent import AgentTurn
from llm_behavior_ci.runtime.appworld import EvaluationResult, TaskContext, ToolResult
from llm_behavior_ci.runtime.episode import RuntimeDependencies, run_episode
from llm_behavior_ci.stats.bootstrap import paired_bootstrap
from llm_behavior_ci.stats.kl import next_token_kl
from llm_behavior_ci.stats.ks import ks_two_sample
from llm_behavior_ci.storage import EpisodeStore
from llm_behavior_ci.tasks.catalog import CatalogEntry, TaskCatalog
from llm_behavior_ci.tasks.selection import select_task_set, verify_task_set
from llm_behavior_ci.tasks.streams import generate_stream

_START = datetime(2026, 9, 27, 15, 0, tzinfo=timezone.utc)


def _catalog() -> TaskCatalog:
    return TaskCatalog(
        appworld_version="0.1.3.post1",
        entries=(
            CatalogEntry("task-a", "scenario-1", "train", 1),
            CatalogEntry("task-b", "scenario-1", "train", 2),
            CatalogEntry("task-c", "scenario-2", "dev", 1),
        ),
    )


def _task_set():
    return select_task_set(
        _catalog(),
        split="train",
        selection_rule="deterministic_sample",
        seed=7,
        count=2,
    )


def _settings(task_set) -> StreamSettings:
    return StreamSettings(
        split=task_set.split,
        selection_rule=task_set.selection_rule,
        selection_seed=task_set.selection_seed,
        task_set_hash=task_set.task_set_hash,
        stream_seed=3,
        arrival_rate_per_second=2.0,
        concurrency=1,
        with_replacement=False,
        task_mix_rule="uniform",
    )


def _configuration(task_set) -> RunConfiguration:
    return RunConfiguration.from_dict(
        {
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
                "step_limit": 4,
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
                "appworld_version": task_set.appworld_version,
                "split": task_set.split,
                "selection_rule": task_set.selection_rule,
                "selection_seed": task_set.selection_seed,
                "task_count": task_set.task_count,
                "task_set_hash": task_set.task_set_hash,
            },
            "run_seed": 7,
            "git_commit": "a" * 40,
            "protocol_hash": "e" * 64,
        }
    )


def _clock():
    current = _START

    def tick() -> datetime:
        nonlocal current
        value = current
        current = current + timedelta(minutes=1)
        return value

    return tick


def _turn(output_text: str, action: str | None, started_at: datetime) -> AgentTurn:
    return AgentTurn(
        prompt_text="plan the next action",
        output_text=output_text,
        top_k_logprobs=((TokenLogprob(token_id=7, logprob=-0.5, rank=0),),),
        latency_seconds=0.1,
        started_at=started_at,
        action=action,
        app_name=None,
        api_name=None,
    )


class _Session:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.execute_count = 0
        self.evaluate_count = 0
        self.close_count = 0

    def context(self) -> TaskContext:
        return TaskContext(
            task_id=self.task_id,
            instruction="solve the task",
            api_documentation="calendar docs",
        )

    def execute(self, action: str) -> ToolResult:
        del action
        self.execute_count += 1
        return ToolResult(
            output_text="ok",
            error_message=None,
            recoverable=False,
            app_name=None,
            api_name=None,
        )

    def evaluate(self) -> EvaluationResult:
        self.evaluate_count += 1
        return EvaluationResult(
            success=True,
            passed_requirements=1,
            total_requirements=1,
            difficulty=1,
        )

    def close(self) -> None:
        self.close_count += 1


class _Agent:
    def __init__(self, clock) -> None:
        self._clock = clock
        self._calls = 0

    def begin(self, context: TaskContext, config: RunConfiguration) -> None:
        del context, config

    def next_turn(self, *, tool_output: str | None) -> AgentTurn:
        del tool_output
        self._calls += 1
        started_at = self._clock()
        if self._calls == 1:
            return _turn("calendar.lookup()", "calendar.lookup()", started_at)
        return _turn("STOP", None, started_at)


def _protected_keys(payload: object) -> list[str]:
    found: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in PROTECTED_FIELDS:
                found.append(key)
            found.extend(_protected_keys(value))
    elif isinstance(payload, list):
        for item in payload:
            found.extend(_protected_keys(item))
    return found


class WorkstreamContractTests(unittest.TestCase):
    def test_catalog_verifies_and_replays_a_stream(self) -> None:
        task_set = _task_set()
        verify_task_set(
            TaskConfiguration(
                appworld_version=task_set.appworld_version,
                split=task_set.split,
                selection_rule=task_set.selection_rule,
                selection_seed=task_set.selection_seed,
                task_count=task_set.task_count,
                task_set_hash=task_set.task_set_hash,
            ),
            task_set,
        )
        settings = _settings(task_set)
        first = list(generate_stream(task_set, settings))
        second = list(generate_stream(task_set, settings))
        self.assertEqual(first, second)
        self.assertEqual(len(first), task_set.task_count)
        self.assertEqual({item.task_id for item in first}, set(task_set.task_ids))
        self.assertEqual(first[0].scheduled_offset_seconds, 0.0)
        self.assertEqual(first[1].scheduled_offset_seconds, 0.5)

    def test_episode_callback_round_trips_through_the_store(self) -> None:
        task_set = _task_set()
        arrival = next(generate_stream(task_set, _settings(task_set)))
        config = _configuration(task_set)
        clock = _clock()
        session = _Session(arrival.task_id)
        captured: list[object] = []
        result = run_episode(
            arrival.task_id,
            config,
            "execute",
            run=new_run_identity(config),
            runtime=RuntimeDependencies(
                session_factory=lambda task_id: session,
                agent=_Agent(clock),
                clock=clock,
            ),
            on_step=captured.append,
        )
        self.assertEqual(result.task.task_id, arrival.task_id)
        self.assertEqual(session.execute_count, 1)
        self.assertEqual(session.evaluate_count, 1)
        self.assertEqual(session.close_count, 1)
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(captured), 3)
        with tempfile.TemporaryDirectory() as temporary:
            store = EpisodeStore(Path(temporary) / "episodes.sqlite")
            store.start_episode(result.episode, result.run, result.task.task_id)
            for step in captured:
                store.append_step(result.episode.episode_id, step)
            store.finish_episode(result)
            loaded = store.load_episode(result.episode.episode_id)
            self.assertEqual(loaded.to_dict(), result.to_dict())
            store.close()

    def test_statistics_accept_numeric_inputs_without_runtime_imports(self) -> None:
        bootstrap = paired_bootstrap(
            [1.0, 0.0],
            [1.0, 1.0],
            confidence_level=0.9,
            resamples=20,
            seed=1,
        )
        self.assertAlmostEqual(bootstrap.mean_difference, -0.5)
        kl = next_token_kl(
            ((math.log(0.5), math.log(0.5)),),
            ((math.log(0.5), math.log(0.5)),),
        )
        self.assertEqual(kl.mean_kl_nats, 0.0)
        ks = ks_two_sample((0.0, 1.0), (0.0, 1.0))
        self.assertEqual(ks.statistic, 0.0)
        root = Path(__file__).parents[2] / "src" / "llm_behavior_ci" / "stats"
        for path in root.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("appworld", text)
            self.assertNotIn("fastapi", text)
            self.assertNotIn("llm_behavior_ci.runtime", text)

    def test_public_export_keeps_episode_fields_out(self) -> None:
        task_set = _task_set()
        run = new_run_identity(_configuration(task_set))
        aggregate = AggregateResults(
            aggregates=(
                AggregateRecord(
                    split="train",
                    configuration_hash=run.configuration_hash,
                    task_set_hash=task_set.task_set_hash,
                    scenario_count=task_set.scenario_count,
                    task_count=task_set.task_count,
                    episode_count=1,
                    metric="task_success",
                    value=1.0,
                ),
            ),
            evidence=(
                StatisticalEvidence(
                    method="paired_bootstrap",
                    split="train",
                    configuration_hash=run.configuration_hash,
                    estimate=0.0,
                    sample_size=2,
                    unit="task_success",
                ),
            ),
            decisions=(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "public.json"
            export_public_results(aggregate, output_path=output)
            payload = json.loads(output.read_text(encoding="utf-8"))
        encoded = json.dumps(payload)
        self.assertEqual(_protected_keys(payload), [])
        self.assertIn(task_set.task_set_hash, encoded)
        self.assertNotIn("task-a", encoded)
        self.assertNotIn("task-b", encoded)
        with self.assertRaises(ExportError):
            export_public_results(EpisodeResult, output_path=Path("unused.json"))
