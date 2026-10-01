import io
import json
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory

from llm_behavior_ci.config import RunConfiguration
from llm_behavior_ci.runtime.aa_capture import main
from llm_behavior_ci.runtime.provenance import (
    ProvenanceError,
    RepositoryState,
    enforce_committed_provenance,
    read_repository_state,
    relevant_dirty_paths,
    repository_root,
    require_committed_provenance,
)
from llm_behavior_ci.tasks.selection import (
    TaskSet,
    canonical_task_set_bytes,
    task_set_hash_from_bytes,
)
from llm_behavior_ci.tasks.streams import StreamSettings

_HEAD = "a" * 40
_OTHER = "b" * 40


class _Completed:
    def __init__(self, returncode: int, stdout: str) -> None:
        self.returncode = returncode
        self.stdout = stdout


def _configuration(git_commit: str = _HEAD) -> RunConfiguration:
    document = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "configs/models/qwen3_4b_production.json"
        ).read_text(encoding="utf-8")
    )
    document["task"] = {
        "appworld_version": "0.1.3.post1",
        "split": "train",
        "selection_rule": "deterministic_sample",
        "selection_seed": 17,
        "task_count": 1,
        "task_set_hash": "c" * 64,
    }
    document["run_seed"] = 17
    document["git_commit"] = git_commit
    document["protocol_hash"] = None
    return RunConfiguration.from_dict(document)


class ProvenanceTests(unittest.TestCase):
    def test_head_mismatch_is_refused(self) -> None:
        state = RepositoryState(head=_OTHER, dirty_paths=())
        with self.assertRaisesRegex(ProvenanceError, "HEAD"):
            require_committed_provenance(_configuration(), state)

    def test_dirty_source_is_refused_and_unrelated_paths_are_not(self) -> None:
        configuration = _configuration()
        clean = RepositoryState(head=_HEAD, dirty_paths=())
        require_committed_provenance(configuration, clean)
        docs = RepositoryState(
            head=_HEAD,
            dirty_paths=("docs/EVAL_PROTOCOL.md", "tests/unit/test_provenance.py"),
        )
        require_committed_provenance(configuration, docs)
        dirty = RepositoryState(
            head=_HEAD,
            dirty_paths=("src/llm_behavior_ci/config.py",),
        )
        with self.assertRaisesRegex(ProvenanceError, "dirty"):
            require_committed_provenance(configuration, dirty)
        self.assertEqual(
            relevant_dirty_paths(
                (
                    "src/llm_behavior_ci/runtime/episode.py",
                    "configs/models/qwen3_4b_spotify_capability.json",
                    "requirements.txt",
                    "data/processed/smoke_c4/run.json",
                )
            ),
            (
                "src/llm_behavior_ci/runtime/episode.py",
                "configs/models/qwen3_4b_spotify_capability.json",
                "requirements.txt",
            ),
        )

    def test_reader_uses_the_injected_runner(self) -> None:
        calls: list[list[str]] = []

        def runner(command: list[str], **_kwargs: object) -> _Completed:
            calls.append(command)
            if command[1] == "rev-parse":
                return _Completed(0, _HEAD + "\n")
            return _Completed(
                0,
                " M src/llm_behavior_ci/config.py\0"
                "R  configs/old.json\0configs/models/qwen3_4b_production.json\0"
                " M docs/EVAL_PROTOCOL.md\0",
            )

        state = read_repository_state(Path("/not/a/repository"), runner=runner)
        self.assertEqual(state.head, _HEAD)
        self.assertEqual(
            state.dirty_paths,
            (
                "src/llm_behavior_ci/config.py",
                "configs/old.json",
                "configs/models/qwen3_4b_production.json",
                "docs/EVAL_PROTOCOL.md",
            ),
        )
        self.assertEqual(calls[0][:2], ["git", "rev-parse"])
        self.assertIn("--porcelain", calls[1])
        with self.assertRaisesRegex(ProvenanceError, "dirty"):
            enforce_committed_provenance(
                _configuration(),
                state=state,
            )

    def test_repository_root_does_not_invoke_git(self) -> None:
        self.assertEqual(repository_root(), Path(__file__).resolve().parents[2])

    def test_capture_refuses_a_mismatched_or_dirty_state_without_git(self) -> None:
        tasks = (("task-a", "scenario-1"),)
        digest = canonical_task_set_bytes(
            appworld_version="0.1.3.post1",
            split="train",
            selection_rule="deterministic_sample",
            selection_seed=17,
            tasks=tasks,
        )
        task_hash = task_set_hash_from_bytes(digest)
        task_set = TaskSet(
            appworld_version="0.1.3.post1",
            split="train",
            selection_rule="deterministic_sample",
            selection_seed=17,
            task_count=1,
            scenario_count=1,
            task_ids=("task-a",),
            scenario_ids=("scenario-1",),
            task_set_hash=task_hash,
        )
        configuration = _configuration()
        document = configuration.to_dict()
        document["task"]["task_set_hash"] = task_hash
        document["task"]["task_count"] = 1
        configuration = RunConfiguration.from_dict(document)
        settings = StreamSettings(
            split="train",
            selection_rule="deterministic_sample",
            selection_seed=17,
            task_set_hash=task_hash,
            stream_seed=3,
            arrival_rate_per_second=1.0,
            concurrency=1,
            with_replacement=False,
            task_mix_rule="uniform",
        )
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "configuration.json"
            task_path = root / "task-set.json"
            stream_path = root / "stream.json"
            output = root / "capture.json"
            config_path.write_text(json.dumps(configuration.to_dict()), encoding="utf-8")
            task_path.write_text(
                json.dumps(
                    {
                        "appworld_version": task_set.appworld_version,
                        "split": task_set.split,
                        "selection_rule": task_set.selection_rule,
                        "selection_seed": task_set.selection_seed,
                        "task_count": task_set.task_count,
                        "scenario_count": task_set.scenario_count,
                        "task_ids": list(task_set.task_ids),
                        "scenario_ids": list(task_set.scenario_ids),
                        "task_set_hash": task_set.task_set_hash,
                    }
                ),
                encoding="utf-8",
            )
            stream_path.write_text(json.dumps(settings.to_dict()), encoding="utf-8")
            argv = [
                "--configuration",
                str(config_path),
                "--task-set",
                str(task_path),
                "--stream-settings",
                str(stream_path),
                "--repetitions",
                "1",
                "--concurrency",
                "1",
                "--modes",
                "execute",
                "--output",
                str(output),
                "--vllm-base-url",
                "http://127.0.0.1:9",
                "--results-root",
                str(root / "results"),
            ]
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                mismatched = main(
                    argv,
                    repository_state=RepositoryState(head=_OTHER, dirty_paths=()),
                )
            self.assertEqual(mismatched, 1)
            self.assertIn("HEAD", stderr.getvalue())
            self.assertFalse(output.exists())
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                dirty = main(
                    argv,
                    repository_state=RepositoryState(
                        head=_HEAD,
                        dirty_paths=("configs/models/qwen3_4b_production.json",),
                    ),
                )
            self.assertEqual(dirty, 1)
            self.assertIn("dirty", stderr.getvalue())
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
