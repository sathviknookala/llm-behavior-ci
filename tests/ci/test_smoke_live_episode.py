from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = "scripts/evaluation/smoke_live_episode.py"


def _run(args: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    if env is None:
        merged = {**os.environ, "PYTHONPATH": "src"}
    else:
        merged = {**env, "PYTHONPATH": "src"}
    return subprocess.run(
        [sys.executable, _SCRIPT, *args],
        cwd=_ROOT,
        env=merged,
        capture_output=True,
        text=True,
        check=False,
    )


class SmokeLiveEpisodeCliTests(unittest.TestCase):
    def test_refuses_results_store_without_creating_database(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results_dir = root / "results"
            results_dir.mkdir()
            store = results_dir / "smoke.sqlite"
            env = {key: value for key, value in os.environ.items() if key != "APPWORLD_ROOT"}
            completed = _run(
                [
                    "--configuration",
                    str(root / "missing_configuration.json"),
                    "--task-set",
                    str(root / "missing_task_set.json"),
                    "--task-index",
                    "0",
                    "--base-url",
                    "http://127.0.0.1:9",
                    "--store",
                    str(store),
                ],
                env=env,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("store path must not be under results/", completed.stderr)
            self.assertFalse(store.exists())
            self.assertEqual(list(results_dir.iterdir()), [])

    def test_missing_configuration_exits_2(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = root / "smoke.sqlite"
            env = {key: value for key, value in os.environ.items() if key != "APPWORLD_ROOT"}
            completed = _run(
                [
                    "--configuration",
                    str(root / "missing_configuration.json"),
                    "--task-set",
                    str(root / "missing_task_set.json"),
                    "--task-index",
                    "0",
                    "--base-url",
                    "http://127.0.0.1:9",
                    "--store",
                    str(store),
                ],
                env=env,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("missing file:", completed.stderr)
            self.assertFalse(store.exists())


if __name__ == "__main__":
    unittest.main()
