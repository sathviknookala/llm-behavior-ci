import importlib.util
import io
import json
import re
import subprocess
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory

_ROOT = Path(__file__).resolve().parents[2]
_BUILDER = _ROOT / "scripts/data/build_spotify_capability_set.py"
_APPWORLD_TASK_ID = re.compile(r"\b[0-9a-f]{7}_[0-9]+\b")
_TRACKED_ROOTS = ("src", "scripts", "configs", "tests", "environments", ".github")


def _builder():
    spec = importlib.util.spec_from_file_location("build_spotify_capability_set", _BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ids(count: int) -> list[str]:
    return [f"task_{index:04d}" for index in range(count)]


class SpotifyCapabilitySetBuilderTest(unittest.TestCase):
    def test_bare_command_exits_2(self) -> None:
        self.assertEqual(_builder().main([]), 2)

    def test_refuses_tracked_task_id_and_manifest_paths(self) -> None:
        builder = _builder()
        tracked = "configs/tasks/train_spotify_capability.json"
        for args in (
            ["--task-ids", tracked, "--manifest", "data/processed/unused.json"],
            ["--task-ids", "data/processed/unused.json", "--manifest", tracked],
        ):
            with self.subTest(args=args):
                with redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        builder.main(args)
                self.assertEqual(caught.exception.code, 2)

    def test_task_id_list_must_be_twenty_unique_strings(self) -> None:
        builder = _builder()
        cases = {
            "valid": (_ids(20), None),
            "short": (_ids(19), 2),
            "duplicate": (_ids(19) + ["task_0000"], 2),
            "not a list": ({"task_ids": _ids(20)}, 2),
            "empty id": (_ids(19) + [""], 2),
        }
        with TemporaryDirectory() as directory:
            for label, (payload, code) in cases.items():
                with self.subTest(case=label):
                    path = Path(directory) / f"{label}.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    if code is None:
                        self.assertEqual(builder._load_task_ids(path), tuple(payload))
                        continue
                    with redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as caught:
                            builder._load_task_ids(path)
                    self.assertEqual(caught.exception.code, code)


class TrackedTaskIdLiteralTest(unittest.TestCase):
    def test_tracked_code_and_configs_hold_no_appworld_task_ids(self) -> None:
        listed = subprocess.run(
            ["git", "ls-files", "--", *_TRACKED_ROOTS],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        hits = []
        for relative in listed:
            path = _ROOT / relative
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if _APPWORLD_TASK_ID.search(line):
                    hits.append(f"{relative}:{number}")
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
