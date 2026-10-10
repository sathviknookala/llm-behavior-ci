from __future__ import annotations

import os
import re
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _ROOT / ".github" / "workflows" / "release-lifecycle.yml"
_STEP = _ROOT / "scripts" / "ci" / "run_live_release.sh"


def _block(lines: list[str], header: str, indent: int) -> list[str]:
    start = lines.index(" " * indent + header)
    block = []
    for line in lines[start + 1 :]:
        if line.strip() and len(line) - len(line.lstrip(" ")) <= indent:
            break
        block.append(line)
    return block


def _keys(block: list[str], indent: int) -> list[str]:
    pattern = re.compile(r"^ {%d}([A-Za-z_-]+):" % indent)
    return [match.group(1) for line in block if (match := pattern.match(line))]


class ReleaseWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _WORKFLOW.read_text(encoding="utf-8")
        self.lines = self.text.splitlines()

    def test_only_a_manual_dispatch_starts_the_workflow(self) -> None:
        self.assertEqual(_keys(_block(self.lines, "on:", 0), 2), ["workflow_dispatch"])

    def test_live_runs_only_on_main_behind_the_protected_environment(self) -> None:
        live = "\n".join(_block(self.lines, "live:", 2))
        for guard in (
            "inputs.mode == 'live'",
            "github.event_name == 'workflow_dispatch'",
            "github.ref == 'refs/heads/main'",
            "environment: release-live",
            "runs-on: [self-hosted, linux, llm-behavior-ci-release]",
            'test "$CONFIRM_LIVE" = "$RELEASE_ID"',
        ):
            self.assertIn(guard, live)
        synthetic = "\n".join(_block(self.lines, "synthetic:", 2))
        self.assertIn("runs-on: ubuntu-latest", synthetic)
        self.assertNotIn("self-hosted", synthetic)

    def test_no_failure_is_masked_and_nothing_private_leaves_the_runner(self) -> None:
        for forbidden in (
            "continue-on-error",
            "upload-artifact",
            "secrets.",
            "|| true",
            "pull_request",
        ):
            self.assertNotIn(forbidden, self.text)
        run_lines = [line for line in self.lines if line.strip().startswith("run:")]
        self.assertTrue(run_lines)
        for line in run_lines:
            self.assertNotIn("${{", line)


class LiveReleaseStepTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.root = Path(self._tmpdir.name)
        (self.root / "releases" / "r2").mkdir(parents=True)
        self.calls = self.root / "calls.txt"
        fake = self.root / "fake_python"
        fake.write_text(
            '#!/usr/bin/env bash\necho "$@" >> "$FAKE_CALLS"\n'
            'echo "private detail" >&2\nexit "$FAKE_CODE"\n',
            encoding="utf-8",
        )
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        self.fake = fake

    def _step(self, code: int, **overrides: str) -> subprocess.CompletedProcess[str]:
        env = {
            "PATH": os.environ["PATH"],
            "LIFECYCLE_RELEASE_ROOT": str(self.root / "releases"),
            "LIFECYCLE_SERVICE_URL": "http://127.0.0.1:9",
            "LIFECYCLE_PYTHON": str(self.fake),
            "RELEASE_ID": "r2",
            "PREVIOUS_RELEASE_ID": "r1",
            "CANARY_ARRIVALS": "4",
            "PRODUCTION_ARRIVALS": "4",
            "FAKE_CALLS": str(self.calls),
            "FAKE_CODE": str(code),
            "GITHUB_STEP_SUMMARY": str(self.root / "step_summary.md"),
            **overrides,
        }
        return subprocess.run(
            ["bash", str(_STEP)],
            cwd=_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_only_a_promotion_passes_the_deployment_job(self) -> None:
        for code in (0, 1, 2, 3, 4, 5):
            with self.subTest(code=code):
                completed = self._step(code)
                self.assertEqual(completed.returncode, code)
                self.assertNotIn("private detail", completed.stdout + completed.stderr)
        summary = (self.root / "step_summary.md").read_text(encoding="utf-8")
        self.assertIn("release r2: exit 0, promoted", summary)
        self.assertIn("release r2: exit 3, not promoted", summary)

    def test_the_step_passes_the_release_lineage_to_the_lifecycle(self) -> None:
        self._step(0)
        call = self.calls.read_text(encoding="utf-8")
        releases = self.root / "releases"
        self.assertIn("scripts/demo/run_three_tier_dev.py live", call)
        self.assertIn(f"--store {releases}/r2/episodes.sqlite", call)
        self.assertIn(f"--release-summary {releases}/r2/release_summary.json", call)
        self.assertIn(f"--previous-release {releases}/r1/release_summary.json", call)

    def test_bad_release_inputs_exit_2_without_running_the_lifecycle(self) -> None:
        (self.root / "outside").mkdir()
        for name, overrides in {
            "path traversal": {"RELEASE_ID": "../outside"},
            "unprovisioned release": {"RELEASE_ID": "r3"},
            "self lineage": {"PREVIOUS_RELEASE_ID": "r2"},
            "non-integer arrivals": {"CANARY_ARRIVALS": "4;true"},
        }.items():
            with self.subTest(case=name):
                completed = self._step(0, **overrides)
                self.assertEqual(completed.returncode, 2)
                self.assertFalse(self.calls.exists())


if __name__ == "__main__":
    unittest.main()
