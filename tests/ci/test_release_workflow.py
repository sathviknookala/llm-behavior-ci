from __future__ import annotations

import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS = _ROOT / ".github" / "workflows"
_WORKFLOW = _WORKFLOWS / "release-lifecycle.yml"
_STEP = _ROOT / "scripts" / "ci" / "run_live_release.sh"
_ACTIVATE = _ROOT / "scripts" / "ci" / "activate_release.sh"
_RESOLVE = _ROOT / "scripts" / "ci" / "resolve_release_id.sh"
_UNIT = _ROOT / "scripts" / "ci" / "llm-behavior-ci-release.service"


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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


def _executable(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


class ReleaseWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _WORKFLOW.read_text(encoding="utf-8")
        self.lines = self.text.splitlines()
        self.jobs = _block(self.lines, "jobs:", 0)

    def _job(self, name: str) -> str:
        return "\n".join(_block(self.lines, f"{name}:", 2))

    def test_a_push_to_main_or_a_manual_dispatch_starts_the_workflow(self) -> None:
        on = _block(self.lines, "on:", 0)
        self.assertEqual(_keys(on, 2), ["push", "workflow_dispatch"])
        self.assertEqual(
            [line.strip() for line in _block(on, "push:", 2)], ["branches:", "- main"]
        )
        self.assertEqual(_keys(self.jobs, 2), ["cpu", "synthetic", "provision", "live"])

    def test_cpu_verification_runs_every_suite_and_all_four_scenarios(self) -> None:
        cpu = self._job("cpu")
        for suite in ("unit", "ci", "integration", "validity"):
            self.assertIn(f"discover -s tests/{suite}", cpu)
        demo = _load("run_three_tier_dev_workflow", "scripts/demo/run_three_tier_dev.py")
        synthetic = self._job("synthetic")
        matrix = [
            line.strip()[2:]
            for line in _block(synthetic.splitlines(), "scenario:", 8)
            if line.strip().startswith("- ")
        ]
        self.assertEqual(matrix, list(demo.SCENARIOS))
        self.assertEqual(
            matrix, ["blocked", "healthy", "regression", "production_regression"]
        )
        self.assertIn("scripts/ci/check_synthetic.py", synthetic)
        self.assertIn('--exit-code "$code"', synthetic)
        for job in (cpu, synthetic):
            self.assertIn("runs-on: ubuntu-latest", job)
            self.assertNotIn("environment:", job)

    def test_the_paid_stage_runs_only_from_main_behind_the_protected_environment(self) -> None:
        provision = self._job("provision")
        live = self._job("live")
        for job in (provision, live):
            for guard in (
                "github.repository == 'sathviknookala/llm-behavior-ci'",
                "github.ref == 'refs/heads/main'",
                "github.event_name == 'push' && vars.LIFECYCLE_RELEASE_ENABLED == 'true'",
                "github.event_name == 'workflow_dispatch' && inputs.mode == 'release'",
                "runs-on: [self-hosted, linux, llm-behavior-ci-release]",
                "ref: ${{ github.sha }}",
            ):
                self.assertIn(guard, job)
        self.assertIn("needs: [cpu, synthetic]", provision)
        self.assertIn("environment: release-provision", provision)
        self.assertIn("needs: provision", live)
        self.assertIn("environment: release-live", live)
        self.assertNotIn("release-live", provision)
        order = [
            live.index(step)
            for step in (
                "release_ctl.py check-deployed",
                "scripts/ci/activate_release.sh",
                "scripts/ci/run_live_release.sh",
            )
        ]
        self.assertEqual(order, sorted(order))

    def test_the_release_jobs_check_out_the_pinned_commit_with_full_history(self) -> None:
        for name in ("provision", "live"):
            with self.subTest(job=name):
                job = _block(self.lines, f"{name}:", 2)
                start = next(
                    index
                    for index, line in enumerate(job)
                    if line.strip() == "- uses: actions/checkout@v4"
                )
                options = [line.strip() for line in _block(job[start:], "with:", 8)]
                self.assertIn("ref: ${{ github.sha }}", options)
                self.assertIn("fetch-depth: 0", options)
                self.assertIn("persist-credentials: false", options)
                self.assertEqual(sum(line.strip().startswith("- uses: actions/checkout") for line in job), 1)

    def test_only_the_release_jobs_reach_the_trusted_runner(self) -> None:
        for job in ("cpu", "synthetic"):
            self.assertNotIn("self-hosted", self._job(job))
        offline = (_WORKFLOWS / "offline-tests.yml").read_text(encoding="utf-8")
        self.assertIn("pull_request:", offline)
        self.assertNotIn("self-hosted", offline)
        for path in _WORKFLOWS.glob("*.yml"):
            text = path.read_text(encoding="utf-8")
            if "self-hosted" in text:
                self.assertEqual(path, _WORKFLOW)
                self.assertNotIn("pull_request", text)
                self.assertNotIn("pull_request_target", text)

    def test_no_failure_is_masked_and_nothing_private_leaves_the_runner(self) -> None:
        for forbidden in (
            "continue-on-error",
            "upload-artifact",
            "secrets.",
            "|| true",
            "pull_request",
            "cancel-in-progress: true",
        ):
            self.assertNotIn(forbidden, self.text)
        self.assertEqual(
            [line.strip() for line in _block(self.lines, "permissions:", 0) if line.strip()],
            ["contents: read"],
        )
        self.assertEqual(
            self.text.count("persist-credentials: false"),
            self.text.count("uses: actions/checkout@"),
        )
        run_lines = [line for line in self.lines if line.strip().startswith("run:")]
        self.assertTrue(run_lines)
        for line in run_lines:
            self.assertNotIn("${{", line)
        for index, line in enumerate(self.lines):
            if line.strip() == "run: |":
                indent = len(line) - len(line.lstrip(" "))
                for body in self.lines[index + 1 :]:
                    if body.strip() and len(body) - len(body.lstrip(" ")) <= indent:
                        break
                    self.assertNotIn("${{", body)

    def test_the_persistent_service_runs_the_release_code_from_the_active_release(self) -> None:
        unit = _UNIT.read_text(encoding="utf-8")
        self.assertIn("EnvironmentFile=%h/.config/llm-behavior-ci/release.env", unit)
        self.assertIn('"$LIFECYCLE_RELEASE_ROOT/active/code/scripts/ci/release_ctl.py" serve', unit)
        self.assertIn("Restart=on-failure", unit)
        self.assertIn("Environment=PYTHONDONTWRITEBYTECODE=1", unit)
        self.assertNotIn("API_KEY", unit)


def _triggers(text: str) -> list[str]:
    lines = text.splitlines()
    return _keys(_block(lines, "on:", 0), 2)


def _check_names(text: str) -> list[str]:
    lines = text.splitlines()
    jobs = _block(lines, "jobs:", 0)
    names = []
    for job_id in _keys(jobs, 2):
        body = _block(jobs, f"{job_id}:", 2)
        declared = [line.strip()[len("name:") :].strip() for line in body if line.startswith("    name:")]
        names.append(declared[0] if declared else job_id)
    return names


class BranchProtectionDocumentationTests(unittest.TestCase):
    """The checks the operator is told to require must report on every pull request."""

    def setUp(self) -> None:
        text = (_ROOT / "docs" / "RELEASE_PIPELINE.md").read_text(encoding="utf-8")
        match = re.search(r"Required status checks: ((?:`[^`]+`(?:, )?)+)\.", text)
        self.assertIsNotNone(match)
        self.required = re.findall(r"`([^`]+)`", match.group(1))
        self.workflows = {
            path.name: path.read_text(encoding="utf-8") for path in sorted(_WORKFLOWS.glob("*.yml"))
        }

    def test_every_required_check_is_a_job_that_runs_on_pull_requests(self) -> None:
        self.assertEqual(self.required, ["offline-tests"])
        for check in self.required:
            with self.subTest(check=check):
                owners = [
                    name for name, text in self.workflows.items() if check in _check_names(text)
                ]
                self.assertEqual(owners, ["offline-tests.yml"])
                text = self.workflows[owners[0]]
                self.assertIn("pull_request", _triggers(text))
                self.assertNotIn("self-hosted", text)

    def test_no_release_lifecycle_check_can_be_required(self) -> None:
        release = self.workflows["release-lifecycle.yml"]
        self.assertEqual(_triggers(release), ["push", "workflow_dispatch"])
        for check in _check_names(release):
            self.assertNotIn(check, self.required)
        self.assertEqual(
            _check_names(release),
            ["CPU verification", "Synthetic lifecycle", "Provision release", "Deploy release (paid)"],
        )


class ResolveReleaseIdTests(unittest.TestCase):
    def _run(self, **env: str) -> subprocess.CompletedProcess[str]:
        output = Path(self._tmp.name) / "output"
        return subprocess.run(
            ["bash", str(_RESOLVE)],
            env={"PATH": os.environ["PATH"], "GITHUB_OUTPUT": str(output), **env},
            capture_output=True,
            text=True,
            check=False,
        )

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.output = Path(self._tmp.name) / "output"

    def test_a_push_names_the_release_after_its_commit(self) -> None:
        completed = self._run(EVENT_NAME="push", COMMIT="0123456789ab" + "c" * 28)
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(self.output.read_text(), "release_id=main-0123456789ab\n")

    def test_a_dispatch_needs_a_confirmed_release_id(self) -> None:
        commit = "a" * 40
        ok = self._run(
            EVENT_NAME="workflow_dispatch", COMMIT=commit, INPUT_RELEASE_ID="r7", CONFIRM_LIVE="r7"
        )
        self.assertEqual(ok.returncode, 0)
        self.assertEqual(self.output.read_text(), "release_id=r7\n")
        for name, env in {
            "unconfirmed": {"INPUT_RELEASE_ID": "r7", "CONFIRM_LIVE": "r8"},
            "empty": {"INPUT_RELEASE_ID": "", "CONFIRM_LIVE": ""},
            "traversal": {"INPUT_RELEASE_ID": "../x", "CONFIRM_LIVE": "../x"},
            "reserved": {"INPUT_RELEASE_ID": "LATEST", "CONFIRM_LIVE": "LATEST"},
        }.items():
            with self.subTest(case=name):
                self.output.unlink(missing_ok=True)
                completed = self._run(EVENT_NAME="workflow_dispatch", COMMIT=commit, **env)
                self.assertEqual(completed.returncode, 2)
                self.assertFalse(self.output.exists())
        for env in (
            {"EVENT_NAME": "pull_request", "COMMIT": commit},
            {"EVENT_NAME": "push", "COMMIT": "main"},
        ):
            with self.subTest(env=env):
                self.assertEqual(self._run(**env).returncode, 2)


class LiveReleaseStepTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.root = Path(self._tmpdir.name)
        (self.root / "releases" / "r2").mkdir(parents=True)
        (self.root / "releases" / "r2" / "release.json").write_text("{}")
        self.calls = self.root / "calls.txt"
        self.fake = _executable(
            self.root / "fake_python",
            'echo "$@" >> "$FAKE_CALLS"\necho "private detail" >&2\nexit "$FAKE_CODE"\n',
        )

    def _step(self, code: int, **overrides: str) -> subprocess.CompletedProcess[str]:
        env = {
            "PATH": os.environ["PATH"],
            "LIFECYCLE_RELEASE_ROOT": str(self.root / "releases"),
            "LIFECYCLE_PYTHON": str(self.fake),
            "RELEASE_ID": "r2",
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
        self.assertIn("release r2: exit 1, not promoted", summary)
        self.assertIn("release r2: exit 3, not promoted", summary)
        self.assertIn(
            "release r2: exit 5, not promoted: promoted, then rolled back to the previous "
            "production during Tier 3",
            summary,
        )
        stderr_files = sorted((self.root / "releases" / "r2").glob("lifecycle_stderr.*.txt"))
        self.assertEqual(len(stderr_files), 6)

    def test_the_live_command_gives_a_post_promotion_rollback_its_own_failing_code(self) -> None:
        module = _load("run_three_tier_dev_ci", "scripts/demo/run_three_tier_dev.py")
        codes = module.LIVE_EXIT_CODES
        self.assertEqual(codes["production_rolled_back"], 5)
        self.assertEqual([status for status, code in codes.items() if code == 0], ["promoted"])

    def test_the_step_runs_the_release_from_its_manifest(self) -> None:
        self._step(0)
        call = self.calls.read_text(encoding="utf-8")
        releases = self.root / "releases"
        self.assertIn(
            f"scripts/demo/run_three_tier_dev.py release --release-dir {releases}/r2", call
        )

    def test_bad_release_inputs_exit_without_running_the_lifecycle(self) -> None:
        (self.root / "outside").mkdir()
        (self.root / "releases" / "r3").mkdir()
        for name, overrides in {
            "path traversal": {"RELEASE_ID": "../outside"},
            "unprovisioned release": {"RELEASE_ID": "r3"},
            "unknown release": {"RELEASE_ID": "r9"},
            "missing release root": {"LIFECYCLE_RELEASE_ROOT": ""},
            "missing release id": {"RELEASE_ID": ""},
        }.items():
            with self.subTest(case=name):
                completed = self._step(0, **overrides)
                self.assertNotEqual(completed.returncode, 0)
                self.assertFalse(self.calls.exists())


class ActivateReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.root = Path(self._tmpdir.name)
        self.releases = self.root / "releases"
        for name in ("r1", "r2"):
            (self.releases / name).mkdir(parents=True)
            (self.releases / name / "release.json").write_text("{}")
        self.log = self.root / "log.txt"
        self.systemctl = _executable(
            self.root / "systemctl",
            'echo "systemctl $* -> $(readlink "$LIFECYCLE_RELEASE_ROOT/active")" >> "$LOG"\n',
        )
        self.python = _executable(
            self.root / "python",
            'echo "python $* -> $(readlink "$LIFECYCLE_RELEASE_ROOT/active")" >> "$LOG"\n'
            'if [[ "$2" == verify ]]; then exit "$VERIFY_CODE"; fi\n'
            'exit "$REGISTER_CODE"\n',
        )

    def _activate(
        self, release_id: str, register_code: int, verify_code: int = 0
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(_ACTIVATE)],
            cwd=_ROOT,
            env={
                "PATH": os.environ["PATH"],
                "LIFECYCLE_RELEASE_ROOT": str(self.releases),
                "LIFECYCLE_PYTHON": str(self.python),
                "SYSTEMCTL": str(self.systemctl),
                "RELEASE_ID": release_id,
                "REGISTER_CODE": str(register_code),
                "VERIFY_CODE": str(verify_code),
                "LOG": str(self.log),
            },
            capture_output=True,
            text=True,
            check=False,
        )

    def test_a_registered_release_becomes_active(self) -> None:
        completed = self._activate("r1", 0)
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(os.readlink(self.releases / "active"), "r1")
        log = self.log.read_text()
        self.assertLess(
            log.index(f"release_ctl.py verify --release-dir {self.releases}/r1"),
            log.index("systemctl --user restart"),
        )
        self.assertIn("systemctl --user restart llm-behavior-ci-release.service -> r1", log)
        self.assertIn(f"wait-registered --release-dir {self.releases}/r1 -> r1", log)

    def test_a_release_that_does_not_register_restores_the_previous_one(self) -> None:
        self._activate("r1", 0)
        completed = self._activate("r2", 4)
        self.assertEqual(completed.returncode, 4)
        self.assertEqual(os.readlink(self.releases / "active"), "r1")
        self.assertTrue(self.log.read_text().rstrip().endswith("-> r1"))

    def test_a_release_that_does_not_verify_is_never_activated(self) -> None:
        self._activate("r1", 0)
        completed = self._activate("r2", 0, verify_code=2)
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(os.readlink(self.releases / "active"), "r1")
        self.assertEqual(self.log.read_text().count("systemctl"), 1)
        self.assertIn("does not verify", completed.stderr)

    def test_a_first_release_that_does_not_register_leaves_nothing_active(self) -> None:
        completed = self._activate("r1", 4)
        self.assertEqual(completed.returncode, 4)
        self.assertFalse((self.releases / "active").is_symlink())
        self.assertIn("systemctl --user stop", self.log.read_text())

    def test_an_unprovisioned_release_is_not_activated(self) -> None:
        (self.releases / "r3").mkdir()
        for release_id in ("r3", "../r1"):
            with self.subTest(release_id=release_id):
                self.assertEqual(self._activate(release_id, 0).returncode, 2)
        self.assertFalse(self.log.exists())


class CheckSyntheticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load("check_synthetic_ci", "scripts/ci/check_synthetic.py")
        self.summary = {
            "scenario": "production_regression",
            "provenance": "synthetic_fixture",
            "status": "production_rolled_back",
            "verified": True,
            "checks": {name: True for name in self.module.REQUIRED_CHECKS},
        }

    def test_a_verified_scenario_passes(self) -> None:
        self.assertEqual(self.module.failures("production_regression", 0, self.summary), [])

    def test_any_missed_fact_fails_the_job(self) -> None:
        cases = {
            "exit code": (1, self.summary),
            "status": (0, {**self.summary, "status": "promoted"}),
            "unverified": (0, {**self.summary, "verified": False}),
            "check": (
                0,
                {**self.summary, "checks": {**self.summary["checks"], "stored_decisions": False}},
            ),
            "missing check": (0, {**self.summary, "checks": {"status": True}}),
            "live provenance": (0, {**self.summary, "provenance": "gate_run"}),
            "other scenario": (0, {**self.summary, "scenario": "healthy"}),
            "no summary": (0, None),
        }
        for name, (code, summary) in cases.items():
            with self.subTest(case=name):
                self.assertTrue(self.module.failures("production_regression", code, summary))

    def test_the_command_exits_1_on_an_unreadable_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scenario.json"
            path.write_text("not json")
            from contextlib import redirect_stdout
            from io import StringIO

            with redirect_stdout(StringIO()) as output:
                code = self.module.main(
                    ["--scenario", "healthy", "--exit-code", "0", str(path)]
                )
            self.assertEqual(code, 1)
            self.assertIn("no summary", json.loads(output.getvalue())["problems"])
        self.assertEqual(self.module.main([]), 2)


if __name__ == "__main__":
    unittest.main()
