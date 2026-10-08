from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_ENV = {**os.environ, "PYTHONPATH": "src"}


def _bare(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, script],
        cwd=_ROOT,
        env=_ENV,
        capture_output=True,
        text=True,
        check=False,
    )


def _run(script: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, script, *args],
        cwd=_ROOT,
        env=_ENV,
        capture_output=True,
        text=True,
        check=False,
    )


class BareCliExitTests(unittest.TestCase):
    def test_offline_gate_exits_2(self) -> None:
        completed = _bare("scripts/run_offline_gate.py")
        self.assertEqual(completed.returncode, 2)

    def test_lifecycle_benchmark_exits_2(self) -> None:
        completed = _bare("scripts/benchmark/run_lifecycle_benchmark.py")
        self.assertEqual(completed.returncode, 2)

    def test_replay_detectors_exits_2(self) -> None:
        completed = _bare("scripts/replay/replay_detectors.py")
        self.assertEqual(completed.returncode, 2)

    def test_serve_exits_2(self) -> None:
        completed = _bare("scripts/service/serve.py")
        self.assertEqual(completed.returncode, 2)

    def test_lock_protocol_exits_2(self) -> None:
        completed = _bare("scripts/evaluation/lock_protocol.py")
        self.assertEqual(completed.returncode, 2)

    def test_lock_protocol_malformed_canary_exits_1_without_traceback(self) -> None:
        settings = {
            "configurations": [],
            "task_selections": [],
            "harm_labels": [],
            "validation_reports": [],
            "seeds": [1],
            "faults": [],
            "plan_evidence": {
                "plan_format_version": "plan-v1",
                "plan_quality_features": ["numbered_step_count"],
                "plan_quality_weights": [1.0],
                "mmd_features": [],
                "kl_approximation": "top_k",
                "required_statistics": ["plan_quality"],
                "validation_provenance": "synthetic_fixture",
            },
            "gate": {
                "confidence_level": 0.9,
                "bootstrap_resamples": 40,
                "score_margin": -0.02,
                "kl_limit_nats": 0.05,
                "mmd_bandwidth": 1.0,
                "mmd_permutations": 19,
                "mmd_alpha": 0.05,
                "plan_format_version": "plan-v1",
            },
            "canary": "not-an-object",
            "monitor": {},
            "stream": {},
            "analysis_version": "v1",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings_path = root / "settings.json"
            settings_path.write_text(json.dumps(settings), encoding="utf-8")
            completed = _run(
                "scripts/evaluation/lock_protocol.py",
                [
                    "--settings",
                    str(settings_path),
                    "--output",
                    str(root / "lock.json"),
                ],
            )
        self.assertEqual(completed.returncode, 1)
        self.assertNotIn("Traceback", completed.stderr)

    def test_capture_aa_exits_2(self) -> None:
        completed = _bare("scripts/evaluation/capture_aa.py")
        self.assertEqual(completed.returncode, 2)

    def test_validate_method_exits_2(self) -> None:
        completed = _bare("scripts/evaluation/validate_method.py")
        self.assertEqual(completed.returncode, 2)

    def test_assess_harm_study_exits_2(self) -> None:
        completed = _bare("scripts/evaluation/assess_harm_study.py")
        self.assertEqual(completed.returncode, 2)

    def test_characterize_harm_exits_2(self) -> None:
        completed = _bare("scripts/evaluation/characterize_harm.py")
        self.assertEqual(completed.returncode, 2)

    def test_export_public_exits_2(self) -> None:
        completed = _bare("scripts/evaluation/export_public.py")
        self.assertEqual(completed.returncode, 2)

    def test_compare_plan_kl_exits_2(self) -> None:
        completed = _bare("scripts/evaluation/compare_plan_kl.py")
        self.assertEqual(completed.returncode, 2)

    def test_smoke_live_episode_exits_2(self) -> None:
        completed = _bare("scripts/evaluation/smoke_live_episode.py")
        self.assertEqual(completed.returncode, 2)

    def test_new_lifecycle_commands_exit_2(self) -> None:
        for script in (
            "scripts/evaluation/build_run_configuration.py",
            "scripts/evaluation/build_task_selection_allowance.py",
            "scripts/evaluation/calibrate_hosted.py",
            "scripts/evaluation/collect_baseline.py",
            "scripts/evaluation/rehearse_dev_stream.py",
            "scripts/evaluation/simulate_power.py",
            "scripts/evaluation/export_usage.py",
            "scripts/data/annotate_task_metadata.py",
            "scripts/data/plan_specs.py",
            "scripts/benchmark/reconcile_attempts.py",
        ):
            with self.subTest(script=script):
                completed = _bare(script)
                self.assertEqual(completed.returncode, 2, completed.stderr)
                self.assertNotIn("Traceback", completed.stderr)

    def test_benchmark_without_runtime_source_exits_2(self) -> None:
        completed = _run(
            "scripts/benchmark/run_lifecycle_benchmark.py",
            [
                "--protocol", "p.json", "--fault", "f.json", "--train-tasks", "t.json",
                "--test-normal-tasks", "n.json", "--baselines", "b.json",
                "--plan-evidence", "e.json", "--checkpoint", "c.json",
            ],
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("--live-runtime", completed.stderr)


if __name__ == "__main__":
    unittest.main()
