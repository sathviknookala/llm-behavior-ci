from __future__ import annotations

import os
import subprocess
import sys
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


if __name__ == "__main__":
    unittest.main()
