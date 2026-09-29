from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path

from llm_behavior_ci.config import ModelConfiguration
from llm_behavior_ci.runtime.launch_spec import build_vllm_launch_spec

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "service" / "launch_vllm.py"
_PRODUCTION = _ROOT / "configs" / "models" / "qwen3_4b_production.json"


def _load_script():
    spec = importlib.util.spec_from_file_location("launch_vllm", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _production_model() -> ModelConfiguration:
    document = json.loads(_PRODUCTION.read_text(encoding="utf-8"))
    return ModelConfiguration.from_dict(document["model"])


class LaunchVLLMTests(unittest.TestCase):
    def test_bare_invocation_exits_2(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(_SCRIPT)],
            cwd=_ROOT,
            env={**os.environ, "PYTHONPATH": "src"},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)

    def test_interpreter_without_vllm_exits_2(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(_SCRIPT), "--configuration", str(_PRODUCTION), "--dry-run"],
            cwd=_ROOT,
            env={**os.environ, "PYTHONPATH": "src"},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertNotIn("Traceback", completed.stderr)

    def test_command_is_the_launch_spec_with_machine_env(self) -> None:
        script = _load_script()
        model = _production_model()
        argv, env = script.launch_command(
            model,
            vllm_executable=Path("/venv/bin/vllm"),
            ptxas=Path("/venv/nvidia/cuda_nvcc/bin/ptxas"),
            host="127.0.0.1",
            port=8000,
        )
        spec = build_vllm_launch_spec(model)
        self.assertEqual(argv[0], "/venv/bin/vllm")
        self.assertEqual(argv[1:-4], spec.argv[1:])
        self.assertEqual(argv[-4:], ("--host", "127.0.0.1", "--port", "8000"))
        self.assertEqual(
            dict(env),
            {
                "TRITON_PTXAS_BLACKWELL_PATH": "/venv/nvidia/cuda_nvcc/bin/ptxas",
                "VLLM_USE_FLASHINFER_SAMPLER": "0",
            },
        )

    def test_spec_env_is_kept(self) -> None:
        script = _load_script()
        model = _production_model()
        model = replace(model, serving=replace(model.serving, batch_invariant=True))
        _argv, env = script.launch_command(
            model,
            vllm_executable=Path("/venv/bin/vllm"),
            ptxas=Path("/p"),
            host="127.0.0.1",
            port=8000,
        )
        self.assertEqual(dict(env)["VLLM_BATCH_INVARIANT"], "1")


if __name__ == "__main__":
    unittest.main()
