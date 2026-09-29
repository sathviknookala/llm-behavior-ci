from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from llm_behavior_ci.config import ConfigError, ModelConfiguration
from llm_behavior_ci.runtime.launch_spec import build_vllm_launch_spec

PTXAS_BLACKWELL_ENV = "TRITON_PTXAS_BLACKWELL_PATH"


def cuda_12_ptxas() -> Path | None:
    try:
        spec = importlib.util.find_spec("nvidia.cuda_nvcc")
    except ModuleNotFoundError:
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    for location in spec.submodule_search_locations:
        candidate = Path(location) / "bin" / "ptxas"
        if candidate.is_file():
            return candidate
    return None


def machine_env(ptxas: Path) -> tuple[tuple[str, str], ...]:
    return ((PTXAS_BLACKWELL_ENV, str(ptxas)),)


def launch_command(
    model: ModelConfiguration,
    *,
    vllm_executable: Path,
    ptxas: Path,
    host: str,
    port: int,
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    spec = build_vllm_launch_spec(model)
    argv = (
        str(vllm_executable),
        *spec.argv[1:],
        "--host",
        host,
        "--port",
        str(port),
    )
    return argv, spec.env + machine_env(ptxas)


def _model_configuration(path: Path) -> ModelConfiguration:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"cannot read configuration: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    if not isinstance(document, dict) or "model" not in document:
        print("configuration must contain a model section", file=sys.stderr)
        raise SystemExit(2)
    try:
        return ModelConfiguration.from_dict(document["model"])
    except ConfigError as error:
        print(f"invalid model configuration: {error}", file=sys.stderr)
        raise SystemExit(2) from error


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch one vLLM server for a model configuration."
    )
    parser.add_argument("--configuration", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(arguments)
    model = _model_configuration(args.configuration)
    vllm_executable = Path(sys.executable).parent / "vllm"
    ptxas = cuda_12_ptxas()
    if ptxas is None or not vllm_executable.is_file():
        print(
            "run with the vLLM environment's interpreter "
            "(.venv-vllm/bin/python): vllm or the CUDA 12 ptxas is missing",
            file=sys.stderr,
        )
        return 2
    argv, env = launch_command(
        model,
        vllm_executable=vllm_executable,
        ptxas=ptxas,
        host=args.host,
        port=args.port,
    )
    if args.dry_run:
        print(json.dumps({"argv": list(argv), "env": dict(env)}, indent=2))
        return 0
    os.execve(argv[0], list(argv), {**os.environ, **dict(env)})


if __name__ == "__main__":
    sys.exit(main())
