"""Build a full run configuration from a committed template and a local task set.

Requires --template (configs/models/*.json), --task-set (local manifest
with task_ids and scenario_ids), --run-seed, and --output. --git-commit
defaults to repository HEAD; --protocol-hash defaults to null. Refuses an
output under results/ and a tree whose tracked source or config is dirty.
Runs the tool-access preflight against the manifest. Prints the public
configuration summary (hashes and bindings, no task ids). Bare invocation
exits 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from llm_behavior_ci.experiments.run_config import (
    RunConfigError,
    build_run_configuration,
    configuration_document,
    configuration_summary,
    load_local_task_manifest,
    preflight_tool_access,
)
from llm_behavior_ci.runtime.provenance import (
    ProvenanceError,
    RepositoryState,
    read_repository_state,
    repository_root,
    require_committed_provenance,
)


def _under_results(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    base = (root / "results").resolve()
    return resolved == base or base in resolved.parents


def main(
    argv: Sequence[str] | None = None,
    *,
    repository_state: RepositoryState | None = None,
) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        return 2
    parser = argparse.ArgumentParser(description="Build one run configuration.")
    parser.add_argument("--template", required=True)
    parser.add_argument("--task-set", required=True)
    parser.add_argument("--run-seed", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--git-commit", default=None)
    parser.add_argument("--protocol-hash", default=None)
    parser.add_argument("--committed-tasks-dir", default=None)
    try:
        args = parser.parse_args(args_list)
    except SystemExit as error:
        return 2 if error.code is None else int(error.code)
    root = repository_root()
    output = Path(args.output)
    if _under_results(output, root):
        print("output must not be under results/", file=sys.stderr)
        return 2
    try:
        state = repository_state if repository_state is not None else read_repository_state(root)
        commit = args.git_commit or state.head
        manifest = load_local_task_manifest(
            Path(args.task_set),
            committed_tasks_dir=(
                Path(args.committed_tasks_dir)
                if args.committed_tasks_dir
                else root / "configs" / "tasks"
            ),
        )
        template = json.loads(Path(args.template).read_text(encoding="utf-8"))
        configuration = build_run_configuration(
            template,
            manifest.task,
            run_seed=args.run_seed,
            git_commit=commit,
            protocol_hash=args.protocol_hash,
        )
        preflight_tool_access(configuration, manifest)
        require_committed_provenance(configuration, state)
    except (RunConfigError, ProvenanceError, OSError, json.JSONDecodeError) as error:
        print(str(error) or "run configuration build failed", file=sys.stderr)
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(configuration_document(configuration), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(configuration_summary(configuration), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
