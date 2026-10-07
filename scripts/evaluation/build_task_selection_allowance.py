"""Write the task-selection allowance that lets a dev service admit a train gate PASS.

Requires --train-config (the reference run configuration the offline gate
ran on train) and --output. Reads every task-selection leaf from that
configuration, so no hash is typed by hand, and writes a
``task-selection-allowance-v1`` document for ``serve.py
--task-selection-allowance``. Refuses a configuration whose split is not
train and an output under results/. The document holds only split,
selection rule, seed, count, and the train task-set hash. Bare invocation
exits 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from llm_behavior_ci.config import ConfigError, RunConfiguration
from llm_behavior_ci.experiments.protocol import (
    ProtocolError,
    task_selection_allowance_document,
    task_selection_allowance_for,
)
from llm_behavior_ci.runtime.provenance import repository_root


def _under_results(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    base = (root / "results").resolve()
    return resolved == base or base in resolved.parents


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        return 2
    parser = argparse.ArgumentParser(
        description="Write a train-to-dev task-selection allowance."
    )
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--output", required=True)
    try:
        args = parser.parse_args(args_list)
    except SystemExit as error:
        return 2 if error.code is None else int(error.code)
    output = Path(args.output)
    if _under_results(output, repository_root()):
        print("output must not be under results/", file=sys.stderr)
        return 2
    try:
        payload = json.loads(Path(args.train_config).read_text(encoding="utf-8"))
        train = RunConfiguration.from_dict(payload)
        if train.task.split != "train":
            print("train configuration split must be train", file=sys.stderr)
            return 2
        document = task_selection_allowance_document(
            task_selection_allowance_for(train)
        )
    except (OSError, json.JSONDecodeError, ConfigError, ProtocolError) as error:
        print(str(error) or "allowance build failed", file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(document, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
