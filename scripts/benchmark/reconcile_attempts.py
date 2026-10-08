"""Reconcile a lifecycle benchmark checkpoint's attempt ledger without running anything.

Requires --checkpoint. Marks every attempt left reserved or started by a
crash as interrupted, keeps the original caps, and prints the ledger
summary: consumed slots, known starts, and remaining capacity per mode.
Resume by rerunning the original benchmark command with the same
--checkpoint. Bare invocation exits 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from llm_behavior_ci.experiments.benchmark import BenchmarkError, reconcile_checkpoint


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None and len(sys.argv) <= 1:
        return 2
    parser = argparse.ArgumentParser(description="Reconcile benchmark attempts.")
    parser.add_argument("--checkpoint", required=True)
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as error:
        return 2 if error.code is None else int(error.code)
    try:
        summary = reconcile_checkpoint(Path(args.checkpoint))
    except BenchmarkError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {"reconciled": summary.reconciled, "attempts": summary.attempts},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
