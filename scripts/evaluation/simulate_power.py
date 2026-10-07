"""Empirical clustered paired power from a local baseline file.

Requires --baselines (from collect_baseline.py), --harm-margin, --alpha,
--sample-size (repeatable), --simulations, and --seed. Prints a summary
with no task or scenario ids: power, the null rejection rate, and counts.
--output writes the same summary to a path outside results/; nothing
is written under results/ by this command.
The normal approximation in assess_harm_study.py stays advisory. Bare
invocation exits 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from llm_behavior_ci.experiments.power import empirical_harm_power
from llm_behavior_ci.experiments.validation import (
    ValidationError,
    default_results_root,
    load_local_baselines,
    write_public_report,
)


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        return 2
    parser = argparse.ArgumentParser(description="Simulate empirical harm-study power.")
    parser.add_argument("--baselines", required=True)
    parser.add_argument("--harm-margin", required=True, type=float)
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--sample-size", required=True, type=int, action="append")
    parser.add_argument("--simulations", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", default=None)
    parser.add_argument("--results-root", default=None)
    try:
        args = parser.parse_args(args_list)
    except SystemExit as error:
        return 2 if error.code is None else int(error.code)
    try:
        report = empirical_harm_power(
            load_local_baselines(Path(args.baselines)),
            harm_margin=args.harm_margin,
            alpha=args.alpha,
            sample_sizes=tuple(args.sample_size),
            simulations=args.simulations,
            seed=args.seed,
        )
        summary = report.to_dict()
        if args.output is not None:
            root = Path(args.results_root) if args.results_root else default_results_root()
            write_public_report(summary, Path(args.output), results_root=root)
    except ValidationError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
