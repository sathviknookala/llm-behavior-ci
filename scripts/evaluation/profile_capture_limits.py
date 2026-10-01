"""Summarize one local A/A capture as a public limit profile.

Requires --capture, --output, --label, --step-limit, and --max-tokens.
Does not run inference and does not change those limits. With no arguments
it exits 2. The written document is an aggregate: no task text, plans,
trajectories, or evaluator reports.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from llm_behavior_ci.experiments.limit_profile import (
    ProfileError,
    format_summary,
    profile_capture,
    write_public_profile,
)


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        return 2
    parser = argparse.ArgumentParser(description="Profile one local A/A capture.")
    parser.add_argument("--capture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--step-limit", required=True, type=int)
    parser.add_argument("--max-tokens", required=True, type=int)
    args = parser.parse_args(args_list)
    document = json.loads(Path(args.capture).read_text(encoding="utf-8"))
    profile = profile_capture(
        document,
        capture_label=args.label,
        step_limit=args.step_limit,
        max_tokens=args.max_tokens,
    )
    write_public_profile(profile, Path(args.output))
    sys.stdout.write(format_summary(profile))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProfileError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
