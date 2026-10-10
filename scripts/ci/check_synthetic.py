"""Fail a synthetic CI job unless its scenario verified every expected fact.

Reads the JSON ``run_three_tier_dev.py synthetic`` printed and that
command's exit code. Passes only when the code is 0, the summary is a
``synthetic_fixture`` run of the named scenario, its status is the
scenario's expected status, and every check (deployment state, serving
configuration, stored decisions, persisted gate artifact, persisted
episodes) is true. Prints the check names and results only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

EXPECTED_STATUS = {
    "blocked": "blocked",
    "healthy": "promoted",
    "regression": "rolled_back",
    "production_regression": "production_rolled_back",
}
REQUIRED_CHECKS = frozenset(
    {
        "status",
        "deployment_state",
        "serving_configuration",
        "stored_decisions",
        "gate_artifact_persisted",
        "episodes_persisted",
    }
)


def failures(scenario: str, exit_code: int, summary: object) -> list[str]:
    problems = []
    if exit_code != 0:
        problems.append(f"exit code {exit_code}")
    if not isinstance(summary, dict):
        return [*problems, "no summary"]
    if summary.get("scenario") != scenario:
        problems.append("another scenario ran")
    if summary.get("provenance") != "synthetic_fixture":
        problems.append("not a synthetic_fixture run")
    if summary.get("status") != EXPECTED_STATUS[scenario]:
        problems.append(f"status {summary.get('status')}")
    if summary.get("verified") is not True:
        problems.append("not verified")
    checks = summary.get("checks")
    if not isinstance(checks, dict) or set(checks) != REQUIRED_CHECKS:
        problems.append("checks missing")
    else:
        problems.extend(f"check {name} failed" for name, ok in sorted(checks.items()) if ok is not True)
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        return 2
    parser = argparse.ArgumentParser(description="Check one synthetic lifecycle scenario.")
    parser.add_argument("--scenario", required=True, choices=sorted(EXPECTED_STATUS))
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument("summary")
    try:
        args = parser.parse_args(args_list)
    except SystemExit as error:
        return 2 if error.code is None else int(error.code)
    try:
        summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        summary = None
    problems = failures(args.scenario, args.exit_code, summary)
    checks = summary.get("checks") if isinstance(summary, dict) else None
    print(json.dumps({"scenario": args.scenario, "checks": checks, "problems": problems}, sort_keys=True))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
