"""Collect production and do-nothing outcomes on identical fresh worlds.

Requires --configuration (a built run configuration), --task-set (the
local manifest it was built from; train or dev only), --repetitions, and
--output (a local JSON path, refused under results/). Every repetition
runs each task once per role, each episode in a fresh AppWorld world; rows
keep scenario labels, missing evaluator outcomes stay null, and pair keys
are ``<task id>#r<repetition>``. A vLLM configuration needs --endpoint; a
hosted one takes none and reads its key from the environment. The output
feeds ``assess_harm_study.py --baselines`` and ``simulate_power.py``.
Prints only counts. Bare invocation exits 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from llm_behavior_ci.config import ConfigError, RunConfiguration
from llm_behavior_ci.experiments.run_config import RunConfigError, load_local_task_manifest
from llm_behavior_ci.runtime.baseline import collect_baseline_outcomes, write_local_baselines
from llm_behavior_ci.runtime.episode import EpisodeRejected
from llm_behavior_ci.runtime.factory import LiveRuntimeFactory
from llm_behavior_ci.runtime.provenance import ProvenanceError, enforce_committed_provenance
from llm_behavior_ci.tasks.streams import TaskArrival


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        return 2
    parser = argparse.ArgumentParser(description="Collect production and do-nothing outcomes.")
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--task-set", required=True)
    parser.add_argument("--repetitions", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint", default=None)
    try:
        args = parser.parse_args(args_list)
    except SystemExit as error:
        return 2 if error.code is None else int(error.code)
    if args.repetitions < 1:
        print("--repetitions must be at least 1", file=sys.stderr)
        return 2
    try:
        configuration = RunConfiguration.from_dict(
            json.loads(Path(args.configuration).read_text(encoding="utf-8"))
        )
        manifest = load_local_task_manifest(Path(args.task_set))
        if manifest.task_set.task_set_hash != configuration.task.task_set_hash:
            raise ConfigError("task set does not match the configuration")
        enforce_committed_provenance(configuration)
        factory = LiveRuntimeFactory.from_endpoints(production=args.endpoint)
        factory.preflight({"production": configuration})
        arrivals = tuple(
            TaskArrival(
                index=index,
                task_id=task_id,
                scenario_id=scenario_id,
                scheduled_offset_seconds=0.0,
                stream_seed=configuration.run_seed,
            )
            for index, (task_id, scenario_id) in enumerate(
                zip(manifest.task_set.task_ids, manifest.task_set.scenario_ids, strict=True)
            )
        )
        rows = []
        for repetition in range(args.repetitions):
            rows.extend(
                collect_baseline_outcomes(
                    arrivals,
                    configuration,
                    production_runtime=factory(configuration, mode="execute", role="production"),
                    do_nothing_runtime=factory(configuration, mode="execute", role="production"),
                    repetition=repetition,
                    required_apps_for=manifest.required_apps,
                )
            )
        write_local_baselines(Path(args.output), rows)
    except (
        ConfigError,
        RunConfigError,
        EpisodeRejected,
        ProvenanceError,
        OSError,
        json.JSONDecodeError,
    ) as error:
        print(str(error) or "baseline collection failed", file=sys.stderr)
        return 1
    missing = sum(1 for row in rows if row.success is None)
    print(
        json.dumps(
            {
                "record": "baseline_collection",
                "split": configuration.task.split,
                "task_set_hash": configuration.task.task_set_hash,
                "repetitions": args.repetitions,
                "rows": len(rows),
                "missing_outcomes": missing,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
