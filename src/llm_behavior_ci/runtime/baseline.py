"""Local production versus do-nothing outcomes for the existing harm study.

This runner does not choose a harm margin. It builds ``BaselineOutcome``
rows that ``assess_harm_study`` already accepts. Task ids stay in
``pair_key`` on the local records and are not a public summary field.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

from llm_behavior_ci.config import RunConfiguration, new_run_identity
from llm_behavior_ci.experiments.validation import BaselineOutcome, default_results_root
from llm_behavior_ci.records import EpisodeResult
from llm_behavior_ci.runtime.episode import (
    EpisodeRejected,
    RuntimeDependencies,
    run_do_nothing_episode,
    run_episode,
)
from llm_behavior_ci.tasks.streams import TaskArrival

_INFRASTRUCTURE_APPS = frozenset({"admin", "api_docs", "supervisor"})
_OPEN_SPLITS = frozenset({"train", "dev"})


def app_label(required_apps: Sequence[str] | None) -> str | None:
    """One stable label for the task's non-infrastructure required apps.

    Infrastructure apps are omitted. Several remaining apps are joined in
    sorted order. An empty remainder is ``None``.
    """

    if required_apps is None:
        return None
    apps = tuple(
        sorted(app for app in required_apps if app not in _INFRASTRUCTURE_APPS and app)
    )
    if not apps:
        return None
    return "+".join(apps)


def outcome_from_episode(
    episode: EpisodeResult,
    *,
    role: str,
    app: str | None,
) -> BaselineOutcome:
    """Copy evaluator fields. A missing scenario id fails closed."""

    scenario_id = episode.task.scenario_id
    if scenario_id is None or scenario_id == "":
        raise EpisodeRejected("baseline outcome requires a scenario id")
    outcome = episode.evaluator_outcome
    return BaselineOutcome(
        role=role,
        success=None if outcome is None else outcome.success,
        requirement_fraction=None if outcome is None else outcome.requirement_fraction,
        scenario_id=scenario_id,
        app=app,
        difficulty=None if outcome is None else outcome.difficulty,
        pair_key=episode.task.task_id,
    )


def collect_baseline_outcomes(
    arrivals: Sequence[TaskArrival],
    configuration: RunConfiguration,
    *,
    production_runtime: RuntimeDependencies,
    do_nothing_runtime: RuntimeDependencies,
    repetition: int | None = None,
    required_apps_for: Callable[[str], Sequence[str] | None] | None = None,
) -> tuple[BaselineOutcome, ...]:
    """Run production and do-nothing on the same arrivals.

    ``train`` and ``dev`` only. Each arrival keeps its scenario id on both
    roles and uses the task id as the local pair key; with ``repetition``
    the key is ``<task id>#r<repetition>``, so repeated passes over one
    task stay distinct pairs that share a task. The app label is read from
    ``required_apps`` on the production session when that method exists,
    otherwise from ``required_apps_for``. A missing evaluator outcome stays
    ``None`` on its row.
    """

    if repetition is not None and (
        isinstance(repetition, bool) or not isinstance(repetition, int) or repetition < 0
    ):
        raise EpisodeRejected("repetition must be a nonnegative integer")

    if not isinstance(configuration, RunConfiguration):
        raise EpisodeRejected("baseline collection requires a run configuration")
    if configuration.task.split not in _OPEN_SPLITS:
        raise EpisodeRejected("baseline collection is closed on this split")
    if isinstance(arrivals, str) or not isinstance(arrivals, Sequence) or not arrivals:
        raise EpisodeRejected("arrivals must be a non-empty sequence")
    production_run = new_run_identity(configuration)
    nothing_run = new_run_identity(configuration)
    labels: dict[str, tuple[str, ...] | None] = {}
    production_runtime = RuntimeDependencies(
        session_factory=_labeling_factory(production_runtime.session_factory, labels),
        agent=production_runtime.agent,
        clock=production_runtime.clock,
    )
    rows: list[BaselineOutcome] = []
    for arrival in arrivals:
        if not isinstance(arrival, TaskArrival):
            raise EpisodeRejected("stream items must be task arrivals")
        if arrival.scenario_id is None or arrival.scenario_id == "":
            raise EpisodeRejected("baseline arrival requires a scenario id")
        production = run_episode(
            arrival.task_id,
            configuration,
            "execute",
            run=production_run,
            runtime=production_runtime,
            scenario_id=arrival.scenario_id,
        )
        nothing = run_do_nothing_episode(
            arrival.task_id,
            configuration,
            run=nothing_run,
            runtime=do_nothing_runtime,
            scenario_id=arrival.scenario_id,
        )
        apps = labels.get(arrival.task_id)
        if apps is None and required_apps_for is not None:
            apps = required_apps_for(arrival.task_id)
        app = app_label(apps)
        pair = (
            None if repetition is None else f"{arrival.task_id}#r{repetition}"
        )
        for episode, role in ((production, "production"), (nothing, "do_nothing")):
            row = outcome_from_episode(episode, role=role, app=app)
            rows.append(row if pair is None else replace(row, pair_key=pair))
    return tuple(rows)


def write_local_baselines(
    path: Path,
    outcomes: Sequence[BaselineOutcome],
    *,
    results_root: Path | None = None,
) -> None:
    """Write local baseline rows. Refuses a path under ``results/``."""

    root = results_root if results_root is not None else default_results_root()
    resolved = path.resolve()
    base = root.resolve()
    if resolved == base or base in resolved.parents:
        raise EpisodeRejected("baseline output cannot be written as a public result")
    payload = {
        "visibility": "local",
        "outcomes": [
            {
                "role": item.role,
                "success": item.success,
                "requirement_fraction": item.requirement_fraction,
                "scenario_id": item.scenario_id,
                "app": item.app,
                "difficulty": item.difficulty,
                "pair_key": item.pair_key,
            }
            for item in outcomes
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _labeling_factory(inner, labels: dict[str, tuple[str, ...] | None]):
    def factory(task_id: str):
        session = inner(task_id)
        reader = getattr(session, "required_apps", None)
        if callable(reader):
            labels[task_id] = tuple(str(app) for app in reader())
        else:
            labels[task_id] = None
        return session

    return factory
