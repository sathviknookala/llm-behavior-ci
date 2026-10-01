"""Descriptive limits from one local A/A capture.

The profile is a public aggregate. It counts turns, tool calls, generated
tokens, termination reasons, and paired disagreements. It does not copy
task identity, plans, trajectories, or evaluator reports, and it does not
choose ``step_limit`` or ``max_tokens``. Plan token lengths are reported
separately and are not execute-limit evidence.

Generated-token lengths come from ``generated_token_count`` when the step
records that field. Captures that predate the field are counted with
``len(top_k_logprobs)``, which stored one position per generated token.
Lengths are not taken from output text.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from llm_behavior_ci.records import PROTECTED_FIELDS, RecordError, assert_public_payload
from llm_behavior_ci.runtime.actions import ActionRejected, parse_model_output

_CLOSED_SPLITS = frozenset({"test_normal", "test_challenge"})
_PERCENTILES = (50, 90, 95, 99)
_TOKEN_TAILS = (64, 96, 128, 192, 256, 1024)
_PARSER_CATEGORIES = (
    "empty_action",
    "syntax_error",
    "not_single_expression",
    "not_single_call",
    "multiple_calls",
    "not_apis_call",
    "invalid_identifier",
    "other",
)


class ProfileError(ValueError):
    pass


def profile_capture(
    document: Mapping[str, object],
    *,
    capture_label: str,
    step_limit: int,
    max_tokens: int,
) -> dict[str, object]:
    """Summarize one local capture. The return value is public."""

    label = _label(capture_label)
    limit = _positive(step_limit, "step_limit")
    token_limit = _positive(max_tokens, "max_tokens")
    capture = _capture(document)
    records = _records(capture)
    split, configuration_hash, task_set_hash, git_commit = _identity(records)
    task_ids, scenario_ids = _population(records)
    execute_records = [record for record in records if record["mode"] == "execute"]
    plan_records = [record for record in records if record["mode"] == "plan"]
    profile = {
        "visibility": "public",
        "record": "capture_limit_profile",
        "purpose": (
            "descriptive statistics for choosing execute step_limit and "
            "max_tokens later; not a protocol threshold"
        ),
        "percentile_method": "inclusive_linear",
        "capture_label": label,
        "configuration_hash": configuration_hash,
        "task_set_hash": task_set_hash,
        "git_commit": git_commit,
        "split": split,
        "repetitions": _positive(capture.get("repetitions"), "repetitions"),
        "concurrency": _positive(capture.get("concurrency"), "concurrency"),
        "modes": _modes(capture.get("modes")),
        "scenario_count": len(scenario_ids),
        "task_count": len(task_ids),
        "configured_step_limit": limit,
        "configured_max_tokens": token_limit,
        "execute": _execute_profile(execute_records, limit, token_limit),
        "plan": _plan_profile(plan_records),
    }
    _assert_public(profile)
    return profile


def format_summary(profile: Mapping[str, object]) -> str:
    """One public text summary. No task text, plans, or trajectories."""

    execute = _mapping(profile.get("execute"), "execute")
    plan = _mapping(profile.get("plan"), "plan")
    turns = _mapping(execute.get("model_turns"), "model_turns")
    tools = _mapping(execute.get("tool_calls"), "tool_calls")
    turn_tokens = _mapping(
        execute.get("generated_tokens_per_turn"),
        "generated_tokens_per_turn",
    )
    episode_tokens = _mapping(
        execute.get("generated_tokens_per_episode"),
        "generated_tokens_per_episode",
    )
    completion = _mapping(execute.get("completion_turn"), "completion_turn")
    errors = _mapping(execute.get("errors"), "errors")
    paired = _mapping(execute.get("paired"), "paired")
    length = _mapping(paired.get("trajectory_length_difference"), "length")
    plan_tokens = _mapping(plan.get("generated_tokens"), "plan generated_tokens")
    lines = [
        "capture_limit_profile",
        f"capture_label: {profile['capture_label']}",
        f"split: {profile['split']}",
        f"configuration_hash: {profile['configuration_hash']}",
        f"task_set_hash: {profile['task_set_hash']}",
        f"task_count: {profile['task_count']}",
        f"scenario_count: {profile['scenario_count']}",
        f"execute_episodes: {execute['episode_count']}",
        f"execute_pairs: {execute['pair_count']}",
        f"configured_step_limit: {profile['configured_step_limit']}",
        f"configured_max_tokens: {profile['configured_max_tokens']}",
        _percentile_line("model_turns", turns),
        _percentile_line("tool_calls", tools),
        _percentile_line("generated_tokens_per_turn", turn_tokens),
        "generated_tokens_per_turn_at_or_above: "
        + _threshold_text(turn_tokens.get("at_or_above")),
        _percentile_line("generated_tokens_per_episode", episode_tokens),
        (
            "step_limit_hits: "
            f"{execute['step_limit_hit_count']} "
            f"fraction={execute['step_limit_hit_fraction']}"
        ),
        f"turns_at_configured_max_tokens: {execute['turns_at_configured_max_tokens']}",
        "termination: " + _count_text(execute.get("termination_counts")),
        (
            "completion_episodes: "
            f"{completion['count']} "
            f"p50={completion['p50']} p90={completion['p90']} "
            f"p95={completion['p95']} p99={completion['p99']} "
            f"max={completion['max']}"
        ),
        "parser: " + _count_text(_mapping(errors.get("parser"), "parser").get("categories")),
        f"parser_events: {errors['parser']['event_count']}",
        f"invalid_action_episodes: {errors['invalid_action']['episode_count']}",
        (
            "repeated_action_events: "
            f"{errors['repeated_action']['event_count']} "
            f"episodes={errors['repeated_action']['episode_count']}"
        ),
        (
            "tool_error_events: "
            f"{errors['tool']['event_count']} "
            f"recoverable={errors['tool']['recoverable_count']} "
            f"unrecoverable={errors['tool']['unrecoverable_count']}"
        ),
        (
            "evaluator_disagreements: "
            f"{paired['disagreement_count']}/{paired['pair_count']} "
            f"missing_outcomes={paired['missing_outcome_count']}"
        ),
        _percentile_line("trajectory_length_difference", length),
        (
            "plan_generated_tokens "
            "(not for execute limits): "
            + _percentile_line("plan_tokens", plan_tokens).split(": ", 1)[1]
        ),
        "teacher_forced_status: " + _count_text(plan.get("teacher_forced_status")),
        *_behavior_lines(execute.get("by_behavior")),
    ]
    return "\n".join(lines) + "\n"


def write_public_profile(profile: Mapping[str, object], output_path: Path) -> None:
    """Write one public profile. Refuses a payload that carries local fields."""

    if not isinstance(profile, Mapping):
        raise ProfileError("profile must be a mapping")
    if profile.get("visibility") != "public":
        raise ProfileError("profile visibility must be public")
    if profile.get("record") != "capture_limit_profile":
        raise ProfileError("profile record is not a capture limit profile")
    _assert_public(profile)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _execute_profile(
    records: Sequence[Mapping[str, object]],
    step_limit: int,
    max_tokens: int,
) -> dict[str, object]:
    episodes = [_episode(record, role) for record in records for role in ("reference", "candidate")]
    model_turns: list[int] = []
    tool_calls: list[int] = []
    tokens_per_turn: list[int] = []
    tokens_per_episode: list[int] = []
    completion_turns: list[int] = []
    termination: Counter[str] = Counter()
    parser_categories: Counter[str] = Counter()
    parser_episodes = 0
    invalid_action = 0
    repeated_events = 0
    repeated_episodes = 0
    tool_events = 0
    tool_episodes = 0
    recoverable = 0
    unrecoverable = 0
    limited_tool_errors = 0
    limited_repeats = 0
    limited_parser = 0
    other_tool_errors = 0
    other_repeats = 0
    other_parser = 0
    unavailable_turns = 0
    turns_at_cap = 0
    step_hits = 0
    token_sources: set[str] = set()
    for episode in episodes:
        steps = _model_steps(episode)
        tools = _tool_steps(episode)
        reason = _text(episode.get("termination_reason"), "termination_reason")
        termination[reason] += 1
        turn_count = len(steps)
        if turn_count > step_limit or (
            reason == "step_limit" and turn_count != step_limit
        ):
            raise ProfileError("configured step_limit does not match the capture")
        if reason == "step_limit":
            step_hits += 1
        model_turns.append(turn_count)
        tool_calls.append(len(tools))
        if reason == "invalid_action":
            invalid_action += 1
        episode_tokens = 0
        episode_tokens_known = True
        parser_count = 0
        for index, step in enumerate(steps, start=1):
            count, source = _read_generated_tokens(step)
            if source is not None:
                token_sources.add(source)
            if count is None:
                unavailable_turns += 1
                episode_tokens_known = False
            else:
                if count > max_tokens:
                    raise ProfileError(
                        "generated tokens exceed the configured max_tokens"
                    )
                tokens_per_turn.append(count)
                episode_tokens += count
                if count == max_tokens:
                    turns_at_cap += 1
            category = _parser_category(_text(step.get("output_text"), "output"))
            if category is not None:
                parser_categories[category] += 1
                parser_count += 1
            if _is_completion(step, tools):
                completion_turns.append(index)
        if episode_tokens_known:
            tokens_per_episode.append(episode_tokens)
        if parser_count:
            parser_episodes += 1
        if reason == "step_limit":
            limited_parser += parser_count
        else:
            other_parser += parser_count
        repeats = _repeated_actions(tools)
        repeated_events += repeats
        if repeats:
            repeated_episodes += 1
        if reason == "step_limit":
            limited_repeats += repeats
        else:
            other_repeats += repeats
        tool_here = 0
        for tool in tools:
            error = tool.get("error")
            if not isinstance(error, Mapping):
                continue
            tool_events += 1
            tool_here += 1
            if error.get("recoverable") is True:
                recoverable += 1
            else:
                unrecoverable += 1
        if tool_here:
            tool_episodes += 1
        if reason == "step_limit":
            limited_tool_errors += tool_here
        else:
            other_tool_errors += tool_here
    episode_count = len(episodes)
    return {
        "informs_execute_limits": True,
        "episode_count": episode_count,
        "pair_count": len(records),
        "roles": ["reference", "candidate"],
        "generated_token_source": _token_source_label(token_sources),
        "model_turns": _stats(model_turns),
        "tool_calls": _stats(tool_calls),
        "generated_tokens_per_turn": _stats(
            tokens_per_turn,
            at_or_above=_token_cuts((32, 64, 128, 256, 512, max_tokens)),
        ),
        "generated_tokens_per_episode": _stats(
            tokens_per_episode,
            at_or_above=(256, 512, 1024, 2048, 4096),
        ),
        "generated_token_unavailable_turns": unavailable_turns,
        "turns_at_configured_max_tokens": turns_at_cap,
        "step_limit_hit_count": step_hits,
        "step_limit_hit_fraction": _fraction(step_hits, episode_count),
        "termination_counts": _counter(termination),
        "completion_turn": {
            **_stats(completion_turns),
            "definition": "1-based model turn that calls supervisor.complete_task",
        },
        "errors": {
            "parser": {
                "event_count": sum(parser_categories.values()),
                "episode_count": parser_episodes,
                "categories": _counter(parser_categories),
            },
            "invalid_action": {"episode_count": invalid_action},
            "repeated_action": {
                "definition": "consecutive tool calls with equal action text",
                "event_count": repeated_events,
                "episode_count": repeated_episodes,
            },
            "tool": {
                "event_count": tool_events,
                "episode_count": tool_episodes,
                "recoverable_count": recoverable,
                "unrecoverable_count": unrecoverable,
            },
            "on_step_limit_episodes": {
                "parser_events": limited_parser,
                "repeated_action_events": limited_repeats,
                "tool_error_events": limited_tool_errors,
            },
            "on_other_episodes": {
                "parser_events": other_parser,
                "repeated_action_events": other_repeats,
                "tool_error_events": other_tool_errors,
            },
        },
        "paired": _paired(records),
        "by_behavior": _behavior_token_profile(records, max_tokens),
    }


def _behavior_token_profile(
    records: Sequence[Mapping[str, object]],
    max_tokens: int,
) -> dict[str, object]:
    """Token lengths for completed episodes and for step-limit episodes.

    A completed episode is one that terminates by calling
    ``supervisor.complete_task``. Valid tool-call generations are model
    turns on those episodes that parse as one API call. Parser failures
    and ``STOP`` are not valid tool calls. The configured cap is not
    changed here.
    """

    success_turns: list[int] = []
    success_episode_tokens: list[int] = []
    completion_tokens: list[int] = []
    valid_tokens: list[int] = []
    limit_turns: list[int] = []
    limit_episode_tokens: list[int] = []
    success_episodes = 0
    success_pairs = 0
    success_pairs_both = 0
    success_hit = False
    valid_hit = False
    completion_hit = False
    limit_episodes = 0
    limit_pairs = 0
    limit_hit = False
    for record in records:
        sides: list[str] = []
        for role in ("reference", "candidate"):
            episode = _episode(record, role)
            reason = _text(episode.get("termination_reason"), "termination_reason")
            steps = _model_steps(episode)
            tools = _tool_steps(episode)
            turn_tokens: list[int] = []
            valid: list[int] = []
            completions: list[int] = []
            known = True
            for step in steps:
                count = _generated_tokens(step)
                if count is None:
                    known = False
                else:
                    if count > max_tokens:
                        raise ProfileError(
                            "generated tokens exceed the configured max_tokens"
                        )
                    turn_tokens.append(count)
                text = _text(step.get("output_text"), "output")
                if count is not None and _is_valid_tool_call(text):
                    valid.append(count)
                if count is not None and _is_completion(step, tools):
                    completions.append(count)
            if reason == "appworld_completed":
                if not completions:
                    raise ProfileError(
                        "completed episode has no complete_task generation"
                    )
                sides.append("success")
                success_episodes += 1
                success_turns.extend(turn_tokens)
                if known:
                    success_episode_tokens.append(sum(turn_tokens))
                completion_tokens.append(completions[-1])
                valid_tokens.extend(valid)
                if any(count == max_tokens for count in turn_tokens):
                    success_hit = True
                if any(count == max_tokens for count in valid):
                    valid_hit = True
                if completions[-1] == max_tokens:
                    completion_hit = True
            elif reason == "step_limit":
                sides.append("step_limit")
                limit_episodes += 1
                limit_turns.extend(turn_tokens)
                if known:
                    limit_episode_tokens.append(sum(turn_tokens))
                if any(count == max_tokens for count in turn_tokens):
                    limit_hit = True
            else:
                sides.append("other")
        if sides.count("success") == 2:
            success_pairs_both += 1
        if "success" in sides:
            success_pairs += 1
        if "step_limit" in sides:
            limit_pairs += 1
    return {
        "successful_complete_task": {
            "episode_count": success_episodes,
            "pair_count": success_pairs,
            "pairs_completed_on_both_sides": success_pairs_both,
            "generated_tokens_per_turn": _stats(
                success_turns,
                at_or_above=_TOKEN_TAILS,
            ),
            "generated_tokens_per_episode": _stats(success_episode_tokens),
            "complete_task_generation_tokens": _stats(completion_tokens),
            "valid_tool_call_generation_tokens": _stats(valid_tokens),
            "valid_tool_call_max_tokens": None if not valid_tokens else max(valid_tokens),
            "any_generation_hit_configured_max_tokens": success_hit,
            "valid_tool_call_hit_configured_max_tokens": valid_hit,
            "complete_task_hit_configured_max_tokens": completion_hit,
        },
        "step_limit": {
            "episode_count": limit_episodes,
            "pair_count": limit_pairs,
            "generated_tokens_per_turn": _stats(
                limit_turns,
                at_or_above=_TOKEN_TAILS,
            ),
            "generated_tokens_per_episode": _stats(limit_episode_tokens),
            "any_generation_hit_configured_max_tokens": limit_hit,
        },
    }


def _plan_profile(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    tokens: list[int] = []
    statuses: Counter[str] = Counter()
    token_sources: set[str] = set()
    for record in records:
        for role in ("reference", "candidate"):
            episode = _episode(record, role)
            steps = _model_steps(episode)
            if len(steps) != 1:
                raise ProfileError("plan episode does not have one model step")
            count, source = _read_generated_tokens(steps[0])
            if count is None:
                raise ProfileError("plan generated tokens are unavailable")
            if source is not None:
                token_sources.add(source)
            tokens.append(count)
        forced = record.get("teacher_forced_plan_kl")
        if not isinstance(forced, Mapping):
            statuses["teacher_force_unavailable"] += 1
            continue
        status = forced.get("status")
        if not isinstance(status, str) or status == "":
            raise ProfileError("teacher-forced status is missing")
        statuses[status] += 1
    return {
        "informs_execute_limits": False,
        "episode_count": len(records) * 2,
        "pair_count": len(records),
        "generated_token_source": _token_source_label(token_sources),
        "generated_tokens": _stats(tokens),
        "teacher_forced_status": _counter(statuses),
    }


def _paired(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    differences: list[int] = []
    disagreements = 0
    missing = 0
    identical = 0
    for record in records:
        flag = record.get("evaluator_disagreement")
        if flag is None:
            missing += 1
        elif flag is True:
            disagreements += 1
        elif flag is not False:
            raise ProfileError("evaluator disagreement must be a bool or null")
        trajectory = record.get("trajectory")
        if not isinstance(trajectory, Mapping):
            raise ProfileError("trajectory is missing")
        difference = trajectory.get("length_difference")
        if isinstance(difference, bool) or not isinstance(difference, int):
            raise ProfileError("trajectory length difference must be an int")
        differences.append(difference)
        if difference == 0 and trajectory.get("first_divergent_step") is None:
            identical += 1
    return {
        "pair_count": len(records),
        "disagreement_count": disagreements,
        "disagreement_fraction": _fraction(disagreements, len(records)),
        "missing_outcome_count": missing,
        "identical_trajectory_count": identical,
        "trajectory_length_difference": _stats(differences),
    }


def _token_cuts(cuts: Sequence[int]) -> tuple[int, ...]:
    chosen: list[int] = []
    for cut in cuts:
        if cut not in chosen:
            chosen.append(cut)
    return tuple(chosen)


def _stats(
    values: Sequence[int],
    *,
    at_or_above: Sequence[int] = (),
) -> dict[str, object]:
    payload: dict[str, object] = {
        "count": len(values),
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": None if not values else max(values),
        "counts": [
            {"value": value, "count": count}
            for value, count in sorted(Counter(values).items())
        ],
    }
    if at_or_above:
        payload["at_or_above"] = [
            {"threshold": cut, "count": sum(1 for value in values if value >= cut)}
            for cut in at_or_above
        ]
    return payload


def _percentile(values: Sequence[int], percent: int) -> float | int | None:
    if percent not in _PERCENTILES:
        raise ProfileError("percentile is not in the reported set")
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (percent / 100) * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    interpolated = ordered[low] * (1 - weight) + ordered[high] * weight
    rounded = round(interpolated, 6)
    if rounded == int(rounded):
        return int(rounded)
    return rounded


def _generated_tokens(step: Mapping[str, object]) -> int | None:
    count, _source = _read_generated_tokens(step)
    return count


def _read_generated_tokens(
    step: Mapping[str, object],
) -> tuple[int | None, str | None]:
    if "generated_token_count" in step:
        count = step.get("generated_token_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ProfileError(
                "generated_token_count must be a non-negative integer"
            )
        return count, "generated_token_count"
    positions = step.get("top_k_logprobs")
    if isinstance(positions, list):
        return len(positions), "top_k_logprob_positions"
    return None, None


def _token_source_label(sources: set[str]) -> str:
    if not sources:
        return "unavailable"
    return "+".join(sorted(sources))


def _parser_category(text: str) -> str | None:
    if text == "STOP" or text.startswith("STOP\n"):
        return None
    try:
        parse_model_output(text)
    except ActionRejected as error:
        message = str(error)
        if message == "empty action":
            return "empty_action"
        if message.startswith("action is not valid Python"):
            return "syntax_error"
        if message == "action must be a single expression":
            return "not_single_expression"
        if message == "action must be a single call expression":
            return "not_single_call"
        if message == "action must contain exactly one call":
            return "multiple_calls"
        if message == "action must call apis.<app>.<api>":
            return "not_apis_call"
        if message == "app and api names must be identifiers":
            return "invalid_identifier"
        return "other"
    return None


def _is_valid_tool_call(text: str) -> bool:
    if text == "STOP" or text.startswith("STOP\n"):
        return False
    return _parser_category(text) is None


def _is_completion(step: Mapping[str, object], tools: Sequence[Mapping[str, object]]) -> bool:
    index = step.get("index")
    if isinstance(index, bool) or not isinstance(index, int):
        return False
    for tool in tools:
        if tool.get("index") != index + 1:
            continue
        if tool.get("app_name") != "supervisor" or tool.get("api_name") != "complete_task":
            return False
        return not isinstance(tool.get("error"), Mapping)
    return False


def _repeated_actions(tools: Sequence[Mapping[str, object]]) -> int:
    previous: str | None = None
    repeats = 0
    for tool in tools:
        action = tool.get("action")
        if not isinstance(action, str):
            raise ProfileError("tool action is missing")
        if previous is not None and action == previous:
            repeats += 1
        previous = action
    return repeats


def _capture(document: Mapping[str, object]) -> Mapping[str, object]:
    capture = document.get("capture", document)
    if not isinstance(capture, Mapping):
        raise ProfileError("capture must be a mapping")
    return capture


def _records(capture: Mapping[str, object]) -> list[Mapping[str, object]]:
    records = capture.get("records")
    if not isinstance(records, list) or not records:
        raise ProfileError("capture records must be a non-empty list")
    chosen: list[Mapping[str, object]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ProfileError("capture record must be a mapping")
        mode = record.get("mode")
        if mode not in {"plan", "execute"}:
            raise ProfileError("mode must be plan or execute")
        chosen.append(record)
    return chosen


def _identity(
    records: Sequence[Mapping[str, object]],
) -> tuple[str, str, str, str]:
    splits: set[str] = set()
    configurations: set[str] = set()
    task_sets: set[str] = set()
    commits: set[str] = set()
    for record in records:
        for role in ("reference", "candidate"):
            episode = _episode(record, role)
            task = episode.get("task")
            run = episode.get("run")
            if not isinstance(task, Mapping) or not isinstance(run, Mapping):
                raise ProfileError("episode identity is missing")
            split = task.get("split")
            if split in _CLOSED_SPLITS:
                raise ProfileError("closed splits are not profiled")
            if not isinstance(split, str):
                raise ProfileError("split is missing")
            splits.add(split)
            configurations.add(_sha(run.get("configuration_hash"), "configuration_hash"))
            task_sets.add(_sha(run.get("task_set_hash"), "task_set_hash"))
            commit = run.get("git_commit")
            if not isinstance(commit, str) or len(commit) != 40:
                raise ProfileError("git_commit is missing")
            commits.add(commit)
    if len(splits) != 1 or len(configurations) != 1 or len(task_sets) != 1 or len(commits) != 1:
        raise ProfileError("capture identity is not uniform")
    return splits.pop(), configurations.pop(), task_sets.pop(), commits.pop()


def _population(records: Sequence[Mapping[str, object]]) -> tuple[set[str], set[str]]:
    tasks: set[str] = set()
    scenarios: set[str] = set()
    for record in records:
        episode = _episode(record, "reference")
        task = episode.get("task")
        if not isinstance(task, Mapping):
            raise ProfileError("task reference is missing")
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or task_id == "":
            raise ProfileError("task id is missing")
        tasks.add(task_id)
        scenario_id = task.get("scenario_id")
        if isinstance(scenario_id, str) and scenario_id != "":
            scenarios.add(scenario_id)
    return tasks, scenarios


def _episode(record: Mapping[str, object], role: str) -> Mapping[str, object]:
    pair = record.get("pair")
    if not isinstance(pair, Mapping):
        raise ProfileError("pair is missing")
    episode = pair.get(role)
    if not isinstance(episode, Mapping):
        raise ProfileError("episode is missing")
    return episode


def _model_steps(episode: Mapping[str, object]) -> list[Mapping[str, object]]:
    return _steps(episode.get("model_steps"), "model_steps")


def _tool_steps(episode: Mapping[str, object]) -> list[Mapping[str, object]]:
    return _steps(episode.get("tool_steps"), "tool_steps")


def _steps(value: object, name: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        raise ProfileError(f"{name} must be a list")
    steps: list[Mapping[str, object]] = []
    for step in value:
        if not isinstance(step, Mapping):
            raise ProfileError(f"{name} must contain mappings")
        steps.append(step)
    return steps


def _modes(value: object) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ProfileError("modes must be a non-empty list")
    modes: list[str] = []
    for mode in value:
        if mode not in {"plan", "execute"}:
            raise ProfileError("mode must be plan or execute")
        modes.append(mode)
    return modes


def _counter(counts: Counter[str]) -> dict[str, int]:
    return {key: counts[key] for key in sorted(counts)}


def _fraction(numerator: int, denominator: int) -> float | int | None:
    if denominator == 0:
        return None
    value = numerator / denominator
    rounded = round(value, 6)
    if rounded == int(rounded):
        return int(rounded)
    return rounded


def _percentile_line(name: str, stats: Mapping[str, object]) -> str:
    return (
        f"{name}: "
        f"p50={stats['p50']} p90={stats['p90']} p95={stats['p95']} "
        f"p99={stats['p99']} max={stats['max']}"
    )


def _behavior_lines(value: object) -> list[str]:
    behavior = _mapping(value, "by_behavior")
    success = _mapping(behavior.get("successful_complete_task"), "successful")
    limited = _mapping(behavior.get("step_limit"), "step_limit")
    success_turns = _mapping(
        success.get("generated_tokens_per_turn"),
        "successful turns",
    )
    success_episodes = _mapping(
        success.get("generated_tokens_per_episode"),
        "successful episodes",
    )
    completion = _mapping(
        success.get("complete_task_generation_tokens"),
        "complete_task tokens",
    )
    limited_turns = _mapping(
        limited.get("generated_tokens_per_turn"),
        "step_limit turns",
    )
    return [
        (
            "successful_complete_task_episodes: "
            f"{success['episode_count']} "
            f"pairs={success['pair_count']} "
            f"both_sides={success['pairs_completed_on_both_sides']}"
        ),
        _percentile_line("successful_tokens_per_turn", success_turns),
        "successful_tokens_per_turn_at_or_above: "
        + _threshold_text(success_turns.get("at_or_above")),
        _percentile_line("successful_tokens_per_episode", success_episodes),
        _percentile_line("complete_task_generation_tokens", completion),
        (
            "valid_tool_call_generation_max: "
            f"{success['valid_tool_call_max_tokens']}"
        ),
        (
            "successful_generation_hit_configured_max_tokens: "
            f"{str(success['any_generation_hit_configured_max_tokens']).lower()}"
        ),
        (
            "valid_tool_call_hit_configured_max_tokens: "
            f"{str(success['valid_tool_call_hit_configured_max_tokens']).lower()}"
        ),
        (
            "complete_task_hit_configured_max_tokens: "
            f"{str(success['complete_task_hit_configured_max_tokens']).lower()}"
        ),
        (
            "step_limit_episodes: "
            f"{limited['episode_count']} pairs={limited['pair_count']}"
        ),
        "step_limit_tokens_per_turn_at_or_above: "
        + _threshold_text(limited_turns.get("at_or_above")),
        (
            "step_limit_generation_hit_configured_max_tokens: "
            f"{str(limited['any_generation_hit_configured_max_tokens']).lower()}"
        ),
    ]


def _threshold_text(value: object) -> str:
    if not isinstance(value, list) or not value:
        return "none"
    return " ".join(
        f">={item['threshold']}:{item['count']}"
        for item in value
        if isinstance(item, Mapping)
    )


def _count_text(value: object) -> str:
    if not isinstance(value, Mapping) or not value:
        return "none"
    return " ".join(f"{key}={count}" for key, count in value.items())


def _assert_public(profile: Mapping[str, object]) -> None:
    try:
        assert_public_payload(profile)
    except RecordError as error:
        raise ProfileError("profile is not public") from error
    _reject_text(profile)


def _reject_text(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in PROTECTED_FIELDS:
                raise ProfileError("profile is not public")
            if not isinstance(key, str) or len(key) > 80 or "\n" in key:
                raise ProfileError("profile field name is not public")
            _reject_text(item)
        return
    if isinstance(value, list):
        for item in value:
            _reject_text(item)
        return
    if isinstance(value, str) and (len(value) > 180 or "\n" in value):
        raise ProfileError("profile text is not an aggregate field")


def _capture_label_ok(value: str) -> bool:
    return value.replace("_", "").isalnum() and value[0].isalpha()


def _label(value: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 64 or not _capture_label_ok(value):
        raise ProfileError("capture_label must be a short token")
    return value


def _positive(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ProfileError(f"{name} must be a positive integer")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ProfileError(f"{name} must be a string")
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ProfileError(f"{name} must be a sha256 hex string")
    return value


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProfileError(f"{name} is missing")
    return value
