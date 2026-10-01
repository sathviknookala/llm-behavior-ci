import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from llm_behavior_ci.experiments.limit_profile import (
    ProfileError,
    format_summary,
    profile_capture,
    write_public_profile,
)


def _positions(count: int) -> list[list[dict[str, object]]]:
    return [[{"token_id": 1, "logprob": -0.1, "rank": 0}] for _ in range(count)]


def _model(index: int, text: str, tokens: int) -> dict[str, object]:
    return {
        "index": index,
        "output_text": text,
        "top_k_logprobs": _positions(tokens),
    }


def _with_generated_token_counts(document: dict[str, object]) -> dict[str, object]:
    copied = json.loads(json.dumps(document))
    records = copied["capture"]["records"]
    for record in records:
        pair = record["pair"]
        for role in ("reference", "candidate"):
            for step in pair[role]["model_steps"]:
                positions = step["top_k_logprobs"]
                step["generated_token_count"] = len(positions)
                step["top_k_logprobs"] = []
    return copied


def _tool(
    index: int,
    action: str,
    *,
    app_name: str = "gmail",
    api_name: str = "search",
    error: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "index": index,
        "action": action,
        "app_name": app_name,
        "api_name": api_name,
        "error": error,
    }


def _episode(
    reason: str,
    models: list[dict[str, object]],
    tools: list[dict[str, object]],
    *,
    success: bool | None = True,
) -> dict[str, object]:
    return {
        "termination_reason": reason,
        "model_steps": models,
        "tool_steps": tools,
        "plan_text": "SECRET-PLAN-TEXT",
        "task": {
            "task_id": "task-secret-id",
            "scenario_id": "scenario-secret-id",
            "split": "train",
        },
        "run": {
            "configuration_hash": "a" * 64,
            "task_set_hash": "b" * 64,
            "git_commit": "c" * 40,
        },
        "evaluator_outcome": None if success is None else {"success": success},
    }


def _document() -> dict[str, object]:
    complete = "apis.supervisor.complete_task()"
    search = "apis.gmail.search(query='SECRET-ACTION')"
    reference_a = _episode(
        "appworld_completed",
        [_model(0, complete, 4)],
        [_tool(1, complete, app_name="supervisor", api_name="complete_task")],
    )
    candidate_a = _episode(
        "appworld_completed",
        [_model(0, search, 4), _model(2, complete, 10)],
        [
            _tool(1, search),
            _tool(3, complete, app_name="supervisor", api_name="complete_task"),
        ],
    )
    reference_b = _episode(
        "step_limit",
        [_model(0, "hello", 2), _model(1, search, 2), _model(3, search, 8)],
        [
            _tool(2, search),
            _tool(
                4,
                search,
                error={
                    "message": "SECRET-TOOL-ERROR",
                    "recoverable": True,
                    "source": "tool",
                    "step_index": 4,
                },
            ),
        ],
        success=True,
    )
    candidate_b = _episode(
        "agent_stopped",
        [_model(0, "STOP", 3)],
        [],
        success=False,
    )
    plan_reference = _episode("plan_emitted", [_model(0, "SECRET-PLAN-TEXT", 5)], [])
    plan_candidate = _episode("plan_emitted", [_model(0, "SECRET-PLAN-TEXT", 7)], [])
    return {
        "visibility": "local",
        "capture": {
            "repetitions": 2,
            "concurrency": 1,
            "modes": ["plan", "execute"],
            "records": [
                {
                    "mode": "execute",
                    "evaluator_disagreement": False,
                    "trajectory": {
                        "length_difference": 1,
                        "first_divergent_step": 0,
                    },
                    "pair": {"reference": reference_a, "candidate": candidate_a},
                },
                {
                    "mode": "execute",
                    "evaluator_disagreement": True,
                    "trajectory": {
                        "length_difference": -2,
                        "first_divergent_step": 0,
                    },
                    "pair": {"reference": reference_b, "candidate": candidate_b},
                },
                {
                    "mode": "plan",
                    "teacher_forced_plan_kl": {"status": "scored"},
                    "pair": {"reference": plan_reference, "candidate": plan_candidate},
                },
            ],
        },
    }


class LimitProfileTests(unittest.TestCase):
    def test_execute_percentiles_errors_and_plan_status_stay_aggregate(self) -> None:
        profile = profile_capture(
            _document(),
            capture_label="train_c1_off",
            step_limit=3,
            max_tokens=10,
        )
        execute = profile["execute"]
        self.assertEqual(execute["episode_count"], 4)
        self.assertEqual(execute["pair_count"], 2)
        self.assertEqual(execute["model_turns"]["p50"], 1.5)
        self.assertEqual(execute["model_turns"]["max"], 3)
        self.assertEqual(execute["tool_calls"]["p50"], 1.5)
        self.assertEqual(execute["tool_calls"]["max"], 2)
        self.assertEqual(execute["generated_tokens_per_turn"]["p50"], 4)
        self.assertEqual(execute["generated_tokens_per_turn"]["max"], 10)
        self.assertEqual(execute["generated_tokens_per_episode"]["p50"], 8)
        self.assertEqual(execute["generated_tokens_per_episode"]["max"], 14)
        self.assertEqual(execute["step_limit_hit_count"], 1)
        self.assertEqual(execute["step_limit_hit_fraction"], 0.25)
        self.assertEqual(execute["turns_at_configured_max_tokens"], 1)
        self.assertEqual(
            execute["generated_tokens_per_turn"]["at_or_above"][-1],
            {"threshold": 10, "count": 1},
        )
        self.assertEqual(
            execute["errors"]["on_step_limit_episodes"]["tool_error_events"],
            1,
        )
        self.assertEqual(execute["errors"]["on_other_episodes"]["tool_error_events"], 0)
        self.assertEqual(execute["termination_counts"]["step_limit"], 1)
        self.assertEqual(execute["termination_counts"]["appworld_completed"], 2)
        self.assertEqual(execute["completion_turn"]["count"], 2)
        self.assertEqual(execute["completion_turn"]["p50"], 1.5)
        self.assertEqual(execute["completion_turn"]["max"], 2)
        behavior = execute["by_behavior"]
        success = behavior["successful_complete_task"]
        limited = behavior["step_limit"]
        self.assertEqual(success["episode_count"], 2)
        self.assertEqual(success["pair_count"], 1)
        self.assertEqual(success["pairs_completed_on_both_sides"], 1)
        self.assertEqual(success["generated_tokens_per_turn"]["p50"], 4)
        self.assertEqual(success["generated_tokens_per_turn"]["max"], 10)
        self.assertEqual(success["generated_tokens_per_episode"]["p50"], 9)
        self.assertEqual(success["generated_tokens_per_episode"]["max"], 14)
        self.assertEqual(success["complete_task_generation_tokens"]["p50"], 7)
        self.assertEqual(success["complete_task_generation_tokens"]["max"], 10)
        self.assertEqual(success["valid_tool_call_max_tokens"], 10)
        self.assertTrue(success["any_generation_hit_configured_max_tokens"])
        self.assertTrue(success["complete_task_hit_configured_max_tokens"])
        self.assertEqual(
            {item["threshold"]: item["count"] for item in success["generated_tokens_per_turn"]["at_or_above"]},
            {64: 0, 96: 0, 128: 0, 192: 0, 256: 0, 1024: 0},
        )
        self.assertEqual(limited["episode_count"], 1)
        self.assertEqual(limited["pair_count"], 1)
        self.assertFalse(limited["any_generation_hit_configured_max_tokens"])
        self.assertEqual(
            {item["threshold"]: item["count"] for item in limited["generated_tokens_per_turn"]["at_or_above"]},
            {64: 0, 96: 0, 128: 0, 192: 0, 256: 0, 1024: 0},
        )
        self.assertEqual(execute["errors"]["parser"]["event_count"], 1)
        self.assertEqual(
            execute["errors"]["parser"]["categories"]["not_single_call"],
            1,
        )
        self.assertEqual(execute["errors"]["invalid_action"]["episode_count"], 0)
        self.assertEqual(execute["errors"]["repeated_action"]["event_count"], 1)
        self.assertEqual(execute["errors"]["tool"]["recoverable_count"], 1)
        self.assertEqual(execute["paired"]["disagreement_count"], 1)
        self.assertEqual(execute["paired"]["trajectory_length_difference"]["max"], 1)
        self.assertEqual(execute["paired"]["trajectory_length_difference"]["p50"], -0.5)
        plan = profile["plan"]
        self.assertFalse(plan["informs_execute_limits"])
        self.assertTrue(execute["informs_execute_limits"])
        self.assertEqual(plan["generated_tokens"]["p50"], 6)
        self.assertEqual(plan["generated_tokens"]["max"], 7)
        self.assertEqual(plan["teacher_forced_status"]["scored"], 1)
        self.assertEqual(profile["task_count"], 1)
        self.assertEqual(profile["scenario_count"], 1)
        rendered = json.dumps(profile)
        summary = format_summary(profile)
        for secret in (
            "task-secret-id",
            "scenario-secret-id",
            "SECRET-PLAN-TEXT",
            "SECRET-ACTION",
            "SECRET-TOOL-ERROR",
        ):
            self.assertNotIn(secret, rendered)
            self.assertNotIn(secret, summary)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            write_public_profile(profile, path)
            self.assertNotIn("SECRET-ACTION", path.read_text(encoding="utf-8"))

    def test_closed_split_and_limit_mismatch_are_refused(self) -> None:
        document = _document()
        records = document["capture"]["records"]
        records[0]["pair"]["reference"]["task"]["split"] = "test_normal"
        with self.assertRaisesRegex(ProfileError, "closed"):
            profile_capture(
                document,
                capture_label="train_c1_off",
                step_limit=3,
                max_tokens=8,
            )
        document = _document()
        with self.assertRaisesRegex(ProfileError, "max_tokens"):
            profile_capture(
                document,
                capture_label="train_c1_off",
                step_limit=3,
                max_tokens=7,
            )

    def test_new_and_old_captures_share_token_percentiles_and_cap_hits(self) -> None:
        old = profile_capture(
            _document(),
            capture_label="train_c1_off",
            step_limit=3,
            max_tokens=10,
        )
        new = profile_capture(
            _with_generated_token_counts(_document()),
            capture_label="train_c1_off",
            step_limit=3,
            max_tokens=10,
        )
        for profile, source in (
            (old, "top_k_logprob_positions"),
            (new, "generated_token_count"),
        ):
            execute = profile["execute"]
            turns = execute["generated_tokens_per_turn"]
            episodes = execute["generated_tokens_per_episode"]
            self.assertEqual(execute["generated_token_source"], source)
            self.assertEqual(turns["p50"], 4)
            self.assertEqual(turns["p90"], 8.8)
            self.assertEqual(turns["p95"], 9.4)
            self.assertEqual(turns["p99"], 9.88)
            self.assertEqual(turns["max"], 10)
            self.assertEqual(episodes["p50"], 8)
            self.assertEqual(episodes["p90"], 13.4)
            self.assertEqual(episodes["p95"], 13.7)
            self.assertEqual(episodes["p99"], 13.94)
            self.assertEqual(episodes["max"], 14)
            self.assertEqual(execute["turns_at_configured_max_tokens"], 1)
            self.assertEqual(profile["plan"]["generated_token_source"], source)
            self.assertEqual(profile["plan"]["generated_tokens"]["p50"], 6)
            self.assertEqual(profile["plan"]["generated_tokens"]["max"], 7)

    def test_generated_token_count_wins_over_logprob_length(self) -> None:
        document = _document()
        step = document["capture"]["records"][0]["pair"]["reference"]["model_steps"][0]
        step["generated_token_count"] = 4
        step["top_k_logprobs"] = _positions(1)
        profile = profile_capture(
            document,
            capture_label="train_c1_off",
            step_limit=3,
            max_tokens=10,
        )
        execute = profile["execute"]
        self.assertEqual(
            execute["generated_token_source"],
            "generated_token_count+top_k_logprob_positions",
        )
        self.assertEqual(execute["generated_tokens_per_turn"]["p50"], 4)
        self.assertEqual(execute["generated_tokens_per_turn"]["max"], 10)
        self.assertEqual(execute["turns_at_configured_max_tokens"], 1)

    def test_bare_command_exits_2(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        completed = subprocess.run(
            [sys.executable, "scripts/evaluation/profile_capture_limits.py"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(completed.returncode, 2)
