import json
import unittest
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace

from llm_behavior_ci.config import (
    MISSING_HASHED_LEAF,
    ConfigError,
    RunConfiguration,
    hashed_values,
    run_configuration_hash,
)
from llm_behavior_ci.runtime.agent import SmolagentsVLLMAgent
from llm_behavior_ci.runtime.appworld import (
    LiveAppWorldSession,
    _spotify_capability_allows,
)
from llm_behavior_ci.runtime.clock import wall_now
from llm_behavior_ci.runtime.episode import (
    RuntimeDependencies,
    build_runtime,
    is_live_runtime,
)

_ROOT = Path(__file__).resolve().parents[2]
_PROFILE = "spotify_capability_v1"

_ALLOWED = (
    "apis.spotify.login(...)",
    "apis.supervisor.show_profile()",
    "apis.supervisor.show_account_passwords()",
    "apis.supervisor.complete_task()",
    'apis.api_docs.show_api_descriptions(app_name="spotify")',
    'apis.api_docs.show_api_doc(app_name="spotify", api_name="login")',
)
_REJECTED = (
    "apis.supervisor.show_payment_cards()",
    "apis.venmo.search_users(...)",
    "apis.todoist.show_tasks(...)",
    'apis.api_docs.show_api_descriptions(app_name="venmo")',
    'apis.api_docs.show_api_doc(app_name="phone", api_name="...")',
)


def _documentation() -> dict[str, object]:
    def api(description: str) -> dict[str, object]:
        return {"description": description, "parameters": []}

    return {
        "spotify": {
            "login": api("Sign in."),
            "search_tracks": api("Find songs."),
        },
        "venmo": {"search_users": api("Find people.")},
        "todoist": {"show_tasks": api("List chores.")},
        "phone": {"send_message": api("Send a text.")},
        "supervisor": {
            "show_profile": api("Show the profile."),
            "show_account_passwords": api("Show passwords."),
            "complete_task": api("Mark the task done."),
            "show_payment_cards": api("Show cards."),
        },
        "api_docs": {
            "show_api_descriptions": api("List one app."),
            "show_api_doc": api("Show one API."),
            "show_app_descriptions": api("List every app."),
        },
    }


class _World:
    def __init__(self, documentation: object) -> None:
        self.task = SimpleNamespace(instruction="instruction", api_docs=documentation)
        self.actions: list[str] = []

    def execute(self, action: str) -> str:
        self.actions.append(action)
        return "ok"

    def close(self) -> None:
        return None


def _run_configuration(model_file: str) -> RunConfiguration:
    document = json.loads(
        (_ROOT / "configs/models" / model_file).read_text(encoding="utf-8")
    )
    document["task"] = {
        "appworld_version": "0.1.3.post1",
        "split": "train",
        "selection_rule": "deterministic_sample",
        "selection_seed": 17,
        "task_count": 1,
        "task_set_hash": "c" * 64,
    }
    document["run_seed"] = 17
    document["git_commit"] = "a" * 40
    document["protocol_hash"] = None
    return RunConfiguration.from_dict(document)


def _names(rendered: str, app_name: str) -> set[str]:
    prefix = f"{app_name}."
    return {
        line.split(":", 1)[0]
        for line in rendered.splitlines()
        if line.startswith(prefix)
    }


class SpotifyCapabilityPolicyTests(unittest.TestCase):
    def test_allowlist_matches_the_spotify_profile(self) -> None:
        for action in _ALLOWED:
            with self.subTest(action=action):
                self.assertTrue(_spotify_capability_allows(action))
        for action in _REJECTED:
            with self.subTest(action=action):
                self.assertFalse(_spotify_capability_allows(action))
        self.assertFalse(_spotify_capability_allows("apis.api_docs.show_api_descriptions()"))
        self.assertFalse(_spotify_capability_allows("not a call"))

    def test_capability_session_rejects_out_of_profile_calls_before_appworld(self) -> None:
        world = _World(_documentation())
        session = LiveAppWorldSession(
            "task-capability",
            opener=lambda task_id: world,
            tool_access_profile=_PROFILE,
        )
        self.addCleanup(session.close)
        for action in _ALLOWED:
            result = session.execute(action)
            self.assertIsNone(result.error_message, action)
            self.assertIsNotNone(result.output_text, action)
        for action in _REJECTED:
            result = session.execute(action)
            self.assertIsNone(result.output_text, action)
            self.assertEqual(
                result.error_message,
                "API is outside the configured capability profile",
                action,
            )
            self.assertTrue(result.recoverable, action)
        self.assertEqual(len(world.actions), len(_ALLOWED))
        for action in _REJECTED:
            self.assertNotIn(action, "\n".join(world.actions))

    def test_filtered_documentation_keeps_only_the_approved_surface(self) -> None:
        world = _World(_documentation())
        session = LiveAppWorldSession(
            "task-docs",
            opener=lambda task_id: world,
            tool_access_profile=_PROFILE,
        )
        self.addCleanup(session.close)
        rendered = session.context().api_documentation
        self.assertEqual(
            {line.split(".", 1)[0] for line in rendered.splitlines() if line},
            {"api_docs", "spotify", "supervisor"},
        )
        self.assertEqual(
            _names(rendered, "spotify"),
            {"spotify.login", "spotify.search_tracks"},
        )
        self.assertEqual(
            _names(rendered, "supervisor"),
            {
                "supervisor.show_profile",
                "supervisor.show_account_passwords",
                "supervisor.complete_task",
            },
        )
        self.assertEqual(
            _names(rendered, "api_docs"),
            {"api_docs.show_api_descriptions", "api_docs.show_api_doc"},
        )
        for absent in (
            "venmo",
            "todoist",
            "phone",
            "show_payment_cards",
            "show_app_descriptions",
        ):
            self.assertNotIn(absent, rendered)

    def test_unrestricted_session_keeps_every_app_and_executes_it(self) -> None:
        world = _World(_documentation())
        session = LiveAppWorldSession("task-open", opener=lambda task_id: world)
        self.addCleanup(session.close)
        rendered = session.context().api_documentation
        for present in ("venmo", "todoist", "phone", "show_payment_cards"):
            self.assertIn(present, rendered)
        for action in _REJECTED:
            result = session.execute(action)
            self.assertIsNone(result.error_message, action)
        self.assertEqual(len(world.actions), len(_REJECTED))


class SpotifyCapabilityConfigurationTests(unittest.TestCase):
    def test_profile_changes_the_configuration_hash_and_round_trips(self) -> None:
        production = _run_configuration("qwen3_4b_production.json")
        self.assertIsNone(production.agent.tool_access_profile)
        self.assertNotIn("tool_access_profile", production.agent.to_dict())
        self.assertIs(
            hashed_values(production)["agent.tool_access_profile"],
            MISSING_HASHED_LEAF,
        )
        changed = replace(
            production,
            agent=replace(production.agent, tool_access_profile=_PROFILE),
        )
        self.assertNotEqual(
            run_configuration_hash(changed),
            run_configuration_hash(production),
        )
        restored = RunConfiguration.from_dict(changed.to_dict())
        self.assertEqual(restored.agent.tool_access_profile, _PROFILE)
        self.assertEqual(run_configuration_hash(restored), run_configuration_hash(changed))
        with self.assertRaises(ConfigError):
            replace(production.agent, tool_access_profile="  ")

    def test_committed_configs_select_restriction_only_through_the_profile(self) -> None:
        production = _run_configuration("qwen3_4b_production.json")
        capability = _run_configuration("qwen3_4b_spotify_capability.json")
        self.assertIsNone(production.agent.tool_access_profile)
        self.assertEqual(capability.agent.execute_max_model_turns, 20)
        self.assertEqual(capability.agent.execute_turn_limit, 20)
        self.assertEqual(capability.agent.sampling.execute_max_tokens, 192)
        self.assertEqual(capability.agent.sampling.generation_max_tokens("execute"), 192)
        self.assertEqual(capability.agent.sampling.temperature, 0.0)
        self.assertEqual(capability.agent.sampling.seed, 17)
        self.assertEqual(capability.agent.tool_access_profile, _PROFILE)
        agent = SmolagentsVLLMAgent("http://127.0.0.1:9")
        agent.set_mode("execute")
        agent.begin(
            SimpleNamespace(
                task_id="task-1",
                instruction="instruction",
                api_documentation="docs",
            ),
            capability,
        )
        payload = agent.completion_payload(agent.messages())
        self.assertNotIn("logprobs", payload)
        self.assertNotIn("top_logprobs", payload)
        self.assertEqual(payload["max_tokens"], 192)
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["seed"], 17)

        open_runtime = build_runtime(production, "http://127.0.0.1:9", mode="execute")
        closed_runtime = build_runtime(capability, "http://127.0.0.1:9", mode="execute")
        self.assertTrue(is_live_runtime(open_runtime))
        self.assertTrue(is_live_runtime(closed_runtime))
        self.assertIsNone(open_runtime.session_factory.keywords["tool_access_profile"])
        self.assertEqual(
            closed_runtime.session_factory.keywords["tool_access_profile"],
            _PROFILE,
        )
        open_world = _World(_documentation())
        closed_world = _World(_documentation())
        open_session = open_runtime.session_factory(
            "task-open-runtime",
            opener=lambda task_id: open_world,
        )
        closed_session = closed_runtime.session_factory(
            "task-closed-runtime",
            opener=lambda task_id: closed_world,
        )
        self.addCleanup(open_session.close)
        self.addCleanup(closed_session.close)
        self.assertIsNone(open_session.execute(_REJECTED[1]).error_message)
        rejected = closed_session.execute(_REJECTED[1])
        self.assertEqual(
            rejected.error_message,
            "API is outside the configured capability profile",
        )
        self.assertEqual(closed_world.actions, [])
        self.assertEqual(len(open_world.actions), 1)
        spoofed = RuntimeDependencies(
            session_factory=partial(
                LiveAppWorldSession,
                opener=lambda task_id: closed_world,
                tool_access_profile=_PROFILE,
            ),
            agent=SmolagentsVLLMAgent("http://127.0.0.1:9"),
            clock=wall_now,
        )
        self.assertFalse(is_live_runtime(spoofed))


if __name__ == "__main__":
    unittest.main()
