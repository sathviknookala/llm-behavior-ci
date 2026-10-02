import importlib.util
import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from llm_behavior_ci.config import (
    MISSING_HASHED_LEAF,
    ConfigError,
    RunConfiguration,
    TaskConfiguration,
    hashed_values,
    new_run_identity,
    run_configuration_hash,
)
from llm_behavior_ci.records import TokenLogprob
from llm_behavior_ci.runtime.agent import AgentTurn
from llm_behavior_ci.runtime.appworld import LiveAppWorldSession, TaskContext
from llm_behavior_ci.runtime.episode import (
    EpisodeRejected,
    RuntimeDependencies,
    RuntimeUnavailable,
    build_runtime,
    is_live_runtime,
    run_episode,
    run_pair,
)

_PROFILE = "spotify_authenticated_v1"
_CAPABILITY = "spotify_capability_v1"
_SECRET = "runtime-secret"
_FAKE = "fake-token"
_EMAIL = "person@example.com"
_SPOTIFY_PASSWORD = "spotify-secret"
_VENMO_PASSWORD = "venmo-secret"
_LOGPROBS = ((TokenLogprob(token_id=7, logprob=-0.5, rank=0),),)
_TASK_HASH = "a20fe52d28164e1d458266331c242277788d2ed0af29b054b7df926345db3a04"


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parameter(name: str) -> dict[str, object]:
    return {"name": name, "type": "string", "required": True}


def _api(description: str, parameters: list[object]) -> dict[str, object]:
    return {"description": description, "parameters": parameters}


def _documentation() -> dict[str, object]:
    return {
        "spotify": {
            "login": _api("Sign in.", [_parameter("username"), _parameter("password")]),
            "signup": _api("Sign up.", []),
            "create_playlist": _api(
                "Create a playlist.",
                [_parameter("access_token"), _parameter("title")],
            ),
            "search_songs": _api("Search songs.", [_parameter("query")]),
        },
        "venmo": {"search_users": _api("Find people.", [])},
        "supervisor": {
            "show_profile": _api("Show the profile.", []),
            "show_account_passwords": _api("Show passwords.", []),
            "complete_task": _api("Mark the task done.", []),
            "show_payment_cards": _api("Show cards.", []),
        },
        "api_docs": {
            "show_api_descriptions": _api("List one app.", []),
            "show_api_doc": _api("Show one API.", []),
        },
    }


class _Requester:
    def __init__(
        self,
        *,
        login_result: object | None = None,
        fail: str | None = None,
        passwords: list[object] | None = None,
    ) -> None:
        self.calls: list[tuple[str, str, bool, dict[str, object]]] = []
        self.login_result = login_result or {
            "access_token": _SECRET,
            "token_type": "Bearer",
        }
        self.fail = fail
        self.passwords = passwords or [
            {"account_name": "venmo", "password": _VENMO_PASSWORD},
            {"account_name": "spotify", "password": _SPOTIFY_PASSWORD},
        ]

    def request(self, _app_name: str, _api_name: str, track: bool = True, **data: object):
        self.calls.append((_app_name, _api_name, track, dict(data)))
        if self.fail == _api_name:
            raise RuntimeError(f"password={_SPOTIFY_PASSWORD}")
        if _api_name == "show_profile":
            return {"email": _EMAIL, "first_name": "A"}
        if _api_name == "show_account_passwords":
            return self.passwords
        if _api_name == "login":
            return self.login_result
        if _api_name == "signup":
            raise AssertionError("signup invoked")
        raise AssertionError(_api_name)


class _World:
    def __init__(
        self,
        requester: _Requester | None = None,
        *,
        echo: bool = False,
        documentation: object | None = None,
    ) -> None:
        self.requester = requester
        self.echo = echo
        self.task = SimpleNamespace(
            instruction="instruction",
            api_docs=_documentation() if documentation is None else documentation,
        )
        self.actions: list[str] = []

    def execute(self, action: str) -> str:
        self.actions.append(action)
        if "show_api_descriptions" in action:
            return json.dumps(
                [
                    {"name": "login", "description": "Sign in."},
                    {"name": "signup", "description": "Sign up."},
                    {"name": "create_playlist", "description": "Create a playlist."},
                ]
            )
        if "show_api_doc" in action:
            return json.dumps(
                {
                    "api_name": "create_playlist",
                    "description": "Create a playlist.",
                    "parameters": [
                        _parameter("access_token"),
                        _parameter("title"),
                    ],
                    "response_schemas": {
                        "success": {"access_token": "string", "playlist_id": 1}
                    },
                }
            )
        if self.echo:
            return action
        return "ok"

    def evaluate(self) -> SimpleNamespace:
        return SimpleNamespace(success=False, pass_count=1, num_tests=3, difficulty=1)

    def close(self) -> None:
        return None


def _session(
    world: _World,
    *,
    setup: str | None = _PROFILE,
    capability: str | None = _CAPABILITY,
) -> LiveAppWorldSession:
    return LiveAppWorldSession(
        "task-auth",
        opener=lambda task_id: world,
        tool_access_profile=capability,
        appworld_setup_profile=setup,
    )


def _names(rendered: str, app_name: str) -> set[str]:
    prefix = f"{app_name}."
    return {
        line.split(":", 1)[0]
        for line in rendered.splitlines()
        if line.startswith(prefix)
    }


def _configuration(**task_overrides: object) -> RunConfiguration:
    document = json.loads(
        (_root() / "configs/models/qwen3_4b_production.json").read_text(encoding="utf-8")
    )
    document["task"] = {
        "appworld_version": "0.1.3.post1",
        "split": "train",
        "selection_rule": "deterministic_sample",
        "selection_seed": 17,
        "task_count": 1,
        "task_set_hash": "c" * 64,
    }
    document["task"].update(task_overrides)
    document["run_seed"] = 17
    document["git_commit"] = "a" * 40
    document["protocol_hash"] = None
    return RunConfiguration.from_dict(document)


def _clock():
    current = datetime(2026, 10, 1, tzinfo=timezone.utc)

    def tick() -> datetime:
        nonlocal current
        value = current
        current = current + timedelta(seconds=1)
        return value

    return tick


def _turn(
    output_text: str,
    *,
    action: str | None,
    app_name: str | None,
    api_name: str | None,
    started_at: datetime,
) -> AgentTurn:
    return AgentTurn(
        prompt_text="plan the next action",
        output_text=output_text,
        top_k_logprobs=_LOGPROBS,
        generated_token_count=1,
        latency_seconds=0.1,
        started_at=started_at,
        action=action,
        app_name=app_name,
        api_name=api_name,
    )


class _Agent:
    def __init__(self, turns: list[tuple[str, str | None, str | None, str | None]], clock) -> None:
        self._turns = turns
        self._clock = clock
        self._index = 0
        self.events: list[str] = []
        self.tool_outputs: list[str | None] = []

    def begin(self, context: TaskContext, config: RunConfiguration) -> None:
        del context, config
        self.events.append("begin")

    def next_turn(self, *, tool_output: str | None) -> AgentTurn:
        self.tool_outputs.append(tool_output)
        output_text, action, app_name, api_name = self._turns[self._index]
        self._index += 1
        return _turn(
            output_text,
            action=action,
            app_name=app_name,
            api_name=api_name,
            started_at=self._clock(),
        )


class _PairSession:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.prepare_count = 0
        self.auth = None
        self.close_count = 0

    def prepare(self) -> None:
        self.prepare_count += 1
        self.auth = object()

    def initial_state_identity(self) -> str:
        return "same-state"

    def context(self) -> TaskContext:
        return TaskContext(self.task_id, "instruction", "docs")

    def execute(self, action: str):
        del action
        raise AssertionError("plan mode executes")

    def evaluate(self):
        raise AssertionError("plan mode evaluates")

    def close(self) -> None:
        self.close_count += 1


class _PlanAgent:
    def __init__(self, clock) -> None:
        self._clock = clock

    def begin(self, context: TaskContext, config: RunConfiguration) -> None:
        del context, config

    def next_turn(self, *, tool_output: str | None) -> AgentTurn:
        del tool_output
        return _turn(
            "1. read the library",
            action=None,
            app_name=None,
            api_name=None,
            started_at=self._clock(),
        )


def _pilot():
    path = _root() / "scripts/evaluation/run_capability_pilot.py"
    spec = importlib.util.spec_from_file_location("run_capability_pilot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SetupProfileConfigurationTests(unittest.TestCase):
    def test_setup_profile_round_trips_and_changes_the_hash(self) -> None:
        plain = _configuration()
        self.assertIsNone(plain.task.appworld_setup_profile)
        self.assertNotIn("appworld_setup_profile", plain.task.to_dict())
        self.assertIs(
            hashed_values(plain)["task.appworld_setup_profile"],
            MISSING_HASHED_LEAF,
        )
        changed = _configuration(appworld_setup_profile=_PROFILE)
        self.assertEqual(changed.task.appworld_setup_profile, _PROFILE)
        self.assertEqual(
            changed.to_dict()["task"]["appworld_setup_profile"],
            _PROFILE,
        )
        self.assertNotEqual(
            run_configuration_hash(changed),
            run_configuration_hash(plain),
        )
        restored = RunConfiguration.from_dict(changed.to_dict())
        self.assertEqual(restored.task.appworld_setup_profile, _PROFILE)
        self.assertEqual(
            run_configuration_hash(restored),
            run_configuration_hash(changed),
        )
        omitted = RunConfiguration.from_dict(plain.to_dict())
        self.assertIsNone(omitted.task.appworld_setup_profile)
        self.assertEqual(run_configuration_hash(omitted), run_configuration_hash(plain))
        with self.assertRaises(ConfigError):
            _configuration(appworld_setup_profile="  ")

    def test_mismatched_setup_profiles_are_rejected_before_worlds_open(self) -> None:
        reference = _configuration(appworld_setup_profile=_PROFILE)
        candidate = _configuration()
        opened: list[str] = []
        with self.assertRaisesRegex(EpisodeRejected, "setup profiles are not compatible"):
            run_pair(
                "task-1",
                reference,
                candidate,
                reference_run=new_run_identity(reference),
                candidate_run=new_run_identity(candidate),
                runtime=RuntimeDependencies(
                    session_factory=lambda task_id: opened.append(task_id),
                    agent=_PlanAgent(_clock()),
                    clock=_clock(),
                ),
                mode="plan",
            )
        self.assertEqual(opened, [])

    def test_committed_spotify_and_14b_contracts(self) -> None:
        root = _root()
        task = json.loads(
            (root / "configs/tasks/train_spotify_capability.json").read_text(encoding="utf-8")
        )
        pilot = json.loads(
            (root / "configs/models/qwen3_14b_awq_spotify_capability.json").read_text(
                encoding="utf-8"
            )
        )
        previous = json.loads(
            (root / "configs/models/qwen3_4b_spotify_capability.json").read_text(encoding="utf-8")
        )
        self.assertEqual(task["appworld_setup_profile"], _PROFILE)
        self.assertEqual(task["task_set_hash"], _TASK_HASH)
        self.assertEqual(pilot["agent"]["prompt"]["prompt_version"], "prompt-runtime-auth-v1")
        self.assertEqual(pilot["agent"]["tool_access_profile"], _CAPABILITY)
        self.assertEqual(pilot["agent"]["execute_max_model_turns"], 20)
        self.assertEqual(pilot["agent"]["sampling"]["execute_max_tokens"], 192)
        self.assertEqual(pilot["agent"]["sampling"]["temperature"], 0.0)
        self.assertEqual(pilot["agent"]["sampling"]["seed"], 17)
        self.assertEqual(pilot["model"]["model"]["repository"], "Qwen/Qwen3-14B-AWQ")
        self.assertEqual(
            pilot["model"]["model"]["revision"],
            "31c69efc29464b6bb0aee1398b5a7b50a99340c3",
        )
        self.assertEqual(pilot["model"]["quantization"]["method"], "awq")
        serving = pilot["model"]["serving"]
        self.assertEqual(serving["dtype"], "float16")
        self.assertEqual(serving["kv_cache_dtype"], "float16")
        self.assertEqual(serving["max_model_len"], 32768)
        self.assertEqual(serving["gpu_memory_utilization"], 0.8)
        self.assertEqual(serving["max_num_seqs"], 1)
        self.assertEqual(serving["max_num_batched_tokens"], 8192)
        self.assertEqual(previous["agent"]["prompt"]["prompt_version"], "prompt-v4")
        public = {
            key: task[key]
            for key in (
                "appworld_version",
                "split",
                "selection_rule",
                "selection_seed",
                "task_count",
                "task_set_hash",
            )
        }
        adopted = _pilot()._adopt_committed_setup_profile(
            TaskConfiguration.from_dict(public)
        )
        self.assertIsNotNone(adopted)
        assert adopted is not None
        self.assertEqual(adopted.appworld_setup_profile, _PROFILE)

    def test_runtime_receives_the_setup_profile(self) -> None:
        plain = _configuration()
        changed = _configuration(appworld_setup_profile=_PROFILE)
        open_runtime = build_runtime(plain, "http://127.0.0.1:9", mode="execute")
        closed_runtime = build_runtime(changed, "http://127.0.0.1:9", mode="execute")
        self.assertTrue(is_live_runtime(open_runtime))
        self.assertTrue(is_live_runtime(closed_runtime))
        self.assertIsNone(open_runtime.session_factory.keywords["appworld_setup_profile"])
        self.assertEqual(
            closed_runtime.session_factory.keywords["appworld_setup_profile"],
            _PROFILE,
        )


class SpotifyPreparationTests(unittest.TestCase):
    def _open(self, world: _World, **kwargs: object) -> LiveAppWorldSession:
        session = _session(world, **kwargs)
        self.addCleanup(session.close)
        return session

    def test_no_profile_preparation_is_a_no_op(self) -> None:
        requester = _Requester()
        world = _World(requester)
        session = self._open(world, setup=None, capability=None)
        session.prepare()
        session.prepare()
        self.assertEqual(requester.calls, [])
        self.assertEqual(world.actions, [])
        self.assertIn("spotify.login", session.context().api_documentation)

    def test_preparation_uses_the_existing_spotify_account_once(self) -> None:
        requester = _Requester()
        world = _World(requester)
        session = self._open(world)
        session.prepare()
        session.prepare()
        names = [call[1] for call in requester.calls]
        self.assertEqual(
            names,
            ["show_profile", "show_account_passwords", "login"],
        )
        app_name, api_name, tracked, data = requester.calls[-1]
        self.assertEqual((app_name, api_name, tracked), ("spotify", "login", False))
        self.assertEqual(data["username"], _EMAIL)
        self.assertEqual(data["password"], _SPOTIFY_PASSWORD)
        self.assertNotEqual(data["password"], _VENMO_PASSWORD)
        self.assertNotIn("signup", names)
        self.assertEqual(world.actions, [])
        self.assertNotIn(_SECRET, repr(requester.calls))

    def test_missing_spotify_account_or_token_is_a_setup_failure(self) -> None:
        missing = _Requester(
            passwords=[{"account_name": "venmo", "password": _VENMO_PASSWORD}]
        )
        session = self._open(_World(missing))
        with self.assertRaises(RuntimeUnavailable) as caught:
            session.prepare()
        self.assertEqual(str(caught.exception), "spotify authentication setup failed")
        self.assertNotIn(_VENMO_PASSWORD, str(caught.exception))
        self.assertNotIn("login", [call[1] for call in missing.calls])
        self.assertNotIn("signup", [call[1] for call in missing.calls])

        denied = _Requester(login_result={"message": "Invalid credentials."})
        denied_session = self._open(_World(denied))
        with self.assertRaises(RuntimeUnavailable) as denied_caught:
            denied_session.prepare()
        self.assertNotIn(_SPOTIFY_PASSWORD, str(denied_caught.exception))
        self.assertNotIn("signup", [call[1] for call in denied.calls])

        broken = _Requester(fail="show_profile")
        broken_session = self._open(_World(broken))
        with self.assertRaises(RuntimeUnavailable) as broken_caught:
            broken_session.prepare()
        self.assertNotIn(_SPOTIFY_PASSWORD, str(broken_caught.exception))
        self.assertIsNone(broken_session._access_token)

    def test_unknown_profile_fails_before_any_call(self) -> None:
        requester = _Requester()
        session = self._open(_World(requester), setup="other_setup")
        with self.assertRaises(RuntimeUnavailable):
            session.prepare()
        self.assertEqual(requester.calls, [])


class TokenInjectionTests(unittest.TestCase):
    def test_runtime_token_replaces_a_supplied_token_and_stays_out_of_the_result(self) -> None:
        requester = _Requester()
        world = _World(requester, echo=True)
        session = _session(world)
        self.addCleanup(session.close)
        session.prepare()
        plain = 'apis.spotify.create_playlist(title="Morning")'
        supplied = 'apis.spotify.create_playlist(title="Morning", access_token="fake-token")'
        public = 'apis.spotify.search_songs(query="a", access_token="fake-token")'
        plain_result = session.execute(plain)
        supplied_result = session.execute(supplied)
        public_result = session.execute(public)
        self.assertIsNone(plain_result.error_message)
        self.assertIn(_SECRET, world.actions[0])
        self.assertIn("Morning", world.actions[0])
        self.assertNotIn(_SECRET, plain_result.output_text or "")
        self.assertIn(_SECRET, world.actions[1])
        self.assertNotIn(_FAKE, world.actions[1])
        self.assertNotIn(_SECRET, supplied_result.output_text or "")
        self.assertNotIn(_SECRET, world.actions[2])
        self.assertNotIn(_FAKE, world.actions[2])
        self.assertNotIn(_SECRET, public_result.output_text or "")
        self.assertIn("query", world.actions[2])

    def test_authenticated_surface_hides_credentials_and_sanitizes_helpers(self) -> None:
        world = _World()
        session = _session(world)
        self.addCleanup(session.close)
        rendered = session.context().api_documentation
        self.assertEqual(
            _names(rendered, "spotify"),
            {"spotify.create_playlist", "spotify.search_songs"},
        )
        self.assertEqual(_names(rendered, "supervisor"), {"supervisor.complete_task"})
        self.assertNotIn("access_token", rendered)
        self.assertNotIn("venmo", rendered)
        self.assertNotIn("show_payment_cards", rendered)
        for action in (
            "apis.spotify.login(username='a', password='b')",
            "apis.spotify.signup(first_name='a', last_name='b', email='a@b.c', password='c')",
            "apis.supervisor.show_profile()",
            "apis.supervisor.show_account_passwords()",
            'apis.api_docs.show_api_doc(app_name="spotify", api_name="login")',
        ):
            result = session.execute(action)
            self.assertIsNone(result.output_text, action)
            self.assertIsNotNone(result.error_message, action)
        self.assertEqual(world.actions, [])
        descriptions = session.execute(
            'apis.api_docs.show_api_descriptions(app_name="spotify")'
        )
        listed = json.loads(descriptions.output_text or "")
        self.assertEqual(
            [item["name"] for item in listed],
            ["create_playlist"],
        )
        document = json.loads(
            session.execute(
                'apis.api_docs.show_api_doc(app_name="spotify", api_name="create_playlist")'
            ).output_text
            or ""
        )
        self.assertEqual(
            [item["name"] for item in document["parameters"]],
            ["title"],
        )
        self.assertNotIn("access_token", json.dumps(document))
        complete = session.execute("apis.supervisor.complete_task()")
        self.assertIsNone(complete.error_message)
        self.assertEqual(len(world.actions), 3)

    def test_setup_profile_without_the_capability_allowlist_keeps_other_apps(self) -> None:
        world = _World()
        session = _session(world, capability=None)
        self.addCleanup(session.close)
        rendered = session.context().api_documentation
        self.assertIn("venmo.search_users", rendered)
        self.assertNotIn("spotify.login", rendered)
        self.assertNotIn("supervisor.show_profile", rendered)
        self.assertIn("supervisor.complete_task", rendered)
        self.assertNotIn("access_token", rendered)


class EpisodeBoundaryTests(unittest.TestCase):
    def test_preparation_precedes_the_agent_and_does_not_spend_turns(self) -> None:
        requester = _Requester()
        world = _World(requester, echo=True)
        session = _session(world)
        self.addCleanup(session.close)
        events: list[str] = []
        original_prepare = session.prepare
        original_context = session.context

        def prepare() -> None:
            events.append("prepare")
            original_prepare()

        def context() -> TaskContext:
            events.append("context")
            return original_context()

        session.prepare = prepare
        session.context = context
        clock = _clock()
        call = 'apis.spotify.create_playlist(title="Morning", access_token="fake-token")'
        agent = _Agent(
            [
                (call, call, "spotify", "create_playlist"),
                ("STOP", None, None, None),
            ],
            clock,
        )
        original_begin = agent.begin

        def begin(context: TaskContext, config: RunConfiguration) -> None:
            events.append("begin")
            original_begin(context, config)

        agent.begin = begin
        config = _configuration(appworld_setup_profile=_PROFILE)
        result = run_episode(
            "task-auth",
            config,
            "execute",
            run=new_run_identity(config),
            runtime=RuntimeDependencies(
                session_factory=lambda task_id: session,
                agent=agent,
                clock=clock,
            ),
        )
        self.assertEqual(events, ["prepare", "context", "begin"])
        self.assertEqual(len(result.model_steps), 2)
        self.assertEqual(len(result.tool_steps), 1)
        stored = result.tool_steps[0]
        self.assertEqual(stored.action, call)
        self.assertNotIn(_SECRET, stored.action)
        self.assertNotIn(_SECRET, stored.output_text or "")
        self.assertIn(_FAKE, stored.action)
        self.assertIn(_SECRET, world.actions[0])
        self.assertNotIn(_FAKE, world.actions[0])
        self.assertEqual(
            [call[1] for call in requester.calls],
            ["show_profile", "show_account_passwords", "login"],
        )
        self.assertEqual(result.termination_reason, "agent_stopped")

    def test_setup_failure_is_a_runtime_error_without_an_evaluator(self) -> None:
        class _Boom:
            def __init__(self) -> None:
                self.close_count = 0

            def prepare(self) -> None:
                raise RuntimeError(f"password={_SPOTIFY_PASSWORD}")

            def context(self) -> TaskContext:
                raise AssertionError("context")

            def execute(self, action: str):
                raise AssertionError(action)

            def evaluate(self):
                raise AssertionError("evaluate")

            def close(self) -> None:
                self.close_count += 1

        began: list[str] = []

        class _Idle:
            def begin(self, context: TaskContext, config: RunConfiguration) -> None:
                del context, config
                began.append("begin")

            def next_turn(self, *, tool_output: str | None) -> AgentTurn:
                del tool_output
                raise AssertionError("turn")

        session = _Boom()
        config = _configuration(appworld_setup_profile=_PROFILE)
        result = run_episode(
            "task-auth",
            config,
            "execute",
            run=new_run_identity(config),
            runtime=RuntimeDependencies(
                session_factory=lambda task_id: session,
                agent=_Idle(),
                clock=_clock(),
            ),
        )
        self.assertEqual(result.termination_reason, "runtime_error")
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.model_steps, ())
        self.assertEqual(result.tool_steps, ())
        self.assertIsNone(result.evaluator_outcome)
        self.assertEqual(result.episode_errors[0].message, "environment setup failed")
        self.assertNotIn(_SPOTIFY_PASSWORD, result.episode_errors[0].message)
        self.assertEqual(began, [])
        self.assertEqual(session.close_count, 1)

    def test_agent_still_decides_complete_task(self) -> None:
        requester = _Requester()
        world = _World(requester)
        session = _session(world)
        self.addCleanup(session.close)
        clock = _clock()
        call = "apis.supervisor.complete_task()"
        agent = _Agent([(call, call, "supervisor", "complete_task")], clock)
        config = _configuration(appworld_setup_profile=_PROFILE)
        result = run_episode(
            "task-auth",
            config,
            "execute",
            run=new_run_identity(config),
            runtime=RuntimeDependencies(
                session_factory=lambda task_id: session,
                agent=agent,
                clock=clock,
            ),
        )
        self.assertEqual(result.termination_reason, "appworld_completed")
        self.assertEqual(len(result.model_steps), 1)
        self.assertEqual(result.tool_steps[0].action, call)
        self.assertEqual([item[1] for item in requester.calls], [
            "show_profile",
            "show_account_passwords",
            "login",
        ])
        self.assertTrue(any("complete_task" in action for action in world.actions))


class IndependentPreparationTests(unittest.TestCase):
    def test_pair_prepares_each_world_with_the_same_profile(self) -> None:
        config = _configuration(appworld_setup_profile=_PROFILE)
        created: list[_PairSession] = []

        def factory(task_id: str) -> _PairSession:
            session = _PairSession(task_id)
            created.append(session)
            return session

        clock = _clock()
        run_pair(
            "task-1",
            config,
            config,
            reference_run=new_run_identity(config),
            candidate_run=new_run_identity(config),
            runtime=RuntimeDependencies(
                session_factory=factory,
                agent=_PlanAgent(clock),
                clock=clock,
            ),
            mode="plan",
        )
        self.assertEqual(len(created), 2)
        self.assertIsNot(created[0], created[1])
        self.assertEqual([session.prepare_count for session in created], [1, 1])
        self.assertIsNotNone(created[0].auth)
        self.assertIsNotNone(created[1].auth)
        self.assertIsNot(created[0].auth, created[1].auth)
        self.assertEqual(config.task.appworld_setup_profile, _PROFILE)


if __name__ == "__main__":
    unittest.main()
