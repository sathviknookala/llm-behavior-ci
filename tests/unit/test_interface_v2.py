import json
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from llm_behavior_ci.config import (
    AgentConfiguration,
    ModelConfiguration,
    RunConfiguration,
    TaskConfiguration,
    hashed_values,
    run_configuration_hash,
)
from llm_behavior_ci.runtime.actions import ActionRejected, parse_model_output
from llm_behavior_ci.runtime.agent import AgentTurn, SmolagentsVLLMAgent
from llm_behavior_ci.runtime.api_docs import resolve_api_documentation
from llm_behavior_ci.runtime.appworld import TaskContext, ToolResult, render_api_documentation
from llm_behavior_ci.runtime.prompts import PROMPT_RUNTIME_AUTH_V2, render_system_text
from llm_behavior_ci.runtime.workflow import (
    WorkflowControlledAgent,
    workflow_instruction,
)

_ROOT = Path(__file__).resolve().parents[2]
_GIT = "a" * 40
_START = datetime(2026, 10, 3, tzinfo=timezone.utc)
_PLAN = ["Read the library", "Apply the change"]
_SHOW = "apis.spotify.show_playlist_library()"
_ERROR = "Execution failed. page_limit must be <= 20"


def _model_payload(name: str) -> dict[str, object]:
    payload = json.loads((_ROOT / "configs" / "models" / name).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError("model configuration must be an object")
    return payload


def _task() -> TaskConfiguration:
    payload = json.loads(
        (_ROOT / "configs" / "tasks" / "train_spotify_capability.json").read_text(
            encoding="utf-8"
        )
    )
    public = {
        field: payload[field]
        for field in (
            "appworld_version",
            "split",
            "selection_rule",
            "selection_seed",
            "task_count",
            "task_set_hash",
            "appworld_setup_profile",
        )
    }
    return TaskConfiguration.from_dict(public)


def _configuration(name: str) -> RunConfiguration:
    payload = _model_payload(name)
    task = _task()
    return RunConfiguration(
        model=ModelConfiguration.from_dict(payload["model"]),
        agent=AgentConfiguration.from_dict(payload["agent"]),
        task=task,
        run_seed=task.selection_seed,
        git_commit=_GIT,
        protocol_hash=None,
    )


def _render(**overrides: object) -> str:
    values = {
        "prompt_version": PROMPT_RUNTIME_AUTH_V2,
        "plan_format_version": "plan-v1",
        "thinking_enabled": False,
        "action_interface": "code",
        "mode": "execute",
    }
    values.update(overrides)
    return render_system_text(**values)


def _docs() -> dict[str, object]:
    return {
        "spotify": {
            "search_songs": {
                "description": "Search songs.",
                "parameters": [
                    {
                        "name": "page_index",
                        "type": "integer",
                        "required": False,
                        "constraints": ["value >= 0.0"],
                    },
                    {
                        "name": "page_limit",
                        "type": "integer",
                        "required": False,
                        "constraints": ["value >= 1.0, <= 20.0"],
                    },
                    {
                        "name": "genre",
                        "type": "string",
                        "required": False,
                        "constraints": ["  value in ['rock', 'pop']  ", ""],
                    },
                ],
            },
            "review_song": {
                "description": "Review a song.",
                "parameters": [
                    {
                        "name": "rating",
                        "type": "integer",
                        "required": True,
                        "constraints": ["value >= 1.0, <= 5.0"],
                    }
                ],
            },
            "show_playlist": {
                "description": "Show one playlist.",
                "parameters": [
                    {"name": "playlist_id", "type": "integer", "required": True}
                ],
            },
            "create_playlist": {
                "description": "Flag.",
                "parameters": [
                    {
                        "name": "is_public",
                        "type": "boolean",
                        "required": False,
                        "constraints": [],
                    },
                    {
                        "name": "title",
                        "type": "string",
                        "required": True,
                        "constraints": [{"minimum": 1}, "   "],
                    },
                ],
            },
        },
        "supervisor": {
            "complete_task": {
                "description": "Mark the task complete.",
                "parameters": [
                    {
                        "name": "answer",
                        "type": "number | integer | string",
                        "required": False,
                        "constraints": [],
                    },
                    {
                        "name": "status",
                        "type": "string",
                        "required": False,
                        "constraints": ["value in ['success', 'fail']"],
                    },
                ],
            }
        },
    }


class _Base:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, object]] = []

    def begin(self, context: TaskContext, config: RunConfiguration) -> None:
        del context, config

    def generate_turn(
        self,
        *,
        tool_output: str | None,
        extra_instruction: str | None = None,
        parse_action: bool = True,
    ) -> AgentTurn:
        del parse_action
        self.calls.append(
            {"tool_output": tool_output, "extra_instruction": extra_instruction}
        )
        return AgentTurn(
            prompt_text=extra_instruction or "",
            output_text=self.outputs.pop(0),
            top_k_logprobs=(),
            generated_token_count=1,
            latency_seconds=0.0,
            started_at=_START,
            action=None,
            app_name=None,
            api_name=None,
        )


def _first(action: str) -> str:
    return json.dumps({"plan": _PLAN, "active_step": 1, "action": action})


class InterfaceV2Tests(unittest.TestCase):
    def test_constraints_render_only_when_the_metadata_defines_them(self) -> None:
        source = _docs()
        plain = render_api_documentation(source)
        rich = render_api_documentation(source, include_constraints=True)
        self.assertEqual(plain, render_api_documentation(source, include_constraints=False))
        self.assertNotIn("#", plain)
        self.assertIn("page_limit:integer?  # value >= 1.0, <= 20.0", rich)
        self.assertIn("page_index:integer?  # value >= 0.0", rich)
        self.assertIn("genre:string?  # value in ['rock', 'pop']", rich)
        self.assertIn("rating:integer  # value >= 1.0, <= 5.0", rich)
        self.assertIn("status:string?  # value in ['success', 'fail']", rich)
        self.assertIn("playlist_id:integer", rich)
        self.assertNotIn("playlist_id:integer  #", rich)
        self.assertIn("is_public:boolean?", rich)
        self.assertNotIn("is_public:boolean?  #", rich)
        self.assertIn("title:string", rich)
        self.assertNotIn("title:string  #", rich)
        self.assertNotIn("True", rich)
        self.assertNotIn("False", rich)
        compressed = render_api_documentation(
            {
                "spotify": {
                    "search_songs": {
                        "description": "Search.",
                        "parameters": {
                            "required": ["query"],
                            "optional": ["page_limit"],
                        },
                    }
                }
            },
            include_constraints=True,
        )
        self.assertEqual(
            compressed,
            "spotify.search_songs: Search. | query, page_limit?\n",
        )
        self.assertNotIn("#", compressed)

    def test_execute_instructions_state_the_parser_contract(self) -> None:
        text = _render()
        self.assertIn("exactly one", text)
        self.assertIn("apis.<app>.<api>(...)", text)
        self.assertIn("Use argument=value.", text)
        self.assertIn("Do not use argument:value.", text)
        self.assertIn("apis.spotify.search_songs(page_limit:20)", text)
        self.assertIn("apis.spotify.search_songs(page_limit=20)", text)
        self.assertIn('"app_name": "spotify"', text)
        self.assertIn('"api_name": "show_playlist"', text)
        self.assertIn("True", text)
        self.assertIn("False", text)
        self.assertIn("None", text)
        self.assertIn("true", text)
        self.assertIn("false", text)
        self.assertIn("null", text)
        self.assertIn("Outer response: JSON", text)
        self.assertIn('"action" value: Python-style API call string', text)
        self.assertIn("leave the complete_task answer empty", text)
        self.assertIn("apis.supervisor.complete_task(answer=None)", text)
        self.assertIn("do not put a prose status summary in the answer field", text)
        self.assertIn("put the requested answer in the complete_task answer field", text)
        self.assertIn("provide the actual requested value/entity", text)
        self.assertNotIn("evaluator", text.lower())
        previous = _render(prompt_version="prompt-runtime-auth-v1")
        self.assertNotIn("Do not use argument:value.", previous)
        self.assertNotIn("answer=None", previous)
        self.assertNotIn('"app_name": "spotify"', previous)
        self.assertIn("apis.<app>.<api>(...)", previous)

    def test_plan_mode_keeps_the_plan_format_and_omits_action_syntax(self) -> None:
        previous = _render(prompt_version="prompt-runtime-auth-v1", mode="plan")
        current = _render(mode="plan")
        self.assertEqual(
            previous.replace("prompt-runtime-auth-v1", PROMPT_RUNTIME_AUTH_V2),
            current,
        )
        self.assertIn("Emit a numbered plan before acting.", current)
        self.assertNotIn("argument=value", current)
        self.assertNotIn("answer=None", current)
        with self.assertRaisesRegex(ValueError, "prompt-runtime-auth-v2 does not support"):
            _render(action_interface="tool_calling")

    def test_prompt_v2_shows_constraints_and_v1_keeps_the_old_lines(self) -> None:
        source = _docs()
        context = TaskContext(
            task_id="task-1",
            instruction="Follow the instruction.",
            api_documentation=render_api_documentation(source),
            api_documentation_source=source,
        )
        corrected = _configuration(
            "qwen3_14b_awq_spotify_capability_v2_interface.json"
        )
        previous = _configuration("qwen3_14b_awq_spotify_capability_v2.json")
        agent = SmolagentsVLLMAgent("http://127.0.0.1:9")
        agent.begin(context, corrected)
        visible = agent.messages()[1]["content"]
        self.assertIn("page_limit:integer?  # value >= 1.0, <= 20.0", visible)
        self.assertIn("status:string?  # value in ['success', 'fail']", visible)
        agent.begin(context, previous)
        historical = agent.messages()[1]["content"]
        self.assertIn("page_limit:integer?", historical)
        self.assertNotIn("#", historical)
        self.assertNotIn("value >= 1.0, <= 20.0", historical)
        corrupted_agent = replace(
            corrected.agent,
            api_docs_version="api-docs-corrupt-v1",
            api_docs_app="spotify",
        )
        agent.begin(context, replace(corrected, agent=corrupted_agent))
        redacted = agent.messages()[1]["content"]
        self.assertIn("spotify.search_songs: [documentation removed]", redacted)
        self.assertNotIn("value >= 1.0, <= 20.0", redacted)
        self.assertEqual(
            resolve_api_documentation(
                render_api_documentation(source),
                api_docs_version=None,
                api_docs_app=None,
            ),
            render_api_documentation(source),
        )

    def test_recoverable_errors_add_generic_recovery_only_on_the_next_turn(self) -> None:
        config = _configuration("qwen3_14b_awq_spotify_capability_v2_interface.json")
        base = _Base([_first(_SHOW), "{", _first(_SHOW)])
        controller = WorkflowControlledAgent(base, config.agent.workflow)
        controller.begin(
            TaskContext("task-1", "Follow the instruction.", "docs"),
            config,
        )
        controller.next_turn(tool_output=None)
        self.assertNotIn(
            "The previous API call was rejected.",
            str(base.calls[0]["extra_instruction"]),
        )
        controller.observe_tool_result(
            _SHOW,
            ToolResult(
                output_text=None,
                error_message=_ERROR,
                recoverable=True,
                app_name="spotify",
                api_name="show_playlist_library",
            ),
        )
        controller.next_turn(tool_output=_ERROR)
        instruction = str(base.calls[1]["extra_instruction"])
        self.assertEqual(base.calls[1]["tool_output"], _ERROR)
        self.assertIn("The previous API call was rejected.", instruction)
        self.assertIn(
            "Read the returned error message and correct the offending API name "
            "or argument.",
            instruction,
        )
        self.assertIn(
            "Do not blindly repeat the same invalid call unchanged.",
            instruction,
        )
        self.assertNotIn("search_songs", instruction)
        self.assertNotIn("evaluator", instruction.lower())
        self.assertNotIn(
            "The previous API call was rejected.",
            workflow_instruction(controller._state(), config.agent.workflow),
        )
        controller.next_turn(tool_output="Return exactly the required workflow JSON object.")
        follow = str(base.calls[2]["extra_instruction"])
        self.assertNotIn("The previous API call was rejected.", follow)
        historical = _configuration("qwen3_14b_awq_spotify_capability_v2.json")
        old = _Base([_first(_SHOW), _first(_SHOW)])
        old_controller = WorkflowControlledAgent(old, historical.agent.workflow)
        old_controller.begin(
            TaskContext("task-1", "Follow the instruction.", "docs"),
            historical,
        )
        old_controller.next_turn(tool_output=None)
        old_controller.observe_tool_result(
            _SHOW,
            ToolResult(
                output_text=None,
                error_message=_ERROR,
                recoverable=True,
                app_name="spotify",
                api_name="show_playlist_library",
            ),
        )
        old_controller.next_turn(tool_output=_ERROR)
        self.assertNotIn(
            "The previous API call was rejected.",
            str(old.calls[1]["extra_instruction"]),
        )

    def test_parser_still_rejects_the_invalid_representations(self) -> None:
        parsed, app_name, api_name = parse_model_output(
            "apis.spotify.search_songs(page_index=0, page_limit=20)"
        )
        self.assertEqual(app_name, "spotify")
        self.assertEqual(api_name, "search_songs")
        self.assertEqual(parsed, "apis.spotify.search_songs(page_index=0, page_limit=20)")
        with self.assertRaises(ActionRejected):
            parse_model_output("apis.spotify.search_songs(page_limit:20)")
        with self.assertRaises(ActionRejected):
            parse_model_output(
                '{"app_name": "spotify", "api_name": "show_playlist", '
                '"params": {"playlist_id": 37}}'
            )

    def test_interface_config_changes_only_the_prompt_hash(self) -> None:
        previous = _configuration("qwen3_14b_awq_spotify_capability_v2.json")
        corrected = _configuration(
            "qwen3_14b_awq_spotify_capability_v2_interface.json"
        )
        self.assertEqual(previous.agent.prompt.prompt_version, "prompt-runtime-auth-v1")
        self.assertEqual(corrected.agent.prompt.prompt_version, PROMPT_RUNTIME_AUTH_V2)
        self.assertEqual(previous.agent.workflow, corrected.agent.workflow)
        self.assertEqual(previous.model, corrected.model)
        self.assertNotEqual(
            run_configuration_hash(previous),
            run_configuration_hash(corrected),
        )
        left = hashed_values(previous)
        right = hashed_values(corrected)
        changed = {
            path: (left[path], right[path])
            for path in left
            if left[path] != right[path]
        }
        self.assertEqual(
            changed,
            {
                "agent.prompt.prompt_version": (
                    "prompt-runtime-auth-v1",
                    PROMPT_RUNTIME_AUTH_V2,
                )
            },
        )


if __name__ == "__main__":
    unittest.main()
