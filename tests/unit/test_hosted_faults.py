from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_behavior_ci.config import RunConfiguration, TaskConfiguration, hashed_values
from llm_behavior_ci.experiments.faults import (
    FaultError,
    FaultPatch,
    FaultSpec,
    apply_fault,
    live_fault_available,
    load_fault,
    load_fault_catalog,
    provider_fault_support,
)
from llm_behavior_ci.experiments.run_config import build_run_configuration
from llm_behavior_ci.runtime.agent import SmolagentsOpenAICompatibleAgent
from llm_behavior_ci.runtime.appworld import TaskContext

_ROOT = Path(__file__).resolve().parents[2]
_HOSTED_CATALOG = _ROOT / "configs" / "faults" / "hosted_zai"
_CATALOG = _ROOT / "configs" / "faults"
_KEY = "zai-test-secret-value-0123456789"


def _task() -> TaskConfiguration:
    document = json.loads((_ROOT / "configs" / "tasks" / "train_smoke.json").read_text(encoding="utf-8"))
    document.pop("scenario_count", None)
    return TaskConfiguration.from_dict(document)


def _config(template: str) -> RunConfiguration:
    document = json.loads((_ROOT / "configs" / "models" / f"{template}.json").read_text(encoding="utf-8"))
    return build_run_configuration(document, _task(), run_seed=17, git_commit="a" * 40)


def _glm() -> RunConfiguration:
    return _config("glm_5_3_spotify_capability")


def _qwen() -> RunConfiguration:
    return _config("qwen3_4b_production")


class _Response:
    def __init__(self, raw: bytes) -> None:
        self._raw = raw

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def _payload(config: RunConfiguration, mode: str) -> dict[str, object]:
    agent = SmolagentsOpenAICompatibleAgent("zai", _KEY)
    agent.set_mode(mode)
    agent.begin(
        TaskContext(
            task_id="task-1",
            instruction="List the playlists",
            api_documentation="spotify.show_playlist_library: list playlists",
        ),
        config,
    )
    captured: list[object] = []
    body = {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "1. list playlists"},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4},
    }

    def urlopen(request: object, timeout: object = None) -> _Response:
        captured.append(request)
        return _Response(json.dumps(body).encode("utf-8"))

    with patch("urllib.request.urlopen", side_effect=urlopen):
        agent.generate_turn(tool_output=None, parse_action=False)
    return json.loads(captured[0].data.decode("utf-8"))


def _hosted(name: str) -> FaultSpec:
    return load_fault(_HOSTED_CATALOG / f"{name}.v1.json")


class HostedFaultPayloadTests(unittest.TestCase):
    def test_token_fault_lowers_plan_and_execute_caps_in_the_request(self) -> None:
        base = _glm()
        faulted = apply_fault(base, _hosted("glm_token_limit_truncation"))
        self.assertEqual(_payload(base, "plan")["max_tokens"], 1024)
        self.assertEqual(_payload(base, "execute")["max_tokens"], 192)
        self.assertEqual(_payload(faulted, "plan")["max_tokens"], 96)
        self.assertEqual(_payload(faulted, "execute")["max_tokens"], 96)

    def test_legacy_token_fault_now_reaches_mode_specific_caps(self) -> None:
        faulted = apply_fault(_glm(), load_fault(_CATALOG / "token_limit_truncation.v1.json"))
        self.assertEqual(_payload(faulted, "plan")["max_tokens"], 16)
        self.assertEqual(_payload(faulted, "execute")["max_tokens"], 16)

    def test_token_fault_on_a_config_without_mode_caps_keeps_max_tokens(self) -> None:
        base = _qwen()
        faulted = apply_fault(base, load_fault(_CATALOG / "token_limit_truncation.v1.json"))
        before, after = hashed_values(base), hashed_values(faulted)
        changed = {path for path in before if before[path] != after[path]}
        self.assertEqual(changed, {"agent.sampling.max_tokens"})

    def test_token_fault_that_lowers_no_cap_is_refused(self) -> None:
        fault = FaultSpec(
            fault_id="token_wide",
            version="1",
            kind="token_limit",
            patches=(FaultPatch("agent.sampling.max_tokens", 4096),),
        )
        with self.assertRaisesRegex(FaultError, "does not lower any effective cap"):
            apply_fault(_glm(), fault)

    def test_reasoning_fault_changes_thinking_type(self) -> None:
        base = _glm()
        faulted = apply_fault(base, _hosted("glm_reasoning_disabled"))
        self.assertEqual(_payload(base, "execute")["thinking"]["type"], "enabled")
        self.assertEqual(_payload(faulted, "execute")["thinking"]["type"], "disabled")

    def test_greedy_fault_drops_sampling_fields(self) -> None:
        base = _glm()
        faulted = apply_fault(base, _hosted("glm_sampling_greedy"))
        original = _payload(base, "execute")
        greedy = _payload(faulted, "execute")
        self.assertTrue(original["do_sample"])
        self.assertIn("temperature", original)
        self.assertFalse(greedy["do_sample"])
        self.assertNotIn("temperature", greedy)
        self.assertNotIn("top_p", greedy)

    def test_prompt_fault_changes_the_system_message(self) -> None:
        base = _glm()
        faulted = apply_fault(base, _hosted("glm_prompt_remove_api_guidance"))
        self.assertNotEqual(
            _payload(base, "execute")["messages"][0]["content"],
            _payload(faulted, "execute")["messages"][0]["content"],
        )

    def test_step_fault_retargets_the_execute_horizon(self) -> None:
        faulted = apply_fault(_glm(), _hosted("glm_step_limit_reduced"))
        self.assertEqual(faulted.agent.execute_max_model_turns, 8)
        self.assertEqual(faulted.agent.step_limit, 40)


class HostedFaultSupportTests(unittest.TestCase):
    def test_every_hosted_fault_applies_to_both_glm_templates(self) -> None:
        for template in ("glm_5_3_spotify_capability", "glm_5_3_general_experimental"):
            base = _config(template)
            for fault in load_fault_catalog(_HOSTED_CATALOG):
                with self.subTest(template=template, fault=fault.fault_version):
                    apply_fault(base, fault)
                    self.assertTrue(live_fault_available(fault, base).available)

    def test_ineffective_change_is_refused_with_an_actionable_message(self) -> None:
        fault = load_fault(_CATALOG / "sampling_temperature_one.v1.json")
        with self.assertRaisesRegex(FaultError, "declared_noop"):
            apply_fault(_glm(), fault)

    def test_declared_noop_is_a_benign_control_that_returns_the_base(self) -> None:
        base = _glm()
        self.assertEqual(apply_fault(base, _hosted("glm_benign_temperature_noop")), base)
        with self.assertRaisesRegex(FaultError, "declares a no-op but changes"):
            apply_fault(_qwen(), _hosted("glm_benign_temperature_noop"))

    def test_weight_and_template_faults_are_reported_unsupported_on_hosted(self) -> None:
        base = _glm()
        for name in (
            "fp8_weights",
            "nvfp4_weights",
            "lora_off_distribution",
            "model_downgrade_qwen3_1_7b",
            "template_thinking_enabled",
            "benign_batch_invariant",
        ):
            fault = load_fault(_CATALOG / f"{name}.v1.json")
            with self.subTest(fault=name):
                reason = provider_fault_support(base, fault)
                self.assertIsNotNone(reason)
                availability = live_fault_available(fault, base)
                self.assertFalse(availability.available)
                self.assertEqual(availability.reason, reason)
                with self.assertRaisesRegex(FaultError, "unsupported on this base"):
                    apply_fault(base, fault)

    def test_hosted_only_faults_are_unsupported_on_vllm(self) -> None:
        base = _qwen()
        for name in ("glm_reasoning_disabled", "glm_sampling_greedy"):
            with self.subTest(fault=name):
                self.assertIsNotNone(provider_fault_support(base, _hosted(name)))

    def test_vllm_catalog_is_unchanged(self) -> None:
        self.assertEqual(len(load_fault_catalog(_CATALOG)), 14)
        base = _qwen()
        for fault in load_fault_catalog(_CATALOG):
            with self.subTest(fault=fault.fault_version):
                self.assertIsNone(provider_fault_support(base, fault))


if __name__ == "__main__":
    unittest.main()
