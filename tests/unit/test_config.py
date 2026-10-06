import copy
import hashlib
import json
import os
import subprocess
import unittest
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from unittest.mock import patch

import llm_behavior_ci.config as configuration_module
from llm_behavior_ci.config import (
    HASHED_FIELDS,
    MISSING_HASHED_LEAF,
    RUNTIME_IDS,
    AgentConfiguration,
    ConfigError,
    EpisodeIdentity,
    LoRASettings,
    ModelConfiguration,
    RunConfiguration,
    RunIdentity,
    TaskConfiguration,
    canonical_configuration_json,
    hashed_values,
    load_model_configuration,
    new_episode_identity,
    new_pair_id,
    new_run_identity,
    run_configuration_hash,
)

_REVISION = "0123456789abcdef0123456789abcdef01234567"
_TOKENIZER_REVISION = "fedcba9876543210fedcba9876543210fedcba98"
_OTHER_REVISION = "1" * 40
_OTHER_TOKENIZER_REVISION = "2" * 40
_GIT_COMMIT = "a" * 40
_OTHER_GIT_COMMIT = "b" * 40
_TASK_SET_HASH = "c" * 64
_OTHER_TASK_SET_HASH = "d" * 64
_PROTOCOL_HASH = "e" * 64
_OTHER_PROTOCOL_HASH = "f" * 64
_LORA_REVISION = "3" * 40
_OTHER_LORA_REVISION = "4" * 40
_ANTHROPIC_HASHED_FIELDS = frozenset(
    {
        "model.provider",
        "model.model_id",
        "model.api_version",
        "model.thinking_mode",
        "model.effort",
    }
)

_REPLACEMENTS = {
    "model.model.repository": "Qwen/Qwen3-1.7B",
    "model.model.revision": _OTHER_REVISION,
    "model.tokenizer.repository": "Qwen/Qwen3-4B-Base",
    "model.tokenizer.revision": _OTHER_TOKENIZER_REVISION,
    "model.quantization.method": "fp8",
    "model.vllm_version": "0.31.0",
    "model.serving.dtype": "float16",
    "model.serving.max_model_len": 8193,
    "model.serving.gpu_memory_utilization": 0.5,
    "model.serving.max_num_seqs": 17,
    "model.serving.max_num_batched_tokens": 8193,
    "model.serving.kv_cache_dtype": "fp8",
    "model.serving.enable_prefix_caching": True,
    "model.serving.enable_chunked_prefill": True,
    "model.serving.enforce_eager": True,
    "model.serving.cpu_offload_gb": 2.0,
    "model.serving.tensor_parallel_size": 2,
    "model.serving.max_logprobs": 21,
    "model.serving.batch_invariant": True,
    "model.serving.sampler_backend": "flashinfer",
    "model.lora.repository": "org/adapter-other",
    "model.lora.revision": _OTHER_LORA_REVISION,
    "agent.smolagents_version": "1.22.1",
    "agent.action_interface": "tool_calling",
    "agent.prompt.prompt_version": "prompt-v2",
    "agent.prompt.plan_format_version": "plan-v2",
    "agent.prompt.thinking_enabled": True,
    "agent.step_limit": 41,
    "agent.execute_max_model_turns": 21,
    "agent.tool_access_profile": "other_profile",
    "agent.sampling.temperature": 1.0,
    "agent.sampling.top_p": 0.9,
    "agent.sampling.top_k": -1,
    "agent.sampling.min_p": 0.1,
    "agent.sampling.seed": 18,
    "agent.sampling.max_tokens": 513,
    "agent.sampling.execute_max_tokens": 193,
    "agent.sampling.plan_max_tokens": 1025,
    "agent.api_docs_version": "api-docs-corrupt-v2",
    "agent.api_docs_app": "other_app",
    "agent.workflow.repeat_action_limit": 3,
    "agent.workflow.no_progress_turns": 4,
    "agent.workflow.completion_gate": False,
    "agent.workflow.max_plan_steps": 6,
    "task.appworld_version": "0.1.3.post2",
    "task.split": "dev",
    "task.selection_rule": "fixed-v2",
    "task.selection_seed": 20260927,
    "task.task_count": 51,
    "task.task_set_hash": _OTHER_TASK_SET_HASH,
    "task.appworld_setup_profile": "other_setup",
    "run_seed": 8,
    "git_commit": _OTHER_GIT_COMMIT,
    "protocol_hash": _OTHER_PROTOCOL_HASH,
}


@dataclass(frozen=True)
class _MarkedRun(RunConfiguration):
    marker: str = "leak"


def _payload() -> dict[str, object]:
    return {
        "model": {
            "model": {
                "repository": "Qwen/Qwen3-4B",
                "revision": _REVISION,
            },
            "tokenizer": {
                "repository": "Qwen/Qwen3-4B",
                "revision": _TOKENIZER_REVISION,
            },
            "quantization": {"method": "none"},
            "vllm_version": "0.30.0",
            "serving": {
                "dtype": "bfloat16",
                "max_model_len": 8192,
                "gpu_memory_utilization": 0.9,
                "max_num_seqs": 16,
                "max_num_batched_tokens": 8192,
                "kv_cache_dtype": "bfloat16",
                "enable_prefix_caching": False,
                "enable_chunked_prefill": False,
                "enforce_eager": False,
                "cpu_offload_gb": 1.0,
                "tensor_parallel_size": 1,
                "max_logprobs": 20,
                "batch_invariant": False,
                "sampler_backend": "native",
            },
            "lora": {
                "repository": "org/adapter-base",
                "revision": _LORA_REVISION,
            },
        },
        "agent": {
            "smolagents_version": "1.22.0",
            "action_interface": "code",
            "prompt": {
                "prompt_version": "prompt-v1",
                "plan_format_version": "plan-v1",
                "thinking_enabled": False,
            },
            "step_limit": 40,
            "execute_max_model_turns": 20,
            "tool_access_profile": "base_profile",
            "sampling": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 20,
                "min_p": 0.0,
                "seed": 17,
                "max_tokens": 512,
                "execute_max_tokens": 192,
                "plan_max_tokens": 1024,
            },
            "api_docs_version": "api-docs-corrupt-v1",
            "api_docs_app": "calendar",
            "workflow": {
                "policy": "plan_progress_v1",
                "repeat_action_limit": 2,
                "no_progress_turns": 3,
                "completion_gate": True,
                "max_plan_steps": 5,
            },
        },
        "task": {
            "appworld_version": "0.1.3.post1",
            "split": "train",
            "selection_rule": "fixed-v1",
            "selection_seed": 20260926,
            "task_count": 50,
            "task_set_hash": _TASK_SET_HASH,
            "appworld_setup_profile": "base_setup",
        },
        "run_seed": 7,
        "git_commit": _GIT_COMMIT,
        "protocol_hash": _PROTOCOL_HASH,
    }


def _assign(payload: dict[str, object], path: str, value: object) -> None:
    cursor = payload
    keys = path.split(".")
    for key in keys[:-1]:
        child = cursor[key]
        if not isinstance(child, dict):
            raise AssertionError(path)
        cursor = child
    cursor[keys[-1]] = value


def _with(path: str, value: object) -> dict[str, object]:
    payload = _payload()
    _assign(payload, path, value)
    return payload


def _lookup(payload: dict[str, object], path: str) -> object:
    cursor: object = payload
    for key in path.split("."):
        if not isinstance(cursor, dict):
            raise AssertionError(path)
        cursor = cursor[key]
    return cursor


def _reverse(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _reverse(item)
            for key, item in reversed(tuple(value.items()))
        }
    return value


def _leaves(payload: dict[str, object], prefix: str = "") -> set[str]:
    found = set()
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            found.update(_leaves(value, path))
        else:
            found.add(path)
    return found


def _same_types(case: unittest.TestCase, left: object, right: object) -> None:
    case.assertIs(type(left), type(right))
    if is_dataclass(left):
        for field in fields(left):
            _same_types(
                case,
                getattr(left, field.name),
                getattr(right, field.name),
            )


def _configuration() -> RunConfiguration:
    return RunConfiguration.from_dict(_payload())


class ConfigurationTests(unittest.TestCase):
    def test_round_trip_preserves_types(self) -> None:
        original = _configuration()
        restored = RunConfiguration.from_dict(original.to_dict())
        self.assertEqual(restored, original)
        _same_types(self, original, restored)
        self.assertIsInstance(restored.agent.step_limit, int)
        self.assertNotIsInstance(restored.agent.step_limit, bool)
        self.assertIsInstance(restored.agent.sampling.temperature, float)
        self.assertIsInstance(restored.agent.prompt.thinking_enabled, bool)
        self.assertIsInstance(restored.protocol_hash, str)

        document = canonical_configuration_json(original)
        from_json = RunConfiguration.from_dict(json.loads(document))
        self.assertEqual(from_json, original)
        _same_types(self, original, from_json)

    def test_omitted_protocol_hash_round_trips_as_null(self) -> None:
        payload = _payload()
        del payload["protocol_hash"]
        explicit = _payload()
        explicit["protocol_hash"] = None
        omitted = RunConfiguration.from_dict(payload)
        present = RunConfiguration.from_dict(explicit)
        self.assertIsNone(omitted.protocol_hash)
        self.assertEqual(omitted, present)
        self.assertIsNone(omitted.to_dict()["protocol_hash"])
        self.assertEqual(
            run_configuration_hash(omitted),
            run_configuration_hash(present),
        )
        self.assertNotEqual(
            run_configuration_hash(omitted),
            run_configuration_hash(_configuration()),
        )
        self.assertIn('"protocol_hash":null', canonical_configuration_json(omitted))

    def test_loading_does_not_mutate_or_read_the_environment(self) -> None:
        payload = _payload()
        del payload["protocol_hash"]
        snapshot = copy.deepcopy(payload)
        with patch.dict(
            os.environ,
            {
                "PROTOCOL_HASH": "d" * 64,
                "GIT_COMMIT": "e" * 40,
                "TASK_SET_HASH": "f" * 64,
                "LLM_BEHAVIOR_CI_CONFIG": "production",
            },
        ):
            loaded = RunConfiguration.from_dict(payload)
        self.assertEqual(payload, snapshot)
        self.assertIsNone(loaded.protocol_hash)
        self.assertEqual(loaded.git_commit, _GIT_COMMIT)
        self.assertEqual(loaded.task.task_set_hash, _TASK_SET_HASH)
        source = Path(configuration_module.__file__).read_text(encoding="utf-8")
        self.assertNotIn("os.environ", source)
        self.assertNotIn("getenv", source)
        self.assertNotIn("subprocess", source)
        self.assertNotIn("rev-parse", source)

    def test_rejects_unknown_fields_and_malformed_objects(self) -> None:
        cases = {
            "task ids": ("task.task_ids", ["task_0001"]),
            "task content": ("task.instruction", "book a ride"),
            "misspelled utilization": (
                "model.serving.gpu_memory_utilisation",
                0.5,
            ),
            "run id": ("run_id", "runtime"),
            "production flag": ("production", True),
            "configuration hash": ("configuration_hash", _PROTOCOL_HASH),
        }
        for label, (path, value) in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ConfigError) as caught:
                    RunConfiguration.from_dict(_with(path, value))
                self.assertIn(path.rsplit(".", 1)[-1], str(caught.exception))

        payload = _payload()
        payload["model"] = ["Qwen/Qwen3-4B"]
        with self.assertRaises(ConfigError):
            RunConfiguration.from_dict(payload)
        payload = _payload()
        payload[1] = "x"
        with self.assertRaises(ConfigError):
            RunConfiguration.from_dict(payload)
        with self.assertRaises(ConfigError):
            RunConfiguration.from_dict(["not", "an", "object"])
        incomplete = _payload()
        del incomplete["run_seed"]
        with self.assertRaises(ConfigError) as caught:
            RunConfiguration.from_dict(incomplete)
        self.assertIn("run_seed", str(caught.exception))

    def test_rejects_invalid_values_and_coercions(self) -> None:
        cases = {
            "bool step limit": ("agent.step_limit", True),
            "float step limit": ("agent.step_limit", 8.0),
            "zero step limit": ("agent.step_limit", 0),
            "string task count": ("task.task_count", "50"),
            "zero task count": ("task.task_count", 0),
            "bool seed": ("agent.sampling.seed", True),
            "float seed": ("run_seed", 7.0),
            "int temperature": ("agent.sampling.temperature", 0),
            "nan temperature": ("agent.sampling.temperature", float("nan")),
            "infinite temperature": (
                "agent.sampling.temperature",
                float("inf"),
            ),
            "negative temperature": ("agent.sampling.temperature", -0.1),
            "zero top_p": ("agent.sampling.top_p", 0.0),
            "zero top_k": ("agent.sampling.top_k", 0),
            "min_p above one": ("agent.sampling.min_p", 1.1),
            "zero max tokens": ("agent.sampling.max_tokens", 0),
            "zero context": ("model.serving.max_model_len", 0),
            "full gpu rejected at zero": (
                "model.serving.gpu_memory_utilization",
                0.0,
            ),
            "gpu above one": ("model.serving.gpu_memory_utilization", 1.1),
            "bool batch flag": ("model.serving.batch_invariant", 0),
            "string thinking flag": ("agent.prompt.thinking_enabled", "false"),
            "branch revision": ("model.model.revision", "main"),
            "short git commit": ("git_commit", "abc1234"),
            "uppercase git commit": ("git_commit", "A" * 40),
            "git-length protocol hash": ("protocol_hash", "a" * 40),
            "uppercase task hash": ("task.task_set_hash", "C" * 64),
            "empty rule": ("task.selection_rule", "  fixed-v1"),
            "bare test split": ("task.split", "test"),
            "class-name interface": ("agent.action_interface", "CodeAgent"),
            "uppercase quantization": ("model.quantization.method", "FP8"),
            "auto dtype": ("model.serving.dtype", "auto"),
            "auto kv cache": ("model.serving.kv_cache_dtype", "auto"),
            "unknown sampler": ("model.serving.sampler_backend", "triton"),
        }
        for label, (path, value) in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ConfigError):
                    RunConfiguration.from_dict(_with(path, value))

        with self.assertRaises(ConfigError):
            TaskConfiguration(
                appworld_version="0.1.3.post1",
                split="train",
                selection_rule="fixed-v1",
                selection_seed=1,
                task_count=0,
                task_set_hash=_TASK_SET_HASH,
            )

    def test_accepts_boundary_values(self) -> None:
        accepted = {
            "disabled top_k": ("agent.sampling.top_k", -1),
            "full gpu": ("model.serving.gpu_memory_utilization", 1.0),
            "closed min_p": ("agent.sampling.min_p", 1.0),
            "negative sampling seed": ("agent.sampling.seed", -1),
            "zero run seed": ("run_seed", 0),
            "one task": ("task.task_count", 1),
            "nvfp4": ("model.quantization.method", "nvfp4"),
            "awq": ("model.quantization.method", "awq"),
            "float32": ("model.serving.dtype", "float32"),
            "challenge split": ("task.split", "test_challenge"),
            "normal split": ("task.split", "test_normal"),
            "tool interface": ("agent.action_interface", "tool_calling"),
        }
        for label, (path, value) in accepted.items():
            with self.subTest(case=label):
                loaded = RunConfiguration.from_dict(_with(path, value))
                self.assertEqual(_lookup(loaded.to_dict(), path), value)
                self.assertIs(type(_lookup(loaded.to_dict(), path)), type(value))

    def test_configuration_is_immutable(self) -> None:
        configuration = _configuration()
        with self.assertRaises(AttributeError):
            configuration.run_seed = 9
        with self.assertRaises(AttributeError):
            configuration.agent.sampling.temperature = 1.0

    def test_hash_uses_sorted_utf8_json_and_documented_fields(self) -> None:
        configuration = _configuration()
        document = canonical_configuration_json(configuration)
        self.assertEqual(
            document,
            json.dumps(
                json.loads(document),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ),
        )
        self.assertNotIn(": ", document)
        self.assertNotIn(", ", document)
        self.assertTrue(document.startswith('{"agent":'))
        digest = hashlib.sha256(document.encode("utf-8")).hexdigest()
        self.assertEqual(run_configuration_hash(configuration), digest)
        self.assertEqual(len(digest), 64)
        self.assertEqual(
            _leaves(configuration.to_dict()) | _ANTHROPIC_HASHED_FIELDS,
            set(HASHED_FIELDS),
        )
        self.assertTrue(
            _ANTHROPIC_HASHED_FIELDS.isdisjoint(_leaves(configuration.to_dict()))
        )
        for path in HASHED_FIELDS:
            self.assertIn(path, configuration_module.__doc__)
        self.assertTrue(RUNTIME_IDS.isdisjoint(HASHED_FIELDS))
        for name in RUNTIME_IDS:
            self.assertNotIn(f'"{name}"', document)

        unicode_payload = _payload()
        unicode_payload["task"]["selection_rule"] = "规则-v1"
        unicode_configuration = RunConfiguration.from_dict(unicode_payload)
        unicode_document = canonical_configuration_json(unicode_configuration)
        self.assertIn("规则", unicode_document)
        self.assertNotIn("\\u", unicode_document)
        escaped = json.dumps(
            unicode_configuration.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        self.assertNotEqual(
            hashlib.sha256(unicode_document.encode("utf-8")).hexdigest(),
            hashlib.sha256(escaped.encode("utf-8")).hexdigest(),
        )

    def test_hash_ignores_insertion_order_and_changes_with_every_field(self) -> None:
        configuration = _configuration()
        base_hash = run_configuration_hash(configuration)
        reversed_payload = _reverse(_payload())
        self.assertIsInstance(reversed_payload, dict)
        reordered = RunConfiguration.from_dict(reversed_payload)
        self.assertEqual(reordered, configuration)
        self.assertEqual(
            canonical_configuration_json(reordered),
            canonical_configuration_json(configuration),
        )
        self.assertEqual(
            set(HASHED_FIELDS) - set(_REPLACEMENTS) - _ANTHROPIC_HASHED_FIELDS,
            {"agent.workflow.policy"},
        )
        payload = _payload()
        for path, value in _REPLACEMENTS.items():
            with self.subTest(path=path):
                self.assertNotEqual(_lookup(payload, path), value)
                changed = RunConfiguration.from_dict(_with(path, value))
                self.assertNotEqual(run_configuration_hash(changed), base_hash)

    def test_extra_runtime_state_does_not_enter_the_hash(self) -> None:
        configuration = _configuration()
        base_hash = run_configuration_hash(configuration)
        values = {
            field.name: getattr(configuration, field.name)
            for field in fields(RunConfiguration)
        }
        marked = _MarkedRun(**values, marker="leak")
        self.assertEqual(run_configuration_hash(marked), base_hash)
        self.assertNotIn("marker", marked.to_dict())
        self.assertNotIn("leak", canonical_configuration_json(marked))
        object.__setattr__(
            configuration.agent.sampling,
            "temperature",
            float("nan"),
        )
        with self.assertRaises(ConfigError):
            run_configuration_hash(configuration)

    def test_run_and_episode_identity_keeps_provenance_explicit(self) -> None:
        configuration = _configuration()
        base_hash = run_configuration_hash(configuration)
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(configuration_module.__file__).parents[2],
            text=True,
        ).strip()
        self.assertNotEqual(configuration.git_commit, head)
        first = new_run_identity(configuration)
        second = new_run_identity(configuration)
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertEqual(
            first.configuration_hash,
            run_configuration_hash(configuration),
        )
        self.assertEqual(second.configuration_hash, first.configuration_hash)
        self.assertTrue(first.run_id.startswith(first.configuration_hash + "."))
        self.assertEqual(first.task_set_hash, configuration.task.task_set_hash)
        self.assertEqual(first.protocol_hash, configuration.protocol_hash)
        self.assertEqual(first.git_commit, configuration.git_commit)
        self.assertEqual(run_configuration_hash(configuration), base_hash)

        pair = new_pair_id()
        left = new_episode_identity(first, pair_id=pair)
        right = new_episode_identity(first, pair_id=pair)
        alone = new_episode_identity(first)
        self.assertEqual(left.pair_id, right.pair_id)
        self.assertNotEqual(left.episode_id, right.episode_id)
        self.assertEqual(left.run_id, first.run_id)
        self.assertEqual(right.run_id, first.run_id)
        self.assertTrue(left.episode_id.startswith(first.run_id + "."))
        self.assertIsNone(alone.pair_id)
        self.assertNotEqual(alone.episode_id, left.episode_id)

        other = new_run_identity(RunConfiguration.from_dict(_with("run_seed", 99)))
        shared = new_pair_id()
        candidate = new_episode_identity(first, pair_id=shared)
        production = new_episode_identity(other, pair_id=shared)
        self.assertEqual(candidate.pair_id, production.pair_id)
        self.assertNotEqual(candidate.run_id, production.run_id)
        self.assertNotEqual(candidate.episode_id, production.episode_id)
        self.assertEqual(
            run_configuration_hash(configuration),
            run_configuration_hash(_configuration()),
        )

        with self.assertRaises(ConfigError):
            new_episode_identity(first, pair_id="pair-1")
        with self.assertRaises(ConfigError):
            RunIdentity(
                run_id="ab" * 16,
                configuration_hash=_PROTOCOL_HASH,
                task_set_hash=_TASK_SET_HASH,
                protocol_hash=None,
                git_commit=_GIT_COMMIT,
            )
        with self.assertRaises(ConfigError):
            EpisodeIdentity(
                episode_id=f"{first.run_id}.{'c' * 32}",
                run_id="not-a-run",
                pair_id=pair,
            )


class OptionalHashedLeafTests(unittest.TestCase):
    def test_unset_lora_and_api_docs_are_omitted_from_canonical_json(self) -> None:
        payload = _payload()
        del payload["model"]["lora"]
        del payload["agent"]["api_docs_version"]
        del payload["agent"]["api_docs_app"]
        configuration = RunConfiguration.from_dict(payload)
        self.assertIsNone(configuration.model.lora)
        self.assertIsNone(configuration.agent.api_docs_version)
        self.assertIsNone(configuration.agent.api_docs_app)
        document = configuration.to_dict()
        self.assertNotIn("lora", document["model"])
        self.assertNotIn("api_docs_version", document["agent"])
        self.assertNotIn("api_docs_app", document["agent"])
        text = canonical_configuration_json(configuration)
        self.assertNotIn("lora", text)
        self.assertNotIn("api_docs", text)
        restored = RunConfiguration.from_dict(document)
        self.assertEqual(restored, configuration)

    def test_setting_or_changing_either_optional_pair_changes_the_hash(self) -> None:
        base_payload = _payload()
        del base_payload["model"]["lora"]
        del base_payload["agent"]["api_docs_version"]
        del base_payload["agent"]["api_docs_app"]
        unset = RunConfiguration.from_dict(base_payload)
        unset_hash = run_configuration_hash(unset)

        with_lora = RunConfiguration.from_dict(_payload())
        self.assertNotEqual(run_configuration_hash(with_lora), unset_hash)

        other_lora_payload = _payload()
        other_lora_payload["model"]["lora"]["repository"] = "org/adapter-other"
        other_lora = RunConfiguration.from_dict(other_lora_payload)
        self.assertNotEqual(
            run_configuration_hash(other_lora),
            run_configuration_hash(with_lora),
        )

        other_docs_payload = _payload()
        other_docs_payload["agent"]["api_docs_app"] = "other_app"
        other_docs = RunConfiguration.from_dict(other_docs_payload)
        self.assertNotEqual(
            run_configuration_hash(other_docs),
            run_configuration_hash(with_lora),
        )

    def test_mode_limit_overrides_are_optional_hashed_leaves(self) -> None:
        unset_payload = _payload()
        del unset_payload["agent"]["execute_max_model_turns"]
        del unset_payload["agent"]["sampling"]["execute_max_tokens"]
        del unset_payload["agent"]["sampling"]["plan_max_tokens"]
        unset = RunConfiguration.from_dict(unset_payload)
        document = json.loads(canonical_configuration_json(unset))
        self.assertNotIn("execute_max_model_turns", document["agent"])
        self.assertNotIn("execute_max_tokens", document["agent"]["sampling"])
        self.assertNotIn("plan_max_tokens", document["agent"]["sampling"])
        self.assertEqual(unset.agent.execute_turn_limit, 40)
        self.assertEqual(unset.agent.sampling.generation_max_tokens("execute"), 512)
        self.assertEqual(unset.agent.sampling.generation_max_tokens("plan"), 512)
        values = hashed_values(unset)
        for path in (
            "agent.execute_max_model_turns",
            "agent.sampling.execute_max_tokens",
            "agent.sampling.plan_max_tokens",
        ):
            self.assertIs(values[path], MISSING_HASHED_LEAF)
        unset_hash = run_configuration_hash(unset)
        for path, value in (
            ("agent.execute_max_model_turns", 20),
            ("agent.sampling.execute_max_tokens", 192),
            ("agent.sampling.plan_max_tokens", 1024),
        ):
            with self.subTest(path=path):
                one = json.loads(json.dumps(unset_payload))
                _assign(one, path, value)
                changed = RunConfiguration.from_dict(one)
                self.assertNotEqual(run_configuration_hash(changed), unset_hash)
                _assign(one, path, value + 1)
                self.assertNotEqual(
                    run_configuration_hash(RunConfiguration.from_dict(one)),
                    run_configuration_hash(changed),
                )
        configured = _configuration()
        self.assertEqual(configured.agent.execute_turn_limit, 20)
        self.assertEqual(configured.agent.sampling.generation_max_tokens("execute"), 192)
        self.assertEqual(configured.agent.sampling.generation_max_tokens("plan"), 1024)
        for path in (
            "agent.execute_max_model_turns",
            "agent.sampling.execute_max_tokens",
            "agent.sampling.plan_max_tokens",
        ):
            for bad in (0, -1, True, 1.5):
                with self.subTest(path=path, bad=bad):
                    with self.assertRaises(ConfigError):
                        RunConfiguration.from_dict(_with(path, bad))

    def test_api_docs_version_and_app_must_be_set_together(self) -> None:
        only_version = _payload()
        del only_version["agent"]["api_docs_app"]
        with self.assertRaises(ConfigError) as ctx:
            RunConfiguration.from_dict(only_version)
        self.assertIn("together", str(ctx.exception))

        only_app = _payload()
        del only_app["agent"]["api_docs_version"]
        with self.assertRaises(ConfigError) as ctx:
            RunConfiguration.from_dict(only_app)
        self.assertIn("together", str(ctx.exception))

    def test_lora_requires_both_fields(self) -> None:
        with self.assertRaises(ConfigError):
            LoRASettings.from_dict({"repository": "org/adapter"})
        with self.assertRaises(ConfigError):
            ModelConfiguration.from_dict(
                {
                    **_payload()["model"],
                    "lora": {"repository": "org/adapter"},
                }
            )

    def test_default_cpu_offload_is_omitted_and_keeps_existing_hashes(self) -> None:
        payload = _payload()
        del payload["model"]["serving"]["cpu_offload_gb"]
        omitted = RunConfiguration.from_dict(payload)
        self.assertEqual(omitted.model.serving.cpu_offload_gb, 0.0)
        self.assertNotIn("cpu_offload_gb", omitted.model.serving.to_dict())
        self.assertIs(
            hashed_values(omitted)["model.serving.cpu_offload_gb"],
            MISSING_HASHED_LEAF,
        )
        explicit = RunConfiguration.from_dict(_with("model.serving.cpu_offload_gb", 0.0))
        self.assertEqual(explicit.model.serving.cpu_offload_gb, 0.0)
        self.assertEqual(
            run_configuration_hash(explicit),
            run_configuration_hash(omitted),
        )
        offloaded = RunConfiguration.from_dict(_with("model.serving.cpu_offload_gb", 2.5))
        self.assertEqual(offloaded.model.serving.cpu_offload_gb, 2.5)
        self.assertEqual(offloaded.model.serving.to_dict()["cpu_offload_gb"], 2.5)
        self.assertNotEqual(
            run_configuration_hash(offloaded),
            run_configuration_hash(omitted),
        )
        self.assertEqual(
            hashed_values(offloaded)["model.serving.cpu_offload_gb"],
            2.5,
        )
        for bad in (-1.0, True, 1, float("nan"), "1.0"):
            with self.subTest(bad=bad):
                with self.assertRaises(ConfigError):
                    RunConfiguration.from_dict(_with("model.serving.cpu_offload_gb", bad))

        root = Path(configuration_module.__file__).parents[2]
        task = TaskConfiguration.from_dict(
            {
                "appworld_version": "0.1.3.post1",
                "split": "train",
                "selection_rule": "fixed_spotify_capability",
                "selection_seed": 17,
                "task_count": 20,
                "task_set_hash": "a20fe52d28164e1d458266331c242277788d2ed0af29b054b7df926345db3a04",
                "appworld_setup_profile": "spotify_authenticated_v1",
            }
        )
        expected = {
            "qwen3_14b_awq_spotify_capability.json": (
                "165c1a0fcbd3851d77068666562d0c749517a640927dec0393fe3fb055a3fa37"
            ),
            "qwen3_14b_awq_spotify_capability_v2.json": (
                "a4b2b909503bbde2380983b3161a1e4d88ae26e51e0b398c46200210cef05d70"
            ),
            "qwen3_14b_awq_spotify_capability_v2_interface.json": (
                "b58eae2cd1a5b01d21076dbf453c72e22296d7faee833f970b7c655a1a7bf43b"
            ),
            "qwen3_4b_production.json": (
                "573c849928904174962bc6e22c0c18d2b8adbe1b81efbe6479956492ecc622f7"
            ),
            "qwen3_4b_spotify_capability.json": (
                "a033866b6ef626cd60eb31fdf6a1f46675fedf47dad01eb5470b7ca2bee13d3f"
            ),
        }
        for name, digest in expected.items():
            with self.subTest(name=name):
                document = json.loads((root / "configs" / "models" / name).read_text())
                self.assertNotIn("cpu_offload_gb", document["model"]["serving"])
                self.assertIsInstance(
                    load_model_configuration(document["model"]),
                    ModelConfiguration,
                )
                configuration = RunConfiguration(
                    model=load_model_configuration(document["model"]),
                    agent=AgentConfiguration.from_dict(document["agent"]),
                    task=task,
                    run_seed=17,
                    git_commit="a" * 40,
                    protocol_hash=None,
                )
                self.assertEqual(configuration.model.serving.cpu_offload_gb, 0.0)
                self.assertEqual(run_configuration_hash(configuration), digest)

    def test_later_vllm_configurations_keep_their_hashes(self) -> None:
        root = Path(configuration_module.__file__).parents[2]
        task = TaskConfiguration.from_dict(
            {
                "appworld_version": "0.1.3.post1",
                "split": "train",
                "selection_rule": "fixed_spotify_capability",
                "selection_seed": 17,
                "task_count": 20,
                "task_set_hash": "a20fe52d28164e1d458266331c242277788d2ed0af29b054b7df926345db3a04",
                "appworld_setup_profile": "spotify_authenticated_v1",
            }
        )
        expected = {
            "qwen3_32b_awq_spotify_capability_v2_interface.json": (
                "d7ddbb5db23642a1855c3e9921c789b8e700d32a0ef14273cd74d1c992f78fb0"
            ),
            "qwen3_32b_awq_spotify_capability_v2_interface_28672.json": (
                "c1b7d7e0b4ffec6ebc2c0fe511e3abe184ef99a1d8a589afa0efa3838002b735"
            ),
            "qwen3_32b_awq_spotify_capability_v2_interface_32768.json": (
                "8719f07ad06a65f9980c2baa549829bcd8588bb7c34fc7c4323fd725ae51cbfe"
            ),
            "qwen3_32b_awq_spotify_short_horizon_diagnostic.json": (
                "8719f07ad06a65f9980c2baa549829bcd8588bb7c34fc7c4323fd725ae51cbfe"
            ),
        }
        for name, digest in expected.items():
            with self.subTest(name=name):
                document = json.loads((root / "configs" / "models" / name).read_text())
                model = load_model_configuration(document["model"])
                self.assertIsInstance(model, ModelConfiguration)
                configuration = RunConfiguration(
                    model=model,
                    agent=AgentConfiguration.from_dict(document["agent"]),
                    task=task,
                    run_seed=17,
                    git_commit="a" * 40,
                    protocol_hash=None,
                )
                self.assertEqual(run_configuration_hash(configuration), digest)
                self.assertNotIn("provider", configuration.model.to_dict())

    def test_hashed_values_reads_unset_optional_leaves_as_missing(self) -> None:
        payload = _payload()
        del payload["model"]["lora"]
        del payload["agent"]["api_docs_version"]
        del payload["agent"]["api_docs_app"]
        del payload["agent"]["tool_access_profile"]
        unset = RunConfiguration.from_dict(payload)
        values = hashed_values(unset)
        self.assertIs(values["model.lora.repository"], MISSING_HASHED_LEAF)
        self.assertIs(values["model.lora.revision"], MISSING_HASHED_LEAF)
        self.assertIs(values["agent.api_docs_version"], MISSING_HASHED_LEAF)
        self.assertIs(values["agent.api_docs_app"], MISSING_HASHED_LEAF)
        self.assertIs(values["agent.tool_access_profile"], MISSING_HASHED_LEAF)
        self.assertEqual(MISSING_HASHED_LEAF, MISSING_HASHED_LEAF)

        present = hashed_values(_configuration())
        self.assertEqual(present["model.lora.repository"], "org/adapter-base")
        self.assertEqual(present["agent.api_docs_app"], "calendar")
        self.assertEqual(set(values), set(HASHED_FIELDS))

    def test_optional_agent_config_field_omitted_from_generic_flatten(self) -> None:
        without_docs = {
            key: value
            for key, value in _payload()["agent"].items()
            if key not in ("api_docs_version", "api_docs_app")
        }
        agent = AgentConfiguration.from_dict(without_docs)
        self.assertIsNone(agent.api_docs_version)
        document = agent.to_dict()
        self.assertNotIn("api_docs_version", document)
        self.assertNotIn("api_docs_app", document)


_WORKFLOW_LEAVES = (
    "agent.workflow.policy",
    "agent.workflow.repeat_action_limit",
    "agent.workflow.no_progress_turns",
    "agent.workflow.completion_gate",
    "agent.workflow.max_plan_steps",
)


class WorkflowConfigurationTests(unittest.TestCase):
    def test_omitted_workflow_is_absent_and_hashes_as_missing(self) -> None:
        payload = _payload()
        del payload["agent"]["workflow"]
        configuration = RunConfiguration.from_dict(payload)
        self.assertIsNone(configuration.agent.workflow)
        self.assertNotIn("workflow", configuration.agent.to_dict())
        self.assertNotIn("workflow", configuration.to_dict()["agent"])
        values = hashed_values(configuration)
        for path in _WORKFLOW_LEAVES:
            self.assertIs(values[path], MISSING_HASHED_LEAF)

    def test_configured_workflow_round_trips(self) -> None:
        configuration = _configuration()
        workflow = configuration.agent.workflow
        self.assertIsNotNone(workflow)
        assert workflow is not None
        self.assertEqual(workflow.policy, "plan_progress_v1")
        self.assertEqual(workflow.repeat_action_limit, 2)
        self.assertEqual(workflow.no_progress_turns, 3)
        self.assertIs(workflow.completion_gate, True)
        self.assertEqual(workflow.max_plan_steps, 5)
        restored = RunConfiguration.from_dict(configuration.to_dict())
        self.assertEqual(restored.agent.workflow, workflow)
        self.assertIs(restored.agent.workflow.completion_gate, True)

    def test_workflow_policy_changes_the_hash(self) -> None:
        configured = _configuration()
        payload = _payload()
        del payload["agent"]["workflow"]
        omitted = RunConfiguration.from_dict(payload)
        self.assertNotEqual(
            run_configuration_hash(configured),
            run_configuration_hash(omitted),
        )
        self.assertEqual(
            hashed_values(configured)["agent.workflow.policy"],
            "plan_progress_v1",
        )
        self.assertIs(
            hashed_values(omitted)["agent.workflow.policy"],
            MISSING_HASHED_LEAF,
        )

    def test_each_mutable_workflow_leaf_changes_the_hash(self) -> None:
        base_hash = run_configuration_hash(_configuration())
        changes = {
            "agent.workflow.repeat_action_limit": 3,
            "agent.workflow.no_progress_turns": 4,
            "agent.workflow.completion_gate": False,
            "agent.workflow.max_plan_steps": 6,
        }
        for path, value in changes.items():
            with self.subTest(path=path):
                changed = RunConfiguration.from_dict(_with(path, value))
                self.assertNotEqual(run_configuration_hash(changed), base_hash)
                self.assertEqual(hashed_values(changed)[path], value)

    def test_workflow_settings_reject_closed_bounds(self) -> None:
        cases = {
            "unknown policy": ("agent.workflow.policy", "other_policy"),
            "repeat limit 1": ("agent.workflow.repeat_action_limit", 1),
            "max plan steps 1": ("agent.workflow.max_plan_steps", 1),
            "max plan steps 9": ("agent.workflow.max_plan_steps", 9),
        }
        for label, (path, value) in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ConfigError):
                    RunConfiguration.from_dict(_with(path, value))

    def test_capability_configs_enable_workflow_only_on_14b(self) -> None:
        root = Path(configuration_module.__file__).parents[2]
        pilot = json.loads(
            (root / "configs/models/qwen3_14b_awq_spotify_capability.json").read_text(
                encoding="utf-8"
            )
        )
        previous = json.loads(
            (root / "configs/models/qwen3_4b_spotify_capability.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            pilot["agent"]["workflow"],
            {
                "policy": "plan_progress_v1",
                "repeat_action_limit": 2,
                "no_progress_turns": 3,
                "completion_gate": True,
                "max_plan_steps": 5,
            },
        )
        self.assertEqual(
            pilot["agent"]["prompt"]["prompt_version"],
            "prompt-runtime-auth-v1",
        )
        self.assertEqual(pilot["agent"]["execute_max_model_turns"], 20)
        self.assertEqual(pilot["agent"]["sampling"]["execute_max_tokens"], 192)
        self.assertEqual(pilot["agent"]["sampling"]["temperature"], 0.0)
        self.assertEqual(pilot["agent"]["sampling"]["seed"], 17)
        self.assertEqual(pilot["model"]["model"]["repository"], "Qwen/Qwen3-14B-AWQ")
        self.assertEqual(
            pilot["model"]["model"]["revision"],
            "31c69efc29464b6bb0aee1398b5a7b50a99340c3",
        )
        self.assertEqual(pilot["model"]["serving"]["dtype"], "float16")
        self.assertEqual(pilot["model"]["serving"]["kv_cache_dtype"], "float16")
        self.assertNotIn("workflow", previous["agent"])
        successor = json.loads(
            (
                root / "configs/models/qwen3_14b_awq_spotify_capability_v2.json"
            ).read_text(encoding="utf-8")
        )
        v1_agent = dict(pilot["agent"])
        v2_agent = dict(successor["agent"])
        v1_workflow = dict(v1_agent.pop("workflow"))
        v2_workflow = dict(v2_agent.pop("workflow"))
        self.assertEqual(v1_workflow.pop("policy"), "plan_progress_v1")
        self.assertEqual(v2_workflow.pop("policy"), "plan_progress_v2")
        self.assertEqual(v1_workflow, v2_workflow)
        self.assertEqual(v1_agent, v2_agent)
        self.assertEqual(pilot["model"], successor["model"])

    def test_plan_progress_v2_is_a_distinct_hashed_policy(self) -> None:
        base = _configuration()
        changed = RunConfiguration.from_dict(
            _with("agent.workflow.policy", "plan_progress_v2")
        )
        self.assertNotEqual(
            run_configuration_hash(base),
            run_configuration_hash(changed),
        )
        self.assertEqual(
            hashed_values(changed)["agent.workflow.policy"],
            "plan_progress_v2",
        )
        for path in _WORKFLOW_LEAVES:
            if path == "agent.workflow.policy":
                continue
            self.assertEqual(hashed_values(base)[path], hashed_values(changed)[path])


if __name__ == "__main__":
    unittest.main()
