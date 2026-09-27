"""Configuration schemas, the run-configuration hash, and run identity.

A run configuration is one model configuration, one agent configuration,
and one task configuration, plus the run seed, git commit, and protocol
hash. Values are checked when an object is constructed. Dictionary
loading rejects unknown fields, so a misspelled setting cannot be
dropped. The protocol hash may be omitted or null before the protocol
is locked. This module does not read process environment variables, the
git repository, or task contents, and it does not invent a production
configuration.

The configuration hash is the SHA-256 hex digest of the canonical JSON
for a run configuration. Canonical JSON sorts keys inside every object,
uses the separators "," and ":", is encoded as UTF-8, and rejects
non-finite numbers. Insertion order does not change the digest.

HASHED_FIELDS is the complete set of leaves that enter the hash. Changing
any of them changes the digest:

model.model.repository
model.model.revision
model.tokenizer.repository
model.tokenizer.revision
model.quantization.method
model.vllm_version
model.serving.dtype
model.serving.max_model_len
model.serving.gpu_memory_utilization
model.serving.max_num_seqs
model.serving.max_num_batched_tokens
model.serving.kv_cache_dtype
model.serving.enable_prefix_caching
model.serving.enable_chunked_prefill
model.serving.enforce_eager
model.serving.tensor_parallel_size
model.serving.max_logprobs
model.serving.batch_invariant
agent.smolagents_version
agent.action_interface
agent.prompt.prompt_version
agent.prompt.plan_format_version
agent.prompt.thinking_enabled
agent.step_limit
agent.sampling.temperature
agent.sampling.top_p
agent.sampling.top_k
agent.sampling.min_p
agent.sampling.seed
agent.sampling.max_tokens
task.appworld_version
task.split
task.selection_rule
task.selection_seed
task.task_count
task.task_set_hash
run_seed
git_commit
protocol_hash

Serving leaves are the vLLM settings that change numerics, context
truncation, batch formation, or recorded log-probabilities. Sampling
leaves are the request settings that change token choice or length.
thinking_enabled records whether the chat template requests thinking
mode. The task schema has no task id and no task content; task_set_hash
is a supplied digest, not a value computed here.

RUNTIME_IDS are run_id, episode_id, and pair_id. They are minted when a
run or episode is opened, they are not configuration fields, and they
do not enter the hash. task_set_hash, protocol_hash, and git_commit are
copied from the supplied configuration onto a run identity.

Closed tokens: splits are train, dev, test_normal, and test_challenge;
action interfaces are code and tool_calling; quantization methods are
none, fp8, and nvfp4; model dtypes are bfloat16, float16, and float32;
KV-cache dtypes are bfloat16, float16, and fp8. Model revisions,
tokenizer revisions, and git_commit are 40-character lowercase git ids.
task_set_hash and a present protocol_hash are 64-character lowercase
SHA-256 digests. top_k is -1 or a positive integer. The sampling seed
is required.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from dataclasses import dataclass, fields, is_dataclass
from typing import Callable, Mapping, TypeVar, get_type_hints

_T = TypeVar("_T")

SPLITS = frozenset({"train", "dev", "test_normal", "test_challenge"})
ACTION_INTERFACES = frozenset({"code", "tool_calling"})
QUANTIZATION_METHODS = frozenset({"none", "fp8", "nvfp4"})
MODEL_DTYPES = frozenset({"bfloat16", "float16", "float32"})
KV_CACHE_DTYPES = frozenset({"bfloat16", "float16", "fp8"})
HASHED_FIELDS = frozenset(
    {
        "model.model.repository",
        "model.model.revision",
        "model.tokenizer.repository",
        "model.tokenizer.revision",
        "model.quantization.method",
        "model.vllm_version",
        "model.serving.dtype",
        "model.serving.max_model_len",
        "model.serving.gpu_memory_utilization",
        "model.serving.max_num_seqs",
        "model.serving.max_num_batched_tokens",
        "model.serving.kv_cache_dtype",
        "model.serving.enable_prefix_caching",
        "model.serving.enable_chunked_prefill",
        "model.serving.enforce_eager",
        "model.serving.tensor_parallel_size",
        "model.serving.max_logprobs",
        "model.serving.batch_invariant",
        "agent.smolagents_version",
        "agent.action_interface",
        "agent.prompt.prompt_version",
        "agent.prompt.plan_format_version",
        "agent.prompt.thinking_enabled",
        "agent.step_limit",
        "agent.sampling.temperature",
        "agent.sampling.top_p",
        "agent.sampling.top_k",
        "agent.sampling.min_p",
        "agent.sampling.seed",
        "agent.sampling.max_tokens",
        "task.appworld_version",
        "task.split",
        "task.selection_rule",
        "task.selection_seed",
        "task.task_count",
        "task.task_set_hash",
        "run_seed",
        "git_commit",
        "protocol_hash",
    }
)
RUNTIME_IDS = frozenset({"run_id", "episode_id", "pair_id"})

_MAX_TEXT = 256
_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUNTIME_TOKEN = re.compile(r"^[0-9a-f]{32}$")
_RUN_ID = re.compile(r"^[0-9a-f]{64}\.[0-9a-f]{32}$")


class ConfigError(ValueError):
    pass


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or value == "" or value != value.strip():
        raise ConfigError(f"{name} must be a non-empty string")
    if len(value) > _MAX_TEXT or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise ConfigError(
            f"{name} must be a single line of at most {_MAX_TEXT} characters"
        )
    return value


def _git_revision(value: object, name: str) -> str:
    if not isinstance(value, str) or _GIT_REVISION.fullmatch(value) is None:
        raise ConfigError(
            f"{name} must be a 40-character lowercase git revision"
        )
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ConfigError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _optional_sha256(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, name)


def _flag(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be a boolean")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer")
    return value


def _positive(value: object, name: str) -> int:
    number = _integer(value, name)
    if number < 1:
        raise ConfigError(f"{name} must be a positive integer")
    return number


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, float):
        raise ConfigError(f"{name} must be a float")
    if not math.isfinite(value):
        raise ConfigError(f"{name} must be finite")
    return value


def _open_unit(value: object, name: str) -> float:
    number = _finite_float(value, name)
    if not 0.0 < number <= 1.0:
        raise ConfigError(
            f"{name} must be greater than zero and at most one"
        )
    return number


def _closed_unit(value: object, name: str) -> float:
    number = _finite_float(value, name)
    if not 0.0 <= number <= 1.0:
        raise ConfigError(f"{name} must be between zero and one")
    return number


def _temperature(value: object) -> float:
    number = _finite_float(value, "temperature")
    if number < 0.0:
        raise ConfigError("temperature must be zero or greater")
    return number


def _top_k(value: object) -> int:
    number = _integer(value, "top_k")
    if number == -1 or number >= 1:
        return number
    raise ConfigError("top_k must be -1 or a positive integer")


def _choice(value: object, allowed: frozenset[str], name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ConfigError(f"{name} must be one of: {choices}")
    return value


def _kind(value: object, cls: type, name: str) -> None:
    if not isinstance(value, cls):
        raise ConfigError(f"{name} must be a {cls.__name__}")


def _object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be an object")
    return value


def _require_fields(
    payload: Mapping[str, object],
    cls: type,
    name: str,
    optional: frozenset[str] = frozenset(),
) -> None:
    if not all(isinstance(key, str) for key in payload):
        raise ConfigError(f"{name} has a non-string field name")
    allowed = frozenset(field.name for field in fields(cls))
    unknown = sorted(set(payload) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise ConfigError(f"{name} contains unknown fields: {joined}")
    missing = sorted(allowed - optional - set(payload))
    if missing:
        joined = ", ".join(missing)
        raise ConfigError(f"{name} is missing fields: {joined}")


def _construct(name: str, factory: Callable[[], _T]) -> _T:
    try:
        return factory()
    except ConfigError as error:
        message = str(error)
        if message.startswith(f"{name} ") or message.startswith(f"{name}:"):
            raise
        raise ConfigError(f"{name}: {message}") from error


def _plain_dict(value: object, cls: type) -> dict[str, object]:
    hints = get_type_hints(cls)
    payload: dict[str, object] = {}
    for field in fields(cls):
        child = getattr(value, field.name)
        if is_dataclass(hints[field.name]):
            payload[field.name] = _plain_dict(child, hints[field.name])
        else:
            payload[field.name] = child
    return payload


def _runtime_token(value: object, name: str) -> str:
    if not isinstance(value, str) or _RUNTIME_TOKEN.fullmatch(value) is None:
        raise ConfigError(f"{name} must be a 32-character lowercase hex id")
    return value


@dataclass(frozen=True)
class ModelRevision:
    repository: str
    revision: str

    def __post_init__(self) -> None:
        _text(self.repository, "repository")
        _git_revision(self.revision, "revision")

    def to_dict(self) -> dict[str, object]:
        return _plain_dict(self, ModelRevision)

    @classmethod
    def from_dict(
        cls,
        payload: object,
        name: str = "model revision",
    ) -> ModelRevision:
        mapping = _object(payload, name)
        _require_fields(mapping, cls, name)
        return _construct(
            name,
            lambda: cls(
                repository=mapping["repository"],
                revision=mapping["revision"],
            ),
        )


@dataclass(frozen=True)
class QuantizationSettings:
    method: str

    def __post_init__(self) -> None:
        _choice(self.method, QUANTIZATION_METHODS, "method")

    def to_dict(self) -> dict[str, object]:
        return _plain_dict(self, QuantizationSettings)

    @classmethod
    def from_dict(
        cls,
        payload: object,
        name: str = "quantization",
    ) -> QuantizationSettings:
        mapping = _object(payload, name)
        _require_fields(mapping, cls, name)
        return _construct(name, lambda: cls(method=mapping["method"]))


@dataclass(frozen=True)
class VLLMBehaviorSettings:
    dtype: str
    max_model_len: int
    gpu_memory_utilization: float
    max_num_seqs: int
    max_num_batched_tokens: int
    kv_cache_dtype: str
    enable_prefix_caching: bool
    enable_chunked_prefill: bool
    enforce_eager: bool
    tensor_parallel_size: int
    max_logprobs: int
    batch_invariant: bool

    def __post_init__(self) -> None:
        _choice(self.dtype, MODEL_DTYPES, "dtype")
        _positive(self.max_model_len, "max_model_len")
        _open_unit(self.gpu_memory_utilization, "gpu_memory_utilization")
        _positive(self.max_num_seqs, "max_num_seqs")
        _positive(self.max_num_batched_tokens, "max_num_batched_tokens")
        _choice(self.kv_cache_dtype, KV_CACHE_DTYPES, "kv_cache_dtype")
        _flag(self.enable_prefix_caching, "enable_prefix_caching")
        _flag(self.enable_chunked_prefill, "enable_chunked_prefill")
        _flag(self.enforce_eager, "enforce_eager")
        _positive(self.tensor_parallel_size, "tensor_parallel_size")
        _positive(self.max_logprobs, "max_logprobs")
        _flag(self.batch_invariant, "batch_invariant")

    def to_dict(self) -> dict[str, object]:
        return _plain_dict(self, VLLMBehaviorSettings)

    @classmethod
    def from_dict(
        cls,
        payload: object,
        name: str = "serving",
    ) -> VLLMBehaviorSettings:
        mapping = _object(payload, name)
        _require_fields(mapping, cls, name)
        return _construct(
            name,
            lambda: cls(
                dtype=mapping["dtype"],
                max_model_len=mapping["max_model_len"],
                gpu_memory_utilization=mapping["gpu_memory_utilization"],
                max_num_seqs=mapping["max_num_seqs"],
                max_num_batched_tokens=mapping["max_num_batched_tokens"],
                kv_cache_dtype=mapping["kv_cache_dtype"],
                enable_prefix_caching=mapping["enable_prefix_caching"],
                enable_chunked_prefill=mapping["enable_chunked_prefill"],
                enforce_eager=mapping["enforce_eager"],
                tensor_parallel_size=mapping["tensor_parallel_size"],
                max_logprobs=mapping["max_logprobs"],
                batch_invariant=mapping["batch_invariant"],
            ),
        )


@dataclass(frozen=True)
class ModelConfiguration:
    model: ModelRevision
    tokenizer: ModelRevision
    quantization: QuantizationSettings
    vllm_version: str
    serving: VLLMBehaviorSettings

    def __post_init__(self) -> None:
        _kind(self.model, ModelRevision, "model")
        _kind(self.tokenizer, ModelRevision, "tokenizer")
        _kind(self.quantization, QuantizationSettings, "quantization")
        _text(self.vllm_version, "vllm_version")
        _kind(self.serving, VLLMBehaviorSettings, "serving")

    def to_dict(self) -> dict[str, object]:
        return _plain_dict(self, ModelConfiguration)

    @classmethod
    def from_dict(
        cls,
        payload: object,
        name: str = "model configuration",
    ) -> ModelConfiguration:
        mapping = _object(payload, name)
        _require_fields(mapping, cls, name)
        model = ModelRevision.from_dict(mapping["model"], name="model")
        tokenizer = ModelRevision.from_dict(
            mapping["tokenizer"],
            name="tokenizer",
        )
        quantization = QuantizationSettings.from_dict(mapping["quantization"])
        serving = VLLMBehaviorSettings.from_dict(mapping["serving"])
        return _construct(
            name,
            lambda: cls(
                model=model,
                tokenizer=tokenizer,
                quantization=quantization,
                vllm_version=mapping["vllm_version"],
                serving=serving,
            ),
        )


@dataclass(frozen=True)
class PromptSettings:
    prompt_version: str
    plan_format_version: str
    thinking_enabled: bool

    def __post_init__(self) -> None:
        _text(self.prompt_version, "prompt_version")
        _text(self.plan_format_version, "plan_format_version")
        _flag(self.thinking_enabled, "thinking_enabled")

    def to_dict(self) -> dict[str, object]:
        return _plain_dict(self, PromptSettings)

    @classmethod
    def from_dict(
        cls,
        payload: object,
        name: str = "prompt",
    ) -> PromptSettings:
        mapping = _object(payload, name)
        _require_fields(mapping, cls, name)
        return _construct(
            name,
            lambda: cls(
                prompt_version=mapping["prompt_version"],
                plan_format_version=mapping["plan_format_version"],
                thinking_enabled=mapping["thinking_enabled"],
            ),
        )


@dataclass(frozen=True)
class SamplingSettings:
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    seed: int
    max_tokens: int

    def __post_init__(self) -> None:
        _temperature(self.temperature)
        _open_unit(self.top_p, "top_p")
        _top_k(self.top_k)
        _closed_unit(self.min_p, "min_p")
        _integer(self.seed, "seed")
        _positive(self.max_tokens, "max_tokens")

    def to_dict(self) -> dict[str, object]:
        return _plain_dict(self, SamplingSettings)

    @classmethod
    def from_dict(
        cls,
        payload: object,
        name: str = "sampling",
    ) -> SamplingSettings:
        mapping = _object(payload, name)
        _require_fields(mapping, cls, name)
        return _construct(
            name,
            lambda: cls(
                temperature=mapping["temperature"],
                top_p=mapping["top_p"],
                top_k=mapping["top_k"],
                min_p=mapping["min_p"],
                seed=mapping["seed"],
                max_tokens=mapping["max_tokens"],
            ),
        )


@dataclass(frozen=True)
class AgentConfiguration:
    smolagents_version: str
    action_interface: str
    prompt: PromptSettings
    step_limit: int
    sampling: SamplingSettings

    def __post_init__(self) -> None:
        _text(self.smolagents_version, "smolagents_version")
        _choice(self.action_interface, ACTION_INTERFACES, "action_interface")
        _kind(self.prompt, PromptSettings, "prompt")
        _positive(self.step_limit, "step_limit")
        _kind(self.sampling, SamplingSettings, "sampling")

    def to_dict(self) -> dict[str, object]:
        return _plain_dict(self, AgentConfiguration)

    @classmethod
    def from_dict(
        cls,
        payload: object,
        name: str = "agent configuration",
    ) -> AgentConfiguration:
        mapping = _object(payload, name)
        _require_fields(mapping, cls, name)
        prompt = PromptSettings.from_dict(mapping["prompt"])
        sampling = SamplingSettings.from_dict(mapping["sampling"])
        return _construct(
            name,
            lambda: cls(
                smolagents_version=mapping["smolagents_version"],
                action_interface=mapping["action_interface"],
                prompt=prompt,
                step_limit=mapping["step_limit"],
                sampling=sampling,
            ),
        )


@dataclass(frozen=True)
class TaskConfiguration:
    appworld_version: str
    split: str
    selection_rule: str
    selection_seed: int
    task_count: int
    task_set_hash: str

    def __post_init__(self) -> None:
        _text(self.appworld_version, "appworld_version")
        _choice(self.split, SPLITS, "split")
        _text(self.selection_rule, "selection_rule")
        _integer(self.selection_seed, "selection_seed")
        _positive(self.task_count, "task_count")
        _sha256(self.task_set_hash, "task_set_hash")

    def to_dict(self) -> dict[str, object]:
        return _plain_dict(self, TaskConfiguration)

    @classmethod
    def from_dict(
        cls,
        payload: object,
        name: str = "task configuration",
    ) -> TaskConfiguration:
        mapping = _object(payload, name)
        _require_fields(mapping, cls, name)
        return _construct(
            name,
            lambda: cls(
                appworld_version=mapping["appworld_version"],
                split=mapping["split"],
                selection_rule=mapping["selection_rule"],
                selection_seed=mapping["selection_seed"],
                task_count=mapping["task_count"],
                task_set_hash=mapping["task_set_hash"],
            ),
        )


@dataclass(frozen=True)
class RunConfiguration:
    model: ModelConfiguration
    agent: AgentConfiguration
    task: TaskConfiguration
    run_seed: int
    git_commit: str
    protocol_hash: str | None = None

    def __post_init__(self) -> None:
        _kind(self.model, ModelConfiguration, "model")
        _kind(self.agent, AgentConfiguration, "agent")
        _kind(self.task, TaskConfiguration, "task")
        _integer(self.run_seed, "run_seed")
        _git_revision(self.git_commit, "git_commit")
        _optional_sha256(self.protocol_hash, "protocol_hash")

    def to_dict(self) -> dict[str, object]:
        return _plain_dict(self, RunConfiguration)

    @classmethod
    def from_dict(
        cls,
        payload: object,
        name: str = "run configuration",
    ) -> RunConfiguration:
        mapping = _object(payload, name)
        _require_fields(
            mapping,
            cls,
            name,
            optional=frozenset({"protocol_hash"}),
        )
        model = ModelConfiguration.from_dict(mapping["model"])
        agent = AgentConfiguration.from_dict(mapping["agent"])
        task = TaskConfiguration.from_dict(mapping["task"])
        arguments: dict[str, object] = {
            "model": model,
            "agent": agent,
            "task": task,
            "run_seed": mapping["run_seed"],
            "git_commit": mapping["git_commit"],
        }
        if "protocol_hash" in mapping:
            arguments["protocol_hash"] = mapping["protocol_hash"]
        return _construct(name, lambda: cls(**arguments))


def canonical_configuration_json(configuration: RunConfiguration) -> str:
    if not isinstance(configuration, RunConfiguration):
        raise ConfigError("canonical JSON requires a run configuration")
    try:
        return json.dumps(
            configuration.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ConfigError("run configuration is not finite JSON") from error


def run_configuration_hash(configuration: RunConfiguration) -> str:
    document = canonical_configuration_json(configuration)
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def _embeds(value: object, prefix: str, name: str) -> str:
    if not isinstance(value, str) or not value.startswith(prefix):
        raise ConfigError(f"{name} must embed {prefix[:-1]}")
    suffix = value[len(prefix) :]
    if _RUNTIME_TOKEN.fullmatch(suffix) is None:
        raise ConfigError(f"{name} must embed {prefix[:-1]}")
    return value


@dataclass(frozen=True)
class RunIdentity:
    run_id: str
    configuration_hash: str
    task_set_hash: str
    protocol_hash: str | None
    git_commit: str

    def __post_init__(self) -> None:
        _sha256(self.configuration_hash, "configuration_hash")
        _sha256(self.task_set_hash, "task_set_hash")
        _optional_sha256(self.protocol_hash, "protocol_hash")
        _git_revision(self.git_commit, "git_commit")
        _embeds(self.run_id, f"{self.configuration_hash}.", "run_id")


@dataclass(frozen=True)
class EpisodeIdentity:
    episode_id: str
    run_id: str
    pair_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or _RUN_ID.fullmatch(self.run_id) is None:
            raise ConfigError("run_id must tie an episode to a configuration hash")
        _embeds(self.episode_id, f"{self.run_id}.", "episode_id")
        if self.pair_id is not None:
            _runtime_token(self.pair_id, "pair_id")


def new_run_identity(configuration: RunConfiguration) -> RunIdentity:
    if not isinstance(configuration, RunConfiguration):
        raise ConfigError("run identity requires a run configuration")
    configuration_hash = run_configuration_hash(configuration)
    return RunIdentity(
        run_id=f"{configuration_hash}.{uuid.uuid4().hex}",
        configuration_hash=configuration_hash,
        task_set_hash=configuration.task.task_set_hash,
        protocol_hash=configuration.protocol_hash,
        git_commit=configuration.git_commit,
    )


def new_pair_id() -> str:
    return uuid.uuid4().hex


def new_episode_identity(
    run: RunIdentity,
    *,
    pair_id: str | None = None,
) -> EpisodeIdentity:
    if not isinstance(run, RunIdentity):
        raise ConfigError("episode identity requires a run identity")
    return EpisodeIdentity(
        episode_id=f"{run.run_id}.{uuid.uuid4().hex}",
        run_id=run.run_id,
        pair_id=pair_id,
    )
