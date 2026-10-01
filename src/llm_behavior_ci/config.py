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
model.serving.sampler_backend
model.lora.repository
model.lora.revision
agent.smolagents_version
agent.action_interface
agent.prompt.prompt_version
agent.prompt.plan_format_version
agent.prompt.thinking_enabled
agent.step_limit
agent.execute_max_model_turns
agent.sampling.temperature
agent.sampling.top_p
agent.sampling.top_k
agent.sampling.min_p
agent.sampling.seed
agent.sampling.max_tokens
agent.sampling.execute_max_tokens
agent.sampling.plan_max_tokens
agent.api_docs_version
agent.api_docs_app
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

StreamSettings, GateSettings, CanarySettings, and MonitorSettings are
the shared inputs for the task stream, the offline gate, the canary,
and production monitors. Lifecycle code imports them from this module.
Every rate, delay, horizon, fraction, margin, level, and stopping rule
is a constructor argument. These types contain no protocol values, and
none of their fields enter the run-configuration hash.

Closed tokens: splits are train, dev, test_normal, and test_challenge;
action interfaces are code and tool_calling; quantization methods are
none, fp8, and nvfp4; model dtypes are bfloat16, float16, and float32;
KV-cache dtypes are bfloat16, float16, and fp8. Model revisions,
tokenizer revisions, and git_commit are 40-character lowercase git ids.
task_set_hash and a present protocol_hash are 64-character lowercase
SHA-256 digests. top_k is -1 or a positive integer. The sampling seed
is required. KL fidelity modes are full and top_k; runtime/scoring.py
is where full is proven rather than merely claimed.

model.lora.repository, model.lora.revision, agent.api_docs_version,
agent.api_docs_app, agent.execute_max_model_turns,
agent.sampling.execute_max_tokens, and agent.sampling.plan_max_tokens are
optional hashed leaves: ``ModelConfiguration.lora``,
``AgentConfiguration.api_docs_version``/``api_docs_app``,
``AgentConfiguration.execute_max_model_turns``, and the two mode-specific
``SamplingSettings`` token caps default to unset, and an unset leaf is omitted from canonical JSON entirely rather
than serialized as null, so a configuration that never names a LoRA
adapter or a corrupted API-documentation source hashes identically to one
built before these fields existed. Setting, clearing, or changing any of
them still changes the digest, because canonical JSON then differs.
``leaf_value``/``hashed_values`` read these leaves through
``MISSING_HASHED_LEAF`` rather than raising, so a caller comparing two
configurations' hashed fields sees "unset" as one comparable value
instead of a lookup error.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable, Mapping, TypeVar, get_args, get_type_hints

_T = TypeVar("_T")

SPLITS = frozenset({"train", "dev", "test_normal", "test_challenge"})
ACTION_INTERFACES = frozenset({"code", "tool_calling"})
QUANTIZATION_METHODS = frozenset({"none", "fp8", "nvfp4"})
MODEL_DTYPES = frozenset({"bfloat16", "float16", "float32"})
KV_CACHE_DTYPES = frozenset({"bfloat16", "float16", "fp8"})
SAMPLER_BACKENDS = frozenset({"flashinfer", "native"})
KL_FIDELITY_MODES = frozenset({"full", "top_k"})
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
        "model.serving.sampler_backend",
        "model.lora.repository",
        "model.lora.revision",
        "agent.smolagents_version",
        "agent.action_interface",
        "agent.prompt.prompt_version",
        "agent.prompt.plan_format_version",
        "agent.prompt.thinking_enabled",
        "agent.step_limit",
        "agent.execute_max_model_turns",
        "agent.sampling.temperature",
        "agent.sampling.top_p",
        "agent.sampling.top_k",
        "agent.sampling.min_p",
        "agent.sampling.seed",
        "agent.sampling.max_tokens",
        "agent.sampling.execute_max_tokens",
        "agent.sampling.plan_max_tokens",
        "agent.api_docs_version",
        "agent.api_docs_app",
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
_RULE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
MONITOR_SIGNALS = frozenset(
    {
        "task_success",
        "requirement_fraction",
        "tool_error_count",
        "invalid_tool_call_count",
        "trajectory_length",
        "plan_quality_score",
        "plan_kl_mean_nats",
    }
)
"""The closed scalar series a ``MonitorObservation`` may carry.

``plan_quality_score`` folds a semantic plan-feature vector (``lifecycle.
plan_features``) into one bounded [0, 1] read, when a caller schedules
plan-quality scoring alongside production traffic. ``plan_kl_mean_nats``
folds one teacher-forced plan-KL result (``stats.kl`` /
``runtime.scoring``) into one nonnegative read, when a caller schedules
teacher-forced scoring. Both are optional: a production episode with
neither scheduled emits neither observation. ``tool_selection`` and
``task_mix`` are distribution-valued and are never members of this set;
they are typed observations (``ToolSelectionObservation``,
``TaskMixObservation``) routed through ``DistributionalMonitorSettings``
and a windowed categorical test, never coerced into this scalar series.
"""

DISTRIBUTIONAL_SIGNALS = frozenset({"tool_selection", "task_mix"})
DISTRIBUTIONAL_CORRECTIONS = frozenset({"none", "bonferroni"})


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


def _open_probability(value: object, name: str) -> float:
    number = _finite_float(value, name)
    if not 0.0 < number < 1.0:
        raise ConfigError(
            f"{name} must be greater than zero and less than one"
        )
    return number


def _positive_float(value: object, name: str) -> float:
    number = _finite_float(value, name)
    if number <= 0.0:
        raise ConfigError(f"{name} must be greater than zero")
    return number


def _nonnegative_float(value: object, name: str) -> float:
    number = _finite_float(value, name)
    if number < 0.0:
        raise ConfigError(f"{name} must be zero or greater")
    return number


def _rule_name(value: object, name: str) -> str:
    if not isinstance(value, str) or _RULE_NAME.fullmatch(value) is None:
        raise ConfigError(f"{name} must be a lowercase identifier")
    return value


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


def _load(
    cls: type[_T],
    mapping: Mapping[str, object],
    **overrides: object,
) -> _T:
    arguments: dict[str, Any] = {}
    for key, value in mapping.items():
        arguments[key] = value
    arguments.update(overrides)
    return cls(**arguments)


def _optional_dataclass_hint(hint: object) -> type | None:
    """The dataclass a field's type hint names, whether or not it is optional.

    Handles ``SomeDataclass`` directly and ``SomeDataclass | None``
    (equivalently ``Optional[SomeDataclass]``); returns ``None`` for every
    other hint, including a plain ``str | None``.
    """

    if is_dataclass(hint):
        return hint  # type: ignore[return-value]
    args = get_args(hint)
    if args and type(None) in args:
        remaining = [arg for arg in args if arg is not type(None)]
        if len(remaining) == 1 and is_dataclass(remaining[0]):
            return remaining[0]
    return None


def _plain_dict(value: object, cls: type) -> dict[str, object]:
    """Flatten one dataclass into a plain JSON-able dict, field by field.

    A nested dataclass field (required, or ``SomeDataclass | None``) defers
    to that field's own ``to_dict()`` rather than re-flattening it
    structurally here, so a class with a custom ``to_dict()`` (an optional
    nested config that omits its key entirely when unset, such as
    ``ModelConfiguration.lora``) is respected however deeply it is nested.
    A ``None`` optional-dataclass field is omitted from the payload
    entirely; every other field, including a plain ``str | None`` such as
    ``RunConfiguration.protocol_hash``, is copied through unchanged,
    ``None`` included.
    """

    hints = get_type_hints(cls)
    payload: dict[str, object] = {}
    for field in fields(cls):
        child = getattr(value, field.name)
        nested = _optional_dataclass_hint(hints[field.name])
        if nested is not None:
            if child is None:
                continue
            to_dict = getattr(child, "to_dict", None)
            payload[field.name] = (
                to_dict() if callable(to_dict) else _plain_dict(child, nested)
            )
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
        return _construct(name, lambda: _load(cls, mapping))


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
        return _construct(name, lambda: _load(cls, mapping))


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
    sampler_backend: str

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
        _choice(self.sampler_backend, SAMPLER_BACKENDS, "sampler_backend")

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
        return _construct(name, lambda: _load(cls, mapping))


@dataclass(frozen=True)
class LoRASettings:
    """A LoRA adapter's identity, when a configuration serves one.

    ``ModelConfiguration.lora`` is ``None`` for the healthy base-model
    configuration; setting it is the ``lora`` fault kind's whole effect.
    Both fields are required together, matching ``ModelRevision``: a
    partially-identified adapter is not a valid configuration.
    """

    repository: str
    revision: str

    def __post_init__(self) -> None:
        _text(self.repository, "repository")
        _git_revision(self.revision, "revision")

    def to_dict(self) -> dict[str, object]:
        return _plain_dict(self, LoRASettings)

    @classmethod
    def from_dict(
        cls,
        payload: object,
        name: str = "lora",
    ) -> LoRASettings:
        mapping = _object(payload, name)
        _require_fields(mapping, cls, name)
        return _construct(name, lambda: _load(cls, mapping))


@dataclass(frozen=True)
class ModelConfiguration:
    model: ModelRevision
    tokenizer: ModelRevision
    quantization: QuantizationSettings
    vllm_version: str
    serving: VLLMBehaviorSettings
    lora: LoRASettings | None = None

    def __post_init__(self) -> None:
        _kind(self.model, ModelRevision, "model")
        _kind(self.tokenizer, ModelRevision, "tokenizer")
        _kind(self.quantization, QuantizationSettings, "quantization")
        _text(self.vllm_version, "vllm_version")
        _kind(self.serving, VLLMBehaviorSettings, "serving")
        if self.lora is not None:
            _kind(self.lora, LoRASettings, "lora")

    def to_dict(self) -> dict[str, object]:
        document: dict[str, object] = {
            "model": self.model.to_dict(),
            "tokenizer": self.tokenizer.to_dict(),
            "quantization": self.quantization.to_dict(),
            "vllm_version": self.vllm_version,
            "serving": self.serving.to_dict(),
        }
        if self.lora is not None:
            document["lora"] = self.lora.to_dict()
        return document

    @classmethod
    def from_dict(
        cls,
        payload: object,
        name: str = "model configuration",
    ) -> ModelConfiguration:
        mapping = _object(payload, name)
        _require_fields(mapping, cls, name, optional=frozenset({"lora"}))
        model = ModelRevision.from_dict(mapping["model"], name="model")
        tokenizer = ModelRevision.from_dict(
            mapping["tokenizer"],
            name="tokenizer",
        )
        quantization = QuantizationSettings.from_dict(mapping["quantization"])
        serving = VLLMBehaviorSettings.from_dict(mapping["serving"])
        lora = (
            LoRASettings.from_dict(mapping["lora"]) if "lora" in mapping else None
        )
        return _construct(
            name,
            lambda: _load(
                cls,
                mapping,
                model=model,
                tokenizer=tokenizer,
                quantization=quantization,
                serving=serving,
                lora=lora,
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
        return _construct(name, lambda: _load(cls, mapping))


@dataclass(frozen=True)
class SamplingSettings:
    """Request sampling settings.

    ``max_tokens`` is the generation cap for every mode unless the mode's
    optional override is set: ``execute_max_tokens`` for execute turns and
    ``plan_max_tokens`` for plan generation. An unset override is omitted
    from ``to_dict``. ``generation_max_tokens`` resolves the cap.
    """

    temperature: float
    top_p: float
    top_k: int
    min_p: float
    seed: int
    max_tokens: int
    execute_max_tokens: int | None = None
    plan_max_tokens: int | None = None

    def __post_init__(self) -> None:
        _temperature(self.temperature)
        _open_unit(self.top_p, "top_p")
        _top_k(self.top_k)
        _closed_unit(self.min_p, "min_p")
        _integer(self.seed, "seed")
        _positive(self.max_tokens, "max_tokens")
        if self.execute_max_tokens is not None:
            _positive(self.execute_max_tokens, "execute_max_tokens")
        if self.plan_max_tokens is not None:
            _positive(self.plan_max_tokens, "plan_max_tokens")

    def generation_max_tokens(self, mode: str) -> int:
        if mode == "execute" and self.execute_max_tokens is not None:
            return self.execute_max_tokens
        if mode == "plan" and self.plan_max_tokens is not None:
            return self.plan_max_tokens
        return self.max_tokens

    def to_dict(self) -> dict[str, object]:
        document = _plain_dict(self, SamplingSettings)
        for key in ("execute_max_tokens", "plan_max_tokens"):
            if document[key] is None:
                del document[key]
        return document

    @classmethod
    def from_dict(
        cls,
        payload: object,
        name: str = "sampling",
    ) -> SamplingSettings:
        mapping = _object(payload, name)
        _require_fields(
            mapping,
            cls,
            name,
            optional=frozenset({"execute_max_tokens", "plan_max_tokens"}),
        )
        return _construct(name, lambda: _load(cls, mapping))


@dataclass(frozen=True)
class AgentConfiguration:
    """One agent's runtime configuration.

    ``api_docs_version``/``api_docs_app`` are set together or not at all:
    the healthy configuration leaves both unset, and the ``api_documentation``
    fault kind's whole effect is naming a corruption transform
    (``api_docs_version``) and the one app it targets (``api_docs_app``).

    ``execute_max_model_turns`` overrides ``step_limit`` as the execute
    episode horizon when set; ``execute_turn_limit`` resolves it. An unset
    override is omitted from ``to_dict``.
    """

    smolagents_version: str
    action_interface: str
    prompt: PromptSettings
    step_limit: int
    sampling: SamplingSettings
    api_docs_version: str | None = None
    api_docs_app: str | None = None
    execute_max_model_turns: int | None = None

    def __post_init__(self) -> None:
        _text(self.smolagents_version, "smolagents_version")
        _choice(self.action_interface, ACTION_INTERFACES, "action_interface")
        _kind(self.prompt, PromptSettings, "prompt")
        _positive(self.step_limit, "step_limit")
        _kind(self.sampling, SamplingSettings, "sampling")
        if (self.api_docs_version is None) != (self.api_docs_app is None):
            raise ConfigError(
                "api_docs_version and api_docs_app must be set together"
            )
        if self.api_docs_version is not None:
            _text(self.api_docs_version, "api_docs_version")
            _text(self.api_docs_app, "api_docs_app")
        if self.execute_max_model_turns is not None:
            _positive(self.execute_max_model_turns, "execute_max_model_turns")

    @property
    def execute_turn_limit(self) -> int:
        if self.execute_max_model_turns is not None:
            return self.execute_max_model_turns
        return self.step_limit

    def to_dict(self) -> dict[str, object]:
        document: dict[str, object] = {
            "smolagents_version": self.smolagents_version,
            "action_interface": self.action_interface,
            "prompt": self.prompt.to_dict(),
            "step_limit": self.step_limit,
            "sampling": self.sampling.to_dict(),
        }
        if self.api_docs_version is not None:
            document["api_docs_version"] = self.api_docs_version
            document["api_docs_app"] = self.api_docs_app
        if self.execute_max_model_turns is not None:
            document["execute_max_model_turns"] = self.execute_max_model_turns
        return document

    @classmethod
    def from_dict(
        cls,
        payload: object,
        name: str = "agent configuration",
    ) -> AgentConfiguration:
        mapping = _object(payload, name)
        _require_fields(
            mapping,
            cls,
            name,
            optional=frozenset(
                {"api_docs_version", "api_docs_app", "execute_max_model_turns"}
            ),
        )
        prompt = PromptSettings.from_dict(mapping["prompt"])
        sampling = SamplingSettings.from_dict(mapping["sampling"])
        return _construct(
            name,
            lambda: _load(
                cls,
                mapping,
                prompt=prompt,
                sampling=sampling,
                api_docs_version=mapping.get("api_docs_version"),
                api_docs_app=mapping.get("api_docs_app"),
                execute_max_model_turns=mapping.get("execute_max_model_turns"),
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
        return _construct(name, lambda: _load(cls, mapping))


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
        return _construct(
            name,
            lambda: _load(cls, mapping, model=model, agent=agent, task=task),
        )


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


class _MissingHashedLeaf:
    """Sentinel for a ``HASHED_FIELDS`` path an optional leaf omits.

    Compares equal only to itself, so "unset" is one comparable value
    rather than a lookup failure: a fault that sets a previously-unset
    optional leaf, or clears one, is detected as a change the same way a
    fault that edits an always-present leaf is.
    """

    def __repr__(self) -> str:
        return "<missing-hashed-leaf>"


MISSING_HASHED_LEAF = _MissingHashedLeaf()


def leaf_value(document: Mapping[str, object], path: str) -> object:
    """Read one dotted ``HASHED_FIELDS`` path out of a configuration's ``to_dict()``.

    Returns ``MISSING_HASHED_LEAF`` when any segment of ``path`` is absent,
    which is exactly what happens at an optional leaf's parent object
    (``model.lora``, or the omitted ``agent.api_docs_version``/
    ``api_docs_app`` pair) when that leaf is unset.
    """

    node: object = document
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return MISSING_HASHED_LEAF
        node = node[part]
    return node


def hashed_values(configuration: RunConfiguration) -> dict[str, object]:
    """Every ``HASHED_FIELDS`` leaf's value for one configuration.

    An unset optional leaf reads as ``MISSING_HASHED_LEAF`` rather than
    raising, so a caller that diffs two configurations' hashed fields never
    needs a special case for a leaf that is being introduced or cleared.
    """

    if not isinstance(configuration, RunConfiguration):
        raise ConfigError("hashed_values requires a run configuration")
    document = configuration.to_dict()
    return {path: leaf_value(document, path) for path in sorted(HASHED_FIELDS)}


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

    def to_dict(self) -> dict[str, object]:
        return _plain_dict(self, RunIdentity)

    @classmethod
    def from_dict(
        cls,
        payload: object,
        name: str = "run identity",
    ) -> RunIdentity:
        mapping = dict(_object(payload, name))
        _require_fields(
            mapping,
            cls,
            name,
            optional=frozenset({"protocol_hash"}),
        )
        if "protocol_hash" not in mapping:
            mapping["protocol_hash"] = None
        return _construct(name, lambda: _load(cls, mapping))


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

    def to_dict(self) -> dict[str, object]:
        return _plain_dict(self, EpisodeIdentity)

    @classmethod
    def from_dict(
        cls,
        payload: object,
        name: str = "episode identity",
    ) -> EpisodeIdentity:
        mapping = _object(payload, name)
        _require_fields(mapping, cls, name, optional=frozenset({"pair_id"}))
        return _construct(name, lambda: _load(cls, mapping))


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


@dataclass(frozen=True)
class StoppingRule:
    """A caller-supplied stopping rule.

    ``alpha``, ``horizon_episodes``, and ``name`` are required. ``threshold``
    is the decision threshold for rules that use one. ``None`` means the
    named rule has no threshold parameter. No level or threshold is filled in.
    """

    name: str
    alpha: float
    horizon_episodes: int
    threshold: float | None = None

    def __post_init__(self) -> None:
        _rule_name(self.name, "name")
        _open_probability(self.alpha, "alpha")
        _positive(self.horizon_episodes, "horizon_episodes")
        if self.threshold is not None:
            _finite_float(self.threshold, "threshold")

    def to_dict(self) -> dict[str, object]:
        return _plain_dict(self, StoppingRule)

    @classmethod
    def from_dict(
        cls,
        payload: object,
        name: str = "stopping rule",
    ) -> StoppingRule:
        mapping = _object(payload, name)
        _require_fields(mapping, cls, name, optional=frozenset({"threshold"}))
        return _construct(name, lambda: _load(cls, mapping))


@dataclass(frozen=True)
class StreamSettings:
    """Seeded task-stream inputs.

    ``task_mix_rule`` is a public rule name. Resolved task ids stay in the
    local task set. Constructing a ``test_normal`` stream does not run it.
    """

    split: str
    selection_rule: str
    selection_seed: int
    task_set_hash: str
    stream_seed: int
    arrival_rate_per_second: float
    concurrency: int
    with_replacement: bool
    task_mix_rule: str

    def __post_init__(self) -> None:
        _choice(self.split, SPLITS, "split")
        _text(self.selection_rule, "selection_rule")
        _integer(self.selection_seed, "selection_seed")
        _sha256(self.task_set_hash, "task_set_hash")
        _integer(self.stream_seed, "stream_seed")
        _positive_float(self.arrival_rate_per_second, "arrival_rate_per_second")
        _positive(self.concurrency, "concurrency")
        _flag(self.with_replacement, "with_replacement")
        _text(self.task_mix_rule, "task_mix_rule")

    def to_dict(self) -> dict[str, object]:
        return _plain_dict(self, StreamSettings)

    @classmethod
    def from_dict(
        cls,
        payload: object,
        name: str = "stream settings",
    ) -> StreamSettings:
        mapping = _object(payload, name)
        _require_fields(mapping, cls, name)
        return _construct(name, lambda: _load(cls, mapping))


@dataclass(frozen=True)
class GateSettings:
    """Explicit inputs of the plan-only offline gate.

    ``score_margin`` is the margin for the paired score-difference interval.
    ``kl_limit_nats`` is the KL bound. ``mmd_alpha`` is the permutation-test
    level. The gate implementation applies those comparisons. These fields
    do not carry demo constants.
    """

    confidence_level: float
    bootstrap_resamples: int
    score_margin: float
    kl_limit_nats: float
    mmd_bandwidth: float
    mmd_permutations: int
    mmd_alpha: float
    plan_format_version: str

    def __post_init__(self) -> None:
        _open_probability(self.confidence_level, "confidence_level")
        _positive(self.bootstrap_resamples, "bootstrap_resamples")
        _finite_float(self.score_margin, "score_margin")
        _nonnegative_float(self.kl_limit_nats, "kl_limit_nats")
        _positive_float(self.mmd_bandwidth, "mmd_bandwidth")
        _positive(self.mmd_permutations, "mmd_permutations")
        _open_probability(self.mmd_alpha, "mmd_alpha")
        _text(self.plan_format_version, "plan_format_version")

    def to_dict(self) -> dict[str, object]:
        return _plain_dict(self, GateSettings)

    @classmethod
    def from_dict(
        cls,
        payload: object,
        name: str = "gate settings",
    ) -> GateSettings:
        mapping = _object(payload, name)
        _require_fields(mapping, cls, name)
        return _construct(name, lambda: _load(cls, mapping))


METRIC_ORIENTATIONS = frozenset({"higher_is_better", "lower_is_better"})
"""How to read a rise in the canary's paired metric.

``higher_is_better`` is the only metric this repo's runtime currently
feeds the canary (evaluator task success), so it is the default. A future
paired metric where a rise is worse (latency, error rate) sets
``lower_is_better`` instead, so harm and benefit are never read
backwards.
"""

PROMOTION_POLICIES = frozenset({"horizon_reached_without_harm"})
"""The named automatic-promotion policies this controller implements.

``horizon_reached_without_harm`` is the only one: promote once the
configured horizon of paired episodes is reached without a harmful-
direction alarm. It is a stopping-time exposure rule, not a claim that
the candidate is statistically superior to the reference; that would be
a separate, stronger promotion rule this task does not add.
"""


@dataclass(frozen=True)
class CanarySettings:
    """Explicit inputs of paired canary execution.

    ``fraction`` is the share of tasks sent to the canary. ``harm_margin``
    is the drop, in the direction ``metric_orientation`` calls worse, that
    the canary treats as harm. ``outcome_delay_seconds`` is the additional
    delay before evaluator outcomes become visible. Zero is an explicit
    delay of none. ``metric_orientation`` states which direction of the
    paired metric is worse, so a lower-is-better metric is never read as
    if it were higher-is-better. ``promotion_policy`` names the automatic
    promotion rule in force; it is represented separately from the
    stopping rule's rollback evidence, and is not itself a rollback or
    harm signal.
    """

    fraction: float
    outcome_delay_seconds: float
    harm_margin: float
    stopping_rule: StoppingRule
    metric_orientation: str
    promotion_policy: str

    def __post_init__(self) -> None:
        _open_unit(self.fraction, "fraction")
        _nonnegative_float(self.outcome_delay_seconds, "outcome_delay_seconds")
        _positive_float(self.harm_margin, "harm_margin")
        _kind(self.stopping_rule, StoppingRule, "stopping_rule")
        _choice(self.metric_orientation, METRIC_ORIENTATIONS, "metric_orientation")
        _choice(self.promotion_policy, PROMOTION_POLICIES, "promotion_policy")

    def to_dict(self) -> dict[str, object]:
        return _plain_dict(self, CanarySettings)

    @classmethod
    def from_dict(
        cls,
        payload: object,
        name: str = "canary settings",
    ) -> CanarySettings:
        mapping = _object(payload, name)
        _require_fields(mapping, cls, name)
        stopping_rule = StoppingRule.from_dict(mapping["stopping_rule"])
        return _construct(
            name,
            lambda: _load(cls, mapping, stopping_rule=stopping_rule),
        )


def _signals(value: object) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ConfigError("signals must be a list of monitor signals")
    if not value:
        raise ConfigError("signals must name at least one monitor signal")
    chosen: list[str] = []
    for item in value:
        chosen.append(_choice(item, MONITOR_SIGNALS, "signals"))
    if len(set(chosen)) != len(chosen):
        raise ConfigError("signals contains a duplicate")
    return tuple(chosen)


def _stopping_rules(value: object, name: str) -> tuple[StoppingRule, ...]:
    if isinstance(value, (str, StoppingRule)) or not isinstance(
        value, (list, tuple)
    ):
        raise ConfigError(f"{name} must be a list of stopping rules")
    if not value:
        raise ConfigError(f"{name} must contain at least one rule")
    rules: list[StoppingRule] = []
    for item in value:
        if isinstance(item, StoppingRule):
            rules.append(item)
        else:
            rules.append(StoppingRule.from_dict(item, name=name))
    return tuple(rules)


@dataclass(frozen=True)
class MonitorSettings:
    """Explicit inputs of production monitors.

    ``signals`` selects the monitored series. ``task_success`` is the
    evaluator success flag. ``requirement_fraction`` is passed requirements
    over total requirements when that total is positive. ``tool_error_count``
    counts tool executions that returned an error. ``invalid_tool_call_count``
    counts actions that did not parse into a call. ``trajectory_length`` is
    the number of model and tool steps. ``reference_configuration_hash`` is
    the previous known-good configuration.
    """

    reference_configuration_hash: str
    outcome_delay_seconds: float
    signals: tuple[str, ...]
    stopping_rules: tuple[StoppingRule, ...]

    def __post_init__(self) -> None:
        _sha256(self.reference_configuration_hash, "reference_configuration_hash")
        _nonnegative_float(self.outcome_delay_seconds, "outcome_delay_seconds")
        if not isinstance(self.signals, tuple):
            raise ConfigError("signals must be a tuple")
        if not self.signals:
            raise ConfigError("signals must name at least one monitor signal")
        if len(set(self.signals)) != len(self.signals):
            raise ConfigError("signals contains a duplicate")
        for signal in self.signals:
            _choice(signal, MONITOR_SIGNALS, "signals")
        if (
            not isinstance(self.stopping_rules, tuple)
            or not self.stopping_rules
        ):
            raise ConfigError("stopping_rules must be a non-empty tuple")
        for rule in self.stopping_rules:
            _kind(rule, StoppingRule, "stopping_rules")

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_configuration_hash": self.reference_configuration_hash,
            "outcome_delay_seconds": self.outcome_delay_seconds,
            "signals": list(self.signals),
            "stopping_rules": [rule.to_dict() for rule in self.stopping_rules],
        }

    @classmethod
    def from_dict(
        cls,
        payload: object,
        name: str = "monitor settings",
    ) -> MonitorSettings:
        mapping = _object(payload, name)
        _require_fields(mapping, cls, name)
        signals = _signals(mapping["signals"])
        stopping_rules = _stopping_rules(
            mapping["stopping_rules"],
            name="stopping_rules",
        )
        return _construct(
            name,
            lambda: _load(
                cls,
                mapping,
                signals=signals,
                stopping_rules=stopping_rules,
            ),
        )


def _reference_counts(value: object, name: str) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, tuple) or not value:
        raise ConfigError(f"{name} must be a non-empty tuple of (name, count) pairs")
    names: list[str] = []
    pairs: list[tuple[str, int]] = []
    for item in value:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or item[0] == ""
        ):
            raise ConfigError(f"{name} entries must be (name, count) pairs")
        category, count = item
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ConfigError(f"{name} counts must be non-negative integers")
        names.append(category)
        pairs.append((category, count))
    if len(set(names)) != len(names):
        raise ConfigError(f"{name} contains a duplicate category name")
    if sum(count for _category, count in pairs) <= 0:
        raise ConfigError(f"{name} must have a positive total count")
    return tuple(pairs)


@dataclass(frozen=True)
class DistributionalMonitorSettings:
    """Explicit inputs for one windowed categorical drift monitor.

    ``signal`` is ``tool_selection`` or ``task_mix``: a distribution-valued
    series ``MonitorSettings.signals`` cannot name, since ``MONITOR_SIGNALS``
    is the closed scalar set. ``reference_counts`` is the frozen baseline
    category distribution (unnormalized counts; only their proportions
    enter the chi-square test). ``window_episodes`` is how many observations
    accumulate before one look. ``correction`` selects the repeated-look
    adjustment the same way ``lifecycle.detectors.build_hourly_window_detector``
    does for scalar fixed-window tests.
    """

    signal: str
    reference_counts: tuple[tuple[str, int], ...]
    window_episodes: int
    alpha: float
    correction: str

    def __post_init__(self) -> None:
        _choice(self.signal, DISTRIBUTIONAL_SIGNALS, "signal")
        _reference_counts(self.reference_counts, "reference_counts")
        _positive(self.window_episodes, "window_episodes")
        _open_probability(self.alpha, "alpha")
        _choice(self.correction, DISTRIBUTIONAL_CORRECTIONS, "correction")

    def to_dict(self) -> dict[str, object]:
        return {
            "signal": self.signal,
            "reference_counts": [list(item) for item in self.reference_counts],
            "window_episodes": self.window_episodes,
            "alpha": self.alpha,
            "correction": self.correction,
        }

    @classmethod
    def from_dict(
        cls,
        payload: object,
        name: str = "distributional monitor settings",
    ) -> DistributionalMonitorSettings:
        mapping = _object(payload, name)
        _require_fields(mapping, cls, name)
        raw_counts = mapping["reference_counts"]
        if not isinstance(raw_counts, (list, tuple)):
            raise ConfigError(f"{name} reference_counts must be a list of pairs")
        counts = tuple(
            (str(item[0]), int(item[1]))
            for item in raw_counts
            if isinstance(item, (list, tuple)) and len(item) == 2
        )
        if len(counts) != len(raw_counts):
            raise ConfigError(f"{name} reference_counts entries must be pairs")
        return _construct(
            name,
            lambda: _load(cls, mapping, reference_counts=counts),
        )
