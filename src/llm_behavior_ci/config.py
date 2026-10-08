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
model.serving.cpu_offload_gb
model.serving.tensor_parallel_size
model.serving.max_logprobs
model.serving.batch_invariant
model.serving.sampler_backend
model.lora.repository
model.lora.revision
model.provider
model.model_id
model.api_version
model.thinking_mode
model.effort
model.api_base
model.thinking_type
model.clear_thinking
model.reasoning_effort
agent.smolagents_version
agent.action_interface
agent.prompt.prompt_version
agent.prompt.plan_format_version
agent.prompt.thinking_enabled
agent.step_limit
agent.execute_max_model_turns
agent.tool_access_profile
agent.sampling.temperature
agent.sampling.top_p
agent.sampling.top_k
agent.sampling.min_p
agent.sampling.seed
agent.sampling.do_sample
agent.sampling.max_tokens
agent.sampling.execute_max_tokens
agent.sampling.plan_max_tokens
agent.api_docs_version
agent.api_docs_app
agent.workflow.policy
agent.workflow.repeat_action_limit
agent.workflow.no_progress_turns
agent.workflow.completion_gate
agent.workflow.max_plan_steps
task.appworld_version
task.split
task.selection_rule
task.selection_seed
task.task_count
task.task_set_hash
task.appworld_setup_profile
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
none, fp8, nvfp4, and awq; model dtypes are bfloat16, float16, and float32;
KV-cache dtypes are bfloat16, float16, and fp8. Workflow policies are
plan_progress_v1 and plan_progress_v2. Model revisions,
tokenizer revisions, and git_commit are 40-character lowercase git ids.
task_set_hash and a present protocol_hash are 64-character lowercase
SHA-256 digests. top_k is -1 or a positive integer. The sampling seed
is required. KL fidelity modes are full and top_k; runtime/scoring.py
is where full is proven rather than merely claimed.

model.lora.repository, model.lora.revision, agent.api_docs_version,
agent.api_docs_app, agent.execute_max_model_turns, agent.tool_access_profile,
agent.sampling.execute_max_tokens, agent.sampling.plan_max_tokens,
agent.workflow.policy, agent.workflow.repeat_action_limit,
agent.workflow.no_progress_turns, agent.workflow.completion_gate,
agent.workflow.max_plan_steps,
model.serving.cpu_offload_gb, and
task.appworld_setup_profile are
optional hashed leaves: ``ModelConfiguration.lora``,
``AgentConfiguration.api_docs_version``/``api_docs_app``,
``AgentConfiguration.execute_max_model_turns``,
``AgentConfiguration.tool_access_profile``, the two mode-specific
``SamplingSettings`` token caps, ``AgentConfiguration.workflow``,
``TaskConfiguration.appworld_setup_profile``, and
``VLLMBehaviorSettings.cpu_offload_gb``
default to unset, and an unset leaf is omitted from canonical JSON entirely rather
than serialized as null, so a configuration that never names a LoRA
adapter, a corrupted API-documentation source, a tool-access profile,
a workflow controller, an AppWorld setup profile, or CPU weight offload
hashes identically to one built before these fields existed. ``cpu_offload_gb``
defaults to ``0`` and is omitted at that default, which is the no-offload
launch. Setting, clearing, or changing any of them still changes the digest,
because canonical JSON then differs.
``leaf_value``/``hashed_values`` read these leaves through
``MISSING_HASHED_LEAF`` rather than raising, so a caller comparing two
configurations' hashed fields sees "unset" as one comparable value
instead of a lookup error.

``model.provider``, ``model.model_id``, ``model.api_version``,
``model.thinking_mode``, and ``model.effort`` are the Anthropic
behavioral identity. A vLLM configuration omits them, so its canonical
JSON is unchanged. An Anthropic configuration omits the vLLM model,
tokenizer, quantization, serving, and LoRA leaves. The same omission
applies to ``agent.sampling.temperature``, ``agent.sampling.top_p``,
``agent.sampling.top_k``, ``agent.sampling.min_p``, and
``agent.sampling.seed`` when those controls are unset. Anthropic
requests do not send them, and the hash does not record a value the
client ignores. A vLLM configuration still requires the controls and
serializes them as before. ``recorded_execution_seed`` uses the vLLM
sampling seed when it is set, and the run seed otherwise. The run seed
orders the experiment; it does not make Claude generation deterministic.

``model.api_base``, ``model.thinking_type``, ``model.clear_thinking``, and
``model.reasoning_effort`` are the OpenAI-compatible hosted identity,
together with the shared ``model.provider`` and ``model.model_id``. The
API base is hashed because it names the service that runs the model.
The only provider is ``zai``. ``agent.sampling.do_sample`` belongs to
OpenAI-compatible sampling, which also sets ``agent.sampling.temperature``
and ``agent.sampling.top_p`` when ``do_sample`` is true and omits them
when it is false, because the provider then ignores them. It never sets
top-k, min-p, or seed. vLLM and Anthropic configurations omit all of
these leaves, so their canonical JSON is unchanged. An unknown provider
is a ``ConfigError``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.parse
import uuid
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable, Mapping, TypeVar, get_args, get_type_hints

_T = TypeVar("_T")

SPLITS = frozenset({"train", "dev", "test_normal", "test_challenge"})
ACTION_INTERFACES = frozenset({"code", "tool_calling"})
QUANTIZATION_METHODS = frozenset({"none", "fp8", "nvfp4", "awq"})
MODEL_DTYPES = frozenset({"bfloat16", "float16", "float32"})
KV_CACHE_DTYPES = frozenset({"bfloat16", "float16", "fp8"})
SAMPLER_BACKENDS = frozenset({"flashinfer", "native"})
KL_FIDELITY_MODES = frozenset({"full", "top_k"})
WORKFLOW_POLICIES = frozenset({"plan_progress_v1", "plan_progress_v2"})
ANTHROPIC_PROVIDERS = frozenset({"anthropic"})
ANTHROPIC_THINKING_MODES = frozenset({"adaptive", "between_tools"})
ANTHROPIC_EFFORT_LEVELS = frozenset({"low", "medium", "high", "xhigh", "max"})
ANTHROPIC_BETWEEN_TOOLS_EFFORTS = frozenset({"low", "medium", "high"})
OPENAI_COMPATIBLE_PROVIDERS = frozenset({"zai"})
OPENAI_COMPATIBLE_THINKING_TYPES = frozenset({"enabled", "disabled"})
OPENAI_COMPATIBLE_REASONING_EFFORTS = frozenset({"low", "medium", "high"})
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
        "model.serving.cpu_offload_gb",
        "model.serving.tensor_parallel_size",
        "model.serving.max_logprobs",
        "model.serving.batch_invariant",
        "model.serving.sampler_backend",
        "model.lora.repository",
        "model.lora.revision",
        "model.provider",
        "model.model_id",
        "model.api_version",
        "model.thinking_mode",
        "model.effort",
        "model.api_base",
        "model.thinking_type",
        "model.clear_thinking",
        "model.reasoning_effort",
        "agent.smolagents_version",
        "agent.action_interface",
        "agent.prompt.prompt_version",
        "agent.prompt.plan_format_version",
        "agent.prompt.thinking_enabled",
        "agent.step_limit",
        "agent.execute_max_model_turns",
        "agent.tool_access_profile",
        "agent.sampling.temperature",
        "agent.sampling.top_p",
        "agent.sampling.top_k",
        "agent.sampling.min_p",
        "agent.sampling.seed",
        "agent.sampling.do_sample",
        "agent.sampling.max_tokens",
        "agent.sampling.execute_max_tokens",
        "agent.sampling.plan_max_tokens",
        "agent.api_docs_version",
        "agent.api_docs_app",
        "agent.workflow.policy",
        "agent.workflow.repeat_action_limit",
        "agent.workflow.no_progress_turns",
        "agent.workflow.completion_gate",
        "agent.workflow.max_plan_steps",
        "task.appworld_version",
        "task.split",
        "task.selection_rule",
        "task.selection_seed",
        "task.task_count",
        "task.task_set_hash",
        "task.appworld_setup_profile",
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
_ANTHROPIC_MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_API_VERSION = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
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


def _anthropic_model_id(value: object) -> str:
    if not isinstance(value, str) or _ANTHROPIC_MODEL_ID.fullmatch(value) is None:
        raise ConfigError("model_id must be an Anthropic model id")
    return value


def _hosted_model_id(value: object) -> str:
    if not isinstance(value, str) or _ANTHROPIC_MODEL_ID.fullmatch(value) is None:
        raise ConfigError("model_id must be a lowercase hosted model id")
    return value


def _api_base(value: object) -> str:
    if (
        not isinstance(value, str)
        or value == ""
        or len(value) > _MAX_TEXT
        or any(character.isspace() for character in value)
        or value.endswith("/")
    ):
        raise ConfigError("api_base must be an https URL without a trailing slash")
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query != ""
        or parsed.fragment != ""
    ):
        raise ConfigError(
            "api_base must be an https URL with no credentials, query, or fragment"
        )
    return value


def _api_version(value: object) -> str:
    if not isinstance(value, str) or _API_VERSION.fullmatch(value) is None:
        raise ConfigError("api_version must be a YYYY-MM-DD Anthropic API version")
    return value


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
        elif is_dataclass(child) and not isinstance(child, type):
            to_dict = getattr(child, "to_dict", None)
            payload[field.name] = (
                to_dict()
                if callable(to_dict)
                else _plain_dict(child, type(child))
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
    cpu_offload_gb: float = 0.0

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
        _nonnegative_float(self.cpu_offload_gb, "cpu_offload_gb")

    def to_dict(self) -> dict[str, object]:
        document = _plain_dict(self, VLLMBehaviorSettings)
        if document["cpu_offload_gb"] == 0.0:
            del document["cpu_offload_gb"]
        return document

    @classmethod
    def from_dict(
        cls,
        payload: object,
        name: str = "serving",
    ) -> VLLMBehaviorSettings:
        mapping = _object(payload, name)
        _require_fields(mapping, cls, name, optional=frozenset({"cpu_offload_gb"}))
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
class AnthropicModelConfiguration:
    """Behavioral identity of one Anthropic Messages API model.

    The fields are the request values that can change model behavior.
    No Hugging Face repository, tokenizer revision, quantization, or vLLM
    serving setting is invented. The API key is not a field.
    """

    provider: str
    model_id: str
    api_version: str
    thinking_mode: str
    effort: str

    def __post_init__(self) -> None:
        _choice(self.provider, ANTHROPIC_PROVIDERS, "provider")
        _anthropic_model_id(self.model_id)
        _api_version(self.api_version)
        _choice(self.thinking_mode, ANTHROPIC_THINKING_MODES, "thinking_mode")
        _choice(self.effort, ANTHROPIC_EFFORT_LEVELS, "effort")
        if (
            self.thinking_mode == "between_tools"
            and self.effort not in ANTHROPIC_BETWEEN_TOOLS_EFFORTS
        ):
            raise ConfigError(
                "between_tools thinking accepts only low, medium, or high effort"
            )

    def to_dict(self) -> dict[str, object]:
        return _plain_dict(self, AnthropicModelConfiguration)

    @classmethod
    def from_dict(
        cls,
        payload: object,
        name: str = "model configuration",
    ) -> AnthropicModelConfiguration:
        mapping = _object(payload, name)
        _require_fields(mapping, cls, name)
        return _construct(name, lambda: _load(cls, mapping))


@dataclass(frozen=True)
class OpenAICompatibleModelConfiguration:
    """Behavioral identity of one hosted OpenAI-compatible chat model.

    ``api_base`` is the service root; requests go to
    ``{api_base}/chat/completions``. ``thinking_type`` and
    ``clear_thinking`` are sent as the ``thinking`` object and
    ``reasoning_effort`` as its own field. The API key is not a field.
    """

    provider: str
    model_id: str
    api_base: str
    thinking_type: str
    clear_thinking: bool
    reasoning_effort: str

    def __post_init__(self) -> None:
        _choice(self.provider, OPENAI_COMPATIBLE_PROVIDERS, "provider")
        _hosted_model_id(self.model_id)
        _api_base(self.api_base)
        _choice(self.thinking_type, OPENAI_COMPATIBLE_THINKING_TYPES, "thinking_type")
        _flag(self.clear_thinking, "clear_thinking")
        _choice(
            self.reasoning_effort,
            OPENAI_COMPATIBLE_REASONING_EFFORTS,
            "reasoning_effort",
        )

    def to_dict(self) -> dict[str, object]:
        return _plain_dict(self, OpenAICompatibleModelConfiguration)

    @classmethod
    def from_dict(
        cls,
        payload: object,
        name: str = "model configuration",
    ) -> OpenAICompatibleModelConfiguration:
        mapping = _object(payload, name)
        _require_fields(mapping, cls, name)
        return _construct(name, lambda: _load(cls, mapping))


HostedModelConfiguration = (
    AnthropicModelConfiguration | OpenAICompatibleModelConfiguration
)
AnyModelConfiguration = ModelConfiguration | HostedModelConfiguration


def load_model_configuration(
    payload: object,
    name: str = "model configuration",
) -> AnyModelConfiguration:
    """Load a model configuration by explicit provider.

    No ``provider`` is a vLLM ``ModelConfiguration``; existing vLLM JSON
    does not name one and this loader does not add one. ``anthropic`` is
    an ``AnthropicModelConfiguration`` and ``zai`` is an
    ``OpenAICompatibleModelConfiguration``. Any other provider is a
    ``ConfigError``.
    """

    mapping = _object(payload, name)
    if "provider" not in mapping:
        return ModelConfiguration.from_dict(payload, name)
    provider = mapping["provider"]
    if isinstance(provider, str) and provider in ANTHROPIC_PROVIDERS:
        return AnthropicModelConfiguration.from_dict(payload, name)
    if isinstance(provider, str) and provider in OPENAI_COMPATIBLE_PROVIDERS:
        return OpenAICompatibleModelConfiguration.from_dict(payload, name)
    choices = ", ".join(sorted(ANTHROPIC_PROVIDERS | OPENAI_COMPATIBLE_PROVIDERS))
    raise ConfigError(f"{name}: provider must be one of: {choices}")


def hosted_provider(model: object) -> str | None:
    """The hosted provider a model configuration calls, or None for local vLLM.

    A local vLLM model needs an endpoint URL. A hosted model names its
    service in configuration and must not be given one.
    """

    if isinstance(model, ModelConfiguration):
        return None
    if isinstance(model, (AnthropicModelConfiguration, OpenAICompatibleModelConfiguration)):
        return model.provider
    raise ConfigError("model must be a vLLM, Anthropic, or OpenAI-compatible configuration")


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


_VLLM_SAMPLING_CONTROLS = frozenset(
    {"temperature", "top_p", "top_k", "min_p", "seed"}
)


@dataclass(frozen=True)
class HostedSamplingSettings:
    """Generation length for an API provider that rejects vLLM sampling.

    Temperature, top-p, top-k, min-p, and seed are absent rather than
    filled with values the client would ignore. ``max_tokens`` is the
    generation cap unless the mode override is set. Anthropic
    configurations use this class, also exported as
    ``AnthropicSamplingSettings``.
    """

    max_tokens: int
    execute_max_tokens: int | None = None
    plan_max_tokens: int | None = None

    def __post_init__(self) -> None:
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
        document = _plain_dict(self, HostedSamplingSettings)
        for key in ("execute_max_tokens", "plan_max_tokens"):
            if document[key] is None:
                del document[key]
        return document

    @classmethod
    def from_dict(
        cls,
        payload: object,
        name: str = "sampling",
    ) -> HostedSamplingSettings:
        mapping = _object(payload, name)
        _require_fields(
            mapping,
            cls,
            name,
            optional=frozenset({"execute_max_tokens", "plan_max_tokens"}),
        )
        return _construct(name, lambda: _load(cls, mapping))


AnthropicSamplingSettings = HostedSamplingSettings


@dataclass(frozen=True)
class OpenAICompatibleSamplingSettings:
    """Request sampling an OpenAI-compatible hosted provider receives.

    Every field is sent. ``do_sample`` true requires ``temperature`` and
    ``top_p``; ``do_sample`` false requires both unset, because the
    provider then decodes greedily and would ignore them. There is no
    top-k, min-p, or seed field. ``max_tokens`` is the generation cap
    unless the mode override is set.
    """

    do_sample: bool
    max_tokens: int
    temperature: float | None = None
    top_p: float | None = None
    execute_max_tokens: int | None = None
    plan_max_tokens: int | None = None

    def __post_init__(self) -> None:
        _flag(self.do_sample, "do_sample")
        if self.do_sample:
            if self.temperature is None or self.top_p is None:
                raise ConfigError("do_sample true requires temperature and top_p")
            _temperature(self.temperature)
            _open_unit(self.top_p, "top_p")
        elif self.temperature is not None or self.top_p is not None:
            raise ConfigError("do_sample false requires temperature and top_p unset")
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
        document = _plain_dict(self, OpenAICompatibleSamplingSettings)
        for key in ("temperature", "top_p", "execute_max_tokens", "plan_max_tokens"):
            if document[key] is None:
                del document[key]
        return document

    @classmethod
    def from_dict(
        cls,
        payload: object,
        name: str = "sampling",
    ) -> OpenAICompatibleSamplingSettings:
        mapping = _object(payload, name)
        _require_fields(
            mapping,
            cls,
            name,
            optional=frozenset(
                {"temperature", "top_p", "execute_max_tokens", "plan_max_tokens"}
            ),
        )
        return _construct(name, lambda: _load(cls, mapping))


AnySamplingSettings = (
    SamplingSettings | HostedSamplingSettings | OpenAICompatibleSamplingSettings
)


def load_sampling_settings(
    payload: object,
    name: str = "sampling",
) -> AnySamplingSettings:
    """Load sampling settings by schema.

    ``do_sample`` present is OpenAI-compatible sampling. Otherwise all five
    vLLM controls are vLLM sampling and none of them is hosted length-only
    sampling. A partial or mixed schema is a ``ConfigError``.
    """

    mapping = _object(payload, name)
    present = _VLLM_SAMPLING_CONTROLS & set(mapping)
    if "do_sample" in mapping:
        if present - {"temperature", "top_p"}:
            raise ConfigError(
                f"{name} with do_sample must not set top_k, min_p, or seed"
            )
        return OpenAICompatibleSamplingSettings.from_dict(payload, name)
    if present == _VLLM_SAMPLING_CONTROLS:
        return SamplingSettings.from_dict(payload, name)
    if not present:
        return HostedSamplingSettings.from_dict(payload, name)
    raise ConfigError(
        f"{name} must set temperature, top_p, top_k, min_p, and seed together "
        "or omit them"
    )


def _validate_model_sampling(model: object, sampling: object) -> None:
    if isinstance(model, ModelConfiguration):
        if not isinstance(sampling, SamplingSettings):
            raise ConfigError("vLLM configurations require vLLM sampling controls")
        return
    if isinstance(model, AnthropicModelConfiguration):
        if not isinstance(sampling, HostedSamplingSettings):
            raise ConfigError(
                "Anthropic configurations must leave unsupported sampling controls unset"
            )
        return
    if isinstance(model, OpenAICompatibleModelConfiguration):
        if not isinstance(sampling, OpenAICompatibleSamplingSettings):
            raise ConfigError(
                "OpenAI-compatible configurations require OpenAI-compatible sampling"
            )
        return
    raise ConfigError("model must be a vLLM, Anthropic, or OpenAI-compatible configuration")


@dataclass(frozen=True)
class WorkflowSettings:
    policy: str
    repeat_action_limit: int
    no_progress_turns: int
    completion_gate: bool
    max_plan_steps: int

    def __post_init__(self) -> None:
        _choice(self.policy, WORKFLOW_POLICIES, "workflow policy")
        _positive(self.repeat_action_limit, "repeat_action_limit")
        _positive(self.no_progress_turns, "no_progress_turns")
        _flag(self.completion_gate, "completion_gate")
        _positive(self.max_plan_steps, "max_plan_steps")
        if self.repeat_action_limit < 2:
            raise ConfigError("repeat_action_limit must be at least 2")
        if not 2 <= self.max_plan_steps <= 8:
            raise ConfigError("max_plan_steps must be between 2 and 8")

    def to_dict(self) -> dict[str, object]:
        return _plain_dict(self, WorkflowSettings)

    @classmethod
    def from_dict(
        cls,
        payload: object,
        name: str = "workflow",
    ) -> WorkflowSettings:
        mapping = _object(payload, name)
        _require_fields(mapping, cls, name)
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

    ``tool_access_profile`` names an optional runtime allowlist. Unset is
    omitted from ``to_dict``. Setting or changing it changes the run hash.

    ``workflow`` names an optional execute-time controller. Unset is omitted
    from ``to_dict``. Setting or changing it changes the run hash. Plan mode
    does not wrap the agent when this field is set.
    """

    smolagents_version: str
    action_interface: str
    prompt: PromptSettings
    step_limit: int
    sampling: AnySamplingSettings
    api_docs_version: str | None = None
    api_docs_app: str | None = None
    tool_access_profile: str | None = None
    execute_max_model_turns: int | None = None
    workflow: WorkflowSettings | None = None

    def __post_init__(self) -> None:
        _text(self.smolagents_version, "smolagents_version")
        _choice(self.action_interface, ACTION_INTERFACES, "action_interface")
        _kind(self.prompt, PromptSettings, "prompt")
        _positive(self.step_limit, "step_limit")
        if not isinstance(
            self.sampling,
            (SamplingSettings, HostedSamplingSettings, OpenAICompatibleSamplingSettings),
        ):
            raise ConfigError("sampling must be vLLM, hosted, or OpenAI-compatible settings")
        if (self.api_docs_version is None) != (self.api_docs_app is None):
            raise ConfigError(
                "api_docs_version and api_docs_app must be set together"
            )
        if self.api_docs_version is not None:
            _text(self.api_docs_version, "api_docs_version")
            _text(self.api_docs_app, "api_docs_app")
        if self.tool_access_profile is not None:
            _text(self.tool_access_profile, "tool_access_profile")
        if self.execute_max_model_turns is not None:
            _positive(self.execute_max_model_turns, "execute_max_model_turns")
        if self.workflow is not None:
            _kind(self.workflow, WorkflowSettings, "workflow")

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
        if self.tool_access_profile is not None:
            document["tool_access_profile"] = self.tool_access_profile
        if self.api_docs_version is not None:
            document["api_docs_version"] = self.api_docs_version
            document["api_docs_app"] = self.api_docs_app
        if self.execute_max_model_turns is not None:
            document["execute_max_model_turns"] = self.execute_max_model_turns
        if self.workflow is not None:
            document["workflow"] = self.workflow.to_dict()
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
                {
                    "api_docs_version",
                    "api_docs_app",
                    "execute_max_model_turns",
                    "tool_access_profile",
                    "workflow",
                }
            ),
        )
        prompt = PromptSettings.from_dict(mapping["prompt"])
        sampling = load_sampling_settings(mapping["sampling"])
        workflow = (
            None
            if "workflow" not in mapping
            else WorkflowSettings.from_dict(mapping["workflow"])
        )
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
                tool_access_profile=mapping.get("tool_access_profile"),
                workflow=workflow,
            ),
        )


@dataclass(frozen=True)
class TaskConfiguration:
    """One task-set configuration.

    ``appworld_setup_profile`` names optional runtime setup applied to the
    world before the agent starts. Unset is omitted from ``to_dict``.
    Reference and candidate configurations share this field because it is
    the initialized environment, not an agent policy. Setting or changing
    it changes the run hash.
    """

    appworld_version: str
    split: str
    selection_rule: str
    selection_seed: int
    task_count: int
    task_set_hash: str
    appworld_setup_profile: str | None = None

    def __post_init__(self) -> None:
        _text(self.appworld_version, "appworld_version")
        _choice(self.split, SPLITS, "split")
        _text(self.selection_rule, "selection_rule")
        _integer(self.selection_seed, "selection_seed")
        _positive(self.task_count, "task_count")
        _sha256(self.task_set_hash, "task_set_hash")
        if self.appworld_setup_profile is not None:
            _text(self.appworld_setup_profile, "appworld_setup_profile")

    def to_dict(self) -> dict[str, object]:
        document = _plain_dict(self, TaskConfiguration)
        if document.get("appworld_setup_profile") is None:
            del document["appworld_setup_profile"]
        return document

    @classmethod
    def from_dict(
        cls,
        payload: object,
        name: str = "task configuration",
    ) -> TaskConfiguration:
        mapping = _object(payload, name)
        _require_fields(
            mapping,
            cls,
            name,
            optional=frozenset({"appworld_setup_profile"}),
        )
        return _construct(name, lambda: _load(cls, mapping))


@dataclass(frozen=True)
class RunConfiguration:
    model: AnyModelConfiguration
    agent: AgentConfiguration
    task: TaskConfiguration
    run_seed: int
    git_commit: str
    protocol_hash: str | None = None

    def __post_init__(self) -> None:
        _validate_model_sampling(self.model, self.agent.sampling)
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
        model = load_model_configuration(mapping["model"])
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


def recorded_execution_seed(configuration: RunConfiguration) -> int:
    """The integer an episode records as its execution seed.

    A vLLM sampling seed is sent to the server and is the execution seed.
    Anthropic and OpenAI-compatible configurations do not send a
    generation seed. The recorded value is then ``run_seed``, which
    orders the experiment and the task stream. It is not a claim that
    hosted generation is deterministic.
    """

    if not isinstance(configuration, RunConfiguration):
        raise ConfigError("execution seed requires a run configuration")
    sampling = configuration.agent.sampling
    if isinstance(sampling, SamplingSettings):
        return sampling.seed
    return configuration.run_seed


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

    ``slack`` is the CUSUM reference allowance ``k`` in the signal's own
    units, subtracted from every deviation before it accumulates. Only a
    ``cusum`` rule takes it. ``None`` keeps the earlier recorded behavior,
    ``k = 0``, and is omitted from ``to_dict`` so older recorded settings
    are unchanged; a set value is recorded. Positive slack lengthens the
    healthy run length; it is not an α guarantee, and CUSUM stays a
    record-only method in validation.
    """

    name: str
    alpha: float
    horizon_episodes: int
    threshold: float | None = None
    slack: float | None = None

    def __post_init__(self) -> None:
        _rule_name(self.name, "name")
        _open_probability(self.alpha, "alpha")
        _positive(self.horizon_episodes, "horizon_episodes")
        if self.threshold is not None:
            _finite_float(self.threshold, "threshold")
        if self.slack is not None:
            if self.name != "cusum":
                raise ConfigError("slack applies only to a cusum stopping rule")
            _nonnegative_float(self.slack, "slack")

    def to_dict(self) -> dict[str, object]:
        payload = _plain_dict(self, StoppingRule)
        if self.slack is None:
            payload.pop("slack")
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: object,
        name: str = "stopping rule",
    ) -> StoppingRule:
        mapping = _object(payload, name)
        _require_fields(mapping, cls, name, optional=frozenset({"threshold", "slack"}))
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
    does for scalar fixed-window tests. ``slice_reference_counts``, when
    non-empty, gives each slice its own frozen distribution and requires
    ``reference_source``; a slice without an entry is then refused instead
    of being compared with the aggregate distribution.
    """

    signal: str
    reference_counts: tuple[tuple[str, int], ...]
    window_episodes: int
    alpha: float
    correction: str
    slice_reference_counts: tuple[tuple[str, tuple[tuple[str, int], ...]], ...] = ()
    reference_source: str | None = None

    def __post_init__(self) -> None:
        _choice(self.signal, DISTRIBUTIONAL_SIGNALS, "signal")
        _reference_counts(self.reference_counts, "reference_counts")
        _positive(self.window_episodes, "window_episodes")
        _open_probability(self.alpha, "alpha")
        _choice(self.correction, DISTRIBUTIONAL_CORRECTIONS, "correction")
        names = [name for name, _counts in self.slice_reference_counts]
        if len(set(names)) != len(names) or any(
            not isinstance(name, str) or name == "" for name in names
        ):
            raise ConfigError("slice_reference_counts must name each slice once")
        for name, counts in self.slice_reference_counts:
            _reference_counts(counts, f"slice_reference_counts {name}")
        if self.slice_reference_counts and (
            not isinstance(self.reference_source, str) or self.reference_source == ""
        ):
            raise ConfigError("slice_reference_counts require a reference_source")

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "signal": self.signal,
            "reference_counts": [list(item) for item in self.reference_counts],
            "window_episodes": self.window_episodes,
            "alpha": self.alpha,
            "correction": self.correction,
        }
        if self.slice_reference_counts:
            payload["slice_reference_counts"] = [
                [name, [list(item) for item in counts]]
                for name, counts in self.slice_reference_counts
            ]
        if self.reference_source is not None:
            payload["reference_source"] = self.reference_source
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: object,
        name: str = "distributional monitor settings",
    ) -> DistributionalMonitorSettings:
        mapping = _object(payload, name)
        _require_fields(
            mapping,
            cls,
            name,
            optional=frozenset({"slice_reference_counts", "reference_source"}),
        )

        def count_pairs(raw: object, label: str) -> tuple[tuple[str, int], ...]:
            if not isinstance(raw, (list, tuple)):
                raise ConfigError(f"{name} {label} must be a list of pairs")
            pairs = tuple(
                (str(item[0]), int(item[1]))
                for item in raw
                if isinstance(item, (list, tuple)) and len(item) == 2
            )
            if len(pairs) != len(raw):
                raise ConfigError(f"{name} {label} entries must be pairs")
            return pairs

        counts = count_pairs(mapping["reference_counts"], "reference_counts")
        raw_slices = mapping.get("slice_reference_counts", [])
        if not isinstance(raw_slices, (list, tuple)):
            raise ConfigError(f"{name} slice_reference_counts must be a list")
        slices: list[tuple[str, tuple[tuple[str, int], ...]]] = []
        for item in raw_slices:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ConfigError(
                    f"{name} slice_reference_counts entries must be [slice, counts]"
                )
            slices.append((str(item[0]), count_pairs(item[1], "slice_reference_counts")))
        return _construct(
            name,
            lambda: _load(
                cls,
                mapping,
                reference_counts=counts,
                slice_reference_counts=tuple(slices),
            ),
        )
