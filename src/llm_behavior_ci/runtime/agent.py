from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Mapping, Protocol

from llm_behavior_ci.config import (
    ACTION_INTERFACES,
    AnthropicModelConfiguration,
    AnthropicSamplingSettings,
    OpenAICompatibleModelConfiguration,
    OpenAICompatibleSamplingSettings,
    RunConfiguration,
)
from llm_behavior_ci.records import TokenLogprob
from llm_behavior_ci.runtime.actions import ActionRejected, parse_model_output
from llm_behavior_ci.runtime.clock import monotonic, wall_now
from llm_behavior_ci.runtime.api_docs import (
    ApiDocsCorruptionError,
    resolve_api_documentation,
)
from llm_behavior_ci.runtime.appworld import TaskContext, render_api_documentation
from llm_behavior_ci.runtime.prompts import (
    PROMPT_RUNTIME_AUTH_V2,
    UnknownPromptVersion,
    render_system_text,
)

try:
    from smolagents.models import ChatMessage as _SmolChatMessage
    from smolagents.models import Model as _SmolModel
except ImportError:
    _SmolChatMessage = None
    _SmolModel = object

try:
    from smolagents.tools import Tool as _SmolTool
except ImportError:
    _SmolTool = object

try:
    from smolagents.local_python_executor import PythonExecutor as _SmolPythonExecutor
except ImportError:
    _SmolPythonExecutor = object

_TOKEN_ID_PREFIX = "token_id:"
ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_RETRYABLE = frozenset({429, 500, 502, 503, 504, 529})
_ANTHROPIC_ATTEMPTS = 5
OPENAI_COMPATIBLE_KEY_VARIABLES = {"zai": "ZAI_API_KEY"}
_OPENAI_COMPATIBLE_LABELS = {"zai": "Z.AI"}
_OPENAI_COMPATIBLE_RETRYABLE = frozenset({429, 500, 502, 503, 504})
_OPENAI_COMPATIBLE_ATTEMPTS = 5
_OPENAI_COMPATIBLE_FINISHES = frozenset({"stop", "length"})
_OPENAI_COMPATIBLE_PROVIDER_FAILURES = frozenset({"network_error", "sensitive"})


class UnsupportedCapability(RuntimeError):
    """The provider cannot perform this AgentLoop operation."""


def _http_error_body(error: urllib.error.HTTPError) -> str:
    try:
        raw = error.read()
    except Exception as read_error:
        return f"<unreadable body: {type(read_error).__name__}>"
    if raw is None:
        return "<empty body>"
    if isinstance(raw, str):
        text = raw
    else:
        try:
            text = bytes(raw).decode("utf-8")
        except UnicodeDecodeError:
            text = bytes(raw).decode("utf-8", errors="replace")
    if text.strip() == "":
        return "<empty body>"
    return text


def _context_length_exceeded(body: str) -> bool:
    text = body.lower()
    if "context_length_exceeded" in text:
        return True
    if "longer than the maximum model length" in text:
        return True
    if "maximum context length" in text:
        return True
    if "context length" in text and (
        "exceed" in text or "maximum" in text or "too long" in text
    ):
        return True
    return "max_model_len" in text and (
        "longer than" in text or "exceed" in text
    )


def _http_runtime_error(
    error: urllib.error.HTTPError,
    endpoint: str,
) -> RuntimeError:
    from llm_behavior_ci.runtime.episode import RuntimeUnavailable

    body = _http_error_body(error)
    reason = "context_length_exceeded" if _context_length_exceeded(body) else None
    message = (
        f"vLLM request failed with HTTP {error.code} {error.reason}:\n"
        f"{body}\n"
        f"endpoint: {endpoint}"
    )
    return RuntimeUnavailable(message, reason=reason)

_UNCHECKED_IDENTITY_FIELDS = (
    "weights_digest",
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
    "model.serving.batch_invariant",
    "model.serving.sampler_backend",
    "model.quantization.method",
    "model.vllm_version",
    "model.lora.repository",
    "model.lora.revision",
)


@dataclass(frozen=True)
class AgentTurn:
    prompt_text: str
    output_text: str
    top_k_logprobs: tuple[tuple[TokenLogprob, ...], ...]
    generated_token_count: int
    latency_seconds: float
    started_at: datetime
    action: str | None
    app_name: str | None
    api_name: str | None
    rejection: str | None = None
    feedback: str | None = None
    consumes_execute_turn: bool = True


class AgentLoop(Protocol):
    def begin(self, context: TaskContext, config: RunConfiguration) -> None: ...

    def next_turn(self, *, tool_output: str | None) -> AgentTurn: ...

    def teacher_force_plan(
        self,
        *,
        messages: list[dict[str, str]],
        plan_text: str,
    ) -> tuple[tuple[TokenLogprob, ...], ...]: ...


@dataclass
class _EpisodeState:
    context: TaskContext
    config: RunConfiguration
    history: list[dict[str, str]] = field(default_factory=list)
    assistant_blocks: list[list[object]] = field(default_factory=list)


def _logprob_token_id(entry: object) -> int:
    """The token id of one vLLM chat logprob entry.

    vLLM's chat logprob entries carry ``token``, ``bytes``, and ``logprob``
    but no id field. ``completion_payload`` sets
    ``return_tokens_as_token_ids``, which makes ``token`` the string
    ``token_id:<id>``; any other shape fails closed.
    """

    from llm_behavior_ci.runtime.episode import RuntimeUnavailable

    if not isinstance(entry, Mapping):
        raise RuntimeUnavailable("token_id is missing")
    token = entry.get("token")
    if not isinstance(token, str) or not token.startswith(_TOKEN_ID_PREFIX):
        raise RuntimeUnavailable(
            "token_id is missing: logprob token is not a token_id:<id> string"
        )
    digits = token[len(_TOKEN_ID_PREFIX):]
    if not digits.isdigit():
        raise RuntimeUnavailable(
            "token_id is missing: logprob token is not a token_id:<id> string"
        )
    return int(digits)


def generated_token_count(choice: Mapping[object, object]) -> int:
    """The number of generated tokens from a choice's ``token_ids``.

    ``completion_payload`` sets ``return_token_ids``, so vLLM returns the
    generated ids on every choice whether or not logprobs were requested.
    A missing or malformed list fails closed.
    """

    from llm_behavior_ci.runtime.episode import RuntimeUnavailable

    token_ids = choice.get("token_ids")
    if not isinstance(token_ids, list) or not all(
        isinstance(token, int) and not isinstance(token, bool) and token >= 0
        for token in token_ids
    ):
        raise RuntimeUnavailable("endpoint did not return generated token_ids")
    return len(token_ids)


def parse_logprobs(choice: Mapping[object, object]) -> tuple[tuple[TokenLogprob, ...], ...]:
    from llm_behavior_ci.runtime.episode import RuntimeUnavailable

    logprobs = choice["logprobs"]
    if not isinstance(logprobs, Mapping):
        raise RuntimeUnavailable("token_id is missing")
    content = logprobs["content"]
    if not isinstance(content, list):
        raise RuntimeUnavailable("token_id is missing")
    positions: list[tuple[TokenLogprob, ...]] = []
    for item in content:
        chosen_id = _logprob_token_id(item)
        chosen_logprob = float(item["logprob"])
        alternatives = [
            TokenLogprob(token_id=chosen_id, logprob=chosen_logprob, rank=0)
        ]
        seen = {chosen_id}
        rank = 1
        top = item.get("top_logprobs", [])
        if not isinstance(top, list):
            raise RuntimeUnavailable("token_id is missing")
        for alternative in top:
            token_id = _logprob_token_id(alternative)
            if token_id in seen:
                continue
            seen.add(token_id)
            alternatives.append(
                TokenLogprob(
                    token_id=token_id,
                    logprob=float(alternative["logprob"]),
                    rank=rank,
                )
            )
            rank += 1
        positions.append(tuple(alternatives))
    token_ids = choice.get("token_ids")
    if not isinstance(token_ids, list):
        raise RuntimeUnavailable("token_ids is missing")
    if len(token_ids) != len(content):
        raise RuntimeUnavailable(
            "token_ids and logprobs.content length differ"
        )
    for index, position in enumerate(positions):
        if int(token_ids[index]) != position[0].token_id:
            raise RuntimeUnavailable(
                f"rank-0 token_id does not match token_ids at position {index}"
            )
    return tuple(positions)


def resolve_action_interface(action_interface: str) -> str:
    from llm_behavior_ci.runtime.episode import RuntimeUnavailable

    if action_interface not in ACTION_INTERFACES:
        raise RuntimeUnavailable(
            f"unsupported action_interface: {action_interface}"
        )
    return action_interface


def action_execution_backend() -> str:
    return "appworld_session.execute"


def reject_local_python_executor() -> None:
    from llm_behavior_ci.runtime.episode import RuntimeUnavailable

    raise RuntimeUnavailable(
        "smolagents LocalPythonExecutor is not on the action path; "
        "actions execute through AppWorldSession.execute"
    )


def validate_chat_request(config: RunConfiguration) -> None:
    """Check the parts of a configuration a chat request itself carries.

    Serving flags (``model.serving``) are applied when the server is
    launched from ``runtime.launch_spec.build_vllm_launch_spec``, not per
    request, so they are not checked here; ``check_model_identity`` lists
    them as unchecked against the running server.
    """

    resolve_action_interface(config.agent.action_interface)


def check_model_identity(
    config: RunConfiguration,
    reported: Mapping[str, object],
) -> list[str]:
    from llm_behavior_ci.runtime.episode import RuntimeUnavailable

    repository = config.model.model.repository
    revision = config.model.model.revision
    reported_id = reported.get("id")
    if reported_id is None:
        reported_id = reported.get("model")
    reported_root = reported.get("root")
    reported_revision = reported.get("revision") or reported.get(
        "parent"
    )
    if reported_id is not None and str(reported_id) not in {
        repository,
        repository.split("/")[-1],
    }:
        if reported_root is None or str(reported_root) != repository:
            raise RuntimeUnavailable(
                "served model id does not match configured repository"
            )
    if reported_revision is not None and str(reported_revision) != revision:
        raise RuntimeUnavailable(
            "served model revision does not match configured revision"
        )
    unchecked = list(_UNCHECKED_IDENTITY_FIELDS)
    if reported_id is None and reported_root is None:
        unchecked.insert(0, "model.model.repository")
    if reported_revision is None:
        unchecked.insert(0, "model.model.revision")
    return unchecked


class AppWorldActionExecutor(_SmolPythonExecutor):
    """Delegates a ``CodeAgent``-style code action to AppWorld's own shell.

    Structurally satisfies smolagents' ``PythonExecutor`` interface
    (``send_tools``, ``send_variables``, ``__call__``) and, when smolagents
    is installed, is also a genuine subclass of it. It never evaluates the
    action itself; ``__call__`` forwards the raw text straight to
    ``AppWorldSession.execute``, which is the only place a mutation happens
    (`DECISIONS.md` D18/D20). ``smolagents.local_python_executor.LocalPythonExecutor``
    is never constructed on this path.
    """

    def __init__(self, execute: Callable[[str], object]) -> None:
        self._execute = execute
        self._tools: dict[str, object] = {}
        self._variables: dict[str, object] = {}

    def send_tools(self, tools: dict[str, object]) -> None:
        self._tools = dict(tools)

    def send_variables(self, variables: dict[str, object]) -> None:
        self._variables = dict(variables)

    def __call__(self, code_action: str) -> object:
        return self._execute(code_action)


class AppWorldExecuteTool(_SmolTool):
    """Exposes one AppWorld action as a smolagents-style tool.

    This is the ``ToolCallingAgent`` counterpart to
    ``AppWorldActionExecutor``'s ``CodeAgent`` shape (`DECISIONS.md` D18's
    two options). When smolagents is installed this is a genuine
    ``smolagents.Tool`` subclass and passes its own argument and signature
    validation; ``forward`` still only ever calls into
    ``AppWorldSession.execute``. ``output_type`` is ``"object"`` because the
    return value is the local ``ToolResult`` record, not raw text.
    """

    name = "appworld_execute"
    description = (
        "Execute one AppWorld action or tool call against the current "
        "task's isolated world and return the raw result."
    )
    inputs = {
        "action": {
            "type": "string",
            "description": "The action payload to execute through AppWorld.",
        }
    }
    output_type = "object"

    def __init__(self, execute: Callable[[str], object]) -> None:
        if _SmolTool is not object:
            super().__init__()
        else:
            self.is_initialized = False
        self._execute = execute

    def setup(self) -> None:
        self.is_initialized = True

    def forward(self, action: str) -> object:
        return self._execute(action)

    def __call__(self, action: str) -> object:
        if not self.is_initialized:
            self.setup()
        return self.forward(action)


def bind_appworld_action_executor(
    execute: Callable[[str], object],
) -> AppWorldActionExecutor:
    return AppWorldActionExecutor(execute)


def served_model_id(config: RunConfiguration) -> str:
    """The chat-completions ``model`` field this configuration serves as.

    A LoRA request names the adapter, not the base repository: vLLM routes
    a request to the loaded LoRA module by matching ``model`` against the
    name ``runtime.launch_spec.build_vllm_launch_spec`` registered with
    ``--lora-modules``, which is this same ``lora.repository``. The healthy
    configuration (``model.lora`` unset) is unaffected and keeps naming the
    base repository.
    """

    lora = config.model.lora
    if lora is not None:
        return lora.repository
    return config.model.model.repository


def build_appworld_executor(
    action_interface: str,
    execute: Callable[[str], object],
) -> AppWorldActionExecutor | AppWorldExecuteTool:
    """Bind one AppWorld executor matching the configured action interface.

    ``code`` returns the ``PythonExecutor``-shaped ``AppWorldActionExecutor``;
    ``tool_calling`` returns the ``Tool``-shaped ``AppWorldExecuteTool``. Both
    wrap the same ``execute`` callable and both call it, unchanged, from
    their ``__call__``; only the smolagents-facing shape differs. D18's
    action-interface choice is read from configuration rather than decided
    here.
    """

    resolve_action_interface(action_interface)
    if action_interface == "code":
        return AppWorldActionExecutor(execute)
    return AppWorldExecuteTool(execute)


@dataclass(frozen=True)
class _FallbackChatMessage:
    """Stand-in for ``smolagents.ChatMessage`` when smolagents is absent.

    Exposes the same ``content``/``raw``/``tool_calls`` attributes so
    ``SmolagentsVLLMAgent.next_turn`` reads a generated message the same
    way regardless of whether the real package is installed. Only
    ``build_runtime`` requires smolagents to actually be present on the
    live path; CPU tests exercise this exact control flow through this
    fallback.
    """

    role: str
    content: str | None
    tool_calls: None
    raw: Mapping[object, object] | None


def _build_chat_message(
    *, role: str, content: str, raw: Mapping[object, object]
) -> object:
    if _SmolChatMessage is not None:
        return _SmolChatMessage(role=role, content=content, tool_calls=None, raw=raw)
    return _FallbackChatMessage(role=role, content=content, tool_calls=None, raw=raw)


class SmolagentsVLLMAgent(_SmolModel):
    """The one live ``AgentLoop``: a smolagents model served by vLLM.

    Subclasses ``smolagents.Model`` when smolagents is installed, so
    ``generate`` is a genuine smolagents model call, not a re-implementation
    of one; ``next_turn`` drives the per-episode turn loop and calls
    ``self.generate`` for the model step. Falls back to a plain ``object``
    base so the same control flow is exercised by CPU tests that do not
    install smolagents (`CONSTRAINTS.md` keeps the hosted CPU suite
    smolagents-free). ``build_runtime`` is the one path that requires the
    real package and checks its version against
    ``config.agent.smolagents_version``.

    Tool execution is not performed here: ``run_episode`` executes the
    parsed action through ``build_appworld_executor`` against
    ``AppWorldSession.execute`` directly, so AppWorld stays the only place a
    mutation happens regardless of which action interface is configured.
    """

    def __init__(self, base_url: str) -> None:
        if _SmolModel is not object:
            super().__init__(model_id=base_url)
        self._base_url = base_url.rstrip("/")
        self._mode = "execute"
        self._local = threading.local()

    @property
    def base_url(self) -> str:
        return self._base_url

    def set_mode(self, mode: str) -> None:
        if mode not in {"plan", "execute"}:
            from llm_behavior_ci.runtime.episode import RuntimeUnavailable

            raise RuntimeUnavailable("mode must be plan or execute")
        self._mode = mode

    def begin(self, context: TaskContext, config: RunConfiguration) -> None:
        validate_chat_request(config)
        try:
            render_system_text(
                prompt_version=config.agent.prompt.prompt_version,
                plan_format_version=config.agent.prompt.plan_format_version,
                thinking_enabled=config.agent.prompt.thinking_enabled,
                action_interface=config.agent.action_interface,
                mode=self._mode,
            )
        except UnknownPromptVersion as error:
            from llm_behavior_ci.runtime.episode import RuntimeUnavailable

            raise RuntimeUnavailable(str(error)) from error
        self._local.state = _EpisodeState(context=context, config=config)

    def _state(self) -> _EpisodeState:
        state = getattr(self._local, "state", None)
        if state is None:
            raise RuntimeError("agent begin was not called")
        return state

    def _system_text(self) -> str:
        state = self._state()
        prompt = state.config.agent.prompt
        try:
            return render_system_text(
                prompt_version=prompt.prompt_version,
                plan_format_version=prompt.plan_format_version,
                thinking_enabled=prompt.thinking_enabled,
                action_interface=state.config.agent.action_interface,
                mode=self._mode,
            )
        except UnknownPromptVersion as error:
            from llm_behavior_ci.runtime.episode import RuntimeUnavailable

            raise RuntimeUnavailable(str(error)) from error

    def _api_documentation(self) -> str:
        state = self._state()
        agent = state.config.agent
        text = state.context.api_documentation
        source = getattr(state.context, "api_documentation_source", None)
        if (
            agent.prompt.prompt_version == PROMPT_RUNTIME_AUTH_V2
            and source is not None
        ):
            text = render_api_documentation(source, include_constraints=True)
        try:
            return resolve_api_documentation(
                text,
                api_docs_version=agent.api_docs_version,
                api_docs_app=agent.api_docs_app,
            )
        except ApiDocsCorruptionError as error:
            from llm_behavior_ci.runtime.episode import RuntimeUnavailable

            raise RuntimeUnavailable(str(error)) from error

    def messages(self, tool_output: str | None = None) -> list[dict[str, str]]:
        state = self._state()
        built = [
            {"role": "system", "content": self._system_text()},
            {
                "role": "user",
                "content": (
                    f"{state.context.instruction}\n"
                    f"{self._api_documentation()}"
                ),
            },
        ]
        built.extend(state.history)
        if tool_output is not None:
            built.append({"role": "user", "content": tool_output})
        return built

    def completion_payload(
        self, messages: list[dict[str, str]]
    ) -> dict[str, object]:
        """The raw vLLM chat-completions body for one agent turn.

        The body is posted as JSON, not through an OpenAI client, so vLLM's
        extension fields (``top_k``, ``min_p``, ``chat_template_kwargs``,
        ``return_tokens_as_token_ids``, ``return_token_ids``) sit at the top
        level; vLLM ignores a literal ``extra_body`` key.

        Plan mode requests ``logprobs`` and ``top_logprobs`` because plan
        capture reads those tables. Execute mode omits both: its A/A metrics
        use the parsed action, the tool trajectory, and the evaluator
        outcome, and its token count comes from ``token_ids``. The
        teacher-forced plan KL request is ``teacher_force_payload``.
        ``max_tokens`` is ``SamplingSettings.generation_max_tokens`` for
        the current mode.
        """

        state = self._state()
        validate_chat_request(state.config)
        sampling = state.config.agent.sampling
        payload: dict[str, object] = {
            "model": served_model_id(state.config),
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.generation_max_tokens(self._mode),
            "seed": sampling.seed,
            "messages": messages,
            "top_k": sampling.top_k,
            "min_p": sampling.min_p,
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
            "chat_template_kwargs": {
                "enable_thinking": state.config.agent.prompt.thinking_enabled
            },
        }
        if self._mode == "plan":
            payload["logprobs"] = True
            payload["top_logprobs"] = state.config.model.serving.max_logprobs
        return payload

    def teacher_force_payload(
        self,
        *,
        messages: list[dict[str, str]],
        plan_text: str,
    ) -> dict[str, object]:
        """The raw vLLM body that scores a frozen plan as prompt tokens.

        ``max_tokens`` is 1 because vLLM rejects 0; the one generated token
        is not read. The scored positions come from ``prompt_logprobs`` and
        ``prompt_token_ids``, requested with the other vLLM extension
        fields at the top level.
        """

        state = self._state()
        validate_chat_request(state.config)
        if plan_text == "":
            from llm_behavior_ci.runtime.episode import RuntimeUnavailable

            raise RuntimeUnavailable("plan_text is required for teacher forcing")
        sampling = state.config.agent.sampling
        forced_messages = list(messages) + [
            {"role": "assistant", "content": plan_text}
        ]
        return {
            "model": served_model_id(state.config),
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": 1,
            "seed": sampling.seed,
            "messages": forced_messages,
            "top_k": sampling.top_k,
            "min_p": sampling.min_p,
            "prompt_logprobs": state.config.model.serving.max_logprobs,
            "return_token_ids": True,
            "add_generation_prompt": False,
            "chat_template_kwargs": {
                "enable_thinking": state.config.agent.prompt.thinking_enabled
            },
        }

    def parse_model_output(
        self, text: str
    ) -> tuple[str | None, str | None, str | None]:
        return parse_model_output(text)

    def generate(
        self,
        messages: list[dict[str, str]],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[object] | None = None,
        **kwargs: object,
    ) -> object:
        """The smolagents ``Model.generate`` call this agent loop runs on.

        Signature-compatible with ``smolagents.Model.generate`` so this
        class is a drop-in model wherever smolagents expects one.
        ``stop_sequences``, ``response_format``, and ``tools_to_call_from``
        are accepted for that compatibility and not sent: prompt-v1/plan-v1
        already carry the action-interface instruction as system text, and
        per-API structured tool schemas need a live AppWorld catalog this
        environment does not have (see the final report). Returns a
        ``smolagents.ChatMessage`` when smolagents is installed, else the
        attribute-compatible ``_FallbackChatMessage``; either way ``.raw``
        holds the full decoded response so logprobs are not lost the way
        they would be through ``smolagents.OpenAIServerModel``.
        """

        del stop_sequences, response_format, tools_to_call_from, kwargs
        payload = self.completion_payload(messages)
        raw = self._post(payload)
        choices = raw["choices"]
        if not isinstance(choices, list) or not choices:
            from llm_behavior_ci.runtime.episode import RuntimeUnavailable

            raise RuntimeUnavailable("token_id is missing")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            from llm_behavior_ci.runtime.episode import RuntimeUnavailable

            raise RuntimeUnavailable("token_id is missing")
        message = choice.get("message", {})
        output_text = ""
        if isinstance(message, Mapping):
            content = message.get("content")
            if isinstance(content, str):
                output_text = content
        return _build_chat_message(role="assistant", content=output_text, raw=raw)

    def generate_turn(
        self,
        *,
        tool_output: str | None,
        extra_instruction: str | None = None,
        parse_action: bool = True,
    ) -> AgentTurn:
        state = self._state()
        started_at = wall_now()
        if tool_output is not None:
            state.history.append({"role": "user", "content": tool_output})
        messages = self.messages()
        if extra_instruction is not None:
            messages.append({"role": "user", "content": extra_instruction})
        began = monotonic()
        chat_message = self.generate(messages)
        latency_seconds = monotonic() - began
        output_text = chat_message.content or ""
        raw = chat_message.raw
        choice = raw["choices"][0]
        token_count = generated_token_count(choice)
        if self._mode == "plan":
            logprobs = parse_logprobs(choice)
            action, app_name, api_name = None, None, None
            rejection = None
        elif parse_action:
            logprobs = ()
            rejection = None
            try:
                action, app_name, api_name = parse_model_output(output_text)
            except ActionRejected as error:
                action, app_name, api_name = None, None, None
                rejection = str(error)
        else:
            logprobs = ()
            action, app_name, api_name = None, None, None
            rejection = None
        state.history.append({"role": "assistant", "content": output_text})
        return AgentTurn(
            prompt_text=messages[-1]["content"],
            output_text=output_text,
            top_k_logprobs=logprobs,
            generated_token_count=token_count,
            latency_seconds=latency_seconds,
            started_at=started_at,
            action=action,
            app_name=app_name,
            api_name=api_name,
            rejection=rejection,
        )

    def next_turn(self, *, tool_output: str | None) -> AgentTurn:
        return self.generate_turn(
            tool_output=tool_output,
            extra_instruction=None,
            parse_action=True,
        )

    def plan_prefix_payload(
        self,
        *,
        messages: list[dict[str, str]],
    ) -> dict[str, object]:
        """Build the ``/tokenize`` request for the prompt that precedes the plan.

        The prefix is the conversation plus the assistant generation header,
        rendered with the same chat-template arguments as the forced request,
        so its tokens are the positions that come before the first plan token.
        """

        state = self._state()
        return {
            "model": served_model_id(state.config),
            "messages": list(messages),
            "add_generation_prompt": True,
            "chat_template_kwargs": {
                "enable_thinking": state.config.agent.prompt.thinking_enabled
            },
        }

    def teacher_force_plan(
        self,
        *,
        messages: list[dict[str, str]],
        plan_text: str,
    ) -> tuple[tuple[TokenLogprob, ...], ...]:
        """Score only the forced plan tokens, not the shared prompt before them.

        Tokenizes the prompt without the plan, requires the forced request's
        ``prompt_token_ids`` to start with exactly those tokens, and returns
        positions from that boundary on: the plan tokens and the chat
        template's end-of-turn tokens after them. A prefix mismatch fails
        closed, because the plan boundary would be unknown.
        """

        from llm_behavior_ci.runtime.episode import RuntimeUnavailable

        payload = self.teacher_force_payload(messages=messages, plan_text=plan_text)
        prefix = self._post(self.plan_prefix_payload(messages=messages), path="/tokenize")
        prefix_tokens = prefix.get("tokens")
        if (
            not isinstance(prefix_tokens, list)
            or not prefix_tokens
            or not all(isinstance(token, int) for token in prefix_tokens)
        ):
            raise RuntimeUnavailable(
                "endpoint did not return prompt tokens for the plan prefix"
            )
        raw = self._post(payload)
        prompt_logprobs = raw.get("prompt_logprobs")
        prompt_token_ids = raw.get("prompt_token_ids")
        if not isinstance(prompt_logprobs, list):
            raise RuntimeUnavailable(
                "endpoint did not return prompt logprobs for the frozen plan"
            )
        if not isinstance(prompt_token_ids, list):
            raise RuntimeUnavailable(
                "endpoint did not return prompt_token_ids; the forced token at "
                "each position cannot be identified"
            )
        boundary = len(prefix_tokens)
        if (
            len(prompt_token_ids) <= boundary
            or [int(token) for token in prompt_token_ids[:boundary]] != prefix_tokens
        ):
            raise RuntimeUnavailable(
                "forced prompt does not start with the plan prefix tokens; "
                "the plan boundary cannot be located"
            )
        return _parse_prompt_logprobs(prompt_logprobs, prompt_token_ids, start=boundary)

    def _post(
        self,
        payload: dict[str, object],
        *,
        path: str = "/v1/chat/completions",
    ) -> dict[str, object]:
        from llm_behavior_ci.runtime.episode import RuntimeUnavailable

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise _http_runtime_error(error, request.full_url) from error
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as error:
            raise RuntimeUnavailable(str(error)) from error


def _parse_prompt_logprobs(
    prompt_logprobs: list[object],
    prompt_token_ids: object,
    *,
    start: int = 0,
) -> tuple[tuple[TokenLogprob, ...], ...]:
    """Parse vLLM's ``prompt_logprobs`` into per-position ``TokenLogprob`` tuples.

    ``prompt_logprobs[i]`` is a mapping of alternative token id to its
    logprob at position ``i``; it never marks which entry is the token
    that was actually forced there, so ``prompt_token_ids`` (from
    ``return_token_ids``) is required to identify it unambiguously. The
    forced token is always placed at ``rank=0``; the remaining
    alternatives are ordered by ascending token id, a canonical,
    model-independent order so that two teacher-forced calls over the
    same support compare as aligned regardless of how each model ranks
    its own probabilities. Positions before ``start`` are validated but not
    returned; position 0 is the only one allowed to carry no logprobs.
    """

    from llm_behavior_ci.runtime.episode import RuntimeUnavailable

    if not isinstance(prompt_token_ids, list):
        raise RuntimeUnavailable(
            "endpoint did not return prompt_token_ids; the forced token at "
            "each position cannot be identified"
        )
    if len(prompt_token_ids) != len(prompt_logprobs):
        raise RuntimeUnavailable(
            "prompt_logprobs and prompt_token_ids length differ"
        )
    positions: list[tuple[TokenLogprob, ...]] = []
    for index, (item, forced_token_id) in enumerate(
        zip(prompt_logprobs, prompt_token_ids)
    ):
        if item is None:
            if index == 0:
                continue
            raise RuntimeUnavailable(
                f"endpoint returned no prompt logprobs at position {index}"
            )
        if not isinstance(item, Mapping):
            raise RuntimeUnavailable(
                "endpoint did not return prompt or echo logprobs for the frozen plan"
            )
        entries: dict[int, float] = {}
        for token_id_key, payload in item.items():
            if not isinstance(payload, Mapping):
                raise RuntimeUnavailable(
                    "endpoint did not return prompt or echo logprobs for the frozen plan"
                )
            token_id = int(payload.get("token_id", token_id_key))
            entries[token_id] = float(payload["logprob"])
        forced_token_id = int(forced_token_id)
        if forced_token_id not in entries:
            raise RuntimeUnavailable(
                f"forced token is missing from returned logprobs at position {index}"
            )
        if index < start:
            continue
        alternatives = [
            TokenLogprob(
                token_id=forced_token_id,
                logprob=entries[forced_token_id],
                rank=0,
            )
        ]
        rank = 1
        for token_id in sorted(entries):
            if token_id == forced_token_id:
                continue
            alternatives.append(
                TokenLogprob(token_id=token_id, logprob=entries[token_id], rank=rank)
            )
            rank += 1
        positions.append(tuple(alternatives))
    if not positions:
        raise RuntimeUnavailable(
            "endpoint did not return prompt or echo logprobs for the frozen plan"
        )
    return tuple(positions)


def _redact_secret(text: str, secret: str) -> str:
    if secret == "":
        return text
    return text.replace(secret, "[redacted]")


def _hosted_context_length(body: str) -> bool:
    if _context_length_exceeded(body):
        return True
    text = body.lower()
    if "model_context_window_exceeded" in text:
        return True
    if "prompt is too long" in text:
        return True
    return "context window" in text and ("exceed" in text or "too long" in text)


def _retry_delay(attempt: int, error: urllib.error.HTTPError | None) -> float:
    if error is not None and error.headers is not None:
        header = error.headers.get("retry-after")
        if isinstance(header, str) and header.isdigit():
            seconds = int(header)
            if 0 < seconds <= 20:
                return float(seconds)
    return min(8.0, 0.5 * (2**attempt))


def _anthropic_http_error(
    error: urllib.error.HTTPError,
    secret: str,
) -> RuntimeError:
    from llm_behavior_ci.runtime.episode import RuntimeUnavailable

    body = _redact_secret(_http_error_body(error), secret)
    if len(body) > 4000:
        body = body[:4000]
    reason = (
        "context_length_exceeded" if _hosted_context_length(body) else None
    )
    message = _redact_secret(
        (
            f"Anthropic request failed with HTTP {error.code} {error.reason}:\n"
            f"{body}\n"
            f"endpoint: {ANTHROPIC_MESSAGES_URL}"
        ),
        secret,
    )
    return RuntimeUnavailable(message, reason=reason)


def _replay_blocks(content: object) -> list[object]:
    if not isinstance(content, list):
        from llm_behavior_ci.runtime.episode import RuntimeUnavailable

        raise RuntimeUnavailable("Anthropic response was malformed")
    return json.loads(json.dumps(content))


def _visible_response_text(content: object) -> str:
    from llm_behavior_ci.runtime.episode import RuntimeUnavailable

    if not isinstance(content, list):
        raise RuntimeUnavailable("Anthropic response was malformed")
    parts: list[str] = []
    for block in content:
        if not isinstance(block, Mapping):
            raise RuntimeUnavailable("Anthropic response was malformed")
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise RuntimeUnavailable("Anthropic response was malformed")
            parts.append(text)
            continue
        if block_type in {"thinking", "redacted_thinking", "refusal"}:
            continue
        raise RuntimeUnavailable("Anthropic response was malformed")
    return "".join(parts)


def _output_token_count(payload: Mapping[object, object]) -> int:
    from llm_behavior_ci.runtime.episode import RuntimeUnavailable

    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        raise RuntimeUnavailable("Anthropic response was malformed")
    count = usage.get("output_tokens")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise RuntimeUnavailable("Anthropic response was malformed")
    return count


class SmolagentsAnthropicAgent:
    """An ``AgentLoop`` whose generations come from the Anthropic Messages API.

    Prompt rendering, history, action parsing, and the workflow controller
    stay on the same path as ``SmolagentsVLLMAgent``. This class replaces
    only the model request. The API key is an HTTP header and is omitted
    from ``repr``, configuration JSON, and exception text.
    """

    def __init__(self, api_key: str) -> None:
        if not isinstance(api_key, str) or api_key.strip() == "":
            from llm_behavior_ci.runtime.episode import EpisodeRejected

            raise EpisodeRejected("ANTHROPIC_API_KEY is required")
        self._api_key = api_key.strip()
        self._mode = "execute"
        self._local = threading.local()

    def __repr__(self) -> str:
        return "SmolagentsAnthropicAgent(provider='anthropic')"

    def set_mode(self, mode: str) -> None:
        if mode not in {"plan", "execute"}:
            from llm_behavior_ci.runtime.episode import RuntimeUnavailable

            raise RuntimeUnavailable("mode must be plan or execute")
        self._mode = mode

    def begin(self, context: TaskContext, config: RunConfiguration) -> None:
        validate_chat_request(config)
        self._require_anthropic(config)
        try:
            render_system_text(
                prompt_version=config.agent.prompt.prompt_version,
                plan_format_version=config.agent.prompt.plan_format_version,
                thinking_enabled=config.agent.prompt.thinking_enabled,
                action_interface=config.agent.action_interface,
                mode=self._mode,
            )
        except UnknownPromptVersion as error:
            from llm_behavior_ci.runtime.episode import RuntimeUnavailable

            raise RuntimeUnavailable(str(error)) from error
        self._local.state = _EpisodeState(context=context, config=config)

    def _state(self) -> _EpisodeState:
        state = getattr(self._local, "state", None)
        if state is None:
            raise RuntimeError("agent begin was not called")
        return state

    def _require_anthropic(self, config: RunConfiguration) -> AnthropicModelConfiguration:
        from llm_behavior_ci.runtime.episode import RuntimeUnavailable

        model = config.model
        sampling = config.agent.sampling
        if not isinstance(model, AnthropicModelConfiguration):
            raise RuntimeUnavailable(
                "Anthropic agent requires an Anthropic model configuration"
            )
        if not isinstance(sampling, AnthropicSamplingSettings):
            raise RuntimeUnavailable(
                "Anthropic agent requires unset vLLM sampling controls"
            )
        return model

    def _system_text(self) -> str:
        state = self._state()
        prompt = state.config.agent.prompt
        try:
            return render_system_text(
                prompt_version=prompt.prompt_version,
                plan_format_version=prompt.plan_format_version,
                thinking_enabled=prompt.thinking_enabled,
                action_interface=state.config.agent.action_interface,
                mode=self._mode,
            )
        except UnknownPromptVersion as error:
            from llm_behavior_ci.runtime.episode import RuntimeUnavailable

            raise RuntimeUnavailable(str(error)) from error

    def _api_documentation(self) -> str:
        state = self._state()
        agent = state.config.agent
        text = state.context.api_documentation
        source = getattr(state.context, "api_documentation_source", None)
        if (
            agent.prompt.prompt_version == PROMPT_RUNTIME_AUTH_V2
            and source is not None
        ):
            text = render_api_documentation(source, include_constraints=True)
        try:
            return resolve_api_documentation(
                text,
                api_docs_version=agent.api_docs_version,
                api_docs_app=agent.api_docs_app,
            )
        except ApiDocsCorruptionError as error:
            from llm_behavior_ci.runtime.episode import RuntimeUnavailable

            raise RuntimeUnavailable(str(error)) from error

    def messages(self, tool_output: str | None = None) -> list[dict[str, str]]:
        state = self._state()
        built = [
            {"role": "system", "content": self._system_text()},
            {
                "role": "user",
                "content": (
                    f"{state.context.instruction}\n"
                    f"{self._api_documentation()}"
                ),
            },
        ]
        built.extend(state.history)
        if tool_output is not None:
            built.append({"role": "user", "content": tool_output})
        return built

    def completion_payload(
        self, messages: list[dict[str, str]]
    ) -> dict[str, object]:
        """The Anthropic Messages body for one agent turn.

        The system prompt is the top-level ``system`` field. Conversation
        messages keep the task, API documentation, and append-only history.
        Sampling controls that Claude Sonnet 5.5 rejects are omitted.
        ``max_tokens`` is the mode-specific generation cap.
        """

        state = self._state()
        model = self._require_anthropic(state.config)
        sampling = state.config.agent.sampling
        if not isinstance(sampling, AnthropicSamplingSettings):
            from llm_behavior_ci.runtime.episode import RuntimeUnavailable

            raise RuntimeUnavailable(
                "Anthropic agent requires unset vLLM sampling controls"
            )
        if not messages or messages[0].get("role") != "system":
            from llm_behavior_ci.runtime.episode import RuntimeUnavailable

            raise RuntimeUnavailable("system prompt is missing")
        system = messages[0].get("content")
        if not isinstance(system, str) or system == "":
            from llm_behavior_ci.runtime.episode import RuntimeUnavailable

            raise RuntimeUnavailable("system prompt is missing")
        conversation: list[dict[str, object]] = []
        for message in messages[1:]:
            role = message.get("role")
            content = message.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                from llm_behavior_ci.runtime.episode import RuntimeUnavailable

                raise RuntimeUnavailable("Anthropic conversation was malformed")
            conversation.append({"role": role, "content": content})
        if not conversation or conversation[-1]["role"] != "user":
            from llm_behavior_ci.runtime.episode import RuntimeUnavailable

            raise RuntimeUnavailable(
                "Anthropic requests must end with a user message"
            )
        block_index = 0
        for message in conversation:
            if message["role"] != "assistant":
                continue
            if block_index >= len(state.assistant_blocks):
                from llm_behavior_ci.runtime.episode import RuntimeUnavailable

                raise RuntimeUnavailable(
                    "Anthropic assistant history is missing response blocks"
                )
            message["content"] = state.assistant_blocks[block_index]
            block_index += 1
        if block_index != len(state.assistant_blocks):
            from llm_behavior_ci.runtime.episode import RuntimeUnavailable

            raise RuntimeUnavailable(
                "Anthropic assistant history is missing response blocks"
            )
        return {
            "model": model.model_id,
            "max_tokens": sampling.generation_max_tokens(self._mode),
            "system": system,
            "messages": conversation,
            "thinking": {"type": model.thinking_mode},
            "output_config": {"effort": model.effort},
        }

    def parse_model_output(
        self, text: str
    ) -> tuple[str | None, str | None, str | None]:
        return parse_model_output(text)

    def _headers(self) -> dict[str, str]:
        model = self._require_anthropic(self._state().config)
        return {
            "content-type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": model.api_version,
        }

    def _read_message(self, payload: Mapping[object, object]) -> tuple[str, int]:
        from llm_behavior_ci.runtime.episode import RuntimeUnavailable

        if payload.get("stop_reason") == "model_context_window_exceeded":
            raise RuntimeUnavailable(
                "context_length_exceeded\n"
                "Anthropic stop_reason model_context_window_exceeded",
                reason="context_length_exceeded",
            )
        return _visible_response_text(payload.get("content")), _output_token_count(
            payload
        )

    def generate_turn(
        self,
        *,
        tool_output: str | None,
        extra_instruction: str | None = None,
        parse_action: bool = True,
    ) -> AgentTurn:
        if self._mode == "plan":
            raise UnsupportedCapability(
                "Anthropic does not expose prompt-token logprobs; "
                "plan-mode scoring is unsupported"
            )
        state = self._state()
        started_at = wall_now()
        if tool_output is not None:
            state.history.append({"role": "user", "content": tool_output})
        messages = self.messages()
        if extra_instruction is not None:
            messages.append({"role": "user", "content": extra_instruction})
        payload = self.completion_payload(messages)
        began = monotonic()
        raw = self._post(payload)
        latency_seconds = monotonic() - began
        output_text, token_count = self._read_message(raw)
        replay = _replay_blocks(raw.get("content"))
        if parse_action:
            rejection = None
            try:
                action, app_name, api_name = parse_model_output(output_text)
            except ActionRejected as error:
                action, app_name, api_name = None, None, None
                rejection = str(error)
        else:
            action, app_name, api_name = None, None, None
            rejection = None
        state.history.append({"role": "assistant", "content": output_text})
        state.assistant_blocks.append(replay)
        return AgentTurn(
            prompt_text=messages[-1]["content"],
            output_text=output_text,
            top_k_logprobs=(),
            generated_token_count=token_count,
            latency_seconds=latency_seconds,
            started_at=started_at,
            action=action,
            app_name=app_name,
            api_name=api_name,
            rejection=rejection,
        )

    def next_turn(self, *, tool_output: str | None) -> AgentTurn:
        return self.generate_turn(
            tool_output=tool_output,
            extra_instruction=None,
            parse_action=True,
        )

    def teacher_force_plan(
        self,
        *,
        messages: list[dict[str, str]],
        plan_text: str,
    ) -> tuple[tuple[TokenLogprob, ...], ...]:
        del messages, plan_text
        raise UnsupportedCapability(
            "Anthropic does not expose prompt-token logprobs; "
            "teacher-forced plan KL is unsupported"
        )

    def _post(self, payload: dict[str, object]) -> dict[str, object]:
        from llm_behavior_ci.runtime.episode import RuntimeUnavailable

        body = json.dumps(payload).encode("utf-8")
        if self._api_key.encode("utf-8") in body:
            raise RuntimeUnavailable("Anthropic request was malformed")
        headers = self._headers()
        last_error: BaseException | None = None
        for attempt in range(_ANTHROPIC_ATTEMPTS):
            request = urllib.request.Request(
                ANTHROPIC_MESSAGES_URL,
                data=body,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=240) as response:
                    raw = response.read()
                break
            except urllib.error.HTTPError as error:
                last_error = error
                if (
                    error.code in _ANTHROPIC_RETRYABLE
                    and attempt + 1 < _ANTHROPIC_ATTEMPTS
                ):
                    time.sleep(_retry_delay(attempt, error))
                    continue
                raise _anthropic_http_error(error, self._api_key) from error
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last_error = error
                if attempt + 1 < _ANTHROPIC_ATTEMPTS:
                    time.sleep(_retry_delay(attempt, None))
                    continue
                raise RuntimeUnavailable(
                    _redact_secret(str(error), self._api_key)
                ) from error
        else:
            raise RuntimeUnavailable(
                _redact_secret(str(last_error), self._api_key)
            )
        try:
            if isinstance(raw, str):
                decoded = json.loads(raw)
            else:
                decoded = json.loads(bytes(raw).decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeUnavailable("Anthropic response was malformed") from error
        if not isinstance(decoded, dict):
            raise RuntimeUnavailable("Anthropic response was malformed")
        return decoded


def _openai_compatible_http_error(
    error: urllib.error.HTTPError,
    *,
    label: str,
    endpoint: str,
    secret: str,
) -> RuntimeError:
    from llm_behavior_ci.runtime.episode import RuntimeUnavailable

    body = _redact_secret(_http_error_body(error), secret)
    if len(body) > 4000:
        body = body[:4000]
    reason = "context_length_exceeded" if _hosted_context_length(body) else None
    message = _redact_secret(
        (
            f"{label} request failed with HTTP {error.code} {error.reason}:\n"
            f"{body}\n"
            f"endpoint: {endpoint}"
        ),
        secret,
    )
    return RuntimeUnavailable(message, reason=reason)


def _read_chat_completion(
    payload: Mapping[object, object],
    label: str,
) -> tuple[str, int]:
    """Visible content and completion-token count of one chat completion.

    Only ``choices[0].message.content`` is read as model output;
    ``reasoning_content`` and any other message field are ignored. A
    ``length`` finish keeps whatever visible content was returned, and an
    absent content under ``length`` is the empty string. A context-window
    finish is a context-length runtime failure; ``network_error`` and
    ``sensitive`` are provider failures. Any other shape fails closed.
    """

    from llm_behavior_ci.runtime.episode import RuntimeUnavailable

    malformed = f"{label} response was malformed"
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeUnavailable(malformed)
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise RuntimeUnavailable(malformed)
    finish = choice.get("finish_reason")
    if finish == "model_context_window_exceeded":
        raise RuntimeUnavailable(
            "context_length_exceeded\n"
            f"{label} finish_reason model_context_window_exceeded",
            reason="context_length_exceeded",
        )
    if finish in _OPENAI_COMPATIBLE_PROVIDER_FAILURES:
        raise RuntimeUnavailable(f"{label} provider failure: finish_reason {finish}")
    if finish not in _OPENAI_COMPATIBLE_FINISHES:
        raise RuntimeUnavailable(f"{label} response had an unsupported finish_reason")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise RuntimeUnavailable(malformed)
    content = message.get("content")
    if content is None and finish == "length":
        content = ""
    if not isinstance(content, str):
        raise RuntimeUnavailable(malformed)
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        raise RuntimeUnavailable(malformed)
    count = usage.get("completion_tokens")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise RuntimeUnavailable(malformed)
    return content, count


class SmolagentsOpenAICompatibleAgent:
    """An ``AgentLoop`` whose generations come from a hosted Chat Completions API.

    Prompt rendering, history, action parsing, and the workflow controller
    stay on the same path as ``SmolagentsVLLMAgent``. This class replaces
    only the model request: a non-streaming ``POST`` to
    ``{api_base}/chat/completions`` with Bearer authentication and no
    native tools. Hidden reasoning is never read into model output or
    history. The API key is an HTTP header and is omitted from ``repr``,
    configuration JSON, request bodies, and exception text. The agent
    accepts only configurations whose provider matches its own, so a key
    is never sent to another provider's endpoint.
    """

    def __init__(self, provider: str, api_key: str) -> None:
        from llm_behavior_ci.runtime.episode import EpisodeRejected

        variable = OPENAI_COMPATIBLE_KEY_VARIABLES.get(provider)
        if variable is None:
            raise EpisodeRejected("unsupported OpenAI-compatible provider")
        if not isinstance(api_key, str) or api_key.strip() == "":
            raise EpisodeRejected(f"{variable} is required")
        self._provider = provider
        self._label = _OPENAI_COMPATIBLE_LABELS[provider]
        self._api_key = api_key.strip()
        self._mode = "execute"
        self._local = threading.local()

    @property
    def provider(self) -> str:
        return self._provider

    def __repr__(self) -> str:
        return f"SmolagentsOpenAICompatibleAgent(provider={self._provider!r})"

    def set_mode(self, mode: str) -> None:
        if mode not in {"plan", "execute"}:
            from llm_behavior_ci.runtime.episode import RuntimeUnavailable

            raise RuntimeUnavailable("mode must be plan or execute")
        self._mode = mode

    def begin(self, context: TaskContext, config: RunConfiguration) -> None:
        validate_chat_request(config)
        self._require_configuration(config)
        try:
            render_system_text(
                prompt_version=config.agent.prompt.prompt_version,
                plan_format_version=config.agent.prompt.plan_format_version,
                thinking_enabled=config.agent.prompt.thinking_enabled,
                action_interface=config.agent.action_interface,
                mode=self._mode,
            )
        except UnknownPromptVersion as error:
            from llm_behavior_ci.runtime.episode import RuntimeUnavailable

            raise RuntimeUnavailable(str(error)) from error
        self._local.state = _EpisodeState(context=context, config=config)

    def _state(self) -> _EpisodeState:
        state = getattr(self._local, "state", None)
        if state is None:
            raise RuntimeError("agent begin was not called")
        return state

    def _require_configuration(
        self, config: RunConfiguration
    ) -> tuple[OpenAICompatibleModelConfiguration, OpenAICompatibleSamplingSettings]:
        from llm_behavior_ci.runtime.episode import RuntimeUnavailable

        model = config.model
        sampling = config.agent.sampling
        if not isinstance(model, OpenAICompatibleModelConfiguration):
            raise RuntimeUnavailable(
                "OpenAI-compatible agent requires an OpenAI-compatible model configuration"
            )
        if model.provider != self._provider:
            raise RuntimeUnavailable(
                "configuration provider does not match the agent provider"
            )
        if not model.clear_thinking:
            raise RuntimeUnavailable(
                "clear_thinking false requires reasoning replay, which this client "
                "does not send"
            )
        if not isinstance(sampling, OpenAICompatibleSamplingSettings):
            raise RuntimeUnavailable(
                "OpenAI-compatible agent requires OpenAI-compatible sampling"
            )
        return model, sampling

    def _system_text(self) -> str:
        state = self._state()
        prompt = state.config.agent.prompt
        try:
            return render_system_text(
                prompt_version=prompt.prompt_version,
                plan_format_version=prompt.plan_format_version,
                thinking_enabled=prompt.thinking_enabled,
                action_interface=state.config.agent.action_interface,
                mode=self._mode,
            )
        except UnknownPromptVersion as error:
            from llm_behavior_ci.runtime.episode import RuntimeUnavailable

            raise RuntimeUnavailable(str(error)) from error

    def _api_documentation(self) -> str:
        state = self._state()
        agent = state.config.agent
        text = state.context.api_documentation
        source = getattr(state.context, "api_documentation_source", None)
        if (
            agent.prompt.prompt_version == PROMPT_RUNTIME_AUTH_V2
            and source is not None
        ):
            text = render_api_documentation(source, include_constraints=True)
        try:
            return resolve_api_documentation(
                text,
                api_docs_version=agent.api_docs_version,
                api_docs_app=agent.api_docs_app,
            )
        except ApiDocsCorruptionError as error:
            from llm_behavior_ci.runtime.episode import RuntimeUnavailable

            raise RuntimeUnavailable(str(error)) from error

    def messages(self, tool_output: str | None = None) -> list[dict[str, str]]:
        state = self._state()
        built = [
            {"role": "system", "content": self._system_text()},
            {
                "role": "user",
                "content": (
                    f"{state.context.instruction}\n"
                    f"{self._api_documentation()}"
                ),
            },
        ]
        built.extend(state.history)
        if tool_output is not None:
            built.append({"role": "user", "content": tool_output})
        return built

    def endpoint(self) -> str:
        model, _ = self._require_configuration(self._state().config)
        return f"{model.api_base}/chat/completions"

    def completion_payload(
        self, messages: list[dict[str, str]]
    ) -> dict[str, object]:
        """The Chat Completions body for one agent turn.

        Messages keep the system prompt, the task and API documentation,
        and the append-only visible history, in order. Temperature and
        top-p are sent only when ``do_sample`` is true. ``max_tokens`` is
        the mode-specific generation cap. No tools, logprobs, top-k,
        min-p, or seed are sent.
        """

        from llm_behavior_ci.runtime.episode import RuntimeUnavailable

        state = self._state()
        model, sampling = self._require_configuration(state.config)
        if not messages or messages[0].get("role") != "system":
            raise RuntimeUnavailable("system prompt is missing")
        conversation: list[dict[str, str]] = []
        for index, message in enumerate(messages):
            role = message.get("role")
            content = message.get("content")
            allowed = {"system"} if index == 0 else {"user", "assistant"}
            if role not in allowed or not isinstance(content, str):
                raise RuntimeUnavailable(f"{self._label} conversation was malformed")
            conversation.append({"role": role, "content": content})
        if conversation[0]["content"] == "":
            raise RuntimeUnavailable("system prompt is missing")
        if len(conversation) < 2 or conversation[-1]["role"] != "user":
            raise RuntimeUnavailable(
                f"{self._label} requests must end with a user message"
            )
        payload: dict[str, object] = {
            "model": model.model_id,
            "messages": conversation,
        }
        if sampling.do_sample:
            payload["temperature"] = sampling.temperature
            payload["top_p"] = sampling.top_p
        payload["do_sample"] = sampling.do_sample
        payload["max_tokens"] = sampling.generation_max_tokens(self._mode)
        payload["stream"] = False
        payload["thinking"] = {
            "type": model.thinking_type,
            "clear_thinking": model.clear_thinking,
        }
        payload["reasoning_effort"] = model.reasoning_effort
        return payload

    def parse_model_output(
        self, text: str
    ) -> tuple[str | None, str | None, str | None]:
        return parse_model_output(text)

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    def generate_turn(
        self,
        *,
        tool_output: str | None,
        extra_instruction: str | None = None,
        parse_action: bool = True,
    ) -> AgentTurn:
        if self._mode == "plan":
            raise UnsupportedCapability(
                f"{self._label} does not expose prompt-token logprobs; "
                "plan-mode scoring is unsupported"
            )
        state = self._state()
        started_at = wall_now()
        if tool_output is not None:
            state.history.append({"role": "user", "content": tool_output})
        messages = self.messages()
        if extra_instruction is not None:
            messages.append({"role": "user", "content": extra_instruction})
        payload = self.completion_payload(messages)
        began = monotonic()
        raw = self._post(payload)
        latency_seconds = monotonic() - began
        output_text, token_count = _read_chat_completion(raw, self._label)
        if parse_action:
            rejection = None
            try:
                action, app_name, api_name = parse_model_output(output_text)
            except ActionRejected as error:
                action, app_name, api_name = None, None, None
                rejection = str(error)
        else:
            action, app_name, api_name = None, None, None
            rejection = None
        state.history.append({"role": "assistant", "content": output_text})
        return AgentTurn(
            prompt_text=messages[-1]["content"],
            output_text=output_text,
            top_k_logprobs=(),
            generated_token_count=token_count,
            latency_seconds=latency_seconds,
            started_at=started_at,
            action=action,
            app_name=app_name,
            api_name=api_name,
            rejection=rejection,
        )

    def next_turn(self, *, tool_output: str | None) -> AgentTurn:
        return self.generate_turn(
            tool_output=tool_output,
            extra_instruction=None,
            parse_action=True,
        )

    def teacher_force_plan(
        self,
        *,
        messages: list[dict[str, str]],
        plan_text: str,
    ) -> tuple[tuple[TokenLogprob, ...], ...]:
        del messages, plan_text
        raise UnsupportedCapability(
            f"{self._label} does not expose prompt-token logprobs; "
            "teacher-forced plan KL is unsupported"
        )

    def _post(self, payload: dict[str, object]) -> dict[str, object]:
        from llm_behavior_ci.runtime.episode import RuntimeUnavailable

        malformed = f"{self._label} response was malformed"
        body = json.dumps(payload).encode("utf-8")
        if self._api_key.encode("utf-8") in body:
            raise RuntimeUnavailable(f"{self._label} request was malformed")
        endpoint = self.endpoint()
        headers = self._headers()
        last_error: BaseException | None = None
        for attempt in range(_OPENAI_COMPATIBLE_ATTEMPTS):
            request = urllib.request.Request(
                endpoint,
                data=body,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=240) as response:
                    raw = response.read()
                break
            except urllib.error.HTTPError as error:
                last_error = error
                if (
                    error.code in _OPENAI_COMPATIBLE_RETRYABLE
                    and attempt + 1 < _OPENAI_COMPATIBLE_ATTEMPTS
                ):
                    time.sleep(_retry_delay(attempt, error))
                    continue
                raise _openai_compatible_http_error(
                    error,
                    label=self._label,
                    endpoint=endpoint,
                    secret=self._api_key,
                ) from error
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last_error = error
                if attempt + 1 < _OPENAI_COMPATIBLE_ATTEMPTS:
                    time.sleep(_retry_delay(attempt, None))
                    continue
                raise RuntimeUnavailable(
                    _redact_secret(
                        f"{self._label} request failed: {error}", self._api_key
                    )
                ) from error
        else:
            raise RuntimeUnavailable(
                _redact_secret(str(last_error), self._api_key)
            )
        try:
            if isinstance(raw, str):
                decoded = json.loads(raw)
            else:
                decoded = json.loads(bytes(raw).decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeUnavailable(malformed) from error
        if not isinstance(decoded, dict):
            raise RuntimeUnavailable(malformed)
        return decoded
