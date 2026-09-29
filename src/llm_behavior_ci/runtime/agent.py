from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Mapping, Protocol

from llm_behavior_ci.config import ACTION_INTERFACES, RunConfiguration
from llm_behavior_ci.records import TokenLogprob
from llm_behavior_ci.runtime.api_docs import (
    ApiDocsCorruptionError,
    resolve_api_documentation,
)
from llm_behavior_ci.runtime.appworld import TaskContext
from llm_behavior_ci.runtime.prompts import (
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

_CALL = re.compile(r"^CALL ([^ \n]+) ([^ \n]+)\n([\s\S]+)$")

_TOKEN_ID_PREFIX = "token_id:"

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
    latency_seconds: float
    started_at: datetime
    action: str | None
    app_name: str | None
    api_name: str | None


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


def parse_model_output(text: str) -> tuple[str | None, str | None, str | None]:
    if text == "STOP" or text.startswith("STOP\n"):
        return None, None, None
    matched = _CALL.match(text)
    if matched is not None:
        return matched.group(3), matched.group(1), matched.group(2)
    return text, None, None


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
        try:
            return resolve_api_documentation(
                state.context.api_documentation,
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
        ``return_tokens_as_token_ids``) sit at the top level; vLLM ignores a
        literal ``extra_body`` key.
        """

        state = self._state()
        validate_chat_request(state.config)
        sampling = state.config.agent.sampling
        return {
            "model": served_model_id(state.config),
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
            "seed": sampling.seed,
            "logprobs": True,
            "top_logprobs": state.config.model.serving.max_logprobs,
            "messages": messages,
            "top_k": sampling.top_k,
            "min_p": sampling.min_p,
            "return_tokens_as_token_ids": True,
            "chat_template_kwargs": {
                "enable_thinking": state.config.agent.prompt.thinking_enabled
            },
        }

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

    def next_turn(self, *, tool_output: str | None) -> AgentTurn:
        state = self._state()
        started_at = datetime.now(timezone.utc)
        if tool_output is not None:
            state.history.append({"role": "user", "content": tool_output})
        messages = self.messages()
        began = time.perf_counter()
        chat_message = self.generate(messages)
        latency_seconds = time.perf_counter() - began
        output_text = chat_message.content or ""
        raw = chat_message.raw
        choice = raw["choices"][0]
        logprobs = parse_logprobs(choice)
        action, app_name, api_name = parse_model_output(output_text)
        state.history.append({"role": "assistant", "content": output_text})
        return AgentTurn(
            prompt_text=messages[-1]["content"],
            output_text=output_text,
            top_k_logprobs=logprobs,
            latency_seconds=latency_seconds,
            started_at=started_at,
            action=action,
            app_name=app_name,
            api_name=api_name,
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
