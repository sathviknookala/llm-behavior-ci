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
from llm_behavior_ci.runtime.appworld import TaskContext
from llm_behavior_ci.runtime.prompts import (
    UnknownPromptVersion,
    render_system_text,
)

_CALL = re.compile(r"^CALL ([^ \n]+) ([^ \n]+)\n([\s\S]+)$")

_REQUEST_UNSUPPORTED_SERVING_FLAGS = (
    "batch_invariant",
    "enforce_eager",
    "enable_prefix_caching",
    "enable_chunked_prefill",
)

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
    "model.quantization.method",
    "model.vllm_version",
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
        if not isinstance(item, Mapping) or "token_id" not in item:
            raise RuntimeUnavailable("token_id is missing")
        chosen_id = int(item["token_id"])
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
            if not isinstance(alternative, Mapping) or "token_id" not in alternative:
                raise RuntimeUnavailable("token_id is missing")
            token_id = int(alternative["token_id"])
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
    from llm_behavior_ci.runtime.episode import RuntimeUnavailable

    resolve_action_interface(config.agent.action_interface)
    serving = config.model.serving
    for name in _REQUEST_UNSUPPORTED_SERVING_FLAGS:
        if bool(getattr(serving, name)):
            raise RuntimeUnavailable(
                f"serving.{name} cannot be applied via chat completions"
            )
    if serving.tensor_parallel_size != 1:
        raise RuntimeUnavailable(
            "serving.tensor_parallel_size cannot be applied via chat completions"
        )


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


class AppWorldActionExecutor:
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


def bind_appworld_action_executor(
    execute: Callable[[str], object],
) -> AppWorldActionExecutor:
    return AppWorldActionExecutor(execute)


class VLLMAgent:
    def __init__(self, base_url: str) -> None:
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

    def messages(self, tool_output: str | None = None) -> list[dict[str, str]]:
        state = self._state()
        built = [
            {"role": "system", "content": self._system_text()},
            {
                "role": "user",
                "content": (
                    f"{state.context.instruction}\n"
                    f"{state.context.api_documentation}"
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
        state = self._state()
        validate_chat_request(state.config)
        sampling = state.config.agent.sampling
        return {
            "model": state.config.model.model.repository,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
            "seed": sampling.seed,
            "logprobs": True,
            "top_logprobs": state.config.model.serving.max_logprobs,
            "messages": messages,
            "extra_body": {
                "top_k": sampling.top_k,
                "min_p": sampling.min_p,
                "chat_template_kwargs": {
                    "enable_thinking": state.config.agent.prompt.thinking_enabled
                },
            },
        }

    def teacher_force_payload(
        self,
        *,
        messages: list[dict[str, str]],
        plan_text: str,
    ) -> dict[str, object]:
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
            "model": state.config.model.model.repository,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": 0,
            "seed": sampling.seed,
            "logprobs": True,
            "top_logprobs": state.config.model.serving.max_logprobs,
            "messages": forced_messages,
            "extra_body": {
                "top_k": sampling.top_k,
                "min_p": sampling.min_p,
                "prompt_logprobs": state.config.model.serving.max_logprobs,
                "add_generation_prompt": False,
                "chat_template_kwargs": {
                    "enable_thinking": state.config.agent.prompt.thinking_enabled
                },
            },
        }

    def parse_model_output(
        self, text: str
    ) -> tuple[str | None, str | None, str | None]:
        return parse_model_output(text)

    def next_turn(self, *, tool_output: str | None) -> AgentTurn:
        state = self._state()
        started_at = datetime.now(timezone.utc)
        if tool_output is not None:
            state.history.append({"role": "user", "content": tool_output})
        messages = self.messages()
        payload = self.completion_payload(messages)
        began = time.perf_counter()
        raw = self._post(payload)
        latency_seconds = time.perf_counter() - began
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

    def teacher_force_plan(
        self,
        *,
        messages: list[dict[str, str]],
        plan_text: str,
    ) -> tuple[tuple[TokenLogprob, ...], ...]:
        from llm_behavior_ci.runtime.episode import RuntimeUnavailable

        payload = self.teacher_force_payload(messages=messages, plan_text=plan_text)
        raw = self._post(payload)
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeUnavailable(
                "teacher-forced plan logprobs are unavailable from the endpoint"
            )
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise RuntimeUnavailable(
                "teacher-forced plan logprobs are unavailable from the endpoint"
            )
        if choice.get("logprobs") is None and "prompt_logprobs" not in raw:
            raise RuntimeUnavailable(
                "endpoint did not return prompt or echo logprobs for the frozen plan"
            )
        if "prompt_logprobs" in raw:
            prompt_logprobs = raw["prompt_logprobs"]
            if not isinstance(prompt_logprobs, list):
                raise RuntimeUnavailable(
                    "endpoint did not return prompt or echo logprobs for the frozen plan"
                )
            return _parse_prompt_logprobs(prompt_logprobs)
        return parse_logprobs(choice)

    def _post(self, payload: dict[str, object]) -> dict[str, object]:
        from llm_behavior_ci.runtime.episode import RuntimeUnavailable

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}/v1/chat/completions",
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
) -> tuple[tuple[TokenLogprob, ...], ...]:
    from llm_behavior_ci.runtime.episode import RuntimeUnavailable

    positions: list[tuple[TokenLogprob, ...]] = []
    for item in prompt_logprobs:
        if item is None:
            continue
        if not isinstance(item, Mapping):
            raise RuntimeUnavailable(
                "endpoint did not return prompt or echo logprobs for the frozen plan"
            )
        alternatives: list[TokenLogprob] = []
        rank = 0
        for token_id_key, payload in item.items():
            if not isinstance(payload, Mapping):
                raise RuntimeUnavailable(
                    "endpoint did not return prompt or echo logprobs for the frozen plan"
                )
            token_id = int(payload.get("token_id", token_id_key))
            alternatives.append(
                TokenLogprob(
                    token_id=token_id,
                    logprob=float(payload["logprob"]),
                    rank=rank,
                )
            )
            rank += 1
        if alternatives:
            positions.append(tuple(alternatives))
    if not positions:
        raise RuntimeUnavailable(
            "endpoint did not return prompt or echo logprobs for the frozen plan"
        )
    return tuple(positions)
