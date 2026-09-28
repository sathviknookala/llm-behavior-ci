from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Protocol
from urllib.error import URLError

from llm_behavior_ci.config import RunConfiguration
from llm_behavior_ci.records import TokenLogprob
from llm_behavior_ci.runtime.appworld import TaskContext

_CALL = re.compile(r"^CALL ([^ \n]+) ([^ \n]+)\n([\s\S]+)$")


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


class VLLMAgent:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url
        self._context: TaskContext | None = None
        self._config: RunConfiguration | None = None
        self._mode = "execute"
        self._history: list[dict[str, str]] = []

    def set_mode(self, mode: str) -> None:
        self._mode = mode

    def begin(self, context: TaskContext, config: RunConfiguration) -> None:
        self._context = context
        self._config = config
        self._history = []

    def _system_text(self) -> str:
        if self._config is None:
            raise RuntimeError("agent begin was not called")
        prompt = self._config.agent.prompt
        text = (
            f"prompt_version={prompt.prompt_version}\n"
            f"plan_format_version={prompt.plan_format_version}\n"
            f"thinking_enabled={prompt.thinking_enabled}\n"
            f"action_interface={self._config.agent.action_interface}\n"
        )
        if self._mode == "plan":
            return text + "Emit a plan.\n"
        return text + "Emit the next action.\n"

    def messages(self, tool_output: str | None = None) -> list[dict[str, str]]:
        if self._context is None:
            raise RuntimeError("agent begin was not called")
        built = [
            {"role": "system", "content": self._system_text()},
            {
                "role": "user",
                "content": (
                    f"{self._context.instruction}\n"
                    f"{self._context.api_documentation}"
                ),
            },
        ]
        built.extend(self._history)
        if tool_output is not None:
            built.append({"role": "user", "content": tool_output})
        return built

    def completion_payload(
        self, messages: list[dict[str, str]]
    ) -> dict[str, object]:
        if self._config is None:
            raise RuntimeError("agent begin was not called")
        sampling = self._config.agent.sampling
        return {
            "model": self._config.model.model.repository,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
            "seed": sampling.seed,
            "logprobs": True,
            "top_logprobs": self._config.model.serving.max_logprobs,
            "messages": messages,
            "extra_body": {
                "top_k": sampling.top_k,
                "min_p": sampling.min_p,
                "chat_template_kwargs": {
                    "enable_thinking": self._config.agent.prompt.thinking_enabled
                },
            },
        }

    def parse_model_output(
        self, text: str
    ) -> tuple[str | None, str | None, str | None]:
        return parse_model_output(text)

    def next_turn(self, *, tool_output: str | None) -> AgentTurn:
        if self._context is None:
            raise RuntimeError("agent begin was not called")
        started_at = datetime.now(timezone.utc)
        if tool_output is not None:
            self._history.append({"role": "user", "content": tool_output})
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
        self._history.append({"role": "assistant", "content": output_text})
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

    def _post(self, payload: dict[str, object]) -> dict[str, object]:
        from llm_behavior_ci.runtime.episode import RuntimeUnavailable

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url.rstrip('/')}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request) as response:
                return json.loads(response.read().decode("utf-8"))
        except (URLError, TimeoutError, json.JSONDecodeError, OSError) as error:
            raise RuntimeUnavailable(str(error)) from error
