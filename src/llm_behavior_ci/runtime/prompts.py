from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    version: str
    system_body: str


@dataclass(frozen=True)
class PlanFormatTemplate:
    version: str
    format_body: str


_PROMPT_REGISTRY: dict[str, PromptTemplate] = {
    "prompt-v1": PromptTemplate(
        version="prompt-v1",
        system_body=(
            "You are an AppWorld tool-using agent.\n"
            "Follow the task instruction and the API documentation.\n"
            "Mutate state only through AppWorld-executed actions.\n"
            "Do not invent APIs that are absent from the documentation.\n"
        ),
    ),
    "prompt-no-api-guidance": PromptTemplate(
        version="prompt-no-api-guidance",
        system_body=(
            "You are an AppWorld tool-using agent.\n"
            "Follow the task instruction.\n"
            "Mutate state only through AppWorld-executed actions.\n"
        ),
    ),
}

_PLAN_FORMAT_REGISTRY: dict[str, PlanFormatTemplate] = {
    "plan-v1": PlanFormatTemplate(
        version="plan-v1",
        format_body=(
            "Emit a numbered plan before acting.\n"
            "Each step names the app, the API, and the intended effect.\n"
            "Do not execute tools while planning.\n"
        ),
    ),
}


class UnknownPromptVersion(ValueError):
    pass


def registered_prompt_versions() -> frozenset[str]:
    return frozenset(_PROMPT_REGISTRY)


def registered_plan_format_versions() -> frozenset[str]:
    return frozenset(_PLAN_FORMAT_REGISTRY)


def resolve_prompt_template(prompt_version: str) -> PromptTemplate:
    try:
        return _PROMPT_REGISTRY[prompt_version]
    except KeyError as error:
        raise UnknownPromptVersion(
            f"unknown prompt_version: {prompt_version}"
        ) from error


def resolve_plan_format_template(plan_format_version: str) -> PlanFormatTemplate:
    try:
        return _PLAN_FORMAT_REGISTRY[plan_format_version]
    except KeyError as error:
        raise UnknownPromptVersion(
            f"unknown plan_format_version: {plan_format_version}"
        ) from error


def render_system_text(
    *,
    prompt_version: str,
    plan_format_version: str,
    thinking_enabled: bool,
    action_interface: str,
    mode: str,
) -> str:
    prompt = resolve_prompt_template(prompt_version)
    plan_format = resolve_plan_format_template(plan_format_version)
    if mode == "plan":
        mode_instruction = (
            f"{plan_format.format_body}"
            "Respond with the plan text only.\n"
        )
    elif mode == "execute":
        if action_interface == "code":
            mode_instruction = (
                "Emit the next Python action for AppWorld execution, "
                "or STOP when the task is done.\n"
                "Format a tool call as:\n"
                "CALL <app> <api>\n"
                "<python action>\n"
            )
        elif action_interface == "tool_calling":
            mode_instruction = (
                "Emit the next tool call for AppWorld execution, "
                "or STOP when the task is done.\n"
                "Format a tool call as:\n"
                "CALL <app> <api>\n"
                "<arguments>\n"
            )
        else:
            raise ValueError(f"unsupported action_interface: {action_interface}")
    else:
        raise ValueError(f"unsupported agent mode: {mode}")
    thinking = "enabled" if thinking_enabled else "disabled"
    return (
        f"{prompt.system_body}"
        f"Prompt registry id: {prompt.version}\n"
        f"Plan format registry id: {plan_format.version}\n"
        f"Thinking tokens: {thinking}\n"
        f"Action interface: {action_interface}\n"
        f"{mode_instruction}"
    )
