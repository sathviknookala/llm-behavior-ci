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
    "prompt-v2": PromptTemplate(
        version="prompt-v2",
        system_body=(
            "You are an AppWorld tool-using agent.\n"
            "Follow the task instruction and the API documentation.\n"
            "Mutate state only through AppWorld-executed actions.\n"
            "Do not invent APIs that are absent from the documentation.\n"
            "When an app requires authentication, obtain credentials through "
            "the documented AppWorld and supervisor APIs.\n"
            "Do not guess usernames, passwords, access tokens, IDs, or other "
            "credentials.\n"
            "Reuse credential and token values returned by earlier API calls "
            "when a later call requires them.\n"
        ),
    ),
    "prompt-v3": PromptTemplate(
        version="prompt-v3",
        system_body=(
            "You are an AppWorld tool-using agent.\n"
            "Follow the task instruction and the API documentation.\n"
            "Mutate state only through AppWorld-executed actions.\n"
            "Do not invent APIs that are absent from the documentation.\n"
            "Before calling any app API that requires authentication, first obtain "
            "the existing user's credentials using the documented supervisor "
            "credential APIs, then log into that app using exactly the returned "
            "credentials.\n"
            "Never fabricate usernames, passwords, access tokens, IDs, or other "
            "authentication state.\n"
            "Do not create a new account unless the task explicitly requires "
            "account creation.\n"
            "Reuse credentials, tokens, IDs, and other values returned by earlier "
            "API calls whenever later calls require them.\n"
        ),
    ),
    "prompt-v4": PromptTemplate(
        version="prompt-v4",
        system_body=(
            "You are an AppWorld tool-using agent.\n"
            "Follow the task instruction and the API documentation.\n"
            "Mutate state only through AppWorld-executed actions.\n"
            "Do not invent APIs that are absent from the documentation.\n"
            "Before calling any app API that requires authentication, retrieve "
            "the existing user's credentials using the documented supervisor "
            "credential APIs.\n"
            "Use the returned username and password exactly to log into that app.\n"
            "Reuse the returned access token for later authenticated calls.\n"
            "Never fabricate usernames, passwords, tokens, IDs, or "
            "authentication state.\n"
            "Do not call nonexistent or undocumented authentication APIs such "
            "as supervisor.login.\n"
            "Do not create a new account unless the user task explicitly "
            "requests account creation.\n"
            "If an authenticated call fails because authentication is missing "
            "or invalid, return to the credential-retrieval and login flow "
            "rather than continuing with guesses.\n"
            "Reuse credentials, tokens, IDs, and other values returned by "
            "earlier API calls whenever later calls require them.\n"
            "Call complete_task() only as the final action, after the requested "
            "work has actually been performed.\n"
            "Never call complete_task() merely because authentication or "
            "another tool call failed.\n"
            "Before completing, verify from the conversation and tool results "
            "that the requested lookup, mutation, or answer-producing work has "
            "been carried out.\n"
            "If a tool call fails, do not repeat the same failed call unchanged.\n"
            "If a read succeeds, do not repeatedly issue the identical "
            "successful read without using its returned information to advance "
            "the task.\n"
            "Continue to the next unresolved subgoal instead of looping on the "
            "current action.\n"
        ),
    ),
    "prompt-runtime-auth-v1": PromptTemplate(
        version="prompt-runtime-auth-v1",
        system_body=(
            "You are an AppWorld tool-using agent.\n"
            "Follow the task instruction and the API documentation.\n"
            "Mutate state only through AppWorld-executed actions.\n"
            "Do not invent APIs that are absent from the documentation.\n"
            "Authentication and session credentials are managed by the runtime.\n"
            "Use tool results to progress toward the requested task.\n"
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

_LEGACY_CODE_EXECUTE = (
    "Emit the next Python action for AppWorld execution, "
    "or STOP when the task is done.\n"
    "Format a tool call as:\n"
    "CALL <app> <api>\n"
    "<python action>\n"
)

_LEGACY_TOOL_CALLING_EXECUTE = (
    "Emit the next tool call for AppWorld execution, "
    "or STOP when the task is done.\n"
    "Format a tool call as:\n"
    "CALL <app> <api>\n"
    "<arguments>\n"
)

_NATIVE_CODE_EXECUTE = (
    "Emit exactly one AppWorld-native Python API call per turn, "
    "of the form apis.<app>.<api>(...).\n"
    "Do not emit prose, Markdown fences, or any wrapper syntax "
    "around the call.\n"
    "Pass every argument by keyword.\n"
    "Do not repeat a call that just failed.\n"
    "When the task is done, call apis.supervisor.complete_task(...).\n"
)


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
        mode_instruction = _execute_instruction(
            prompt_version=prompt_version,
            action_interface=action_interface,
        )
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


def _execute_instruction(*, prompt_version: str, action_interface: str) -> str:
    if action_interface == "code":
        if prompt_version in {
            "prompt-v2",
            "prompt-v3",
            "prompt-v4",
            "prompt-runtime-auth-v1",
        }:
            return _NATIVE_CODE_EXECUTE
        return _LEGACY_CODE_EXECUTE
    if action_interface == "tool_calling":
        if prompt_version in {"prompt-v2", "prompt-v4", "prompt-runtime-auth-v1"}:
            raise ValueError(
                f"{prompt_version} does not support tool_calling action interface"
            )
        return _LEGACY_TOOL_CALLING_EXECUTE
    raise ValueError(f"unsupported action_interface: {action_interface}")
