"""Deterministic, versioned corruption of the API-documentation prompt source.

``TaskContext.api_documentation`` is the exact text AppWorld attaches to a
task and ``SmolagentsVLLMAgent.messages`` folds, unchanged, into the user
turn. The ``api_documentation`` fault kind's whole effect is that one prompt
input: ``AgentConfiguration.api_docs_version``/``api_docs_app`` name a
registered transform and the one app it targets, and
``resolve_api_documentation`` is the single place that transform runs,
so every caller sees the same corrupted text for the same inputs and the
healthy configuration (both fields unset) always sees the source text
unchanged.
"""

from __future__ import annotations

import re

API_DOCS_CORRUPTION_VERSIONS = frozenset({"api-docs-corrupt-v1"})

_APP_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\.")
_REDACTED = ": [documentation removed]"


class ApiDocsCorruptionError(ValueError):
    pass


class UnknownApiDocsVersion(ApiDocsCorruptionError):
    pass


def resolve_api_documentation(
    text: str,
    *,
    api_docs_version: str | None,
    api_docs_app: str | None,
) -> str:
    """Return the API-documentation text an episode's prompt actually carries.

    Both ``None`` is the healthy default and returns ``text`` unchanged.
    Setting only one of the pair is a configuration error the same way
    ``AgentConfiguration`` itself rejects it: both must be set for a
    corruption to apply. An unregistered ``api_docs_version`` is rejected
    rather than silently passed through, the same way an unregistered
    ``prompt_version`` is in ``runtime.prompts``.
    """

    if not isinstance(text, str):
        raise ApiDocsCorruptionError("text must be a string")
    if api_docs_version is None and api_docs_app is None:
        return text
    if api_docs_version is None or api_docs_app is None:
        raise ApiDocsCorruptionError(
            "api_docs_version and api_docs_app must be set together"
        )
    if api_docs_version not in API_DOCS_CORRUPTION_VERSIONS:
        raise UnknownApiDocsVersion(
            f"unknown api_docs_version: {api_docs_version}"
        )
    return _corrupt_v1(text, api_docs_app)


def _corrupt_v1(text: str, app: str) -> str:
    """Redact only ``app``'s documented lines, leaving everything else intact.

    A line belongs to ``app`` when it opens with ``app.`` (the same
    ``app.api`` convention ``tasks.plan_specs`` and ``lifecycle.
    plan_features`` read tool references against), case-insensitively. Its
    text after the first colon is replaced with a fixed redaction marker;
    a line with no colon is left as only its ``app.api`` header. Every
    other line, including the task instruction this text is concatenated
    after in ``SmolagentsVLLMAgent.messages``, is returned byte-for-byte.
    """

    target = app.lower()
    lines = text.splitlines(keepends=True)
    corrupted: list[str] = []
    for line in lines:
        matched = _APP_LINE.match(line)
        if matched is None or matched.group(1).lower() != target:
            corrupted.append(line)
            continue
        newline = "\n" if line.endswith("\n") else ""
        body = line[: len(line) - len(newline)] if newline else line
        colon = body.find(":")
        head = body[:colon] if colon != -1 else body
        corrupted.append(f"{head}{_REDACTED}{newline}")
    return "".join(corrupted)
