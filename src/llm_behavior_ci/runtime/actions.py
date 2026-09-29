from __future__ import annotations

import ast
import re

_CALL = re.compile(r"^CALL ([^ \n]+) ([^ \n]+)\n([\s\S]+)$")


class ActionRejected(ValueError):
    pass


def parse_model_output(text: str) -> tuple[str | None, str | None, str | None]:
    if text == "STOP" or text.startswith("STOP\n"):
        return None, None, None
    matched = _CALL.match(text)
    if matched is not None:
        return matched.group(3), matched.group(1), matched.group(2)
    return _parse_native_call(text)


def _parse_native_call(text: str) -> tuple[str, str, str]:
    stripped = text.strip()
    if stripped == "":
        raise ActionRejected("empty action")
    try:
        tree = ast.parse(stripped, mode="exec")
    except SyntaxError as error:
        raise ActionRejected("action is not valid Python") from error
    if len(tree.body) != 1:
        raise ActionRejected("action must be a single expression")
    statement = tree.body[0]
    if not isinstance(statement, ast.Expr):
        raise ActionRejected("action must be a single call expression")
    if not isinstance(statement.value, ast.Call):
        raise ActionRejected("action must be a single call expression")
    call_count = sum(isinstance(node, ast.Call) for node in ast.walk(tree))
    if call_count != 1:
        raise ActionRejected("action must contain exactly one call")
    app_name, api_name = _apis_target(statement.value.func)
    return stripped, app_name, api_name


def _apis_target(func: ast.AST) -> tuple[str, str]:
    if not isinstance(func, ast.Attribute):
        raise ActionRejected("action must call apis.<app>.<api>")
    api_name = func.attr
    inner = func.value
    if not isinstance(inner, ast.Attribute):
        raise ActionRejected("action must call apis.<app>.<api>")
    app_name = inner.attr
    root = inner.value
    if not isinstance(root, ast.Name) or root.id != "apis":
        raise ActionRejected("action must call apis.<app>.<api>")
    if not app_name.isidentifier() or not api_name.isidentifier():
        raise ActionRejected("app and api names must be identifiers")
    return app_name, api_name
