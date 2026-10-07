from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable, NoReturn, Protocol


@dataclass(frozen=True)
class TaskContext:
    task_id: str
    instruction: str
    api_documentation: str
    api_documentation_source: object | None = None


@dataclass(frozen=True)
class ToolResult:
    output_text: str | None
    error_message: str | None
    recoverable: bool
    app_name: str | None
    api_name: str | None


@dataclass(frozen=True)
class EvaluationResult:
    success: bool
    passed_requirements: int
    total_requirements: int
    difficulty: int | None


class AppWorldSession(Protocol):
    def prepare(self) -> None: ...

    def context(self) -> TaskContext: ...

    def execute(self, action: str) -> ToolResult: ...

    def evaluate(self) -> EvaluationResult: ...

    def close(self) -> None: ...


def _open_appworld(task_id: str) -> object:
    try:
        from appworld import AppWorld
    except ImportError as error:
        from llm_behavior_ci.runtime.episode import RuntimeUnavailable

        raise RuntimeUnavailable("AppWorld is not installed") from error
    return AppWorld(task_id=task_id)


def _is_sequence(value: object) -> bool:
    return isinstance(value, (list, tuple))


def _constraint_suffix(item: Mapping[str, object]) -> str:
    raw = item.get("constraints")
    if not isinstance(raw, list):
        return ""
    phrases = [
        " ".join(entry.split())
        for entry in raw
        if isinstance(entry, str) and entry.strip() != ""
    ]
    if not phrases:
        return ""
    return "  # " + "; ".join(phrases)


def _parameter_text(parameters: object, *, include_constraints: bool = False) -> str:
    parts: list[str] = []
    if isinstance(parameters, Mapping):
        required = parameters.get("required", [])
        optional = parameters.get("optional", [])
        if isinstance(required, list):
            parts.extend(
                name for name in required if isinstance(name, str) and name != ""
            )
        if isinstance(optional, list):
            parts.extend(
                f"{name}?"
                for name in optional
                if isinstance(name, str) and name != ""
            )
        return ", ".join(parts)
    if not isinstance(parameters, list):
        return ""
    for item in parameters:
        if not isinstance(item, Mapping):
            continue
        name = item.get("name")
        if not isinstance(name, str) or name == "":
            continue
        type_name = item.get("type")
        piece = name
        if isinstance(type_name, str) and type_name != "":
            piece = f"{name}:{type_name}"
        if not item.get("required"):
            piece += "?"
        if include_constraints:
            piece += _constraint_suffix(item)
        parts.append(piece)
    return ", ".join(parts)


def _api_line(
    app_name: str,
    api_name: str,
    doc: object,
    *,
    include_constraints: bool = False,
) -> str:
    description = ""
    parameters: object = []
    if isinstance(doc, Mapping):
        raw_description = doc.get("description", "")
        if isinstance(raw_description, str):
            description = " ".join(raw_description.split())
        parameters = doc.get("parameters", [])
    parameter_text = _parameter_text(
        parameters,
        include_constraints=include_constraints,
    )
    if parameter_text:
        return f"{app_name}.{api_name}: {description} | {parameter_text}"
    return f"{app_name}.{api_name}: {description}"


def render_api_documentation(
    documentation: object,
    *,
    include_constraints: bool = False,
) -> str:
    """Turn AppWorld's API-doc collection into line-oriented prompt text.

    A string is kept unchanged so fakes and already-rendered text stay
    stable. A mapping of app to API docs becomes one ``app.api:`` line per
    API, apps and APIs sorted, with the description and parameter name and
    type. That is the form ``api-docs-corrupt-v1`` can redact, and it drops
    response schemas. Anything else falls back to ``str``.

    ``include_constraints`` appends each parameter's stored ``constraints``
    list as a comment. The default omits those comments, which is the
    rendering ``prompt-runtime-auth-v1`` still shows. Constraints are copied
    from the metadata; a parameter with none gets no comment.
    """

    if documentation is None:
        return ""
    if isinstance(documentation, str):
        return documentation
    if not isinstance(documentation, Mapping):
        return str(documentation)
    if not documentation:
        return ""
    lines: list[str] = []
    for app_name in sorted(documentation, key=str):
        apis = documentation[app_name]
        if not isinstance(apis, Mapping):
            return str(documentation)
        for api_name in sorted(apis, key=str):
            lines.append(
                _api_line(
                    str(app_name),
                    str(api_name),
                    apis[api_name],
                    include_constraints=include_constraints,
                )
            )
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


_EXECUTION_FAILED = "Execution failed."
_NO_CODE = "No code available to execute."


def _code_for_execute(action: str) -> str:
    """Print a single call so AppWorld's stdout capture keeps the return value.

    AppWorld records stdout and substitutes ``Execution successful.`` when
    that stream is empty. A bare ``apis.<app>.<api>(...)`` expression
    therefore drops the response body, including tokens and records the
    next turn needs. ``print`` of one dict or list is AppWorld's own
    JSON printer. Anything that is not a single call is left unchanged.
    """

    stripped = action.strip()
    try:
        tree = ast.parse(stripped, mode="exec")
    except SyntaxError:
        return action
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        return action
    value = tree.body[0].value
    if not isinstance(value, ast.Call):
        return action
    func = value.func
    if isinstance(func, ast.Name) and func.id == "print":
        return action
    return f"print({stripped})"


def _execute_output_is_error(value: str) -> bool:
    stripped = value.lstrip()
    return stripped.startswith(_EXECUTION_FAILED) or stripped.startswith(_NO_CODE)


_SPOTIFY_CAPABILITY_PROFILE = "spotify_capability_v1"
_SPOTIFY_AUTHENTICATED_PROFILE = "spotify_authenticated_v1"
_HIDDEN_SPOTIFY_APIS = frozenset({"login", "signup"})
_HIDDEN_SUPERVISOR_APIS = frozenset({"show_profile", "show_account_passwords"})
_HIDDEN_RENDERED = frozenset(
    {
        "spotify.login",
        "spotify.signup",
        "supervisor.show_profile",
        "supervisor.show_account_passwords",
    }
)
_ACCESS_TOKEN = "access_token"
_SETUP_FAILED = "spotify authentication setup failed"

_SPOTIFY_SUPERVISOR_APIS = frozenset(
    {
        "show_profile",
        "show_account_passwords",
        "complete_task",
    }
)

_SPOTIFY_API_DOC_APIS = frozenset(
    {
        "show_api_descriptions",
        "show_api_doc",
    }
)


def _call_target(action: str) -> tuple[str, str]:
    """Return ``(app, api)`` for one ``apis.<app>.<api>(...)`` call."""

    stripped = action.strip()
    tree = ast.parse(stripped, mode="exec")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        raise ValueError("action must be one call")
    call = tree.body[0].value
    if not isinstance(call, ast.Call):
        raise ValueError("action must be one call")
    func = call.func
    if not isinstance(func, ast.Attribute):
        raise ValueError("action must target apis.<app>.<api>")
    api_name = func.attr
    app_node = func.value
    if (
        not isinstance(app_node, ast.Attribute)
        or not isinstance(app_node.value, ast.Name)
        or app_node.value.id != "apis"
    ):
        raise ValueError("action must target apis.<app>.<api>")
    return app_node.attr, api_name


def _keyword_string(action: str, name: str) -> str | None:
    """Return one string keyword from a call, or None when it is absent."""

    tree = ast.parse(action.strip(), mode="exec")
    call = tree.body[0].value
    if not isinstance(call, ast.Call):
        return None
    for keyword in call.keywords:
        if keyword.arg != name:
            continue
        if isinstance(keyword.value, ast.Constant) and isinstance(
            keyword.value.value, str
        ):
            return keyword.value.value
    return None


def _spotify_capability_allows(action: str) -> bool:
    """Whether one action stays inside the Spotify capability profile.

    Spotify APIs are allowed. Supervisor is limited to profile, account
    passwords, and task completion. ApiDocs is limited to the two lookup
    helpers, and only when the requested app is Spotify, so those helpers
    cannot browse an unrelated app.
    """

    try:
        app_name, api_name = _call_target(action)
    except (SyntaxError, ValueError):
        return False
    if app_name == "spotify":
        return True
    if app_name == "supervisor":
        return api_name in _SPOTIFY_SUPERVISOR_APIS
    if app_name == "api_docs":
        if api_name not in _SPOTIFY_API_DOC_APIS:
            return False
        return _keyword_string(action, "app_name") == "spotify"
    return False


def _spotify_capability_docs(documentation: object) -> object:
    """Keep Spotify and the approved supervisor and api_docs helpers."""

    if not isinstance(documentation, Mapping):
        return documentation
    selected: dict[str, object] = {}
    if "spotify" in documentation:
        selected["spotify"] = documentation["spotify"]
    supervisor = documentation.get("supervisor")
    if isinstance(supervisor, Mapping):
        selected["supervisor"] = {
            name: doc
            for name, doc in supervisor.items()
            if name in _SPOTIFY_SUPERVISOR_APIS
        }
    api_docs = documentation.get("api_docs")
    if isinstance(api_docs, Mapping):
        selected["api_docs"] = {
            name: doc
            for name, doc in api_docs.items()
            if name in _SPOTIFY_API_DOC_APIS
        }
    return selected


def _raise_setup(error: Exception | None = None) -> NoReturn:
    from llm_behavior_ci.runtime.episode import RuntimeUnavailable

    if error is None:
        raise RuntimeUnavailable(_SETUP_FAILED)
    raise RuntimeUnavailable(_SETUP_FAILED) from error


def _is_access_token_parameter(item: object) -> bool:
    if isinstance(item, str):
        return item == _ACCESS_TOKEN
    if isinstance(item, Mapping):
        return item.get("name") == _ACCESS_TOKEN
    return False


def _without_access_token_parameters(parameters: object) -> object:
    if isinstance(parameters, list):
        return [item for item in parameters if not _is_access_token_parameter(item)]
    if isinstance(parameters, Mapping):
        cleaned: dict[str, object] = {}
        for key, value in parameters.items():
            if isinstance(value, list):
                cleaned[str(key)] = [
                    item for item in value if not _is_access_token_parameter(item)
                ]
            else:
                cleaned[str(key)] = value
        return cleaned
    return parameters


def _without_access_token_schema(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: item
            for key, item in value.items()
            if key != _ACCESS_TOKEN
        }
    if isinstance(value, list):
        return [item for item in value if not _is_access_token_parameter(item)]
    return value


def _without_access_token_doc(doc: object) -> object:
    if not isinstance(doc, Mapping):
        return doc
    updated = dict(doc)
    if "parameters" in updated:
        updated["parameters"] = _without_access_token_parameters(updated["parameters"])
    schemas = updated.get("response_schemas")
    if isinstance(schemas, Mapping):
        updated["response_schemas"] = {
            key: _without_access_token_schema(value) for key, value in schemas.items()
        }
    return updated


def _doc_requires_access_token(doc: object) -> bool:
    if not isinstance(doc, Mapping):
        return False
    parameters = doc.get("parameters")
    if isinstance(parameters, list):
        return any(_is_access_token_parameter(item) for item in parameters)
    if isinstance(parameters, Mapping):
        for value in parameters.values():
            if isinstance(value, list) and any(
                _is_access_token_parameter(item) for item in value
            ):
                return True
    return False


def _spotify_token_apis(documentation: object) -> frozenset[str]:
    if not isinstance(documentation, Mapping):
        return frozenset()
    spotify = documentation.get("spotify")
    if not isinstance(spotify, Mapping):
        return frozenset()
    return frozenset(
        str(api_name)
        for api_name, doc in spotify.items()
        if _doc_requires_access_token(doc)
    )


def _strip_access_token_parameter_text(line: str) -> str:
    if " | " not in line:
        return line
    head, parameters = line.split(" | ", 1)
    kept: list[str] = []
    for piece in parameters.split(","):
        stripped = piece.strip()
        name = stripped.split(":", 1)[0].strip().rstrip("?")
        if name == _ACCESS_TOKEN:
            continue
        kept.append(stripped)
    if not kept:
        return head
    return f"{head} | {', '.join(kept)}"


def _sanitize_rendered_documentation(text: str) -> str:
    kept: list[str] = []
    for line in text.splitlines():
        name = line.split(":", 1)[0].strip()
        if name in _HIDDEN_RENDERED:
            continue
        if name.startswith("spotify."):
            line = _strip_access_token_parameter_text(line)
        kept.append(line)
    if not kept:
        return ""
    rendered = "\n".join(kept)
    if text.endswith("\n"):
        rendered += "\n"
    return rendered


def _plain_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    return value


def _plain_documentation(documentation: Mapping[str, object]) -> dict[str, object]:
    plain: dict[str, object] = {}
    for app_name, apis in documentation.items():
        if isinstance(apis, Mapping):
            plain[str(app_name)] = {
                str(api_name): _plain_value(doc) for api_name, doc in apis.items()
            }
        else:
            plain[str(app_name)] = apis
    return plain


def _authenticated_surface(documentation: object) -> object:
    """Drop credential APIs and Spotify access-token parameters."""

    if isinstance(documentation, str):
        return _sanitize_rendered_documentation(documentation)
    if not isinstance(documentation, Mapping):
        return documentation
    selected = _plain_documentation(documentation)
    spotify = selected.get("spotify")
    if isinstance(spotify, Mapping):
        selected["spotify"] = {
            name: _without_access_token_doc(doc)
            for name, doc in spotify.items()
            if name not in _HIDDEN_SPOTIFY_APIS
        }
    supervisor = selected.get("supervisor")
    if isinstance(supervisor, Mapping):
        selected["supervisor"] = {
            name: doc
            for name, doc in supervisor.items()
            if name not in _HIDDEN_SUPERVISOR_APIS
        }
    return selected


def _authenticated_surface_allows(action: str) -> bool:
    try:
        app_name, api_name = _call_target(action)
    except (SyntaxError, ValueError):
        return True
    if app_name == "spotify" and api_name in _HIDDEN_SPOTIFY_APIS:
        return False
    if app_name == "supervisor" and api_name in _HIDDEN_SUPERVISOR_APIS:
        return False
    if app_name != "api_docs":
        return True
    requested_app = _keyword_string(action, "app_name")
    requested_api = _keyword_string(action, "api_name")
    if requested_app == "spotify" and requested_api in _HIDDEN_SPOTIFY_APIS:
        return False
    if requested_app == "supervisor" and requested_api in _HIDDEN_SUPERVISOR_APIS:
        return False
    return True


def _profile_email(profile: object) -> str:
    if not isinstance(profile, Mapping):
        _raise_setup()
    email = profile.get("email")
    if not isinstance(email, str) or email.strip() == "":
        _raise_setup()
    return email


def _spotify_account_password(accounts: object) -> str:
    if not isinstance(accounts, list):
        _raise_setup()
    matches: list[str] = []
    for item in accounts:
        if not isinstance(item, Mapping):
            continue
        name = item.get("account_name")
        password = item.get("password")
        if not isinstance(name, str) or name.lower() != "spotify":
            continue
        if not isinstance(password, str) or password == "":
            _raise_setup()
        matches.append(password)
    if len(matches) != 1:
        _raise_setup()
    return matches[0]


def _spotify_access_token(payload: object) -> str:
    if not isinstance(payload, Mapping):
        _raise_setup()
    token = payload.get(_ACCESS_TOKEN)
    if not isinstance(token, str) or token == "":
        _raise_setup()
    return token


def _rewrite_access_token(action: str, token: str | None) -> str:
    try:
        tree = ast.parse(action.strip(), mode="exec")
        call = tree.body[0].value
        if not isinstance(call, ast.Call):
            raise ValueError("not a call")
        keywords = [
            keyword for keyword in call.keywords if keyword.arg != _ACCESS_TOKEN
        ]
        if token is not None:
            keywords.append(
                ast.keyword(arg=_ACCESS_TOKEN, value=ast.Constant(token))
            )
        call.keywords = keywords
        return ast.unparse(call)
    except (SyntaxError, ValueError, TypeError) as error:
        _raise_setup(error)


def _call_supplies_access_token(action: str) -> bool:
    try:
        tree = ast.parse(action.strip(), mode="exec")
        call = tree.body[0].value
    except (SyntaxError, ValueError, IndexError, AttributeError):
        return False
    if not isinstance(call, ast.Call):
        return False
    return any(keyword.arg == _ACCESS_TOKEN for keyword in call.keywords)


def _filter_api_descriptions(payload: object, app_name: str | None) -> object:
    hidden = {
        "spotify": _HIDDEN_SPOTIFY_APIS,
        "supervisor": _HIDDEN_SUPERVISOR_APIS,
    }.get(app_name or "", frozenset())
    if not isinstance(payload, list) or not hidden:
        return payload
    return [
        item
        for item in payload
        if not (isinstance(item, Mapping) and item.get("name") in hidden)
    ]


def _json_text(payload: object, original: str) -> str:
    encoded = json.dumps(payload, ensure_ascii=False)
    indent = 1 if len(encoded) >= 100 else None
    text = json.dumps(payload, indent=indent, ensure_ascii=False)
    if original.endswith("\n"):
        text += "\n"
    return text


def _sanitize_helper_output(action: str, api_name: str, text: str) -> str:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text
    if api_name == "show_api_descriptions":
        payload = _filter_api_descriptions(payload, _keyword_string(action, "app_name"))
    elif api_name == "show_api_doc":
        payload = _without_access_token_doc(payload)
    return _json_text(payload, text)


class LiveAppWorldSession:
    """AppWorld session adapter that imports AppWorld only when a world is opened.

    A second session for the same task does not open its world while
    another live world is still open. AppWorld starts nested time
    freezers that cannot be stopped safely in that state. The deferred
    session serves ``context`` from the open world, then opens its own
    world after that one has closed.

    ``context`` reads the task instruction and renders API docs without
    executing. ``execute`` prints a single call expression before handing
    it to AppWorld, so the return value is on stdout instead of being
    replaced by ``Execution successful.``. It treats AppWorld's returned
    ``Execution failed.`` text as a recoverable tool error; a raised
    exception is the same.
    ``evaluate`` reads ``pass_count`` and ``num_tests`` from the
    ``TestTracker``. ``close`` is idempotent. The live world has no
    ``initial_state_identity`` method.

    ``tool_access_profile`` ``spotify_capability_v1`` renders only the
    Spotify app plus the approved supervisor and api_docs helpers, and
    ``execute`` rejects every other API before AppWorld runs it. Any
    other value, including unset, leaves the existing surface unchanged.

    ``appworld_setup_profile`` ``spotify_authenticated_v1`` makes
    ``prepare`` log into the existing Spotify account and keep the access
    token on the session. Later Spotify calls that require ``access_token``
    receive that token inside AppWorld. The stored agent action is the
    call the model submitted. Model-visible documentation, including
    later api_docs helper responses, omits login, signup, supervisor
    credential retrieval, and the access-token parameter. ``prepare`` is
    idempotent: a second call does not log in again. No profile is a no-op.
    """

    _open_stack: list[LiveAppWorldSession] = []

    def __init__(
        self,
        task_id: str,
        *,
        opener: Callable[[str], object] | None = None,
        tool_access_profile: str | None = None,
        appworld_setup_profile: str | None = None,
    ) -> None:
        self._closed = False
        self._task_id = task_id
        self._opener = opener or _open_appworld
        self._tool_access_profile = tool_access_profile
        self._setup_profile = appworld_setup_profile
        self._prepared = False
        self._access_token: str | None = None
        self._token_api_names: frozenset[str] | None = None
        self._world: object | None = None
        if any(
            session._world is not None and not session._closed
            for session in type(self)._open_stack
        ):
            return
        self._open_world()

    def _open_world(self) -> None:
        try:
            self._world = self._opener(self._task_id)
        except ImportError as error:
            from llm_behavior_ci.runtime.episode import RuntimeUnavailable

            raise RuntimeUnavailable("AppWorld is not installed") from error
        type(self)._open_stack.append(self)

    def required_apps(self) -> tuple[str, ...] | None:
        """Ground-truth required apps, or ``None`` when the world does not expose them.

        AppWorld's default minimal ground-truth mode leaves ``required_apps``
        unset, so ``None`` means unknown. An explicitly empty list stays an
        empty tuple.
        """

        if self._world is None:
            self._open_world()
        ground_truth = getattr(self._world.task, "ground_truth", None)
        if ground_truth is None:
            return None
        apps = getattr(ground_truth, "required_apps", None)
        if apps is None:
            return None
        return tuple(str(app) for app in apps)

    def complete_without_work(self) -> None:
        from llm_behavior_ci.runtime.episode import RuntimeUnavailable

        result = self.execute("apis.supervisor.complete_task()")
        if result.error_message is not None:
            raise RuntimeUnavailable("do-nothing completion failed")

    def prepare(self) -> None:
        """Authenticate the world before the agent starts, when a profile says so.

        ``spotify_authenticated_v1`` reads the supervisor profile and account
        passwords, selects the existing Spotify account, and logs in with
        those returned values. The access token stays on this session.
        Credential APIs and login are not model turns. A second call after
        success does not log in again. Any failed step raises
        ``RuntimeUnavailable`` and leaves the token unset.
        """

        if self._prepared:
            return
        if self._setup_profile is None:
            self._prepared = True
            return
        if self._setup_profile != _SPOTIFY_AUTHENTICATED_PROFILE:
            _raise_setup()
        if self._world is None:
            self._open_world()
        try:
            documentation = getattr(self._world.task, "api_docs", "")
            token_apis = _spotify_token_apis(documentation)
            if not token_apis:
                _raise_setup()
            profile = self._call_setup_api("supervisor", "show_profile")
            passwords = self._call_setup_api("supervisor", "show_account_passwords")
            email = _profile_email(profile)
            password = _spotify_account_password(passwords)
            login = self._call_setup_api(
                "spotify",
                "login",
                username=email,
                password=password,
            )
            token = _spotify_access_token(login)
        except Exception as error:
            from llm_behavior_ci.runtime.episode import RuntimeUnavailable

            if isinstance(error, RuntimeUnavailable):
                raise
            _raise_setup(error)
        self._token_api_names = token_apis
        self._access_token = token
        self._prepared = True

    def _call_setup_api(self, app_name: str, api_name: str, **kwargs: object) -> object:
        if self._world is None:
            self._open_world()
        requester = getattr(self._world, "requester", None)
        request = getattr(requester, "request", None)
        if not callable(request):
            _raise_setup()
        try:
            return request(
                _app_name=app_name,
                _api_name=api_name,
                track=False,
                **kwargs,
            )
        except Exception as error:
            from llm_behavior_ci.runtime.episode import RuntimeUnavailable

            if isinstance(error, RuntimeUnavailable):
                raise
            _raise_setup(error)

    def _visible_documentation(self, raw_docs: object) -> object:
        if self._tool_access_profile == _SPOTIFY_CAPABILITY_PROFILE:
            raw_docs = _spotify_capability_docs(raw_docs)
        if self._setup_profile == _SPOTIFY_AUTHENTICATED_PROFILE:
            raw_docs = _authenticated_surface(raw_docs)
        return raw_docs

    def _allows(self, action: str) -> bool:
        if self._tool_access_profile == _SPOTIFY_CAPABILITY_PROFILE:
            if not _spotify_capability_allows(action):
                return False
        if self._setup_profile == _SPOTIFY_AUTHENTICATED_PROFILE:
            if not _authenticated_surface_allows(action):
                return False
        return True

    def _token_apis(self) -> frozenset[str]:
        if self._token_api_names is not None:
            return self._token_api_names
        if self._world is None:
            return frozenset()
        return _spotify_token_apis(getattr(self._world.task, "api_docs", ""))

    def _with_runtime_auth(self, action: str) -> str:
        if self._setup_profile != _SPOTIFY_AUTHENTICATED_PROFILE:
            return action
        try:
            app_name, api_name = _call_target(action)
        except (SyntaxError, ValueError):
            return action
        if app_name != "spotify":
            return action
        requires_token = api_name in self._token_apis()
        supplied = _call_supplies_access_token(action)
        if not requires_token:
            if supplied:
                return _rewrite_access_token(action, None)
            return action
        if not self._access_token:
            _raise_setup()
        return _rewrite_access_token(action, self._access_token)

    def _redact(self, text: str | None) -> str | None:
        token = self._access_token
        if text is None or token is None or token not in text:
            return text
        return text.replace(token, "")

    def _visible_result(self, action: str, result: ToolResult) -> ToolResult:
        output_text = result.output_text
        error_message = result.error_message
        if (
            self._setup_profile == _SPOTIFY_AUTHENTICATED_PROFILE
            and output_text is not None
            and error_message is None
        ):
            try:
                app_name, api_name = _call_target(action)
            except (SyntaxError, ValueError):
                app_name, api_name = None, None
            if app_name == "api_docs" and api_name in _SPOTIFY_API_DOC_APIS:
                output_text = _sanitize_helper_output(action, api_name, output_text)
        output_text = self._redact(output_text)
        error_message = self._redact(error_message)
        if output_text == result.output_text and error_message == result.error_message:
            return result
        return ToolResult(
            output_text=output_text,
            error_message=error_message,
            recoverable=result.recoverable,
            app_name=result.app_name,
            api_name=result.api_name,
        )

    def context(self) -> TaskContext:
        if self._world is None:
            for session in reversed(type(self)._open_stack):
                if (
                    session._task_id == self._task_id
                    and session._world is not None
                    and not session._closed
                ):
                    return session.context()
            self._open_world()
        task = self._world.task
        raw_docs = self._visible_documentation(getattr(task, "api_docs", ""))
        source = raw_docs if isinstance(raw_docs, Mapping) else None
        return TaskContext(
            task_id=self._task_id,
            instruction=task.instruction,
            api_documentation=render_api_documentation(raw_docs),
            api_documentation_source=source,
        )

    def execute(self, action: str) -> ToolResult:
        if not self._allows(action):
            return ToolResult(
                output_text=None,
                error_message="API is outside the configured capability profile",
                recoverable=True,
                app_name=None,
                api_name=None,
            )
        submitted = self._with_runtime_auth(action)
        if self._world is None:
            self._open_world()
        try:
            value = self._world.execute(_code_for_execute(submitted))
        except Exception as error:
            return self._visible_result(
                action,
                ToolResult(
                    output_text=None,
                    error_message=str(error),
                    recoverable=True,
                    app_name=None,
                    api_name=None,
                ),
            )
        text = value if isinstance(value, str) else str(value)
        if _execute_output_is_error(text):
            return self._visible_result(
                action,
                ToolResult(
                    output_text=None,
                    error_message=text,
                    recoverable=True,
                    app_name=None,
                    api_name=None,
                ),
            )
        return self._visible_result(
            action,
            ToolResult(
                output_text=text,
                error_message=None,
                recoverable=False,
                app_name=None,
                api_name=None,
            ),
        )

    def evaluate(self) -> EvaluationResult:
        from llm_behavior_ci.runtime.episode import RuntimeUnavailable

        if self._world is None:
            self._open_world()
        raw = self._world.evaluate()
        success = bool(raw.success)
        if hasattr(raw, "pass_count") and hasattr(raw, "num_tests"):
            passed = int(raw.pass_count)
            total = int(raw.num_tests)
        else:
            passes = getattr(raw, "passes", None)
            failures = getattr(raw, "failures", None)
            if _is_sequence(passes) and _is_sequence(failures):
                passed = len(passes)
                total = passed + len(failures)
            else:
                raise RuntimeUnavailable("evaluation result is missing counts")
        difficulty = getattr(raw, "difficulty", None)
        return EvaluationResult(
            success=success,
            passed_requirements=passed,
            total_requirements=total,
            difficulty=difficulty,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._world is None:
            return
        stack = type(self)._open_stack
        if self in stack:
            stack.remove(self)
        closer = getattr(self._world, "close", None)
        if callable(closer):
            closer()
