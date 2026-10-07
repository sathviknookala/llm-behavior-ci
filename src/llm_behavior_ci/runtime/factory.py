"""Configuration-, mode-, and role-aware live runtime construction.

Every live command builds its runtimes here: the offline gate, the A/A
capture, the baseline collector, harm characterization, the service, and
the lifecycle benchmark. A runtime is always built for one configuration,
one mode, and one role, so plan mode never receives the execute workflow
wrapper and two roles never share an agent, even when their
configurations are byte-identical.

A vLLM configuration needs that role's endpoint. A hosted configuration
must not be given one; its provider key is read from the process
environment when the runtime is built, so a factory carries no secret and
pickles into a spawned worker as plain configuration and endpoint data.
``preflight`` reports a missing endpoint or key before any world opens or
any provider is called.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime

from llm_behavior_ci.config import (
    AnthropicModelConfiguration,
    OpenAICompatibleModelConfiguration,
    RunConfiguration,
    hosted_provider,
    run_configuration_hash,
)
from llm_behavior_ci.runtime.episode import EpisodeRejected, RuntimeDependencies

ROLES = ("reference", "candidate", "production", "do_nothing")


class RuntimeFactoryError(EpisodeRejected):
    """A runtime cannot be built for a role; the message names what is missing."""


def provider_key_variable(configuration: RunConfiguration) -> str | None:
    """The environment variable a hosted configuration reads, or ``None`` for vLLM."""

    from llm_behavior_ci.runtime.agent import OPENAI_COMPATIBLE_KEY_VARIABLES

    model = configuration.model
    if isinstance(model, AnthropicModelConfiguration):
        return "ANTHROPIC_API_KEY"
    if isinstance(model, OpenAICompatibleModelConfiguration):
        variable = OPENAI_COMPATIBLE_KEY_VARIABLES.get(model.provider)
        if variable is None:
            raise RuntimeFactoryError("unsupported OpenAI-compatible provider")
        return variable
    return None


def is_hosted(configuration: RunConfiguration) -> bool:
    return hosted_provider(configuration.model) is not None


def _role(role: str) -> str:
    if role not in ROLES:
        raise RuntimeFactoryError(f"runtime role must be one of: {', '.join(ROLES)}")
    return role


class RuntimeFactory:
    """Builds one runtime for ``(configuration, mode, role)``."""

    def __call__(
        self,
        configuration: RunConfiguration,
        *,
        mode: str,
        role: str = "reference",
    ) -> RuntimeDependencies:
        raise NotImplementedError

    def preflight(
        self,
        roles: Mapping[str, RunConfiguration],
        *,
        require_distinct_endpoints: bool = True,
    ) -> None:
        del roles, require_distinct_endpoints


@dataclass(frozen=True)
class LiveRuntimeFactory(RuntimeFactory):
    """The live factory. ``endpoints`` maps a role to its vLLM base URL.

    Only roles whose configuration is vLLM need an entry. Credentials are
    resolved from the environment of the process that calls the factory,
    never stored on it.
    """

    endpoints: tuple[tuple[str, str], ...] = ()
    clock: Callable[[], datetime] | None = None

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for role, url in self.endpoints:
            _role(role)
            if role in seen:
                raise RuntimeFactoryError(f"{role} endpoint is given twice")
            if not isinstance(url, str) or url.strip() == "":
                raise RuntimeFactoryError(f"{role} endpoint is empty")
            seen.add(role)

    @classmethod
    def from_endpoints(
        cls,
        clock: Callable[[], datetime] | None = None,
        **endpoints: str | None,
    ) -> LiveRuntimeFactory:
        pairs = tuple(
            (role, url.strip())
            for role, url in sorted(endpoints.items())
            if url is not None and url.strip() != ""
        )
        return cls(endpoints=pairs, clock=clock)

    def endpoint_for(self, role: str) -> str | None:
        _role(role)
        for name, url in self.endpoints:
            if name == role:
                return url
        return None

    def _endpoint(self, configuration: RunConfiguration, role: str) -> str | None:
        endpoint = self.endpoint_for(role)
        if is_hosted(configuration):
            if endpoint is not None:
                raise RuntimeFactoryError(
                    f"{role} uses a hosted provider and must not be given an endpoint"
                )
            return None
        if endpoint is None:
            raise RuntimeFactoryError(
                f"{role} uses a vLLM configuration and needs its endpoint "
                f"(--{role.replace('_', '-')}-endpoint)"
            )
        return endpoint

    def __call__(
        self,
        configuration: RunConfiguration,
        *,
        mode: str,
        role: str = "reference",
    ) -> RuntimeDependencies:
        from llm_behavior_ci.runtime.episode import build_runtime

        if not isinstance(configuration, RunConfiguration):
            raise RuntimeFactoryError("runtime requires a run configuration")
        _role(role)
        return build_runtime(
            configuration,
            self._endpoint(configuration, role),
            mode=mode,
            clock=self.clock,
        )

    def preflight(
        self,
        roles: Mapping[str, RunConfiguration],
        *,
        require_distinct_endpoints: bool = True,
    ) -> None:
        """Fail before any episode when a role lacks its endpoint or key.

        With ``require_distinct_endpoints``, two vLLM roles whose
        configuration hashes differ must use different endpoints. The
        error names the variable or flag, never a key value.
        """

        local: dict[str, tuple[str, str]] = {}
        for role, configuration in roles.items():
            _role(role)
            endpoint = self._endpoint(configuration, role)
            _require_key(configuration, role)
            if endpoint is not None:
                local[role] = (endpoint, run_configuration_hash(configuration))
        if not require_distinct_endpoints:
            return
        items = sorted(local.items())
        for index, (left_role, (left_url, left_hash)) in enumerate(items):
            for right_role, (right_url, right_hash) in items[index + 1:]:
                if left_url == right_url and left_hash != right_hash:
                    raise RuntimeFactoryError(
                        f"{left_role} and {right_role} serve different vLLM "
                        "configurations and must use distinct endpoints"
                    )


def _require_key(configuration: RunConfiguration, label: str) -> None:
    variable = provider_key_variable(configuration)
    if variable is not None and os.environ.get(variable, "").strip() == "":
        raise RuntimeFactoryError(
            f"{label} uses provider {hosted_provider(configuration.model)}; "
            f"{variable} is required in the environment"
        )


@dataclass(frozen=True)
class ConfigurationRoutedRuntimeFactory(RuntimeFactory):
    """Routes by configuration hash rather than by role.

    The service uses it: after promotion the candidate configuration serves
    production traffic from the server it was admitted on, so the endpoint
    follows the configuration. ``routes`` pairs each registered hash with
    its vLLM endpoint, or ``None`` for a hosted configuration.
    """

    routes: tuple[tuple[str, str | None], ...]
    clock: Callable[[], datetime] | None = None

    @classmethod
    def for_configurations(
        cls,
        entries: Mapping[str, tuple[RunConfiguration, str | None]],
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> ConfigurationRoutedRuntimeFactory:
        """Validate ``label -> (configuration, endpoint)`` and build the routes.

        A hosted entry must have no endpoint and its key in the environment;
        a vLLM entry needs an endpoint, and two different vLLM
        configurations need distinct endpoints.
        """

        routes: dict[str, str | None] = {}
        claimed: dict[str, str] = {}
        for label, (configuration, endpoint) in entries.items():
            url = None if endpoint is None or endpoint.strip() == "" else endpoint.strip()
            if is_hosted(configuration):
                if url is not None:
                    raise RuntimeFactoryError(
                        f"{label} uses a hosted provider and must not be given a base URL"
                    )
                _require_key(configuration, label)
            elif url is None:
                raise RuntimeFactoryError(
                    f"{label} uses a vLLM configuration and needs its base URL "
                    f"(--{label}-base-url)"
                )
            digest = run_configuration_hash(configuration)
            if digest in routes and routes[digest] != url:
                raise RuntimeFactoryError(
                    "one configuration cannot be routed to two endpoints"
                )
            if url is not None and claimed.get(url, digest) != digest:
                raise RuntimeFactoryError(
                    "different vLLM configurations must use distinct base URLs"
                )
            if url is not None:
                claimed[url] = digest
            routes[digest] = url
        return cls(routes=tuple(sorted(routes.items())), clock=clock)

    def __call__(
        self,
        configuration: RunConfiguration,
        *,
        mode: str,
        role: str = "reference",
    ) -> RuntimeDependencies:
        from llm_behavior_ci.runtime.episode import RuntimeUnavailable, build_runtime

        _role(role)
        digest = run_configuration_hash(configuration)
        for route_hash, endpoint in self.routes:
            if route_hash == digest:
                return build_runtime(configuration, endpoint, mode=mode, clock=self.clock)
        raise RuntimeUnavailable("configuration is not in the registry")


@dataclass(frozen=True)
class StaticRuntimeFactory(RuntimeFactory):
    """Returns supplied runtimes by role. CPU tests and injected hooks use it.

    ``candidate`` falls back to ``reference`` only when it is omitted. When
    the agent exposes ``set_mode``, the requested mode is applied, as the
    legacy single-runtime path did.
    """

    reference: RuntimeDependencies
    candidate: RuntimeDependencies | None = None
    others: tuple[tuple[str, RuntimeDependencies], ...] = ()

    def __call__(
        self,
        configuration: RunConfiguration,
        *,
        mode: str,
        role: str = "reference",
    ) -> RuntimeDependencies:
        del configuration
        _role(role)
        runtime = self.reference
        if role == "candidate" and self.candidate is not None:
            runtime = self.candidate
        for name, item in self.others:
            if name == role:
                runtime = item
        setter = getattr(runtime.agent, "set_mode", None)
        if callable(setter):
            setter(mode)
        return runtime


@dataclass(frozen=True)
class ConfiguredRuntimeFactory:
    """A picklable ``mode -> runtime`` callable for one configuration and role.

    The A/A capture sends this to spawned workers. It carries the run
    configuration and the endpoint, if any, and builds the runtime in the
    worker, which reads its own provider key.
    """

    configuration: RunConfiguration
    role: str
    factory: LiveRuntimeFactory

    def __call__(self, mode: str) -> RuntimeDependencies:
        return self.factory(self.configuration, mode=mode, role=self.role)
