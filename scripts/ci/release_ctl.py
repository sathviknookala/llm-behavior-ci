"""Provision, check, and serve releases on the trusted release machine.

``provision --release-root R --release-id ID --spec SPEC`` builds
``R/ID`` from a local ``release-spec-v1`` at the clean HEAD
(``lifecycle.release.provision_release``). The previous release is
``R/LATEST``; its service must be running and serving what that release
left serving. ``--reuse-existing`` accepts an already provisioned
``R/ID`` only when its manifest verifies at HEAD; nothing is rewritten.

``check-deployed --release-dir D`` runs just before a release takes the
service over: D verifies at HEAD, still continues ``LATEST``, its previous
summary is unchanged, and the running service serves D's production with
no canary open (or, for a first release, nothing answers on D's port).

``wait-registered --release-dir D`` polls ``GET /deployment`` until the
service registers D's production and candidate with no canary state.

``serve --release-dir D`` is the persistent service's entry point. It runs
from ``D/code`` (the release commit's ``git archive``), refuses a manifest
or code tree that is not that release's, and starts ``serve.py`` with
arguments read only from the manifest.

Output is hashes and outcomes only. Exit 0 on success, 2 for an invalid or
unsafe release, 4 when the service cannot be reached or does not register.
Bare invocation exits 2.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from llm_behavior_ci.config import ConfigError  # noqa: E402
from llm_behavior_ci.lifecycle.connected import ConnectedLifecycleError  # noqa: E402
from llm_behavior_ci.lifecycle.release import (  # noqa: E402
    CODE_COMMIT_FILE,
    CODE_DIR,
    ReleaseError,
    load_release_spec,
    provision_release,
    read_latest,
    require_current_lineage,
    require_deployed,
    require_previous_summary,
    require_release_id,
    service_argv,
    service_url,
    verify_release_manifest,
)
from llm_behavior_ci.runtime.clock import wall_now  # noqa: E402
from llm_behavior_ci.runtime.provenance import (  # noqa: E402
    ProvenanceError,
    read_repository_state,
)

EXIT_INVALID = 2
EXIT_UNAVAILABLE = 4


class ServiceUnreachable(RuntimeError):
    pass


def fetch_deployment(url: str, *, timeout: float = 5.0) -> dict[str, object]:
    import httpx

    try:
        response = httpx.get(f"{url}/deployment", timeout=timeout)
    except httpx.HTTPError as error:
        raise ServiceUnreachable(f"no service answers at {url}") from error
    if response.status_code != 200:
        raise ServiceUnreachable(f"GET /deployment returned {response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise ServiceUnreachable("GET /deployment did not return an object")
    return payload


def _previous_service_url(release_root: Path) -> str | None:
    latest = read_latest(release_root)
    if latest is None:
        return None
    return service_url(verify_release_manifest(release_root / latest))


def _public(manifest: dict[str, object]) -> dict[str, object]:
    configurations = manifest["configurations"]
    service = manifest["service"]
    assert isinstance(configurations, dict) and isinstance(service, dict)
    previous = manifest.get("previous")
    return {
        "release_id": manifest["release_id"],
        "release_commit": manifest["release_commit"],
        "previous_release_id": previous.get("release_id") if isinstance(previous, dict) else None,
        "configurations": {
            role: {"hash": entry["hash"], "git_commit": entry["git_commit"]}
            for role, entry in sorted(configurations.items())
        },
        "runtime": manifest["runtime"],
        "budget": manifest["budget"],
        "post_promotion_rollback": service["post_promotion_rollback"],
    }


def provision(args: argparse.Namespace, fetch: Callable[[str], dict[str, object]]) -> int:
    release_root = Path(args.release_root)
    release_id = require_release_id(args.release_id)
    state = read_repository_state()
    existing = release_root / release_id
    if existing.exists():
        if not args.reuse_existing:
            raise ReleaseError(f"release {release_id} already exists; releases are never overwritten")
        manifest = verify_release_manifest(existing, head=state.head)
        require_current_lineage(release_root, manifest)
        print(json.dumps({"reused": True, **_public(manifest)}, sort_keys=True))
        return 0
    previous_url = _previous_service_url(release_root)
    deployment = None if previous_url is None else fetch(previous_url)
    manifest = provision_release(
        load_release_spec(Path(args.spec)),
        release_root=release_root,
        release_id=release_id,
        state=state,
        deployment=deployment,
        now=wall_now(),
    )
    print(json.dumps({"reused": False, **_public(manifest)}, sort_keys=True))
    return 0


def check_deployed(args: argparse.Namespace, fetch: Callable[[str], dict[str, object]]) -> int:
    release_dir = Path(args.release_dir)
    release_root = release_dir.parent
    state = read_repository_state()
    manifest = verify_release_manifest(release_dir, head=state.head)
    require_current_lineage(release_root, manifest)
    previous_summary = require_previous_summary(release_root, manifest)
    if previous_summary is None:
        try:
            fetch(service_url(manifest))
        except ServiceUnreachable:
            print(json.dumps({"release_id": release_dir.name, "deployed": None}))
            return 0
        raise ReleaseError(
            "a first release needs its port free, but a service already answers there"
        )
    previous_url = _previous_service_url(release_root)
    assert previous_url is not None
    configurations = manifest["configurations"]
    assert isinstance(configurations, dict)
    production = configurations["production"]["hash"]
    require_deployed(fetch(previous_url), production)
    print(json.dumps({"release_id": release_dir.name, "deployed": production}))
    return 0


def wait_registered(
    args: argparse.Namespace,
    fetch: Callable[[str], dict[str, object]],
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    release_dir = Path(args.release_dir)
    manifest = verify_release_manifest(release_dir)
    configurations = manifest["configurations"]
    assert isinstance(configurations, dict)
    expected = (configurations["production"]["hash"], configurations["candidate"]["hash"])
    url = service_url(manifest)
    deadline = time.monotonic() + float(args.timeout_seconds)
    while True:
        try:
            deployment = fetch(url)
        except ServiceUnreachable:
            deployment = None
        if deployment is not None:
            registered = (
                deployment.get("production_configuration_hash"),
                deployment.get("candidate_configuration_hash"),
            )
            if registered == expected and "state" not in deployment:
                print(json.dumps({"release_id": release_dir.name, "registered": True}))
                return 0
            if registered != expected and None not in registered:
                print("the service on the release port serves another release", file=sys.stderr)
                return EXIT_UNAVAILABLE
        if time.monotonic() >= deadline:
            print("the release service did not register in time", file=sys.stderr)
            return EXIT_UNAVAILABLE
        sleep(1.0)


def _load_serve():
    spec = importlib.util.spec_from_file_location(
        "serve_for_release", _REPO / "scripts" / "service" / "serve.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def serve(args: argparse.Namespace) -> int:
    release_dir = Path(args.release_dir).resolve()
    manifest = verify_release_manifest(release_dir)
    code = release_dir / CODE_DIR
    if _REPO != code.resolve():
        raise ReleaseError("the release service must run from the release's own code")
    commit_file = code / CODE_COMMIT_FILE
    if not commit_file.is_file() or commit_file.read_text(encoding="utf-8").strip() != manifest[
        "release_commit"
    ]:
        raise ReleaseError("release code is not the release commit")
    import llm_behavior_ci

    if code.resolve() not in Path(llm_behavior_ci.__file__).resolve().parents:
        raise ReleaseError("llm_behavior_ci is imported from outside the release code")
    return int(_load_serve().main(service_argv(release_dir, manifest)))


def main(
    argv: Sequence[str] | None = None,
    *,
    fetch: Callable[[str], dict[str, object]] = fetch_deployment,
) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        return EXIT_INVALID
    parser = argparse.ArgumentParser(description="Provision, check, and serve releases.")
    commands = parser.add_subparsers(dest="command", required=True)
    provision_parser = commands.add_parser("provision")
    provision_parser.add_argument("--release-root", required=True)
    provision_parser.add_argument("--release-id", required=True)
    provision_parser.add_argument("--spec", required=True)
    provision_parser.add_argument("--reuse-existing", action="store_true")
    for name in ("check-deployed", "serve"):
        commands.add_parser(name).add_argument("--release-dir", required=True)
    wait = commands.add_parser("wait-registered")
    wait.add_argument("--release-dir", required=True)
    wait.add_argument("--timeout-seconds", type=float, default=180.0)
    try:
        args = parser.parse_args(args_list)
    except SystemExit as error:
        return EXIT_INVALID if error.code is None else int(error.code)
    try:
        if args.command == "provision":
            return provision(args, fetch)
        if args.command == "check-deployed":
            return check_deployed(args, fetch)
        if args.command == "wait-registered":
            return wait_registered(args, fetch)
        return serve(args)
    except ServiceUnreachable as error:
        print(str(error), file=sys.stderr)
        return EXIT_UNAVAILABLE
    except (
        ReleaseError,
        ConfigError,
        ConnectedLifecycleError,
        ProvenanceError,
        OSError,
        ValueError,
        KeyError,
    ) as error:
        print(str(error) or type(error).__name__, file=sys.stderr)
        return EXIT_INVALID


if __name__ == "__main__":
    raise SystemExit(main())
