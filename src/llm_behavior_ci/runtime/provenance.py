"""Refuse a live run that would record a commit other than the code executing.

``smoke_live_episode`` is the development smoke. It stamps the current
HEAD onto a one-off episode and does not call this module. A capture,
a live gate, a harm measurement, and a benchmark do.

A release may keep serving a production configuration built at an earlier
commit. ``execution_fingerprint`` and ``is_ancestor`` are the git facts the
release contract (``lifecycle.release``) reads to decide whether code at
HEAD executes that configuration as its own commit did.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from llm_behavior_ci.config import RunConfiguration

_RELEVANT_PREFIXES = (
    "configs/",
    "environments/",
    "scripts/",
    "src/",
)
_RELEVANT_FILES = frozenset({"requirements.txt"})
EXECUTION_PATHS = (
    "requirements.txt",
    "environments",
    "src/llm_behavior_ci/config.py",
    "src/llm_behavior_ci/records.py",
    "src/llm_behavior_ci/runtime",
    "src/llm_behavior_ci/tasks",
)
_ABSENT = "absent"


class ProvenanceError(ValueError):
    pass


@dataclass(frozen=True)
class RepositoryState:
    head: str
    dirty_paths: tuple[str, ...]


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def relevant_dirty_paths(paths: Sequence[str]) -> tuple[str, ...]:
    """Tracked paths whose contents the live run executes or configures."""

    chosen: list[str] = []
    for path in paths:
        if not isinstance(path, str) or path == "":
            raise ProvenanceError("dirty path must be a non-empty string")
        normalized = path.replace("\\", "/").lstrip("./")
        if normalized in _RELEVANT_FILES or normalized.startswith(_RELEVANT_PREFIXES):
            chosen.append(normalized)
    return tuple(chosen)


def require_committed_provenance(
    configuration: RunConfiguration,
    state: RepositoryState,
) -> None:
    """Refuse unless ``configuration.git_commit`` is ``state.head`` and the tree is clean.

    A dirty path outside source, scripts, configs, environments, and
    ``requirements.txt`` does not describe the code this run executes.
    """

    if not isinstance(configuration, RunConfiguration):
        raise ProvenanceError("provenance requires a run configuration")
    if not isinstance(state, RepositoryState):
        raise ProvenanceError("provenance requires a repository state")
    if state.head != configuration.git_commit:
        raise ProvenanceError("configured git_commit does not match HEAD")
    dirty = relevant_dirty_paths(state.dirty_paths)
    if dirty:
        raise ProvenanceError("tracked source or config files are dirty")


def enforce_committed_provenance(
    *configurations: RunConfiguration,
    state: RepositoryState | None = None,
    root: Path | None = None,
    reader: Callable[[Path], RepositoryState] | None = None,
) -> None:
    """Check each configuration against HEAD, reading git when no state is supplied."""

    if not configurations:
        raise ProvenanceError("provenance requires a run configuration")
    resolved = state
    if resolved is None:
        load = reader or read_repository_state
        resolved = load(root or repository_root())
    for configuration in configurations:
        require_committed_provenance(configuration, resolved)


def read_repository_state(
    root: Path | None = None,
    *,
    runner: Callable[..., object] = subprocess.run,
) -> RepositoryState:
    """Read HEAD and tracked dirty paths. ``runner`` stands in for git in tests."""

    repo = root if root is not None else repository_root()
    head_completed = _run_git(
        runner,
        ["git", "rev-parse", "HEAD"],
        repo,
        "git rev-parse HEAD failed",
    )
    head = _stdout(head_completed, "git rev-parse HEAD failed").strip()
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise ProvenanceError("git rev-parse HEAD failed")
    status_completed = _run_git(
        runner,
        ["git", "status", "--porcelain", "-uno", "-z"],
        repo,
        "git status failed",
    )
    payload = _stdout(status_completed, "git status failed")
    return RepositoryState(head=head, dirty_paths=_porcelain_paths(payload))


def _run_git(
    runner: Callable[..., object],
    command: list[str],
    root: Path,
    message: str,
) -> object:
    try:
        completed = runner(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise ProvenanceError(message) from error
    if getattr(completed, "returncode", 1) != 0:
        raise ProvenanceError(message)
    return completed


def _stdout(completed: object, message: str) -> str:
    payload = getattr(completed, "stdout", None)
    if not isinstance(payload, str):
        raise ProvenanceError(message)
    return payload


def _porcelain_paths(payload: str) -> tuple[str, ...]:
    if payload == "":
        return ()
    parts = payload.split("\0")
    paths: list[str] = []
    index = 0
    while index < len(parts):
        entry = parts[index]
        index += 1
        if entry == "":
            continue
        if len(entry) < 4 or entry[2] != " ":
            raise ProvenanceError("git status output is not porcelain")
        paths.append(entry[3:])
        status = entry[:2]
        if status[0] in {"R", "C"}:
            if index >= len(parts) or parts[index] == "":
                raise ProvenanceError("git status output is not porcelain")
            paths.append(parts[index])
            index += 1
    return tuple(paths)


def require_commit_id(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProvenanceError(f"{label} must be a 40-character lowercase git commit")
    return value


def is_ancestor(
    ancestor: str,
    descendant: str,
    root: Path | None = None,
    *,
    runner: Callable[..., object] = subprocess.run,
) -> bool:
    """Whether ``ancestor`` is ``descendant`` or reachable from it."""

    require_commit_id(ancestor, "ancestor")
    require_commit_id(descendant, "descendant")
    repo = root if root is not None else repository_root()
    try:
        completed = runner(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise ProvenanceError("git merge-base failed") from error
    code = getattr(completed, "returncode", 128)
    if code == 0:
        return True
    if code == 1:
        return False
    raise ProvenanceError("git merge-base failed; is the commit in this repository?")


def execution_fingerprint(
    commit: str,
    root: Path | None = None,
    *,
    runner: Callable[..., object] = subprocess.run,
    paths: Sequence[str] = EXECUTION_PATHS,
) -> str:
    """SHA-256 over the git object ids of the paths that execute an episode at ``commit``.

    Two commits with one fingerprint run a configuration through
    byte-identical runtime, configuration, task, record, and dependency
    code. A path absent at ``commit`` contributes ``absent``.
    """

    require_commit_id(commit, "commit")
    repo = root if root is not None else repository_root()
    _run_git(
        runner,
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        repo,
        "commit is not in this repository",
    )
    lines = []
    for path in paths:
        try:
            completed = runner(
                ["git", "rev-parse", "--verify", "--quiet", f"{commit}:{path}"],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as error:
            raise ProvenanceError("git rev-parse failed") from error
        code = getattr(completed, "returncode", 128)
        if code == 0:
            object_id = _stdout(completed, "git rev-parse failed").strip()
        elif code == 1:
            object_id = _ABSENT
        else:
            raise ProvenanceError("git rev-parse failed")
        lines.append(f"{path}\t{object_id}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
