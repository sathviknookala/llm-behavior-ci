from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from llm_behavior_ci.config import (
    HASHED_FIELDS,
    ConfigError,
    RunConfiguration,
    hashed_values,
    new_run_identity,
    run_configuration_hash,
)
from llm_behavior_ci.runtime.episode import (
    EpisodeRejected,
    RuntimeDependencies,
    evaluator_difference,
    run_pair,
)
from llm_behavior_ci.stats.bootstrap import clustered_paired_bootstrap
from llm_behavior_ci.tasks.selection import SelectionError, TaskSet, verify_task_set

_FAULT_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_VERSION = re.compile(r"^[1-9][0-9]{0,8}$")
_FAULT_VERSION = re.compile(r"^[a-z][a-z0-9_]{0,63}:[1-9][0-9]{0,8}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_SCHEMA = 256
_FLOAT_PATCH_PATHS = frozenset(
    {
        "agent.sampling.temperature",
        "agent.sampling.top_p",
        "agent.sampling.min_p",
    }
)
_KINDS = frozenset(
    {
        "model",
        "quantization",
        "prompt",
        "template",
        "sampling",
        "token_limit",
        "step_limit",
        "api_documentation",
        "lora",
        "benign_control",
    }
)
_SCHEMA_KINDS: frozenset[str] = frozenset()
"""Fault kinds whose hashed leaves the current schema cannot yet represent.

Empty: every catalog ``kind`` in ``_KINDS`` has a real ``_KIND_PATHS`` entry
now, including ``api_documentation`` and ``lora``. Kept as a named, checked
set (not deleted) so a future fault class that genuinely needs a schema
before it can be patched has somewhere to declare that gap the same way
these two once did, via ``schema_request``, rather than reusing this
mechanism only implicitly.
"""
_LIVE_UNAVAILABLE_KINDS = frozenset(
    {
        "quantization",
        "lora",
    }
)
"""Representable fault kinds that still need a different vLLM server process.

Both change a ``build_runtime``/launch-spec input
(``runtime.launch_spec.build_vllm_launch_spec``) that only takes effect when
a server is (re)started with different weights: ``--quantization`` needs
quantized weights on disk, ``--enable-lora`` needs the adapter's weights.
Neither can be applied to an already-running server via a chat-completions
request the way ``sampling``, ``prompt``, ``template``, ``token_limit``,
``step_limit``, and ``model`` (a different *base* repository, still loaded
fresh) can. ``api_documentation`` is deliberately not here: correcting or
corrupting the API-documentation text a prompt is built from
(``runtime.api_docs.resolve_api_documentation``) changes only the request
content sent to whichever server is already running, so it needs no
relaunch and is live like the request-level kinds.
"""
_LIVE_UNAVAILABLE_CONTROLS = frozenset({"batch_invariant"})
_LIVE_UNAVAILABLE_REASONS: Mapping[str, str] = {
    "quantization": (
        "the launch spec would carry --quantization, but quantized weights "
        "and a running vLLM process are unavailable here"
    ),
    "lora": (
        "the launch spec would carry --enable-lora, but adapter weights and "
        "a running vLLM process are unavailable here"
    ),
    "batch_invariant": (
        "the launch spec would carry VLLM_BATCH_INVARIANT=1, but a running "
        "vLLM process is unavailable here"
    ),
}
_BENIGN_CONTROLS = frozenset(
    {
        "identical",
        "noop_redeploy",
        "batch_invariant",
        "logging_refactor",
    }
)
_KIND_PATHS: Mapping[str, frozenset[str]] = {
    "model": frozenset(
        {
            "model.model.repository",
            "model.model.revision",
            "model.tokenizer.repository",
            "model.tokenizer.revision",
        }
    ),
    "quantization": frozenset({"model.quantization.method"}),
    "prompt": frozenset(
        {
            "agent.prompt.prompt_version",
            "agent.prompt.plan_format_version",
        }
    ),
    "template": frozenset({"agent.prompt.thinking_enabled"}),
    "sampling": frozenset(
        {
            "agent.sampling.temperature",
            "agent.sampling.top_p",
            "agent.sampling.top_k",
            "agent.sampling.min_p",
        }
    ),
    "token_limit": frozenset({"agent.sampling.max_tokens"}),
    "step_limit": frozenset({"agent.step_limit"}),
    "api_documentation": frozenset(
        {
            "agent.api_docs_version",
            "agent.api_docs_app",
        }
    ),
    "lora": frozenset(
        {
            "model.lora.repository",
            "model.lora.revision",
        }
    ),
}
_SCHEMA_KIND_PATHS: Mapping[str, frozenset[str]] = {}
_BENIGN_PATHS: Mapping[str, frozenset[str]] = {
    "identical": frozenset(),
    "noop_redeploy": frozenset({"git_commit"}),
    "batch_invariant": frozenset({"model.serving.batch_invariant"}),
    "logging_refactor": frozenset({"git_commit"}),
}
_CATALOG_KEYS = frozenset(
    {
        "fault_id",
        "version",
        "kind",
        "patches",
        "control",
        "schema_request",
    }
)
_REQUIRED_CATALOG_KEYS = frozenset({"fault_id", "version", "kind", "patches"})
_PATCH_KEYS = frozenset({"path", "value"})
_KNOWN_LEAVES = frozenset(HASHED_FIELDS)


class FaultError(ValueError):
    pass


@dataclass(frozen=True)
class FaultPatch:
    path: str
    value: object


@dataclass(frozen=True)
class FaultSpec:
    fault_id: str
    version: str
    kind: str
    patches: tuple[FaultPatch, ...]
    control: str | None = None
    schema_request: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.fault_id, str) or _FAULT_ID.fullmatch(self.fault_id) is None:
            raise FaultError("fault_id must be a lowercase identifier")
        if not isinstance(self.version, str) or _VERSION.fullmatch(self.version) is None:
            raise FaultError("version must be a positive decimal integer string")
        if self.kind not in _KINDS:
            raise FaultError(f"kind must be one of: {', '.join(sorted(_KINDS))}")
        if not isinstance(self.patches, tuple) or any(
            not isinstance(patch, FaultPatch) for patch in self.patches
        ):
            raise FaultError("patches must be a tuple of FaultPatch")
        paths = tuple(patch.path for patch in self.patches)
        if len(paths) != len(set(paths)):
            raise FaultError("patches have duplicate paths")
        if self.kind == "benign_control":
            if self.control not in _BENIGN_CONTROLS:
                raise FaultError(
                    "control must be one of: "
                    + ", ".join(sorted(_BENIGN_CONTROLS))
                )
            if self.schema_request is not None:
                raise FaultError("schema_request must be None")
            allowed = _BENIGN_PATHS[self.control]
            path_set = frozenset(paths)
            for path in paths:
                _validate_patch_path(path, allowed)
            if path_set != allowed:
                raise FaultError(
                    f"benign control {self.control} requires paths {sorted(allowed)}"
                )
            return
        if self.control is not None:
            raise FaultError("control must be None")
        if self.kind in _SCHEMA_KINDS:
            if paths:
                raise FaultError(f"{self.kind} patches must be empty")
            _schema_request(self.schema_request)
            return
        if self.schema_request is not None:
            raise FaultError("schema_request must be None")
        if not paths:
            raise FaultError(f"{self.kind} requires at least one patch")
        allowed = _KIND_PATHS[self.kind]
        for path in paths:
            _validate_patch_path(path, allowed)

    @property
    def fault_version(self) -> str:
        return f"{self.fault_id}:{self.version}"

    @property
    def representable(self) -> bool:
        return self.schema_request is None

    @property
    def schema_supported(self) -> bool:
        if self.kind not in _KINDS:
            return False
        if self.kind in _SCHEMA_KINDS:
            return (
                self.schema_request is not None
                and self.kind in _SCHEMA_KIND_PATHS
                and len(self.patches) == 0
            )
        return True


@dataclass(frozen=True)
class LiveFaultAvailability:
    available: bool
    reason: str | None = None


def live_fault_available(fault: FaultSpec) -> LiveFaultAvailability:
    """Report whether a catalog fault can be executed live.

    Schema-gap and GPU-backed faults stay in coverage reports as unavailable
    rather than absent. ``apply_fault`` still rejects schema-gap faults and
    does not apply an empty patch for them.
    """

    if not isinstance(fault, FaultSpec):
        raise FaultError("live_fault_available requires a fault spec")
    if fault.schema_request is not None:
        return LiveFaultAvailability(available=False, reason=str(fault.schema_request))
    if fault.kind in _LIVE_UNAVAILABLE_KINDS:
        return LiveFaultAvailability(
            available=False,
            reason=_LIVE_UNAVAILABLE_REASONS[fault.kind],
        )
    if fault.kind == "benign_control" and fault.control in _LIVE_UNAVAILABLE_CONTROLS:
        return LiveFaultAvailability(
            available=False,
            reason=_LIVE_UNAVAILABLE_REASONS[fault.control],
        )
    return LiveFaultAvailability(available=True, reason=None)


@dataclass(frozen=True)
class HarmLabel:
    """Clustered paired harm label for one representable fault on a frozen dev task set.

    ``effect_estimate`` is the clustered paired mean of candidate evaluator
    success minus base evaluator success. A fault is harmful when the drop,
    which is the negation of that estimate, is at least ``margin``. The
    interval is the clustered paired bootstrap percentile interval of that
    same difference. Plan text, tool traces, and requirement fractions are
    not the effect. AppWorld evaluator success is the only outcome.
    ``test_normal`` is not run. The label stores no task id, plan text, or
    prompt.
    """

    fault_version: str
    base_configuration_hash: str
    candidate_configuration_hash: str
    task_set_hash: str
    effect_estimate: float
    interval_low: float
    interval_high: float
    margin: float
    harmful: bool
    split: str
    confidence_level: float
    resamples: int
    seed: int

    def __post_init__(self) -> None:
        if self.split != "dev":
            raise FaultError("split must be dev")
        if (
            not isinstance(self.fault_version, str)
            or _FAULT_VERSION.fullmatch(self.fault_version) is None
        ):
            raise FaultError("fault_version must identify a fault and version")
        for name in (
            "base_configuration_hash",
            "candidate_configuration_hash",
            "task_set_hash",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise FaultError(f"{name} must be a lowercase SHA-256 hex digest")
        effect = _require_finite_float(self.effect_estimate, "effect_estimate")
        low = _require_finite_float(self.interval_low, "interval_low")
        high = _require_finite_float(self.interval_high, "interval_high")
        if low > high:
            raise FaultError("interval_low must not exceed interval_high")
        margin = _require_margin(self.margin)
        _require_confidence_level(self.confidence_level)
        _require_resamples(self.resamples)
        _require_seed(self.seed)
        if not isinstance(self.harmful, bool):
            raise FaultError("harmful must be a bool")
        if self.harmful != ((-effect) >= margin):
            raise FaultError("harmful does not match the effect and margin")


def _schema_request(value: object) -> str:
    if not isinstance(value, str) or value == "" or value != value.strip():
        raise FaultError("schema_request must be a non-empty string")
    if len(value) > _MAX_SCHEMA or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise FaultError(
            f"schema_request must be a single line of at most {_MAX_SCHEMA} characters"
        )
    return value


def _validate_patch_path(path: object, allowed: frozenset[str]) -> None:
    if not isinstance(path, str) or path == "":
        raise FaultError("patch path must be a non-empty string")
    if path not in _KNOWN_LEAVES:
        raise FaultError(f"unknown path {path}")
    if path not in allowed:
        raise FaultError(f"change outside the declared fault: {path}")


def _coerce_patch_value(path: str, value: object) -> object:
    if (
        path in _FLOAT_PATCH_PATHS
        and isinstance(value, int)
        and not isinstance(value, bool)
    ):
        return float(value)
    return value


def _set_leaf(document: dict[str, object], path: str, value: object) -> None:
    """Set one already-validated ``HASHED_FIELDS`` path, creating missing parents.

    Every path reaching here was already checked against ``_KIND_PATHS`` (or
    ``_BENIGN_PATHS``) in ``FaultSpec.__post_init__``, so auto-creating a
    missing intermediate object is safe: it only ever happens at an optional
    leaf's parent (``model.lora``, or ``agent`` before ``api_docs_version``/
    ``api_docs_app`` exist) on a base configuration that has not set it.
    """

    parts = path.split(".")
    node: dict[str, object] = document
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def _under(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    base = root.resolve()
    return resolved == base or base in resolved.parents


def _require_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, float) or not math.isfinite(value):
        raise FaultError(f"{name} must be a finite float")
    return value


def _require_margin(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, float):
        raise FaultError("margin must be a float")
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise FaultError("margin must be in (0, 1]")
    return value


def _require_confidence_level(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, float):
        raise FaultError("confidence_level must be a float")
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise FaultError("confidence_level must be in (0, 1)")
    return value


def _require_resamples(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise FaultError("resamples must be a positive integer")
    return value


def _require_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FaultError("seed must be an integer")
    return value


def fault_from_mapping(payload: object) -> FaultSpec:
    """Build a ``FaultSpec`` from a catalog mapping.

    Coerces JSON ints to floats only for temperature, top_p, and min_p.
    Rejects unknown keys.
    """

    if not isinstance(payload, Mapping):
        raise FaultError("fault document must be an object")
    if not all(isinstance(key, str) for key in payload):
        raise FaultError("fault document has a non-string field name")
    unknown = sorted(set(payload) - _CATALOG_KEYS)
    if unknown:
        raise FaultError(f"fault document contains unknown fields: {', '.join(unknown)}")
    missing = sorted(_REQUIRED_CATALOG_KEYS - set(payload))
    if missing:
        raise FaultError(f"fault document is missing fields: {', '.join(missing)}")
    patches_raw = payload["patches"]
    if not isinstance(patches_raw, list):
        raise FaultError("patches must be a list")
    patches: list[FaultPatch] = []
    for item in patches_raw:
        if not isinstance(item, Mapping):
            raise FaultError("patch must be an object")
        if set(item) != _PATCH_KEYS:
            raise FaultError("patch must contain exactly path and value")
        path = item["path"]
        if not isinstance(path, str):
            raise FaultError("patch path must be a string")
        patches.append(
            FaultPatch(path=path, value=_coerce_patch_value(path, item["value"]))
        )
    return FaultSpec(
        fault_id=payload["fault_id"],
        version=payload["version"],
        kind=payload["kind"],
        patches=tuple(patches),
        control=payload.get("control"),
        schema_request=payload.get("schema_request"),
    )


def fault_to_mapping(fault: FaultSpec) -> dict[str, object]:
    """The catalog-shaped mapping ``fault_from_mapping`` would reconstruct.

    The inverse of ``fault_from_mapping``, so a protocol lock can bind the
    exact declared faults it authorizes without re-deriving the catalog
    JSON shape at every call site that needs to persist one.
    """

    if not isinstance(fault, FaultSpec):
        raise FaultError("fault_to_mapping requires a FaultSpec")
    document: dict[str, object] = {
        "fault_id": fault.fault_id,
        "version": fault.version,
        "kind": fault.kind,
        "patches": [
            {"path": patch.path, "value": patch.value} for patch in fault.patches
        ],
    }
    if fault.control is not None:
        document["control"] = fault.control
    if fault.schema_request is not None:
        document["schema_request"] = fault.schema_request
    return document


def load_fault(path: Path) -> FaultSpec:
    """Load one fault catalog JSON file into a validated ``FaultSpec``."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FaultError(f"fault file is not readable JSON: {path}") from error
    return fault_from_mapping(payload)


def load_fault_catalog(directory: Path) -> tuple[FaultSpec, ...]:
    """Load every ``*.json`` fault under ``directory``, sorted by id and version."""

    root = Path(directory)
    if not root.is_dir():
        raise FaultError(f"fault catalog is not a directory: {root}")
    loaded: list[FaultSpec] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(root.glob("*.json")):
        fault = load_fault(path)
        key = (fault.fault_id, fault.version)
        if key in seen:
            raise FaultError(
                f"duplicate fault_id and version: {fault.fault_version}"
            )
        seen.add(key)
        loaded.append(fault)
    loaded.sort(key=lambda item: (item.fault_id, item.version))
    return tuple(loaded)


def apply_fault(base: RunConfiguration, fault: FaultSpec) -> RunConfiguration:
    """Return a new configuration with only the fault's declared hashed leaves changed.

    Does not mutate ``base``. Schema-gap faults raise and do not return a
    configuration. Live-unavailable representable faults may still patch
    hashed leaves; ``live_fault_available`` reports execution status. After
    reconstruction, the set of changed ``HASHED_FIELDS`` must equal the
    declared patch paths.
    """

    if not isinstance(base, RunConfiguration):
        raise FaultError("apply_fault requires a run configuration")
    if not isinstance(fault, FaultSpec):
        raise FaultError("apply_fault requires a fault spec")
    if not fault.representable:
        raise FaultError(str(fault.schema_request))
    document = copy.deepcopy(base.to_dict())
    declared = frozenset(patch.path for patch in fault.patches)
    for patch in fault.patches:
        _set_leaf(document, patch.path, patch.value)
    try:
        produced = RunConfiguration.from_dict(document)
    except ConfigError as error:
        raise FaultError(f"invalid value: {error}") from error
    before = hashed_values(base)
    after = hashed_values(produced)
    changed = frozenset(
        path for path in HASHED_FIELDS if before[path] != after[path]
    )
    if changed != declared:
        raise FaultError("outside the declared fault")
    return produced


def measure_harm(
    base: RunConfiguration,
    candidate: RunConfiguration,
    task_set: TaskSet,
    *,
    margin: float,
    runtime: RuntimeDependencies,
    fault: FaultSpec,
    confidence_level: float,
    resamples: int,
    seed: int,
) -> HarmLabel:
    """Label whether a declared fault is harmful on a frozen ``dev`` task set.

    ``effect_estimate`` is the clustered paired mean of candidate evaluator
    success minus base evaluator success. A fault is harmful when the drop,
    which is the negation of that estimate, is at least margin. The interval
    is the clustered paired bootstrap percentile interval of that same
    difference. Plan text, tool traces, and requirement fractions are not
    the effect. AppWorld evaluator success is the only outcome.
    ``test_normal`` is not run.
    """

    if not isinstance(task_set, TaskSet):
        raise FaultError("measure_harm requires a task set")
    if task_set.split != "dev":
        raise FaultError("measure_harm requires a dev task set")
    margin_value = _require_margin(margin)
    confidence_value = _require_confidence_level(confidence_level)
    resample_count = _require_resamples(resamples)
    seed_value = _require_seed(seed)
    if not isinstance(fault, FaultSpec):
        raise FaultError("measure_harm requires a fault spec")
    if not fault.representable:
        raise FaultError(str(fault.schema_request))
    if not isinstance(base, RunConfiguration) or not isinstance(
        candidate, RunConfiguration
    ):
        raise FaultError("measure_harm requires run configurations")
    if not isinstance(runtime, RuntimeDependencies):
        raise FaultError("measure_harm requires runtime dependencies")
    produced = apply_fault(base, fault)
    if produced != candidate:
        raise FaultError("candidate is not the declared fault")
    try:
        verify_task_set(base.task, task_set)
        verify_task_set(candidate.task, task_set)
    except SelectionError as error:
        raise FaultError(str(error)) from error
    reference_run = new_run_identity(base)
    candidate_run = new_run_identity(candidate)
    candidate_scores: list[float] = []
    base_scores: list[float] = []
    clusters: list[str] = []
    lone_index = 0
    try:
        for index, task_id in enumerate(task_set.task_ids):
            scenario_id = task_set.scenario_ids[index]
            pair = run_pair(
                task_id,
                base,
                candidate,
                reference_run=reference_run,
                candidate_run=candidate_run,
                runtime=runtime,
                mode="execute",
                scenario_id=scenario_id,
            )
            difference = evaluator_difference(pair)
            reference_outcome = pair.reference.evaluator_outcome
            candidate_outcome = pair.candidate.evaluator_outcome
            if (
                difference is None
                or reference_outcome is None
                or candidate_outcome is None
            ):
                raise FaultError("missing evaluator outcome")
            candidate_scores.append(float(candidate_outcome.success))
            base_scores.append(float(reference_outcome.success))
            if scenario_id is not None:
                clusters.append(f"scenario:{scenario_id}")
            else:
                clusters.append(f"lone:{lone_index}")
                lone_index += 1
    except EpisodeRejected as error:
        raise FaultError(str(error)) from error
    bootstrap = clustered_paired_bootstrap(
        candidate_scores,
        base_scores,
        clusters,
        confidence_level=confidence_value,
        resamples=resample_count,
        seed=seed_value,
    )
    effect = bootstrap.mean_difference
    return HarmLabel(
        fault_version=fault.fault_version,
        base_configuration_hash=run_configuration_hash(base),
        candidate_configuration_hash=run_configuration_hash(candidate),
        task_set_hash=task_set.task_set_hash,
        effect_estimate=effect,
        interval_low=bootstrap.confidence_low,
        interval_high=bootstrap.confidence_high,
        margin=margin_value,
        harmful=(-effect) >= margin_value,
        split="dev",
        confidence_level=confidence_value,
        resamples=resample_count,
        seed=seed_value,
    )


def default_results_root() -> Path:
    return Path(__file__).resolve().parents[3] / "results"


def _harm_label_document(label: HarmLabel) -> dict[str, object]:
    return {
        "record": "harm_label",
        "fault_version": label.fault_version,
        "base_configuration_hash": label.base_configuration_hash,
        "candidate_configuration_hash": label.candidate_configuration_hash,
        "task_set_hash": label.task_set_hash,
        "effect_estimate": label.effect_estimate,
        "interval_low": label.interval_low,
        "interval_high": label.interval_high,
        "margin": label.margin,
        "harmful": label.harmful,
        "split": label.split,
        "confidence_level": label.confidence_level,
        "resamples": label.resamples,
        "seed": label.seed,
    }


def _canonical_harm_label_json(label: HarmLabel) -> bytes:
    return json.dumps(
        _harm_label_document(label),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def freeze_harm_label(
    label: HarmLabel,
    destination: Path,
    *,
    results_root: Path | None = None,
) -> None:
    """Record a harm label once. Refuses paths under ``results_root`` and overwrites."""

    if not isinstance(label, HarmLabel):
        raise FaultError("freeze_harm_label requires a harm label")
    if label.split != "dev":
        raise FaultError("split must be dev")
    root = default_results_root() if results_root is None else Path(results_root)
    output = Path(destination)
    if _under(output, root):
        raise FaultError("harm label cannot be written under results")
    payload = _canonical_harm_label_json(label)
    output.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(output, flags)
    except FileExistsError as error:
        try:
            existing = output.read_bytes()
        except OSError as read_error:
            raise FaultError("already frozen") from read_error
        if existing == payload:
            return
        raise FaultError("already frozen") from error
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
    except Exception:
        try:
            output.unlink()
        except OSError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """Print catalog fault versions and schema or live-execution status."""

    parser = argparse.ArgumentParser(
        description="List fault catalog entries without measuring harm."
    )
    parser.add_argument("--catalog", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    catalog = load_fault_catalog(Path(args.catalog))
    for fault in catalog:
        availability = live_fault_available(fault)
        if not fault.representable:
            status = "schema_gap"
        elif availability.available:
            status = "live"
        else:
            status = "live_unavailable"
        print(f"{fault.fault_version}\t{fault.kind}\t{status}")
    return 0
