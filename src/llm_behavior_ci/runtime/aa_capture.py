"""A/A capture over a supplied finite task stream.

The same configuration is paired with itself. Repetitions replay that
stream. Concurrency does not change task order, membership, or the
recorded offsets. Evaluator disagreement and requirement fractions are
recorded only from evaluator outcomes. A missing outcome is not an
agreement and is not a success.

Plan-scoring inputs are the plan texts and top-k log probabilities from
independent generation. Beside them, plan-mode pairs also record a
teacher-forced top-k KL on one frozen reference plan scored under both
A sides. This module does not apply a KL limit, a margin, or a stopping
rule. Tool-selection homogeneity reuses ``chi_square_homogeneity`` when
the counts meet that function's contract. The p-value is not a decision.

``test_normal`` is rejected here. Hardware fields stay empty unless a
snapshot is taken. A caller-supplied probe is not GPU evidence;
``nvidia-smi`` is. Synthetic captures leave memory and wall time unset.
With no expected-process allowance, any compute process refuses a timed
start. An explicit caller allowance names the baseline server by pid
and/or exact process name; unexpected concurrent GPU work still refuses.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from llm_behavior_ci.config import (
    ConfigError,
    RunConfiguration,
    StreamSettings,
    new_run_identity,
)
from llm_behavior_ci.records import EpisodeResult, PairedResult, TokenLogprob
from llm_behavior_ci.runtime.episode import (
    EpisodeRejected,
    PairExecution,
    RuntimeDependencies,
    RuntimeUnavailable,
    evaluator_difference,
    pair_execution,
    run_pair,
)
from llm_behavior_ci.runtime.scoring import ScoringError, score_top_k
from llm_behavior_ci.stats.chi_square import ChiSquareError, ChiSquareResult, chi_square_homogeneity
from llm_behavior_ci.tasks.selection import SelectionError, TaskSet, verify_task_set
from llm_behavior_ci.tasks.streams import StreamError, TaskArrival, generate_stream

ModeName = Literal["plan", "execute"]


class GpuBusy(RuntimeError):
    """The GPU already has compute work, so a timed capture must not start."""


@dataclass(frozen=True)
class GpuProcess:
    """One compute process reported on the first GPU."""

    pid: int
    name: str | None = None


@dataclass(frozen=True)
class AllowedGpuProcess:
    """Caller-named allowance for one expected baseline compute process.

    At least one of ``pid`` or ``name`` must be set. A name match is exact
    equality with the reported process name, not a substring.
    """

    pid: int | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.pid is None and (self.name is None or self.name == ""):
            raise EpisodeRejected("allowed GPU process needs a pid or a name")
        if self.pid is not None and (
            isinstance(self.pid, bool) or not isinstance(self.pid, int) or self.pid < 1
        ):
            raise EpisodeRejected("allowed GPU pid must be a positive integer")
        if self.name is not None and not isinstance(self.name, str):
            raise EpisodeRejected("allowed GPU process name must be a string")


@dataclass(frozen=True)
class GpuSnapshot:
    """One reading of the first GPU reported by the probe."""

    memory_used_mib: int
    memory_total_mib: int
    process_count: int
    processes: tuple[GpuProcess, ...] = ()


@dataclass(frozen=True)
class ExecutionCost:
    """Cost fields for one capture.

    ``hardware_observed`` is true only for an ``nvidia-smi`` reading.
    A probe can fill memory and wall time for a dry run, and
    ``hardware_observed`` stays false. An unobserved capture leaves the
    numeric fields as ``None`` rather than zero. When hardware is
    observed, ``initial`` and ``final`` keep the raw snapshots, including
    process count and which processes matched the caller allowance.
    """

    hardware_observed: bool
    memory_used_mib: int | None
    memory_total_mib: int | None
    wall_seconds: float | None
    source: str
    initial: GpuSnapshot | None = None
    final: GpuSnapshot | None = None
    allowed_processes: tuple[GpuProcess, ...] = ()


@dataclass(frozen=True)
class ScheduledInput:
    """One replay of a supplied arrival. Offsets are copied, not extended."""

    repetition: int
    arrival_index: int
    task_id: str
    scenario_id: str | None
    stream_seed: int
    scheduled_offset_seconds: float


@dataclass(frozen=True)
class TrajectoryDivergence:
    """First mismatched step and length gap for one pair.

    Execute pairs compare tool actions and record tool-name counts.
    Plan pairs compare model output text and record empty tool counts.
    ``first_divergent_step`` is ``None`` when the sequences are equal.
    A longer sequence diverges at the first extra step.
    """

    first_divergent_step: int | None
    length_difference: int
    reference_tool_counts: tuple[tuple[str, int], ...]
    candidate_tool_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class PlanScoringInputs:
    """Plan text and top-k tables, with no score and no threshold applied.

    Each step is a tuple of positions, and each position is a tuple of
    ``(token_id, logprob)`` pairs. These are independently generated plan
    variation inputs, not the teacher-forced A/A floor. They are not
    full-vocabulary log probabilities.
    """

    reference_plan_text: str | None
    candidate_plan_text: str | None
    reference_steps: tuple[tuple[tuple[tuple[int, float], ...], ...], ...]
    candidate_steps: tuple[tuple[tuple[tuple[int, float], ...], ...], ...]


@dataclass(frozen=True)
class TeacherForcedPlanKL:
    """Teacher-forced top-k plan KL on one frozen reference plan.

    Both A sides score the same ``frozen_plan_text``. ``mean_kl_nats`` is
    top-k at the configuration's ``max_logprobs``, not a full-vocabulary
    claim. Fields stay empty when teacher-force is unavailable. No
    threshold or pass/fail is applied.
    """

    frozen_plan_text: str | None = None
    mean_kl_nats: float | None = None
    position_kl_nats: tuple[float, ...] | None = None
    reference_top_k: tuple[tuple[tuple[int, float], ...], ...] | None = None
    candidate_top_k: tuple[tuple[tuple[int, float], ...], ...] | None = None


@dataclass(frozen=True)
class AAPairRecord:
    """One paired capture on the supplied schedule."""

    schedule: ScheduledInput
    mode: str
    pair: PairedResult
    execution: PairExecution
    evaluator_disagreement: bool | None
    reference_requirement_fraction: float | None
    candidate_requirement_fraction: float | None
    trajectory: TrajectoryDivergence
    plan_scoring_inputs: PlanScoringInputs | None
    teacher_forced_plan_kl: TeacherForcedPlanKL | None = None


@dataclass(frozen=True)
class AACaptureResult:
    """Local A/A capture. Task ids in ``schedule`` stay in the local file."""

    configuration_hash: str
    task_set_hash: str
    repetitions: int
    concurrency: int
    modes: tuple[str, ...]
    schedule: tuple[ScheduledInput, ...]
    records: tuple[AAPairRecord, ...]
    evaluated_pairs: int
    disagreement_count: int
    missing_outcome_count: int
    tool_selection: ChiSquareResult | None
    cost: ExecutionCost


def _positive(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EpisodeRejected(f"{name} must be a positive integer")
    return value


def _modes(modes: Sequence[str]) -> tuple[ModeName, ...]:
    if isinstance(modes, str) or not isinstance(modes, tuple) or not modes:
        raise EpisodeRejected("modes must be a non-empty tuple")
    chosen: list[ModeName] = []
    for mode in modes:
        if mode == "plan" or mode == "execute":
            chosen.append(mode)
        else:
            raise EpisodeRejected("mode must be plan or execute")
    if len(set(chosen)) != len(chosen):
        raise EpisodeRejected("modes contains a duplicate")
    return tuple(chosen)


def repeated_schedule(
    arrivals: Sequence[TaskArrival],
    *,
    repetitions: int,
) -> tuple[ScheduledInput, ...]:
    """Replay ``arrivals`` ``repetitions`` times in arrival order.

    The result does not depend on concurrency. Each repetition copies the
    supplied offsets and stream seed.
    """

    count = _positive(repetitions, "repetitions")
    if isinstance(arrivals, str) or not isinstance(arrivals, Sequence):
        raise EpisodeRejected("arrivals must be a finite sequence")
    if not arrivals:
        raise EpisodeRejected("stream is empty")
    rows: list[ScheduledInput] = []
    for repetition in range(count):
        for arrival in arrivals:
            if not isinstance(arrival, TaskArrival):
                raise EpisodeRejected("stream items must be task arrivals")
            rows.append(
                ScheduledInput(
                    repetition=repetition,
                    arrival_index=arrival.index,
                    task_id=arrival.task_id,
                    scenario_id=arrival.scenario_id,
                    stream_seed=arrival.stream_seed,
                    scheduled_offset_seconds=arrival.scheduled_offset_seconds,
                )
            )
    return tuple(rows)


def _first_divergent(left: Sequence[str], right: Sequence[str]) -> int | None:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    if len(left) != len(right):
        return limit
    return None


def _tool_counts(episode: EpisodeResult) -> tuple[tuple[str, int], ...]:
    counts: Counter[str] = Counter()
    for step in episode.tool_steps:
        name = step.api_name if step.api_name is not None else step.action
        counts[name] += 1
    return tuple(sorted(counts.items()))


def trajectory_divergence(pair: PairedResult) -> TrajectoryDivergence:
    """Compare paired trajectories without applying a threshold."""

    if pair.reference.mode == "plan":
        left = tuple(step.output_text for step in pair.reference.model_steps)
        right = tuple(step.output_text for step in pair.candidate.model_steps)
        left_counts: tuple[tuple[str, int], ...] = ()
        right_counts: tuple[tuple[str, int], ...] = ()
    else:
        left = tuple(step.action for step in pair.reference.tool_steps)
        right = tuple(step.action for step in pair.candidate.tool_steps)
        left_counts = _tool_counts(pair.reference)
        right_counts = _tool_counts(pair.candidate)
    return TrajectoryDivergence(
        first_divergent_step=_first_divergent(left, right),
        length_difference=len(right) - len(left),
        reference_tool_counts=left_counts,
        candidate_tool_counts=right_counts,
    )


def _logprob_steps(
    episode: EpisodeResult,
) -> tuple[tuple[tuple[tuple[int, float], ...], ...], ...]:
    steps = []
    for step in episode.model_steps:
        positions = []
        for position in step.top_k_logprobs:
            positions.append(
                tuple((item.token_id, item.logprob) for item in position)
            )
        steps.append(tuple(positions))
    return tuple(steps)


def plan_scoring_inputs(pair: PairedResult) -> PlanScoringInputs | None:
    """Return plan-scoring inputs for a plan pair and ``None`` otherwise."""

    if pair.reference.mode != "plan":
        return None
    return PlanScoringInputs(
        reference_plan_text=pair.reference.plan_text,
        candidate_plan_text=pair.candidate.plan_text,
        reference_steps=_logprob_steps(pair.reference),
        candidate_steps=_logprob_steps(pair.candidate),
    )


def _matched_messages(
    agent: object,
    context: object,
    config: RunConfiguration,
) -> list[dict[str, str]] | None:
    begin = getattr(agent, "begin", None)
    if not callable(begin):
        return None
    begin(context, config)
    messages_fn = getattr(agent, "messages", None)
    if not callable(messages_fn):
        return None
    built = messages_fn()
    if (
        isinstance(built, list)
        and built
        and all(isinstance(item, dict) for item in built)
    ):
        return [
            {"role": str(item["role"]), "content": str(item["content"])}
            for item in built
        ]
    return None


def _top_k_tables(
    positions: Sequence[Sequence[TokenLogprob]],
) -> tuple[tuple[tuple[int, float], ...], ...]:
    return tuple(
        tuple((item.token_id, item.logprob) for item in position)
        for position in positions
    )


def _logprob_arrays(
    positions: Sequence[Sequence[TokenLogprob]],
) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(item.logprob for item in position) for position in positions)


def teacher_forced_plan_kl(
    *,
    runtime: RuntimeDependencies,
    task_id: str,
    configuration: RunConfiguration,
    frozen_plan_text: str,
) -> TeacherForcedPlanKL:
    """Teacher-force one frozen plan under both identical A sides.

    Reuses the offline-gate pattern: same messages, same plan text, score
    with ``score_top_k``. Missing teacher-force leaves KL fields empty
    rather than falling back to independently generated plans.
    """

    teacher_force = getattr(runtime.agent, "teacher_force_plan", None)
    if not callable(teacher_force):
        return TeacherForcedPlanKL()
    session = runtime.session_factory(task_id)
    try:
        context = session.context()
        messages = _matched_messages(runtime.agent, context, configuration)
        if messages is None:
            return TeacherForcedPlanKL()
        runtime.agent.begin(context, configuration)
        try:
            reference_positions = teacher_force(
                messages=messages,
                plan_text=frozen_plan_text,
            )
        except Exception as error:
            if "unsupported" in str(error).lower():
                return TeacherForcedPlanKL()
            raise
        runtime.agent.begin(context, configuration)
        try:
            candidate_positions = teacher_force(
                messages=messages,
                plan_text=frozen_plan_text,
            )
        except Exception as error:
            if "unsupported" in str(error).lower():
                return TeacherForcedPlanKL()
            raise
    finally:
        session.close()
    if not isinstance(reference_positions, tuple) or not isinstance(
        candidate_positions, tuple
    ):
        return TeacherForcedPlanKL(frozen_plan_text=frozen_plan_text)
    if not reference_positions or not candidate_positions:
        return TeacherForcedPlanKL(frozen_plan_text=frozen_plan_text)
    try:
        scored = score_top_k(
            _logprob_arrays(reference_positions),
            _logprob_arrays(candidate_positions),
        )
    except ScoringError:
        return TeacherForcedPlanKL(frozen_plan_text=frozen_plan_text)
    return TeacherForcedPlanKL(
        frozen_plan_text=frozen_plan_text,
        mean_kl_nats=scored.mean_kl_nats,
        position_kl_nats=scored.position_kl_nats,
        reference_top_k=_top_k_tables(reference_positions),
        candidate_top_k=_top_k_tables(candidate_positions),
    )


def _fraction(episode: EpisodeResult) -> float | None:
    outcome = episode.evaluator_outcome
    if outcome is None:
        return None
    return outcome.requirement_fraction


def _disagreement(pair: PairedResult) -> bool | None:
    difference = evaluator_difference(pair)
    if difference is None:
        return None
    return difference.success_difference != 0


def _record(
    item: ScheduledInput,
    mode: str,
    pair: PairedResult,
    *,
    forced_kl: TeacherForcedPlanKL | None = None,
) -> AAPairRecord:
    return AAPairRecord(
        schedule=item,
        mode=mode,
        pair=pair,
        execution=pair_execution(pair),
        evaluator_disagreement=_disagreement(pair),
        reference_requirement_fraction=_fraction(pair.reference),
        candidate_requirement_fraction=_fraction(pair.candidate),
        trajectory=trajectory_divergence(pair),
        plan_scoring_inputs=plan_scoring_inputs(pair),
        teacher_forced_plan_kl=forced_kl,
    )


def _tool_selection(records: Sequence[AAPairRecord]) -> ChiSquareResult | None:
    counts_left: Counter[str] = Counter()
    counts_right: Counter[str] = Counter()
    for record in records:
        if record.mode != "execute":
            continue
        for name, count in record.trajectory.reference_tool_counts:
            counts_left[name] += count
        for name, count in record.trajectory.candidate_tool_counts:
            counts_right[name] += count
    keys = tuple(sorted(set(counts_left) | set(counts_right)))
    if len(keys) < 2:
        return None
    try:
        return chi_square_homogeneity(
            [counts_left[key] for key in keys],
            [counts_right[key] for key in keys],
        )
    except ChiSquareError:
        return None


def _run_smi(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError as error:
        raise RuntimeUnavailable("nvidia-smi is not available") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeUnavailable("nvidia-smi timed out") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeUnavailable("nvidia-smi failed") from error
    return completed.stdout


def parse_gpu_snapshot(memory: str, processes: str) -> GpuSnapshot:
    """Parse the first GPU from ``nvidia-smi`` CSV text.

    Memory is ``used, total`` in MiB, nounits. Process lines are ``pid`` or
    ``pid, process_name``. A blank process list is idle. Non-numeric process
    lines are ignored. Incomplete text raises ``RuntimeUnavailable`` instead
    of returning zeros.
    """

    lines = [line.strip() for line in memory.splitlines() if line.strip()]
    if not lines:
        raise RuntimeUnavailable("nvidia-smi returned no GPU memory")
    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) != 2:
        raise RuntimeUnavailable("nvidia-smi memory reading is incomplete")
    try:
        used = int(parts[0])
        total = int(parts[1])
    except ValueError as error:
        raise RuntimeUnavailable("nvidia-smi memory reading is incomplete") from error
    if used < 0 or total <= 0:
        raise RuntimeUnavailable("nvidia-smi memory reading is incomplete")
    parsed: list[GpuProcess] = []
    for line in processes.splitlines():
        text = line.strip()
        if not text:
            continue
        columns = [part.strip() for part in text.split(",")]
        if not columns or not columns[0].isdigit():
            continue
        pid = int(columns[0])
        name: str | None = None
        if len(columns) >= 2 and columns[1] and not columns[1].isdigit():
            name = columns[1]
        parsed.append(GpuProcess(pid=pid, name=name))
    return GpuSnapshot(
        memory_used_mib=used,
        memory_total_mib=total,
        process_count=len(parsed),
        processes=tuple(parsed),
    )


def read_nvidia_smi_snapshot() -> GpuSnapshot:
    """Read memory and compute-process count for the first reported GPU.

    This is the only hardware source. A failure raises
    ``RuntimeUnavailable`` and does not invent a zero reading.
    """

    memory = _run_smi(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    processes = _run_smi(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ]
    )
    return parse_gpu_snapshot(memory, processes)


def _process_matches(process: GpuProcess, allowed: AllowedGpuProcess) -> bool:
    if allowed.pid is not None and allowed.name is not None:
        return process.pid == allowed.pid and process.name == allowed.name
    if allowed.pid is not None:
        return process.pid == allowed.pid
    return process.name is not None and process.name == allowed.name


def assert_gpu_processes_allowed(
    snapshot: GpuSnapshot,
    allowed: Sequence[AllowedGpuProcess] = (),
) -> tuple[GpuProcess, ...]:
    """Raise ``GpuBusy`` unless every compute process is explicitly allowed.

    An idle GPU (zero processes) always proceeds. With no allowance, any
    compute process refuses. Allowed does not mean idle: matched processes
    are returned and the snapshot's process count is unchanged. A count
    without process identities cannot be matched and refuses.
    """

    if snapshot.process_count == 0:
        return ()
    if not allowed or not snapshot.processes:
        raise GpuBusy("GPU already has compute processes")
    if len(snapshot.processes) != snapshot.process_count:
        raise GpuBusy("GPU already has compute processes")
    matched: list[GpuProcess] = []
    for process in snapshot.processes:
        if not any(_process_matches(process, spec) for spec in allowed):
            raise GpuBusy("GPU already has compute processes")
        matched.append(process)
    return tuple(matched)


def _snapshot(probe: Callable[[], GpuSnapshot] | None) -> tuple[GpuSnapshot, str]:
    if probe is None:
        return read_nvidia_smi_snapshot(), "nvidia-smi"
    snapshot = probe()
    if (
        not isinstance(snapshot, GpuSnapshot)
        or snapshot.memory_used_mib < 0
        or snapshot.memory_total_mib <= 0
        or snapshot.process_count < 0
    ):
        raise RuntimeUnavailable("GPU probe did not return a snapshot")
    return snapshot, "probe"


def _unobserved() -> ExecutionCost:
    return ExecutionCost(
        hardware_observed=False,
        memory_used_mib=None,
        memory_total_mib=None,
        wall_seconds=None,
        source="not_observed",
    )


def _cost(
    before: GpuSnapshot,
    after: GpuSnapshot,
    wall_seconds: float,
    source: str,
    *,
    allowed_processes: tuple[GpuProcess, ...] = (),
) -> ExecutionCost:
    return ExecutionCost(
        hardware_observed=source == "nvidia-smi",
        memory_used_mib=max(before.memory_used_mib, after.memory_used_mib),
        memory_total_mib=before.memory_total_mib,
        wall_seconds=wall_seconds,
        source=source,
        initial=before,
        final=after,
        allowed_processes=allowed_processes,
    )


def capture_aa(
    configuration: RunConfiguration,
    arrivals: Sequence[TaskArrival],
    *,
    task_set_hash: str,
    repetitions: int,
    concurrency: int,
    modes: tuple[ModeName, ...],
    runtime_factory: Callable[[str], RuntimeDependencies],
    observe_hardware: bool,
    gpu_probe: Callable[[], GpuSnapshot] | None = None,
    allowed_gpu_processes: Sequence[AllowedGpuProcess] = (),
) -> AACaptureResult:
    """Pair one configuration with itself on a supplied stream.

    ``runtime_factory`` receives the mode and must return a fresh runtime
    for that pair. Reference and candidate runs are distinct identities of
    this same configuration. ``observe_hardware`` false leaves cost numbers
    unset and does not shell out to ``nvidia-smi``. When it is true, every
    compute process must be covered by ``allowed_gpu_processes`` or the
    capture raises ``GpuBusy`` before any episode starts.
    """

    if not isinstance(configuration, RunConfiguration):
        raise EpisodeRejected("capture requires a run configuration")
    if configuration.task.split == "test_normal":
        raise EpisodeRejected("test_normal capture is closed")
    if configuration.task.task_set_hash != task_set_hash:
        raise EpisodeRejected("stream task set does not match the configuration")
    chosen = _modes(modes)
    worker_count = _positive(concurrency, "concurrency")
    schedule = repeated_schedule(arrivals, repetitions=repetitions)
    allowed = tuple(allowed_gpu_processes)
    for item in allowed:
        if not isinstance(item, AllowedGpuProcess):
            raise EpisodeRejected("allowed GPU processes must be AllowedGpuProcess")
    before: GpuSnapshot | None = None
    source = "not_observed"
    matched_allowed: tuple[GpuProcess, ...] = ()
    if observe_hardware:
        before, source = _snapshot(gpu_probe)
        matched_allowed = assert_gpu_processes_allowed(before, allowed)
    reference_run = new_run_identity(configuration)
    candidate_run = new_run_identity(configuration)
    jobs: list[tuple[int, ScheduledInput, ModeName]] = []
    for item in schedule:
        for mode in chosen:
            jobs.append((len(jobs), item, mode))
    slots: list[AAPairRecord | None] = [None] * len(jobs)

    def run_job(index: int, item: ScheduledInput, mode: ModeName) -> None:
        runtime = runtime_factory(mode)
        pair = run_pair(
            item.task_id,
            configuration,
            configuration,
            reference_run=reference_run,
            candidate_run=candidate_run,
            runtime=runtime,
            mode=mode,
            scenario_id=item.scenario_id,
        )
        forced_kl: TeacherForcedPlanKL | None = None
        if mode == "plan":
            frozen = pair.reference.plan_text
            if frozen is None or not str(frozen).strip():
                forced_kl = TeacherForcedPlanKL()
            else:
                forced_kl = teacher_forced_plan_kl(
                    runtime=runtime,
                    task_id=item.task_id,
                    configuration=configuration,
                    frozen_plan_text=frozen,
                )
        slots[index] = _record(item, mode, pair, forced_kl=forced_kl)

    started = time.perf_counter()
    if worker_count == 1:
        for index, item, mode in jobs:
            run_job(index, item, mode)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = [
                pool.submit(run_job, index, item, mode)
                for index, item, mode in jobs
            ]
            for future in futures:
                future.result()
    wall_seconds = time.perf_counter() - started
    if before is None:
        cost = _unobserved()
    else:
        after, after_source = _snapshot(gpu_probe)
        if after_source != source:
            raise RuntimeUnavailable("GPU probe source changed during capture")
        cost = _cost(
            before,
            after,
            wall_seconds,
            source,
            allowed_processes=matched_allowed,
        )
    records = tuple(record for record in slots if record is not None)
    if len(records) != len(slots):
        raise EpisodeRejected("capture dropped a paired episode")
    evaluated = tuple(
        record
        for record in records
        if record.mode == "execute" and record.evaluator_disagreement is not None
    )
    missing = tuple(
        record
        for record in records
        if record.mode == "execute" and record.evaluator_disagreement is None
    )
    return AACaptureResult(
        configuration_hash=reference_run.configuration_hash,
        task_set_hash=task_set_hash,
        repetitions=len({item.repetition for item in schedule}),
        concurrency=worker_count,
        modes=chosen,
        schedule=schedule,
        records=records,
        evaluated_pairs=len(evaluated),
        disagreement_count=sum(
            1 for record in evaluated if record.evaluator_disagreement
        ),
        missing_outcome_count=len(missing),
        tool_selection=_tool_selection(records),
        cost=cost,
    )


def format_summary(result: AACaptureResult) -> str:
    """Return counts only. Task ids, plans, and trajectories stay out."""

    lines = [
        f"pairs: {len(result.records)}",
        f"evaluated: {result.evaluated_pairs}",
        f"disagreements: {result.disagreement_count}",
        f"missing_outcomes: {result.missing_outcome_count}",
        f"hardware_observed: {str(result.cost.hardware_observed).lower()}",
        f"hardware_source: {result.cost.source}",
    ]
    if result.cost.hardware_observed:
        lines.append(f"memory_used_mib: {result.cost.memory_used_mib}")
        lines.append(f"wall_seconds: {result.cost.wall_seconds}")
    return "\n".join(lines) + "\n"


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        exporter = getattr(value, "to_dict", None)
        if callable(exporter):
            return exporter()
        return {
            item.name: _jsonable(getattr(value, item.name)) for item in fields(value)
        }
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _under(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    base = root.resolve()
    return resolved == base or base in resolved.parents


def write_local_capture(
    result: AACaptureResult,
    output_path: Path,
    *,
    results_root: Path,
) -> None:
    """Write a local capture. Refuses a path inside ``results_root``."""

    if _under(output_path, results_root):
        raise EpisodeRejected("capture output cannot be written as a public result")
    payload = {"visibility": "local", "capture": _jsonable(result)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def live_runtime_factory(base_url: str) -> Callable[[str], RuntimeDependencies]:
    """Build a runtime that opens one AppWorld world and talks to vLLM.

    Import of the live adapters happens when a pair is created. AppWorld
    itself is imported only when a world opens.
    """

    def factory(mode: str) -> RuntimeDependencies:
        from llm_behavior_ci.runtime.agent import SmolagentsVLLMAgent
        from llm_behavior_ci.runtime.appworld import LiveAppWorldSession

        agent = SmolagentsVLLMAgent(base_url)
        agent.set_mode(mode)
        return RuntimeDependencies(
            session_factory=LiveAppWorldSession,
            agent=agent,
            clock=lambda: datetime.now(timezone.utc),
        )

    return factory


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EpisodeRejected("capture input is not readable JSON") from error


def load_task_set(payload: object) -> TaskSet:
    """Build a task set from a local mapping. The hash is verified by the caller."""

    if not isinstance(payload, dict):
        raise EpisodeRejected("task set must be an object")
    try:
        scenario_ids = tuple(payload["scenario_ids"])
        task_ids = tuple(payload["task_ids"])
        return TaskSet(
            appworld_version=payload["appworld_version"],
            split=payload["split"],
            selection_rule=payload["selection_rule"],
            selection_seed=payload["selection_seed"],
            task_count=payload["task_count"],
            scenario_count=payload["scenario_count"],
            task_ids=task_ids,
            scenario_ids=scenario_ids,
            task_set_hash=payload["task_set_hash"],
        )
    except (KeyError, TypeError, SelectionError) as error:
        raise EpisodeRejected("task set file is incomplete") from error


def default_results_root() -> Path:
    return Path(__file__).resolve().parents[3] / "results"


def main(argv: Sequence[str] | None = None) -> int:
    """Run one A/A capture. Missing arguments exit through argparse."""

    parser = argparse.ArgumentParser(description="Capture paired A/A episodes.")
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--task-set", required=True)
    parser.add_argument("--stream-settings", required=True)
    parser.add_argument("--repetitions", required=True, type=int)
    parser.add_argument("--concurrency", required=True, type=int)
    parser.add_argument("--modes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vllm-base-url", required=True)
    parser.add_argument("--observe-hardware", action="store_true")
    parser.add_argument(
        "--allow-gpu-pid",
        action="append",
        type=int,
        default=[],
        dest="allow_gpu_pids",
    )
    parser.add_argument(
        "--allow-gpu-process-name",
        action="append",
        default=[],
        dest="allow_gpu_process_names",
    )
    parser.add_argument("--results-root", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        configuration = RunConfiguration.from_dict(
            _load_json(Path(args.configuration))
        )
        task_set = load_task_set(_load_json(Path(args.task_set)))
        settings = StreamSettings.from_dict(_load_json(Path(args.stream_settings)))
        verify_task_set(configuration.task, task_set)
        if settings.with_replacement:
            raise EpisodeRejected("capture requires a finite stream")
        modes = _modes(
            tuple(part.strip() for part in str(args.modes).split(",") if part.strip())
        )
        results_root = (
            Path(args.results_root)
            if args.results_root is not None
            else default_results_root()
        )
        output_path = Path(args.output)
        if _under(output_path, results_root):
            raise EpisodeRejected(
                "capture output cannot be written as a public result"
            )
        allowed = tuple(
            [AllowedGpuProcess(pid=pid) for pid in args.allow_gpu_pids]
            + [
                AllowedGpuProcess(name=name)
                for name in args.allow_gpu_process_names
            ]
        )
        arrivals = tuple(generate_stream(task_set, settings))
        snapshot = read_nvidia_smi_snapshot()
        assert_gpu_processes_allowed(snapshot, allowed)
        result = capture_aa(
            configuration,
            arrivals,
            task_set_hash=task_set.task_set_hash,
            repetitions=args.repetitions,
            concurrency=args.concurrency,
            modes=modes,
            runtime_factory=live_runtime_factory(args.vllm_base_url),
            observe_hardware=args.observe_hardware,
            allowed_gpu_processes=allowed,
        )
        write_local_capture(result, output_path, results_root=results_root)
    except (
        EpisodeRejected,
        ConfigError,
        SelectionError,
        StreamError,
        GpuBusy,
        RuntimeUnavailable,
    ) as error:
        print(str(error), file=sys.stderr, flush=True)
        return 1
    print(format_summary(result), end="", flush=True)
    return 0
