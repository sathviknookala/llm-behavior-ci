from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Callable, Literal

from llm_behavior_ci.config import (
    EpisodeIdentity,
    RunConfiguration,
    RunIdentity,
    new_episode_identity,
    new_pair_id,
    run_configuration_hash,
)
from llm_behavior_ci.records import (
    EpisodeResult,
    EvaluatorOutcome,
    LocalTaskRef,
    ModelStep,
    PairedResult,
    RecordedError,
    ToolStep,
)
from llm_behavior_ci.runtime.actions import ActionRejected
from llm_behavior_ci.runtime.agent import (
    AgentLoop,
    AgentTurn,
    build_appworld_executor,
    validate_chat_request,
)
from llm_behavior_ci.runtime.clock import wall_now
from llm_behavior_ci.runtime.appworld import (
    AppWorldSession,
    EvaluationResult,
    ToolResult,
)

_COMPLETED = frozenset({"plan_emitted", "agent_stopped", "appworld_completed"})


class EpisodeRejected(ValueError):
    pass


class RuntimeUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeDependencies:
    session_factory: Callable[[str], AppWorldSession]
    agent: AgentLoop
    clock: Callable[[], datetime]


def is_live_runtime(runtime: RuntimeDependencies) -> bool:
    """True only when both the session and every underlying agent are live.

    Imports the live adapters lazily, as ``build_runtime`` does, so this
    stays CPU-testable without AppWorld or smolagents installed. A wrapper
    around several agents (a switching agent used to run reference and
    candidate through one gate call) exposes them through
    ``underlying_agents()``; every one of them must be a real
    ``SmolagentsVLLMAgent`` for the runtime to count as live.
    """

    from llm_behavior_ci.runtime.agent import SmolagentsVLLMAgent
    from llm_behavior_ci.runtime.appworld import LiveAppWorldSession

    if runtime.session_factory is not LiveAppWorldSession:
        return False
    underlying = getattr(runtime.agent, "underlying_agents", None)
    agents = underlying() if callable(underlying) else (runtime.agent,)
    return len(agents) > 0 and all(
        isinstance(agent, SmolagentsVLLMAgent) for agent in agents
    )


def _reject(
    task_id: str,
    config: RunConfiguration,
    mode: str,
    run: RunIdentity,
) -> None:
    del task_id
    if run.configuration_hash != run_configuration_hash(config):
        raise EpisodeRejected("run identity does not match the configuration")
    if run.task_set_hash != config.task.task_set_hash:
        raise EpisodeRejected("run identity does not match the task set")
    if run.git_commit != config.git_commit:
        raise EpisodeRejected("run identity does not match the git commit")
    if run.protocol_hash != config.protocol_hash:
        raise EpisodeRejected("run identity does not match the protocol")
    if mode not in {"plan", "execute"}:
        raise EpisodeRejected("mode must be plan or execute")


def _outcome(result: EvaluationResult) -> EvaluatorOutcome:
    return EvaluatorOutcome(
        success=result.success,
        passed_requirements=result.passed_requirements,
        total_requirements=result.total_requirements,
        difficulty=result.difficulty,
    )


def _observation(result: ToolResult) -> str | None:
    if result.error_message is None:
        return result.output_text
    if result.output_text is not None:
        return result.output_text
    message = result.error_message
    if message.startswith("Execution failed."):
        lines = [line.strip() for line in message.splitlines() if line.strip()]
        if lines:
            return lines[-1]
    return message


def _model_step(index: int, turn: AgentTurn) -> ModelStep:
    return ModelStep(
        index=index,
        prompt_text=turn.prompt_text,
        output_text=turn.output_text,
        top_k_logprobs=turn.top_k_logprobs,
        latency_seconds=turn.latency_seconds,
        started_at=turn.started_at,
    )


def _finish(
    *,
    identity,
    run: RunIdentity,
    task_id: str,
    config: RunConfiguration,
    scenario_id: str | None,
    mode: str,
    started_at: datetime,
    clock: Callable[[], datetime],
    model_steps: list[ModelStep],
    tool_steps: list[ToolStep],
    plan_text: str | None,
    evaluator_outcome: EvaluatorOutcome | None,
    termination_reason: str,
    episode_errors: tuple[RecordedError, ...],
) -> EpisodeResult:
    ended_at = clock()
    status = "completed" if termination_reason in _COMPLETED else "failed"
    return EpisodeResult(
        episode=identity,
        run=run,
        task=LocalTaskRef(
            task_id=task_id,
            scenario_id=scenario_id,
            split=config.task.split,
        ),
        mode=mode,
        execution_seed=config.agent.sampling.seed,
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        model_steps=tuple(model_steps),
        tool_steps=tuple(tool_steps),
        plan_text=plan_text,
        evaluator_outcome=evaluator_outcome,
        termination_reason=termination_reason,
        episode_errors=episode_errors,
        role=None,
    )


def run_episode(
    task_id: str,
    config: RunConfiguration,
    mode: Literal["plan", "execute"],
    *,
    run: RunIdentity,
    runtime: RuntimeDependencies,
    pair_id: str | None = None,
    on_start: Callable[[EpisodeIdentity, RunIdentity], None] | None = None,
    on_step: Callable[[ModelStep | ToolStep], None] | None = None,
    scenario_id: str | None = None,
) -> EpisodeResult:
    """Run one plan or execute episode and return a local ``EpisodeResult``.

    Rejects a run identity that does not match the configuration hash, task
    set, git commit, or protocol. Mints one episode identity, opens a session
    from ``runtime.session_factory``, and always closes that session. Plan
    mode takes one model turn and never executes or evaluates. Execute mode
    enforces the configured step limit and records tool results. It evaluates
    when the agent stops, when ``complete_task`` succeeds, and when the step
    limit is reached. A step-limit evaluation does not change the failed
    status. Unrecoverable runtime and tool failures are not evaluated.
    ``scenario_id`` is stored on the local task reference when the caller
    has one; it is not ground truth.
    """

    _reject(task_id, config, mode, run)
    identity = new_episode_identity(run, pair_id=pair_id)
    if on_start is not None:
        on_start(identity, run)
    session = runtime.session_factory(task_id)
    try:
        started_at = runtime.clock()
        context = session.context()
        runtime.agent.begin(context, config)
        model_steps: list[ModelStep] = []
        tool_steps: list[ToolStep] = []
        if mode == "plan":
            turn = runtime.agent.next_turn(tool_output=None)
            step = _model_step(0, turn)
            model_steps.append(step)
            if on_step is not None:
                on_step(step)
            if turn.output_text.strip() == "":
                return _finish(
                    identity=identity,
                    run=run,
                    task_id=task_id,
                    config=config,
                    scenario_id=scenario_id,
                    mode=mode,
                    started_at=started_at,
                    clock=runtime.clock,
                    model_steps=model_steps,
                    tool_steps=tool_steps,
                    plan_text=None,
                    evaluator_outcome=None,
                    termination_reason="runtime_error",
                    episode_errors=(
                        RecordedError(
                            source="runtime",
                            recoverable=False,
                            message="empty plan",
                            step_index=None,
                        ),
                    ),
                )
            return _finish(
                identity=identity,
                run=run,
                task_id=task_id,
                config=config,
                scenario_id=scenario_id,
                mode=mode,
                started_at=started_at,
                clock=runtime.clock,
                model_steps=model_steps,
                tool_steps=tool_steps,
                plan_text=turn.output_text,
                evaluator_outcome=None,
                termination_reason="plan_emitted",
                episode_errors=(),
            )

        executor = build_appworld_executor(config.agent.action_interface, session.execute)
        model_turns = 0
        tool_output: str | None = None
        next_index = 0
        while True:
            if model_turns >= config.agent.step_limit:
                evaluation = session.evaluate()
                return _finish(
                    identity=identity,
                    run=run,
                    task_id=task_id,
                    config=config,
                    scenario_id=scenario_id,
                    mode=mode,
                    started_at=started_at,
                    clock=runtime.clock,
                    model_steps=model_steps,
                    tool_steps=tool_steps,
                    plan_text=None,
                    evaluator_outcome=_outcome(evaluation),
                    termination_reason="step_limit",
                    episode_errors=(
                        RecordedError(
                            source="step_limit",
                            recoverable=False,
                            message="step limit reached",
                            step_index=None,
                        ),
                    ),
                )
            try:
                turn = runtime.agent.next_turn(tool_output=tool_output)
            except ActionRejected as error:
                return _finish(
                    identity=identity,
                    run=run,
                    task_id=task_id,
                    config=config,
                    scenario_id=scenario_id,
                    mode=mode,
                    started_at=started_at,
                    clock=runtime.clock,
                    model_steps=model_steps,
                    tool_steps=tool_steps,
                    plan_text=None,
                    evaluator_outcome=None,
                    termination_reason="invalid_action",
                    episode_errors=(
                        RecordedError(
                            source="runtime",
                            recoverable=False,
                            message=str(error),
                            step_index=None,
                        ),
                    ),
                )
            step = _model_step(next_index, turn)
            model_steps.append(step)
            next_index += 1
            model_turns += 1
            if on_step is not None:
                on_step(step)
            if turn.rejection is not None:
                tool_output = (
                    "That output was not one apis.<app>.<api>(...) call. "
                    "Emit exactly one call, with keyword arguments, and no other text."
                )
                continue
            if turn.action is None:
                evaluation = session.evaluate()
                return _finish(
                    identity=identity,
                    run=run,
                    task_id=task_id,
                    config=config,
                    scenario_id=scenario_id,
                    mode=mode,
                    started_at=started_at,
                    clock=runtime.clock,
                    model_steps=model_steps,
                    tool_steps=tool_steps,
                    plan_text=None,
                    evaluator_outcome=_outcome(evaluation),
                    termination_reason="agent_stopped",
                    episode_errors=(),
                )
            try:
                tool_started = runtime.clock()
                result = executor(turn.action)
            except Exception as error:
                return _finish(
                    identity=identity,
                    run=run,
                    task_id=task_id,
                    config=config,
                    scenario_id=scenario_id,
                    mode=mode,
                    started_at=started_at,
                    clock=runtime.clock,
                    model_steps=model_steps,
                    tool_steps=tool_steps,
                    plan_text=None,
                    evaluator_outcome=None,
                    termination_reason="runtime_error",
                    episode_errors=(
                        RecordedError(
                            source="runtime",
                            recoverable=False,
                            message=str(error),
                            step_index=None,
                        ),
                    ),
                )
            recorded: RecordedError | None = None
            if result.error_message is not None:
                recorded = RecordedError(
                    source="tool",
                    recoverable=result.recoverable,
                    message=result.error_message,
                    step_index=next_index,
                )
            tool_step = ToolStep(
                index=next_index,
                action=turn.action,
                app_name=turn.app_name,
                api_name=turn.api_name,
                output_text=result.output_text,
                error=recorded,
                latency_seconds=0.0,
                started_at=tool_started,
            )
            tool_steps.append(tool_step)
            next_index += 1
            if on_step is not None:
                on_step(tool_step)
            if result.error_message is not None and not result.recoverable:
                return _finish(
                    identity=identity,
                    run=run,
                    task_id=task_id,
                    config=config,
                    scenario_id=scenario_id,
                    mode=mode,
                    started_at=started_at,
                    clock=runtime.clock,
                    model_steps=model_steps,
                    tool_steps=tool_steps,
                    plan_text=None,
                    evaluator_outcome=None,
                    termination_reason="unrecoverable_tool_error",
                    episode_errors=(),
                )
            if (
                result.error_message is None
                and turn.app_name == "supervisor"
                and turn.api_name == "complete_task"
            ):
                evaluation = session.evaluate()
                return _finish(
                    identity=identity,
                    run=run,
                    task_id=task_id,
                    config=config,
                    scenario_id=scenario_id,
                    mode=mode,
                    started_at=started_at,
                    clock=runtime.clock,
                    model_steps=model_steps,
                    tool_steps=tool_steps,
                    plan_text=None,
                    evaluator_outcome=_outcome(evaluation),
                    termination_reason="appworld_completed",
                    episode_errors=(),
                )
            tool_output = _observation(result)
    finally:
        session.close()


@dataclass(frozen=True)
class EvaluatorDifference:
    """Candidate minus reference on AppWorld evaluator outcomes.

    ``success_difference`` is the candidate success flag minus the reference
    success flag, each as 0 or 1. ``requirement_fraction_difference`` is set
    only when both fractions are defined. A zero requirement total leaves
    that fraction undefined, so it is not treated as zero.
    """

    success_difference: int
    requirement_fraction_difference: float | None


@dataclass(frozen=True)
class PairExecution:
    """Local facts about one paired run that ``PairedResult`` does not store.

    ``execution_order`` is the role order in which ``run_episode`` was called.
    The seeds are the sampling seeds recorded on the episodes and the run
    seeds of the two configurations. ``initial_state_identity`` is the shared
    identity captured before either world was mutated.
    """

    pair_id: str
    execution_order: tuple[str, str]
    reference_episode_id: str
    candidate_episode_id: str
    reference_seed: int
    candidate_seed: int
    reference_run_seed: int
    candidate_run_seed: int
    initial_state_identity: str


_PAIR_EXECUTIONS: dict[str, PairExecution] = {}
_PAIR_LOCK = threading.Lock()


def pair_execution(pair: PairedResult) -> PairExecution:
    """Return the execution record ``run_pair`` stored for this pair."""

    pair_id = pair.reference.episode.pair_id
    if pair_id is None:
        raise EpisodeRejected("pair execution was not recorded")
    with _PAIR_LOCK:
        record = _PAIR_EXECUTIONS.get(pair_id)
    if record is None:
        raise EpisodeRejected("pair execution was not recorded")
    return record


def restore_pair_execution(record: PairExecution) -> None:
    """Store a checkpointed pair execution so ``pair_execution`` can read it.

    Resume replays stored pairs through ``CanaryController.observe`` without
    calling ``run_pair`` again. A different record already stored for the
    same pair id is rejected.
    """

    if not isinstance(record, PairExecution):
        raise EpisodeRejected("pair execution record is required")
    with _PAIR_LOCK:
        existing = _PAIR_EXECUTIONS.get(record.pair_id)
        if existing is not None and existing != record:
            raise EpisodeRejected("pair execution was already recorded")
        _PAIR_EXECUTIONS[record.pair_id] = record


def evaluator_difference(pair: PairedResult) -> EvaluatorDifference | None:
    """Return evaluator differences only when both outcomes exist.

    A missing evaluator outcome is not a success and is not a zero. The
    difference is ``None`` until both episodes have been evaluated.
    """

    reference = pair.reference.evaluator_outcome
    candidate = pair.candidate.evaluator_outcome
    if reference is None or candidate is None:
        return None
    fraction_difference: float | None = None
    if (
        reference.requirement_fraction is not None
        and candidate.requirement_fraction is not None
    ):
        fraction_difference = (
            candidate.requirement_fraction - reference.requirement_fraction
        )
    return EvaluatorDifference(
        success_difference=int(candidate.success) - int(reference.success),
        requirement_fraction_difference=fraction_difference,
    )


def _close_sessions(sessions: list[AppWorldSession]) -> None:
    first_error: Exception | None = None
    closed: list[AppWorldSession] = []
    while sessions:
        session = sessions.pop(0)
        if any(session is item for item in closed):
            continue
        closed.append(session)
        try:
            session.close()
        except Exception as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error


def _initial_state_identity(session: AppWorldSession) -> str:
    reader = getattr(session, "initial_state_identity", None)
    if callable(reader):
        value = reader()
        if not isinstance(value, str) or value == "":
            raise EpisodeRejected("initial state identity is missing")
        return value
    context = session.context()
    payload = "\n".join(
        (context.task_id, context.instruction, context.api_documentation)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _compatible_pair(
    reference_config: RunConfiguration,
    candidate_config: RunConfiguration,
    reference_run: RunIdentity,
    candidate_run: RunIdentity,
) -> None:
    if reference_run.run_id == candidate_run.run_id:
        raise EpisodeRejected("paired runs must have distinct run identities")
    if (
        reference_config.task.task_set_hash != candidate_config.task.task_set_hash
        or reference_config.task.split != candidate_config.task.split
    ):
        raise EpisodeRejected("task sets are not compatible")
    if (
        reference_config.agent.sampling.seed
        != candidate_config.agent.sampling.seed
    ):
        raise EpisodeRejected("pair episodes must share an execution seed")


def _open_worlds(
    task_id: str,
    runtime: RuntimeDependencies,
) -> tuple[AppWorldSession, AppWorldSession, str]:
    first = runtime.session_factory(task_id)
    try:
        second = runtime.session_factory(task_id)
    except Exception:
        _close_sessions([first])
        raise
    try:
        if first is second:
            raise EpisodeRejected("paired worlds must be separate")
        first_identity = _initial_state_identity(first)
        second_identity = _initial_state_identity(second)
        if first_identity != second_identity:
            raise EpisodeRejected("paired worlds do not share an initial state")
    except Exception:
        _close_sessions([first, second])
        raise
    return first, second, first_identity


def build_runtime(
    configuration: RunConfiguration,
    endpoint_url: str,
    *,
    mode: Literal["plan", "execute"] = "execute",
    clock: Callable[[], datetime] | None = None,
) -> RuntimeDependencies:
    """Build runtime dependencies for one configuration and HTTP endpoint.

    Imports the live AppWorld session adapter only when called. Does not
    import vLLM or smolagents; vLLM stays a served HTTP endpoint, never an
    in-process backend, and smolagents stays an optional import so this
    function, like the rest of this module, is CPU-testable without it
    (`CONSTRAINTS.md` keeps the hosted CPU suite smolagents-free).
    ``SmolagentsVLLMAgent`` genuinely subclasses ``smolagents.Model`` when
    the package is present and falls back to a plain object base otherwise;
    either way its control flow, and the version recorded on
    ``configuration.agent.smolagents_version``, are identical. Prompt
    versions, action interface, and chat-expressible serving flags are
    checked before the runtime is returned. Actions still execute through
    ``AppWorldSession.execute``, reached by ``run_episode`` via
    ``build_appworld_executor`` and never by smolagents'
    ``LocalPythonExecutor``.
    """

    if not isinstance(configuration, RunConfiguration):
        raise EpisodeRejected("runtime requires a run configuration")
    if not isinstance(endpoint_url, str) or endpoint_url.strip() == "":
        raise EpisodeRejected("endpoint url is required")
    if mode not in {"plan", "execute"}:
        raise EpisodeRejected("mode must be plan or execute")
    validate_chat_request(configuration)
    from llm_behavior_ci.runtime.agent import SmolagentsVLLMAgent
    from llm_behavior_ci.runtime.appworld import LiveAppWorldSession
    from llm_behavior_ci.runtime.prompts import UnknownPromptVersion, render_system_text

    try:
        render_system_text(
            prompt_version=configuration.agent.prompt.prompt_version,
            plan_format_version=configuration.agent.prompt.plan_format_version,
            thinking_enabled=configuration.agent.prompt.thinking_enabled,
            action_interface=configuration.agent.action_interface,
            mode=mode,
        )
    except UnknownPromptVersion as error:
        raise RuntimeUnavailable(str(error)) from error
    except ValueError as error:
        raise RuntimeUnavailable(str(error)) from error

    agent = SmolagentsVLLMAgent(endpoint_url)
    agent.set_mode(mode)
    return RuntimeDependencies(
        session_factory=LiveAppWorldSession,
        agent=agent,
        clock=clock or wall_now,
    )


def run_pair(
    task_id: str,
    reference_config: RunConfiguration,
    candidate_config: RunConfiguration,
    *,
    reference_run: RunIdentity,
    candidate_run: RunIdentity,
    runtime: RuntimeDependencies,
    mode: Literal["plan", "execute"],
    scenario_id: str | None = None,
    candidate_runtime: RuntimeDependencies | None = None,
    on_start: Callable[[EpisodeIdentity, RunIdentity], None] | None = None,
    on_step: Callable[[ModelStep | ToolStep], None] | None = None,
) -> PairedResult:
    """Run one reference episode and one candidate episode as a pair.

    Mints one pair id and calls ``run_episode`` twice, reference first.
    Both worlds are opened from ``task_id`` before either episode runs.
    The session factory must return a distinct world each time; one world
    is never passed the other's tool output. Run identities and episode
    identities stay distinct. Task-set hash, split, and sampling seed must
    already be compatible with ``PairedResult``.

    When ``candidate_runtime`` is omitted, both episodes use ``runtime``.
    When it is supplied, the reference episode uses ``runtime.agent`` and
    the candidate episode uses ``candidate_runtime.agent``; worlds still
    come from ``runtime.session_factory`` opened before either episode.

    A failed episode is kept on the pair. Missing evaluation is not rewritten
    as success. Evaluator differences are available from
    ``evaluator_difference`` only when both outcomes exist. Ordering, seeds,
    and the initial-state identity are available from ``pair_execution``.
    """

    _reject(task_id, reference_config, mode, reference_run)
    _reject(task_id, candidate_config, mode, candidate_run)
    _compatible_pair(
        reference_config,
        candidate_config,
        reference_run,
        candidate_run,
    )
    pair_id = new_pair_id()
    reference_session, candidate_session, initial_state_identity = _open_worlds(
        task_id,
        runtime,
    )
    pending = [reference_session, candidate_session]

    def factory(requested: str) -> AppWorldSession:
        if requested != task_id:
            raise EpisodeRejected("paired episode requested a different task")
        if not pending:
            raise EpisodeRejected("paired world was reused")
        return pending.pop(0)

    reference_paired = RuntimeDependencies(
        session_factory=factory,
        agent=runtime.agent,
        clock=runtime.clock,
    )
    if candidate_runtime is None:
        candidate_paired = reference_paired
    else:
        candidate_paired = RuntimeDependencies(
            session_factory=factory,
            agent=candidate_runtime.agent,
            clock=candidate_runtime.clock,
        )
    try:
        reference_episode = run_episode(
            task_id,
            reference_config,
            mode,
            run=reference_run,
            runtime=reference_paired,
            pair_id=pair_id,
            on_start=on_start,
            on_step=on_step,
            scenario_id=scenario_id,
        )
        candidate_episode = run_episode(
            task_id,
            candidate_config,
            mode,
            run=candidate_run,
            runtime=candidate_paired,
            pair_id=pair_id,
            on_start=on_start,
            on_step=on_step,
            scenario_id=scenario_id,
        )
    finally:
        _close_sessions(pending)
    pair = PairedResult(
        reference=replace(reference_episode, role="reference"),
        candidate=replace(candidate_episode, role="candidate"),
    )
    record = PairExecution(
        pair_id=pair_id,
        execution_order=("reference", "candidate"),
        reference_episode_id=pair.reference.episode.episode_id,
        candidate_episode_id=pair.candidate.episode.episode_id,
        reference_seed=pair.reference.execution_seed,
        candidate_seed=pair.candidate.execution_seed,
        reference_run_seed=reference_config.run_seed,
        candidate_run_seed=candidate_config.run_seed,
        initial_state_identity=initial_state_identity,
    )
    with _PAIR_LOCK:
        _PAIR_EXECUTIONS[pair_id] = record
    return pair
