from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Literal

from llm_behavior_ci.config import (
    RunConfiguration,
    RunIdentity,
    new_episode_identity,
    run_configuration_hash,
)
from llm_behavior_ci.records import (
    EpisodeResult,
    EvaluatorOutcome,
    LocalTaskRef,
    ModelStep,
    RecordedError,
    ToolStep,
)
from llm_behavior_ci.runtime.agent import AgentLoop, AgentTurn
from llm_behavior_ci.runtime.appworld import (
    AppWorldSession,
    EvaluationResult,
)

_COMPLETED = frozenset({"plan_emitted", "agent_stopped"})


class EpisodeRejected(ValueError):
    pass


class RuntimeUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeDependencies:
    session_factory: Callable[[str], AppWorldSession]
    agent: AgentLoop
    clock: Callable[[], datetime]


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
            scenario_id=None,
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
    on_step: Callable[[ModelStep | ToolStep], None] | None = None,
) -> EpisodeResult:
    """Run one plan or execute episode and return a local ``EpisodeResult``.

    Rejects a run identity that does not match the configuration hash, task
    set, git commit, or protocol. Mints one episode identity, opens a session
    from ``runtime.session_factory``, and always closes that session. Plan
    mode takes one model turn and never executes or evaluates. Execute mode
    enforces the configured step limit, records tool results, and evaluates
    only after the agent stops.
    """

    _reject(task_id, config, mode, run)
    identity = new_episode_identity(run, pair_id=pair_id)
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

        model_turns = 0
        tool_output: str | None = None
        next_index = 0
        while True:
            if model_turns >= config.agent.step_limit:
                return _finish(
                    identity=identity,
                    run=run,
                    task_id=task_id,
                    config=config,
                    mode=mode,
                    started_at=started_at,
                    clock=runtime.clock,
                    model_steps=model_steps,
                    tool_steps=tool_steps,
                    plan_text=None,
                    evaluator_outcome=None,
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
            turn = runtime.agent.next_turn(tool_output=tool_output)
            step = _model_step(next_index, turn)
            model_steps.append(step)
            next_index += 1
            model_turns += 1
            if on_step is not None:
                on_step(step)
            if turn.action is None:
                evaluation = session.evaluate()
                return _finish(
                    identity=identity,
                    run=run,
                    task_id=task_id,
                    config=config,
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
                result = session.execute(turn.action)
            except Exception as error:
                return _finish(
                    identity=identity,
                    run=run,
                    task_id=task_id,
                    config=config,
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
            if result.error_message is not None:
                if result.output_text is not None:
                    tool_output = result.output_text
                else:
                    tool_output = result.error_message
            else:
                tool_output = result.output_text
    finally:
        session.close()
