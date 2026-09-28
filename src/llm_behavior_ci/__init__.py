"""Shared contracts for the tool-using agent lifecycle.

Run configuration and identity live in ``config``. Episode, pair, evidence,
and decision records live in ``records``. The paired bootstrap, next-token
KL, and MMD implementations live in ``stats`` and are re-exported here.
"""

from llm_behavior_ci.config import (
    MONITOR_SIGNALS,
    SPLITS,
    CanarySettings,
    EpisodeIdentity,
    GateSettings,
    MonitorSettings,
    RunConfiguration,
    RunIdentity,
    StoppingRule,
    StreamSettings,
    new_episode_identity,
    new_pair_id,
    new_run_identity,
    run_configuration_hash,
)
from llm_behavior_ci.records import (
    PROTECTED_FIELDS,
    AggregateRecord,
    EpisodeResult,
    EvaluatorOutcome,
    LifecycleDecision,
    LocalTaskRef,
    ModelStep,
    MonitorObservation,
    PairedResult,
    RecordError,
    RecordedError,
    StatisticalEvidence,
    ToolStep,
    assert_public_payload,
    public_record_dict,
)
from llm_behavior_ci.stats.bootstrap import (
    PairedBootstrapError,
    PairedBootstrapResult,
    paired_bootstrap,
)
from llm_behavior_ci.stats.kl import NextTokenKLError, NextTokenKLResult, next_token_kl
from llm_behavior_ci.stats.mmd import MMDError, MMDResult, mmd_permutation_test

__all__ = [
    "MONITOR_SIGNALS",
    "PROTECTED_FIELDS",
    "SPLITS",
    "AggregateRecord",
    "CanarySettings",
    "EpisodeIdentity",
    "EpisodeResult",
    "EvaluatorOutcome",
    "GateSettings",
    "LifecycleDecision",
    "LocalTaskRef",
    "MMDError",
    "MMDResult",
    "ModelStep",
    "MonitorObservation",
    "MonitorSettings",
    "NextTokenKLError",
    "NextTokenKLResult",
    "PairedBootstrapError",
    "PairedBootstrapResult",
    "PairedResult",
    "RecordError",
    "RecordedError",
    "RunConfiguration",
    "RunIdentity",
    "StatisticalEvidence",
    "StoppingRule",
    "StreamSettings",
    "ToolStep",
    "assert_public_payload",
    "mmd_permutation_test",
    "new_episode_identity",
    "new_pair_id",
    "new_run_identity",
    "next_token_kl",
    "paired_bootstrap",
    "public_record_dict",
    "run_configuration_hash",
]
