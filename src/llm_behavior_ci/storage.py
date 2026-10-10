"""Local episode log.

Steps are committed as they arrive under one writer lock. An open episode
is not an EpisodeResult. Interrupted open episodes reload with the steps
already appended. Alert and deployment-decision rows are local only.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, TypeVar

from llm_behavior_ci.config import EpisodeIdentity, RunIdentity
from llm_behavior_ci.lifecycle.validation_artifact import ValidationArtifact
from llm_behavior_ci.records import EpisodeResult, ModelStep, PairedResult, ToolStep, RecordError

_T = TypeVar("_T")
_DUMP = {"sort_keys": True, "separators": (",", ":"), "ensure_ascii": False}


class StorageError(ValueError):
    pass


@dataclass(frozen=True)
class OpenEpisode:
    identity: EpisodeIdentity
    run: RunIdentity
    task_id: str
    steps: tuple[ModelStep | ToolStep, ...]


@dataclass(frozen=True)
class AlertRecord:
    configuration_hash: str
    reference_configuration_hash: str
    signal: str
    slice_name: str
    method: str
    estimate: float
    boundary: float | None
    sample_size: int
    raised_at: datetime
    period_id: str | None = None
    attributed_slices: tuple[str, ...] = ()

    @property
    def incident_key(self) -> tuple[str, str, str, str] | None:
        """``(period, configuration, signal, slice)`` when a period is set.

        One incident alerts once per monitoring period, whichever detector
        method fired first, and the key is read back from SQLite so a
        restarted process with the same period does not alert again.
        """

        if self.period_id is None:
            return None
        return (self.period_id, self.configuration_hash, self.signal, self.slice_name)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "configuration_hash": self.configuration_hash,
            "reference_configuration_hash": self.reference_configuration_hash,
            "signal": self.signal,
            "slice_name": self.slice_name,
            "method": self.method,
            "estimate": self.estimate,
            "boundary": self.boundary,
            "sample_size": self.sample_size,
            "raised_at": self.raised_at.isoformat(),
        }
        if self.period_id is not None:
            payload["period_id"] = self.period_id
        if self.attributed_slices:
            payload["attributed_slices"] = list(self.attributed_slices)
        return payload

    @classmethod
    def from_dict(cls, payload: object) -> AlertRecord:
        if not isinstance(payload, dict):
            raise StorageError("alert record must be an object")
        raised_at = payload.get("raised_at")
        if not isinstance(raised_at, str):
            raise StorageError("alert record raised_at is invalid")
        try:
            parsed = datetime.fromisoformat(raised_at)
        except ValueError as error:
            raise StorageError("alert record raised_at is invalid") from error
        boundary = payload.get("boundary")
        if boundary is not None and (
            isinstance(boundary, bool) or not isinstance(boundary, (int, float))
        ):
            raise StorageError("alert record boundary is invalid")
        estimate = payload.get("estimate")
        sample_size = payload.get("sample_size")
        if isinstance(estimate, bool) or not isinstance(estimate, (int, float)):
            raise StorageError("alert record estimate is invalid")
        if isinstance(sample_size, bool) or not isinstance(sample_size, int):
            raise StorageError("alert record sample_size is invalid")
        for key in (
            "configuration_hash",
            "reference_configuration_hash",
            "signal",
            "slice_name",
            "method",
        ):
            value = payload.get(key)
            if not isinstance(value, str) or value == "":
                raise StorageError(f"alert record {key} is invalid")
        period_id = payload.get("period_id")
        if period_id is not None and (not isinstance(period_id, str) or period_id == ""):
            raise StorageError("alert record period_id is invalid")
        attributed = payload.get("attributed_slices", [])
        if not isinstance(attributed, list) or any(
            not isinstance(item, str) or item == "" for item in attributed
        ):
            raise StorageError("alert record attributed_slices is invalid")
        return cls(
            configuration_hash=str(payload["configuration_hash"]),
            reference_configuration_hash=str(
                payload["reference_configuration_hash"]
            ),
            signal=str(payload["signal"]),
            slice_name=str(payload["slice_name"]),
            method=str(payload["method"]),
            estimate=float(estimate),
            boundary=None if boundary is None else float(boundary),
            sample_size=sample_size,
            raised_at=parsed,
            period_id=period_id,
            attributed_slices=tuple(attributed),
        )


@dataclass(frozen=True)
class DeploymentDecisionRecord:
    """A local record of one deployment-lifecycle decision.

    ``evidence_artifact_id`` is the ``ValidationArtifact.artifact_id`` that
    authorized the candidate this decision concerns, when the decision has
    one (an admission decision always does; a canary promote/rollback
    decision may not, since ``lifecycle.canary`` is not extended here to
    thread it through). ``None`` means no artifact backs this row, not
    that verification was skipped for a row that needed it.
    """

    configuration_hash: str
    reference_configuration_hash: str
    signal: str
    slice_name: str
    decision: str
    method: str
    estimate: float
    boundary: float | None
    sample_size: int
    decided_at: datetime
    evidence_artifact_id: str | None = None
    evidence_source: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "configuration_hash": self.configuration_hash,
            "reference_configuration_hash": self.reference_configuration_hash,
            "signal": self.signal,
            "slice_name": self.slice_name,
            "decision": self.decision,
            "method": self.method,
            "estimate": self.estimate,
            "boundary": self.boundary,
            "sample_size": self.sample_size,
            "decided_at": self.decided_at.isoformat(),
            "evidence_artifact_id": self.evidence_artifact_id,
            "evidence_source": self.evidence_source,
        }

    @classmethod
    def from_dict(cls, payload: object) -> DeploymentDecisionRecord:
        if not isinstance(payload, dict):
            raise StorageError("deployment decision must be an object")
        decided_at = payload.get("decided_at")
        if not isinstance(decided_at, str):
            raise StorageError("deployment decision decided_at is invalid")
        try:
            parsed = datetime.fromisoformat(decided_at)
        except ValueError as error:
            raise StorageError("deployment decision decided_at is invalid") from error
        boundary = payload.get("boundary")
        if boundary is not None and (
            isinstance(boundary, bool) or not isinstance(boundary, (int, float))
        ):
            raise StorageError("deployment decision boundary is invalid")
        estimate = payload.get("estimate")
        sample_size = payload.get("sample_size")
        if isinstance(estimate, bool) or not isinstance(estimate, (int, float)):
            raise StorageError("deployment decision estimate is invalid")
        if isinstance(sample_size, bool) or not isinstance(sample_size, int):
            raise StorageError("deployment decision sample_size is invalid")
        for key in (
            "configuration_hash",
            "reference_configuration_hash",
            "signal",
            "slice_name",
            "decision",
            "method",
        ):
            value = payload.get(key)
            if not isinstance(value, str) or value == "":
                raise StorageError(f"deployment decision {key} is invalid")
        evidence_artifact_id = payload.get("evidence_artifact_id")
        if evidence_artifact_id is not None and (
            not isinstance(evidence_artifact_id, str) or evidence_artifact_id == ""
        ):
            raise StorageError("deployment decision evidence_artifact_id is invalid")
        evidence_source = payload.get("evidence_source")
        if evidence_source is not None and (
            not isinstance(evidence_source, str) or evidence_source == ""
        ):
            raise StorageError("deployment decision evidence_source is invalid")
        return cls(
            configuration_hash=str(payload["configuration_hash"]),
            reference_configuration_hash=str(
                payload["reference_configuration_hash"]
            ),
            signal=str(payload["signal"]),
            slice_name=str(payload["slice_name"]),
            decision=str(payload["decision"]),
            method=str(payload["method"]),
            estimate=float(estimate),
            boundary=None if boundary is None else float(boundary),
            sample_size=sample_size,
            decided_at=parsed,
            evidence_artifact_id=evidence_artifact_id,
            evidence_source=evidence_source,
        )


@dataclass(frozen=True)
class MonitorMetadataRecord:
    """The caller-supplied metadata behind one persisted monitor observation.

    Paired with the already-persisted ``EpisodeResult`` for ``episode_id``,
    this is everything ``lifecycle.monitoring.normalize_episode`` and the
    typed observation builders need to reconstruct the exact detector
    input a live ``ProductionMonitor.update_from_episode`` call once used,
    for one ``signal`` (a scalar ``MONITOR_SIGNALS`` member, or
    ``tool_selection``/``task_mix``).
    """

    episode_id: str
    signal: str
    completion_index: int
    difficulty: int | None
    task_mix: str | None
    slice_id: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "signal": self.signal,
            "completion_index": self.completion_index,
            "difficulty": self.difficulty,
            "task_mix": self.task_mix,
            "slice_id": self.slice_id,
        }

    @classmethod
    def from_dict(cls, payload: object) -> MonitorMetadataRecord:
        if not isinstance(payload, dict):
            raise StorageError("monitor metadata record must be an object")
        episode_id = payload.get("episode_id")
        signal = payload.get("signal")
        completion_index = payload.get("completion_index")
        if not isinstance(episode_id, str) or episode_id == "":
            raise StorageError("monitor metadata record episode_id is invalid")
        if not isinstance(signal, str) or signal == "":
            raise StorageError("monitor metadata record signal is invalid")
        if isinstance(completion_index, bool) or not isinstance(
            completion_index, int
        ):
            raise StorageError(
                "monitor metadata record completion_index is invalid"
            )
        difficulty = payload.get("difficulty")
        if difficulty is not None and (
            isinstance(difficulty, bool) or not isinstance(difficulty, int)
        ):
            raise StorageError("monitor metadata record difficulty is invalid")
        task_mix = payload.get("task_mix")
        if task_mix is not None and not isinstance(task_mix, str):
            raise StorageError("monitor metadata record task_mix is invalid")
        slice_id = payload.get("slice_id")
        if slice_id is not None and not isinstance(slice_id, str):
            raise StorageError("monitor metadata record slice_id is invalid")
        return cls(
            episode_id=episode_id,
            signal=signal,
            completion_index=completion_index,
            difficulty=difficulty,
            task_mix=task_mix,
            slice_id=slice_id,
        )


def _dumps(payload: object) -> str:
    return json.dumps(payload, **_DUMP)


class EpisodeStore:
    """SQLite episode log for one writer.

    All mutating and reading methods take the process-local writer lock.
    Concurrent writers are unsupported; one thread should own the store.
    """

    def __init__(self, path: Path) -> None:
        if not path.parent.is_dir():
            raise StorageError("parent directory does not exist")
        try:
            connection = sqlite3.connect(
                path,
                check_same_thread=True,
                isolation_level="",
            )
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                  episode_id TEXT PRIMARY KEY,
                  run_id TEXT NOT NULL,
                  task_id TEXT NOT NULL,
                  state TEXT NOT NULL,
                  identity_json TEXT NOT NULL,
                  run_json TEXT NOT NULL,
                  result_json TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS steps (
                  episode_id TEXT NOT NULL,
                  step_index INTEGER NOT NULL,
                  kind TEXT NOT NULL,
                  step_json TEXT NOT NULL,
                  PRIMARY KEY (episode_id, step_index)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS pairs (
                  pair_id TEXT NOT NULL PRIMARY KEY,
                  pair_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                  alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  signal TEXT NOT NULL,
                  slice_name TEXT NOT NULL,
                  method TEXT NOT NULL,
                  raised_at TEXT NOT NULL,
                  alert_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS deployment_decisions (
                  decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  decided_at TEXT NOT NULL,
                  decision_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS validation_artifacts (
                  artifact_id TEXT NOT NULL PRIMARY KEY,
                  created_at TEXT NOT NULL,
                  artifact_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS monitor_metadata (
                  episode_id TEXT NOT NULL,
                  signal TEXT NOT NULL,
                  metadata_json TEXT NOT NULL,
                  PRIMARY KEY (episode_id, signal)
                )
                """
            )
            connection.commit()
        except sqlite3.Error as error:
            raise StorageError("open failed") from error
        self._connection: sqlite3.Connection | None = connection
        self._lock = threading.Lock()

    def _run(self, operation: str, callback: Callable[[sqlite3.Connection], _T]) -> _T:
        with self._lock:
            connection = self._connection
            if connection is None:
                raise StorageError(f"{operation} failed")
            try:
                return callback(connection)
            except StorageError:
                raise
            except sqlite3.Error as error:
                raise StorageError(f"{operation} failed") from error

    def start_episode(
        self,
        identity: EpisodeIdentity,
        run: RunIdentity,
        task_id: str,
    ) -> None:
        if identity.run_id != run.run_id:
            raise StorageError("start_episode failed")
        if not isinstance(task_id, str) or task_id == "":
            raise StorageError("start_episode failed")

        def write(connection: sqlite3.Connection) -> None:
            connection.execute(
                """
                INSERT INTO episodes (
                  episode_id, run_id, task_id, state,
                  identity_json, run_json, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identity.episode_id,
                    run.run_id,
                    task_id,
                    "open",
                    _dumps(identity.to_dict()),
                    _dumps(run.to_dict()),
                    None,
                ),
            )
            connection.commit()

        self._run("start_episode", write)

    def append_step(self, episode_id: str, step: ModelStep | ToolStep) -> None:
        if isinstance(step, ModelStep):
            kind = "model"
        elif isinstance(step, ToolStep):
            kind = "tool"
        else:
            raise StorageError("append_step failed")

        def write(connection: sqlite3.Connection) -> None:
            row = connection.execute(
                "SELECT state FROM episodes WHERE episode_id = ?",
                (episode_id,),
            ).fetchone()
            if row is None or row[0] != "open":
                raise StorageError("append_step failed")
            maximum = connection.execute(
                "SELECT MAX(step_index) FROM steps WHERE episode_id = ?",
                (episode_id,),
            ).fetchone()
            next_index = 0 if maximum[0] is None else maximum[0] + 1
            if step.index != next_index:
                raise StorageError("append_step failed")
            connection.execute(
                """
                INSERT INTO steps (episode_id, step_index, kind, step_json)
                VALUES (?, ?, ?, ?)
                """,
                (episode_id, step.index, kind, _dumps(step.to_dict())),
            )
            connection.commit()

        self._run("append_step", write)

    def finish_episode(self, episode: EpisodeResult) -> None:
        if not isinstance(episode, EpisodeResult):
            raise StorageError("finish_episode failed")
        episode_id = episode.episode.episode_id

        def write(connection: sqlite3.Connection) -> None:
            row = connection.execute(
                "SELECT state FROM episodes WHERE episode_id = ?",
                (episode_id,),
            ).fetchone()
            if row is None or row[0] != "open":
                raise StorageError("finish_episode failed")
            stored = connection.execute(
                """
                SELECT step_json FROM steps
                WHERE episode_id = ?
                ORDER BY step_index
                """,
                (episode_id,),
            ).fetchall()
            stored_payloads = [json.loads(item[0]) for item in stored]
            expected = [
                step.to_dict()
                for step in sorted(
                    (*episode.model_steps, *episode.tool_steps),
                    key=lambda item: item.index,
                )
            ]
            if stored_payloads != expected:
                raise StorageError("finish_episode failed")
            connection.execute(
                """
                UPDATE episodes
                SET state = ?, result_json = ?
                WHERE episode_id = ?
                """,
                ("finished", _dumps(episode.to_dict()), episode_id),
            )
            connection.commit()

        self._run("finish_episode", write)

    def load_episode(self, episode_id: str) -> EpisodeResult:
        def read(connection: sqlite3.Connection) -> EpisodeResult:
            row = connection.execute(
                "SELECT state, result_json FROM episodes WHERE episode_id = ?",
                (episode_id,),
            ).fetchone()
            if row is None or row[0] != "finished" or row[1] is None:
                raise StorageError("load_episode failed")
            return EpisodeResult.from_dict(json.loads(row[1]))

        return self._run("load_episode", read)

    def load_finished_episodes(self) -> tuple[EpisodeResult, ...]:
        def read(connection: sqlite3.Connection) -> tuple[EpisodeResult, ...]:
            rows = connection.execute(
                "SELECT result_json FROM episodes WHERE state = 'finished' "
                "AND result_json IS NOT NULL ORDER BY rowid"
            ).fetchall()
            return tuple(EpisodeResult.from_dict(json.loads(row[0])) for row in rows)

        return self._run("load_finished_episodes", read)

    def load_open_episode(self, episode_id: str) -> OpenEpisode:
        def read(connection: sqlite3.Connection) -> OpenEpisode:
            row = connection.execute(
                """
                SELECT state, identity_json, run_json, task_id
                FROM episodes
                WHERE episode_id = ?
                """,
                (episode_id,),
            ).fetchone()
            if row is None or row[0] != "open":
                raise StorageError("load_open_episode failed")
            identity = EpisodeIdentity.from_dict(json.loads(row[1]))
            run = RunIdentity.from_dict(json.loads(row[2]))
            step_rows = connection.execute(
                """
                SELECT kind, step_json FROM steps
                WHERE episode_id = ?
                ORDER BY step_index
                """,
                (episode_id,),
            ).fetchall()
            steps: list[ModelStep | ToolStep] = []
            for kind, step_json in step_rows:
                payload = json.loads(step_json)
                if kind == "model":
                    steps.append(ModelStep.from_dict(payload))
                elif kind == "tool":
                    steps.append(ToolStep.from_dict(payload))
                else:
                    raise StorageError("load_open_episode failed")
            return OpenEpisode(
                identity=identity,
                run=run,
                task_id=row[3],
                steps=tuple(steps),
            )

        return self._run("load_open_episode", read)

    def append_pair(self, pair: PairedResult) -> None:
        if not isinstance(pair, PairedResult):
            raise StorageError("append_pair failed")
        pair_id = pair.reference.episode.pair_id
        if pair_id is None:
            raise StorageError("append_pair failed")

        def write(connection: sqlite3.Connection) -> None:
            connection.execute(
                "INSERT INTO pairs (pair_id, pair_json) VALUES (?, ?)",
                (pair_id, _dumps(pair.to_dict())),
            )
            connection.commit()

        self._run("append_pair", write)

    def load_pair(self, pair_id: str) -> PairedResult:
        def read(connection: sqlite3.Connection) -> PairedResult:
            row = connection.execute(
                "SELECT pair_json FROM pairs WHERE pair_id = ?",
                (pair_id,),
            ).fetchone()
            if row is None:
                raise StorageError("load_pair failed")
            return PairedResult.from_dict(json.loads(row[0]))

        return self._run("load_pair", read)

    def append_alert(
        self,
        alert: AlertRecord,
        *,
        dedup_seconds: float,
    ) -> AlertRecord:
        record, _inserted = self.append_alert_with_status(
            alert,
            dedup_seconds=dedup_seconds,
        )
        return record

    def append_alert_with_status(
        self,
        alert: AlertRecord,
        *,
        dedup_seconds: float,
    ) -> tuple[AlertRecord, bool]:
        if not isinstance(alert, AlertRecord):
            raise StorageError("append_alert failed")
        if isinstance(dedup_seconds, bool) or not isinstance(
            dedup_seconds, (int, float)
        ):
            raise StorageError("append_alert failed")
        window = float(dedup_seconds)
        if window != window or window < 0.0:
            raise StorageError("append_alert failed")

        def write(connection: sqlite3.Connection) -> tuple[AlertRecord, bool]:
            incident = alert.incident_key
            if incident is not None:
                rows = connection.execute(
                    """
                    SELECT alert_json FROM alerts
                    WHERE signal = ? AND slice_name = ?
                    ORDER BY alert_id
                    """,
                    (alert.signal, alert.slice_name),
                ).fetchall()
                for (payload,) in rows:
                    existing = AlertRecord.from_dict(json.loads(payload))
                    if existing.incident_key == incident:
                        return existing, False
            elif window > 0.0:
                rows = connection.execute(
                    """
                    SELECT alert_json FROM alerts
                    WHERE signal = ? AND slice_name = ? AND method = ?
                    ORDER BY alert_id DESC
                    """,
                    (alert.signal, alert.slice_name, alert.method),
                ).fetchall()
                for (payload,) in rows:
                    existing = AlertRecord.from_dict(json.loads(payload))
                    delta = (alert.raised_at - existing.raised_at).total_seconds()
                    if 0.0 <= delta < window:
                        return existing, False
            connection.execute(
                """
                INSERT INTO alerts (
                  signal, slice_name, method, raised_at, alert_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    alert.signal,
                    alert.slice_name,
                    alert.method,
                    alert.raised_at.isoformat(),
                    _dumps(alert.to_dict()),
                ),
            )
            connection.commit()
            return alert, True

        return self._run("append_alert", write)

    def load_alerts(
        self,
        *,
        signal: str | None = None,
    ) -> tuple[AlertRecord, ...]:
        def read(connection: sqlite3.Connection) -> tuple[AlertRecord, ...]:
            if signal is None:
                rows = connection.execute(
                    """
                    SELECT alert_json FROM alerts
                    ORDER BY alert_id
                    """
                ).fetchall()
            else:
                if not isinstance(signal, str) or signal == "":
                    raise StorageError("load_alerts failed")
                rows = connection.execute(
                    """
                    SELECT alert_json FROM alerts
                    WHERE signal = ?
                    ORDER BY alert_id
                    """,
                    (signal,),
                ).fetchall()
            return tuple(
                AlertRecord.from_dict(json.loads(payload)) for (payload,) in rows
            )

        return self._run("load_alerts", read)

    def append_deployment_decision(
        self,
        decision: DeploymentDecisionRecord,
    ) -> DeploymentDecisionRecord:
        if not isinstance(decision, DeploymentDecisionRecord):
            raise StorageError("append_deployment_decision failed")

        def write(connection: sqlite3.Connection) -> DeploymentDecisionRecord:
            connection.execute(
                """
                INSERT INTO deployment_decisions (decided_at, decision_json)
                VALUES (?, ?)
                """,
                (decision.decided_at.isoformat(), _dumps(decision.to_dict())),
            )
            connection.commit()
            return decision

        return self._run("append_deployment_decision", write)

    def load_deployment_decisions(self) -> tuple[DeploymentDecisionRecord, ...]:
        def read(
            connection: sqlite3.Connection,
        ) -> tuple[DeploymentDecisionRecord, ...]:
            rows = connection.execute(
                """
                SELECT decision_json FROM deployment_decisions
                ORDER BY decision_id
                """
            ).fetchall()
            return tuple(
                DeploymentDecisionRecord.from_dict(json.loads(payload))
                for (payload,) in rows
            )

        return self._run("load_deployment_decisions", read)

    def append_validation_artifact(
        self,
        artifact: ValidationArtifact,
    ) -> ValidationArtifact:
        """Persist one gate-emitted artifact, keyed by its content hash.

        Content-addressed, so a second call with an artifact that hashes
        to an id already in the store is a no-op rather than a conflict:
        the row already holds this exact content. This is what lets
        candidate admission trust an ``artifact_id`` handed to it over the
        wire: the content behind that id is whatever a real
        ``run_offline_gate`` call wrote here, never whatever the admission
        caller separately claims.
        """

        if not isinstance(artifact, ValidationArtifact):
            raise StorageError("append_validation_artifact failed")
        artifact_id = artifact.artifact_id

        def write(connection: sqlite3.Connection) -> ValidationArtifact:
            connection.execute(
                """
                INSERT INTO validation_artifacts (
                  artifact_id, created_at, artifact_json
                ) VALUES (?, ?, ?)
                ON CONFLICT(artifact_id) DO NOTHING
                """,
                (
                    artifact_id,
                    artifact.created_at.isoformat(),
                    _dumps(artifact.to_dict()),
                ),
            )
            connection.commit()
            return artifact

        return self._run("append_validation_artifact", write)

    def load_validation_artifact_ids(self) -> tuple[str, ...]:
        def read(connection: sqlite3.Connection) -> tuple[str, ...]:
            rows = connection.execute(
                "SELECT artifact_id FROM validation_artifacts ORDER BY artifact_id"
            ).fetchall()
            return tuple(artifact_id for (artifact_id,) in rows)

        return self._run("load_validation_artifact_ids", read)

    def load_validation_artifact(
        self,
        artifact_id: str,
    ) -> ValidationArtifact | None:
        """Look up a persisted artifact by content hash, or ``None``.

        A missing, malformed, or tampered row (one whose recomputed
        ``artifact_id`` disagrees with the row it was stored under) is
        rejected the same way: ``None``, so a caller cannot distinguish
        "never written" from "corrupt" and treat the latter as an
        invitation to fall back to a caller-supplied document instead.
        """

        if not isinstance(artifact_id, str) or artifact_id == "":
            raise StorageError("load_validation_artifact failed")

        def read(connection: sqlite3.Connection) -> ValidationArtifact | None:
            row = connection.execute(
                """
                SELECT artifact_json FROM validation_artifacts
                WHERE artifact_id = ?
                """,
                (artifact_id,),
            ).fetchone()
            if row is None:
                return None
            try:
                artifact = ValidationArtifact.from_dict(json.loads(row[0]))
            except (RecordError, json.JSONDecodeError):
                return None
            if artifact.artifact_id != artifact_id:
                return None
            return artifact

        return self._run("load_validation_artifact", read)

    def append_monitor_metadata(self, record: MonitorMetadataRecord) -> None:
        """Persist one observation's metadata, keyed by episode and signal.

        Content-addressed by ``(episode_id, signal)`` rather than
        auto-incremented: a signal is fed from one episode at most once
        (``ProductionMonitor`` itself rejects a repeated episode/signal
        pair), so a second call with the same key is a no-op replay of the
        same content rather than a conflicting overwrite.
        """

        if not isinstance(record, MonitorMetadataRecord):
            raise StorageError("append_monitor_metadata failed")

        def write(connection: sqlite3.Connection) -> None:
            connection.execute(
                """
                INSERT INTO monitor_metadata (episode_id, signal, metadata_json)
                VALUES (?, ?, ?)
                ON CONFLICT(episode_id, signal) DO NOTHING
                """,
                (record.episode_id, record.signal, _dumps(record.to_dict())),
            )
            connection.commit()

        self._run("append_monitor_metadata", write)

    def load_monitor_metadata(
        self,
        episode_id: str,
        signal: str,
    ) -> MonitorMetadataRecord | None:
        if not isinstance(episode_id, str) or episode_id == "":
            raise StorageError("load_monitor_metadata failed")
        if not isinstance(signal, str) or signal == "":
            raise StorageError("load_monitor_metadata failed")

        def read(connection: sqlite3.Connection) -> MonitorMetadataRecord | None:
            row = connection.execute(
                """
                SELECT metadata_json FROM monitor_metadata
                WHERE episode_id = ? AND signal = ?
                """,
                (episode_id, signal),
            ).fetchone()
            if row is None:
                return None
            return MonitorMetadataRecord.from_dict(json.loads(row[0]))

        return self._run("load_monitor_metadata", read)

    def load_monitor_metadata_for_episode(
        self,
        episode_id: str,
    ) -> tuple[MonitorMetadataRecord, ...]:
        if not isinstance(episode_id, str) or episode_id == "":
            raise StorageError("load_monitor_metadata_for_episode failed")

        def read(
            connection: sqlite3.Connection,
        ) -> tuple[MonitorMetadataRecord, ...]:
            rows = connection.execute(
                """
                SELECT metadata_json FROM monitor_metadata
                WHERE episode_id = ?
                ORDER BY signal
                """,
                (episode_id,),
            ).fetchall()
            return tuple(
                MonitorMetadataRecord.from_dict(json.loads(payload))
                for (payload,) in rows
            )

        return self._run("load_monitor_metadata_for_episode", read)

    def close(self) -> None:
        with self._lock:
            connection = self._connection
            if connection is None:
                return
            self._connection = None
            try:
                connection.close()
            except sqlite3.Error as error:
                raise StorageError("close failed") from error
