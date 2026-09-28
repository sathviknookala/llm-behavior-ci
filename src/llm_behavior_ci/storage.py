"""Local episode log.

Steps are committed as they arrive. An open episode is not an EpisodeResult.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

from llm_behavior_ci.config import EpisodeIdentity, RunIdentity
from llm_behavior_ci.records import EpisodeResult, ModelStep, PairedResult, ToolStep

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


def _dumps(payload: object) -> str:
    return json.dumps(payload, **_DUMP)


class EpisodeStore:
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
            connection.commit()
        except sqlite3.Error as error:
            raise StorageError("open failed") from error
        self._connection: sqlite3.Connection | None = connection

    def _run(self, operation: str, callback: Callable[[sqlite3.Connection], _T]) -> _T:
        connection = self._connection
        if connection is None:
            raise StorageError(f"{operation} failed")
        try:
            return callback(connection)
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

    def close(self) -> None:
        connection = self._connection
        if connection is None:
            return
        self._connection = None
        try:
            connection.close()
        except sqlite3.Error as error:
            raise StorageError("close failed") from error
