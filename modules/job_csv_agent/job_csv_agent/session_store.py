from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from hr_agent_contracts import ActorContext

from .agent import AgentState, initial_state
from .governance import AuthorizationError, VersionConflictError


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class SqliteSessionStore:
    """Durable optimistic-concurrency session store suitable for multiple workers."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_sessions (
                    session_id TEXT PRIMARY KEY,
                    company_id TEXT NOT NULL,
                    operator_id TEXT NOT NULL,
                    state_version INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def create(self, actor: ActorContext) -> tuple[str, AgentState]:
        state = initial_state()
        state["company_id"] = actor.company_id
        state["operator_id"] = actor.operator_id
        state["roles"] = list(actor.roles)
        session_id = uuid.uuid4().hex
        state["session_id"] = session_id
        now = _now()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO agent_sessions VALUES (?, ?, ?, 0, ?, ?, ?)",
                (session_id, actor.company_id, actor.operator_id,
                 json.dumps(state, ensure_ascii=False), now, now),
            )
            conn.commit()
        return session_id, state

    def apply(
        self, session_id: str, actor: ActorContext,
        callback: Callable[[AgentState], AgentState], expected_version: int | None = None,
    ) -> AgentState:
        # SQLite has a database-wide writer lock. Do not keep a write transaction
        # open while the Agent invokes governed capabilities against the same DB.
        # The process lock serializes local sessions; the conditional UPDATE below
        # remains the final optimistic-concurrency guard. Distributed deployment
        # uses PostgresSessionStore and row-level locks instead.
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM agent_sessions WHERE session_id = ?", (session_id,),
                ).fetchone()
            if row is None:
                raise KeyError(session_id)
            if row["company_id"] != actor.company_id or row["operator_id"] != actor.operator_id:
                raise AuthorizationError("会话不属于当前企业操作者")
            version = int(row["state_version"])
            if expected_version is not None and expected_version != version:
                raise VersionConflictError(expected_version, version)
            current: AgentState = json.loads(row["state_json"])
            state = callback(current)
            state["state_version"] = version + 1
            state["company_id"] = actor.company_id
            state["operator_id"] = actor.operator_id
            state["roles"] = list(actor.roles)
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                updated = conn.execute(
                    "UPDATE agent_sessions SET state_version = ?, state_json = ?, updated_at = ? "
                    "WHERE session_id = ? AND state_version = ?",
                    (version + 1, json.dumps(state, ensure_ascii=False), _now(), session_id, version),
                )
                if updated.rowcount != 1:
                    conn.rollback()
                    actual_row = conn.execute(
                        "SELECT state_version FROM agent_sessions WHERE session_id = ?", (session_id,),
                    ).fetchone()
                    actual = int(actual_row["state_version"]) if actual_row else version + 1
                    raise VersionConflictError(version, actual)
                conn.commit()
            return state

    def get(self, session_id: str, actor: ActorContext) -> AgentState:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM agent_sessions WHERE session_id = ?", (session_id,),
            ).fetchone()
            if row is None:
                raise KeyError(session_id)
            if row["company_id"] != actor.company_id or row["operator_id"] != actor.operator_id:
                raise AuthorizationError("会话不属于当前企业操作者")
            return json.loads(row["state_json"])


class PostgresSessionStore:
    """Distributed session store sharing the production PostgreSQL control plane."""

    def __init__(self, control: Any) -> None:
        self.control = control
        with self.control._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_sessions (
                    session_id TEXT PRIMARY KEY,
                    company_id TEXT NOT NULL,
                    operator_id TEXT NOT NULL,
                    state_version INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def create(self, actor: ActorContext) -> tuple[str, AgentState]:
        session_id = uuid.uuid4().hex
        state = initial_state()
        state.update({
            "company_id": actor.company_id, "operator_id": actor.operator_id,
            "roles": list(actor.roles), "session_id": session_id,
        })
        now = _now()
        with self.control._connection() as conn:
            conn.execute(
                "INSERT INTO agent_sessions VALUES (?, ?, ?, 0, ?, ?, ?)",
                (session_id, actor.company_id, actor.operator_id,
                 json.dumps(state, ensure_ascii=False), now, now),
            )
        return session_id, state

    def apply(
        self, session_id: str, actor: ActorContext,
        callback: Callable[[AgentState], AgentState], expected_version: int | None = None,
    ) -> AgentState:
        with self.control._connection() as conn:
            row = conn.execute(
                "SELECT * FROM agent_sessions WHERE session_id = ? FOR UPDATE", (session_id,),
            ).fetchone()
            if row is None:
                raise KeyError(session_id)
            if row["company_id"] != actor.company_id or row["operator_id"] != actor.operator_id:
                raise AuthorizationError("会话不属于当前企业操作者")
            version = int(row["state_version"])
            if expected_version is not None and expected_version != version:
                raise VersionConflictError(expected_version, version)
            state = callback(json.loads(row["state_json"]))
            state.update({
                "state_version": version + 1, "company_id": actor.company_id,
                "operator_id": actor.operator_id, "roles": list(actor.roles),
                "session_id": session_id,
            })
            updated = conn.execute(
                "UPDATE agent_sessions SET state_version = ?, state_json = ?, updated_at = ? "
                "WHERE session_id = ? AND state_version = ?",
                (version + 1, json.dumps(state, ensure_ascii=False), _now(), session_id, version),
            )
            if updated.rowcount != 1:
                raise VersionConflictError(version, version + 1)
            return state

    def get(self, session_id: str, actor: ActorContext) -> AgentState:
        with self.control._connection() as conn:
            row = conn.execute(
                "SELECT * FROM agent_sessions WHERE session_id = ?", (session_id,),
            ).fetchone()
            if row is None:
                raise KeyError(session_id)
            if row["company_id"] != actor.company_id or row["operator_id"] != actor.operator_id:
                raise AuthorizationError("会话不属于当前企业操作者")
            return json.loads(row["state_json"])
