from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from hr_agent_contracts import (
    ActorContext,
    AuditEvent,
    JobLifecycleStatus,
    JobProfile,
    JobProfileReview,
    JobProfileVersion,
)


class AuthorizationError(PermissionError):
    pass


class VersionConflictError(RuntimeError):
    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(f"岗位版本冲突：请求版本 {expected}，当前版本 {actual}")
        self.expected = expected
        self.actual = actual


class LifecycleConflictError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class SqliteControlPlane:
    """Canonical tenant, version, lifecycle and audit store for the MVP."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._lock, self._connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS governed_jobs (
                    job_id TEXT PRIMARY KEY,
                    company_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_version INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    published_at TEXT,
                    closed_at TEXT,
                    deleted_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_governed_jobs_company_status
                    ON governed_jobs(company_id, status);

                CREATE TABLE IF NOT EXISTS job_profile_versions (
                    job_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    profile_json TEXT NOT NULL,
                    review_json TEXT,
                    change_reason TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    PRIMARY KEY (job_id, version),
                    FOREIGN KEY (job_id) REFERENCES governed_jobs(job_id)
                );

                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    company_id TEXT NOT NULL,
                    operator_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    resource_id TEXT,
                    occurred_at TEXT NOT NULL,
                    request_id TEXT,
                    details_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_audit_company_time
                    ON audit_events(company_id, occurred_at DESC);

                CREATE TABLE IF NOT EXISTS workflow_runs (
                    run_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    company_id TEXT NOT NULL,
                    operator_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    status TEXT NOT NULL,
                    nodes_json TEXT NOT NULL,
                    error_type TEXT
                );
                """
            )

    @staticmethod
    def _ensure_write(actor: ActorContext) -> None:
        if not actor.can_write_jobs():
            raise AuthorizationError("当前操作者没有岗位写入权限")

    def _ensure_owner(self, conn: sqlite3.Connection, actor: ActorContext, job_id: str) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM governed_jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None or row["company_id"] != actor.company_id:
            raise AuthorizationError("岗位不存在或不属于当前企业")
        return row

    def append_audit(
        self, conn: sqlite3.Connection, actor: ActorContext, action: str,
        resource_type: str, resource_id: str | None, details: dict[str, Any] | None = None,
    ) -> AuditEvent:
        event = AuditEvent(
            event_id=f"audit_{uuid.uuid4().hex}", company_id=actor.company_id,
            operator_id=actor.operator_id, action=action, resource_type=resource_type,
            resource_id=resource_id, occurred_at=_now(), request_id=actor.request_id,
            details=details or {},
        )
        conn.execute(
            "INSERT INTO audit_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event.event_id, event.company_id, event.operator_id, event.action,
                event.resource_type, event.resource_id, event.occurred_at, event.request_id,
                json.dumps(event.details, ensure_ascii=False, separators=(",", ":")),
            ),
        )
        return event

    def create_job(
        self, actor: ActorContext, profile: JobProfile, review: JobProfileReview,
        *, change_reason: str = "created",
    ) -> JobProfileVersion:
        self._ensure_write(actor)
        if not profile.job_id:
            raise ValueError("profile.job_id is required")
        now = _now()
        status = (
            JobLifecycleStatus.REVIEWED if review.publishable
            else JobLifecycleStatus.NEEDS_CLARIFICATION
        )
        with self._lock, self._connection() as conn:
            if conn.execute("SELECT 1 FROM governed_jobs WHERE job_id = ?", (profile.job_id,)).fetchone():
                raise ValueError(f"岗位已存在：{profile.job_id}")
            conn.execute(
                "INSERT INTO governed_jobs(job_id, company_id, status, current_version, created_at, updated_at, created_by) "
                "VALUES (?, ?, ?, 1, ?, ?, ?)",
                (profile.job_id, actor.company_id, status.value, now, now, actor.operator_id),
            )
            self._insert_version(conn, profile.job_id, 1, status, profile, review, change_reason, actor, now)
            self.append_audit(conn, actor, "job.created", "job", profile.job_id, {"version": 1})
        return JobProfileVersion(
            job_id=profile.job_id, version=1, status=status, profile=profile,
            review=review, created_at=now, created_by=actor.operator_id,
        )

    def add_version(
        self, actor: ActorContext, profile: JobProfile, review: JobProfileReview,
        *, expected_version: int | None = None, change_reason: str = "updated",
    ) -> JobProfileVersion:
        self._ensure_write(actor)
        if not profile.job_id:
            raise ValueError("profile.job_id is required")
        with self._lock, self._connection() as conn:
            job = self._ensure_owner(conn, actor, profile.job_id)
            actual = int(job["current_version"])
            if expected_version is not None and expected_version != actual:
                raise VersionConflictError(expected_version, actual)
            if job["status"] == JobLifecycleStatus.DELETED.value:
                raise LifecycleConflictError("已删除岗位不能直接修改，请先恢复")
            version = actual + 1
            now = _now()
            status = (
                JobLifecycleStatus.REVIEWED if review.publishable
                else JobLifecycleStatus.NEEDS_CLARIFICATION
            )
            conn.execute(
                "UPDATE governed_jobs SET status = ?, current_version = ?, updated_at = ?, "
                "published_at = NULL, closed_at = NULL WHERE job_id = ?",
                (status.value, version, now, profile.job_id),
            )
            self._insert_version(
                conn, profile.job_id, version, status, profile, review,
                change_reason, actor, now,
            )
            self.append_audit(conn, actor, "job.version_created", "job", profile.job_id, {
                "version": version, "previous_version": actual, "reason": change_reason,
            })
        return JobProfileVersion(
            job_id=profile.job_id, version=version, status=status, profile=profile,
            review=review, created_at=now, created_by=actor.operator_id,
        )

    @staticmethod
    def _insert_version(
        conn: sqlite3.Connection, job_id: str, version: int, status: JobLifecycleStatus,
        profile: JobProfile, review: JobProfileReview | None, change_reason: str,
        actor: ActorContext, now: str,
    ) -> None:
        conn.execute(
            "INSERT INTO job_profile_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job_id, version, status.value, profile.model_dump_json(),
                review.model_dump_json() if review else None, change_reason, now,
                actor.operator_id,
            ),
        )

    def latest(self, actor: ActorContext, job_id: str, *, include_deleted: bool = False) -> JobProfileVersion:
        with self._lock, self._connection() as conn:
            job = self._ensure_owner(conn, actor, job_id)
            if job["status"] == JobLifecycleStatus.DELETED.value and not include_deleted:
                raise AuthorizationError("岗位不存在或已删除")
            version = conn.execute(
                "SELECT * FROM job_profile_versions WHERE job_id = ? AND version = ?",
                (job_id, job["current_version"]),
            ).fetchone()
            if version is None:
                raise LookupError(job_id)
            return self._version_from_rows(job, version)

    @staticmethod
    def _version_from_rows(job: sqlite3.Row, version: sqlite3.Row) -> JobProfileVersion:
        return JobProfileVersion(
            job_id=job["job_id"], version=int(version["version"]),
            status=JobLifecycleStatus(job["status"]),
            profile=JobProfile.model_validate_json(version["profile_json"]),
            review=(JobProfileReview.model_validate_json(version["review_json"])
                    if version["review_json"] else None),
            created_at=version["created_at"], created_by=version["created_by"],
        )

    def list_latest(self, actor: ActorContext, *, include_deleted: bool = False) -> list[JobProfileVersion]:
        with self._lock, self._connection() as conn:
            clauses = ["j.company_id = ?"]
            parameters: list[Any] = [actor.company_id]
            if not include_deleted:
                clauses.append("j.status != ?")
                parameters.append(JobLifecycleStatus.DELETED.value)
            rows = conn.execute(
                "SELECT j.*, v.profile_json, v.review_json, v.change_reason, "
                "v.created_at AS version_created_at, v.created_by AS version_created_by, "
                "v.version AS version_number FROM governed_jobs j JOIN job_profile_versions v "
                "ON v.job_id = j.job_id AND v.version = j.current_version WHERE "
                + " AND ".join(clauses) + " ORDER BY j.updated_at DESC",
                parameters,
            ).fetchall()
            return [JobProfileVersion(
                job_id=row["job_id"], version=int(row["version_number"]),
                status=JobLifecycleStatus(row["status"]),
                profile=JobProfile.model_validate_json(row["profile_json"]),
                review=(JobProfileReview.model_validate_json(row["review_json"])
                        if row["review_json"] else None),
                created_at=row["version_created_at"], created_by=row["version_created_by"],
            ) for row in rows]

    def history(self, actor: ActorContext, job_id: str) -> list[JobProfileVersion]:
        with self._lock, self._connection() as conn:
            job = self._ensure_owner(conn, actor, job_id)
            rows = conn.execute(
                "SELECT * FROM job_profile_versions WHERE job_id = ? ORDER BY version DESC",
                (job_id,),
            ).fetchall()
            return [self._version_from_rows(job, row) for row in rows]

    def publish(self, actor: ActorContext, job_id: str, expected_version: int | None = None) -> JobProfileVersion:
        if not actor.can_publish_jobs():
            raise AuthorizationError("当前操作者没有岗位发布权限")
        with self._lock, self._connection() as conn:
            job = self._ensure_owner(conn, actor, job_id)
            actual = int(job["current_version"])
            if expected_version is not None and expected_version != actual:
                raise VersionConflictError(expected_version, actual)
            version = conn.execute(
                "SELECT * FROM job_profile_versions WHERE job_id = ? AND version = ?",
                (job_id, actual),
            ).fetchone()
            review = JobProfileReview.model_validate_json(version["review_json"]) if version["review_json"] else None
            if review is None or not review.publishable:
                raise LifecycleConflictError("岗位画像未通过质量门，不能发布")
            now = _now()
            conn.execute(
                "UPDATE governed_jobs SET status = ?, published_at = ?, closed_at = NULL, updated_at = ? WHERE job_id = ?",
                (JobLifecycleStatus.PUBLISHED.value, now, now, job_id),
            )
            self.append_audit(conn, actor, "job.published", "job", job_id, {"version": actual})
            updated = dict(job)
            updated["status"] = JobLifecycleStatus.PUBLISHED.value
            return self._version_from_rows(updated, version)  # type: ignore[arg-type]

    def close(self, actor: ActorContext, job_id: str) -> None:
        self._ensure_write(actor)
        with self._lock, self._connection() as conn:
            job = self._ensure_owner(conn, actor, job_id)
            if job["status"] not in {JobLifecycleStatus.PUBLISHED.value, JobLifecycleStatus.REVIEWED.value}:
                raise LifecycleConflictError("只有已审核或已发布岗位可以关闭")
            now = _now()
            conn.execute(
                "UPDATE governed_jobs SET status = ?, closed_at = ?, updated_at = ? WHERE job_id = ?",
                (JobLifecycleStatus.CLOSED.value, now, now, job_id),
            )
            self.append_audit(conn, actor, "job.closed", "job", job_id, {
                "version": int(job["current_version"]),
            })

    def soft_delete(self, actor: ActorContext, job_id: str) -> None:
        self._ensure_write(actor)
        with self._lock, self._connection() as conn:
            job = self._ensure_owner(conn, actor, job_id)
            now = _now()
            conn.execute(
                "UPDATE governed_jobs SET status = ?, deleted_at = ?, updated_at = ? WHERE job_id = ?",
                (JobLifecycleStatus.DELETED.value, now, now, job_id),
            )
            self.append_audit(conn, actor, "job.soft_deleted", "job", job_id, {
                "version": int(job["current_version"]),
            })

    def list_audit(self, actor: ActorContext, limit: int = 100) -> list[AuditEvent]:
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_events WHERE company_id = ? ORDER BY occurred_at DESC LIMIT ?",
                (actor.company_id, max(1, min(limit, 500))),
            ).fetchall()
            return [AuditEvent(
                event_id=row["event_id"], company_id=row["company_id"],
                operator_id=row["operator_id"], action=row["action"],
                resource_type=row["resource_type"], resource_id=row["resource_id"],
                occurred_at=row["occurred_at"], request_id=row["request_id"],
                details=json.loads(row["details_json"]),
            ) for row in rows]

    def record_workflow_run(
        self, *, run_id: str, session_id: str | None, actor: ActorContext,
        started_at: str, duration_ms: float, status: str, nodes: list[str],
        error_type: str | None = None,
    ) -> None:
        with self._lock, self._connection() as conn:
            conn.execute(
                "INSERT INTO workflow_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, session_id, actor.company_id, actor.operator_id, started_at,
                 duration_ms, status, json.dumps(nodes, ensure_ascii=False), error_type),
            )

    def metrics(self, actor: ActorContext) -> dict[str, Any]:
        with self._lock, self._connection() as conn:
            jobs = conn.execute(
                "SELECT status, COUNT(*) AS count FROM governed_jobs WHERE company_id = ? GROUP BY status",
                (actor.company_id,),
            ).fetchall()
            runs = conn.execute(
                "SELECT COUNT(*) AS count, AVG(duration_ms) AS avg_ms FROM workflow_runs WHERE company_id = ?",
                (actor.company_id,),
            ).fetchone()
            return {
                "jobs_by_status": {row["status"]: int(row["count"]) for row in jobs},
                "workflow_runs": int(runs["count"] or 0),
                "workflow_avg_duration_ms": round(float(runs["avg_ms"] or 0), 2),
            }


class _PostgresConnection:
    """Minimal DB-API compatibility layer for the control-plane SQL."""

    def __init__(self, raw: Any) -> None:
        self.raw = raw

    def execute(self, sql: str, parameters: Any = ()) -> Any:
        cursor = self.raw.cursor()
        cursor.execute(sql.replace("?", "%s"), parameters)
        return cursor

    def executescript(self, script: str) -> None:
        for statement in script.split(";"):
            if statement.strip():
                self.execute(statement)

    def commit(self) -> None:
        self.raw.commit()

    def rollback(self) -> None:
        self.raw.rollback()

    def close(self) -> None:
        self.raw.close()


class PostgresControlPlane(SqliteControlPlane):
    """Production control-plane backend; selected with HR_AGENT_DATABASE_URL."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.path = Path("postgres-control-plane")
        self._lock = threading.RLock()
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL 已配置，但未安装 psycopg；请安装 requirements.txt。"
            ) from exc
        connection = _PostgresConnection(
            psycopg.connect(self.database_url, row_factory=dict_row),
        )
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def create_control_plane(sqlite_path: Path, database_url: str | None = None) -> SqliteControlPlane:
    return PostgresControlPlane(database_url) if database_url else SqliteControlPlane(sqlite_path)
