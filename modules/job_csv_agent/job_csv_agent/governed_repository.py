from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from difflib import SequenceMatcher
from typing import Any, Iterator

from hr_agent_contracts import ActorContext, JobProfileVersion

from .critic import BoundedProfileCritic
from .governance import SqliteControlPlane
from .profiles import fields_from_profile, profile_from_fields
from .repository import CsvJobRepository, JobNotFoundError
from .schemas import JSON_FIELDS


_actor_var: ContextVar[ActorContext | None] = ContextVar("hr_actor", default=None)
_expected_job_version_var: ContextVar[int | None] = ContextVar("expected_job_version", default=None)


class GovernedJobRepository:
    """Tenant-scoped canonical repository with CSV maintained as a read-model projection."""

    def __init__(
        self, projection: CsvJobRepository, control: SqliteControlPlane,
        critic: BoundedProfileCritic, default_actor: ActorContext,
    ) -> None:
        self.projection = projection
        self.control = control
        self.critic = critic
        self.default_actor = default_actor

    @contextmanager
    def bind(
        self, actor: ActorContext, expected_job_version: int | None = None,
    ) -> Iterator[None]:
        actor_token = _actor_var.set(actor)
        version_token = _expected_job_version_var.set(expected_job_version)
        try:
            yield
        finally:
            _expected_job_version_var.reset(version_token)
            _actor_var.reset(actor_token)

    @property
    def actor(self) -> ActorContext:
        return _actor_var.get() or self.default_actor

    def bootstrap_legacy(self) -> int:
        imported = 0
        existing = {item.job_id for item in self.control.list_latest(self.default_actor, include_deleted=True)}
        for row in self.projection.list_all():
            job_id = str(row.get("job_id", ""))
            if not job_id or job_id in existing:
                continue
            profile = profile_from_fields(row, self.default_actor.company_id).model_copy(
                update={"job_id": job_id},
            )
            review = self.critic.review(profile, 1)
            self.control.create_job(
                self.default_actor, profile, review, change_reason="legacy_csv_import",
            )
            imported += 1
        return imported

    @staticmethod
    def _as_row(item: JobProfileVersion) -> dict[str, Any]:
        row = fields_from_profile(item.profile)
        row.update({
            "job_id": item.job_id,
            "profile_version": item.version,
            "lifecycle_status": item.status.value,
        })
        return row

    def get(self, job_id: str) -> dict[str, Any]:
        try:
            return self._as_row(self.control.latest(self.actor, job_id))
        except (LookupError, PermissionError) as exc:
            raise JobNotFoundError(job_id) from exc

    def create(self, fields: dict[str, Any]) -> dict[str, Any]:
        row = self.projection.create(fields)
        profile = profile_from_fields(row, self.actor.company_id).model_copy(
            update={"job_id": row["job_id"]},
        )
        review = self.critic.review(profile, 1)
        version = self.control.create_job(self.actor, profile, review)
        return {**row, "profile_version": version.version, "lifecycle_status": version.status.value}

    def update(
        self, job_id: str, fields: dict[str, Any], clear_fields: list[str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        current = self.control.latest(self.actor, job_id)
        expected = _expected_job_version_var.get()
        if expected is not None and expected != current.version:
            from .governance import VersionConflictError
            raise VersionConflictError(expected, current.version)
        before_projection, after_projection = self.projection.update(job_id, fields, clear_fields)
        profile = profile_from_fields(after_projection, self.actor.company_id).model_copy(
            update={"job_id": job_id},
        )
        review = self.critic.review(profile, current.version + 1)
        version = self.control.add_version(
            self.actor, profile, review, expected_version=current.version,
        )
        return (
            {**before_projection, "profile_version": current.version,
             "lifecycle_status": current.status.value},
            {**after_projection, "profile_version": version.version,
             "lifecycle_status": version.status.value},
        )

    def delete(self, job_id: str) -> dict[str, Any]:
        current = self.control.latest(self.actor, job_id)
        deleted = self.projection.delete(job_id)
        self.control.soft_delete(self.actor, job_id)
        return {**deleted, "profile_version": current.version,
                "lifecycle_status": "DELETED"}

    @staticmethod
    def _matches(row: dict[str, Any], filters: dict[str, str]) -> bool:
        for field in ("company_name", "title", "city"):
            needle = str(filters.get(field, "")).strip().casefold()
            if needle and needle not in str(row.get(field, "")).casefold():
                return False
        recruitment = str(filters.get("recruitment", "")).strip().casefold()
        return not recruitment or str(row.get("recruitment", "")).casefold() == recruitment

    def _rows(self, filters: dict[str, str] | None = None) -> list[dict[str, Any]]:
        rows = [self._as_row(item) for item in self.control.list_latest(self.actor)]
        return [row for row in rows if self._matches(row, filters or {})]

    def count(self, filters: dict[str, str] | None = None) -> int:
        return len(self._rows(filters))

    def list_summaries(
        self, filters: dict[str, str] | None = None, limit: int = 10,
    ) -> list[dict[str, str]]:
        rows = self._rows(filters)[:max(1, min(int(limit), 10))]
        return [
            {key: str(row.get(key, "")) for key in (
                "job_id", "company_name", "title", "city", "profile_version", "lifecycle_status",
            )}
            for row in rows
        ]

    def find_job_ids(self, filters: dict[str, str] | None = None, limit: int = 10) -> list[str]:
        return [str(row["job_id"]) for row in self._rows(filters)[:max(1, min(int(limit), 10))]]

    def get_public_detail(self, job_id: str) -> dict[str, Any]:
        row = self.get(job_id)
        detail: dict[str, Any] = {}
        for key, value in row.items():
            if value in (None, "", []):
                continue
            if key in JSON_FIELDS and isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    value = []
            detail[key] = value
        return detail

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        needle = " ".join(query.casefold().split())
        if not needle:
            return []
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in self._rows():
            company = str(row.get("company_name", ""))
            title = str(row.get("title", ""))
            job_id = str(row.get("job_id", ""))
            haystack = f"{company} {title} {job_id}".casefold()
            overlap = sum(1 for token in needle.split() if token in haystack)
            score = SequenceMatcher(None, needle, haystack).ratio() + overlap * 0.3
            if needle in haystack:
                score += 1.0
            if overlap or score >= 0.28:
                scored.append((score, row))
        scored.sort(key=lambda item: (-item[0], str(item[1]["job_id"])))
        return [row for _, row in scored[:limit]]
