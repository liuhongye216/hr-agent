from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import portalocker

from .schemas import (
    BUSINESS_COLUMNS, CREATE_CONTENT_FIELDS, EDITABLE_FIELDS, JSON_FIELDS, JobFields,
    REQUIRED_CREATE_FIELDS, SemanticFact,
)
from .guards import SEMANTIC_FIELD_BY_CATEGORY, text_key


class JobNotFoundError(LookupError):
    pass


class SemanticValidationError(ValueError):
    pass


class CsvJobRepository:
    """Pandas-backed repository with a sidecar lock and atomic replacement."""

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = Path(csv_path)
        self.lock_path = self.csv_path.with_suffix(self.csv_path.suffix + ".lock")

    @contextmanager
    def _locked_frame(self) -> Iterator[pd.DataFrame]:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(self.lock_path), mode="a+", timeout=10):
            if self.csv_path.exists() and self.csv_path.stat().st_size:
                frame = pd.read_csv(self.csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
            else:
                frame = pd.DataFrame(columns=BUSINESS_COLUMNS)
            missing = [column for column in BUSINESS_COLUMNS if column not in frame.columns]
            if missing:
                raise ValueError(f"CSV schema 缺少字段：{', '.join(missing)}")
            if frame["job_id"].duplicated().any():
                raise ValueError("CSV 中存在重复 job_id，已拒绝写入")
            yield frame.loc[:, list(BUSINESS_COLUMNS)].copy()

    def _atomic_write(self, frame: pd.DataFrame) -> None:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8-sig", newline="", suffix=".tmp",
                prefix=f".{self.csv_path.name}.", dir=self.csv_path.parent, delete=False,
            ) as handle:
                temporary = Path(handle.name)
                frame.to_csv(handle, index=False, lineterminator="\n")
            os.replace(temporary, self.csv_path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _serialize_fields(fields: dict[str, Any]) -> dict[str, str]:
        result: dict[str, str] = {}
        for key, value in fields.items():
            if key not in EDITABLE_FIELDS:
                raise ValueError(f"字段不可编辑：{key}")
            if key in JSON_FIELDS:
                result[key] = json.dumps(value or [], ensure_ascii=False, separators=(",", ":"))
            elif value is None:
                result[key] = ""
            else:
                result[key] = str(value)
        return result

    @staticmethod
    def _content_hash(row: dict[str, Any]) -> str:
        editable = {key: row.get(key, "") for key in EDITABLE_FIELDS}
        payload = json.dumps(editable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _validation_payload(row: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key in EDITABLE_FIELDS:
            value = row.get(key)
            if value in (None, ""):
                continue
            if key in JSON_FIELDS and isinstance(value, str):
                value = json.loads(value)
            payload[key] = value
        return payload

    @staticmethod
    def _list_value(fields: dict[str, Any], key: str) -> list[str]:
        value = fields.get(key, [])
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise SemanticValidationError(f"{key} 不是有效 JSON 数组") from exc
        return [str(item).strip() for item in value or [] if str(item).strip()]

    @classmethod
    def _validate_semantics(
        cls,
        fields: dict[str, Any],
        semantic_facts: list[dict[str, Any] | SemanticFact] | None = None,
    ) -> None:
        formal_values = {
            field: cls._list_value(fields, field)
            for field in ("requirements_json", "responsibilities_json", "skills_json", "benefits_json")
        }
        parsed_facts = [
            item if isinstance(item, SemanticFact) else SemanticFact.model_validate(item)
            for item in semantic_facts or []
        ]
        for fact in parsed_facts:
            containing_fields = [
                field for field, values in formal_values.items()
                if text_key(fact.value) in {text_key(value) for value in values}
            ]
            if not containing_fields:
                continue
            if fact.source_type == "inferred":
                raise SemanticValidationError(f"推断信息未经用户确认，禁止写入：{fact.value}")
            if fact.category == "unknown":
                raise SemanticValidationError(f"待澄清信息禁止写入：{fact.value}")
            if fact.needs_confirmation and fact.source_type != "user_confirmed":
                raise SemanticValidationError(f"信息仍需用户确认，禁止写入：{fact.value}")
            expected = SEMANTIC_FIELD_BY_CATEGORY.get(fact.category)
            if expected and expected not in containing_fields:
                raise SemanticValidationError(
                    f"语义事实“{fact.value}”分类为 {fact.category}，不能写入 {', '.join(containing_fields)}"
                )

    @classmethod
    def _validate_complete_row(
        cls, row: dict[str, Any], required_fields: tuple[str, ...] = REQUIRED_CREATE_FIELDS,
        *, require_job_content: bool = False,
    ) -> None:
        payload = cls._validation_payload(row)
        JobFields.model_validate(payload)
        missing = [key for key in required_fields if not payload.get(key)]
        if missing:
            raise ValueError(f"岗位不能缺少必填字段：{', '.join(missing)}")
        if require_job_content and not any(payload.get(key) for key in CREATE_CONTENT_FIELDS):
            raise ValueError("岗位至少需要一项任职要求、岗位职责或可抽取的完整 JD 内容")

    def get(self, job_id: str) -> dict[str, str]:
        with self._locked_frame() as frame:
            matches = frame.loc[frame["job_id"] == job_id]
            if matches.empty:
                raise JobNotFoundError(job_id)
            return matches.iloc[0].to_dict()

    def create(
        self,
        fields: dict[str, Any],
        semantic_facts: list[dict[str, Any] | SemanticFact] | None = None,
    ) -> dict[str, str]:
        validated = JobFields.model_validate(fields).model_dump(exclude_none=True)
        missing = [key for key in REQUIRED_CREATE_FIELDS if not validated.get(key)]
        if missing:
            raise ValueError(f"新建岗位缺少必填字段：{', '.join(missing)}")
        if not any(validated.get(key) for key in CREATE_CONTENT_FIELDS):
            raise ValueError("新建岗位至少需要一项任职要求、岗位职责或可抽取的完整 JD 内容")
        self._validate_semantics(validated, semantic_facts)
        values = self._serialize_fields(validated)
        with self._locked_frame() as frame:
            job_id = f"job_{uuid.uuid4().hex[:16]}"
            while (frame["job_id"] == job_id).any():
                job_id = f"job_{uuid.uuid4().hex[:16]}"
            row = {column: "" for column in BUSINESS_COLUMNS}
            row.update({field: "[]" for field in JSON_FIELDS})
            row.update(values)
            row.update({
                "job_id": job_id,
                "scraped_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "extraction_mode": "manual_agent",
            })
            self._validate_complete_row(row, require_job_content=True)
            row["content_hash"] = self._content_hash(row)
            frame.loc[len(frame)] = row
            self._atomic_write(frame)
            return row

    def update(
        self,
        job_id: str,
        fields: dict[str, Any],
        clear_fields: list[str] | None = None,
        semantic_facts: list[dict[str, Any] | SemanticFact] | None = None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        validated = JobFields.model_validate(fields).model_dump(exclude_none=True)
        self._validate_semantics(validated, semantic_facts)
        values = self._serialize_fields(validated)
        for key in clear_fields or []:
            if key not in EDITABLE_FIELDS:
                raise ValueError(f"字段不可清空：{key}")
            values[key] = "[]" if key in JSON_FIELDS else ""
        if not values:
            raise ValueError("没有可写入的修改字段")
        with self._locked_frame() as frame:
            indexes = frame.index[frame["job_id"] == job_id].tolist()
            if not indexes:
                raise JobNotFoundError(job_id)
            index = indexes[0]
            before = frame.loc[index].to_dict()
            candidate = {**before, **values}
            # Legacy rows may have no requirements_json. Do not block an unrelated edit,
            # but never allow the identity fields to be cleared.
            self._validate_complete_row(candidate, ("company_name", "title"))
            for key, value in values.items():
                frame.at[index, key] = value
            after = frame.loc[index].to_dict()
            after["scraped_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            after["extraction_mode"] = "manual_agent"
            after["content_hash"] = self._content_hash(after)
            for key, value in after.items():
                frame.at[index, key] = value
            self._atomic_write(frame)
            return before, after

    def delete(self, job_id: str) -> dict[str, str]:
        with self._locked_frame() as frame:
            indexes = frame.index[frame["job_id"] == job_id].tolist()
            if not indexes:
                raise JobNotFoundError(job_id)
            deleted = frame.loc[indexes[0]].to_dict()
            self._atomic_write(frame.drop(index=indexes[0]).reset_index(drop=True))
            return deleted

    def search(self, query: str, limit: int = 5) -> list[dict[str, str]]:
        needle = " ".join(query.casefold().split())
        if not needle:
            return []
        with self._locked_frame() as frame:
            scored: list[tuple[float, dict[str, str]]] = []
            for row in frame.to_dict(orient="records"):
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
            scored.sort(key=lambda item: (-item[0], item[1]["job_id"]))
            return [row for _, row in scored[:limit]]
