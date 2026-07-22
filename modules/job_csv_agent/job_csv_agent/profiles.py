from __future__ import annotations

import json
from typing import Any

from hr_agent_contracts import Evidence, JobProfile

from .schemas import JSON_FIELDS


PROFILE_FIELD_MAP = {
    "work_locations_json": "work_locations",
    "graduation_years_json": "graduation_years",
    "major_requirements_json": "major_requirements",
    "student_status_json": "student_status",
    "requirements_json": "requirements",
    "responsibilities_json": "responsibilities",
    "skills_json": "skills",
    "certificates_json": "certificates",
    "benefits_json": "benefits",
    "preferred_requirements_json": "preferred_requirements",
    "not_required_requirements_json": "not_required_requirements",
}


def _decode_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return [str(item) for item in value or [] if str(item).strip()]


def profile_from_fields(
    fields: dict[str, Any], company_id: str, provenance: dict[str, Any] | None = None,
) -> JobProfile:
    payload: dict[str, Any] = {"company_id": company_id}
    valid_fields = set(JobProfile.model_fields)
    for source, value in fields.items():
        target = PROFILE_FIELD_MAP.get(source, source)
        if target not in valid_fields or value in (None, ""):
            continue
        payload[target] = _decode_list(value) if source in JSON_FIELDS else value

    evidence: list[Evidence] = []
    for field, source in (provenance or {}).items():
        entries = source if isinstance(source, list) else [source]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            text = str(entry.get("evidence_text") or entry.get("text") or "").strip()
            if not text:
                continue
            evidence.append(Evidence(
                field=PROFILE_FIELD_MAP.get(field, field),
                text=text,
                source_message_id=entry.get("source_message_id"),
                source_type=entry.get("source", "explicit"),
                confirmed=entry.get("source") in {"explicit", "user_confirmed"},
            ))
    payload["evidence"] = evidence
    return JobProfile.model_validate(payload)


def fields_from_profile(profile: JobProfile) -> dict[str, Any]:
    reverse = {target: source for source, target in PROFILE_FIELD_MAP.items()}
    data = profile.model_dump(exclude={"schema_version", "company_id", "facts", "evidence"})
    result: dict[str, Any] = {}
    for key, value in data.items():
        if value in (None, "", []):
            continue
        result[reverse.get(key, key)] = value
    return result
