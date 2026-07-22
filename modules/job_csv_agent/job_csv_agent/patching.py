from __future__ import annotations

import json
import re
import unicodedata
from copy import deepcopy
from difflib import SequenceMatcher
from typing import Any, Iterable

from .normalization import (
    clean_markdown, closed_field_values, experience_ranges_equivalent_for_reclassification,
    parse_experience_range,
)
from .schemas import DraftPatch, EvidenceSpan, IgnoredFragment, JSON_FIELDS, JobFields, PatchItem


REQUIREMENT_MODALITY_FIELDS = (
    "requirements_json", "preferred_requirements_json", "not_required_requirements_json",
)


def text_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[\s，,、；;。.!！：:（）()\-]+", "", normalized)


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = []
    return [clean_markdown(str(item)) for item in value or [] if str(item).strip()]


def normalize_job_fields(fields: dict[str, Any]) -> dict[str, Any]:
    candidate = dict(fields)
    for field in JSON_FIELDS:
        if field not in candidate:
            continue
        seen: set[str] = set()
        values: list[str] = []
        for value in _as_list(candidate[field]):
            key = text_key(value)
            if key and key not in seen:
                seen.add(key)
                values.append(value)
        if values:
            candidate[field] = values
        else:
            candidate.pop(field, None)
    return JobFields.model_validate(candidate).model_dump(exclude_none=True)


def requirement_modality_conflicts(fields: dict[str, Any]) -> list[str]:
    """Return conditions present in more than one mutually exclusive modality list."""
    seen: list[tuple[str, str]] = []
    conflicts: list[str] = []
    for field in REQUIREMENT_MODALITY_FIELDS:
        for value in _as_list(fields.get(field, [])):
            previous_fields = {seen_field for seen_field, item in seen if _matches(value, item)}
            if previous_fields and field not in previous_fields and not any(
                _matches(value, item) for item in conflicts
            ):
                conflicts.append(value)
            seen.append((field, value))
    hard_experience = (
        fields.get("experience_min_months"), fields.get("experience_max_months"),
    )
    hard_requirements = _as_list(fields.get("requirements_json", []))
    if any(value is not None for value in hard_experience):
        for field in ("preferred_requirements_json", "not_required_requirements_json"):
            for value in _as_list(fields.get(field, [])):
                same_hard_fact = not hard_requirements or any(
                    _matches(value, requirement) for requirement in hard_requirements
                )
                if (
                    (value_range := parse_experience_range(value)) is not None
                    and experience_ranges_equivalent_for_reclassification(
                        value_range, hard_experience,
                    )
                    and same_hard_fact
                    and not any(_matches(value, item) for item in conflicts)
                ):
                    conflicts.append(value)
    return conflicts


def _matches(value: str, target: str) -> bool:
    left, right = text_key(value), text_key(target)
    if not left or not right:
        return False
    if left == right:
        return True
    if "经验" in left and "经验" in right:
        left_range = parse_experience_range(value)
        right_range = parse_experience_range(target)
        number = r"(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十]+)"
        left_subject = re.sub(rf"{number}年(?:以上|及以上|以下|以内|起)?", "", left)
        right_subject = re.sub(rf"{number}年(?:以上|及以上|以下|以内|起)?", "", right)
        qualifier = r"^(?:(?:要求|必须|需要|需|具备)?(?:至少|最低|不少于|不低于))"
        left_subject = re.sub(qualifier, "", left_subject)
        right_subject = re.sub(qualifier, "", right_subject)
        if left_range is not None or right_range is not None:
            return left_range == right_range and left_subject == right_subject
    return SequenceMatcher(None, left, right).ratio() >= 0.9


def _item_sources(provenance: dict[str, Any], field: str, values: list[str]) -> list[dict[str, str]]:
    raw = provenance.get(field, [])
    indexed = {
        text_key(item.get("value", "")): item
        for item in raw if isinstance(item, dict)
    } if isinstance(raw, list) else {}
    return [indexed.get(text_key(value), {"value": value, "source": "explicit"}) for value in values]


def _append_item(
    draft: dict[str, Any], provenance: dict[str, Any], field: str, item: PatchItem,
) -> None:
    values = _as_list(draft.get(field, []))
    if any(_matches(value, item.value) for value in values):
        return
    values.append(item.value)
    sources = _item_sources(provenance, field, values[:-1])
    sources.append({"value": item.value, "source": item.source})
    draft[field] = values
    provenance[field] = sources


def apply_draft_patch(
    draft: dict[str, Any], provenance: dict[str, Any], patch: DraftPatch,
) -> tuple[dict[str, Any], dict[str, Any]]:
    next_draft = deepcopy(draft)
    next_provenance = deepcopy(provenance)

    for field in patch.clear_fields:
        next_draft.pop(field, None)
        next_provenance.pop(field, None)

    for field, value in patch.set_fields.items():
        next_draft[field] = value
        next_provenance[field] = {
            "value": value, "source": patch.set_sources.get(field, "explicit"),
        }

    for field, targets in patch.remove_items.items():
        values = _as_list(next_draft.get(field, []))
        sources = _item_sources(next_provenance, field, values)
        keep = [
            index for index, value in enumerate(values)
            if not any(_matches(value, target) for target in targets)
        ]
        next_draft[field] = [values[index] for index in keep]
        next_provenance[field] = [sources[index] for index in keep]

    for replacement in patch.replace_items:
        values = _as_list(next_draft.get(replacement.field, []))
        sources = _item_sources(next_provenance, replacement.field, values)
        index = next((i for i, value in enumerate(values) if _matches(value, replacement.match)), None)
        if index is None:
            _append_item(
                next_draft, next_provenance, replacement.field,
                PatchItem(value=replacement.value, source=replacement.source),
            )
        else:
            values[index] = replacement.value
            sources[index] = {"value": replacement.value, "source": replacement.source}
            next_draft[replacement.field] = values
            next_provenance[replacement.field] = sources

    # A newly assigned modality is authoritative for that item. Remove stale
    # copies from the other mutually exclusive requirement lists atomically.
    for field, items in patch.append_items.items():
        if field not in REQUIREMENT_MODALITY_FIELDS:
            continue
        targets = [item.value for item in items]
        original_experience = (
            draft.get("experience_min_months"), draft.get("experience_max_months"),
        )
        hard_experience = original_experience if any(
            value is not None for value in original_experience
        ) else (
            next_draft.get("experience_min_months"),
            next_draft.get("experience_max_months"),
        )
        hard_requirements = _as_list(draft.get("requirements_json", []))
        experience_hard_requirements = [
            requirement for requirement in hard_requirements
            if parse_experience_range(requirement) is not None
        ]
        reclassified_experience = any(
            (target_range := parse_experience_range(target)) is not None
            and experience_ranges_equivalent_for_reclassification(
                target_range, hard_experience,
            )
            and (
                not experience_hard_requirements
                or any(
                    _matches(target, requirement)
                    for requirement in experience_hard_requirements
                )
            )
            for target in targets
        )
        if (
            field in {"preferred_requirements_json", "not_required_requirements_json"}
            and reclassified_experience
        ):
            for scalar_field in ("experience_min_months", "experience_max_months"):
                next_draft.pop(scalar_field, None)
                next_provenance.pop(scalar_field, None)
        for other_field in REQUIREMENT_MODALITY_FIELDS:
            if other_field == field:
                continue
            values = _as_list(next_draft.get(other_field, []))
            sources = _item_sources(next_provenance, other_field, values)
            keep = [
                index for index, value in enumerate(values)
                if not any(_matches(value, target) for target in targets)
            ]
            next_draft[other_field] = [values[index] for index in keep]
            next_provenance[other_field] = [sources[index] for index in keep]

    for field, items in patch.append_items.items():
        for item in items:
            _append_item(next_draft, next_provenance, field, item)

    normalized = normalize_job_fields(next_draft)
    for field in JSON_FIELDS:
        if field not in normalized:
            next_provenance.pop(field, None)
    return normalized, next_provenance


def changed_fields(before: dict[str, Any], after: dict[str, Any]) -> set[str]:
    return {field for field in set(before) | set(after) if before.get(field) != after.get(field)}


def _span_source(
    span: EvidenceSpan, source_text: str, source_messages: dict[str, str] | None,
) -> str:
    if span.source_message_id and source_messages:
        return clean_markdown(source_messages.get(span.source_message_id, ""))
    return clean_markdown(source_text)


def _field_has_evidence(
    field: str, source_text: str, evidence: Iterable[EvidenceSpan],
    source_messages: dict[str, str] | None = None,
) -> bool:
    return any(
        span.field == field and clean_markdown(span.text) in _span_source(
            span, source_text, source_messages,
        )
        for span in evidence
    )


def sanitize_model_patch(
    patch: DraftPatch,
    source_text: str,
    mentioned_fields: list[str],
    evidence_spans: list[EvidenceSpan],
    ignored_fragments: list[IgnoredFragment],
    source_messages: dict[str, str] | None = None,
) -> DraftPatch:
    """Enforce source evidence and ignored-fragment boundaries in deterministic code."""
    data = patch.model_dump(mode="python")
    sources = [clean_markdown(source_text)]
    sources.extend(clean_markdown(value) for value in (source_messages or {}).values())
    source = "\n".join(value for value in sources if value)
    evidence_by_field = {
        field: _field_has_evidence(field, source_text, evidence_spans, source_messages)
        for field in mentioned_fields
    }
    closed_values = closed_field_values(source_text)

    for field in list(data.get("set_fields", {})):
        value = data["set_fields"][field]
        direct_value = str(value).removesuffix(".0")
        direct_evidence = direct_value in source
        normalized_evidence = closed_values.get(field) == value
        if (
            field in evidence_by_field
            and not evidence_by_field[field]
            and not direct_evidence
            and not normalized_evidence
        ):
            data["set_fields"].pop(field, None)
            data["set_sources"].pop(field, None)

    for field, items in list(data.get("append_items", {}).items()):
        kept = []
        for item in items:
            value = clean_markdown(item["value"])
            directly_present = value in source
            supported_span = any(
                span.field == field
                and clean_markdown(span.text) in _span_source(span, source_text, source_messages)
                and (_matches(value, span.text) or text_key(value) in text_key(span.text))
                for span in evidence_spans
            )
            normalized = item.get("source") == "normalized"
            if directly_present or supported_span or normalized or field not in mentioned_fields:
                item["value"] = value
                kept.append(item)
        if kept:
            data["append_items"][field] = kept
        else:
            data["append_items"].pop(field, None)

    ignored = [clean_markdown(item.text) for item in ignored_fragments]
    for field, value in list(data.get("set_fields", {}).items()):
        if not isinstance(value, str):
            continue
        cleaned = value
        for fragment in ignored:
            cleaned = cleaned.replace(fragment, "").strip(" /｜|-：:，,；;")
        if cleaned:
            data["set_fields"][field] = cleaned
        else:
            data["set_fields"].pop(field, None)
            data["set_sources"].pop(field, None)
    for field, items in list(data.get("append_items", {}).items()):
        kept = []
        for item in items:
            cleaned = item["value"]
            for fragment in ignored:
                cleaned = cleaned.replace(fragment, "").strip(" /｜|-：:，,；;")
            if cleaned:
                item["value"] = cleaned
                kept.append(item)
        if kept:
            data["append_items"][field] = kept
        else:
            data["append_items"].pop(field, None)
    return DraftPatch.model_validate(data)


def unapplied_fields(
    before: dict[str, Any], after: dict[str, Any], patch: DraftPatch, mentioned_fields: list[str],
) -> list[str]:
    failures: list[str] = []
    for field in dict.fromkeys(mentioned_fields):
        if field in patch.set_fields:
            if after.get(field) != patch.set_fields[field]:
                failures.append(field)
            continue
        if field in patch.clear_fields:
            if field in after:
                failures.append(field)
            continue
        if field in JSON_FIELDS:
            expected = [item.value for item in patch.append_items.get(field, [])]
            expected += [item.value for item in patch.replace_items if item.field == field]
            removed = patch.remove_items.get(field, [])
            values = _as_list(after.get(field, []))
            if expected and all(any(_matches(value, item) for value in values) for item in expected):
                continue
            if removed and all(not any(_matches(value, item) for value in values) for item in removed):
                continue
        # An unchanged value is acceptable only when the patch states the same target.
        failures.append(field)
    return failures
