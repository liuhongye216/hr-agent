from __future__ import annotations

import json
import re
import unicodedata
from copy import deepcopy
from difflib import SequenceMatcher
from typing import Any, Iterable

from .normalization import clean_markdown, closed_field_values
from .schemas import DraftPatch, EvidenceSpan, IgnoredFragment, JSON_FIELDS, JobFields, PatchItem


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
        candidate[field] = values
    return JobFields.model_validate(candidate).model_dump(exclude_none=True)


def _matches(value: str, target: str) -> bool:
    left, right = text_key(value), text_key(target)
    if not left or not right:
        return False
    return left == right or SequenceMatcher(None, left, right).ratio() >= 0.9


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

    for field, items in patch.append_items.items():
        for item in items:
            _append_item(next_draft, next_provenance, field, item)

    return normalize_job_fields(next_draft), next_provenance


def changed_fields(before: dict[str, Any], after: dict[str, Any]) -> set[str]:
    return {field for field in set(before) | set(after) if before.get(field) != after.get(field)}


def _field_has_evidence(field: str, source_text: str, evidence: Iterable[EvidenceSpan]) -> bool:
    source = clean_markdown(source_text)
    return any(
        span.field == field and clean_markdown(span.text) in source
        for span in evidence
    )


def sanitize_model_patch(
    patch: DraftPatch,
    source_text: str,
    mentioned_fields: list[str],
    evidence_spans: list[EvidenceSpan],
    ignored_fragments: list[IgnoredFragment],
) -> DraftPatch:
    """Enforce source evidence and ignored-fragment boundaries in deterministic code."""
    data = patch.model_dump(mode="python")
    source = clean_markdown(source_text)
    evidence_by_field = {
        field: _field_has_evidence(field, source_text, evidence_spans)
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
                and clean_markdown(span.text) in source
                and (_matches(value, span.text) or text_key(value) in text_key(span.text))
                for span in evidence_spans
            )
            if directly_present or supported_span or field not in mentioned_fields:
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
