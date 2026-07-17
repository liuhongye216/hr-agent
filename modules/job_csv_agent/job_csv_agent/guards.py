from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import Any

from .schemas import JSON_FIELDS, JobFields, SemanticFact, SemanticIssue, StructuredCommand


SEMANTIC_FIELD_BY_CATEGORY = {
    "requirement": "requirements_json",
    "responsibility": "responsibilities_json",
    "skill": "skills_json",
    "benefit": "benefits_json",
}
SEMANTIC_FIELDS = tuple(SEMANTIC_FIELD_BY_CATEGORY.values())


def text_key(value: str) -> str:
    """Canonical key for equality/evidence checks, never for semantic inference."""
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[\s，,、；;：:。.!！?？()（）\-]+", "", normalized)


def evidence_is_present(evidence: str, source: str) -> bool:
    needle = text_key(evidence)
    return bool(needle) and needle in text_key(source)


def normalize_job_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Apply only schema, whitespace, punctuation and exact-deduplication rules."""
    validated = JobFields.model_validate(fields).model_dump(exclude_none=True)
    for field in JSON_FIELDS:
        if field not in validated:
            continue
        values: list[str] = []
        seen: set[str] = set()
        for raw in validated[field] or []:
            value = " ".join(str(raw).split()).strip(" ，,；;。.!！")
            key = text_key(value)
            if value and key and key not in seen:
                seen.add(key)
                values.append(value)
        validated[field] = values
    return JobFields.model_validate(validated).model_dump(exclude_none=True)


def _eligible_fact(fact: SemanticFact, source_text: str) -> bool:
    if fact.category == "unknown" or fact.needs_confirmation:
        return False
    if fact.source_type == "inferred":
        return False
    if fact.source_type == "explicit" and not evidence_is_present(fact.evidence_text, source_text):
        return False
    return True


def guard_model_command(command: StructuredCommand, source_text: str) -> StructuredCommand:
    """Reject unsupported projections while preserving the model's semantic decisions.

    This function never decides what a phrase means.  It only checks that the typed
    decision is traceable to the current user turn and that a fact is projected to
    the field dictated by its declared category.
    """
    fields = normalize_job_fields(command.fields.model_dump(exclude_none=True))
    questions = list(command.clarification_questions)
    eligible: dict[str, set[str]] = {field: set() for field in SEMANTIC_FIELDS}
    guarded_facts: list[SemanticFact] = []
    guarded_coverage = []

    for fact in command.semantic_facts:
        allowed = _eligible_fact(fact, source_text)
        if not allowed and fact.source_type == "explicit":
            fact = fact.model_copy(update={"needs_confirmation": True})
            questions.append(f"请确认这条岗位信息的原文依据：{fact.value}")
        guarded_facts.append(fact)
        target = SEMANTIC_FIELD_BY_CATEGORY.get(fact.category)
        if target and allowed:
            eligible[target].add(text_key(fact.value))

    for field in SEMANTIC_FIELDS:
        if field not in fields:
            continue
        accepted = [value for value in fields[field] if text_key(value) in eligible[field]]
        if len(accepted) != len(fields[field]):
            questions.append("部分岗位内容缺少可核对的原文证据，请补充或确认。")
        fields[field] = accepted

    for item in command.coverage:
        source_item = item.source_item
        offsets_ok = (
            0 <= source_item.source_start < source_item.source_end <= len(source_text)
            and text_key(source_text[source_item.source_start:source_item.source_end])
            == text_key(source_item.text)
        )
        if offsets_ok and evidence_is_present(source_item.text, source_text):
            guarded_coverage.append(item)
        else:
            questions.append(f"JD 第 {source_item.item_index} 条的位置或原文证据需要确认。")

    # Scalar fields are accepted only when the strict model protocol supplied
    # matching, explicit evidence.  Compatibility commands used by deterministic
    # controls/tests may omit field_evidence and are left to Pydantic field guards.
    if command.field_evidence:
        evidence_by_field = {item.field: item for item in command.field_evidence}
        for field in set(fields) - JSON_FIELDS:
            evidence = evidence_by_field.get(field)
            if evidence is None:
                fields.pop(field, None)
                questions.append(f"字段 {field} 缺少原文依据，请重新说明。")
                continue
            same_value = str(evidence.value).casefold() == str(fields[field]).casefold()
            source_ok = evidence.source_type == "user_confirmed" or (
                evidence.source_type == "explicit"
                and evidence_is_present(evidence.evidence_text, source_text)
            )
            if not same_value or not source_ok or evidence.confidence < 0.75:
                fields.pop(field, None)
                questions.append(f"字段 {field} 的依据不充分，请重新说明。")

    return command.model_copy(update={
        "fields": JobFields.model_validate(fields),
        "semantic_facts": guarded_facts,
        "coverage": guarded_coverage,
        "clarification_questions": list(dict.fromkeys(questions)),
    })


def semantic_review(
    fields: dict[str, Any], facts: Iterable[SemanticFact] = (),
) -> list[SemanticIssue]:
    """Deterministic evidence/projection review; no keyword classification."""
    values = {
        field: {text_key(value) for value in fields.get(field, []) or []}
        for field in SEMANTIC_FIELDS
    }
    issues: list[SemanticIssue] = []
    for fact in facts:
        containing = [field for field, items in values.items() if text_key(fact.value) in items]
        if not containing:
            continue
        expected = SEMANTIC_FIELD_BY_CATEGORY.get(fact.category)
        if fact.source_type == "inferred" or fact.needs_confirmation or fact.category == "unknown":
            issues.append(SemanticIssue(
                level="blocking", field=expected, value=fact.value,
                message=f"未经确认或证据不足的信息不能写入岗位：{fact.value}",
            ))
        elif expected and expected not in containing:
            issues.append(SemanticIssue(
                level="blocking", field=expected, value=fact.value,
                message=f"岗位内容与模型给出的分类不一致：{fact.value}",
            ))
    return issues or [SemanticIssue(level="pass", message="证据与字段校验通过")]
