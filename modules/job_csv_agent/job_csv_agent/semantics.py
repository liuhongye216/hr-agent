from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from .schemas import JobFields, SemanticFact, StructuredCommand


SEMANTIC_FIELD_BY_CATEGORY = {
    "requirement": "requirements_json",
    "responsibility": "responsibilities_json",
    "skill": "skills_json",
    "benefit": "benefits_json",
}

_QUALIFICATION_RE = re.compile(
    r"(?:具备|熟悉|掌握|精通|擅长|学历|学位|经验|证书|执照|资格|资质|优先|"
    r"能力|本科|硕士|博士|大专|高中|中专|CET|英语[四六]级)",
    re.IGNORECASE,
)
_RESPONSIBILITY_RE = re.compile(
    r"(?:负责|完成|搭建|制定|推进|交付|参与|承担|建设|开发|设计|维护|清洗|"
    r"预训练|训练|评估|部署|优化|跟进|产出|实现)",
    re.IGNORECASE,
)
_BENEFIT_RE = re.compile(
    r"(?:五险一金|年终奖|带薪年假|餐补|房补|交通补贴|免费餐|弹性工作|股票期权|"
    r"定期体检|节日福利|团建)",
    re.IGNORECASE,
)
_SKILL_TERMS = (
    "Python", "PyTorch", "TensorFlow", "JAX", "Transformer", "DeepSpeed",
    "Megatron-LM", "CUDA", "SQL", "Java", "C++", "Linux", "Spark", "Flink",
)


def split_atomic_clauses(text: str) -> list[str]:
    """Split only at boundaries that normally separate independently classifiable facts."""
    pieces = re.split(r"[。；;！？!?\n]+", text)
    atoms: list[str] = []
    boundary = re.compile(
        r"[,，](?=\s*(?:负责|完成|搭建|制定|推进|交付|参与|承担|具备|熟悉|掌握|精通|"
        r"本科|硕士|博士|大专|学历|有.+经验|经验|证书|优先|提供|享有))"
    )
    for piece in pieces:
        for atom in boundary.split(piece):
            cleaned = re.sub(r"^\s*(?:\d{1,2}[、.．)）-]\s*)?", "", atom).strip(" ，,:：")
            if cleaned and cleaned not in atoms:
                atoms.append(cleaned)
    return atoms


def looks_like_qualification(value: str) -> bool:
    return bool(_QUALIFICATION_RE.search(value))


def looks_like_responsibility(value: str) -> bool:
    return bool(_RESPONSIBILITY_RE.search(value)) and not looks_like_qualification(value)


def classify_atomic_text(text: str, *, source_type: str = "explicit") -> list[SemanticFact]:
    facts: list[SemanticFact] = []
    for atom in split_atomic_clauses(text):
        if _BENEFIT_RE.search(atom):
            category = "benefit"
            importance = "neutral"
        elif looks_like_qualification(atom):
            category = "requirement"
            importance = "preferred" if "优先" in atom else "must"
        elif _RESPONSIBILITY_RE.search(atom):
            category = "responsibility"
            importance = "neutral"
        elif any(re.search(rf"(?i)(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])", atom) for term in _SKILL_TERMS):
            category = "skill"
            importance = "neutral"
        else:
            category = "unknown"
            importance = "unknown"
        facts.append(SemanticFact(
            value=atom,
            category=category,
            importance=importance,
            source_type=source_type,
            evidence_text=atom,
            needs_confirmation=category == "unknown" or source_type == "inferred",
        ))
        if category == "requirement":
            for term in _SKILL_TERMS:
                if re.search(rf"(?i)(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])", atom):
                    facts.append(SemanticFact(
                        value=term,
                        category="skill",
                        importance=importance,
                        source_type=source_type,
                        evidence_text=atom,
                        needs_confirmation=source_type == "inferred",
                    ))
    return _deduplicate_facts(facts)


def inferred_pretraining_suggestions(facts: Iterable[SemanticFact]) -> tuple[list[SemanticFact], list[str]]:
    has_ambiguous_pretraining_duty = any(
        fact.category == "responsibility"
        and fact.source_type in {"explicit", "user_confirmed"}
        and re.search(r"(?:从|自)\s*[0零]\s*开始|从零", fact.value)
        and "预训练" in fact.value
        for fact in facts
    )
    if not has_ambiguous_pretraining_duty:
        return [], []
    evidence = next(fact.evidence_text for fact in facts if "预训练" in fact.value)
    suggestions = [
        SemanticFact(
            value=value, category="skill", importance="unknown", source_type="inferred",
            evidence_text=evidence, needs_confirmation=True,
        )
        for value in ("Python", "深度学习框架", "Transformer", "数据处理", "分布式训练")
    ]
    questions = [
        "“从0开始”是指从零训练基础模型，还是允许候选人零经验入职？",
        "候选人负责数据、模型、训练、评估中的哪些环节？",
        "哪些技能必须具备，哪些可以入职后学习？",
        "是否要求已有大规模预训练或分布式训练经验？",
    ]
    return suggestions, questions


def project_semantic_facts(facts: Iterable[SemanticFact]) -> dict[str, list[str]]:
    projected: dict[str, list[str]] = {}
    for fact in facts:
        eligible = fact.source_type == "user_confirmed" or (
            fact.source_type == "explicit" and not fact.needs_confirmation
        )
        if not eligible or fact.category == "unknown":
            continue
        field = SEMANTIC_FIELD_BY_CATEGORY.get(fact.category)
        if field is None:
            continue
        projected.setdefault(field, [])
        if fact.value not in projected[field]:
            projected[field].append(fact.value)
    return projected


def reconcile_command_semantics(command: StructuredCommand, original_text: str = "") -> StructuredCommand:
    """Rebuild business JSON fields from semantic facts and deterministic safety checks."""
    facts = list(command.semantic_facts)
    raw_fields = command.fields.model_dump(exclude_none=True)

    # Reclassify model-projected items. This is the deterministic backstop for a model
    # that incorrectly puts an action such as “从0开始预训练大模型” in requirements_json.
    for field in ("requirements_json", "responsibilities_json", "skills_json", "benefits_json"):
        for value in raw_fields.get(field, []) or []:
            classified = classify_atomic_text(value)
            fallback_category = {
                "requirements_json": "requirement",
                "responsibilities_json": "responsibility",
                "skills_json": "skill",
                "benefits_json": "benefit",
            }[field]
            if all(fact.category == "unknown" for fact in classified):
                classified = [SemanticFact(
                    value=value, category=fallback_category, importance="neutral", source_type="explicit",
                    evidence_text=value, needs_confirmation=False,
                )]
            facts.extend(classified)

    if not facts and original_text:
        facts.extend(
            fact for fact in classify_atomic_text(original_text)
            if fact.category != "unknown"
        )
    facts = _deduplicate_facts(facts)
    inferred, suggested_questions = inferred_pretraining_suggestions(facts)
    facts = _deduplicate_facts([*facts, *inferred])

    for field in ("requirements_json", "responsibilities_json", "skills_json", "benefits_json"):
        raw_fields.pop(field, None)
    raw_fields.update(project_semantic_facts(facts))
    fields = JobFields.model_validate(raw_fields)
    questions = list(dict.fromkeys([*command.clarification_questions, *suggested_questions]))
    return command.model_copy(update={
        "fields": fields,
        "semantic_facts": facts,
        "clarification_questions": questions,
    })


def persisted_semantic_facts(facts: Iterable[dict[str, Any] | SemanticFact]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in facts:
        fact = item if isinstance(item, SemanticFact) else SemanticFact.model_validate(item)
        eligible = fact.source_type == "user_confirmed" or (
            fact.source_type == "explicit" and not fact.needs_confirmation
        )
        if eligible and fact.category != "unknown":
            result.append(fact.model_dump(mode="json"))
    return result


def _deduplicate_facts(facts: Iterable[SemanticFact]) -> list[SemanticFact]:
    result: list[SemanticFact] = []
    seen: set[tuple[str, str, str]] = set()
    for fact in facts:
        key = (fact.value.casefold(), fact.category, fact.source_type)
        if key not in seen:
            seen.add(key)
            result.append(fact)
    return result
