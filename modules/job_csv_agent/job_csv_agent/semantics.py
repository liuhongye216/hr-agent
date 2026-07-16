from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from .schemas import JobFields, RewriteResult, SemanticFact, StructuredCommand


SEMANTIC_FIELD_BY_CATEGORY = {
    "requirement": "requirements_json",
    "responsibility": "responsibilities_json",
    "skill": "skills_json",
    "benefit": "benefits_json",
}

_QUALIFICATION_RE = re.compile(
    r"(?:具备|熟悉|掌握|精通|擅长|学历|学位|经验|证书|执照|资格|资质|优先|"
    r"能力|本科|硕士|博士|大专|高中|中专|CET|英语[四六]级|懂\s*[A-Za-z]|会用|了解)",
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
    "LLM", "Python", "PyTorch", "TensorFlow", "JAX", "Transformer", "DeepSpeed",
    "Megatron-LM", "CUDA", "SQL", "Java", "C++", "Linux", "Spark", "Flink",
)

_RECRUITMENT_INTENT_RE = re.compile(
    r"^(?:我(?:要|想)(?:招|招聘)|想招(?:聘)?|帮我招|新建(?:一个)?|发布|招聘|招(?:一个|一名)?)"
)
_STRONG_TERMS = ("熟练", "精通", "必须", "独立负责", "专家", "资深")
_HIGH_RISK_RE = re.compile(r"(?:\d+\s*年|本科|硕士|博士|学历|证书|百亿|千亿|薪资|工资)")
_SAFE_REWRITES = {
    ("后训练是训练一个招聘模型", "responsibility"): "负责招聘领域大模型的后训练工作。",
    ("懂llm", "requirement"): "具备大语言模型（LLM）相关基础知识。",
    ("会用python", "requirement"): "能够使用 Python 开展相关开发工作。",
    ("做模型评估", "responsibility"): "负责模型效果评估与结果分析。",
}


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


def is_recruitment_intent_text(text: str) -> bool:
    """Return true for conversational hiring requests that are not JD content."""
    return bool(_RECRUITMENT_INTENT_RE.search("".join(text.split())))


def rewrite_jd_text(
    text: str,
    category: str,
    *,
    source_type: str = "explicit",
    evidence_text: str | None = None,
    model_confidence: float | None = None,
) -> RewriteResult:
    """Normalize JD copy and enforce evidence/risk rules around any rewrite."""
    original = " ".join(text.split()).strip(" ，,。")
    evidence = " ".join((evidence_text or "").split())
    normalized_key = re.sub(r"\s+", "", original).casefold()
    rewritten = _SAFE_REWRITES.get((normalized_key, category), original)
    if rewritten == original:
        rewritten = re.sub(r"^(?:我们?(?:要|想)|这个岗位是|岗位主要是)[:：，,\s]*", "", rewritten).strip()
        if category in {"responsibility", "requirement"} and rewritten and rewritten[-1] not in "。！？":
            rewritten += "。"

    direct_evidence = bool(evidence) and (
        original.casefold() in evidence.casefold() or evidence.casefold() in original.casefold()
    )
    evidence_level = "direct" if direct_evidence else ("contextual" if evidence else "none")
    score = 0.9 if direct_evidence else 0.62 if evidence else 0.35
    if model_confidence is not None:
        # Self-reported confidence is deliberately only a small input.
        score = score * 0.85 + max(0.0, min(1.0, model_confidence)) * 0.15

    comparison_text = evidence or original
    introduced_strong = any(term in rewritten and term not in comparison_text for term in _STRONG_TERMS)
    introduced_high_risk = bool(
        _HIGH_RISK_RE.search(rewritten) and not _HIGH_RISK_RE.search(comparison_text)
    )
    exact_safe = (normalized_key, category) in _SAFE_REWRITES
    if source_type == "inferred":
        return RewriteResult(
            original_text=original, rewritten_text=rewritten, category=category,
            source_type=source_type, evidence_level=evidence_level, confidence=min(score, 0.75),
            decision="needs_confirmation", reason="内容来自岗位上下文建议，尚无用户直接确认。",
        )
    if category == "unknown" or not original or evidence_level == "none":
        return RewriteResult(
            original_text=original or text, rewritten_text=rewritten or text, category=category,
            source_type=source_type, evidence_level=evidence_level, confidence=min(score, 0.54),
            decision="reject_or_clarify", reason="缺少可核对的用户原文或语义类别不明确。",
        )
    if introduced_strong or introduced_high_risk:
        return RewriteResult(
            original_text=original, rewritten_text=rewritten, category=category,
            source_type=source_type, evidence_level=evidence_level, confidence=min(score, 0.54),
            decision="reject_or_clarify", reason="改写新增了招聘门槛、数值或更强语气。",
        )
    if exact_safe or (direct_evidence and score >= 0.85):
        return RewriteResult(
            original_text=original, rewritten_text=rewritten, category=category,
            source_type=source_type, evidence_level=evidence_level, confidence=max(score, 0.9 if exact_safe else score),
            decision="safe_rewrite", reason="仅做等价规范化，保留了用户原意和语气强度。",
        )
    return RewriteResult(
        original_text=original, rewritten_text=rewritten, category=category,
        source_type=source_type, evidence_level=evidence_level, confidence=min(score, 0.84),
        decision="needs_confirmation", reason="内容相关，但改写关系仍需要用户确认。",
    )


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


def inferred_pretraining_suggestions(
    facts: Iterable[SemanticFact], title: str = "",
) -> tuple[list[SemanticFact], list[str]]:
    has_ambiguous_pretraining_duty = any(
        fact.category == "responsibility"
        and fact.source_type in {"explicit", "user_confirmed"}
        and re.search(r"(?:从|自)\s*[0零]\s*开始|从零", fact.value)
        and "预训练" in fact.value
        for fact in facts
    )
    has_post_training = "后训练" in title or any("后训练" in fact.value for fact in facts)
    if has_post_training:
        evidence = next((fact.evidence_text for fact in facts if "后训练" in fact.value), title)
        return [
            SemanticFact(
                value="后训练范围：SFT、RLHF/DPO、奖励模型或模型评估", category="responsibility",
                importance="unknown", source_type="inferred", evidence_text=evidence,
                needs_confirmation=True,
            ),
            SemanticFact(
                value="Python、PyTorch、分布式训练的要求等级", category="skill",
                importance="unknown", source_type="inferred", evidence_text=evidence,
                needs_confirmation=True,
            ),
        ], [
            "请确认后训练范围：SFT、RLHF/DPO、奖励模型、模型评估，还是其他任务？",
            "Python、PyTorch、分布式训练分别是必须、优先，还是不要求？",
        ]
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
    return suggestions, [
        "“从0开始”是指从零训练基础模型，还是允许候选人零经验入职？",
        "候选人负责数据、模型、训练、评估中的哪些环节？",
        "哪些技能必须具备，哪些可以入职后学习？",
    ]


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
    facts: list[SemanticFact] = []
    compact_original = "".join(original_text.split()).casefold()
    for fact in command.semantic_facts:
        if is_recruitment_intent_text(fact.value):
            continue
        compact_evidence = "".join(fact.evidence_text.split()).casefold()
        if (
            original_text and fact.source_type == "explicit"
            and compact_evidence not in compact_original
        ):
            facts.append(fact.model_copy(update={
                "source_type": "inferred", "needs_confirmation": True,
            }))
        else:
            facts.append(fact)
    raw_fields = command.fields.model_dump(exclude_none=True)

    # Reclassify model-projected items. This is the deterministic backstop for a model
    # that incorrectly puts an action such as “从0开始预训练大模型” in requirements_json.
    for field in ("requirements_json", "responsibilities_json", "skills_json", "benefits_json"):
        for value in raw_fields.get(field, []) or []:
            if is_recruitment_intent_text(value):
                continue
            compact_value = "".join(value.split()).casefold()
            source_type = "explicit" if compact_value and compact_value in compact_original else "inferred"
            classified = classify_atomic_text(value, source_type=source_type)
            fallback_category = {
                "requirements_json": "requirement",
                "responsibilities_json": "responsibility",
                "skills_json": "skill",
                "benefits_json": "benefit",
            }[field]
            if all(fact.category == "unknown" for fact in classified):
                classified = [SemanticFact(
                    value=value, category=fallback_category, importance="neutral", source_type=source_type,
                    evidence_text=value if source_type == "explicit" else (original_text or value),
                    needs_confirmation=source_type == "inferred",
                )]
            facts.extend(classified)

    if (
        not facts and original_text and not command.search_query
        and not is_recruitment_intent_text(original_text)
    ):
        facts.extend(
            fact for fact in classify_atomic_text(original_text)
            if fact.category != "unknown"
        )
    recognized_metadata = {
        str(raw_fields.get(key, "")).strip().casefold()
        for key in ("company_name", "title") if raw_fields.get(key)
    }
    facts = [
        fact for fact in facts
        if not (fact.category == "unknown" and fact.value.strip().casefold() in recognized_metadata)
    ]
    rewritten_facts: list[SemanticFact] = []
    for fact in _deduplicate_facts(facts):
        result = rewrite_jd_text(
            fact.value, fact.category, source_type=fact.source_type,
            evidence_text=fact.evidence_text,
        )
        if result.decision == "reject_or_clarify":
            rewritten_facts.append(fact.model_copy(update={"needs_confirmation": True}))
            continue
        rewritten_facts.append(fact.model_copy(update={
            "value": result.rewritten_text,
            "needs_confirmation": result.decision != "safe_rewrite" or fact.needs_confirmation,
        }))
    facts = _deduplicate_facts(rewritten_facts)
    inferred, suggested_questions = inferred_pretraining_suggestions(
        facts, str(raw_fields.get("title", "")),
    )
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
