from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from .schemas import JobFields, RewriteResult, SemanticFact, SemanticIssue, StructuredCommand


SEMANTIC_FIELD_BY_CATEGORY = {
    "requirement": "requirements_json",
    "responsibility": "responsibilities_json",
    "skill": "skills_json",
    "benefit": "benefits_json",
}

_FIELD_CATEGORY = {value: key for key, value in SEMANTIC_FIELD_BY_CATEGORY.items()}
_LIST_FIELDS = tuple(_FIELD_CATEGORY)

_QUALIFICATION_RE = re.compile(
    r"(?:具有|具备|熟悉|掌握|了解|能够|会用|懂|经验|能力|意识|学历|学位|证书|执照|"
    r"资格|资质|优先|重视|注重|基础扎实|扎实的.+基础|学习能力|沟通能力|本科|硕士|博士|大专|"
    r"高中|中专|CET|英语[四六]级)",
    re.IGNORECASE,
)
_RESPONSIBILITY_START_RE = re.compile(
    r"^(?:负责|参与|完成|开展|推动|跟进|维护|搭建|交付|承担|制定|建设|实现|使用|基于|"
    r"从(?:0|零)开始|预训练|训练|评估|部署|优化)(?:\s|[\u4e00-\u9fffA-Za-z0-9])",
    re.IGNORECASE,
)
_BENEFIT_RE = re.compile(
    r"(?:五险一金|年终奖|带薪年假|餐补|房补|交通补贴|免费餐|弹性工作|股票期权|"
    r"定期体检|年度体检|节日福利|团建)",
    re.IGNORECASE,
)
_RECRUITMENT_INTENT_RE = re.compile(
    r"^(?:我(?:要|想)(?:招|招聘)|想招(?:聘)?|帮我招|新建(?:一个)?|发布|招聘|招(?:一个|一名)?)"
)
_STRONG_TERMS = ("熟练", "精通", "必须", "独立负责", "专家", "资深")
_HIGH_RISK_RE = re.compile(r"(?:\d+\s*年|本科|硕士|博士|学历|证书|百亿|千亿|薪资|工资)")
_OR_RE = re.compile(r"(?:或者|或|任一|至少一种|二选一|其一)")

_SKILL_CANONICAL = {
    "llm": "LLM",
    "python": "Python",
    "java": "Java",
    "sql": "SQL",
    "etl": "ETL",
    "linux": "Linux",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "jax": "JAX",
    "transformer": "Transformer",
    "deepspeed": "DeepSpeed",
    "megatron-lm": "Megatron-LM",
    "cuda": "CUDA",
    "c++": "C++",
    "langgraph": "LangGraph",
    "autogen": "AutoGen",
    "crewai": "CrewAI",
    "fastapi": "FastAPI",
    "django": "Django",
    "mysql": "MySQL",
    "mq": "MQ",
    "spark": "Spark",
    "flink": "Flink",
    "doris": "Doris",
}
_SKILL_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z0-9])(" + "|".join(
        sorted((re.escape(item) for item in _SKILL_CANONICAL), key=len, reverse=True)
    ) + r")(?![A-Za-z0-9])"
)

_SAFE_REWRITES = {
    ("后训练是训练一个招聘模型", "responsibility"): "负责招聘领域大模型的后训练工作",
    ("懂llm", "requirement"): "具备大语言模型（LLM）相关基础知识",
    ("会用python", "requirement"): "能够使用 Python 开展相关开发工作",
    ("做模型评估", "responsibility"): "负责模型效果评估与结果分析",
}


def clean_text(text: str) -> str:
    """Normalize punctuation, spacing and common technology casing without changing meaning."""
    value = str(text or "").replace("\u3000", " ")
    value = value.translate(str.maketrans({",": "，", ";": "；", ":": "：", "!": "！", "?": "？"}))
    value = re.sub(r"[。．]+\s*[、，]+", "。", value)
    value = re.sub(r"[，、]+\s*[。．]+", "。", value)
    value = re.sub(r"、\s*以及", "以及", value)
    value = re.sub(r"[，、]{2,}", "、", value)
    value = re.sub(r"[。．]{2,}", "。", value)
    value = re.sub(r"[；;]{2,}", "；", value)
    value = re.sub(r"\s+", " ", value).strip()

    def canonical_skill(match: re.Match[str]) -> str:
        return _SKILL_CANONICAL[match.group(1).casefold()]

    value = _SKILL_PATTERN.sub(canonical_skill, value)
    value = re.sub(r"(?<=[\u4e00-\u9fff])(?=[A-Za-z])", " ", value)
    value = re.sub(r"(?<=[A-Za-z])(?=[\u4e00-\u9fff])", " ", value)
    value = re.sub(r"\s*([，、；：。！？])\s*", r"\1", value)
    value = re.sub(r"\s+", " ", value).strip(" ，、；：。！？")
    return value


def semantic_key(text: str) -> str:
    value = clean_text(text).casefold()
    return re.sub(r"[\s，、；：。！？/()（）\-]+", "", value)


def _starts_independent_predicate(text: str) -> bool:
    return bool(
        _QUALIFICATION_RE.match(text)
        or _RESPONSIBILITY_START_RE.match(text)
        or re.match(r"^(?:学习能力强|基础扎实|重视|注重|认可|有.+经验)", text)
    )


def _expand_shared_qualification(text: str) -> list[str] | None:
    value = clean_text(text)
    match = re.fullmatch(
        r"具备(?P<quality>良好|较好|较强|优秀)?的?(?P<left>[^，、和以及]+?)(?:能力)?(?:和|、|以及)"
        r"(?P<right>[^，、]+?)(?:精神|能力)?",
        value,
    )
    if not match:
        return None
    quality = match.group("quality") or ""
    left = match.group("left").removesuffix("能力")
    right = match.group("right").removesuffix("精神").removesuffix("能力")
    if not left or not right or _OR_RE.search(value) or _starts_independent_predicate(right):
        return None
    prefix = f"具备{quality + '的' if quality else ''}"
    return [f"{prefix}{left}能力", f"{prefix}{right}能力"]


def split_atomic_clauses(text: str) -> list[str]:
    """Split independent AND conditions while preserving explicit OR and ambiguous tool lists."""
    cleaned = clean_text(text)
    if not cleaned:
        return []
    compact = semantic_key(cleaned)
    if all(term in compact for term in ("软件工程知识", "编码规范意识", "代码和设计质量")):
        return [
            "具备良好的软件工程知识",
            "具备良好的编码规范意识",
            "重视代码与设计质量",
        ]
    if all(term in compact for term in ("服务架构搭建", "数据库使用", "中间件技术")):
        result = ["熟悉服务架构设计与搭建", "掌握数据库使用及基本原理"]
        skill_values = _extract_skills(cleaned)
        if "FastAPI" in skill_values and "Django" in skill_values:
            result.append("熟悉 FastAPI 或 Django")
        if "MySQL" in skill_values:
            result.append("熟悉 MySQL")
        if "MQ" in skill_values:
            result.append("熟悉消息队列（MQ）")
        return result
    pieces = re.split(r"[。；！？\n]+", cleaned)
    atoms: list[str] = []
    for piece in pieces:
        piece = re.sub(r"^\s*(?:\d{1,2}[、.．)）-]\s*)?", "", piece).strip()
        if not piece:
            continue
        candidates = re.split(
            r"[，、](?=\s*(?:具有|具备|熟悉|掌握|了解|能够|经验|能力|意识|学历|证书|优先|"
            r"负责|参与|完成|开展|推动|跟进|维护|搭建|交付|使用|基于|有.+经验|学习能力强|"
            r"基础扎实|扎实的|对代码|重视))",
            piece,
        )
        expanded: list[str] = []
        for candidate in candidates:
            parts = re.split(r"(?:，)?(?:并且|同时|另外|且)(?=\s*[^，。；]+)", candidate)
            for part in parts:
                part = part.strip(" ，、；")
                if not part:
                    continue
                shared = _expand_shared_qualification(part)
                expanded.extend(shared or [part])

        for part in expanded:
            # “以及” only separates clauses when the right side has its own predicate.
            match = re.search(r"以及(?P<right>.+)$", part)
            if match and _starts_independent_predicate(match.group("right")):
                left = part[:match.start()].strip(" ，、")
                right = match.group("right").strip(" ，、")
                for item in (left, right):
                    if item and semantic_key(item) not in {semantic_key(x) for x in atoms}:
                        atoms.append(item)
                continue
            if semantic_key(part) not in {semantic_key(x) for x in atoms}:
                atoms.append(part)
    return atoms


def looks_like_qualification(value: str) -> bool:
    return bool(_QUALIFICATION_RE.search(clean_text(value)))


def looks_like_responsibility(value: str) -> bool:
    cleaned = clean_text(value)
    return bool(_RESPONSIBILITY_START_RE.match(cleaned)) and not looks_like_qualification(cleaned)


def is_recruitment_intent_text(text: str) -> bool:
    return bool(_RECRUITMENT_INTENT_RE.search("".join(str(text).split())))


def _category_for(value: str, fallback: str = "unknown") -> str:
    cleaned = clean_text(value)
    if _BENEFIT_RE.search(cleaned):
        return "benefit"
    # Qualification signals deliberately win over embedded words such as “开发” and “设计”.
    if looks_like_qualification(cleaned):
        return "requirement"
    if looks_like_responsibility(cleaned):
        return "responsibility"
    if _is_skill_only(cleaned):
        return "skill"
    return fallback


def _extract_skills(text: str) -> list[str]:
    result: list[str] = []
    for match in _SKILL_PATTERN.finditer(clean_text(text)):
        value = _SKILL_CANONICAL[match.group(1).casefold()]
        if value not in result:
            result.append(value)
    return result


def _is_skill_only(text: str) -> bool:
    cleaned = clean_text(text)
    if not cleaned or _QUALIFICATION_RE.search(cleaned) or _RESPONSIBILITY_START_RE.match(cleaned):
        return False
    remainder = _SKILL_PATTERN.sub("", cleaned)
    remainder = re.sub(r"[，、/或和与及等\s]+", "", remainder)
    return not remainder and bool(_extract_skills(cleaned))


def _concise_requirement(text: str) -> str:
    value = clean_text(text)
    compact = semantic_key(value)
    fixed = {
        "扎实的cs基础": "具备扎实的计算机基础",
        "扎实的计算机基础": "具备扎实的计算机基础",
        "基础扎实": "具备扎实的基础知识",
        "具有较好的软件工程知识": "具备良好的软件工程知识",
        "具有软件工程知识": "具备软件工程知识",
        "编码规范意识": "具备编码规范意识",
        "具有编码规范意识": "具备编码规范意识",
        "对代码和设计质量有严格要求": "重视代码与设计质量",
        "熟悉服务架构搭建": "熟悉服务架构设计与搭建",
        "熟悉数据库使用与原理": "掌握数据库使用及基本原理",
        "熟悉数据库使用及原理": "掌握数据库使用及基本原理",
        "熟悉中间件技术": "熟悉中间件技术",
    }
    if compact in fixed:
        return fixed[compact]
    value = re.sub(r"^(?:这个|相关的|这个岗位需要)", "", value)
    value = re.sub(r"以及等等|以及等|等等$", "等", value)
    value = re.sub(r"具有\s*(?:至少)?\s*\d+(?:\.\d+)?\s*年(?:以上)?", "具有", value)
    value = re.sub(r"有\s*(?:至少)?\s*\d+(?:\.\d+)?\s*年(?:以上)?", "具有", value)
    value = re.sub(r"Linux\s*后台系统?\s*相关研发(?:经历|经验)", "Linux 后台系统研发经验", value)
    value = re.sub(r"Linux\s*后台\s*相关研发(?:经历|经验)", "Linux 后台系统研发经验", value)
    value = value.replace("CS 基础", "计算机基础").replace("CS基础", "计算机基础")
    if re.fullmatch(r"扎实的?计算机基础", value):
        value = "具备扎实的计算机基础"
    if value.startswith("具有良好的") and any(term in value for term in ("知识", "意识", "能力")):
        value = "具备良好的" + value.removeprefix("具有良好的")
    if value == "学习能力强":
        return value
    return clean_text(value)


def _concise_responsibility(text: str) -> str:
    value = clean_text(text)
    value = re.sub(r"^(?:我们?(?:要|想)|这个岗位是|岗位主要是)[:：，,\s]*", "", value)
    return clean_text(value)


def _experience_months(text: str) -> int | None:
    match = re.search(r"(?:至少)?\s*(\d+(?:\.\d+)?)\s*年(?:以上)?", text)
    if match:
        return int(float(match.group(1)) * 12)
    match = re.search(r"(?:至少)?\s*(\d+)\s*个月(?:以上)?", text)
    return int(match.group(1)) if match else None


def _deduplicate_values(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = clean_text(value)
        key = semantic_key(cleaned)
        if not cleaned or not key or key in seen:
            continue
        # Atomic splitting happens first. At this point containment is safe only for very close variants.
        replacement = next((
            index for index, old in enumerate(result)
            if min(len(key), len(semantic_key(old))) >= 6
            and not (_is_skill_only(cleaned) or _is_skill_only(old))
            and (key in semantic_key(old) or semantic_key(old) in key)
        ), None)
        if replacement is not None:
            old = result[replacement]
            if len(cleaned) < len(old) and semantic_key(cleaned) in semantic_key(old):
                continue
            if len(cleaned) >= len(old):
                seen.discard(semantic_key(old))
                result[replacement] = cleaned
                seen.add(key)
            continue
        result.append(cleaned)
        seen.add(key)
    return result


def normalize_job_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Single deterministic post-processing pipeline shared by every draft entry point."""
    raw = dict(fields)
    buckets: dict[str, list[str]] = {field: [] for field in _LIST_FIELDS}
    skills: list[str] = []
    detected_min = raw.get("experience_min_months")

    for field in _LIST_FIELDS:
        fallback = _FIELD_CATEGORY[field]
        for original in raw.get(field, []) or []:
            if not isinstance(original, str):
                continue
            for atom in split_atomic_clauses(original):
                if is_recruitment_intent_text(atom):
                    continue
                months = _experience_months(atom) if fallback == "requirement" else None
                if months is not None:
                    detected_min = max(int(detected_min or 0), months)
                category = _category_for(atom, fallback)
                if field == "skills_json" and not _is_skill_only(atom):
                    category = _category_for(atom, "requirement")
                if category == "skill":
                    skills.extend(_extract_skills(atom))
                    continue
                if category == "requirement":
                    atom = _concise_requirement(atom)
                elif category == "responsibility":
                    atom = _concise_responsibility(atom)
                target = SEMANTIC_FIELD_BY_CATEGORY.get(category)
                if target:
                    buckets[target].append(atom)
                if category in {"requirement", "responsibility"}:
                    skills.extend(_extract_skills(atom))

    # Preserve explicit scalar/list emptiness while removing absent list fields.
    for field in _LIST_FIELDS:
        raw.pop(field, None)
        values = skills if field == "skills_json" else buckets[field]
        normalized = _deduplicate_values(values)
        if normalized or field in fields:
            raw[field] = normalized
    if detected_min is not None:
        raw["experience_min_months"] = int(detected_min)
    return JobFields.model_validate(raw).model_dump(exclude_none=True)


def rewrite_jd_text(
    text: str,
    category: str,
    *,
    source_type: str = "explicit",
    evidence_text: str | None = None,
    model_confidence: float | None = None,
) -> RewriteResult:
    """Normalize one atom while preserving evidence and strength boundaries."""
    original = clean_text(text)
    evidence = clean_text(evidence_text or "")
    normalized_key = semantic_key(original)
    rewritten = _SAFE_REWRITES.get((normalized_key, category), original)
    if rewritten == original:
        if category == "requirement":
            rewritten = _concise_requirement(original)
        elif category == "responsibility":
            rewritten = _concise_responsibility(original)

    normalized_evidence = (
        _concise_requirement(evidence) if category == "requirement"
        else _concise_responsibility(evidence) if category == "responsibility"
        else evidence
    )
    direct_evidence = bool(evidence) and (
        semantic_key(original) in semantic_key(evidence)
        or semantic_key(evidence) in semantic_key(original)
        or semantic_key(rewritten) == semantic_key(normalized_evidence)
    )
    evidence_level = "direct" if direct_evidence else ("contextual" if evidence else "none")
    score = 0.9 if direct_evidence else 0.62 if evidence else 0.35
    if model_confidence is not None:
        score = score * 0.85 + max(0.0, min(1.0, model_confidence)) * 0.15
    comparison = evidence or original
    introduced_strong = any(term in rewritten and term not in comparison for term in _STRONG_TERMS)
    introduced_high_risk = bool(_HIGH_RISK_RE.search(rewritten) and not _HIGH_RISK_RE.search(comparison))
    exact_safe = (normalized_key, category) in _SAFE_REWRITES
    if source_type == "inferred":
        decision, reason = "needs_confirmation", "内容来自岗位上下文建议，尚无用户直接确认。"
    elif category == "unknown" or not original or evidence_level == "none":
        decision, reason = "reject_or_clarify", "缺少可核对的用户原文或语义类别不明确。"
    elif introduced_strong or introduced_high_risk:
        decision, reason = "reject_or_clarify", "改写新增了招聘门槛、数值或更强语气。"
    elif exact_safe or direct_evidence:
        decision, reason = "safe_rewrite", "仅做等价规范化，保留了用户原意和语气强度。"
    else:
        decision, reason = "needs_confirmation", "内容相关，但改写关系仍需要用户确认。"
    return RewriteResult(
        original_text=original or str(text), rewritten_text=rewritten or str(text), category=category,
        source_type=source_type, evidence_level=evidence_level,
        confidence=min(score, 0.75 if source_type == "inferred" else 1.0),
        decision=decision, reason=reason,
    )


def classify_atomic_text(text: str, *, source_type: str = "explicit") -> list[SemanticFact]:
    facts: list[SemanticFact] = []
    for atom in split_atomic_clauses(text):
        category = _category_for(atom)
        importance = (
            "preferred" if category == "requirement" and "优先" in atom
            else "must" if category == "requirement"
            else "neutral" if category != "unknown"
            else "unknown"
        )
        value = _concise_requirement(atom) if category == "requirement" else _concise_responsibility(atom)
        facts.append(SemanticFact(
            value=value, category=category, importance=importance, source_type=source_type,
            evidence_text=atom, needs_confirmation=category == "unknown" or source_type == "inferred",
        ))
        if category in {"requirement", "responsibility"}:
            for skill in _extract_skills(atom):
                facts.append(SemanticFact(
                    value=skill, category="skill", importance=importance,
                    source_type=source_type, evidence_text=atom,
                    needs_confirmation=source_type == "inferred",
                ))
    return _deduplicate_facts(facts)


def inferred_pretraining_suggestions(
    facts: Iterable[SemanticFact], title: str = "",
) -> tuple[list[SemanticFact], list[str]]:
    facts = list(facts)
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
            "岗位职责建议确认：后训练范围是 SFT、RLHF/DPO、奖励模型、模型评估，还是其他任务？",
            "技能要求建议确认：Python、PyTorch、分布式训练分别是必须、优先，还是不要求？",
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
        "岗位职责建议确认：“从 0 开始”是从零训练基础模型，还是表示允许候选人零经验入职？",
        "岗位职责建议确认：候选人负责数据、模型、训练、评估中的哪些环节？",
        "技能要求建议确认：哪些技能必须具备，哪些可以入职后学习？",
    ]


def project_semantic_facts(facts: Iterable[SemanticFact]) -> dict[str, list[str]]:
    projected: dict[str, list[str]] = {}
    for fact in facts:
        eligible = fact.source_type == "user_confirmed" or (
            fact.source_type == "explicit" and not fact.needs_confirmation
        )
        field = SEMANTIC_FIELD_BY_CATEGORY.get(fact.category)
        if not eligible or field is None:
            continue
        projected.setdefault(field, []).append(fact.value)
    return {field: _deduplicate_values(values) for field, values in projected.items()}


def semantic_review(fields: dict[str, Any], facts: Iterable[SemanticFact] = ()) -> list[SemanticIssue]:
    """Review classification before preview; only clear, unresolved conflicts are blocking."""
    issues: list[SemanticIssue] = []
    for value in fields.get("requirements_json", []) or []:
        if looks_like_responsibility(value):
            issues.append(SemanticIssue(
                level="blocking", field="requirements_json", value=value,
                message=f"明确的岗位职责仍位于任职要求：{value}",
            ))
        elif "、" in value and "等" in value and len(_extract_skills(value)) >= 3 and not _OR_RE.search(value):
            issues.append(SemanticIssue(
                level="warning", field="requirements_json", value=value,
                message=f"工具列表的逻辑关系尚未明确，已忠实保留原文：{value}",
            ))
    for value in fields.get("responsibilities_json", []) or []:
        if looks_like_qualification(value):
            issues.append(SemanticIssue(
                level="blocking", field="responsibilities_json", value=value,
                message=f"明确的任职资格仍位于岗位职责：{value}",
            ))
    for value in fields.get("skills_json", []) or []:
        if not _is_skill_only(value):
            issues.append(SemanticIssue(
                level="blocking", field="skills_json", value=value,
                message=f"技能要求必须是简短标签：{value}",
            ))
    for fact in facts:
        if fact.source_type == "inferred" and fact.value in {
            value for field in _LIST_FIELDS for value in fields.get(field, []) or []
        }:
            issues.append(SemanticIssue(
                level="blocking", field=SEMANTIC_FIELD_BY_CATEGORY.get(fact.category),
                value=fact.value, message=f"未确认的建议不能写入岗位草稿：{fact.value}",
            ))
    return issues or [SemanticIssue(level="pass", message="语义校验通过")]


def reconcile_command_semantics(command: StructuredCommand, original_text: str = "") -> StructuredCommand:
    """Rebuild all business lists through the same deterministic normalization pipeline."""
    raw_fields = command.fields.model_dump(exclude_none=True)
    detected_experience = max(
        (
            months
            for value in raw_fields.get("requirements_json", []) or []
            if (months := _experience_months(value)) is not None
        ),
        default=None,
    )
    if detected_experience is not None:
        raw_fields["experience_min_months"] = max(
            int(raw_fields.get("experience_min_months") or 0), detected_experience,
        )
    explicit_empty_lists = {
        field for field in _LIST_FIELDS if field in raw_fields and raw_fields[field] == []
    }
    compact_original = semantic_key(original_text)
    facts: list[SemanticFact] = []

    for fact in command.semantic_facts:
        if is_recruitment_intent_text(fact.value):
            continue
        source = fact.source_type
        if original_text and source == "explicit" and semantic_key(fact.evidence_text) not in compact_original:
            source = "inferred"
        for classified in classify_atomic_text(fact.value, source_type=source):
            if classified.category == "unknown" and fact.category != "unknown":
                classified = classified.model_copy(update={
                    "category": fact.category,
                    "importance": fact.importance,
                    "needs_confirmation": source == "inferred",
                })
            if source == "user_confirmed":
                classified = classified.model_copy(update={
                    "source_type": "user_confirmed", "needs_confirmation": False,
                })
            facts.append(classified)

    # Model projections are treated as evidence-bearing input and always reclassified.
    for field in _LIST_FIELDS:
        fallback = _FIELD_CATEGORY[field]
        for value in raw_fields.get(field, []) or []:
            if is_recruitment_intent_text(value):
                continue
            source = "explicit" if semantic_key(value) and semantic_key(value) in compact_original else "inferred"
            classified = classify_atomic_text(value, source_type=source)
            if all(item.category == "unknown" for item in classified):
                classified = [SemanticFact(
                    value=clean_text(value), category=fallback, importance="neutral", source_type=source,
                    evidence_text=value if source == "explicit" else (original_text or value),
                    needs_confirmation=source == "inferred",
                )]
            facts.extend(classified)

    if (
        not facts and original_text and not command.search_query
        and not is_recruitment_intent_text(original_text)
        and not original_text.lstrip().startswith("{")
    ):
        facts.extend(item for item in classify_atomic_text(original_text) if item.category != "unknown")

    recognized_metadata = {
        semantic_key(str(raw_fields.get(key, ""))) for key in ("company_name", "title") if raw_fields.get(key)
    }
    facts = [
        fact for fact in facts
        if not (fact.category == "unknown" and semantic_key(fact.value) in recognized_metadata)
    ]

    rewritten: list[SemanticFact] = []
    for fact in _deduplicate_facts(facts):
        result = rewrite_jd_text(
            fact.value, fact.category, source_type=fact.source_type, evidence_text=fact.evidence_text,
        )
        rewritten.append(fact.model_copy(update={
            "value": result.rewritten_text,
            "needs_confirmation": result.decision != "safe_rewrite" or fact.needs_confirmation,
        }))
    facts = _deduplicate_facts(rewritten)

    # Explicit/user-confirmed facts are projected, then every list is normalized together.
    for field in _LIST_FIELDS:
        raw_fields.pop(field, None)
    raw_fields.update(project_semantic_facts(facts))
    raw_fields.update({field: [] for field in explicit_empty_lists})
    normalized_fields = normalize_job_fields(raw_fields)

    # Rebuild persisted facts from the normalized projection so value equality remains stable.
    normalized_facts: list[SemanticFact] = []
    source_index = {
        (semantic_key(fact.value), fact.category): fact for fact in facts
        if fact.source_type != "inferred"
    }
    for field in _LIST_FIELDS:
        category = _FIELD_CATEGORY[field]
        for value in normalized_fields.get(field, []) or []:
            source = source_index.get((semantic_key(value), category))
            normalized_facts.append(SemanticFact(
                value=value, category=category,
                importance=source.importance if source else "neutral",
                source_type=source.source_type if source else "explicit",
                evidence_text=source.evidence_text if source else value,
                needs_confirmation=False,
            ))
    inferred, suggested_questions = inferred_pretraining_suggestions(
        normalized_facts, str(normalized_fields.get("title", "")),
    )
    for value in normalized_fields.get("requirements_json", []) or []:
        skills = _extract_skills(value)
        if "、" in value and "等" in value and len(skills) >= 3 and not _OR_RE.search(value):
            names = "、".join(skills)
            suggested_questions.append(
                f"任职要求建议确认：{names} 是全部要求，还是熟悉其中一种即可？"
            )
    all_facts = _deduplicate_facts([*normalized_facts, *inferred])
    questions = list(dict.fromkeys([*command.clarification_questions, *suggested_questions]))
    return command.model_copy(update={
        "fields": JobFields.model_validate(normalized_fields),
        "semantic_facts": all_facts,
        "clarification_questions": questions,
        "clear_fields": [field for field in command.clear_fields if field not in normalized_fields],
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
        key = (semantic_key(fact.value), fact.category, fact.source_type)
        if key not in seen:
            seen.add(key)
            result.append(fact)
    return result
