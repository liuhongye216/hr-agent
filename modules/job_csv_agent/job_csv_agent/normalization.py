from __future__ import annotations

import re
from typing import Any

from .schemas import AtomicMention, DraftPatch, JSON_FIELDS, PatchItem


_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}


def clean_markdown(value: str) -> str:
    """Remove lightweight Markdown wrappers without rewriting the user's wording."""
    # Preserve a single '~' because it commonly denotes a numeric range (2~4年).
    text = re.sub(r"(?<!\\)(?:[*_`]+|~~+)", "", str(value))
    return " ".join(text.split()).strip()


def chinese_number(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    if text.endswith("半"):
        prefix = chinese_number(text[:-1])
        return (prefix or 0) + 0.5
    if text in _DIGITS:
        return float(_DIGITS[text])
    if "十" in text:
        left, right = text.split("十", 1)
        tens = _DIGITS.get(left, 1) if left else 1
        units = _DIGITS.get(right, 0) if right else 0
        return float(tens * 10 + units)
    return None


_NUMBER = r"(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十]+)"


def parse_experience_months(text: str) -> int | None:
    compact = clean_markdown(text).replace(" ", "")
    bounded = re.search(
        rf"(?P<minimum>{_NUMBER})(?:到|至|[-~～—–]){_NUMBER}年", compact,
    )
    if bounded:
        minimum = chinese_number(bounded.group("minimum"))
        if minimum is not None:
            return int(round(minimum * 12))
    match = re.search(rf"(?P<years>{_NUMBER})年(?P<half>半)?", compact)
    if match:
        years = chinese_number(match.group("years"))
        if years is not None:
            return int(round(years * 12 + (6 if match.group("half") else 0)))
    match = re.search(rf"(?P<months>{_NUMBER})个?月", compact)
    if match:
        months = chinese_number(match.group("months"))
        if months is not None:
            return int(round(months))
    return None


def parse_experience_range(text: str) -> tuple[int | None, int | None] | None:
    """Parse bounded and one-sided experience expressions before single-value parsing."""
    compact = clean_markdown(text).replace(" ", "")
    bounded = re.search(
        rf"(?P<minimum>{_NUMBER})(?:到|至|[-~～—–])(?P<maximum>{_NUMBER})年",
        compact,
    )
    if bounded:
        minimum = chinese_number(bounded.group("minimum"))
        maximum = chinese_number(bounded.group("maximum"))
        if minimum is not None and maximum is not None:
            low, high = sorted((int(round(minimum * 12)), int(round(maximum * 12))))
            return low, high
    upper = re.search(rf"(?P<value>{_NUMBER})年(?:以内|以下|以内经验)", compact)
    if upper:
        value = chinese_number(upper.group("value"))
        if value is not None:
            return None, int(round(value * 12))
    lower = re.search(rf"(?:至少)?(?P<value>{_NUMBER})年(?:以上|及以上|起)", compact)
    if lower:
        value = chinese_number(lower.group("value"))
        if value is not None:
            return int(round(value * 12)), None
    if re.search(r"经验不限|不限经验|无需经验", compact):
        return 0, None
    months = parse_experience_months(compact)
    return (months, None) if months is not None else None


def parse_recruitment(text: str) -> str | None:
    if re.search(r"社会招聘|社招", text):
        return "experienced"
    if re.search(r"校园招聘|校招", text):
        return "campus"
    if "实习招聘" in text:
        return "internship"
    return None


def parse_education_level(text: str) -> int | None:
    if re.search(r"博士(?:学历)?(?:及|或)?以上", text):
        return 6
    if re.search(r"硕士(?:学历)?(?:及|或)?以上", text):
        return 5
    if re.search(r"本科(?:学历)?(?:及|或)?以上", text):
        return 4
    if re.search(r"本科(?:生)?(?:或|和|及)研究生在读", text):
        return 4
    return None


def parse_student_statuses(text: str) -> list[str]:
    compact = clean_markdown(text).replace(" ", "")
    if re.search(r"本科(?:生)?(?:或|和|及)研究生在读", compact):
        return ["本科在读", "研究生在读"]
    statuses: list[str] = []
    for label, pattern in (
        ("本科在读", r"本科(?:生)?在读"),
        ("研究生在读", r"研究生在读|硕士(?:生)?在读"),
        ("应届毕业生", r"应届(?:毕业生)?"),
    ):
        if re.search(pattern, compact):
            statuses.append(label)
    return statuses


def parse_internship_min_months(text: str) -> int | None:
    compact = clean_markdown(text).replace(" ", "")
    match = re.search(rf"(?:至少)?(?:连续)?实习(?P<months>{_NUMBER})个?月", compact)
    if not match:
        return None
    value = chinese_number(match.group("months"))
    return int(round(value)) if value is not None else None


def parse_onsite_days_per_week(text: str) -> int | None:
    compact = clean_markdown(text).replace(" ", "")
    match = re.search(rf"每周(?:至少)?(?:到岗|出勤|坐班)(?P<days>{_NUMBER})天", compact)
    if not match:
        return None
    value = chinese_number(match.group("days"))
    return int(round(value)) if value is not None else None


def split_parallel_items(text: str) -> list[str]:
    """Split coordinated entities while allowing them to inherit a shared predicate."""
    cleaned = clean_markdown(text).strip("：:，,。；; ")
    parts = re.split(r"\s*(?:、|，|,|以及|和|与|及)\s*", cleaned)
    return [part.strip("：:，,。；; ") for part in parts if part.strip("：:，,。；; ")]


def _section_items(text: str, label: str) -> list[str]:
    match = re.search(rf"{label}(?:是|包括|：|:)?(?P<body>[^。！？]+)", text)
    if not match:
        return []
    body = re.sub(r"^\s*", "", match.group("body"))
    parts = re.split(r"\s*(?:、|，|,|；|;|以及|和|并且|并(?=向|能|可|具|独))\s*", body)
    return [part.strip("，,。；; ") for part in parts if part.strip("，,。；; ")]


def infer_atomic_mentions(text: str, source_message_id: str | None = None) -> list[AtomicMention]:
    """Deterministic high-precision mentions for common JD constructions."""
    source = clean_markdown(text)
    mentions: list[AtomicMention] = []

    def add(**data: Any) -> None:
        data.setdefault("source_message_id", source_message_id)
        candidate = AtomicMention.model_validate(data)
        key = (candidate.field, str(candidate.value), tuple(candidate.items), candidate.modality)
        existing = {
            (item.field, str(item.value), tuple(item.items), item.modality) for item in mentions
        }
        if key not in existing:
            mentions.append(candidate)

    for field, value in closed_field_values(source).items():
        add(field=field, raw_text=source, value=value, operation="set", source="normalized")
    statuses = parse_student_statuses(source)
    if statuses:
        add(
            field="student_status_json", raw_text=source, items=statuses,
            operation="append", source="normalized",
        )
    internship_months = parse_internship_min_months(source)
    if internship_months is not None:
        add(
            field="internship_min_months", raw_text=source, value=internship_months,
            operation="set", source="normalized",
        )
    onsite_days = parse_onsite_days_per_week(source)
    if onsite_days is not None:
        add(
            field="onsite_days_per_week", raw_text=source, value=onsite_days,
            operation="set", source="normalized",
        )

    for match in re.finditer(
        r"(?:技能要求(?:是|：|:))?\s*(?:熟练使用|熟练掌握|熟悉|掌握|会使用)"
        r"(?P<body>.*?)(?=[，,。；;]|\s+\d+[.、]|$)",
        source,
    ):
        items = split_parallel_items(match.group("body"))
        if items:
            add(
                field="skills_json", raw_text=match.group(0), items=items,
                operation="append", source="explicit",
            )

    # Numbered full JDs are already explicitly sectioned and are handled by the
    # model without flattening their boundaries. This fallback targets prose JDs.
    if not ("岗位职责" in source and "任职要求" in source):
        for item in _section_items(source, "(?:工作内容|岗位职责|主要负责)"):
            add(
                field="responsibilities_json", raw_text=item, items=[item],
                operation="append", source="explicit",
            )
        for item in _section_items(source, "任职要求"):
            if parse_education_level(item) is not None or parse_student_statuses(item):
                continue
            add(
                field="requirements_json", raw_text=item, items=[item],
                operation="append", source="explicit", modality="required",
            )

    modality_patterns = (
        ("not_required", r"不要求(?P<item>[^，,。；;]+)"),
        ("not_required", r"没有(?P<item>[^，,。；;]+?)也可以"),
        ("preferred", r"(?P<item>[^，,。；;]+?)(?:只是|仅是)?(?:加分项|优先项)"),
        ("required", r"(?:但)?必须(?:能够)?(?P<item>[^，,。；;]+)"),
    )
    for modality, pattern in modality_patterns:
        for match in re.finditer(pattern, source):
            item = re.sub(r"^(?:但|而)", "", match.group("item")).strip("，,。；; ")
            if modality == "preferred":
                item = re.sub(r"(?:只是|仅是)$", "", item).strip()
            if modality == "required":
                item = re.sub(r"^能够", "", item).strip()
            if item:
                add(
                    field="requirements_json", raw_text=match.group(0), items=[item],
                    operation="append", source="explicit", modality=modality,
                )
    return mentions


def closed_field_values(text: str) -> dict[str, Any]:
    """Parse only closed, unambiguous scalar fields; never infer JD section semantics."""
    result: dict[str, Any] = {}
    recruitment = parse_recruitment(text)
    if recruitment is not None:
        result["recruitment"] = recruitment
    education = parse_education_level(text)
    if education is not None:
        result["education_min_level"] = education
    if re.search(r"经验|年限|最低经验", text):
        experience = parse_experience_range(text)
        if experience is not None:
            minimum, maximum = experience
            if minimum is not None:
                result["experience_min_months"] = minimum
            if maximum is not None:
                result["experience_max_months"] = maximum
    internship_months = parse_internship_min_months(text)
    if internship_months is not None:
        result["internship_min_months"] = internship_months
    onsite_days = parse_onsite_days_per_week(text)
    if onsite_days is not None:
        result["onsite_days_per_week"] = onsite_days
    return result


def _merge_mentions(data: dict[str, Any], mentions: list[AtomicMention]) -> None:
    set_fields = data.setdefault("set_fields", {})
    set_sources = data.setdefault("set_sources", {})
    append_items = data.setdefault("append_items", {})
    remove_items = data.setdefault("remove_items", {})
    modality_fields = {
        "required": "requirements_json",
        "preferred": "preferred_requirements_json",
        "not_required": "not_required_requirements_json",
    }
    for mention in mentions:
        field = mention.field
        if field == "requirements_json":
            field = modality_fields[mention.modality]
        if field in JSON_FIELDS:
            values = mention.items or ([] if mention.value is None else [str(mention.value)])
            target = remove_items if mention.operation == "remove" else append_items
            if mention.operation == "remove":
                target.setdefault(field, []).extend(values)
            else:
                target.setdefault(field, []).extend(
                    PatchItem(value=value, source=mention.source).model_dump() for value in values
                )
        elif mention.value is not None and mention.operation == "set":
            set_fields[field] = mention.value
            set_sources[field] = mention.source


def _is_canonicalized_requirement(value: str) -> bool:
    text = re.sub(r"^\s*\d+[.、]\s*", "", clean_markdown(value))
    education_only = re.fullmatch(
        r"(?:本科|硕士|博士)(?:生|学历)?(?:及以上|或以上|以上)?(?:学历)?"
        r"|本科(?:生)?(?:或|和|及)研究生在读",
        text,
    )
    skill_only = re.match(r"^(?:熟悉|熟练使用|熟练掌握|掌握|会使用)", text)
    return education_only is not None or skill_only is not None


def normalize_patch(
    patch: DraftPatch, source_text: str, mentions: list[AtomicMention] | None = None,
) -> DraftPatch:
    data = patch.model_dump(mode="python")
    _merge_mentions(data, mentions or [])
    set_fields = data.setdefault("set_fields", {})
    set_sources = data.setdefault("set_sources", {})

    for field, value in list(set_fields.items()):
        if isinstance(value, str):
            set_fields[field] = clean_markdown(value)
    if "experience_min_months" in set_fields and isinstance(set_fields["experience_min_months"], str):
        parsed = parse_experience_months(set_fields["experience_min_months"])
        if parsed is not None:
            set_fields["experience_min_months"] = parsed
    if "recruitment" in set_fields:
        parsed = parse_recruitment(str(set_fields["recruitment"]))
        if parsed is not None:
            set_fields["recruitment"] = parsed
    if "education_min_level" in set_fields and isinstance(set_fields["education_min_level"], str):
        parsed = parse_education_level(set_fields["education_min_level"])
        if parsed is not None:
            set_fields["education_min_level"] = parsed

    for field, value in closed_field_values(source_text).items():
        # Deterministic closed-field parsing is authoritative only when it preserves
        # the complete meaning (for example both ends of an experience range).
        set_fields[field] = value
        set_sources[field] = "normalized"

    for field, items in data.get("append_items", {}).items():
        data["append_items"][field] = [
            PatchItem(value=clean_markdown(item["value"]), source=item["source"]).model_dump()
            for item in items
            if field != "requirements_json" or not _is_canonicalized_requirement(item["value"])
        ]
    data["append_items"] = {
        field: items for field, items in data.get("append_items", {}).items() if items
    }
    for replacement in data.get("replace_items", []):
        replacement["value"] = clean_markdown(replacement["value"])
        replacement["match"] = clean_markdown(replacement["match"])
    return DraftPatch.model_validate(data)
