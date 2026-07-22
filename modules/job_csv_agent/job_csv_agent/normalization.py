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
_REQUIREMENT_MODALITY_FIELDS = {
    "required": "requirements_json",
    "preferred": "preferred_requirements_json",
    "not_required": "not_required_requirements_json",
}
_PREFERRED_SIGNAL_RE = re.compile(
    r"(?:只是|仅是)?(?:加分项|优先项|优选条件)|非硬性要求|"
    r"(?:有|具备).+者优先|(?:经验|年限)(?:者)?优先|优先考虑"
)
_NOT_REQUIRED_SIGNAL_RE = re.compile(r"不要求|无需|没有.+也可以")
_CORRECTION_SIGNAL_RE = re.compile(
    r"刚才|之前|前面|改成|改为|只是|仅是|不是硬性|非硬性|不要求|无需"
)


def _canonical_requirement_text(value: str) -> str:
    """Strip conversational correction framing while preserving the stated fact."""
    text = clean_markdown(value).strip("，,。；;：: ")
    text = re.sub(
        r"^(?:[（(【\[]\s*)?"
        r"(?:加分项|优先项|优选条件|非硬性要求|非必需条件|硬性要求)"
        r"(?:\s*[）)】\]])?\s*(?:[：:\-—]\s*)?",
        "",
        text,
    ).strip()
    text = re.sub(
        r"^(?:我)?(?:刚才|之前|前面)(?:我)?"
        r"(?:说的是|说的|说了|提到的是|提到的|提到|表达的是|表达的|表达了|改成|改为)"
        r"(?:是)?",
        "",
        text,
    ).strip()
    text = re.sub(
        r"(?:只是|仅是)?(?:加分项|优先项|优选条件|非硬性要求)$",
        "",
        text,
    ).strip("，,。；;：: ")
    return text


def _canonical_responsibility_text(value: str) -> str:
    text = clean_markdown(value).strip("，,。；;：: ")
    return re.sub(r"^(?:主要)?负责", "", text).strip() or text


_EXPERIENCE_EXPRESSION_RE = re.compile(
    rf"{_NUMBER}(?:(?:到|至|[-~～—–]){_NUMBER})?年(?:以上|及以上|以下|以内|起)?"
    r"[^，,。；;！？]{0,10}(?:经验|年限)"
)


def _experience_context_is_non_required(source: str, expression: re.Match[str]) -> bool:
    prefix = re.split(r"[，,。；;！？]", source[:expression.start()])[-1][-10:]
    suffix = re.split(r"[，,。；;！？]", source[expression.end():], maxsplit=1)[0][:12]
    window = prefix + expression.group(0) + suffix
    return bool(_PREFERRED_SIGNAL_RE.search(window) or _NOT_REQUIRED_SIGNAL_RE.search(window))


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


def parse_required_experience_range(text: str) -> tuple[int | None, int | None] | None:
    """Select the first hard experience expression without being fooled by preferred clauses."""
    source = clean_markdown(text)
    expressions = list(_EXPERIENCE_EXPRESSION_RE.finditer(source))
    for expression in expressions:
        if _experience_context_is_non_required(source, expression):
            continue
        parsed = parse_experience_range(expression.group(0))
        if parsed is not None:
            return parsed
    if not expressions:
        parsed = parse_experience_range(source)
        if parsed == (0, None):
            return parsed
        if (
            parsed is not None
            and not _PREFERRED_SIGNAL_RE.search(source)
            and not _NOT_REQUIRED_SIGNAL_RE.search(source)
        ):
            return parsed
    return None


def has_experience_context(text: str) -> bool:
    return re.search(
        r"经验|(?:工作|从业|任职|相关|专业|开发|产品)年限", clean_markdown(text),
    ) is not None


def experience_ranges_equivalent_for_reclassification(
    left: tuple[int | None, int | None],
    right: tuple[int | None, int | None],
) -> bool:
    """Treat a model-generated equal upper bound as the same minimum-only fact."""
    if left == right:
        return True
    if left[0] is None or right[0] is None or left[0] != right[0]:
        return False
    return left[1] in {None, left[0]} and right[1] in {None, right[0]}


def _experience_subject(value: str) -> str:
    compact = clean_markdown(value).replace(" ", "")
    compact = re.sub(
        rf"{_NUMBER}年(?:以上|及以上|以下|以内|起)?", "", compact,
    )
    return re.sub(
        r"^(?:要求|必须|需要|需|具备)?(?:至少|最低|不少于|不低于)?", "", compact,
    )


def _same_experience_requirement(left: str, right: str) -> bool:
    left_range = parse_experience_range(left)
    right_range = parse_experience_range(right)
    return bool(
        left_range is not None
        and right_range is not None
        and experience_ranges_equivalent_for_reclassification(left_range, right_range)
        and _experience_subject(left) == _experience_subject(right)
    )


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
    compact = clean_markdown(text).replace(" ", "")
    for label, level in (("博士", 6), ("硕士", 5), ("本科", 4)):
        if re.search(
            rf"(?:最低)?学历(?:要求)?(?:为|是|需|要求|：|:)?{label}(?:学历)?",
            compact,
        ):
            return level
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
    if re.search(r"[?？]|还是|是否|(?:吗|么|呢)(?:[。.!！?？]|$)", source):
        return []
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

    for match in re.finditer(
        r"(?:^|[，,。；;])\s*(?P<item>负责[^，,。；;！？]+)",
        source,
    ):
        item = match.group("item").strip()
        if item:
            add(
                field="responsibilities_json", raw_text=item, items=[item],
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
        (
            "preferred",
            r"(?P<item>[^，,。；;]+?)(?:只是|仅是)?"
            r"(?:加分项|优先项|优选条件|非硬性要求|优先(?:考虑)?)",
        ),
        ("required", r"(?:但)?必须(?:能够)?(?P<item>[^，,。；;]+)"),
    )
    for modality, pattern in modality_patterns:
        for match in re.finditer(pattern, source):
            item = re.sub(r"^(?:但|而)", "", match.group("item")).strip("，,。；; ")
            if modality == "preferred":
                item = _canonical_requirement_text(item)
                if match.group(0).endswith("优先"):
                    item = re.sub(r"者$", "", item).strip()
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
    if has_experience_context(text):
        experience = parse_required_experience_range(text)
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
    for mention in mentions:
        field = mention.field
        if field == "requirements_json":
            field = _REQUIREMENT_MODALITY_FIELDS[mention.modality]
        if field in JSON_FIELDS:
            values = mention.items or ([] if mention.value is None else [str(mention.value)])
            if field in _REQUIREMENT_MODALITY_FIELDS.values():
                values = [_canonical_requirement_text(value) for value in values]
                values = [value for value in values if value]
            target = remove_items if mention.operation == "remove" else append_items
            if mention.operation == "remove":
                target.setdefault(field, []).extend(values)
            else:
                target.setdefault(field, []).extend(
                    PatchItem(value=value, source=mention.source).model_dump() for value in values
                )
        elif mention.operation == "remove":
            clear_fields = data.setdefault("clear_fields", [])
            clear_fields.append(field)
        elif mention.value is not None and mention.operation == "set":
            set_fields[field] = mention.value
            set_sources[field] = mention.source


def _normalize_requirement_patch_items(data: dict[str, Any]) -> None:
    for section in ("append_items", "remove_items"):
        for field in _REQUIREMENT_MODALITY_FIELDS.values():
            values = data.get(section, {}).get(field)
            if not values:
                continue
            if section == "append_items":
                normalized = []
                for item in values:
                    value = _canonical_requirement_text(item["value"])
                    if value:
                        normalized.append({**item, "value": value})
            else:
                normalized = [
                    value
                    for item in values
                    if (value := _canonical_requirement_text(str(item)))
                ]
            data[section][field] = normalized


def _apply_requirement_modality_migrations(
    data: dict[str, Any], mentions: list[AtomicMention], source_text: str,
    current_draft: dict[str, Any] | None,
) -> None:
    append_items = data.setdefault("append_items", {})
    remove_items = data.setdefault("remove_items", {})
    clear_fields = data.setdefault("clear_fields", [])
    set_fields = data.setdefault("set_fields", {})
    set_sources = data.setdefault("set_sources", {})
    current = current_draft or {}
    current_range = (
        current.get("experience_min_months"), current.get("experience_max_months"),
    )
    patch_range = (
        set_fields.get("experience_min_months"), set_fields.get("experience_max_months"),
    )
    hard_range = current_range if any(item is not None for item in current_range) else patch_range
    hard_requirements = current.get("requirements_json") or []

    for mention in mentions:
        if mention.field != "requirements_json" or mention.operation != "append":
            continue
        target = _REQUIREMENT_MODALITY_FIELDS[mention.modality]
        values = mention.items or ([] if mention.value is None else [str(mention.value)])
        values = [
            value
            for item in values
            if (value := _canonical_requirement_text(item))
        ]
        for value in values:
            for field in _REQUIREMENT_MODALITY_FIELDS.values():
                if field == target:
                    continue
                remove_items.setdefault(field, []).append(value)
                append_items[field] = [
                    item
                    for item in append_items.get(field, [])
                    if _canonical_requirement_text(item["value"]) != value
                ]
            target_range = parse_experience_range(value) if re.search(r"经验|年限", value) else None
            reclassifies_existing = (
                mention.source == "contextual"
                or _CORRECTION_SIGNAL_RE.search(source_text) is not None
            )
            if mention.modality != "required" and target_range and reclassifies_existing:
                experience_requirements = [
                    requirement for requirement in hard_requirements
                    if parse_experience_range(requirement) is not None
                ]
                if (
                    not experience_ranges_equivalent_for_reclassification(
                        hard_range, target_range,
                    )
                    or (
                        experience_requirements
                        and not any(
                            _same_experience_requirement(requirement, value)
                            for requirement in experience_requirements
                        )
                    )
                ):
                    continue
                for field in ("experience_min_months", "experience_max_months"):
                    set_fields.pop(field, None)
                    set_sources.pop(field, None)
                    clear_fields.append(field)

    data["append_items"] = {
        field: items for field, items in append_items.items() if items
    }
    data["remove_items"] = {
        field: list(dict.fromkeys(items))
        for field, items in remove_items.items() if items
    }
    data["clear_fields"] = list(dict.fromkeys(clear_fields))


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
    current_draft: dict[str, Any] | None = None,
) -> DraftPatch:
    data = patch.model_dump(mode="python")
    atomic_mentions = mentions or []
    _merge_mentions(data, atomic_mentions)
    _normalize_requirement_patch_items(data)
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

    required_experience = (
        parse_required_experience_range(source_text)
        if has_experience_context(source_text)
        else None
    )
    if required_experience is not None and required_experience[1] is None:
        set_fields.pop("experience_max_months", None)
        set_sources.pop("experience_max_months", None)
        data.setdefault("clear_fields", []).append("experience_max_months")

    # Requirement modality is the final semantic authority. Apply migrations
    # after scalar parsing so an older hard-condition source cannot re-add a
    # value that the user's latest correction moved to preferred/not-required.
    _apply_requirement_modality_migrations(
        data, atomic_mentions, source_text, current_draft,
    )
    for mention in atomic_mentions:
        if mention.operation != "remove" or mention.field in JSON_FIELDS:
            continue
        set_fields.pop(mention.field, None)
        set_sources.pop(mention.field, None)
        data.setdefault("clear_fields", []).append(mention.field)
    data["clear_fields"] = list(dict.fromkeys(data.get("clear_fields", [])))

    for field, items in data.get("append_items", {}).items():
        data["append_items"][field] = [
            PatchItem(
                value=(
                    _canonical_requirement_text(item["value"])
                    if field in _REQUIREMENT_MODALITY_FIELDS.values()
                    else (
                        _canonical_responsibility_text(item["value"])
                        if field == "responsibilities_json"
                        else clean_markdown(item["value"])
                    )
                ),
                source=item["source"],
            ).model_dump()
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
