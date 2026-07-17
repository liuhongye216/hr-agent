from __future__ import annotations

import re
from typing import Any

from .schemas import DraftPatch, PatchItem


_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}


def clean_markdown(value: str) -> str:
    """Remove lightweight Markdown wrappers without rewriting the user's wording."""
    text = re.sub(r"(?<!\\)[*_`~]+", "", str(value))
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
    return None


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
        months = parse_experience_months(text)
        if months is not None:
            result["experience_min_months"] = months
    return result


def normalize_patch(patch: DraftPatch, source_text: str) -> DraftPatch:
    data = patch.model_dump(mode="python")
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
        set_fields[field] = value
        set_sources[field] = "normalized"

    for field, items in data.get("append_items", {}).items():
        data["append_items"][field] = [
            PatchItem(value=clean_markdown(item["value"]), source=item["source"]).model_dump()
            for item in items
        ]
    for replacement in data.get("replace_items", []):
        replacement["value"] = clean_markdown(replacement["value"])
        replacement["match"] = clean_markdown(replacement["match"])
    return DraftPatch.model_validate(data)
