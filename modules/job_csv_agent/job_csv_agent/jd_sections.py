"""Legacy deterministic JD parser, disconnected from the runtime agent path."""

from __future__ import annotations

import re

from .schemas import JDSourceItem


_SECTION_MAP = {
    "职位描述": "job_description",
    "岗位描述": "job_description",
    "岗位职责": "responsibilities",
    "工作职责": "responsibilities",
    "工作内容": "work_content",
    "职位要求": "requirements",
    "岗位要求": "requirements",
    "任职要求": "requirements",
    "任职资格": "qualifications",
    "职位资格": "qualifications",
    "福利待遇": "benefits",
    "薪酬福利": "benefits",
}
_HEADINGS = "|".join(sorted(map(re.escape, _SECTION_MAP), key=len, reverse=True))
_HEADING_RE = re.compile(
    rf"(?m)(?P<prefix>^|\n)\s*(?P<heading>{_HEADINGS})\s*(?:[：:]\s*|(?=\n|$))"
)
_ITEM_RE = re.compile(
    r"(?ms)(?:^|[\n；;])\s*(?:\d{1,3}\s*[、.．)）-]\s*)?(?P<text>.*?)(?=(?:[\n；;]\s*(?:\d{1,3}\s*[、.．)）-]\s*)?)|\Z)"
)


def has_jd_structure(text: str) -> bool:
    return bool(_HEADING_RE.search(text) or re.search(r"(?m)^\s*\d{1,3}\s*[、.．)）-]", text))


def _segment_items(text: str, section: str, offset: int, start_index: int) -> list[JDSourceItem]:
    items: list[JDSourceItem] = []
    for match in _ITEM_RE.finditer(text):
        raw = match.group("text")
        cleaned = re.sub(r"^\s*\d{1,3}\s*[、.．)）-]\s*", "", raw).strip(" \t\r\n，。；;：:")
        if not cleaned:
            continue
        local_start = match.start("text") + max(0, raw.find(cleaned))
        items.append(JDSourceItem(
            text=cleaned,
            section=section,
            item_index=start_index + len(items),
            source_start=offset + local_start,
            source_end=offset + local_start + len(cleaned),
        ))
    return items


def parse_jd_sections(text: str) -> list[JDSourceItem]:
    """Parse headings and numbered source items without rewriting their semantics."""
    source = str(text or "")
    matches = list(_HEADING_RE.finditer(source))
    if not matches:
        return _segment_items(source, "unknown", 0, 1)

    result: list[JDSourceItem] = []
    # Text before the first heading is still accounted for instead of silently dropped.
    prefix = source[:matches[0].start()]
    result.extend(_segment_items(prefix, "unknown", 0, 1))
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        section = _SECTION_MAP[match.group("heading")]
        result.extend(_segment_items(source[start:end], section, start, len(result) + 1))
    # Preserve source order and issue stable, global item positions.
    result.sort(key=lambda item: item.source_start)
    return [item.model_copy(update={"item_index": index}) for index, item in enumerate(result, 1)]
