from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Literal

from .config import BUSINESS_JOBS_CSV, SOURCE_CSV, SOURCE_JSONL
from .llm import extract_long_text_with_llm


SOURCE_FIRST_COLUMNS = (
    "job_id",
    "title",
    "salary",
    "city",
    "experience",
    "education",
    "company_name",
    "work_address",
    "jd_raw",
    "source_url",
    "scraped_at",
)

BUSINESS_COLUMNS = (
    "job_id",
    "title",
    "company_name",
    "city",
    "work_address",
    "salary_min",
    "salary_max",
    "salary_currency",
    "salary_period",
    "recruitment",
    "employment",
    "work_mode",
    "education_min_level",
    "experience_min_months",
    "experience_max_months",
    "requirements_json",
    "responsibilities_json",
    "skills_json",
    "certificates_json",
    "benefits_json",
    "source_url",
    "scraped_at",
    "content_hash",
    "extraction_mode",
)

RESPONSIBILITY_MARKERS = ("岗位职责", "工作职责", "工作内容", "职位描述", "岗位描述", "职位内容")
REQUIREMENT_MARKERS = ("岗位要求", "任职要求", "任职资格", "职位要求", "招聘要求", "基本要求")
SECTION_MARKERS = RESPONSIBILITY_MARKERS + REQUIREMENT_MARKERS + (
    "薪资福利",
    "福利待遇",
    "岗位亮点",
    "公司介绍",
)

SKILL_TERMS = (
    "Excel", "Word", "PPT", "PowerPoint", "Python", "Java", "JavaScript", "TypeScript",
    "SQL", "CAD", "SolidWorks", "Photoshop", "PS", "AI", "剪映", "ERP", "CRM", "英语",
)
CERTIFICATE_TERMS = (
    "健康证", "教师资格证", "毕业证", "学位证", "驾驶证", "会计证", "电工证", "焊工证",
    "普通话证书", "英语四级", "英语六级", "CET-4", "CET-6",
)


def _json_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _write_csv(path: Path, columns: Iterable[str], rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8-sig",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _json_cell(row.get(key)) for key in writer.fieldnames})
                count += 1
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return count


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            job_id = str(row.get("job_id") or "").strip()
            if not job_id:
                raise ValueError(f"Missing job_id at {path}:{line_number}")
            if job_id in seen:
                raise ValueError(f"Duplicate job_id {job_id!r} at {path}:{line_number}")
            seen.add(job_id)
            rows.append(row)
    return rows


def export_source_csv(
    source_jsonl: Path = SOURCE_JSONL,
    output_path: Path = SOURCE_CSV,
) -> dict[str, Any]:
    """Losslessly flatten JSONL top-level fields into a reviewable source CSV."""
    rows = _read_jsonl(source_jsonl)
    discovered: list[str] = []
    for row in rows:
        for key in row:
            if key not in discovered:
                discovered.append(key)
    columns = [key for key in SOURCE_FIRST_COLUMNS if key in discovered]
    columns.extend(key for key in discovered if key not in columns)
    count = _write_csv(output_path, columns, rows)
    return {"source_rows": count, "source_columns": len(columns), "output": str(output_path)}


def _read_source_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _number(value: str, unit: str) -> float:
    number = float(value)
    if unit.lower() == "k" or unit == "千":
        number *= 1_000
    elif unit.lower() == "w" or unit == "万":
        number *= 10_000
    return number


def _parse_salary(text: str) -> tuple[float | None, float | None, str | None, str | None]:
    value = _clean(text)
    period = "per_order" if re.search(r"/\s*(?:单|件)|每单|按单", value) else None
    if period is None:
        if re.search(r"/\s*(?:天|日)|元/天", value):
            period = "day"
        elif re.search(r"/\s*(?:小时|时)", value):
            period = "hour"
        elif re.search(r"/\s*年", value):
            period = "year"
        else:
            period = "month"
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*([kKwW万千]?)\s*[-~—–至到]\s*(\d+(?:\.\d+)?)\s*([kKwW万千]?)",
        value,
    )
    if match:
        left_unit = match.group(2) or match.group(4)
        right_unit = match.group(4) or match.group(2)
        return _number(match.group(1), left_unit), _number(match.group(3), right_unit), "CNY", period
    match = re.search(r"(\d+(?:\.\d+)?)\s*([kKwW万千]?)\s*(?:以上|起|\+)", value)
    if match:
        return _number(match.group(1), match.group(2)), None, "CNY", period
    return None, None, None, None


def _education_level(text: str) -> int | None:
    value = _clean(text)
    if re.search(r"不限|初中及以下", value):
        return 0
    for pattern, level in (
        (r"初中", 1), (r"高中|中专|中技|中职", 2), (r"大专|专科", 3),
        (r"本科", 4), (r"硕士|研究生", 5), (r"博士", 6),
    ):
        if re.search(pattern, value):
            return level
    return None


def _experience_range(value: str, jd_raw: str) -> tuple[int | None, int | None]:
    if re.search(r"经验不限|无需经验|无经验(?:也)?可|有无经验(?:均|都)?可|接受无经验", jd_raw):
        return 0, None
    text = _clean(value)
    match = re.search(r"(\d+)\s*[-~—–至到]\s*(\d+)\s*年", text)
    if match:
        return int(match.group(1)) * 12, int(match.group(2)) * 12
    match = re.search(r"(\d+)\s*年以内", text)
    if match:
        return None, int(match.group(1)) * 12
    match = re.search(r"(\d+)\s*年以上", text)
    if match:
        return int(match.group(1)) * 12, None
    if "经验不限" in text:
        return 0, None
    return None, None


def _employment(row: dict[str, str]) -> str:
    text = f"{row.get('title', '')} {row.get('jd_raw', '')}"
    full = "全职" in text
    part = bool(re.search(r"兼职|小时工|半日工", text))
    if full and part:
        return "full_or_part_time"
    if part:
        return "part_time"
    if re.search(r"实习生|实习岗位|日常实习", row.get("title", "")):
        return "internship"
    if "合同工" in text:
        return "contract"
    if "临时工" in text:
        return "temporary"
    return "full_time" if full else "unknown"


def _recruitment(row: dict[str, str]) -> str:
    text = f"{row.get('title', '')} {row.get('experience', '')} {row.get('jd_raw', '')}"
    internship = bool(re.search(r"实习|在校", text))
    campus = bool(re.search(r"应届|校招|毕业生", text))
    experienced = bool(re.search(r"社招|社会招聘|工作经验", text))
    if sum((internship or campus, experienced)) > 1:
        return "mixed"
    if internship:
        return "internship"
    if campus:
        return "campus"
    if experienced:
        return "experienced"
    return "unknown"


def _work_mode(text: str) -> str:
    if "远程" in text:
        return "hybrid" if re.search(r"混合|部分到岗|线下", text) else "remote"
    return "onsite" if re.search(r"到岗|坐班|办公地点|工作地址", text) else "unknown"


def _heading(raw: str, marker: str, offset: int = 0) -> re.Match[str] | None:
    pattern = rf"(?:^|[\s【\[。；;！？!?])({re.escape(marker)})(?=\s|[】\]:：]|$)\s*(?:[】\]]|[:：])?"
    return re.search(pattern, raw[offset:])


def _extract_section(raw: str, starts: tuple[str, ...]) -> str:
    selected = next(((marker, _heading(raw, marker)) for marker in starts if _heading(raw, marker)), None)
    if selected is None:
        return ""
    # Marker order expresses specificity. For example, prefer “工作内容” over an
    # earlier generic “职位描述” that may only contain company background.
    _, match = selected
    start = match.end()
    stops = []
    for marker in SECTION_MARKERS:
        stop = _heading(raw, marker, start)
        if stop:
            stops.append(start + stop.start())
    end = min(stops) if stops else len(raw)
    return raw[start:end]


def _split_items(text: str) -> list[str]:
    items = re.split(r"[；;。！？!?\n]|(?=\d{1,2}\s*[、.．)）](?!\d))", text)
    result: list[str] = []
    for item in items:
        cleaned = re.sub(r"^\s*\d{1,2}\s*[、.．)）-]\s*", "", _clean(item)).strip(" ：:;；")
        if 4 <= len(cleaned) <= 220 and cleaned not in result:
            result.append(cleaned)
    return result


def _split_semantic_atoms(text: str) -> list[str]:
    atoms: list[str] = []
    boundary = re.compile(
        r"[,，](?=\s*(?:负责|完成|搭建|制定|推进|交付|参与|承担|具备|熟悉|掌握|"
        r"本科|硕士|博士|大专|学历|有.+经验|经验|证书|优先))"
    )
    for item in _split_items(text):
        for atom in boundary.split(item):
            cleaned = _clean(atom).strip(" ，,:：")
            if cleaned and cleaned not in atoms:
                atoms.append(cleaned)
    return atoms


def _semantic_bucket(value: str, default: str) -> str:
    qualification = re.search(
        r"具备|熟悉|掌握|精通|学历|学位|经验|证书|执照|资格|资质|优先|"
        r"本科|硕士|博士|大专|高中|中专",
        value,
    )
    if qualification:
        return "requirements"
    if re.search(
        r"负责|完成|搭建|制定|推进|交付|参与|承担|建设|开发|设计|维护|整理|清洗|"
        r"预训练|训练|评估|部署|优化|跟进|产出|实现",
        value,
    ):
        return "responsibilities"
    return default


def _rule_long_text(raw: str) -> dict[str, list[str]]:
    responsibilities: list[str] = []
    requirements: list[str] = []
    sections = (
        (_extract_section(raw, RESPONSIBILITY_MARKERS), "responsibilities"),
        (_extract_section(raw, REQUIREMENT_MARKERS), "requirements"),
    )
    for section, default in sections:
        for atom in _split_semantic_atoms(section):
            bucket = _semantic_bucket(atom, default)
            target = requirements if bucket == "requirements" else responsibilities
            if atom not in target:
                target.append(atom)
    if not any(section for section, _ in sections):
        for atom in _split_semantic_atoms(raw):
            bucket = _semantic_bucket(atom, "unknown")
            if bucket == "requirements" and atom not in requirements:
                requirements.append(atom)
            elif bucket == "responsibilities" and atom not in responsibilities:
                responsibilities.append(atom)
    skills = [term for term in SKILL_TERMS if re.search(rf"(?i)(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])", raw)]
    certificates = [term for term in CERTIFICATE_TERMS if term.lower() in raw.lower()]
    return {
        "requirements": requirements,
        "responsibilities": responsibilities,
        "skills": skills,
        "certificates": certificates,
        "benefits": [],
    }


def _merge_unique(first: Iterable[str], second: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in (*first, *second):
        cleaned = _clean(item)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            result.append(cleaned)
            seen.add(key)
    return result


def _parse_json_list(value: str) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [_clean(item) for item in parsed if _clean(item)] if isinstance(parsed, list) else []


def _business_row(row: dict[str, str], mode: Literal["rules", "hybrid"]) -> dict[str, Any]:
    raw = row.get("jd_raw") or ""
    salary_min, salary_max, currency, period = _parse_salary(row.get("salary") or "")
    exp_min, exp_max = _experience_range(row.get("experience") or "", raw)
    extracted = _rule_long_text(raw)
    if mode == "hybrid":
        llm = extract_long_text_with_llm(row)
        extracted = {
            key: _merge_unique(extracted[key], getattr(llm, key))
            for key in ("requirements", "responsibilities", "skills", "certificates", "benefits")
        }
    return {
        "job_id": row.get("job_id"),
        "title": row.get("title"),
        "company_name": row.get("company_name"),
        "city": row.get("city"),
        "work_address": row.get("work_address"),
        "salary_min": salary_min,
        "salary_max": salary_max,
        "salary_currency": currency,
        "salary_period": period,
        "recruitment": _recruitment(row),
        "employment": _employment(row),
        "work_mode": _work_mode(raw),
        "education_min_level": _education_level(row.get("education") or ""),
        "experience_min_months": exp_min,
        "experience_max_months": exp_max,
        "requirements_json": extracted["requirements"],
        "responsibilities_json": extracted["responsibilities"],
        "skills_json": extracted["skills"],
        "certificates_json": extracted["certificates"],
        "benefits_json": _merge_unique(
            _parse_json_list(row.get("job_tags") or ""), extracted["benefits"],
        ),
        "source_url": row.get("source_url") or row.get("page_url"),
        "scraped_at": row.get("scraped_at"),
        "content_hash": hashlib.sha256(_clean(raw).encode("utf-8")).hexdigest(),
        "extraction_mode": mode,
    }


def build_business_csv(
    source_csv: Path = SOURCE_CSV,
    output_path: Path = BUSINESS_JOBS_CSV,
    *,
    mode: Literal["rules", "hybrid"] = "hybrid",
    allow_external_llm: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Build the single business table; LLM is used only for long JD text."""
    if mode == "hybrid" and not allow_external_llm:
        raise PermissionError(
            "Hybrid extraction sends jd_raw to DeepSeek; pass --allow-external-llm explicitly"
        )
    rows = _read_source_csv(source_csv)
    if limit is not None:
        rows = rows[:limit]
    business_rows = [_business_row(row, mode) for row in rows]
    count = _write_csv(output_path, BUSINESS_COLUMNS, business_rows)
    return {
        "business_rows": count,
        "business_columns": len(BUSINESS_COLUMNS),
        "llm_calls": count if mode == "hybrid" else 0,
        "output": str(output_path),
    }
