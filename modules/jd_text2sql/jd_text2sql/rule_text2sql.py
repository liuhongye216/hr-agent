from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EDUCATION_LEVELS = {
    "初中": 1, "高中": 2, "中专": 2, "大专": 3, "专科": 3,
    "本科": 4, "硕士": 5, "研究生": 5, "博士": 6,
}


@dataclass(frozen=True)
class SQLDraft:
    sql: str
    parameters: tuple[Any, ...]
    explanation: str
    matched_rules: tuple[str, ...]


def _known_cities(db_path: Path) -> list[str]:
    conn = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        return [row[0] for row in conn.execute("SELECT DISTINCT city FROM jobs WHERE city IS NOT NULL")]
    finally:
        conn.close()


def _salary_number(raw: str, unit: str) -> float:
    value = float(raw)
    if unit.lower() == "k" or unit == "千":
        return value * 1_000
    if unit == "万":
        return value * 10_000
    return value


def generate_rule_sql(question: str, db_path: Path) -> SQLDraft:
    q = question.strip()
    if not q:
        raise ValueError("Question is empty")
    conditions: list[str] = []
    parameters: list[Any] = []
    rules: list[str] = []

    for city in sorted(_known_cities(db_path), key=len, reverse=True):
        if city and city in q:
            conditions.append("j.city = ?")
            parameters.append(city)
            rules.append("city")
            break

    if "学历不限" in q or "不限学历" in q:
        conditions.append("j.education_min_level = 0")
        rules.append("education_unrestricted")
    else:
        for word, level in EDUCATION_LEVELS.items():
            if word in q:
                conditions.append("j.education_min_level >= ?")
                parameters.append(level)
                rules.append("education")
                break

    if any(token in q for token in ("全职兼职均可", "全职、兼职均可", "全职或兼职都可以")):
        conditions.append("j.employment = 'full_or_part_time'")
        rules.append("employment:flexible")
    elif "兼职" in q:
        conditions.append("j.employment IN ('part_time', 'full_or_part_time')")
        rules.append("employment:part_time")
    elif "全职" in q:
        conditions.append("j.employment IN ('full_time', 'full_or_part_time')")
        rules.append("employment:full_time")

    period = None
    if "月薪" in q:
        period = "month"
    elif "日薪" in q:
        period = "day"
    elif "时薪" in q or "小时" in q:
        period = "hour"
    elif "按单" in q or "每单" in q or "元/单" in q:
        period = "per_order"
    salary_match = re.search(
        r"(?:月薪|日薪|时薪|薪资|工资|每单|按单)[^0-9]{0,10}(\d+(?:\.\d+)?)\s*([kK千万元]?)\s*(?:以上|起|及以上)",
        q,
    )
    if salary_match:
        conditions.extend(["j.salary_min >= ?", "j.salary_currency = 'CNY'"])
        parameters.append(_salary_number(salary_match.group(1), salary_match.group(2)))
        rules.append("salary_min")
        if period:
            conditions.append("j.salary_period = ?")
            parameters.append(period)
            rules.append("salary_period")

    if any(token in q for token in ("经验不限", "无需经验", "无经验")):
        conditions.append("COALESCE(j.experience_min_months, 0) = 0")
        rules.append("no_experience")

    # JSON arrays keep the table count at one while remaining queryable in SQLite.
    for label, column in (("技能", "skills_json"), ("证书", "certificates_json"), ("福利", "benefits_json")):
        match = re.search(rf"(?:包含|要求|会|有)([^，。；\s]{{1,30}}){label}", q)
        if match:
            conditions.append(
                f"EXISTS (SELECT 1 FROM json_each(j.{column}) x WHERE x.value LIKE ?)"
            )
            parameters.append(f"%{match.group(1)}%")
            rules.append(column)

    is_count = any(token in q for token in ("多少", "数量", "几条", "统计", "总数"))
    select = "SELECT COUNT(*) AS job_count" if is_count else (
        "SELECT j.job_id, j.title, j.company_name, j.city, j.education_min_level, "
        "j.experience_min_months, j.salary_min, j.salary_max, j.salary_currency, j.salary_period"
    )
    sql = f"{select}\nFROM jobs j"
    if conditions:
        sql += "\nWHERE " + "\n  AND ".join(conditions)
    if not is_count:
        sql += "\nORDER BY j.job_id"
    explanation = "规则命中：" + ("、".join(rules) if rules else "无筛选条件")
    return SQLDraft(sql, tuple(parameters), explanation, tuple(rules))
