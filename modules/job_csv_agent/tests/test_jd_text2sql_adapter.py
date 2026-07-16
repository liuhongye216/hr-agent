from __future__ import annotations

import csv
from pathlib import Path

from job_csv_agent.jd_text2sql_adapter import JDText2SQLAdapter
from job_csv_agent.repository import CsvJobRepository
from job_csv_agent.schemas import BUSINESS_COLUMNS


def test_confirmed_csv_rows_are_queryable_by_existing_text2sql(tmp_path: Path) -> None:
    csv_path = tmp_path / "jobs.csv"
    db_path = tmp_path / "jobs.sqlite"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.DictWriter(handle, fieldnames=BUSINESS_COLUMNS).writeheader()
    CsvJobRepository(csv_path).create({
        "company_name": "示例科技", "title": "Python 开发",
        "requirements_json": ["本科及以上"], "education_min_level": 4,
    })

    adapter = JDText2SQLAdapter(csv_path, db_path)
    adapter.rebuild()
    result = adapter.ask("本科及以上岗位有多少")

    assert result["generator"] == "rules"
    assert result["rows"] == [{"job_count": 1}]
