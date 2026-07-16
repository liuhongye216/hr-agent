from __future__ import annotations

from pathlib import Path

import pytest

from jd_text2sql.db import build_database, execute_readonly
from jd_text2sql.rule_text2sql import generate_rule_sql
from jd_text2sql.sql_guard import guard_sql
from jd_text2sql.config import INTERNAL_RELATIONS


@pytest.fixture(scope="module")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("db") / "jobs.sqlite"
    counts = build_database(db_path=path)
    assert counts["jobs"] == 100
    return path


def test_city_and_education_count(db_path: Path) -> None:
    draft = generate_rule_sql("北京有多少本科及以上岗位", db_path)
    guarded = guard_sql(draft.sql, draft.parameters, INTERNAL_RELATIONS)
    rows = execute_readonly(db_path, guarded.sql, guarded.parameters)
    assert rows == [{"job_count": 1}]


def test_salary_query_forces_currency_and_period(db_path: Path) -> None:
    draft = generate_rule_sql("月薪下限10k以上的岗位", db_path)
    assert "salary_currency = 'CNY'" in draft.sql
    assert "salary_period = ?" in draft.sql
    guarded = guard_sql(draft.sql, draft.parameters, INTERNAL_RELATIONS)
    rows = execute_readonly(db_path, guarded.sql, guarded.parameters)
    assert rows
    assert all(row["salary_period"] == "month" for row in rows)
    assert all(row["salary_min"] >= 10000 for row in rows)


def test_database_has_one_business_table(db_path: Path) -> None:
    rows = execute_readonly(
        db_path,
        r"SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE '\_%' ESCAPE '\' ORDER BY name",
    )
    assert rows == [{"name": "jobs"}]
