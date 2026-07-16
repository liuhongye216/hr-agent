from __future__ import annotations

import pytest

from jd_text2sql.sql_guard import SQLGuardError, guard_sql


ALLOWED = {"jobs", "json_each"}


def test_allows_parameterized_select_and_adds_limit() -> None:
    guarded = guard_sql(
        "SELECT * FROM jobs WHERE city = ?",
        ["北京"],
        ALLOWED,
        max_rows=25,
    )
    assert guarded.parameters == ("北京",)
    assert guarded.sql.endswith("LIMIT 25")


def test_allows_cte_that_reads_allowlisted_view() -> None:
    guarded = guard_sql(
        "WITH filtered AS (SELECT job_id FROM jobs) SELECT * FROM filtered",
        [],
        ALLOWED,
    )
    assert "filtered" in guarded.relations


def test_allows_json_array_search_on_business_table() -> None:
    guarded = guard_sql(
        "SELECT j.job_id FROM jobs j WHERE EXISTS "
        "(SELECT 1 FROM json_each(j.skills_json) x WHERE x.value = ?)",
        ["Excel"],
        ALLOWED,
    )
    assert guarded.relations == ("jobs", "json_each")


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM jobs",
        "SELECT raw_jd_text FROM job_sources",
        "SELECT * FROM sqlite_master",
        "SELECT * FROM jobs; SELECT * FROM jobs",
        "SELECT * FROM jobs -- unsafe",
    ],
)
def test_blocks_unsafe_sql(sql: str) -> None:
    with pytest.raises(SQLGuardError):
        guard_sql(sql, [], ALLOWED)


def test_rejects_placeholder_mismatch() -> None:
    with pytest.raises(SQLGuardError):
        guard_sql("SELECT * FROM jobs WHERE city = ?", [], ALLOWED)
