from __future__ import annotations

import csv
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

from .config import BUSINESS_JOBS_CSV, DEFAULT_DB_PATH


INTEGER_COLUMNS = {
    "education_min_level", "experience_min_months", "experience_max_months",
    "headcount", "internship_min_months", "onsite_days_per_week",
}
REAL_COLUMNS = {"salary_min", "salary_max"}


def _coerce(column: str, value: str | None) -> Any:
    if value in (None, ""):
        return None
    if column in INTEGER_COLUMNS:
        return int(value)
    if column in REAL_COLUMNS:
        return float(value)
    return value


def _column_type(column: str) -> str:
    if column in INTEGER_COLUMNS:
        return "INTEGER"
    if column in REAL_COLUMNS:
        return "REAL"
    return "TEXT"


def build_database(
    business_csv: Path = BUSINESS_JOBS_CSV,
    db_path: Path = DEFAULT_DB_PATH,
) -> dict[str, int]:
    """Build a disposable SQLite database from the single business CSV."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    with business_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if not columns:
            raise ValueError(f"CSV has no header: {business_csv}")
        rows = [tuple(_coerce(column, row.get(column)) for column in columns) for row in reader]

    conn = sqlite3.connect(db_path)
    try:
        definitions = ", ".join(f'"{column}" {_column_type(column)}' for column in columns)
        conn.execute(f"CREATE TABLE jobs ({definitions}, PRIMARY KEY (job_id))")
        placeholders = ", ".join("?" for _ in columns)
        quoted = ", ".join(f'"{column}"' for column in columns)
        conn.executemany(f"INSERT INTO jobs ({quoted}) VALUES ({placeholders})", rows)
        conn.executescript(
            """
            CREATE INDEX idx_jobs_city ON jobs(city);
            CREATE INDEX idx_jobs_salary ON jobs(salary_currency, salary_period, salary_min);
            CREATE INDEX idx_jobs_education ON jobs(education_min_level);
            CREATE INDEX idx_jobs_employment ON jobs(employment);
            CREATE TABLE _dataset_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )
        conn.executemany(
            "INSERT INTO _dataset_meta(key, value) VALUES (?, ?)",
            [
                ("schema", "single_jobs_csv"),
                ("built_at", time.strftime("%Y-%m-%dT%H:%M:%S%z")),
                ("source", str(business_csv)),
            ],
        )
        conn.commit()
        return {"jobs": len(rows)}
    finally:
        conn.close()


def execute_readonly(
    db_path: Path,
    sql: str,
    parameters: Iterable[Any] = (),
    *,
    timeout_seconds: float = 3.0,
) -> list[dict[str, Any]]:
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    started = time.monotonic()

    def progress() -> int:
        return 1 if time.monotonic() - started > timeout_seconds else 0

    try:
        conn.execute("PRAGMA query_only = ON")
        conn.set_progress_handler(progress, 1000)
        cursor = conn.execute(sql, tuple(parameters))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()
