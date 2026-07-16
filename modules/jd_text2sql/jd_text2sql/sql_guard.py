from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Collection, Sequence


FORBIDDEN_KEYWORDS = re.compile(
    r"\b(?:insert|update|delete|drop|alter|create|replace|truncate|attach|detach|"
    r"pragma|vacuum|reindex|analyze|copy|grant|revoke|load_extension)\b",
    re.IGNORECASE,
)
COMMENT_PATTERN = re.compile(r"--|/\*|\*/")
RAW_FIELD_PATTERN = re.compile(
    r"\b(?:job_sources|raw_jd_text|raw_payload|raw_company_facts|recruiter_name|recruiter_meta)\b",
    re.IGNORECASE,
)
RELATION_PATTERN = re.compile(r"\b(?:from|join)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
CTE_PATTERN = re.compile(r"(?:\bwith|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s+as\s*\(", re.IGNORECASE)


@dataclass(frozen=True)
class GuardedSQL:
    sql: str
    parameters: tuple[Any, ...]
    relations: tuple[str, ...]


class SQLGuardError(ValueError):
    pass


def guard_sql(
    sql: str,
    parameters: Sequence[Any],
    allowed_relations: Collection[str],
    *,
    max_rows: int = 100,
) -> GuardedSQL:
    candidate = sql.strip()
    if not candidate:
        raise SQLGuardError("SQL is empty")
    if COMMENT_PATTERN.search(candidate):
        raise SQLGuardError("SQL comments are not allowed")
    if candidate.endswith(";"):
        candidate = candidate[:-1].rstrip()
    if ";" in candidate:
        raise SQLGuardError("Only one SQL statement is allowed")
    if not re.match(r"^(?:select|with)\b", candidate, re.IGNORECASE):
        raise SQLGuardError("Only SELECT or WITH ... SELECT is allowed")
    if FORBIDDEN_KEYWORDS.search(candidate):
        raise SQLGuardError("Write, DDL, PRAGMA, and extension operations are forbidden")
    if RAW_FIELD_PATTERN.search(candidate):
        raise SQLGuardError("Raw JD and source tables are outside the Text2SQL semantic layer")
    if candidate.count("?") != len(parameters):
        raise SQLGuardError("Placeholder count does not match parameters")

    ctes = {name.lower() for name in CTE_PATTERN.findall(candidate)}
    relations = tuple(name.lower() for name in RELATION_PATTERN.findall(candidate))
    allowed = {name.lower() for name in allowed_relations}
    unexpected = sorted({name for name in relations if name not in allowed and name not in ctes})
    if unexpected:
        raise SQLGuardError(f"Relations are not allowlisted: {', '.join(unexpected)}")
    if not relations:
        raise SQLGuardError("SQL must query at least one allowlisted relation")

    limit_match = re.search(r"\blimit\s+(\d+)\s*$", candidate, re.IGNORECASE)
    if limit_match:
        current = int(limit_match.group(1))
        if current > max_rows:
            candidate = candidate[: limit_match.start(1)] + str(max_rows)
    elif not re.search(r"\blimit\s+\?\s*$", candidate, re.IGNORECASE):
        candidate = f"{candidate}\nLIMIT {max_rows}"

    return GuardedSQL(candidate, tuple(parameters), relations)
