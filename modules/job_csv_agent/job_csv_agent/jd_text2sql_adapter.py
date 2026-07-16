from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

from .config import AGENT_ROOT, BUSINESS_CSV, SQLITE_DB


JD_TEXT2SQL_ROOT = AGENT_ROOT / "modules" / "jd_text2sql"
if str(JD_TEXT2SQL_ROOT) not in sys.path:
    sys.path.insert(0, str(JD_TEXT2SQL_ROOT))

from jd_text2sql.config import INTERNAL_RELATIONS  # noqa: E402
from jd_text2sql.db import build_database, execute_readonly  # noqa: E402
from jd_text2sql.llm import text2sql_with_llm  # noqa: E402
from jd_text2sql.rule_text2sql import SQLDraft, generate_rule_sql  # noqa: E402
from jd_text2sql.sql_guard import guard_sql  # noqa: E402


class JDText2SQLAdapter:
    """Makes CSV writes immediately visible to the existing guarded query module."""

    _db_lock = threading.Lock()

    def __init__(self, csv_path: Path = BUSINESS_CSV, db_path: Path = SQLITE_DB) -> None:
        self.csv_path = Path(csv_path)
        self.db_path = Path(db_path)

    def rebuild(self) -> dict[str, int]:
        with self._db_lock:
            return build_database(self.csv_path, self.db_path)

    def ask(self, question: str, max_rows: int = 20) -> dict[str, Any]:
        if not self.db_path.exists() or self.db_path.stat().st_mtime < self.csv_path.stat().st_mtime:
            self.rebuild()
        rule_draft = generate_rule_sql(question, self.db_path)
        generator = "rules"
        draft: SQLDraft = rule_draft
        if not rule_draft.matched_rules:
            llm_draft = text2sql_with_llm(question)
            if llm_draft.needs_clarification:
                return {"needs_clarification": True, "message": llm_draft.clarification_question}
            draft = SQLDraft(
                sql=llm_draft.sql,
                parameters=tuple(llm_draft.parameters()),
                explanation=llm_draft.explanation,
                matched_rules=(),
            )
            generator = "llm"
        guarded = guard_sql(draft.sql, draft.parameters, INTERNAL_RELATIONS, max_rows=max_rows)
        rows = execute_readonly(self.db_path, guarded.sql, guarded.parameters)
        return {
            "needs_clarification": False,
            "generator": generator,
            "explanation": draft.explanation,
            "rows": rows,
        }
