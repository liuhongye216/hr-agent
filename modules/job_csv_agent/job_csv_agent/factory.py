from __future__ import annotations

from .agent import JobCsvAgent
from .config import BUSINESS_CSV
from .jd_text2sql_adapter import JDText2SQLAdapter
from .llm import StructuredInterpreter
from .repository import CsvJobRepository


def create_default_agent() -> JobCsvAgent:
    return JobCsvAgent(
        repository=CsvJobRepository(BUSINESS_CSV),
        interpreter=StructuredInterpreter(),
        query_adapter=JDText2SQLAdapter(),
    )
