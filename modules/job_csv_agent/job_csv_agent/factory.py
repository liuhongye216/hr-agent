from __future__ import annotations

from .agent import JobCsvAgent
from .config import BUSINESS_CSV
from .llm import StructuredInterpreter
from .repository import CsvJobRepository


def create_default_agent() -> JobCsvAgent:
    return JobCsvAgent(
        repository=CsvJobRepository(BUSINESS_CSV),
        interpreter=StructuredInterpreter(),
    )
