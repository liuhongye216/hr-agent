from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from job_csv_agent.agent import JobCsvAgent, initial_state
from job_csv_agent.llm import StructuredInterpreter
from job_csv_agent.repository import CsvJobRepository
from job_csv_agent.schemas import BUSINESS_COLUMNS


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_DEEPSEEK") != "1",
    reason="set RUN_LIVE_DEEPSEEK=1 to run real DeepSeek regression conversations",
)


def empty_repository(path: Path) -> CsvJobRepository:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.DictWriter(handle, fieldnames=BUSINESS_COLUMNS).writeheader()
    return CsvJobRepository(path)


def test_live_modality_repair_and_confirmation_safety(tmp_path: Path) -> None:
    repository = empty_repository(tmp_path / "jobs.csv")
    agent = JobCsvAgent(repository, StructuredInterpreter())

    state = agent.handle(
        initial_state(),
        "远山科技招聘产品经理，负责企业协同产品规划，要求三年以上产品经验。",
    )
    assert state["draft"]["experience_min_months"] == 36
    assert state["draft"]["responsibilities_json"] == ["企业协同产品规划"]

    state = agent.handle(state, "我刚才说了三年以上产品经验只是加分项。")
    assert "experience_min_months" not in state["draft"], state["message"]
    assert state["draft"].get("requirements_json", []) == []
    assert state["draft"]["preferred_requirements_json"] == ["三年以上产品经验"]
    assert "我刚才说了" not in str(state["draft"])

    state = agent.handle(state, "你现在的字段对吗？重新检查一下。")
    assert "experience_min_months" not in state["draft"]
    assert state["draft"]["preferred_requirements_json"] == ["三年以上产品经验"]

    state = agent.handle(state, "？")
    assert repository.count({"company_name": "远山科技"}) == 0
    assert state["phase"] == "CONFIRMING"
    assert state["can_confirm"] is True


def test_live_structured_requirement_and_finish_editing_flow(tmp_path: Path) -> None:
    repository = empty_repository(tmp_path / "jobs.csv")
    agent = JobCsvAgent(repository, StructuredInterpreter())

    state = agent.handle(initial_state(), "新增乙公司的运营专员岗位。")
    assert state["missing_fields"] == ["job_content"]
    assert "任职要求或岗位职责" in state["message"]

    state = agent.handle(state, "月薪最低一万。")
    assert state["draft"]["salary_min"] == 10000
    assert state["missing_fields"] == ["job_content"]
    assert "岗位职责" in state["message"]

    state = agent.handle(state, "月薪上限15000，学历要求本科，城市在杭州。")
    assert state["draft"]["education_min_level"] == 4
    assert state["storage_valid"] is True
    assert state["extraction_complete"] is True, state["message"]
    assert state["can_confirm"] is True

    state = agent.handle(state, "不补充了。")
    assert repository.count({"company_name": "乙公司"}) == 0
    assert state["phase"] == "CONFIRMING"
    assert state["can_confirm"] is True
