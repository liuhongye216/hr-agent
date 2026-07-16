from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from job_csv_agent.agent import JobCsvAgent, initial_state
from job_csv_agent.repository import CsvJobRepository
from job_csv_agent.schemas import BUSINESS_COLUMNS, Phase, StructuredCommand


class FakeInterpreter:
    def __init__(self, commands: dict[str, dict]) -> None:
        self.commands = commands

    def extract(self, text: str, context: dict) -> StructuredCommand:
        return StructuredCommand.model_validate(self.commands[text])


def empty_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.DictWriter(handle, fieldnames=BUSINESS_COLUMNS).writeheader()


def row_count(path: Path) -> int:
    return len(pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig"))


def test_create_requires_confirmation_before_write(tmp_path: Path) -> None:
    path = tmp_path / "jobs.csv"
    empty_csv(path)
    repository = CsvJobRepository(path)
    interpreter = FakeInterpreter({
        "新建": {"intent": "create", "fields": {"title": "Python 开发", "salary_min": 20_000}},
        "补充": {"intent": "update", "fields": {
            "company_name": "示例科技", "requirements_json": ["三年开发经验"],
        }},
    })
    agent = JobCsvAgent(repository, interpreter)

    state = agent.handle(initial_state(), "新建")
    assert state["phase"] == Phase.CREATING.value
    assert set(state["missing_fields"]) == {"company_name", "job_content"}
    assert row_count(path) == 0

    state = agent.handle(state, "补充")
    assert state["phase"] == Phase.CONFIRMING.value
    assert state["can_confirm"] is True
    assert row_count(path) == 0

    state = agent.handle(state, "确认写入", {"intent": "confirm"})
    assert state["phase"] == Phase.IDLE.value
    assert row_count(path) == 1


def test_edit_disambiguates_then_confirms(tmp_path: Path) -> None:
    path = tmp_path / "jobs.csv"
    empty_csv(path)
    repository = CsvJobRepository(path)
    first = repository.create({
        "company_name": "甲公司", "title": "Python 后台", "requirements_json": ["本科"],
        "salary_min": 20_000,
    })
    repository.create({
        "company_name": "乙公司", "title": "Python 数据", "requirements_json": ["本科"],
        "salary_min": 21_000,
    })
    interpreter = FakeInterpreter({
        "修改 Python 岗位": {
            "intent": "update", "search_query": "Python",
            "fields": {"salary_min": 25_000, "salary_max": 30_000},
        },
        "加福利": {"intent": "update", "fields": {"benefits_json": ["年度体检"]}},
    })
    agent = JobCsvAgent(repository, interpreter)

    state = agent.handle(initial_state(), "修改 Python 岗位")
    assert state["phase"] == Phase.EDITING.value
    assert len(state["candidates"]) == 2

    state = agent.handle(state, "1", {"intent": "unknown", "selection_index": 1})
    assert state["phase"] == Phase.CONFIRMING.value
    selected_id = state["target_id"]
    assert repository.get(selected_id)["salary_min"] in {"20000.0", "21000.0"}

    state = agent.handle(state, "继续修改", {"intent": "update"})
    assert state["phase"] == Phase.EDITING.value
    state = agent.handle(state, "加福利")
    assert state["phase"] == Phase.CONFIRMING.value
    assert state["pending_fields"]["salary_min"] == 25_000
    assert state["pending_fields"]["benefits_json"] == ["年度体检"]

    state = agent.handle(state, "确认写入", {"intent": "confirm"})
    assert state["phase"] == Phase.IDLE.value
    assert repository.get(selected_id)["salary_min"] == "25000.0"
    assert repository.get(selected_id)["benefits_json"] == '["年度体检"]'
    assert repository.get(first["job_id"])["job_id"] == first["job_id"]


def test_create_with_responsibility_only_and_inferred_suggestions(tmp_path: Path) -> None:
    path = tmp_path / "jobs.csv"
    empty_csv(path)
    repository = CsvJobRepository(path)
    interpreter = FakeInterpreter({
        "新建大模型岗位": {
            "intent": "create",
            "fields": {
                "company_name": "示例科技", "title": "大模型工程师",
                # Simulate the historical model error. The deterministic semantic layer repairs it.
                "requirements_json": ["从0开始预训练大模型"],
            },
        },
    })
    agent = JobCsvAgent(repository, interpreter)

    state = agent.handle(initial_state(), "新建大模型岗位")
    assert state["phase"] == Phase.CONFIRMING.value
    assert state["pending_fields"].get("requirements_json") is None
    assert state["pending_fields"]["responsibilities_json"] == ["从0开始预训练大模型"]
    assert "inferred，不会写入 CSV" in state["message"]
    assert "从零训练基础模型" in state["message"]
    assert row_count(path) == 0

    state = agent.handle(state, "确认写入", {"intent": "confirm"})
    assert state["phase"] == Phase.IDLE.value
    row = repository.search("示例科技 大模型工程师")[0]
    assert row["requirements_json"] == "[]"
    assert row["responsibilities_json"] == '["从0开始预训练大模型"]'
    assert row["skills_json"] == "[]"
