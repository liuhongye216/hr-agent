from __future__ import annotations

import csv
from pathlib import Path

from fastapi.testclient import TestClient

from job_csv_agent.agent import JobCsvAgent
from job_csv_agent.api import create_app
from job_csv_agent.repository import CsvJobRepository
from job_csv_agent.schemas import BUSINESS_COLUMNS, StructuredCommand


class SessionInterpreter:
    def extract(self, text: str, context: dict) -> StructuredCommand:
        company = "甲公司" if "甲" in text else "乙公司"
        return StructuredCommand.model_validate({
            "intent": "create",
            "task_relation": "start_new",
            "patch": {
                "set_fields": {"company_name": company, "title": "测试工程师"},
                "append_items": {"responsibilities_json": [{
                    "value": f"负责{company}的软件测试", "source": "explicit",
                }]},
            },
        })


def empty_repository(path: Path) -> CsvJobRepository:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.DictWriter(handle, fieldnames=BUSINESS_COLUMNS).writeheader()
    return CsvJobRepository(path)


def test_single_confirmation_writes_once_and_resets(tmp_path: Path) -> None:
    repository = empty_repository(tmp_path / "jobs.csv")
    app = create_app(JobCsvAgent(repository, SessionInterpreter()))
    with TestClient(app) as client:
        session_id = client.post("/sessions").json()["session_id"]
        preview = client.post(
            f"/sessions/{session_id}/messages", json={"content": "新增甲公司的测试工程师岗位，负责甲公司的软件测试"},
        )
        assert preview.json()["phase"] == "CONFIRMING"
        confirmed = client.post(
            f"/sessions/{session_id}/confirm?expected_version={preview.json()['state_version']}"
        )
        assert confirmed.json()["phase"] == "IDLE"
        assert confirmed.json()["can_confirm"] is False
    assert len(repository.search("甲公司")) == 1


def test_api_sessions_do_not_share_drafts(tmp_path: Path) -> None:
    repository = empty_repository(tmp_path / "jobs.csv")
    app = create_app(JobCsvAgent(repository, SessionInterpreter()))
    with TestClient(app) as client:
        first = client.post("/sessions").json()["session_id"]
        second = client.post("/sessions").json()["session_id"]
        client.post(f"/sessions/{first}/messages", json={"content": "新增甲公司测试工程师，负责甲公司的软件测试"})
        client.post(f"/sessions/{second}/messages", json={"content": "新增乙公司测试工程师，负责乙公司的软件测试"})
        first_view = client.post(f"/sessions/{first}/messages", json={"content": "你现在的字段是什么"}).json()
        second_view = client.post(f"/sessions/{second}/messages", json={"content": "你现在的字段是什么"}).json()
    assert "甲公司" in first_view["message"] and "乙公司" not in first_view["message"]
    assert "乙公司" in second_view["message"] and "甲公司" not in second_view["message"]


def test_modify_and_delete_keep_target_selection(tmp_path: Path) -> None:
    class TargetInterpreter:
        def extract(self, text: str, context: dict) -> StructuredCommand:
            commands = {
                "修改测试岗位": {"intent": "update", "search_query": "测试工程师"},
                "薪资改成3000": {
                    "intent": "update",
                    "patch": {
                        "set_fields": {"salary_min": 3000},
                        "set_sources": {"salary_min": "explicit"},
                    },
                },
                "删除测试岗位": {"intent": "delete", "search_query": "测试工程师"},
            }
            return StructuredCommand.model_validate(commands[text])

    repository = empty_repository(tmp_path / "jobs.csv")
    first = repository.create({
        "company_name": "甲公司", "title": "测试工程师",
        "responsibilities_json": ["负责软件测试"],
    })
    second = repository.create({
        "company_name": "乙公司", "title": "测试工程师",
        "responsibilities_json": ["负责软件测试"],
    })
    agent = JobCsvAgent(repository, TargetInterpreter())

    from job_csv_agent.agent import initial_state

    choosing = agent.handle(initial_state(), "修改测试岗位")
    assert len(choosing["candidates"]) == 2
    selected = agent.handle(choosing, "2", {"intent": "unknown", "selection_index": 2})
    target_id = selected["target_id"]
    preview = agent.handle(selected, "薪资改成3000")
    saved = agent.handle(preview, "确认写入", {"intent": "confirm"})
    assert saved["phase"] == "IDLE"
    assert repository.get(target_id)["salary_min"] == "3000.0"
    other_id = first["job_id"] if target_id == second["job_id"] else second["job_id"]
    assert repository.get(other_id)["salary_min"] == ""

    deleting = agent.handle(initial_state(), "删除测试岗位")
    selected_delete = agent.handle(deleting, "1", {"intent": "unknown", "selection_index": 1})
    deleted_id = selected_delete["target_id"]
    agent.handle(selected_delete, "确认删除", {"intent": "confirm"})
    try:
        repository.get(deleted_id)
    except LookupError:
        pass
    else:
        raise AssertionError("selected row was not deleted")
