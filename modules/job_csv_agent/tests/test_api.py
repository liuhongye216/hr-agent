from __future__ import annotations

import csv
from pathlib import Path

from fastapi.testclient import TestClient

from job_csv_agent.agent import JobCsvAgent
from job_csv_agent.api import create_app
from job_csv_agent.llm import StructuredInterpreter
from job_csv_agent.repository import CsvJobRepository
from job_csv_agent.schemas import BUSINESS_COLUMNS, StructuredCommand


class FakeInterpreter:
    def extract(self, text: str, context: dict) -> StructuredCommand:
        return StructuredCommand.model_validate({
            "intent": "create",
            "fields": {
                "company_name": "API 示例公司", "title": "测试工程师",
                "requirements_json": ["熟悉自动化测试"],
            },
            "semantic_facts": [{
                "value": "熟悉自动化测试", "category": "requirement", "importance": "must",
                "source_type": "explicit", "evidence_text": "熟悉自动化测试",
            }],
        })


def test_fastapi_session_confirmation_flow(tmp_path: Path) -> None:
    csv_path = tmp_path / "jobs.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.DictWriter(handle, fieldnames=BUSINESS_COLUMNS).writeheader()
    app = create_app(JobCsvAgent(CsvJobRepository(csv_path), FakeInterpreter()))

    with TestClient(app) as client:
        created = client.post("/sessions")
        assert created.status_code == 200
        session_id = created.json()["session_id"]

        preview = client.post(
            f"/sessions/{session_id}/messages", json={"content": "创建测试岗位，熟悉自动化测试"},
        )
        assert preview.json()["phase"] == "CONFIRMING"
        assert len(CsvJobRepository(csv_path).search("API 示例公司")) == 0

        confirmed = client.post(f"/sessions/{session_id}/confirm")
        assert confirmed.json()["phase"] == "CONFIRMING"
        confirmed = client.post(f"/sessions/{session_id}/confirm")
        assert confirmed.json()["phase"] == "IDLE"
        assert len(CsvJobRepository(csv_path).search("API 示例公司")) == 1


def test_placeholder_api_key_returns_actionable_422(tmp_path: Path, monkeypatch) -> None:
    csv_path = tmp_path / "jobs.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.DictWriter(handle, fieldnames=BUSINESS_COLUMNS).writeheader()
    monkeypatch.setattr("job_csv_agent.llm.DEEPSEEK_API_KEY", "replace-me")
    app = create_app(JobCsvAgent(CsvJobRepository(csv_path), StructuredInterpreter()))

    with TestClient(app) as client:
        session_id = client.post("/sessions").json()["session_id"]
        response = client.post(
            f"/sessions/{session_id}/messages", json={"content": "帮我处理一个岗位"},
        )

    assert response.status_code == 422
    assert "示例占位值" in response.json()["detail"]


def test_second_confirmation_is_not_rejected_by_fuzzy_semantics(tmp_path: Path) -> None:
    class QualificationInterpreter:
        def extract(self, text: str, context: dict) -> StructuredCommand:
            requirement = "具有较好的软件工程知识和编码规范意识，对代码和设计质量有严格要求"
            return StructuredCommand.model_validate({
                "intent": "create",
                "fields": {
                    "company_name": "美团", "title": "大模型实习生",
                    "requirements_json": [requirement],
                },
                "semantic_facts": [{
                    "value": requirement, "category": "requirement", "importance": "must",
                    "source_type": "explicit", "evidence_text": requirement,
                }],
            })

    csv_path = tmp_path / "jobs.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.DictWriter(handle, fieldnames=BUSINESS_COLUMNS).writeheader()
    app = create_app(JobCsvAgent(CsvJobRepository(csv_path), QualificationInterpreter()))

    with TestClient(app) as client:
        session_id = client.post("/sessions").json()["session_id"]
        preview = client.post(
            f"/sessions/{session_id}/messages",
            json={"content": "创建美团大模型实习生，具有较好的软件工程知识和编码规范意识，对代码和设计质量有严格要求"},
        )
        assert preview.status_code == 200
        first = client.post(f"/sessions/{session_id}/confirm")
        assert first.status_code == 200
        second = client.post(f"/sessions/{session_id}/confirm")
        assert second.status_code == 200
        assert second.json()["phase"] == "IDLE"
