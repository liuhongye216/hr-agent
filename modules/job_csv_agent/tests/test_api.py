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
