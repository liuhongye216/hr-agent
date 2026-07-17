from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from job_csv_agent.agent import JobCsvAgent, initial_state
from job_csv_agent.api import create_app
from job_csv_agent.repository import CsvJobRepository
from job_csv_agent.schemas import BUSINESS_COLUMNS, LLMInterpretation, StructuredCommand


FULL_JD = """后浪澎湃有限公司
TEG技术
**混元大语言模型后训练算法工程师**
工作地点：深圳/北京/上海
招聘类型：社招
硕士及以上学历，三年以上工作经验
岗位职责：
**1. 负责大语言模型后训练算法研发**
2. 负责训练数据构建与质量分析
3. 负责强化学习训练流程优化
4. 负责模型效果评测与问题分析
5. 负责算法工程化落地
任职要求：
1. 硕士及以上学历
2. 三年以上机器学习工作经验
3. 有大模型后训练经验者优先
4. 熟悉 Python 与深度学习框架
5. 具备良好的算法基础
6. 有强化学习项目经验者优先
7. 具备清晰的技术沟通能力
"""


RESPONSIBILITIES = [
    "1. 负责大语言模型后训练算法研发",
    "2. 负责训练数据构建与质量分析",
    "3. 负责强化学习训练流程优化",
    "4. 负责模型效果评测与问题分析",
    "5. 负责算法工程化落地",
]
REQUIREMENTS = [
    "1. 硕士及以上学历",
    "2. 三年以上机器学习工作经验",
    "3. 有大模型后训练经验者优先",
    "4. 熟悉 Python 与深度学习框架",
    "5. 具备良好的算法基础",
    "6. 有强化学习项目经验者优先",
    "7. 具备清晰的技术沟通能力",
]


def empty_repository(path: Path) -> CsvJobRepository:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.DictWriter(handle, fieldnames=BUSINESS_COLUMNS).writeheader()
    return CsvJobRepository(path)


class HunyuanInterpreter:
    def extract(self, text: str, context: dict) -> StructuredCommand:
        if text == FULL_JD.strip().replace("\n", " "):
            return StructuredCommand.model_validate({
                "intent": "create",
                "task_relation": "start_new",
                "mentioned_fields": [
                    "company_name", "title", "city", "recruitment",
                    "education_min_level", "experience_min_months",
                    "responsibilities_json", "requirements_json",
                ],
                "evidence_spans": [
                    {"field": "company_name", "text": "后浪澎湃有限公司"},
                    {"field": "title", "text": "混元大语言模型后训练算法工程师"},
                    {"field": "city", "text": "深圳/北京/上海"},
                    {"field": "recruitment", "text": "社招"},
                    {"field": "education_min_level", "text": "硕士及以上学历"},
                    {"field": "experience_min_months", "text": "三年以上工作经验"},
                    *({"field": "responsibilities_json", "text": item} for item in RESPONSIBILITIES),
                    *({"field": "requirements_json", "text": item} for item in REQUIREMENTS),
                ],
                "ignored_fragments": [{"text": "TEG技术", "reason": "当前 schema 不保存部门"}],
                "patch": {
                    "set_fields": {
                        "company_name": "后浪澎湃有限公司",
                        "title": "TEG技术/混元大语言模型后训练算法工程师",
                        "city": "深圳/北京/上海",
                    },
                    "append_items": {
                        "responsibilities_json": [
                            {"value": f"**{item}**" if item.startswith("1.") else item}
                            for item in RESPONSIBILITIES
                        ],
                        "requirements_json": [{"value": item} for item in REQUIREMENTS],
                    },
                },
            })
        if "招聘类型为社招" in text:
            return StructuredCommand.model_validate({
                "intent": "update",
                "task_relation": "continue_current",
                "mentioned_fields": ["recruitment", "experience_min_months"],
                "evidence_spans": [
                    {"field": "recruitment", "text": "社招"},
                    {"field": "experience_min_months", "text": "三年"},
                ],
                "patch": {"set_fields": {
                    "recruitment": "社招", "experience_min_months": "三年",
                }},
            })
        raise AssertionError(text)


def test_full_hunyuan_jd_preserves_items_and_has_no_invented_content(tmp_path: Path) -> None:
    agent = JobCsvAgent(empty_repository(tmp_path / "jobs.csv"), HunyuanInterpreter())
    state = agent.handle(initial_state(), FULL_JD)

    assert state["draft"]["recruitment"] == "experienced"
    assert state["draft"]["experience_min_months"] == 36
    assert state["draft"]["education_min_level"] == 5
    assert state["draft"]["city"] == "深圳/北京/上海"
    assert state["draft"]["responsibilities_json"] == RESPONSIBILITIES
    assert state["draft"]["requirements_json"] == REQUIREMENTS
    assert "TEG技术" not in str(state["draft"])
    assert "配合团队" not in str(state["draft"])
    assert "**" not in str(state["draft"])
    assert state["can_confirm"] is True


def test_multiturn_explicit_closed_fields_update_ready_draft(tmp_path: Path) -> None:
    agent = JobCsvAgent(empty_repository(tmp_path / "jobs.csv"), HunyuanInterpreter())
    first = agent.handle(initial_state(), FULL_JD)
    second = agent.handle(first, "招聘类型为社招；最低经验月数为三年")

    assert second["draft"]["recruitment"] == "experienced"
    assert second["draft"]["experience_min_months"] == 36
    assert second["can_confirm"] is True
    assert "本轮已更新：招聘类型＝社会招聘；最低经验（月）＝36个月" in second["message"]
    assert "还缺少" not in second["message"]


def test_empty_patch_with_mentioned_field_retries_then_blocks_confirmation(tmp_path: Path) -> None:
    class EmptyPatchInterpreter:
        calls = 0

        def extract(self, text: str, context: dict) -> StructuredCommand:
            self.calls += 1
            return StructuredCommand.model_validate({
                "intent": "update",
                "mentioned_fields": ["city"],
                "evidence_spans": [{"field": "city", "text": "北京"}],
            })

    interpreter = EmptyPatchInterpreter()
    agent = JobCsvAgent(empty_repository(tmp_path / "jobs.csv"), interpreter)
    state = initial_state()
    state.update({
        "pending_action": "create",
        "phase": "CONFIRMING",
        "draft": {
            "company_name": "甲公司", "title": "算法工程师",
            "responsibilities_json": ["1. 负责算法研发"],
        },
        "can_confirm": True,
        "draft_revision": 1,
        "preview_revision": 1,
    })
    before = deepcopy(state["draft"])

    failed = agent.handle(state, "工作城市是北京")
    assert interpreter.calls == 2
    assert failed["draft"] == before
    assert failed["can_confirm"] is False
    assert failed["preview_revision"] is None
    assert "本轮字段未成功应用，尚未保存" in failed["message"]
    assert "城市" in failed["message"]
    assert "岗位草稿" not in failed["message"]


def test_stale_confirmation_returns_409_and_latest_version_saves(tmp_path: Path) -> None:
    class CreateInterpreter:
        def extract(self, text: str, context: dict) -> StructuredCommand:
            return StructuredCommand.model_validate({
                "intent": "create", "task_relation": "start_new",
                "patch": {
                    "set_fields": {"company_name": "甲公司", "title": "测试工程师"},
                    "append_items": {"responsibilities_json": [{"value": "负责软件测试"}]},
                },
            })

    repository = empty_repository(tmp_path / "jobs.csv")
    app = create_app(JobCsvAgent(repository, CreateInterpreter()))
    with TestClient(app) as client:
        session_id = client.post("/sessions").json()["session_id"]
        preview = client.post(
            f"/sessions/{session_id}/messages",
            json={"content": "新增甲公司的测试工程师，负责软件测试", "expected_version": 0},
        )
        assert preview.status_code == 200
        assert client.post(f"/sessions/{session_id}/confirm").status_code == 422
        assert client.post(f"/sessions/{session_id}/confirm?expected_version=0").status_code == 409
        saved = client.post(f"/sessions/{session_id}/confirm?expected_version=1")
        assert saved.status_code == 200
    assert repository.count({"company_name": "甲公司"}) == 1


def test_llm_protocol_requires_declared_evidenced_fields_and_query_plan() -> None:
    with pytest.raises(ValueError, match="missing from mentioned_fields"):
        LLMInterpretation.model_validate({
            "intent": "create",
            "patch": {"set_fields": {"company_name": "甲公司"}},
        })
    with pytest.raises(ValueError, match="require a QueryPlan"):
        LLMInterpretation.model_validate({"intent": "search"})
