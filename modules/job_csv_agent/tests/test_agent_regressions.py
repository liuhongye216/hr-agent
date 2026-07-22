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
CANONICAL_REQUIREMENTS = [
    item for item in REQUIREMENTS
    if "学历" not in item and not item.startswith("4. 熟悉")
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
    assert state["draft"]["requirements_json"] == CANONICAL_REQUIREMENTS
    assert state["draft"]["skills_json"] == ["Python", "深度学习框架"]
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
    assert "本轮已更新" not in second["message"]
    assert "还缺少" not in second["message"]


def test_empty_patch_with_mentioned_field_keeps_draft_and_tracks_unresolved(tmp_path: Path) -> None:
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
    assert failed["preview_revision"] == failed["draft_revision"]
    assert failed["extraction_complete"] is False
    assert "尚未完整归类" in failed["message"]
    assert "城市" in failed["message"]
    assert "岗位草稿" in failed["message"]


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


class ScenarioInterpreter:
    def extract(self, text: str, context: dict) -> StructuredCommand:
        if "星河科技" in text:
            return StructuredCommand.model_validate({
                "intent": "create", "task_relation": "start_new",
                "patch": {
                    "set_fields": {
                        "company_name": "星河科技", "title": "高级后端工程师",
                        "experience_min_months": 96,
                    },
                    "append_items": {"requirements_json": [{"value": "熟悉Java"}]},
                },
            })
        if "云帆科技" in text:
            return StructuredCommand.model_validate({
                "intent": "create", "task_relation": "start_new",
                "patch": {"set_fields": {
                    "company_name": "云帆科技", "title": "数据分析实习生",
                    "recruitment": "internship",
                }},
            })
        if "青禾教育" in text:
            return StructuredCommand.model_validate({
                "intent": "create", "task_relation": "start_new",
                "patch": {"set_fields": {
                    "company_name": "青禾教育", "title": "课程运营",
                }},
            })
        raise AssertionError(text)


def test_range_parallel_skills_and_preview_separator(tmp_path: Path) -> None:
    text = (
        "星河科技现在要招聘一名高级后端工程师。主要负责交易系统设计、服务性能优化和代码评审。"
        "要求本科及以上学历，5到8年后端开发经验，熟悉Java、Spring Boot、MySQL和Redis。"
    )
    agent = JobCsvAgent(empty_repository(tmp_path / "jobs.csv"), ScenarioInterpreter())
    state = agent.handle(initial_state(), text)

    assert state["draft"]["experience_min_months"] == 60
    assert state["draft"]["experience_max_months"] == 96
    assert state["draft"]["education_min_level"] == 4
    assert state["draft"]["skills_json"] == ["Java", "Spring Boot", "MySQL", "Redis"]
    assert state["draft"]["responsibilities_json"] == ["交易系统设计", "服务性能优化", "代码评审"]
    assert "熟悉Java" not in state["draft"].get("requirements_json", [])
    assert "Redis\n\n当前草稿已完整抽取" in state["message"]


def test_compound_education_and_internship_conditions(tmp_path: Path) -> None:
    text = (
        "云帆科技招聘数据分析实习生。工作内容是整理经营数据、搭建销售分析报表，并向业务部门解释数据变化。"
        "任职要求是本科或研究生在读，逻辑清晰，能够独立沟通。"
        "技能要求是熟练使用Excel、SQL和Tableau。实习要求是至少连续实习4个月，每周到岗4天。"
    )
    agent = JobCsvAgent(empty_repository(tmp_path / "jobs.csv"), ScenarioInterpreter())
    state = agent.handle(initial_state(), text)

    assert state["draft"]["education_min_level"] == 4
    assert state["draft"]["student_status_json"] == ["本科在读", "研究生在读"]
    assert state["draft"]["internship_min_months"] == 4
    assert state["draft"]["onsite_days_per_week"] == 4
    assert state["draft"]["skills_json"] == ["Excel", "SQL", "Tableau"]
    assert state["draft"]["requirements_json"] == ["逻辑清晰", "能够独立沟通"]
    assert state["draft"]["responsibilities_json"] == [
        "整理经营数据", "搭建销售分析报表", "向业务部门解释数据变化",
    ]


def test_required_preferred_and_not_required_are_not_inverted(tmp_path: Path) -> None:
    text = (
        "青禾教育招聘课程运营。不要求师范专业，没有教师资格证也可以。"
        "英语只是加分项，不是硬性要求，但必须能够独立完成课程上线。"
    )
    agent = JobCsvAgent(empty_repository(tmp_path / "jobs.csv"), ScenarioInterpreter())
    state = agent.handle(initial_state(), text)

    assert state["draft"]["requirements_json"] == ["独立完成课程上线"]
    assert state["draft"]["preferred_requirements_json"] == ["英语"]
    assert state["draft"]["not_required_requirements_json"] == ["师范专业", "教师资格证"]
    assert state["can_confirm"] is True


def test_historical_evidence_repair_reextracts_previous_user_source(tmp_path: Path) -> None:
    original = "远山公司招聘产品助理，负责整理产品文档。学历要求方面，希望候选人为本科层次。"

    class RepairInterpreter:
        def extract(self, text: str, context: dict) -> StructuredCommand:
            if text == original:
                return StructuredCommand.model_validate({
                    "intent": "create", "task_relation": "start_new",
                    "patch": {
                        "set_fields": {"company_name": "远山公司", "title": "产品助理"},
                        "append_items": {"responsibilities_json": [{"value": "整理产品文档"}]},
                    },
                    "unresolved_fragments": [{
                        "text": "希望候选人为本科层次", "reason": "学历表达待规范化",
                        "suggested_fields": ["education_min_level"], "source_message_id": "user_1",
                    }],
                    "extraction_complete": False,
                })
            assert context["repair"]["source_message_ids"] == ["user_1"]
            return StructuredCommand.model_validate({
                "intent": "update", "task_relation": "continue_current",
                "mentioned_fields": ["education_min_level"],
                "evidence_spans": [{
                    "field": "education_min_level", "text": "希望候选人为本科层次",
                    "source_message_id": "user_1",
                }],
                "patch": {"set_fields": {"education_min_level": 4}},
            })

    agent = JobCsvAgent(empty_repository(tmp_path / "jobs.csv"), RepairInterpreter())
    first = agent.handle(initial_state(), original)
    assert first["extraction_complete"] is False
    repaired = agent.handle(first, "学历信息呢？你再仔细看看")
    assert repaired["draft"]["education_min_level"] == 4
    assert repaired["unresolved_fragments"] == []
    assert repaired["extraction_complete"] is True
    assert repaired["can_confirm"] is True


def test_complex_patch_keeps_successful_fields_and_tracks_only_failures(tmp_path: Path) -> None:
    class PartialInterpreter:
        def extract(self, text: str, context: dict) -> StructuredCommand:
            return StructuredCommand.model_validate({
                "intent": "update",
                "mentioned_fields": ["city", "requirements_json"],
                "evidence_spans": [
                    {"field": "city", "text": "上海"},
                    {"field": "requirements_json", "text": "沟通要求"},
                ],
                "patch": {"set_fields": {"city": "上海"}},
            })

    agent = JobCsvAgent(empty_repository(tmp_path / "jobs.csv"), PartialInterpreter())
    state = initial_state()
    state.update({
        "pending_action": "create", "phase": "CREATING",
        "draft": {
            "company_name": "甲公司", "title": "运营专员",
            "responsibilities_json": ["负责日常运营"],
        },
    })
    result = agent.handle(state, "城市改成上海，沟通要求也调整")
    assert result["draft"]["city"] == "上海"
    assert result["extraction_complete"] is False
    assert result["unresolved_fragments"][0]["suggested_fields"] == ["requirements_json"]
    assert result["can_confirm"] is False


def test_irrelevant_answer_is_recorded_while_requested_field_is_reasked(tmp_path: Path) -> None:
    class RequestedFieldInterpreter:
        def extract(self, text: str, context: dict) -> StructuredCommand:
            if text == "新增乙公司的运营专员岗位":
                return StructuredCommand.model_validate({
                    "intent": "create", "task_relation": "start_new",
                    "patch": {"set_fields": {"company_name": "乙公司", "title": "运营专员"}},
                    "requires_clarification": True,
                    "clarification_question": "请提供岗位职责。",
                    "clarification_fields": ["responsibilities_json"],
                })
            return StructuredCommand.model_validate({
                "intent": "update", "task_relation": "continue_current",
                "mentioned_fields": ["salary_min"],
                "evidence_spans": [{"field": "salary_min", "text": "月薪最低一万"}],
                "patch": {"set_fields": {"salary_min": 10000}},
            })

    agent = JobCsvAgent(empty_repository(tmp_path / "jobs.csv"), RequestedFieldInterpreter())
    first = agent.handle(initial_state(), "新增乙公司的运营专员岗位")
    assert first["requested_fields"] == ["responsibilities_json"]
    second = agent.handle(first, "月薪最低一万")
    assert second["draft"]["salary_min"] == 10000
    assert "本轮其他信息已记录" in second["message"]
    assert "岗位职责" in second["message"]
    assert second["requested_fields"] == ["responsibilities_json"]
