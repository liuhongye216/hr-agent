from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import Mock

from job_csv_agent.agent import JobCsvAgent, initial_state
from job_csv_agent.llm import deterministic_command
from job_csv_agent.repository import CsvJobRepository
from job_csv_agent.schemas import BUSINESS_COLUMNS, Phase, StructuredCommand
from job_csv_agent.semantics import reconcile_command_semantics, rewrite_jd_text


class NoCallInterpreter:
    def extract(self, text: str, context: dict) -> StructuredCommand:
        commands = {
            "我要招正式工": {
                "intent": "create", "fields": {"employment": "full_time"},
                "route": {"category": "job_write", "action": "create", "confidence": 0.99},
            },
            "后浪澎湃公司 保洁员": {
                "intent": "update", "fields": {"company_name": "后浪澎湃公司", "title": "保洁员"},
                "route": {"category": "job_write", "action": "update", "confidence": 0.99},
            },
            "我要招大模型后训练实习生": {
                "intent": "create",
                "fields": {"title": "大模型后训练实习生", "employment": "internship", "recruitment": "internship"},
                "clarification_questions": ["后训练范围是 SFT、RLHF/DPO、奖励模型、模型评估，还是其他任务？"],
                "route": {"category": "job_write", "action": "create", "confidence": 0.99},
            },
            "后浪澎湃科技有限公司，后训练是训练一个招聘模型，技能主要是懂llm": {
                "intent": "update",
                "fields": {
                    "company_name": "后浪澎湃科技有限公司",
                    "responsibilities_json": ["负责招聘领域大模型的后训练工作"],
                    "requirements_json": ["具备大语言模型（LLM）相关基础知识"],
                },
                "semantic_facts": [
                    {"value": "负责招聘领域大模型的后训练工作", "category": "responsibility", "importance": "neutral", "source_type": "explicit", "evidence_text": "后训练是训练一个招聘模型"},
                    {"value": "具备大语言模型（LLM）相关基础知识", "category": "requirement", "importance": "must", "source_type": "explicit", "evidence_text": "懂llm"},
                ],
                "route": {"category": "job_write", "action": "update", "confidence": 0.99},
            },
            "修改后浪澎湃公司的保洁员岗位": {
                "intent": "update", "search_query": "后浪澎湃公司 保洁员",
                "route": {"category": "job_write", "action": "update", "confidence": 0.99},
            },
        }
        if text not in commands:
            raise AssertionError(f"missing model stub for: {text}")
        return StructuredCommand.model_validate(commands[text])


def empty_repository(path: Path) -> CsvJobRepository:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.DictWriter(handle, fieldnames=BUSINESS_COLUMNS).writeheader()
    return CsvJobRepository(path)


def assert_no_internal_terms(message: str) -> None:
    forbidden = (
        "job_id", "requirements_json", "responsibilities_json", "skills_json",
        "csv", "schema", "intent", "phase", "extraction_mode", "inferred",
    )
    lowered = message.casefold()
    assert not [term for term in forbidden if term in lowered]


def test_deterministic_business_language_routing_is_disabled() -> None:
    assert deterministic_command("我要招正式工") is None
    assert deterministic_command("修改后浪澎湃公司的保洁员岗位") is None
    assert deterministic_command("查询后浪澎湃公司的保洁员岗位") is None
    assert deterministic_command("删除后浪澎湃公司的保洁员岗位") is None


def test_creating_state_inherits_company_and_title_without_search(tmp_path: Path) -> None:
    repository = empty_repository(tmp_path / "jobs.csv")
    repository.search = Mock(wraps=repository.search)
    agent = JobCsvAgent(repository, NoCallInterpreter())

    state = agent.handle(initial_state(), "我要招正式工")
    assert state["phase"] == Phase.CREATING.value
    assert state["draft"]["employment"] == "full_time"

    state = agent.handle(state, "后浪澎湃公司 保洁员")
    assert state["phase"] == Phase.CREATING.value
    assert state["draft"]["company_name"] == "后浪澎湃公司"
    assert state["draft"]["title"] == "保洁员"
    repository.search.assert_not_called()
    assert "没有找到" not in state["message"]
    assert_no_internal_terms(state["message"])


def test_recruitment_request_is_not_projected_as_job_content() -> None:
    command = StructuredCommand.model_validate({
        "intent": "create",
        "fields": {
            "title": "大模型后训练实习生", "employment": "internship",
            "recruitment": "internship",
            "responsibilities_json": ["我要招大模型后训练实习生"],
        },
        "semantic_facts": [{
            "value": "我要招大模型后训练实习生", "category": "responsibility",
            "importance": "neutral", "source_type": "explicit",
            "evidence_text": "我要招大模型后训练实习生",
        }],
    })
    cleaned = reconcile_command_semantics(command, "我要招大模型后训练实习生")
    assert cleaned.fields.responsibilities_json is None
    assert all("我要招" not in fact.value for fact in cleaned.semantic_facts)


def test_safe_rewrite_and_confidence_hard_rules() -> None:
    duty = rewrite_jd_text(
        "后训练是训练一个招聘模型", "responsibility",
        evidence_text="后训练是训练一个招聘模型",
    )
    assert duty.decision == "safe_rewrite"
    assert duty.rewritten_text == "负责招聘领域大模型的后训练工作"

    requirement = rewrite_jd_text("懂llm", "requirement", evidence_text="懂llm")
    assert requirement.decision == "safe_rewrite"
    assert requirement.rewritten_text == "具备大语言模型（LLM）相关基础知识"
    assert "精通" not in requirement.rewritten_text

    suggestion = rewrite_jd_text(
        "熟悉 PyTorch", "requirement", source_type="inferred",
        evidence_text="大模型后训练岗位",
        model_confidence=0.99,
    )
    assert suggestion.decision == "needs_confirmation"

    overclaim = rewrite_jd_text("精通 Python", "requirement", evidence_text="会用Python")
    assert overclaim.decision == "reject_or_clarify"


def test_large_model_end_to_end_preview_and_question_dedup(tmp_path: Path) -> None:
    repository = empty_repository(tmp_path / "jobs.csv")
    agent = JobCsvAgent(repository, NoCallInterpreter())

    first = agent.handle(initial_state(), "我要招大模型后训练实习生")
    assert first["phase"] == Phase.CREATING.value
    assert first["draft"]["title"] == "大模型后训练实习生"
    assert first["draft"]["employment"] == "internship"
    assert first["draft"].get("responsibilities_json") is None
    assert first["message"].count("？") <= 3
    assert_no_internal_terms(first["message"])

    scope_question = "后训练范围是 SFT、RLHF/DPO、奖励模型、模型评估，还是其他任务？"
    assert scope_question in first["message"]

    second = agent.handle(
        first,
        "后浪澎湃科技有限公司，后训练是训练一个招聘模型，技能主要是懂llm",
    )
    assert second["phase"] == Phase.CONFIRMING.value
    pending = second["pending_fields"]
    assert pending["company_name"] == "后浪澎湃科技有限公司"
    assert pending["title"] == "大模型后训练实习生"
    assert pending["responsibilities_json"] == ["负责招聘领域大模型的后训练工作"]
    assert pending["requirements_json"] == ["具备大语言模型（LLM）相关基础知识"]
    assert scope_question not in second["message"]
    assert second["message"].count("？") <= 3
    assert_no_internal_terms(second["message"])

    formal_text = " ".join(
        item for field in ("responsibilities_json", "requirements_json", "skills_json")
        for item in pending.get(field, [])
    )
    for unconfirmed in ("RLHF", "SFT", "DPO", "Python", "PyTorch", "分布式训练"):
        assert unconfirmed not in formal_text


def test_recognized_company_is_removed_from_unknown() -> None:
    command = StructuredCommand.model_validate({
        "intent": "update",
        "fields": {"company_name": "后浪澎湃科技有限公司"},
        "semantic_facts": [{
            "value": "后浪澎湃科技有限公司", "category": "unknown",
            "importance": "unknown", "source_type": "explicit",
            "evidence_text": "后浪澎湃科技有限公司", "needs_confirmation": True,
        }],
    })
    cleaned = reconcile_command_semantics(command, "后浪澎湃科技有限公司")
    assert not cleaned.semantic_facts


def test_user_confirmed_suggestion_can_enter_draft() -> None:
    command = StructuredCommand.model_validate({
        "intent": "update",
        "fields": {"responsibilities_json": ["负责模型效果评估与结果分析。"]},
        "semantic_facts": [{
            "value": "负责模型效果评估与结果分析。", "category": "responsibility",
            "importance": "neutral", "source_type": "user_confirmed",
            "evidence_text": "确认需要做模型评估",
        }],
    })
    cleaned = reconcile_command_semantics(command, "确认需要做模型评估")
    assert cleaned.fields.responsibilities_json == ["负责模型效果评估与结果分析"]


def test_explicit_update_not_found_message_is_natural(tmp_path: Path) -> None:
    repository = empty_repository(tmp_path / "jobs.csv")
    agent = JobCsvAgent(repository, NoCallInterpreter())
    state = agent.handle(initial_state(), "修改后浪澎湃公司的保洁员岗位")
    assert state["phase"] == Phase.EDITING.value
    assert "没有找到对应岗位" in state["message"]
    assert_no_internal_terms(state["message"])
