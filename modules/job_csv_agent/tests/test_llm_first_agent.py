from __future__ import annotations

import csv
from pathlib import Path

import pytest
from pydantic import ValidationError

from job_csv_agent.agent import JobCsvAgent, initial_state
from job_csv_agent.guards import guard_model_command
from job_csv_agent.llm import LLMServiceError
from job_csv_agent.repository import CsvJobRepository
from job_csv_agent.schemas import BUSINESS_COLUMNS, LLMInterpretation, Phase, StructuredCommand


def empty_repository(path: Path) -> CsvJobRepository:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.DictWriter(handle, fieldnames=BUSINESS_COLUMNS).writeheader()
    return CsvJobRepository(path)


def interpretation(**updates: object) -> LLMInterpretation:
    payload: dict[str, object] = {
        "intent": "create",
        "route": {"category": "job_write", "action": "create", "confidence": 0.99, "reason": "语义明确"},
        "fields": {},
        "field_evidence": [],
        "semantic_facts": [],
        "confidence": 0.99,
        "requires_clarification": False,
        "clarification_questions": [],
        "coverage": [],
        "natural_reply": None,
        "task_relation": "start_new",
    }
    payload.update(updates)
    return LLMInterpretation.model_validate(payload)


class MappingInterpreter:
    def __init__(self, commands: dict[str, StructuredCommand]) -> None:
        self.commands = commands

    def extract(self, text: str, context: dict) -> StructuredCommand:
        return self.commands[text]


def test_recruitment_only_starts_create_with_empty_title(tmp_path: Path) -> None:
    agent = JobCsvAgent(
        empty_repository(tmp_path / "jobs.csv"),
        MappingInterpreter({"我要招聘": interpretation()}),
    )

    state = agent.handle(initial_state(), "我要招聘")

    assert state["phase"] == Phase.CREATING.value
    assert state["draft"].get("title") is None
    assert "title" in state["missing_fields"]


def test_recruitment_with_title_extracts_complete_name(tmp_path: Path) -> None:
    command = interpretation(
        fields={"title": "算法工程师"},
        field_evidence=[{
            "field": "title", "value": "算法工程师", "evidence_text": "算法工程师",
            "confidence": 0.99, "source_type": "explicit",
        }],
    )
    agent = JobCsvAgent(
        empty_repository(tmp_path / "jobs.csv"),
        MappingInterpreter({"我要招聘算法工程师": command}),
    )

    state = agent.handle(initial_state(), "我要招聘算法工程师")

    assert state["phase"] == Phase.CREATING.value
    assert state["draft"]["title"] == "算法工程师"


def test_creating_conversation_does_not_mutate_draft(tmp_path: Path) -> None:
    commands = {
        "我要招聘算法工程师": interpretation(
            fields={"title": "算法工程师"},
            field_evidence=[{
                "field": "title", "value": "算法工程师", "evidence_text": "算法工程师",
                "confidence": 0.99,
            }],
        ),
        "今天天气不错": interpretation(
            intent="conversation",
            route={"category": "conversation", "action": "chat", "confidence": 0.98, "reason": "闲聊"},
            natural_reply="确实不错。你的岗位草稿还在，我们可以随时继续。",
            task_relation="not_applicable",
        ),
    }
    agent = JobCsvAgent(empty_repository(tmp_path / "jobs.csv"), MappingInterpreter(commands))
    state = agent.handle(initial_state(), "我要招聘算法工程师")
    before = dict(state["draft"])

    state = agent.handle(state, "今天天气不错")

    assert state["phase"] == Phase.CREATING.value
    assert state["draft"] == before
    assert "天气" not in str(state["draft"])


def test_creating_explicit_new_task_replaces_unsaved_draft(tmp_path: Path) -> None:
    commands = {
        "甲公司要招聘算法工程师": interpretation(
            fields={"title": "算法工程师", "company_name": "甲公司"},
            field_evidence=[
                {"field": "title", "value": "算法工程师", "evidence_text": "算法工程师", "confidence": 0.99},
                {"field": "company_name", "value": "甲公司", "evidence_text": "甲公司", "confidence": 0.99},
            ],
        ),
        "另起一个数据工程师岗位": interpretation(
            fields={"title": "数据工程师"},
            field_evidence=[{
                "field": "title", "value": "数据工程师", "evidence_text": "数据工程师", "confidence": 0.99,
            }],
            task_relation="start_new",
        ),
    }
    agent = JobCsvAgent(empty_repository(tmp_path / "jobs.csv"), MappingInterpreter(commands))
    state = agent.handle(initial_state(), "甲公司要招聘算法工程师")
    assert state["draft"]["company_name"] == "甲公司"

    state = agent.handle(state, "另起一个数据工程师岗位")

    assert state["phase"] == Phase.CREATING.value
    assert state["draft"]["title"] == "数据工程师"
    assert state["draft"].get("company_name") is None


def test_creating_read_query_is_read_only_and_keeps_draft(tmp_path: Path) -> None:
    class CountAdapter:
        def ask(self, text: str) -> dict:
            return {"kind": "scalar", "scalar_name": "job_count", "scalar_value": 3}

    commands = {
        "我要招聘算法工程师": interpretation(
            fields={"title": "算法工程师"},
            field_evidence=[{
                "field": "title", "value": "算法工程师", "evidence_text": "算法工程师", "confidence": 0.99,
            }],
        ),
        "现在有多少岗位": interpretation(
            intent="search",
            route={"category": "job_read", "action": "count", "confidence": 0.99, "reason": "只读统计"},
            search_query="现在有多少岗位",
            task_relation="not_applicable",
        ),
    }
    agent = JobCsvAgent(
        empty_repository(tmp_path / "jobs.csv"), MappingInterpreter(commands), CountAdapter(),
    )
    state = agent.handle(initial_state(), "我要招聘算法工程师")
    before = dict(state["draft"])

    state = agent.handle(state, "现在有多少岗位")

    assert state["phase"] == Phase.CREATING.value
    assert state["draft"] == before
    assert state["message"] == "当前共有 3 条岗位记录。"


def test_low_confidence_asks_without_guessing(tmp_path: Path) -> None:
    command = interpretation(
        fields={"title": "算法工程师"},
        field_evidence=[{
            "field": "title", "value": "算法工程师", "evidence_text": "算法工程师",
            "confidence": 0.40,
        }],
        route={"category": "job_write", "action": "create", "confidence": 0.40, "reason": "可能是新岗位"},
        confidence=0.40,
        requires_clarification=True,
        clarification_questions=["你是想新建算法工程师岗位，还是查询已有岗位？"],
        natural_reply="你是想新建算法工程师岗位，还是查询已有岗位？",
    )
    agent = JobCsvAgent(
        empty_repository(tmp_path / "jobs.csv"), MappingInterpreter({"算法工程师": command}),
    )

    state = agent.handle(initial_state(), "算法工程师")

    assert state["phase"] == Phase.IDLE.value
    assert state["draft"] == {}
    assert "还是" in state["message"]


def test_model_unavailable_keeps_existing_draft_and_adds_no_fields(tmp_path: Path) -> None:
    class UnavailableInterpreter:
        def extract(self, text: str, context: dict) -> StructuredCommand:
            raise LLMServiceError("offline")

    state = initial_state()
    state.update({"phase": Phase.CREATING.value, "draft": {"title": "算法工程师"}})
    agent = JobCsvAgent(empty_repository(tmp_path / "jobs.csv"), UnavailableInterpreter())

    result = agent.handle(state, "公司可能叫示例科技")

    assert result["phase"] == Phase.CREATING.value
    assert result["draft"] == {"title": "算法工程师"}
    assert result["draft"].get("company_name") is None
    assert "没有修改任何字段" in result["message"]


def test_one_character_title_is_rejected_by_field_format_guard() -> None:
    with pytest.raises(ValidationError):
        interpretation(
            fields={"title": "聘"},
            field_evidence=[{
                "field": "title", "value": "聘", "evidence_text": "聘", "confidence": 0.99,
            }],
        )


def test_jd_projection_requires_typed_fact_with_source_evidence() -> None:
    source = "岗位职责：负责数据平台建设"
    accepted = interpretation(
        fields={"responsibilities_json": ["建设数据平台"]},
        semantic_facts=[{
            "value": "建设数据平台", "category": "responsibility", "importance": "neutral",
            "source_type": "explicit", "evidence_text": "负责数据平台建设",
        }],
    )
    guarded = guard_model_command(accepted, source)
    assert guarded.fields.responsibilities_json == ["建设数据平台"]

    ungrounded = interpretation(
        fields={"requirements_json": ["精通 Python"]},
        semantic_facts=[{
            "value": "精通 Python", "category": "requirement", "importance": "must",
            "source_type": "explicit", "evidence_text": "精通 Python",
        }],
    )
    guarded = guard_model_command(ungrounded, source)
    assert guarded.fields.requirements_json == []
    assert guarded.clarification_questions


def test_delete_requires_confirmation_and_cancel_does_not_write(tmp_path: Path) -> None:
    repository = empty_repository(tmp_path / "jobs.csv")
    created = repository.create({
        "company_name": "示例科技", "title": "算法工程师",
        "responsibilities_json": ["负责模型评估"],
    })
    delete_command = interpretation(
        intent="delete",
        route={"category": "job_write", "action": "delete", "confidence": 0.99, "reason": "删除已有岗位"},
        search_query="示例科技 算法工程师",
        task_relation="not_applicable",
    )
    agent = JobCsvAgent(
        repository,
        MappingInterpreter({"删除示例科技的算法工程师": delete_command}),
    )

    pending = agent.handle(initial_state(), "删除示例科技的算法工程师")
    assert pending["phase"] == Phase.CONFIRMING.value
    assert repository.get(created["job_id"])["title"] == "算法工程师"

    cancelled = agent.handle(pending, "取消")
    assert cancelled["phase"] == Phase.IDLE.value
    assert repository.get(created["job_id"])["title"] == "算法工程师"

    pending = agent.handle(cancelled, "删除示例科技的算法工程师")
    deleted = agent.handle(pending, "确认")
    assert deleted["phase"] == Phase.IDLE.value
    assert repository.search("示例科技 算法工程师") == []
