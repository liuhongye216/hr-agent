from __future__ import annotations

import csv
from pathlib import Path

import pytest

from job_csv_agent.agent import JobCsvAgent, initial_state
from job_csv_agent.jd_text2sql_adapter import JDText2SQLAdapter
from job_csv_agent.repository import CsvJobRepository
from job_csv_agent.schemas import (
    BUSINESS_COLUMNS, ConstrainedRewriteBatch, Phase, StructuredCommand,
)
from job_csv_agent.semantics import (
    analyze_jd_text, apply_constrained_rewrites, reconcile_command_semantics,
)


class NoCallInterpreter:
    def extract(self, text: str, context: dict) -> StructuredCommand:
        if text in {
            "统计有多少条目", "一共有几个岗位", "目前岗位数量是多少", "帮我数一下岗位",
            "现在收录了多少条招聘信息", "本科及以上岗位有多少", "北京有多少实习岗位",
        }:
            return StructuredCommand.model_validate({
                "intent": "search",
                "route": {"category": "job_read", "action": "count", "confidence": 0.99},
                "search_query": text,
            })
        if text == "你好":
            return StructuredCommand.model_validate({
                "intent": "conversation",
                "route": {"category": "conversation", "action": "chat", "confidence": 0.99},
                "natural_reply": "你好！我在。",
            })
        if text == "我要招正式工":
            return StructuredCommand.model_validate({
                "intent": "create",
                "route": {"category": "job_write", "action": "create", "confidence": 0.99},
                "fields": {"employment": "full_time"},
            })
        if text == "你能做什么":
            return StructuredCommand.model_validate({
                "intent": "help",
                "route": {"category": "help", "action": "explain_capabilities", "confidence": 0.99},
                "natural_reply": "我可以查询岗位、整理 JD；所有写入都要经过你的明确确认。",
            })
        if text == "示例科技有限公司 清洁工程师":
            return StructuredCommand.model_validate({
                "intent": "update",
                "route": {"category": "job_write", "action": "update", "confidence": 0.99},
                "fields": {"company_name": "示例科技有限公司", "title": "清洁工程师"},
            })
        raise AssertionError(f"missing model stub for: {text}")


def repository_with_jobs(tmp_path: Path) -> tuple[CsvJobRepository, JDText2SQLAdapter]:
    csv_path = tmp_path / "jobs.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.DictWriter(handle, fieldnames=BUSINESS_COLUMNS).writeheader()
    repository = CsvJobRepository(csv_path)
    repository.create({
        "company_name": "甲公司", "title": "算法实习生", "city": "北京",
        "recruitment": "internship", "employment": "internship",
        "education_min_level": 4, "responsibilities_json": ["参与模型评测"],
    })
    repository.create({
        "company_name": "乙公司", "title": "数据工程师", "city": "上海",
        "education_min_level": 4, "responsibilities_json": ["负责数据平台建设"],
    })
    repository.create({
        "company_name": "丙公司", "title": "运营专员", "city": "北京",
        "education_min_level": 3, "responsibilities_json": ["负责内容运营"],
    })
    adapter = JDText2SQLAdapter(csv_path, tmp_path / "jobs.sqlite")
    return repository, adapter


@pytest.mark.parametrize("question", [
    "统计有多少条目",
    "一共有几个岗位",
    "目前岗位数量是多少",
    "帮我数一下岗位",
    "现在收录了多少条招聘信息",
])
def test_count_synonyms_render_scalar(tmp_path: Path, question: str) -> None:
    repository, adapter = repository_with_jobs(tmp_path)
    state = JobCsvAgent(repository, NoCallInterpreter(), adapter).handle(initial_state(), question)
    assert state["phase"] == Phase.IDLE.value
    assert state["message"] == "当前共有 3 条岗位记录。"
    assert "找到 1 个岗位" not in state["message"]


@pytest.mark.parametrize(("question", "expected"), [
    ("本科及以上岗位有多少", 2),
    ("北京有多少实习岗位", 1),
])
def test_conditional_counts(tmp_path: Path, question: str, expected: int) -> None:
    repository, adapter = repository_with_jobs(tmp_path)
    state = JobCsvAgent(repository, NoCallInterpreter(), adapter).handle(initial_state(), question)
    assert state["message"] == f"当前共有 {expected} 条岗位记录。"


def test_chat_is_read_only_and_natural(tmp_path: Path) -> None:
    repository, adapter = repository_with_jobs(tmp_path)
    agent = JobCsvAgent(repository, NoCallInterpreter(), adapter)
    before = adapter.ask("一共有几个岗位")["scalar_value"]
    state = agent.handle(initial_state(), "你好")
    after = adapter.ask("一共有几个岗位")["scalar_value"]
    assert "你好" in state["message"]
    assert before == after == 3


def test_help_interrupts_and_resumes_creation_without_mutating_draft(tmp_path: Path) -> None:
    csv_path = tmp_path / "jobs.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.DictWriter(handle, fieldnames=BUSINESS_COLUMNS).writeheader()
    agent = JobCsvAgent(CsvJobRepository(csv_path), NoCallInterpreter())
    state = agent.handle(initial_state(), "我要招正式工")
    before = dict(state["draft"])
    state = agent.handle(state, "你能做什么")
    assert state["draft"] == before
    assert "所有写入都要经过你的明确确认" in state["message"]
    state = agent.handle(state, "示例科技有限公司 清洁工程师")
    assert state["draft"]["company_name"] == "示例科技有限公司"
    assert state["draft"]["title"] == "清洁工程师"
    assert "你能做什么" not in str(state["draft"])


def test_section_aware_classification_shared_object_and_coverage() -> None:
    text = """岗位职责：
1、协助推进全链路情报分析与风险决策能力的建模和优化
2、支持复杂任务下 Agent 能力评测、训练数据构造和效果分析
3、参与可扩展 Agent 训练环境的设计、搭建与迭代
职位要求
1、硕士研究生及以上在读，熟悉 Python 和 PyTorch
"""
    command = reconcile_command_semantics(StructuredCommand(
        intent="create", fields={"company_name": "示例公司", "title": "Agent 研究实习生"},
    ), text)
    fields = command.fields.model_dump(exclude_none=True)
    assert fields["responsibilities_json"] == [
        "协助推进全链路情报分析与风险决策能力的建模和优化",
        "支持复杂任务下 Agent 能力评测、训练数据构造和效果分析",
        "参与可扩展 Agent 训练环境的设计、搭建与迭代",
    ]
    assert all("搭建与迭代" != item for item in fields["responsibilities_json"])
    assert fields["education_min_level"] == 5
    assert fields["requirements_json"] == ["硕士研究生及以上在读", "熟悉 Python 和 PyTorch"]
    assert fields["skills_json"] == ["Python", "PyTorch"]
    assert all("职位要求" not in item and not item.startswith("1、") for item in fields["requirements_json"])
    assert len(command.coverage) == 4
    assert all(item.status != "ignored" for item in command.coverage)


def test_every_numbered_item_has_a_coverage_decision() -> None:
    analysis = analyze_jd_text("岗位职责\n1、负责数据整理\n2、团队氛围很好")
    assert len(analysis.coverage) == 2
    assert {item.status for item in analysis.coverage} == {"projected_responsibility", "needs_clarification"}
    assert analysis.clarification_questions


def test_traceable_llm_rewrite_is_accepted_without_losing_shared_object() -> None:
    source = "参与可扩展 Agent 训练环境的设计、搭建与迭代"
    command = reconcile_command_semantics(StructuredCommand.model_validate({
        "intent": "create",
        "fields": {"company_name": "示例公司", "title": "Agent 工程师", "responsibilities_json": [source]},
    }), source)
    rewritten = apply_constrained_rewrites(command, ConstrainedRewriteBatch.model_validate({
        "rewrites": [{
            "rewritten_text": source,
            "category": "responsibility",
            "evidence_indices": [0],
            "preserves_strength": True,
        }],
    }), source)
    assert rewritten.fields.responsibilities_json == [source]
    assert rewritten.semantic_facts[0].evidence_texts == [source]


def test_llm_rewrite_cannot_inject_unmentioned_training_techniques() -> None:
    source = "负责大模型后训练"
    command = reconcile_command_semantics(StructuredCommand.model_validate({
        "intent": "create",
        "fields": {"company_name": "示例公司", "title": "后训练工程师", "responsibilities_json": [source]},
    }), source)
    rewritten = apply_constrained_rewrites(command, ConstrainedRewriteBatch.model_validate({
        "rewrites": [{
            "rewritten_text": "负责 SFT、RLHF、DPO 和分布式训练",
            "category": "responsibility",
            "evidence_indices": [0],
            "preserves_strength": True,
        }],
    }), source)
    formal = " ".join(rewritten.fields.responsibilities_json or [])
    assert all(term not in formal for term in ("SFT", "RLHF", "DPO", "分布式训练"))
    assert formal == source


def test_mechanical_awareness_ability_phrase_is_repaired() -> None:
    command = reconcile_command_semantics(StructuredCommand.model_validate({
        "intent": "create",
        "fields": {
            "company_name": "示例公司", "title": "工程师",
            "requirements_json": ["具备良好的团队合作意识能力"],
        },
    }), "具备良好的团队合作意识能力")
    assert command.fields.requirements_json == ["具备良好的团队合作意识"]
