from __future__ import annotations

import csv
import json
import re
import sqlite3
from pathlib import Path

import pytest

from job_csv_agent.agent import JobCsvAgent, _grouped_suggestions, initial_state
from job_csv_agent.jd_text2sql_adapter import JDText2SQLAdapter
from job_csv_agent.repository import CsvJobRepository, SemanticValidationError
from job_csv_agent.schemas import BUSINESS_COLUMNS, Phase, StructuredCommand
from job_csv_agent.semantics import (
    clean_text, classify_atomic_text, normalize_job_fields, reconcile_command_semantics,
    semantic_review, split_atomic_clauses,
)


class NoCallInterpreter:
    def extract(self, text: str, context: dict) -> StructuredCommand:
        raise AssertionError(f"unexpected external interpreter call: {text}")


def empty_repository(path: Path) -> CsvJobRepository:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.DictWriter(handle, fieldnames=BUSINESS_COLUMNS).writeheader()
    return CsvJobRepository(path)


def create_preview(agent: JobCsvAgent) -> dict:
    return agent.handle(
        initial_state(),
        "创建美团大模型实习生，熟练掌握Python或Java，具备扎实的编程基础，负责大模型应用开发，技能Python、Java",
        {
        "intent": "create",
        "fields": {
            "company_name": "美团", "title": "大模型实习生",
            "recruitment": "internship", "employment": "internship", "city": "北京",
            "requirements_json": ["熟练掌握Python或Java，具备扎实的编程基础。"],
            "responsibilities_json": ["负责大模型应用开发"],
            "skills_json": ["Python", "Java"],
        },
        },
    )


def message_json(message: str) -> dict:
    match = re.search(r"```json\n(.*?)\n```", message, re.DOTALL)
    assert match, message
    return json.loads(match.group(1))


def test_abnormal_punctuation_is_cleaned() -> None:
    cleaned = clean_text("熟练掌握Python。、，、、、以及Java。。")
    assert all(token not in cleaned for token in ("。、", "，、", "、以及", "。。"))
    assert not cleaned.endswith(("。", "、", "，"))


def test_and_clauses_are_atomic() -> None:
    assert split_atomic_clauses("具备沟通能力，并且具备团队协作能力") == [
        "具备沟通能力", "具备团队协作能力",
    ]


def test_yiji_splits_only_independent_conditions() -> None:
    assert split_atomic_clauses("具备沟通能力以及具备团队协作能力") == [
        "具备沟通能力", "具备团队协作能力",
    ]


@pytest.mark.parametrize("text", ["熟练掌握Python或Java", "熟悉FastAPI或Django"])
def test_explicit_or_is_preserved(text: str) -> None:
    atoms = split_atomic_clauses(text)
    assert len(atoms) == 1
    assert " 或 " in atoms[0]


def test_ambiguous_tool_list_is_not_rewritten_as_and_or() -> None:
    fields = normalize_job_fields({"requirements_json": ["熟悉Spark、Flink、Doris等大数据工具"]})
    assert fields["requirements_json"] == ["熟悉 Spark、Flink、Doris 等大数据工具"]
    issues = semantic_review(fields)
    assert any(item.level == "warning" for item in issues)


def test_experience_is_scalar_and_removed_from_requirement_text() -> None:
    fields = normalize_job_fields({
        "requirements_json": ["具有至少1年以上linux后台系统相关研发经历，扎实的CS基础。"],
    })
    assert fields["experience_min_months"] == 12
    assert fields["requirements_json"] == [
        "具有 Linux 后台系统研发经验", "具备扎实的计算机基础",
    ]
    assert "1年" not in "".join(fields["requirements_json"])


def test_shared_predicates_are_completed() -> None:
    fields = normalize_job_fields({
        "requirements_json": ["具备良好的沟通能力和团队协作精神，学习能力强"],
    })
    assert fields["requirements_json"] == [
        "具备良好的沟通能力", "具备良好的团队协作能力", "学习能力强",
    ]


def test_software_engineering_requirement_is_split_and_not_a_duty() -> None:
    text = "具有较好的软件工程知识和编码规范意识，对代码和设计质量有严格要求。"
    fields = normalize_job_fields({"requirements_json": [text]})
    assert fields["requirements_json"] == [
        "具备良好的软件工程知识", "具备良好的编码规范意识", "重视代码与设计质量",
    ]
    assert not fields.get("responsibilities_json")
    assert all(item.level != "blocking" for item in semantic_review(fields))


def test_service_stack_is_atomized_without_strengthening() -> None:
    fields = normalize_job_fields({
        "requirements_json": [
            "熟悉服务架构搭建、数据库使用与原理、以及中间件技术，如FastAPI/Django、MySQL、MQ等。"
        ],
    })
    assert fields["requirements_json"] == [
        "熟悉服务架构设计与搭建", "掌握数据库使用及基本原理", "熟悉 FastAPI 或 Django",
        "熟悉 MySQL", "熟悉消息队列（MQ）",
    ]
    assert fields["skills_json"] == ["FastAPI", "Django", "MySQL", "MQ"]


def test_duplicate_or_requirement_is_removed() -> None:
    fields = normalize_job_fields({
        "requirements_json": ["熟练掌握Python或Java。", "熟练掌握 Python 或 Java、"],
    })
    assert fields["requirements_json"] == ["熟练掌握 Python 或 Java"]


def test_skill_list_contains_labels_only_and_moves_full_requirement() -> None:
    fields = normalize_job_fields({
        "skills_json": ["SQL", "MySQL", "具有 Linux 后台系统研发经验"],
    })
    assert fields["skills_json"] == ["SQL", "MySQL", "Linux"]
    assert fields["requirements_json"] == ["具有 Linux 后台系统研发经验"]


def test_existing_content_is_not_repeated_as_suggestion() -> None:
    draft = {
        "requirements_json": ["熟练掌握 Python 或 Java"],
        "responsibilities_json": ["负责大模型应用评测"],
        "skills_json": ["Python", "Java"],
    }
    facts = [
        {"value": "Python", "category": "skill", "source_type": "inferred"},
        {"value": "负责大模型应用评测", "category": "responsibility", "source_type": "inferred"},
    ]
    assert _grouped_suggestions(draft, facts, []) == {}


def test_suggestions_are_grouped_and_limited_to_three() -> None:
    grouped = _grouped_suggestions({}, [], [
        "岗位职责建议确认：是否负责模型部署？",
        "任职要求建议确认：Spark、Flink、Doris 是全部要求吗？",
        "技能要求建议确认：是否加入 Linux？",
        "工作信息建议确认：请补充工作城市？",
    ])
    assert set(grouped) == {"岗位职责", "任职要求", "技能要求"}
    assert sum(len(items) for items in grouped.values()) == 3


def test_forced_command_does_not_generate_semantic_suggestions(tmp_path: Path) -> None:
    repository = empty_repository(tmp_path / "jobs.csv")
    agent = JobCsvAgent(repository, NoCallInterpreter())
    state = agent.handle(initial_state(), "创建预训练岗位，职责是从0开始预训练大模型", {
        "intent": "create",
        "fields": {
            "company_name": "示例科技", "title": "预训练工程师",
            "responsibilities_json": ["从0开始预训练大模型"],
        },
    })
    assert "建议补充" not in state["message"]


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("具有开发经验", "requirement"),
        ("具有较好的软件工程知识和编码规范意识，对代码和设计质量有严格要求", "requirement"),
        ("负责系统开发", "responsibility"),
    ],
)
def test_qualification_priority_and_clear_action_structure(text: str, category: str) -> None:
    core = [fact for fact in classify_atomic_text(text) if fact.category != "skill"]
    assert core
    assert {fact.category for fact in core} == {category}


def test_repository_warning_text_does_not_become_value_error(tmp_path: Path) -> None:
    repository = empty_repository(tmp_path / "jobs.csv")
    row = repository.create({
        "company_name": "美团", "title": "大模型实习生",
        "requirements_json": ["具有较好的软件工程知识和编码规范意识，对代码和设计质量有严格要求"],
    })
    assert row["job_id"]


def test_repository_still_blocks_unconfirmed_inference(tmp_path: Path) -> None:
    repository = empty_repository(tmp_path / "jobs.csv")
    fact = [{
        "value": "熟悉 Linux", "category": "requirement", "importance": "unknown",
        "source_type": "inferred", "evidence_text": "后台研发", "needs_confirmation": True,
    }]
    with pytest.raises(SemanticValidationError, match="未经用户确认"):
        repository.create({
            "company_name": "美团", "title": "后台工程师", "requirements_json": ["熟悉 Linux"],
        }, fact)


def test_preview_contains_strict_business_json(tmp_path: Path) -> None:
    state = create_preview(JobCsvAgent(empty_repository(tmp_path / "jobs.csv"), NoCallInterpreter()))
    preview = message_json(state["message"])
    assert preview["公司名称"] == "美团"
    assert preview["任职要求"] == ["熟练掌握Python或Java，具备扎实的编程基础"]
    assert "requirements_json" not in state["message"]


def test_editable_chinese_json_updates_draft_and_runs_pipeline(tmp_path: Path) -> None:
    agent = JobCsvAgent(empty_repository(tmp_path / "jobs.csv"), NoCallInterpreter())
    state = create_preview(agent)
    edited = message_json(state["message"])
    edited["任职要求"] = ["具有至少1年以上linux后台系统相关研发经历，扎实的CS基础。"]
    state = agent.handle(state, json.dumps(edited, ensure_ascii=False))
    assert state["phase"] == Phase.CONFIRMING.value
    assert state["pending_fields"].get("experience_min_months") is None
    assert state["pending_fields"]["requirements_json"] == [
        "具有至少1年以上linux后台系统相关研发经历，扎实的CS基础",
    ]


def test_invalid_json_keeps_original_draft(tmp_path: Path) -> None:
    agent = JobCsvAgent(empty_repository(tmp_path / "jobs.csv"), NoCallInterpreter())
    state = create_preview(agent)
    before = dict(state["pending_fields"])
    after = agent.handle(state, '{"公司名称": "美团",')
    assert after["pending_fields"] == before
    assert "JSON 格式错误" in after["message"]
    assert "原岗位草稿已保留" in after["message"]


def test_empty_array_clears_list_and_null_marks_scalar_unfilled(tmp_path: Path) -> None:
    agent = JobCsvAgent(empty_repository(tmp_path / "jobs.csv"), NoCallInterpreter())
    state = create_preview(agent)
    edited = message_json(state["message"])
    edited["任职要求"] = []
    edited["工作城市"] = None
    state = agent.handle(state, json.dumps(edited, ensure_ascii=False))
    assert state["pending_fields"]["requirements_json"] == []
    assert "city" not in state["pending_fields"]
    assert state["pending_fields"]["responsibilities_json"] == ["负责大模型应用开发"]


def test_reconcile_adds_ambiguous_list_question() -> None:
    command = StructuredCommand.model_validate({
        "intent": "create",
        "fields": {
            "company_name": "美团", "title": "大数据工程师",
            "requirements_json": ["熟悉Spark、Flink、Doris等大数据工具"],
        },
    })
    result = reconcile_command_semantics(command, "美团大数据工程师，熟悉Spark、Flink、Doris等大数据工具")
    assert any("全部要求" in item and "其中一种" in item for item in result.clarification_questions)


def test_meituan_sample_end_to_end_json_edit_confirm_and_sqlite_sync(tmp_path: Path) -> None:
    repository = empty_repository(tmp_path / "jobs.csv")
    adapter = JDText2SQLAdapter(tmp_path / "jobs.csv", tmp_path / "jobs.sqlite")
    agent = JobCsvAgent(repository, NoCallInterpreter(), adapter)
    requirements = [
        "具有至少1年以上linux后台系统相关研发经历，扎实的CS基础。",
        "熟练掌握Python或Java，具备扎实的编程基础。",
        "熟悉服务架构搭建、数据库使用与原理、以及中间件技术，如FastAPI/Django、MySQL、MQ等。",
        "具有较好的软件工程知识和编码规范意识，对代码和设计质量有严格要求。",
        "熟悉Spark、Flink、Doris等大数据工具",
        "了解SQL解析原理及相关应用",
        "具有大模型应用评测经验者优先",
        "具有大模型应用调优经验者优先",
        "具备良好的沟通能力和团队协作精神，学习能力强",
    ]
    responsibilities = [
        "负责大数据领域 SQL/ETL Agent 的设计、开发与优化",
        "使用 Python 或 Java 开展大模型应用开发并推动业务落地",
        "基于 LangGraph、AutoGen、CrewAI 等框架完成项目开发与集成",
        "开展大模型应用评测并持续优化模型效果与性能",
    ]
    original = "美团大模型实习生。" + "。".join([*requirements, *responsibilities])
    state = agent.handle(initial_state(), original, {
        "intent": "create",
        "fields": {
            "company_name": "美团", "title": "大模型实习生",
            "recruitment": "internship", "employment": "internship",
            "requirements_json": requirements, "responsibilities_json": responsibilities,
        },
    })
    preview = message_json(state["message"])
    assert preview["最低经验月数"] is None
    assert preview["任职要求"] == [item.rstrip("。") for item in requirements]
    assert preview["技能要求"] == []
    assert all("。、" not in item and not item.endswith("。") for item in preview["任职要求"])

    preview["工作城市"] = "北京"
    state = agent.handle(state, json.dumps(preview, ensure_ascii=False))
    assert message_json(state["message"])["工作城市"] == "北京"

    state = agent.handle(state, "确认写入", {"intent": "confirm"})
    assert state["phase"] == Phase.CONFIRMING.value
    state = agent.handle(state, "确认写入", {"intent": "confirm"})
    assert state["phase"] == Phase.IDLE.value
    assert len(repository.search("美团 大模型实习生")) == 1
    assert adapter.db_path.exists()
    with sqlite3.connect(adapter.db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
