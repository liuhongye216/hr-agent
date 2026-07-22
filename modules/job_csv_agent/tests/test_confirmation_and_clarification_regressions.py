from __future__ import annotations

import csv
from pathlib import Path

from fastapi.testclient import TestClient

from job_csv_agent.agent import JobCsvAgent, initial_state
from job_csv_agent.api import create_app
from job_csv_agent.llm import LLMServiceError
from job_csv_agent.normalization import infer_atomic_mentions, normalize_patch
from job_csv_agent.patching import apply_draft_patch, requirement_modality_conflicts
from job_csv_agent.repository import CsvJobRepository
from job_csv_agent.schemas import AtomicMention, BUSINESS_COLUMNS, DraftPatch, StructuredCommand


def empty_repository(path: Path) -> CsvJobRepository:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.DictWriter(handle, fieldnames=BUSINESS_COLUMNS).writeheader()
    return CsvJobRepository(path)


class MisclassifyingConfirmInterpreter:
    def extract(self, text: str, context: dict) -> StructuredCommand:
        if text == "？":
            return StructuredCommand(intent="confirm")
        return StructuredCommand.model_validate({
            "intent": "create",
            "task_relation": "start_new",
            "patch": {
                "set_fields": {"company_name": "远山科技", "title": "产品经理"},
                "append_items": {
                    "responsibilities_json": [{"value": "负责企业协同产品规划"}],
                },
            },
        })


class AlwaysConfirmInterpreter:
    def extract(self, text: str, context: dict) -> StructuredCommand:
        return StructuredCommand(intent="confirm")


def test_ordinary_message_cannot_use_llm_confirm_to_write(tmp_path: Path) -> None:
    repository = empty_repository(tmp_path / "jobs.csv")
    app = create_app(JobCsvAgent(repository, MisclassifyingConfirmInterpreter()))

    with TestClient(app) as client:
        session_id = client.post("/sessions").json()["session_id"]
        preview = client.post(
            f"/sessions/{session_id}/messages",
            json={"content": "远山科技招聘产品经理，负责企业协同产品规划"},
        ).json()
        assert preview["can_confirm"] is True

        question = client.post(
            f"/sessions/{session_id}/messages",
            json={"content": "？", "expected_version": preview["state_version"]},
        ).json()

        assert repository.count({"company_name": "远山科技"}) == 0
        assert question["phase"] == "CONFIRMING"
        assert question["can_confirm"] is True
        assert "未执行保存" in question["message"]

        saved = client.post(
            f"/sessions/{session_id}/confirm?expected_version={question['state_version']}",
        ).json()
        assert saved["phase"] == "IDLE"

    assert repository.count({"company_name": "远山科技"}) == 1


def test_all_ambiguous_confirmation_variants_are_read_only(tmp_path: Path) -> None:
    repository = empty_repository(tmp_path / "jobs.csv")
    agent = JobCsvAgent(repository, AlwaysConfirmInterpreter())
    state = editing_state({
        "company_name": "远山科技",
        "title": "产品经理",
        "responsibilities_json": ["负责企业协同产品规划"],
    })
    state.update({"phase": "CONFIRMING", "can_confirm": True, "preview_revision": 1})

    for text in ("?", "你确定吗", "字段对吗", "再检查一下", "我还要补充", "不补充了", "好的"):
        result = agent.handle(state, text)
        assert repository.count({"company_name": "远山科技"}) == 0, text
        assert result["phase"] != "IDLE", text


def test_create_cannot_be_confirmed_with_delete_specific_wording(tmp_path: Path) -> None:
    repository = empty_repository(tmp_path / "jobs.csv")
    agent = JobCsvAgent(repository, MustNotCallInterpreter())
    state = editing_state({
        "company_name": "远山科技",
        "title": "产品经理",
        "responsibilities_json": ["负责企业协同产品规划"],
    })
    state.update({"phase": "CONFIRMING", "can_confirm": True, "preview_revision": 1})

    result = agent.handle(state, "确认删除")

    assert repository.count({"company_name": "远山科技"}) == 0
    assert result["phase"] == "CONFIRMING"
    assert "确认指令与当前操作不匹配" in result["message"]


def test_delete_cannot_be_confirmed_with_save_specific_wording(tmp_path: Path) -> None:
    repository = empty_repository(tmp_path / "jobs.csv")
    row = repository.create({
        "company_name": "远山科技",
        "title": "产品经理",
        "responsibilities_json": ["负责企业协同产品规划"],
    })
    agent = JobCsvAgent(repository, MustNotCallInterpreter())
    state = initial_state()
    state.update({
        "pending_action": "delete",
        "target_id": row["job_id"],
        "phase": "CONFIRMING",
        "can_confirm": True,
        "can_cancel": True,
        "draft_revision": 0,
        "preview_revision": 0,
    })

    result = agent.handle(state, "确认保存")

    assert repository.get(row["job_id"])["job_id"] == row["job_id"]
    assert result["phase"] == "CONFIRMING"
    assert "确认指令与当前操作不匹配" in result["message"]


class MustNotCallInterpreter:
    def extract(self, text: str, context: dict) -> StructuredCommand:
        raise AssertionError(f"确定性结束编辑语句不应调用模型：{text}")


def editing_state(draft: dict) -> dict:
    state = initial_state()
    state.update({
        "pending_action": "create",
        "phase": "CREATING",
        "draft": draft,
        "draft_revision": 1,
        "preview_revision": None,
        "can_cancel": True,
    })
    return state


def test_finish_editing_valid_draft_enters_confirmation_without_writing(tmp_path: Path) -> None:
    repository = empty_repository(tmp_path / "jobs.csv")
    agent = JobCsvAgent(repository, MustNotCallInterpreter())
    state = editing_state({
        "company_name": "乙公司",
        "title": "运营专员",
        "education_min_level": 4,
    })

    result = agent.handle(state, "不补充了")

    assert repository.count({"company_name": "乙公司"}) == 0
    assert result["phase"] == "CONFIRMING"
    assert result["can_confirm"] is True
    assert "最低学历等级：本科及以上" in result["message"]
    assert "当前草稿已完整抽取并达到保存条件" in result["message"]


def test_finish_editing_accepts_natural_sentence_punctuation(tmp_path: Path) -> None:
    repository = empty_repository(tmp_path / "jobs.csv")
    agent = JobCsvAgent(repository, MustNotCallInterpreter())
    state = editing_state({
        "company_name": "乙公司",
        "title": "运营专员",
        "education_min_level": 4,
    })

    result = agent.handle(state, "不补充了。")

    assert repository.count({"company_name": "乙公司"}) == 0
    assert result["phase"] == "CONFIRMING"
    assert result["can_confirm"] is True


def test_finish_editing_invalid_draft_names_the_blocker(tmp_path: Path) -> None:
    agent = JobCsvAgent(empty_repository(tmp_path / "jobs.csv"), MustNotCallInterpreter())
    state = editing_state({"company_name": "乙公司", "title": "运营专员"})

    result = agent.handle(state, "不补充了")

    assert result["phase"] == "CREATING"
    assert result["can_confirm"] is False
    assert result["requested_fields"] == ["responsibilities_json", "requirements_json"]
    assert "还缺少：任职要求或岗位职责" in result["message"]
    assert "当前没有等待确认的写入操作" not in result["message"]


class OptionalQuestionInterpreter:
    def extract(self, text: str, context: dict) -> StructuredCommand:
        return StructuredCommand.model_validate({
            "intent": "create",
            "task_relation": "start_new",
            "patch": {
                "set_fields": {"company_name": "乙公司", "title": "运营专员"},
            },
            "requires_clarification": True,
            "clarification_question": "请问月薪、学历、经验和城市分别是什么？",
            "clarification_fields": ["salary_min", "education_min_level", "experience_min_months", "city"],
        })


class MissingFieldsReportedAsUnresolvedInterpreter:
    def extract(self, text: str, context: dict) -> StructuredCommand:
        return StructuredCommand.model_validate({
            "intent": "create",
            "task_relation": "start_new",
            "patch": {
                "set_fields": {"company_name": "乙公司", "title": "运营专员"},
            },
            "mentioned_fields": ["company_name", "title"],
            "evidence_spans": [
                {"field": "company_name", "text": "乙公司"},
                {"field": "title", "text": "运营专员"},
            ],
            "unresolved_fragments": [{
                "text": "新增乙公司的运营专员岗位。",
                "reason": "用户仅提供了公司名称和岗位名称，缺少岗位职责、任职要求、薪资等关键信息。",
                "suggested_fields": [
                    "responsibilities_json", "requirements_json", "salary_min", "city",
                ],
            }],
            "extraction_complete": False,
            "requires_clarification": True,
            "clarification_question": "请补充岗位职责、任职要求、薪资和城市。",
            "clarification_fields": [
                "responsibilities_json", "requirements_json", "salary_min", "city",
            ],
        })


class CoveredSourceReportedAsUnresolvedInterpreter:
    def extract(self, text: str, context: dict) -> StructuredCommand:
        return StructuredCommand.model_validate({
            "intent": "create",
            "task_relation": "start_new",
            "patch": {
                "set_fields": {
                    "company_name": "远山科技",
                    "title": "产品经理",
                    "experience_min_months": 36,
                },
                "append_items": {
                    "responsibilities_json": [{"value": "负责企业协同产品规划"}],
                },
            },
            "mentioned_fields": [
                "company_name", "title", "experience_min_months", "responsibilities_json",
            ],
            "evidence_spans": [
                {"field": "company_name", "text": "远山科技"},
                {"field": "title", "text": "产品经理"},
                {"field": "responsibilities_json", "text": "负责企业协同产品规划"},
                {"field": "experience_min_months", "text": "三年以上产品经验"},
            ],
            "unresolved_fragments": [{
                "text": "远山科技招聘产品经理，负责企业协同产品规划，要求三年以上产品经验。",
                "reason": "该经验片段已完成归一化，但被模型重复标记。",
                "suggested_fields": ["experience_min_months"],
            }],
            "extraction_complete": False,
        })


class FabricatedMissingFieldFragmentInterpreter:
    def extract(self, text: str, context: dict) -> StructuredCommand:
        return StructuredCommand.model_validate({
            "intent": "update",
            "task_relation": "continue_current",
            "patch": {
                "set_fields": {
                    "salary_max": 15000,
                    "education_min_level": 4,
                    "city": "杭州",
                },
            },
            "mentioned_fields": ["salary_max", "education_min_level", "city"],
            "evidence_spans": [
                {"field": "salary_max", "text": "月薪上限15000"},
                {"field": "education_min_level", "text": "学历要求本科"},
                {"field": "city", "text": "城市在杭州"},
            ],
            "unresolved_fragments": [{
                "text": "经验年限未提供",
                "reason": "该字段被模型标记为待补充。",
                "suggested_fields": ["experience_min_months"],
            }],
            "extraction_complete": False,
            "requires_clarification": True,
            "clarification_question": "还需要补充经验年限吗？",
            "clarification_fields": ["experience_min_months"],
        })


class AppliedSubspanReportedAsUnresolvedInterpreter:
    def extract(self, text: str, context: dict) -> StructuredCommand:
        return StructuredCommand.model_validate({
            "intent": "update",
            "task_relation": "continue_current",
            "patch": {
                "set_fields": {"salary_min": 10000, "salary_period": "month"},
            },
            "mentioned_fields": ["salary_min", "salary_period"],
            "evidence_spans": [
                {"field": "salary_min", "text": "月薪最低一万"},
                {"field": "salary_period", "text": "月薪"},
            ],
            "unresolved_fragments": [{
                "text": "最低一万",
                "reason": "薪资表达仍需确认。",
                "suggested_fields": ["salary_min"],
            }],
            "extraction_complete": False,
        })


class ParaphrasedAmbiguityInterpreter:
    def extract(self, text: str, context: dict) -> StructuredCommand:
        return StructuredCommand.model_validate({
            "intent": "update",
            "task_relation": "continue_current",
            "unresolved_fragments": [{
                "text": "之前的三年经验条件",
                "reason": "用户没有明确提供该条件的要求属性。",
                "suggested_fields": [
                    "requirements_json", "preferred_requirements_json",
                ],
            }],
            "extraction_complete": False,
        })


class UnrelatedRequirementInterpreter:
    def extract(self, text: str, context: dict) -> StructuredCommand:
        return StructuredCommand.model_validate({
            "intent": "update",
            "task_relation": "continue_current",
            "patch": {
                "append_items": {
                    "requirements_json": [{"value": "沟通能力良好"}],
                },
            },
            "mentioned_fields": ["requirements_json"],
            "evidence_spans": [{
                "field": "requirements_json", "text": "沟通能力良好",
            }],
        })


class ExistingFieldClarificationInterpreter:
    def extract(self, text: str, context: dict) -> StructuredCommand:
        return StructuredCommand.model_validate({
            "intent": "update",
            "task_relation": "continue_current",
            "extraction_complete": False,
            "requires_clarification": True,
            "clarification_question": "现有三年经验要求是硬性条件还是加分项？",
            "clarification_fields": [
                "requirements_json", "preferred_requirements_json",
            ],
        })


class ExplicitExperienceAmbiguityInterpreter:
    def extract(self, text: str, context: dict) -> StructuredCommand:
        return StructuredCommand.model_validate({
            "intent": "update",
            "task_relation": "continue_current",
            "patch": {"set_fields": {"experience_min_months": 36}},
            "mentioned_fields": ["experience_min_months"],
            "evidence_spans": [{
                "field": "experience_min_months",
                "text": "要求三年以上产品经验",
            }],
            "unresolved_fragments": [{
                "text": "要求三年以上产品经验，这算硬性条件还是加分项",
                "reason": "无法确定该经验条件是硬性要求还是加分项。",
                "suggested_fields": [
                    "requirements_json", "preferred_requirements_json",
                ],
            }],
            "extraction_complete": False,
        })


def test_required_job_content_question_outranks_optional_model_question(tmp_path: Path) -> None:
    agent = JobCsvAgent(empty_repository(tmp_path / "jobs.csv"), OptionalQuestionInterpreter())

    result = agent.handle(initial_state(), "新增乙公司的运营专员岗位")

    assert result["requested_fields"] == ["responsibilities_json", "requirements_json"]
    assert "还缺少：任职要求或岗位职责" in result["message"]
    assert "月薪、学历、经验和城市" not in result["message"]


def test_absent_optional_fields_are_not_unresolved_source_text(tmp_path: Path) -> None:
    agent = JobCsvAgent(
        empty_repository(tmp_path / "jobs.csv"), MissingFieldsReportedAsUnresolvedInterpreter(),
    )

    result = agent.handle(initial_state(), "新增乙公司的运营专员岗位。")

    assert result["unresolved_fragments"] == []
    assert result["missing_fields"] == ["job_content"]
    assert result["requested_fields"] == ["responsibilities_json", "requirements_json"]
    assert "还缺少：任职要求或岗位职责" in result["message"]
    assert "尚未完整归类" not in result["message"]


def test_fully_evidenced_source_cannot_remain_unresolved(tmp_path: Path) -> None:
    agent = JobCsvAgent(
        empty_repository(tmp_path / "jobs.csv"), CoveredSourceReportedAsUnresolvedInterpreter(),
    )

    result = agent.handle(
        initial_state(),
        "远山科技招聘产品经理，负责企业协同产品规划，要求三年以上产品经验。",
    )

    assert result["unresolved_fragments"] == []
    assert result["extraction_complete"] is True
    assert result["can_confirm"] is True


def test_unstated_optional_field_cannot_be_invented_as_unresolved(tmp_path: Path) -> None:
    agent = JobCsvAgent(
        empty_repository(tmp_path / "jobs.csv"), FabricatedMissingFieldFragmentInterpreter(),
    )
    state = editing_state({
        "company_name": "乙公司",
        "title": "运营专员",
        "salary_min": 10000,
        "salary_period": "month",
    })

    result = agent.handle(state, "月薪上限15000，学历要求本科，城市在杭州。")

    assert result["unresolved_fragments"] == []
    assert result["requested_fields"] == []
    assert result["extraction_complete"] is True
    assert result["storage_valid"] is True
    assert result["can_confirm"] is True
    assert "经验年限" not in result["message"]


def test_applied_evidence_subspan_cannot_remain_unresolved(tmp_path: Path) -> None:
    agent = JobCsvAgent(
        empty_repository(tmp_path / "jobs.csv"), AppliedSubspanReportedAsUnresolvedInterpreter(),
    )
    state = editing_state({"company_name": "乙公司", "title": "运营专员"})
    state["requested_fields"] = ["responsibilities_json", "requirements_json"]

    result = agent.handle(state, "月薪最低一万。")

    assert result["draft"]["salary_min"] == 10000
    assert result["unresolved_fragments"] == []
    assert result["extraction_complete"] is True
    assert result["missing_fields"] == ["job_content"]
    assert "岗位职责" in result["message"]
    assert "还有信息尚未完整归类" not in result["message"]


def test_paraphrased_genuine_ambiguity_still_blocks_confirmation(tmp_path: Path) -> None:
    agent = JobCsvAgent(
        empty_repository(tmp_path / "jobs.csv"), ParaphrasedAmbiguityInterpreter(),
    )
    state = editing_state({
        "company_name": "远山科技",
        "title": "产品经理",
        "experience_min_months": 36,
        "responsibilities_json": ["负责企业协同产品规划"],
    })

    result = agent.handle(state, "这个经验条件的属性还不明确。")

    assert result["unresolved_fragments"][0]["text"] == "之前的三年经验条件"
    assert result["extraction_complete"] is False
    assert result["can_confirm"] is False
    assert "尚未完整归类" in result["message"]


def test_existing_field_ambiguity_without_new_patch_still_requires_clarification(
    tmp_path: Path,
) -> None:
    agent = JobCsvAgent(
        empty_repository(tmp_path / "jobs.csv"), ExistingFieldClarificationInterpreter(),
    )
    state = editing_state({
        "company_name": "远山科技",
        "title": "产品经理",
        "experience_min_months": 36,
        "responsibilities_json": ["负责企业协同产品规划"],
    })

    result = agent.handle(state, "请重新确认已有经验条件的属性。")

    assert result["extraction_complete"] is False
    assert result["can_confirm"] is False
    assert result["requested_fields"] == [
        "requirements_json", "preferred_requirements_json",
    ]
    assert "硬性条件还是加分项" in result["message"]


def test_explicit_experience_wording_does_not_erase_user_modality_question(
    tmp_path: Path,
) -> None:
    agent = JobCsvAgent(
        empty_repository(tmp_path / "jobs.csv"), ExplicitExperienceAmbiguityInterpreter(),
    )
    state = editing_state({
        "company_name": "远山科技",
        "title": "产品经理",
        "responsibilities_json": ["负责企业协同产品规划"],
    })

    result = agent.handle(state, "要求三年以上产品经验，这算硬性条件还是加分项？")

    assert result["draft"]["experience_min_months"] == 36
    assert result["extraction_complete"] is False
    assert result["can_confirm"] is False
    assert result["unresolved_fragments"]


def test_unrelated_requirement_does_not_resolve_previous_modality_ambiguity(
    tmp_path: Path,
) -> None:
    agent = JobCsvAgent(
        empty_repository(tmp_path / "jobs.csv"), UnrelatedRequirementInterpreter(),
    )
    state = editing_state({
        "company_name": "远山科技",
        "title": "产品经理",
        "experience_min_months": 36,
        "responsibilities_json": ["企业协同产品规划"],
    })
    state["unresolved_fragments"] = [{
        "text": "三年以上产品经验的要求属性",
        "reason": "尚不确定是硬性条件还是加分项",
        "suggested_fields": ["requirements_json", "preferred_requirements_json"],
        "source_message_id": "user_1",
    }]
    state["extraction_complete"] = False

    result = agent.handle(state, "再补充一条，沟通能力良好。")

    assert result["draft"]["requirements_json"] == ["沟通能力良好"]
    assert result["unresolved_fragments"][0]["text"] == "三年以上产品经验的要求属性"
    assert result["can_confirm"] is False


class StructuredEducationInterpreter:
    def extract(self, text: str, context: dict) -> StructuredCommand:
        return StructuredCommand.model_validate({
            "intent": "create",
            "task_relation": "start_new",
            "patch": {
                "set_fields": {
                    "company_name": "乙公司",
                    "title": "运营专员",
                    "education_min_level": 4,
                },
            },
            "mentioned_fields": ["education_min_level"],
            "evidence_spans": [{"field": "education_min_level", "text": "学历要求本科"}],
        })


class IrrelevantContentQuestionInterpreter:
    def extract(self, text: str, context: dict) -> StructuredCommand:
        return StructuredCommand.model_validate({
            "intent": "create",
            "task_relation": "start_new",
            "mentioned_fields": ["education_min_level"],
            "evidence_spans": [{"field": "education_min_level", "text": "学历要求本科"}],
            "patch": {
                "set_fields": {
                    "company_name": "乙公司",
                    "title": "运营专员",
                    "education_min_level": 4,
                },
            },
            "requires_clarification": True,
            "clarification_question": "还需要补充岗位职责和任职要求，请提供。",
            "clarification_fields": ["responsibilities_json", "requirements_json"],
            "extraction_complete": False,
        })


def test_structured_education_counts_as_job_requirement_content(tmp_path: Path) -> None:
    agent = JobCsvAgent(empty_repository(tmp_path / "jobs.csv"), StructuredEducationInterpreter())

    result = agent.handle(
        initial_state(),
        "新增乙公司运营专员，月薪一万到一万五，学历要求本科，城市杭州",
    )

    assert result["storage_valid"] is True
    assert result["can_confirm"] is True
    assert result["missing_fields"] == []


def test_irrelevant_model_question_cannot_block_structured_requirement(
    tmp_path: Path,
) -> None:
    agent = JobCsvAgent(
        empty_repository(tmp_path / "jobs.csv"), IrrelevantContentQuestionInterpreter(),
    )

    result = agent.handle(initial_state(), "新增乙公司运营专员，学历要求本科")

    assert result["storage_valid"] is True
    assert result["extraction_complete"] is True
    assert result["can_confirm"] is True
    assert "还需要补充岗位职责和任职要求" not in result["message"]


def test_salary_and_city_alone_do_not_count_as_job_requirement_content(tmp_path: Path) -> None:
    state = editing_state({
        "company_name": "乙公司",
        "title": "运营专员",
        "salary_min": 10000,
        "salary_max": 15000,
        "city": "杭州",
    })
    agent = JobCsvAgent(empty_repository(tmp_path / "jobs.csv"), MustNotCallInterpreter())

    result = agent.handle(state, "不补充了")

    assert result["storage_valid"] is False
    assert result["missing_fields"] == ["job_content"]
    assert "任职要求或岗位职责" in result["message"]


def test_explicit_no_experience_requirement_counts_as_structured_content(tmp_path: Path) -> None:
    state = editing_state({
        "company_name": "乙公司",
        "title": "运营专员",
        "experience_min_months": 0,
    })
    agent = JobCsvAgent(empty_repository(tmp_path / "jobs.csv"), MustNotCallInterpreter())

    result = agent.handle(state, "不补充了")

    assert result["storage_valid"] is True
    assert result["can_confirm"] is True

    saved = agent.handle(result, "确认保存")
    assert saved["phase"] == "IDLE"
    assert agent.repository.count({"company_name": "乙公司"}) == 1


def test_correction_meta_language_is_not_extracted_as_requirement() -> None:
    mentions = infer_atomic_mentions("我刚才说了3年以上产品经验只是加分项")

    preferred = [
        item for mention in mentions
        if mention.field == "requirements_json" and mention.modality == "preferred"
        for item in mention.items
    ]
    scalar_fields = {mention.field for mention in mentions if not mention.items}

    assert preferred == ["3年以上产品经验"]
    assert "experience_min_months" not in scalar_fields


def test_common_correction_prefixes_are_removed_from_requirement_fact() -> None:
    for text in (
        "我刚才说的是3年以上产品经验只是加分项",
        "刚才我说的3年以上产品经验只是加分项",
        "我之前提到的是3年以上产品经验只是加分项",
        "我刚才改成3年以上产品经验只是加分项",
    ):
        preferred = [
            item for mention in infer_atomic_mentions(text)
            if mention.field == "requirements_json" and mention.modality == "preferred"
            for item in mention.items
        ]
        assert preferred == ["3年以上产品经验"], text


def test_preferred_experience_patch_atomically_clears_hard_requirement() -> None:
    model_patch = DraftPatch.model_validate({
        "set_fields": {"experience_min_months": 36, "experience_max_months": 36},
        "append_items": {
            "requirements_json": [{"value": "（加分项）三年以上产品经验"}],
            "preferred_requirements_json": [{"value": "我刚才说了3年以上产品经验"}],
        },
    })
    mention = AtomicMention.model_validate({
        "field": "requirements_json",
        "raw_text": "三年以上产品经验",
        "items": ["三年以上产品经验"],
        "operation": "append",
        "modality": "preferred",
        "source": "contextual",
    })

    patch = normalize_patch(
        model_patch,
        "我刚才说了3年以上产品经验只是加分项",
        [mention],
    )
    draft, _ = apply_draft_patch(
        {
            "company_name": "远山科技",
            "title": "产品经理",
            "experience_min_months": 36,
            "experience_max_months": 36,
            "requirements_json": ["三年以上产品经验"],
            "responsibilities_json": ["负责企业协同产品规划"],
        },
        {},
        patch,
    )

    assert "experience_min_months" not in draft
    assert "experience_max_months" not in draft
    assert draft.get("requirements_json", []) == []
    assert draft["preferred_requirements_json"] == ["3年以上产品经验"]
    assert "我刚才说了" not in str(draft)


def test_unbounded_experience_source_discards_model_generated_upper_bound() -> None:
    patch = normalize_patch(
        DraftPatch.model_validate({
            "set_fields": {
                "experience_min_months": 36,
                "experience_max_months": 36,
            },
        }),
        "要求三年以上产品经验",
    )

    assert patch.set_fields["experience_min_months"] == 36
    assert "experience_max_months" not in patch.set_fields
    assert "experience_max_months" in patch.clear_fields


def test_non_experience_duration_does_not_clear_experience_upper_bound() -> None:
    patch = normalize_patch(
        DraftPatch(set_fields={"city": "北京"}),
        "工作地点改为北京，劳动合同年限至少3年",
        current_draft={
            "experience_min_months": 36,
            "experience_max_months": 60,
        },
    )

    assert "experience_min_months" not in patch.set_fields
    assert "experience_max_months" not in patch.clear_fields


def test_unrelated_hard_requirement_does_not_block_experience_reclassification() -> None:
    mention = AtomicMention.model_validate({
        "field": "requirements_json",
        "raw_text": "三年以上产品经验只是加分项",
        "items": ["三年以上产品经验"],
        "operation": "append",
        "modality": "preferred",
        "source": "contextual",
    })

    patch = normalize_patch(
        DraftPatch(),
        "我刚才说的三年以上产品经验只是加分项",
        [mention],
        current_draft={
            "experience_min_months": 36,
            "requirements_json": ["沟通能力良好"],
        },
    )

    assert "experience_min_months" in patch.clear_fields


def test_patch_only_reclassification_ignores_unrelated_hard_requirement() -> None:
    draft, _ = apply_draft_patch(
        {
            "experience_min_months": 36,
            "requirements_json": ["沟通能力良好"],
        },
        {},
        DraftPatch.model_validate({
            "append_items": {
                "preferred_requirements_json": [{"value": "3年以上产品经验"}],
            },
        }),
    )

    assert "experience_min_months" not in draft
    assert draft["requirements_json"] == ["沟通能力良好"]
    assert draft["preferred_requirements_json"] == ["3年以上产品经验"]


def test_latest_preferred_correction_wins_over_older_hard_source() -> None:
    mention = AtomicMention.model_validate({
        "field": "requirements_json",
        "raw_text": "三年以上产品经验只是加分项",
        "items": ["三年以上产品经验"],
        "operation": "append",
        "modality": "preferred",
        "source": "contextual",
    })

    patch = normalize_patch(
        DraftPatch(),
        "要求三年以上产品经验\n我刚才说了三年以上产品经验只是加分项",
        [mention],
    )

    assert "experience_min_months" not in patch.set_fields
    assert "experience_min_months" in patch.clear_fields


def test_scalar_remove_mention_is_a_valid_clear_operation() -> None:
    mention = AtomicMention.model_validate({
        "field": "experience_min_months",
        "raw_text": "三年以上产品经验只是加分项",
        "operation": "remove",
        "source": "contextual",
    })

    patch = normalize_patch(
        DraftPatch(set_fields={"experience_min_months": 36}),
        "优选条件",
        [mention],
    )

    assert patch.clear_fields == ["experience_min_months"]
    assert "experience_min_months" not in patch.set_fields


def test_modality_migration_does_not_leave_empty_list_fields() -> None:
    patch = DraftPatch.model_validate({
        "append_items": {"requirements_json": [{"value": "沟通能力良好"}]},
    })

    draft, _ = apply_draft_patch({}, {}, patch)

    assert draft == {"requirements_json": ["沟通能力良好"]}


def test_patch_layer_clears_hard_experience_for_direct_preferred_assignment() -> None:
    patch = DraftPatch.model_validate({
        "set_fields": {"experience_min_months": 36},
        "append_items": {
            "preferred_requirements_json": [{"value": "3年以上产品经验"}],
        },
    })

    draft, _ = apply_draft_patch(
        {
            "company_name": "远山科技",
            "title": "产品经理",
            "experience_min_months": 36,
            "requirements_json": ["三年以上产品经验"],
        },
        {},
        patch,
    )

    assert "experience_min_months" not in draft
    assert draft.get("requirements_json", []) == []
    assert draft["preferred_requirements_json"] == ["3年以上产品经验"]


def test_different_experience_ranges_are_not_modality_conflicts() -> None:
    conflicts = requirement_modality_conflicts({
        "requirements_json": ["3年以上产品经验"],
        "preferred_requirements_json": ["5年以上产品经验"],
    })

    assert conflicts == []


def test_equivalent_experience_phrasings_are_modality_conflicts() -> None:
    conflicts = requirement_modality_conflicts({
        "requirements_json": ["3年以上产品经验"],
        "preferred_requirements_json": ["至少3年产品经验"],
    })

    assert conflicts == ["至少3年产品经验"]

    draft, _ = apply_draft_patch(
        {
            "experience_min_months": 36,
            "requirements_json": ["3年以上产品经验"],
        },
        {},
        DraftPatch.model_validate({
            "append_items": {
                "preferred_requirements_json": [{"value": "至少3年产品经验"}],
            },
        }),
    )
    assert "experience_min_months" not in draft
    assert draft.get("requirements_json", []) == []


def test_unrelated_preferred_experience_preserves_hard_experience_scalar() -> None:
    patch = DraftPatch.model_validate({
        "append_items": {
            "preferred_requirements_json": [{"value": "5年以上SaaS产品经验"}],
        },
    })

    draft, _ = apply_draft_patch(
        {
            "company_name": "远山科技",
            "title": "产品经理",
            "experience_min_months": 36,
            "requirements_json": ["3年以上产品经验"],
        },
        {},
        patch,
    )

    assert draft["experience_min_months"] == 36
    assert draft["requirements_json"] == ["3年以上产品经验"]
    assert draft["preferred_requirements_json"] == ["5年以上SaaS产品经验"]


def test_normalization_preserves_unrelated_hard_and_preferred_experience() -> None:
    mention = AtomicMention.model_validate({
        "field": "requirements_json",
        "raw_text": "5年以上SaaS产品经验为加分项",
        "items": ["5年以上SaaS产品经验"],
        "operation": "append",
        "modality": "preferred",
        "source": "explicit",
    })

    patch = normalize_patch(
        DraftPatch(set_fields={"experience_min_months": 36}),
        "要求3年以上产品经验，5年以上SaaS产品经验为加分项",
        [mention],
    )

    assert patch.set_fields["experience_min_months"] == 36
    assert "experience_min_months" not in patch.clear_fields


def test_contextual_unrelated_preferred_experience_does_not_clear_current_scalar() -> None:
    mention = AtomicMention.model_validate({
        "field": "requirements_json",
        "raw_text": "5年以上SaaS产品经验",
        "items": ["5年以上SaaS产品经验"],
        "operation": "append",
        "modality": "preferred",
        "source": "contextual",
    })

    patch = normalize_patch(
        DraftPatch(),
        "优选条件",
        [mention],
        current_draft={"experience_min_months": 36},
    )

    assert "experience_min_months" not in patch.clear_fields


class ModalityCorrectionInterpreter:
    def extract(self, text: str, context: dict) -> StructuredCommand:
        return StructuredCommand.model_validate({
            "intent": "update",
            "task_relation": "continue_current",
            "mentioned_fields": ["requirements_json"],
            "evidence_spans": [{"field": "requirements_json", "text": "3年以上产品经验"}],
            "mentions": [{
                "field": "requirements_json",
                "raw_text": "3年以上产品经验",
                "items": ["3年以上产品经验"],
                "operation": "append",
                "modality": "preferred",
                "source": "contextual",
            }],
            "patch": {
                "set_fields": {"experience_min_months": 36},
                "append_items": {
                    "requirements_json": [{"value": "3年以上产品经验"}],
                    "preferred_requirements_json": [{"value": "我刚才说了3年以上产品经验"}],
                },
            },
        })


class FailingCorrectionInterpreter:
    def extract(self, text: str, context: dict) -> StructuredCommand:
        raise LLMServiceError("模拟模型结构化失败")


def test_agent_moves_corrected_requirement_to_preferred_only(tmp_path: Path) -> None:
    agent = JobCsvAgent(empty_repository(tmp_path / "jobs.csv"), ModalityCorrectionInterpreter())
    state = editing_state({
        "company_name": "远山科技",
        "title": "产品经理",
        "experience_min_months": 36,
        "requirements_json": ["3年以上产品经验"],
        "responsibilities_json": ["负责企业协同产品规划"],
    })

    result = agent.handle(state, "我刚才说了3年以上产品经验只是加分项")

    assert "experience_min_months" not in result["draft"]
    assert result["draft"].get("requirements_json", []) == []
    assert result["draft"]["preferred_requirements_json"] == ["3年以上产品经验"]
    assert result["can_confirm"] is True


def test_deterministic_correction_survives_llm_failure(tmp_path: Path) -> None:
    agent = JobCsvAgent(empty_repository(tmp_path / "jobs.csv"), FailingCorrectionInterpreter())
    state = editing_state({
        "company_name": "远山科技",
        "title": "产品经理",
        "experience_min_months": 36,
        "requirements_json": ["三年以上产品经验"],
        "responsibilities_json": ["负责企业协同产品规划"],
    })

    result = agent.handle(state, "我刚才说了三年以上产品经验只是加分项。")

    assert "experience_min_months" not in result["draft"]
    assert result["draft"].get("requirements_json", []) == []
    assert result["draft"]["preferred_requirements_json"] == ["三年以上产品经验"]


def test_short_modality_answer_moves_previous_condition_atomically(tmp_path: Path) -> None:
    agent = JobCsvAgent(empty_repository(tmp_path / "jobs.csv"), ModalityCorrectionInterpreter())
    state = editing_state({
        "company_name": "远山科技",
        "title": "产品经理",
        "experience_min_months": 36,
        "requirements_json": ["3年以上产品经验"],
        "responsibilities_json": ["负责企业协同产品规划"],
    })
    state["last_assistant_question"] = "您提到的‘3年以上产品经验’，是硬性要求还是优选条件？"
    state["source_messages"] = [{
        "id": "user_1",
        "content": "远山科技招聘产品经理，负责企业协同产品规划，要求3年以上产品经验",
    }]

    result = agent.handle(state, "优选条件")

    assert "experience_min_months" not in result["draft"]
    assert result["draft"].get("requirements_json", []) == []
    assert result["draft"]["preferred_requirements_json"] == ["3年以上产品经验"]


class RepairPrecedenceInterpreter:
    called = False

    def extract(self, text: str, context: dict) -> StructuredCommand:
        self.called = True
        assert context["repair"]["source_message_ids"] == ["user_1"]
        return StructuredCommand.model_validate({
            "intent": "update",
            "task_relation": "continue_current",
            "mentioned_fields": ["city"],
            "evidence_spans": [{"field": "city", "text": "北京", "source_message_id": "user_1"}],
            "patch": {"set_fields": {"city": "北京"}},
        })


def test_recheck_request_uses_repair_flow_before_show_current_shortcut(tmp_path: Path) -> None:
    interpreter = RepairPrecedenceInterpreter()
    agent = JobCsvAgent(empty_repository(tmp_path / "jobs.csv"), interpreter)
    state = editing_state({
        "company_name": "远山科技",
        "title": "产品经理",
        "responsibilities_json": ["负责企业协同产品规划"],
    })
    state["source_messages"] = [{
        "id": "user_1",
        "content": "远山科技招聘产品经理，负责企业协同产品规划，工作地点北京",
    }]

    result = agent.handle(state, "你现在的字段对吗？重新检查一下")

    assert interpreter.called is True
    assert result["draft"]["city"] == "北京"


def test_inaccurate_identification_feedback_enters_repair_flow(tmp_path: Path) -> None:
    interpreter = RepairPrecedenceInterpreter()
    agent = JobCsvAgent(empty_repository(tmp_path / "jobs.csv"), interpreter)
    state = editing_state({
        "company_name": "远山科技",
        "title": "产品经理",
        "responsibilities_json": ["负责企业协同产品规划"],
    })
    state["source_messages"] = [{
        "id": "user_1",
        "content": "远山科技招聘产品经理，负责企业协同产品规划，工作地点北京",
    }]

    result = agent.handle(state, "你识别得不准确呀")

    assert interpreter.called is True
    assert result["draft"]["city"] == "北京"


def test_conflicting_requirement_modalities_block_confirmation(tmp_path: Path) -> None:
    repository = empty_repository(tmp_path / "jobs.csv")
    agent = JobCsvAgent(repository, MustNotCallInterpreter())
    state = editing_state({
        "company_name": "远山科技",
        "title": "产品经理",
        "requirements_json": ["3年以上产品经验"],
        "preferred_requirements_json": ["3年以上产品经验"],
    })
    state.update({"phase": "CONFIRMING", "can_confirm": True, "preview_revision": 1})

    result = agent.handle(state, "确认写入", {"intent": "confirm"})

    assert repository.count({"company_name": "远山科技"}) == 0
    assert result["can_confirm"] is False
    assert "要求属性冲突" in result["message"]
    assert "3年以上产品经验" in result["message"]


def test_preferred_experience_conflicting_with_hard_scalar_blocks_confirmation(
    tmp_path: Path,
) -> None:
    repository = empty_repository(tmp_path / "jobs.csv")
    agent = JobCsvAgent(repository, MustNotCallInterpreter())
    state = editing_state({
        "company_name": "远山科技",
        "title": "产品经理",
        "experience_min_months": 36,
        "preferred_requirements_json": ["3年以上产品经验"],
    })
    state.update({"phase": "CONFIRMING", "can_confirm": True, "preview_revision": 1})

    result = agent.handle(state, "确认写入", {"intent": "confirm"})

    assert repository.count({"company_name": "远山科技"}) == 0
    assert result["can_confirm"] is False
    assert "要求属性冲突" in result["message"]


def test_same_duration_different_experience_subjects_are_not_conflicts() -> None:
    conflicts = requirement_modality_conflicts({
        "experience_min_months": 36,
        "requirements_json": ["3年以上产品经验"],
        "preferred_requirements_json": ["3年以上SaaS经验"],
    })

    assert conflicts == []


def test_show_current_explains_why_confirmation_is_unavailable(tmp_path: Path) -> None:
    agent = JobCsvAgent(empty_repository(tmp_path / "jobs.csv"), MustNotCallInterpreter())
    state = editing_state({"company_name": "乙公司", "title": "运营专员"})

    result = agent.handle(state, "当前草稿是什么")

    assert result["can_confirm"] is False
    assert "当前不能保存" in result["message"]
    assert "任职要求或岗位职责" in result["message"]


class DuplicateSummaryInterpreter:
    def extract(self, text: str, context: dict) -> StructuredCommand:
        return StructuredCommand.model_validate({
            "intent": "create",
            "task_relation": "start_new",
            "patch": {
                "set_fields": {"company_name": "远山科技", "title": "产品经理"},
                "append_items": {
                    "responsibilities_json": [
                        {"value": "企业协同产品规划"},
                        {"value": "企业协同产品规划"},
                    ],
                },
            },
        })


def test_update_summary_counts_final_deduplicated_diff(tmp_path: Path) -> None:
    agent = JobCsvAgent(empty_repository(tmp_path / "jobs.csv"), DuplicateSummaryInterpreter())

    result = agent.handle(initial_state(), "远山科技招聘产品经理，负责企业协同产品规划")

    assert result["draft"]["responsibilities_json"] == ["企业协同产品规划"]
    assert "岗位职责新增1条" in result["message"]
    assert "岗位职责新增2条" not in result["message"]
