from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


BUSINESS_COLUMNS = (
    "job_id", "title", "company_name", "department", "job_category", "headcount",
    "city", "work_address", "work_locations_json",
    "salary_min", "salary_max", "salary_currency", "salary_period",
    "recruitment", "employment", "work_mode", "education_min_level",
    "experience_min_months", "experience_max_months", "requirements_json",
    "responsibilities_json", "skills_json", "certificates_json", "benefits_json",
    "preferred_requirements_json", "not_required_requirements_json",
    "student_status_json", "major_requirements_json", "graduation_years_json",
    "recruitment_batch", "application_deadline", "internship_min_months",
    "onsite_days_per_week",
    "source_url", "scraped_at", "content_hash", "extraction_mode",
)

JSON_FIELDS = frozenset({
    "requirements_json", "responsibilities_json", "skills_json",
    "certificates_json", "benefits_json", "preferred_requirements_json",
    "not_required_requirements_json", "student_status_json", "major_requirements_json",
    "graduation_years_json", "work_locations_json",
})
SYSTEM_FIELDS = frozenset({"job_id", "scraped_at", "content_hash", "extraction_mode"})
EDITABLE_FIELDS = tuple(column for column in BUSINESS_COLUMNS if column not in SYSTEM_FIELDS)
SCALAR_FIELDS = frozenset(set(EDITABLE_FIELDS) - JSON_FIELDS)
REQUIRED_CREATE_FIELDS = ("company_name", "title")
CREATE_REQUIREMENT_FIELDS = (
    "requirements_json", "preferred_requirements_json", "not_required_requirements_json",
    "skills_json", "certificates_json", "education_min_level", "experience_min_months",
    "experience_max_months", "student_status_json", "major_requirements_json",
    "graduation_years_json", "internship_min_months",
    "onsite_days_per_week",
)
CREATE_CONTENT_FIELDS = ("responsibilities_json", *CREATE_REQUIREMENT_FIELDS)


def has_meaningful_value(value: Any) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def has_create_content(fields: dict[str, Any]) -> bool:
    return any(has_meaningful_value(fields.get(field)) for field in CREATE_CONTENT_FIELDS)


FIELD_LABELS = {
    "job_id": "岗位编号", "title": "岗位名称", "company_name": "公司名称",
    "department": "部门", "job_category": "岗位类别", "headcount": "招聘人数",
    "city": "城市", "work_address": "工作地址", "salary_min": "最低薪资",
    "work_locations_json": "工作地点列表",
    "salary_max": "最高薪资", "salary_currency": "薪资币种",
    "salary_period": "薪资周期", "recruitment": "招聘类型",
    "employment": "用工类型", "work_mode": "办公模式",
    "education_min_level": "最低学历等级", "experience_min_months": "最低经验（月）",
    "experience_max_months": "最高经验（月）", "requirements_json": "任职要求",
    "responsibilities_json": "岗位职责", "skills_json": "技能要求",
    "certificates_json": "证书", "benefits_json": "福利", "source_url": "来源链接",
    "preferred_requirements_json": "加分项", "not_required_requirements_json": "非必需条件",
    "student_status_json": "学籍状态", "major_requirements_json": "专业要求",
    "graduation_years_json": "毕业届别", "recruitment_batch": "招聘批次",
    "application_deadline": "投递截止时间",
    "internship_min_months": "最短实习期（月）",
    "onsite_days_per_week": "每周到岗天数",
    "job_content": "任职要求或岗位职责",
}


class Phase(StrEnum):
    IDLE = "IDLE"
    CREATING = "CREATING"
    EDITING = "EDITING"
    CONFIRMING = "CONFIRMING"


class QueryPlan(BaseModel):
    """A small read-only query language. It deliberately has no SQL escape hatch."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["count", "list", "detail"]
    company_name: str | None = None
    title: str | None = None
    city: str | None = None
    recruitment: Literal["internship", "campus", "experienced", "mixed", "unknown"] | None = None
    limit: int = Field(default=10, ge=1, le=10)

    def filters(self) -> dict[str, str]:
        return {
            field: str(value)
            for field in ("company_name", "title", "city", "recruitment")
            if (value := getattr(self, field)) is not None
        }


class JobFields(BaseModel):
    """All user-editable CSV fields; system-managed columns are deliberately absent."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    company_name: str | None = None
    department: str | None = None
    job_category: str | None = None
    headcount: int | None = Field(default=None, ge=1)
    city: str | None = None
    work_address: str | None = None
    work_locations_json: list[str] | None = None
    salary_min: float | None = Field(default=None, ge=0)
    salary_max: float | None = Field(default=None, ge=0)
    salary_currency: str | None = None
    salary_period: Literal["hour", "day", "month", "year", "per_order"] | None = None
    recruitment: Literal["internship", "campus", "experienced", "mixed", "unknown"] | None = None
    employment: Literal[
        "full_time", "part_time", "full_or_part_time", "internship", "contract",
        "temporary", "unknown",
    ] | None = None
    work_mode: Literal["onsite", "remote", "hybrid", "unknown"] | None = None
    education_min_level: int | None = Field(default=None, ge=0, le=6)
    experience_min_months: int | None = Field(default=None, ge=0)
    experience_max_months: int | None = Field(default=None, ge=0)
    internship_min_months: int | None = Field(default=None, ge=0)
    onsite_days_per_week: int | None = Field(default=None, ge=0, le=7)
    recruitment_batch: str | None = None
    application_deadline: str | None = None
    requirements_json: list[str] | None = None
    responsibilities_json: list[str] | None = None
    skills_json: list[str] | None = None
    certificates_json: list[str] | None = None
    benefits_json: list[str] | None = None
    preferred_requirements_json: list[str] | None = None
    not_required_requirements_json: list[str] | None = None
    student_status_json: list[str] | None = None
    major_requirements_json: list[str] | None = None
    graduation_years_json: list[str] | None = None
    source_url: str | None = None

    @field_validator("title", "company_name")
    @classmethod
    def required_text_cannot_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be blank")
        return value.strip() if value is not None else None

    @field_validator("title")
    @classmethod
    def title_must_be_complete(cls, value: str | None) -> str | None:
        if value is not None and len(value) < 2:
            raise ValueError("title must contain at least two characters")
        return value

    @field_validator(*JSON_FIELDS)
    @classmethod
    def clean_lists(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        result: list[str] = []
        for item in value:
            cleaned = " ".join(str(item).split()).strip()
            if cleaned and cleaned not in result:
                result.append(cleaned)
        return result

    @model_validator(mode="after")
    def validate_ranges(self) -> "JobFields":
        if (
            self.salary_min is not None and self.salary_max is not None
            and self.salary_max < self.salary_min
        ):
            raise ValueError("salary_max must be greater than or equal to salary_min")
        if (
            self.experience_min_months is not None and self.experience_max_months is not None
            and self.experience_max_months < self.experience_min_months
        ):
            raise ValueError("experience_max_months must be greater than or equal to experience_min_months")
        return self


ProvenanceSource = Literal["explicit", "contextual", "normalized"]
ModelPatchSource = Literal["explicit", "contextual", "normalized"]


class PatchItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str = Field(min_length=1)
    source: ModelPatchSource = "explicit"

    @field_validator("value")
    @classmethod
    def clean_value(cls, value: str) -> str:
        return " ".join(value.split()).strip(" ，,；;。.!！")


class ListReplacement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: Literal[
        "requirements_json", "responsibilities_json", "skills_json",
        "certificates_json", "benefits_json", "preferred_requirements_json",
        "not_required_requirements_json", "student_status_json",
        "major_requirements_json", "graduation_years_json", "work_locations_json",
    ]
    match: str = Field(min_length=1)
    value: str = Field(min_length=1)
    source: ModelPatchSource = "explicit"


class EvidenceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_message_id: str | None = None


RequirementModality = Literal["required", "preferred", "not_required"]


class AtomicMention(BaseModel):
    """One source-grounded fact before it is converted into an incremental patch."""

    model_config = ConfigDict(extra="forbid")

    field: str = Field(min_length=1)
    raw_text: str = Field(min_length=1)
    value: Any | None = None
    items: list[str] = Field(default_factory=list)
    operation: Literal["set", "append", "remove"] = "set"
    modality: RequirementModality = "required"
    source: ModelPatchSource = "explicit"
    source_message_id: str | None = None

    @model_validator(mode="after")
    def validate_mention(self) -> "AtomicMention":
        if self.field not in EDITABLE_FIELDS:
            raise ValueError(f"atomic mention contains unknown field: {self.field}")
        scalar_clear = self.operation == "remove" and self.field not in JSON_FIELDS
        if self.value is None and not self.items and not scalar_clear:
            raise ValueError("atomic mention requires value or items")
        if self.field not in JSON_FIELDS and self.items:
            raise ValueError("scalar atomic mention cannot contain items")
        return self


class UnresolvedFragment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    suggested_fields: list[str] = Field(default_factory=list)
    source_message_id: str | None = None


class IgnoredFragment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class DraftPatch(BaseModel):
    """Incremental edit protocol. List fields are never replaced wholesale."""

    model_config = ConfigDict(extra="forbid")

    set_fields: dict[str, Any] = Field(default_factory=dict)
    set_sources: dict[str, ModelPatchSource] = Field(default_factory=dict)
    append_items: dict[str, list[PatchItem]] = Field(default_factory=dict)
    replace_items: list[ListReplacement] = Field(default_factory=list)
    remove_items: dict[str, list[str]] = Field(default_factory=dict)
    clear_fields: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_patch_fields(self) -> "DraftPatch":
        invalid_set = sorted(set(self.set_fields) - SCALAR_FIELDS)
        invalid_sources = sorted(set(self.set_sources) - set(self.set_fields))
        invalid_lists = sorted((set(self.append_items) | set(self.remove_items)) - JSON_FIELDS)
        invalid_clear = sorted(set(self.clear_fields) - set(EDITABLE_FIELDS))
        if invalid_set:
            raise ValueError(f"set_fields contains non-scalar or unknown fields: {', '.join(invalid_set)}")
        if invalid_sources:
            raise ValueError(f"set_sources has no matching value: {', '.join(invalid_sources)}")
        if invalid_lists:
            raise ValueError(f"invalid list fields: {', '.join(invalid_lists)}")
        if invalid_clear:
            raise ValueError(f"non-editable fields: {', '.join(invalid_clear)}")
        self.clear_fields = list(dict.fromkeys(self.clear_fields))
        return self


class PendingDecision(BaseModel):
    """One machine-actionable yes/no decision; never a second full draft."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["accept_patch"] = "accept_patch"
    prompt: str = Field(min_length=1)
    patch: DraftPatch
    reason: str = "该信息存在真实歧义，需要用户确认"


class StructuredCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Literal[
        "create", "update", "delete", "search", "count", "detail", "confirm", "cancel", "unknown",
        "help", "conversation", "unsupported",
    ]
    search_query: str | None = None
    query_plan: QueryPlan | None = None
    selection_index: int | None = Field(default=None, ge=1)
    patch: DraftPatch = Field(default_factory=DraftPatch)
    mentions: list[AtomicMention] = Field(default_factory=list)
    mentioned_fields: list[str] = Field(default_factory=list)
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    ignored_fragments: list[IgnoredFragment] = Field(default_factory=list)
    unresolved_fragments: list[UnresolvedFragment] = Field(default_factory=list)
    extraction_complete: bool = True
    pending_decision: PendingDecision | None = None
    clarification_question: str | None = None
    requires_clarification: bool = False
    natural_reply: str | None = None
    task_relation: Literal["continue_current", "start_new", "not_applicable"] = "not_applicable"
    clarification_fields: list[str] = Field(default_factory=list)

    @field_validator("mentioned_fields")
    @classmethod
    def validate_mentioned_fields(cls, value: list[str]) -> list[str]:
        invalid = sorted(set(value) - set(EDITABLE_FIELDS))
        if invalid:
            raise ValueError(f"mentioned_fields contains unknown fields: {', '.join(invalid)}")
        return list(dict.fromkeys(value))

    @field_validator("clarification_fields")
    @classmethod
    def validate_clarification_fields(cls, value: list[str]) -> list[str]:
        invalid = sorted(set(value) - set(EDITABLE_FIELDS))
        if invalid:
            raise ValueError(f"clarification_fields contains unknown fields: {', '.join(invalid)}")
        return list(dict.fromkeys(value))


class LLMInterpretation(StructuredCommand):
    """Strict typed result accepted from the language model."""

    @model_validator(mode="after")
    def validate_model_protocol(self) -> "LLMInterpretation":
        if self.intent in {"search", "count", "detail"} and self.query_plan is None:
            raise ValueError("query intents require a QueryPlan")
        touched = (
            set(self.patch.set_fields) | set(self.patch.append_items)
            | set(self.patch.remove_items) | set(self.patch.clear_fields)
            | {item.field for item in self.patch.replace_items}
        )
        touched |= {item.field for item in self.mentions}
        if self.pending_decision is not None:
            pending = self.pending_decision.patch
            touched |= (
                set(pending.set_fields) | set(pending.append_items)
                | set(pending.remove_items) | set(pending.clear_fields)
                | {item.field for item in pending.replace_items}
            )
        undeclared = sorted(touched - set(self.mentioned_fields))
        if undeclared:
            raise ValueError(f"patch fields missing from mentioned_fields: {', '.join(undeclared)}")
        evidenced = {span.field for span in self.evidence_spans}
        missing_evidence = sorted(set(self.mentioned_fields) - evidenced)
        if missing_evidence:
            raise ValueError(f"mentioned_fields missing evidence: {', '.join(missing_evidence)}")
        if self.unresolved_fragments and self.extraction_complete:
            raise ValueError("extraction_complete cannot be true with unresolved_fragments")
        return self


class ChatRequest(BaseModel):
    content: str = Field(min_length=1, max_length=10_000)
    expected_version: int | None = Field(default=None, ge=0)


class ChatResponse(BaseModel):
    session_id: str
    phase: Phase
    message: str
    missing_fields: list[str] = Field(default_factory=list)
    candidates: list[dict[str, str]] = Field(default_factory=list)
    can_confirm: bool = False
    can_cancel: bool = False
    storage_valid: bool = False
    extraction_complete: bool = True
    unresolved_fragments: list[str] = Field(default_factory=list)
    requested_fields: list[str] = Field(default_factory=list)
    state_version: int = Field(default=0, ge=0)
    job_id: str | None = None
    job_profile_version: int | None = Field(default=None, ge=1)
    lifecycle_status: str | None = None
    active_question: dict[str, Any] | None = None
    workflow_trace: list[str] = Field(default_factory=list)
    conversation_phase: str = "IDLE"
