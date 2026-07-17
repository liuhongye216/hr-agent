from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


BUSINESS_COLUMNS = (
    "job_id", "title", "company_name", "city", "work_address",
    "salary_min", "salary_max", "salary_currency", "salary_period",
    "recruitment", "employment", "work_mode", "education_min_level",
    "experience_min_months", "experience_max_months", "requirements_json",
    "responsibilities_json", "skills_json", "certificates_json", "benefits_json",
    "source_url", "scraped_at", "content_hash", "extraction_mode",
)

JSON_FIELDS = frozenset({
    "requirements_json", "responsibilities_json", "skills_json",
    "certificates_json", "benefits_json",
})
SYSTEM_FIELDS = frozenset({"job_id", "scraped_at", "content_hash", "extraction_mode"})
EDITABLE_FIELDS = tuple(column for column in BUSINESS_COLUMNS if column not in SYSTEM_FIELDS)
REQUIRED_CREATE_FIELDS = ("company_name", "title")
CREATE_CONTENT_FIELDS = ("requirements_json", "responsibilities_json")

FIELD_LABELS = {
    "job_id": "岗位编号", "title": "岗位名称", "company_name": "公司名称",
    "city": "城市", "work_address": "工作地址", "salary_min": "最低薪资",
    "salary_max": "最高薪资", "salary_currency": "薪资币种",
    "salary_period": "薪资周期", "recruitment": "招聘类型",
    "employment": "用工类型", "work_mode": "办公模式",
    "education_min_level": "最低学历等级", "experience_min_months": "最低经验（月）",
    "experience_max_months": "最高经验（月）", "requirements_json": "任职要求",
    "responsibilities_json": "岗位职责", "skills_json": "技能要求",
    "certificates_json": "证书", "benefits_json": "福利", "source_url": "来源链接",
    "job_content": "任职要求、岗位职责或完整 JD 文本",
}


class Phase(StrEnum):
    IDLE = "IDLE"
    CREATING = "CREATING"
    EDITING = "EDITING"
    CONFIRMING = "CONFIRMING"


class JobFields(BaseModel):
    """LLM-editable fields. System fields are deliberately absent."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    company_name: str | None = None
    city: str | None = None
    work_address: str | None = None
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
    requirements_json: list[str] | None = None
    responsibilities_json: list[str] | None = None
    skills_json: list[str] | None = None
    certificates_json: list[str] | None = None
    benefits_json: list[str] | None = None
    source_url: str | None = None

    @field_validator("title", "company_name")
    @classmethod
    def required_text_cannot_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be blank")
        return value.strip() if value is not None else None

    @field_validator(*JSON_FIELDS)
    @classmethod
    def clean_lists(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        result: list[str] = []
        for item in value:
            cleaned = " ".join(item.split())
            if cleaned and cleaned not in result:
                result.append(cleaned)
        return result

    @field_validator("experience_max_months")
    @classmethod
    def validate_experience_range(cls, value: int | None, info: Any) -> int | None:
        minimum = info.data.get("experience_min_months")
        if value is not None and minimum is not None and value < minimum:
            raise ValueError("must be greater than or equal to experience_min_months")
        return value

    @field_validator("salary_max")
    @classmethod
    def validate_salary_range(cls, value: float | None, info: Any) -> float | None:
        minimum = info.data.get("salary_min")
        if value is not None and minimum is not None and value < minimum:
            raise ValueError("must be greater than or equal to salary_min")
        return value


class SemanticFact(BaseModel):
    """Internal evidence-bearing fact; CSV projection deliberately stores only value strings."""

    model_config = ConfigDict(extra="forbid")

    value: str = Field(min_length=1)
    category: Literal["responsibility", "requirement", "skill", "benefit", "unknown"]
    importance: Literal["must", "preferred", "neutral", "unknown"] = "unknown"
    source_type: Literal["explicit", "inferred", "user_confirmed"]
    evidence_text: str = Field(min_length=1)
    needs_confirmation: bool = False

    @field_validator("value", "evidence_text")
    @classmethod
    def clean_text(cls, value: str) -> str:
        return " ".join(value.split())

    @model_validator(mode="after")
    def enforce_confirmation_boundary(self) -> "SemanticFact":
        if self.source_type == "inferred" or self.category == "unknown":
            self.needs_confirmation = True
        return self


class RewriteResult(BaseModel):
    """Internal safety decision for one piece of JD copy."""

    model_config = ConfigDict(extra="forbid")

    original_text: str = Field(min_length=1)
    rewritten_text: str = Field(min_length=1)
    category: Literal["responsibility", "requirement", "skill", "benefit", "unknown"]
    source_type: Literal["explicit", "inferred", "user_confirmed"]
    evidence_level: Literal["direct", "contextual", "none"]
    confidence: float = Field(ge=0, le=1)
    decision: Literal["safe_rewrite", "needs_confirmation", "reject_or_clarify"]
    reason: str = Field(min_length=1)

    @field_validator("original_text", "rewritten_text", "reason")
    @classmethod
    def clean_rewrite_text(cls, value: str) -> str:
        return " ".join(value.split())


class SemanticIssue(BaseModel):
    """User-facing semantic review result produced before persistence."""

    model_config = ConfigDict(extra="forbid")

    level: Literal["pass", "warning", "blocking"]
    field: str | None = None
    value: str | None = None
    message: str


class StructuredCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Literal["create", "update", "delete", "search", "confirm", "cancel", "unknown"]
    search_query: str | None = None
    selection_index: int | None = Field(default=None, ge=1)
    fields: JobFields = Field(default_factory=JobFields)
    clear_fields: list[str] = Field(default_factory=list)
    semantic_facts: list[SemanticFact] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)

    @field_validator("clear_fields")
    @classmethod
    def only_editable_clear_fields(cls, value: list[str]) -> list[str]:
        invalid = sorted(set(value) - set(EDITABLE_FIELDS))
        if invalid:
            raise ValueError(f"non-editable fields: {', '.join(invalid)}")
        return list(dict.fromkeys(value))

    @field_validator("clarification_questions")
    @classmethod
    def clean_questions(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(" ".join(item.split()) for item in value if item.strip()))


class ChatRequest(BaseModel):
    content: str = Field(min_length=1, max_length=10_000)


class ChatResponse(BaseModel):
    session_id: str
    phase: Phase
    message: str
    missing_fields: list[str] = Field(default_factory=list)
    candidates: list[dict[str, str]] = Field(default_factory=list)
    can_confirm: bool = False
    can_cancel: bool = False
