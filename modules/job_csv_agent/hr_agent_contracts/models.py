from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA_VERSION = "1.0.0"


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION


class ConversationPhase(StrEnum):
    IDLE = "IDLE"
    COLLECTING = "COLLECTING"
    WAITING_ANSWER = "WAITING_ANSWER"
    WAITING_CONFIRMATION = "WAITING_CONFIRMATION"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class JobLifecycleStatus(StrEnum):
    DRAFT = "DRAFT"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    REVIEWED = "REVIEWED"
    PUBLISHED = "PUBLISHED"
    CLOSED = "CLOSED"
    EXPIRED = "EXPIRED"
    DELETED = "DELETED"


class ActorContext(StrictContract):
    company_id: str = Field(min_length=1, max_length=128)
    operator_id: str = Field(min_length=1, max_length=128)
    roles: list[str] = Field(default_factory=lambda: ["recruiter"])
    request_id: str | None = None

    def can_write_jobs(self) -> bool:
        return bool({"recruiter", "hr_admin", "company_admin"} & set(self.roles))

    def can_publish_jobs(self) -> bool:
        return bool({"hr_admin", "company_admin"} & set(self.roles))


class Evidence(StrictContract):
    field: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_message_id: str | None = None
    source_type: Literal["explicit", "contextual", "normalized", "user_confirmed"] = "explicit"
    confirmed: bool = False


class FieldFact(StrictContract):
    field: str = Field(min_length=1)
    value: Any
    modality: Literal["must", "preferred", "not_required", "neutral", "unknown"] = "neutral"
    status: Literal["confirmed", "unconfirmed", "unknown"] = "unconfirmed"
    evidence: list[Evidence] = Field(default_factory=list)


class JobProfile(StrictContract):
    job_id: str | None = None
    company_id: str = Field(min_length=1)
    company_name: str | None = None
    title: str | None = None
    department: str | None = None
    job_category: str | None = None
    headcount: int | None = Field(default=None, ge=1)
    city: str | None = None
    work_address: str | None = None
    work_locations: list[str] = Field(default_factory=list)
    salary_min: float | None = Field(default=None, ge=0)
    salary_max: float | None = Field(default=None, ge=0)
    salary_currency: str | None = None
    salary_period: str | None = None
    recruitment: str | None = None
    recruitment_batch: str | None = None
    employment: str | None = None
    work_mode: str | None = None
    application_deadline: str | None = None
    education_min_level: int | None = Field(default=None, ge=0, le=6)
    experience_min_months: int | None = Field(default=None, ge=0)
    experience_max_months: int | None = Field(default=None, ge=0)
    internship_min_months: int | None = Field(default=None, ge=0)
    onsite_days_per_week: int | None = Field(default=None, ge=0, le=7)
    graduation_years: list[str] = Field(default_factory=list)
    major_requirements: list[str] = Field(default_factory=list)
    student_status: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    certificates: list[str] = Field(default_factory=list)
    benefits: list[str] = Field(default_factory=list)
    preferred_requirements: list[str] = Field(default_factory=list)
    not_required_requirements: list[str] = Field(default_factory=list)
    source_url: str | None = None
    facts: list[FieldFact] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ranges(self) -> "JobProfile":
        if self.salary_min is not None and self.salary_max is not None and self.salary_max < self.salary_min:
            raise ValueError("salary_max must be greater than or equal to salary_min")
        if (
            self.experience_min_months is not None
            and self.experience_max_months is not None
            and self.experience_max_months < self.experience_min_months
        ):
            raise ValueError("experience_max_months must be greater than or equal to experience_min_months")
        return self


class JobPatch(StrictContract):
    job_id: str | None = None
    expected_version: int | None = Field(default=None, ge=0)
    set_fields: dict[str, Any] = Field(default_factory=dict)
    clear_fields: list[str] = Field(default_factory=list)
    append_items: dict[str, list[Any]] = Field(default_factory=dict)
    remove_items: dict[str, list[Any]] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    reason: str = "user_requested_change"


class ClarificationQuestion(StrictContract):
    question_id: str = Field(min_length=1)
    job_id: str | None = None
    fields: list[str] = Field(min_length=1)
    prompt: str = Field(min_length=1)
    priority: Literal["blocking", "high", "normal", "low"] = "normal"
    reason: str = Field(min_length=1)
    sequence: int = Field(default=1, ge=1)


class ClarificationAnswer(StrictContract):
    question_id: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    source_message_id: str | None = None


class ReviewIssue(StrictContract):
    code: str = Field(min_length=1)
    severity: Literal["blocking", "warning", "info"]
    fields: list[str] = Field(default_factory=list)
    message: str = Field(min_length=1)
    suggested_question: str | None = None


class JobProfileReview(StrictContract):
    status: Literal["pass", "needs_clarification", "conflict"]
    reviewed_profile_version: int = Field(default=0, ge=0)
    reviewer: Literal["deterministic", "llm_critic", "combined"] = "deterministic"
    issues: list[ReviewIssue] = Field(default_factory=list)

    @property
    def publishable(self) -> bool:
        return self.status == "pass" and not any(issue.severity == "blocking" for issue in self.issues)


class JobProfileVersion(StrictContract):
    job_id: str
    version: int = Field(ge=1)
    status: JobLifecycleStatus
    profile: JobProfile
    review: JobProfileReview | None = None
    created_at: str
    created_by: str


class AuditEvent(StrictContract):
    event_id: str
    company_id: str
    operator_id: str
    action: str
    resource_type: str
    resource_id: str | None = None
    occurred_at: str
    request_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
