from __future__ import annotations

import re
from typing import Protocol

from hr_agent_contracts import JobProfile, JobProfileReview, ReviewIssue


_SENSITIVE_RE = re.compile(
    r"(?:限|只要|仅限)?(?:男性|女性|男生|女生|男士|女士)|"
    r"(?:未婚|已婚|婚育|生育)|(?:汉族|少数民族)|(?:年龄|岁以下|岁以上)"
)


class AdvisoryReviewer(Protocol):
    def review(self, profile: JobProfile, version: int) -> JobProfileReview: ...


class DeterministicProfileCritic:
    """Read-only publish gate. It reports issues and never mutates the profile."""

    def review(self, profile: JobProfile, version: int = 0) -> JobProfileReview:
        issues: list[ReviewIssue] = []
        if not profile.company_name:
            issues.append(ReviewIssue(
                code="missing_company_name", severity="blocking", fields=["company_name"],
                message="缺少公司名称。", suggested_question="请补充公司名称。",
            ))
        if not profile.title:
            issues.append(ReviewIssue(
                code="missing_title", severity="blocking", fields=["title"],
                message="缺少岗位名称。", suggested_question="请补充岗位名称。",
            ))
        has_content = bool(
            profile.responsibilities or profile.requirements or profile.skills
            or profile.education_min_level is not None or profile.experience_min_months is not None
            or profile.major_requirements
        )
        if not has_content:
            issues.append(ReviewIssue(
                code="missing_job_content", severity="blocking",
                fields=["responsibilities", "requirements"],
                message="岗位画像缺少职责或任职要求。",
                suggested_question="请至少补充一项岗位职责或任职要求。",
            ))

        groups = {
            "must": set(profile.requirements),
            "preferred": set(profile.preferred_requirements),
            "not_required": set(profile.not_required_requirements),
        }
        for left, right in (("must", "preferred"), ("must", "not_required"), ("preferred", "not_required")):
            overlap = sorted(groups[left] & groups[right])
            if overlap:
                issues.append(ReviewIssue(
                    code="requirement_modality_conflict", severity="blocking",
                    fields=["requirements", "preferred_requirements", "not_required_requirements"],
                    message=f"同一条件存在属性冲突：{'、'.join(overlap[:3])}。",
                    suggested_question="请确认这些条件最终属于硬性要求、加分项还是非必需条件。",
                ))

        source_text = "\n".join(
            [*profile.requirements, *profile.preferred_requirements,
             *profile.not_required_requirements, *profile.responsibilities]
        )
        if _SENSITIVE_RE.search(source_text):
            issues.append(ReviewIssue(
                code="sensitive_employment_condition", severity="warning",
                fields=["requirements"],
                message="岗位条件可能包含年龄、性别、婚育或民族等敏感限制，需要企业合规人员复核。",
            ))

        blocking = [issue for issue in issues if issue.severity == "blocking"]
        conflict = any(issue.code.endswith("conflict") for issue in blocking)
        status = "conflict" if conflict else ("needs_clarification" if blocking else "pass")
        return JobProfileReview(
            status=status,
            reviewed_profile_version=version,
            reviewer="deterministic",
            issues=issues,
        )


class BoundedProfileCritic:
    """Runs deterministic checks and at most one optional advisory review."""

    def __init__(self, advisory: AdvisoryReviewer | None = None) -> None:
        self.deterministic = DeterministicProfileCritic()
        self.advisory = advisory

    def review(self, profile: JobProfile, version: int = 0) -> JobProfileReview:
        base = self.deterministic.review(profile, version)
        if self.advisory is None or not base.publishable:
            return base
        advisory = self.advisory.review(profile, version)
        issues = [*base.issues, *advisory.issues]
        blocking = [issue for issue in issues if issue.severity == "blocking"]
        status = "needs_clarification" if blocking else "pass"
        return JobProfileReview(
            status=status,
            reviewed_profile_version=version,
            reviewer="combined",
            issues=issues,
        )
