from __future__ import annotations

from .agent import JobCsvAgent
from hr_agent_contracts import ActorContext

from .clarification import ClarificationPolicy
from .config import (
    BUSINESS_CSV, CONTROL_PLANE_DB, DATABASE_URL, DEFAULT_COMPANY_ID, DEFAULT_OPERATOR_ID,
    DEFAULT_ROLES, MAX_CLARIFICATION_TURNS, SKILL_ROOT,
)
from .critic import BoundedProfileCritic
from .enterprise_agent import EnterpriseHrAgent
from .governance import create_control_plane
from .governed_repository import GovernedJobRepository
from .llm import StructuredInterpreter
from .repository import CsvJobRepository
from .skills import SkillRegistry


def create_default_agent() -> EnterpriseHrAgent:
    actor = ActorContext(
        company_id=DEFAULT_COMPANY_ID, operator_id=DEFAULT_OPERATOR_ID,
        roles=list(DEFAULT_ROLES),
    )
    control = create_control_plane(CONTROL_PLANE_DB, DATABASE_URL)
    critic = BoundedProfileCritic()
    repository = GovernedJobRepository(
        CsvJobRepository(BUSINESS_CSV), control, critic, actor,
    )
    repository.bootstrap_legacy()
    capability = JobCsvAgent(
        repository=repository,
        interpreter=StructuredInterpreter(),
        clarification_policy=ClarificationPolicy(MAX_CLARIFICATION_TURNS),
    )
    skills = SkillRegistry(
        SKILL_ROOT,
        allowlist={"jd_profile_v1", "clarification_v1", "profile_critic_v1"},
    ).load()
    return EnterpriseHrAgent(capability, repository, control, skills)
