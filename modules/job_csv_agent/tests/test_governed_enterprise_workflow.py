from __future__ import annotations

from pathlib import Path
import time

import pytest
from fastapi.testclient import TestClient

from hr_agent_contracts import ActorContext, JobLifecycleStatus
from job_csv_agent.agent import JobCsvAgent
from job_csv_agent.api import InMemorySessionStore, create_app
from job_csv_agent.auth import AuthenticationError, IdentityProvider
from job_csv_agent.clarification import ClarificationPolicy
from job_csv_agent.critic import BoundedProfileCritic
from job_csv_agent.enterprise_agent import EnterpriseHrAgent
from job_csv_agent.governance import AuthorizationError, SqliteControlPlane
from job_csv_agent.governed_repository import GovernedJobRepository
from job_csv_agent.repository import CsvJobRepository
from job_csv_agent.schemas import DraftPatch, EvidenceSpan, StructuredCommand
from job_csv_agent.skills import SkillRegistry
from job_csv_agent.session_store import SqliteSessionStore


class CreateInterpreter:
    def extract(self, text: str, context: dict) -> StructuredCommand:
        del text, context
        return StructuredCommand(
            intent="create", task_relation="start_new",
            patch=DraftPatch(
                set_fields={"company_name": "甲公司", "title": "算法工程师"},
                set_sources={"company_name": "explicit", "title": "explicit"},
                append_items={
                    "responsibilities_json": [{"value": "负责模型训练", "source": "explicit"}],
                },
            ),
            mentioned_fields=["company_name", "title", "responsibilities_json"],
            evidence_spans=[
                EvidenceSpan(field="company_name", text="甲公司"),
                EvidenceSpan(field="title", text="算法工程师"),
                EvidenceSpan(field="responsibilities_json", text="负责模型训练"),
            ],
        )


def _service(tmp_path: Path) -> tuple[EnterpriseHrAgent, SqliteControlPlane]:
    actor = ActorContext(
        company_id="company_a", operator_id="admin_a",
        roles=["recruiter", "hr_admin"],
    )
    control = SqliteControlPlane(tmp_path / "control.sqlite")
    critic = BoundedProfileCritic()
    repository = GovernedJobRepository(
        CsvJobRepository(tmp_path / "jobs.csv"), control, critic, actor,
    )
    skills_root = Path(__file__).resolve().parents[3] / "skills"
    skills = SkillRegistry(skills_root).load()
    capability = JobCsvAgent(
        repository, CreateInterpreter(), ClarificationPolicy(max_turns=3),
    )
    return EnterpriseHrAgent(capability, repository, control, skills), control


def test_governed_graph_confirm_publish_and_tenant_isolation(tmp_path: Path) -> None:
    service, control = _service(tmp_path)
    identity = IdentityProvider(
        None, allow_demo=True, default_company_id="company_a",
        default_operator_id="admin_a", default_roles=["recruiter", "hr_admin"],
    )
    app = create_app(service, InMemorySessionStore(), identity)
    headers = {"X-Company-ID": "company_a", "X-Operator-ID": "admin_a", "X-Roles": "recruiter,hr_admin"}
    with TestClient(app) as client:
        created_session = client.post("/sessions", headers=headers).json()
        session_id = created_session["session_id"]
        preview = client.post(
            f"/sessions/{session_id}/messages", headers=headers,
            json={"content": "甲公司招聘算法工程师，负责模型训练", "expected_version": 0},
        ).json()
        assert preview["phase"] == "CONFIRMING"
        assert preview["workflow_trace"] == [
            "authorize", "select_skills", "route_lifecycle",
            "invoke_job_capability", "attach_quality_status",
        ]

        saved = client.post(
            f"/sessions/{session_id}/confirm?expected_version={preview['state_version']}",
            headers=headers,
        ).json()
        assert saved["lifecycle_status"] == "REVIEWED"
        assert saved["job_profile_version"] == 1
        assert saved["job_id"]

        published = client.post(
            f"/sessions/{session_id}/publish?expected_version={saved['state_version']}",
            headers=headers,
        ).json()
        assert published["lifecycle_status"] == "PUBLISHED"

        audit = client.get("/audit", headers=headers).json()
        assert {item["action"] for item in audit} >= {"job.created", "job.published"}

    foreign_actor = ActorContext(
        company_id="company_b", operator_id="admin_b", roles=["hr_admin"],
    )
    with pytest.raises(AuthorizationError):
        control.latest(foreign_actor, saved["job_id"])


def test_soft_delete_retains_version_history(tmp_path: Path) -> None:
    service, control = _service(tmp_path)
    actor = ActorContext(
        company_id="company_a", operator_id="admin_a", roles=["recruiter", "hr_admin"],
    )
    with service.repository.bind(actor):
        row = service.repository.create({
            "company_name": "甲公司", "title": "测试工程师",
            "responsibilities_json": ["负责测试"],
        })
        service.repository.delete(row["job_id"])
    deleted = control.latest(actor, row["job_id"], include_deleted=True)
    assert deleted.status == JobLifecycleStatus.DELETED
    assert len(control.history(actor, row["job_id"])) == 1


def test_skill_registry_rejects_unknown_skill(tmp_path: Path) -> None:
    registry = SkillRegistry(tmp_path).load()
    with pytest.raises(KeyError):
        registry.compose_instructions(["untrusted_skill"])


def test_signed_identity_cannot_be_overridden_by_headers() -> None:
    provider = IdentityProvider(
        "test-secret", allow_demo=False, default_company_id="unused",
        default_operator_id="unused", default_roles=[],
    )
    token = provider.issue_for_testing({
        "company_id": "signed_company", "operator_id": "signed_operator",
        "roles": ["hr_admin"], "exp": time.time() + 60,
    })
    actor = provider.resolve(
        f"Bearer {token}", "spoofed_company", "spoofed_operator", "company_admin", "request_1",
    )
    assert actor.company_id == "signed_company"
    assert actor.operator_id == "signed_operator"
    assert actor.roles == ["hr_admin"]

    expired = provider.issue_for_testing({
        "company_id": "signed_company", "operator_id": "signed_operator",
        "roles": ["hr_admin"], "exp": time.time() - 1,
    })
    with pytest.raises(AuthenticationError):
        provider.resolve(f"Bearer {expired}", None, None, None, None)


def test_sqlite_session_and_control_plane_can_share_one_database(tmp_path: Path) -> None:
    database = tmp_path / "shared.sqlite"
    actor = ActorContext(
        company_id="company_a", operator_id="admin_a",
        roles=["recruiter", "hr_admin"],
    )
    control = SqliteControlPlane(database)
    critic = BoundedProfileCritic()
    repository = GovernedJobRepository(
        CsvJobRepository(tmp_path / "jobs.csv"), control, critic, actor,
    )
    skills_root = Path(__file__).resolve().parents[3] / "skills"
    service = EnterpriseHrAgent(
        JobCsvAgent(repository, CreateInterpreter()), repository, control,
        SkillRegistry(skills_root).load(),
    )
    identity = IdentityProvider(
        None, allow_demo=True, default_company_id="company_a",
        default_operator_id="admin_a", default_roles=["recruiter", "hr_admin"],
    )
    app = create_app(service, SqliteSessionStore(database), identity)
    with TestClient(app) as client:
        session = client.post("/sessions").json()
        response = client.post(
            f"/sessions/{session['session_id']}/messages",
            json={"content": "甲公司招聘算法工程师，负责模型训练", "expected_version": 0},
        )
        assert response.status_code == 200
        assert response.json()["workflow_trace"][-1] == "attach_quality_status"
