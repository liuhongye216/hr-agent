from __future__ import annotations

import re
import time
import uuid
from datetime import datetime
from typing import Any

from hr_agent_contracts import ActorContext

from .agent import AgentState, JobCsvAgent
from .governance import SqliteControlPlane
from .governed_repository import GovernedJobRepository
from .skills import SkillRegistry
from .workflow import END, ExplicitStateGraph


_PUBLISH_RE = re.compile(r"^(?:请)?发布(?:当前|这个)?岗位[。.!！]?$|^(?:确认)?发布[。.!！]?$")
_CLOSE_RE = re.compile(r"^(?:请)?关闭(?:当前|这个)?岗位[。.!！]?$|^停止招聘[。.!！]?$")
_HISTORY_RE = re.compile(r"^(?:查看|显示)(?:当前|这个)?岗位(?:版本|历史)[。.!！]?$")


class EnterpriseHrAgent:
    """Governed company-side orchestrator; the legacy CSV agent is one capability node."""

    def __init__(
        self, capability: JobCsvAgent, repository: GovernedJobRepository,
        control: SqliteControlPlane, skills: SkillRegistry,
    ) -> None:
        self.capability = capability
        self.repository = repository
        self.control = control
        self.skills = skills
        self.graph = self._build_graph()

    @staticmethod
    def _actor(state: AgentState) -> ActorContext:
        return ActorContext(
            company_id=state.get("company_id", "company_demo"),
            operator_id=state.get("operator_id", "operator_demo"),
            roles=list(state.get("roles", ["recruiter", "hr_admin"])),
            request_id=state.get("request_id"),
        )

    def _build_graph(self) -> ExplicitStateGraph:
        graph = ExplicitStateGraph(max_steps=10)
        graph.add_node("authorize", self._authorize)
        graph.add_node("select_skills", self._select_skills)
        graph.add_node("route_lifecycle", self._route_lifecycle)
        graph.add_node("invoke_job_capability", self._invoke_job_capability)
        graph.add_node("attach_quality_status", self._attach_quality_status)
        graph.add_edge("authorize", "select_skills")
        graph.add_edge("select_skills", "route_lifecycle")
        graph.add_edge("invoke_job_capability", "attach_quality_status")
        graph.add_edge("attach_quality_status", END)
        graph.set_entrypoint("authorize")
        return graph

    def _authorize(self, workflow: dict[str, Any]) -> None:
        actor = self._actor(workflow["agent_state"])
        workflow["actor"] = actor

    def _select_skills(self, workflow: dict[str, Any]) -> None:
        state: AgentState = workflow["agent_state"]
        text = workflow["text"]
        lifecycle = bool(_PUBLISH_RE.fullmatch(text) or _CLOSE_RE.fullmatch(text) or _HISTORY_RE.fullmatch(text))
        skill_ids = [] if lifecycle else ["jd_profile_v1", "clarification_v1"]
        state["active_skills"] = skill_ids
        state["skill_instructions"] = self.skills.compose_instructions(skill_ids)

    def _route_lifecycle(self, workflow: dict[str, Any]) -> str:
        state: AgentState = workflow["agent_state"]
        forced = workflow.get("forced") or {}
        text = workflow["text"]
        intent = forced.get("intent")
        if intent in {"publish", "close", "history"}:
            operation = intent
        elif _PUBLISH_RE.fullmatch(text):
            operation = "publish"
        elif _CLOSE_RE.fullmatch(text):
            operation = "close"
        elif _HISTORY_RE.fullmatch(text):
            operation = "history"
        else:
            return "invoke_job_capability"
        job_id = forced.get("job_id") or state.get("last_job_id") or state.get("target_id")
        if not job_id:
            state["message"] = "当前没有可操作的岗位，请先保存或选择一个岗位。"
            workflow["result"] = state
            return END
        actor: ActorContext = workflow["actor"]
        expected = forced.get("expected_job_version") or state.get("last_profile_version")
        if operation == "publish":
            version = self.control.publish(actor, str(job_id), expected)
            state["last_lifecycle_status"] = version.status.value
            state["last_profile_version"] = version.version
            state["message"] = f"岗位已发布：{version.profile.company_name} - {version.profile.title}（版本 {version.version}）。"
            state["conversation_phase"] = "COMPLETED"
        elif operation == "close":
            self.control.close(actor, str(job_id))
            state["last_lifecycle_status"] = "CLOSED"
            state["message"] = "岗位已关闭，不再处于发布状态。"
            state["conversation_phase"] = "COMPLETED"
        else:
            history = self.control.history(actor, str(job_id))
            state["message"] = "\n".join(
                f"- 版本 {item.version}：{item.status.value}，由 {item.created_by} 于 {item.created_at} 创建"
                for item in history
            )
            state["conversation_phase"] = "COMPLETED"
        workflow["result"] = state
        return END

    def _invoke_job_capability(self, workflow: dict[str, Any]) -> None:
        state: AgentState = workflow["agent_state"]
        actor: ActorContext = workflow["actor"]
        with self.repository.bind(actor, state.get("target_profile_version")):
            workflow["result"] = self.capability.handle(
                state, workflow["text"], workflow.get("forced"),
            )

    def _attach_quality_status(self, workflow: dict[str, Any]) -> None:
        state: AgentState = workflow["result"]
        if state.get("last_operation") not in {"create", "update"} or not state.get("last_job_id"):
            return
        version = self.control.latest(workflow["actor"], str(state["last_job_id"]))
        state["last_profile_version"] = version.version
        state["last_lifecycle_status"] = version.status.value
        if version.review and version.review.issues:
            warnings = [issue.message for issue in version.review.issues if issue.severity == "warning"]
            if warnings:
                state["message"] += "\n\n质量门提示：" + "；".join(warnings)
        state["message"] += f"\n\n岗位画像版本：{version.version}；状态：{version.status.value}。"

    def handle(
        self, state: AgentState, text: str, forced_command: dict[str, Any] | None = None,
    ) -> AgentState:
        started = time.perf_counter()
        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        run_id = f"run_{uuid.uuid4().hex}"
        actor = self._actor(state)
        workflow: dict[str, Any] = {
            "agent_state": state, "text": " ".join(text.split()), "forced": forced_command,
        }
        status = "completed"
        error_type: str | None = None
        trace: list[str] = []
        try:
            completed = self.graph.invoke(workflow)
            trace = completed.get("workflow_trace", [])
            result: AgentState = completed.get("result", state)
            result["workflow_trace"] = trace
            return result
        except Exception as exc:
            status = "failed"
            error_type = type(exc).__name__
            raise
        finally:
            self.control.record_workflow_run(
                run_id=run_id, session_id=state.get("session_id"), actor=actor,
                started_at=started_at, duration_ms=(time.perf_counter() - started) * 1000,
                status=status, nodes=trace, error_type=error_type,
            )
