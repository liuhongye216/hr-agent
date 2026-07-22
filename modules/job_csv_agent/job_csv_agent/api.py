from __future__ import annotations

import logging
import threading
import uuid
from typing import Any, Protocol

from fastapi import Depends, FastAPI, Header, HTTPException

from hr_agent_contracts import ActorContext

from .agent import AgentState, JobCsvAgent, initial_state
from .auth import AuthenticationError, IdentityProvider
from .config import (
    ALLOW_DEMO_IDENTITY, AUTH_SECRET, BUSINESS_CSV, CONTROL_PLANE_DB, DATABASE_URL,
    DEEPSEEK_MODEL, DEFAULT_COMPANY_ID, DEFAULT_OPERATOR_ID, DEFAULT_ROLES,
    is_valid_api_key,
)
from .factory import create_default_agent
from .governance import (
    AuthorizationError, LifecycleConflictError, VersionConflictError,
)
from .llm import LLMConfigurationError, LLMServiceError
from .repository import JobNotFoundError
from .schemas import FIELD_LABELS, ChatRequest, ChatResponse, Phase
from .session_store import PostgresSessionStore, SqliteSessionStore


class AgentService(Protocol):
    def handle(
        self, state: AgentState, text: str, forced_command: dict[str, Any] | None = None,
    ) -> AgentState: ...


class SessionStore(Protocol):
    def create(self, actor: ActorContext) -> tuple[str, AgentState]: ...
    def apply(
        self, session_id: str, actor: ActorContext, callback: Any,
        expected_version: int | None = None,
    ) -> AgentState: ...


class InMemorySessionStore:
    """Test/local injected-agent store; production default uses durable SQLite."""

    def __init__(self) -> None:
        self._states: dict[str, tuple[ActorContext, AgentState]] = {}
        self._lock = threading.RLock()

    def create(self, actor: ActorContext) -> tuple[str, AgentState]:
        with self._lock:
            session_id = uuid.uuid4().hex
            state = initial_state()
            state.update({
                "company_id": actor.company_id, "operator_id": actor.operator_id,
                "roles": list(actor.roles), "session_id": session_id,
            })
            self._states[session_id] = (actor, state)
            return session_id, state

    def apply(
        self, session_id: str, actor: ActorContext, callback: Any,
        expected_version: int | None = None,
    ) -> AgentState:
        with self._lock:
            if session_id not in self._states:
                raise KeyError(session_id)
            owner, current = self._states[session_id]
            if owner.company_id != actor.company_id or owner.operator_id != actor.operator_id:
                raise AuthorizationError("会话不属于当前企业操作者")
            version = int(current.get("state_version", 0))
            if expected_version is not None and expected_version != version:
                raise VersionConflictError(expected_version, version)
            state = callback(current)
            state.update({
                "state_version": version + 1, "company_id": actor.company_id,
                "operator_id": actor.operator_id, "roles": list(actor.roles),
                "session_id": session_id,
            })
            self._states[session_id] = (owner, state)
            return state


def _response(session_id: str, state: AgentState) -> ChatResponse:
    return ChatResponse(
        session_id=session_id,
        phase=Phase(state.get("phase", Phase.IDLE.value)),
        message=state.get("message", ""),
        missing_fields=[FIELD_LABELS.get(field, field) for field in state.get("missing_fields", [])],
        candidates=[
            {key: str(row.get(key, "")) for key in ("company_name", "title", "city")}
            for row in state.get("candidates", [])
        ],
        can_confirm=state.get("can_confirm", False),
        can_cancel=state.get("can_cancel", False),
        storage_valid=state.get("storage_valid", False),
        extraction_complete=state.get("extraction_complete", True),
        unresolved_fragments=[str(item.get("text", "")) for item in state.get("unresolved_fragments", [])],
        requested_fields=[FIELD_LABELS.get(field, field) for field in state.get("requested_fields", [])],
        state_version=state.get("state_version", 0),
        job_id=state.get("last_job_id") or state.get("target_id"),
        job_profile_version=state.get("last_profile_version") or state.get("target_profile_version"),
        lifecycle_status=state.get("last_lifecycle_status"),
        active_question=state.get("active_question"),
        workflow_trace=list(state.get("workflow_trace", [])),
        conversation_phase=state.get("conversation_phase", "IDLE"),
    )


def create_app(
    agent: AgentService | None = None, session_store: SessionStore | None = None,
    identity_provider: IdentityProvider | None = None,
) -> FastAPI:
    service = agent or create_default_agent()
    if session_store is not None:
        sessions = session_store
    elif agent is not None:
        sessions = InMemorySessionStore()
    elif DATABASE_URL:
        sessions = PostgresSessionStore(getattr(service, "control"))
    else:
        sessions = SqliteSessionStore(CONTROL_PLANE_DB)
    identity = identity_provider or IdentityProvider(
        AUTH_SECRET, allow_demo=ALLOW_DEMO_IDENTITY,
        default_company_id=DEFAULT_COMPANY_ID, default_operator_id=DEFAULT_OPERATOR_ID,
        default_roles=list(DEFAULT_ROLES),
    )
    app = FastAPI(title="企业侧招聘岗位 Agent", version="1.0.0")

    def actor_dependency(
        authorization: str | None = Header(default=None),
        x_company_id: str | None = Header(default=None),
        x_operator_id: str | None = Header(default=None),
        x_roles: str | None = Header(default=None),
        x_request_id: str | None = Header(default=None),
    ) -> ActorContext:
        try:
            return identity.resolve(
                authorization, x_company_id, x_operator_id, x_roles,
                x_request_id or f"request_{uuid.uuid4().hex}",
            )
        except AuthenticationError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    @app.get("/health")
    def health() -> dict[str, str | bool]:
        configured = is_valid_api_key()
        return {
            "status": "ok" if configured and BUSINESS_CSV.exists() else "degraded",
            "llm_configured": configured,
            "business_csv_exists": BUSINESS_CSV.exists(),
            "model": DEEPSEEK_MODEL,
            "durable_sessions": not isinstance(sessions, InMemorySessionStore),
            "signed_identity_required": bool(AUTH_SECRET),
        }

    @app.post("/sessions", response_model=ChatResponse)
    def create_session(actor: ActorContext = Depends(actor_dependency)) -> ChatResponse:
        session_id, state = sessions.create(actor)
        state["session_id"] = session_id
        return _response(session_id, state)

    def run(
        session_id: str, actor: ActorContext, text: str,
        forced: dict[str, Any] | None = None, expected_version: int | None = None,
    ) -> ChatResponse:
        try:
            state = sessions.apply(
                session_id, actor,
                lambda current: service.handle(current, text, forced),
                expected_version,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="session 不存在") from exc
        except (AuthenticationError, AuthorizationError) as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except VersionConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": str(exc), "expected_version": exc.expected,
                    "current_version": exc.actual,
                },
            ) from exc
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="岗位不存在") from exc
        except LifecycleConflictError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except LLMConfigurationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except LLMServiceError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except (ValueError, RuntimeError, OSError) as exc:
            detail = str(exc)
            internal_terms = (
                "job_id", "requirements_json", "responsibilities_json", "skills_json",
                "schema", "extraction_mode",
            )
            if any(term in detail.casefold() for term in internal_terms):
                logging.getLogger(__name__).warning("Job validation failed: %s", detail)
                detail = "岗位信息校验未通过，请检查已填写的内容后重试。"
            raise HTTPException(status_code=422, detail=detail) from exc
        except Exception as exc:
            logging.getLogger(__name__).exception("Unhandled enterprise HR agent error")
            raise HTTPException(status_code=500, detail="后端发生未预期错误，请查看 FastAPI 日志。") from exc
        return _response(session_id, state)

    @app.post("/sessions/{session_id}/messages", response_model=ChatResponse)
    def message(
        session_id: str, request: ChatRequest,
        actor: ActorContext = Depends(actor_dependency),
    ) -> ChatResponse:
        return run(session_id, actor, request.content, expected_version=request.expected_version)

    @app.post("/sessions/{session_id}/confirm", response_model=ChatResponse)
    def confirm(
        session_id: str, expected_version: int,
        actor: ActorContext = Depends(actor_dependency),
    ) -> ChatResponse:
        return run(session_id, actor, "确认写入", {"intent": "confirm"}, expected_version)

    @app.post("/sessions/{session_id}/cancel", response_model=ChatResponse)
    def cancel(
        session_id: str, expected_version: int | None = None,
        actor: ActorContext = Depends(actor_dependency),
    ) -> ChatResponse:
        return run(session_id, actor, "取消", {"intent": "cancel"}, expected_version)

    @app.post("/sessions/{session_id}/select/{selection_index}", response_model=ChatResponse)
    def select(
        session_id: str, selection_index: int, expected_version: int | None = None,
        actor: ActorContext = Depends(actor_dependency),
    ) -> ChatResponse:
        return run(
            session_id, actor, str(selection_index),
            {"intent": "unknown", "selection_index": selection_index}, expected_version,
        )

    @app.post("/sessions/{session_id}/publish", response_model=ChatResponse)
    def publish(
        session_id: str, expected_version: int,
        expected_job_version: int | None = None, job_id: str | None = None,
        actor: ActorContext = Depends(actor_dependency),
    ) -> ChatResponse:
        return run(
            session_id, actor, "发布岗位",
            {"intent": "publish", "job_id": job_id,
             "expected_job_version": expected_job_version}, expected_version,
        )

    @app.post("/sessions/{session_id}/close", response_model=ChatResponse)
    def close(
        session_id: str, expected_version: int, job_id: str | None = None,
        actor: ActorContext = Depends(actor_dependency),
    ) -> ChatResponse:
        return run(
            session_id, actor, "关闭岗位", {"intent": "close", "job_id": job_id},
            expected_version,
        )

    @app.get("/audit")
    def audit(
        limit: int = 100, actor: ActorContext = Depends(actor_dependency),
    ) -> list[dict[str, Any]]:
        control = getattr(service, "control", None)
        if control is None:
            raise HTTPException(status_code=404, detail="当前 Agent 未启用治理控制面")
        return [event.model_dump(mode="json") for event in control.list_audit(actor, limit)]

    @app.get("/metrics")
    def metrics(actor: ActorContext = Depends(actor_dependency)) -> dict[str, Any]:
        control = getattr(service, "control", None)
        if control is None:
            raise HTTPException(status_code=404, detail="当前 Agent 未启用治理控制面")
        return control.metrics(actor)

    return app


app = create_app()
