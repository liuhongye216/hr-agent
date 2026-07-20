from __future__ import annotations

import threading
import uuid
import logging
from typing import Any

from fastapi import FastAPI, HTTPException

from .agent import AgentState, JobCsvAgent, initial_state
from .factory import create_default_agent
from .config import BUSINESS_CSV, DEEPSEEK_MODEL, is_valid_api_key
from .llm import LLMConfigurationError, LLMServiceError
from .schemas import FIELD_LABELS, ChatRequest, ChatResponse, Phase


class InMemorySessionStore:
    """MVP session store. Replace with Redis for multi-process production deployment."""

    def __init__(self) -> None:
        self._states: dict[str, AgentState] = {}
        self._lock = threading.RLock()

    def create(self) -> tuple[str, AgentState]:
        with self._lock:
            session_id = uuid.uuid4().hex
            state = initial_state()
            self._states[session_id] = state
            return session_id, state

    def apply(
        self, session_id: str, callback: Any, expected_version: int | None = None,
    ) -> AgentState:
        with self._lock:
            if session_id not in self._states:
                raise KeyError(session_id)
            current = self._states[session_id]
            version = int(current.get("state_version", 0))
            if expected_version is not None and expected_version != version:
                raise StateConflictError(expected_version, version)
            state = callback(current)
            state["state_version"] = version + 1
            self._states[session_id] = state
            return state


class StateConflictError(RuntimeError):
    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(f"会话状态已变化：请求版本 {expected}，当前版本 {actual}")
        self.expected = expected
        self.actual = actual


def _response(session_id: str, state: AgentState) -> ChatResponse:
    return ChatResponse(
        session_id=session_id,
        phase=Phase(state.get("phase", Phase.IDLE.value)),
        message=state.get("message", ""),
        missing_fields=[
            FIELD_LABELS.get(field, field) for field in state.get("missing_fields", [])
        ],
        candidates=[
            {key: str(row.get(key, "")) for key in ("company_name", "title", "city")}
            for row in state.get("candidates", [])
        ],
        can_confirm=state.get("can_confirm", False),
        can_cancel=state.get("can_cancel", False),
        storage_valid=state.get("storage_valid", False),
        extraction_complete=state.get("extraction_complete", True),
        unresolved_fragments=[
            str(item.get("text", "")) for item in state.get("unresolved_fragments", [])
        ],
        requested_fields=[
            FIELD_LABELS.get(field, field) for field in state.get("requested_fields", [])
        ],
        state_version=state.get("state_version", 0),
    )


def create_app(agent: JobCsvAgent | None = None) -> FastAPI:
    service = agent or create_default_agent()
    sessions = InMemorySessionStore()
    app = FastAPI(title="招聘岗位自然语言 CSV Agent", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str | bool]:
        configured = is_valid_api_key()
        return {
            "status": "ok" if configured and BUSINESS_CSV.exists() else "degraded",
            "llm_configured": configured,
            "business_csv_exists": BUSINESS_CSV.exists(),
            "model": DEEPSEEK_MODEL,
        }

    @app.post("/sessions", response_model=ChatResponse)
    def create_session() -> ChatResponse:
        session_id, state = sessions.create()
        return _response(session_id, state)

    def run(
        session_id: str,
        text: str,
        forced: dict[str, Any] | None = None,
        expected_version: int | None = None,
    ) -> ChatResponse:
        try:
            state = sessions.apply(
                session_id, lambda current: service.handle(current, text, forced), expected_version,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="session 不存在") from exc
        except StateConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "会话已被另一条消息更新，请等待上一条回复后重试。",
                    "expected_version": exc.expected,
                    "current_version": exc.actual,
                },
            ) from exc
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
            logging.getLogger(__name__).exception("Unhandled job CSV agent error")
            raise HTTPException(status_code=500, detail="后端发生未预期错误，请查看 FastAPI 日志。") from exc
        return _response(session_id, state)

    @app.post("/sessions/{session_id}/messages", response_model=ChatResponse)
    def message(session_id: str, request: ChatRequest) -> ChatResponse:
        return run(session_id, request.content, expected_version=request.expected_version)

    @app.post("/sessions/{session_id}/confirm", response_model=ChatResponse)
    def confirm(session_id: str, expected_version: int) -> ChatResponse:
        return run(
            session_id, "确认写入", {"intent": "confirm"}, expected_version=expected_version,
        )

    @app.post("/sessions/{session_id}/cancel", response_model=ChatResponse)
    def cancel(session_id: str, expected_version: int | None = None) -> ChatResponse:
        return run(session_id, "取消", {"intent": "cancel"}, expected_version=expected_version)

    @app.post("/sessions/{session_id}/select/{selection_index}", response_model=ChatResponse)
    def select(
        session_id: str, selection_index: int, expected_version: int | None = None,
    ) -> ChatResponse:
        return run(
            session_id, str(selection_index),
            {"intent": "unknown", "selection_index": selection_index},
            expected_version=expected_version,
        )

    return app


app = create_app()
