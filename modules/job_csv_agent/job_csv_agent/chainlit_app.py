from __future__ import annotations

import asyncio
import os
from typing import Any

import chainlit as cl
import httpx


API_URL = os.getenv("JOB_AGENT_API_URL", "http://127.0.0.1:8000").rstrip("/")


async def _request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.request(method, f"{API_URL}{path}", **kwargs)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except ValueError:
            detail = exc.response.text
        raise RuntimeError(f"FastAPI 请求失败：{detail}") from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"无法连接 FastAPI 后端 {API_URL}：{exc}") from exc


async def _render(payload: dict[str, Any]) -> None:
    actions: list[cl.Action] = []
    if payload.get("can_confirm"):
        actions.extend([
            cl.Action(name="confirm_write", label="按当前内容保存", payload={"value": "confirm"}),
            cl.Action(name="continue_edit", label="继续补充", payload={"value": "edit"}),
        ])
    if payload.get("can_cancel"):
        actions.append(cl.Action(name="cancel_operation", label="取消", payload={"value": "cancel"}))
    if payload.get("lifecycle_status") == "REVIEWED" and payload.get("job_id"):
        actions.append(cl.Action(name="publish_job", label="发布岗位", payload={"value": "publish"}))
    if payload.get("lifecycle_status") == "PUBLISHED" and payload.get("job_id"):
        actions.append(cl.Action(name="close_job", label="关闭岗位", payload={"value": "close"}))
    if payload.get("phase") == "EDITING":
        for index, row in enumerate(payload.get("candidates", []), 1):
            actions.append(cl.Action(
                name="select_job", label=f"{index}. {row['company_name']} - {row['title']}",
                payload={"index": index},
            ))
    await cl.Message(content=payload.get("message", ""), actions=actions).send()


async def _safe_call(method: str, path: str, **kwargs: Any) -> None:
    lock = cl.user_session.get("api_request_lock")
    if lock is None:
        lock = asyncio.Lock()
        cl.user_session.set("api_request_lock", lock)
    try:
        async with lock:
            version = cl.user_session.get("api_state_version", 0)
            if path.endswith("/messages"):
                request_json = dict(kwargs.get("json", {}))
                request_json["expected_version"] = version
                kwargs["json"] = request_json
            elif any(token in path for token in ("/confirm", "/cancel", "/select/", "/publish", "/close")):
                separator = "&" if "?" in path else "?"
                path = f"{path}{separator}expected_version={version}"
            payload = await _request(method, path, **kwargs)
            cl.user_session.set("api_state_version", payload.get("state_version", 0))
            await _render(payload)
    except RuntimeError as exc:
        await cl.Message(content=str(exc)).send()


def _session_id() -> str:
    session_id = cl.user_session.get("api_session_id")
    if not session_id:
        raise RuntimeError("后端会话尚未创建，请刷新页面重试。")
    return str(session_id)


@cl.on_chat_start
async def on_chat_start() -> None:
    try:
        payload = await _request("POST", "/sessions")
        cl.user_session.set("api_session_id", payload["session_id"])
        cl.user_session.set("api_state_version", payload.get("state_version", 0))
        cl.user_session.set("api_request_lock", asyncio.Lock())
        await _render(payload)
    except RuntimeError as exc:
        await cl.Message(content=str(exc)).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    try:
        session_id = _session_id()
    except RuntimeError as exc:
        await cl.Message(content=str(exc)).send()
        return
    await _safe_call(
        "POST", f"/sessions/{session_id}/messages",
        json={"content": message.content},
    )


@cl.action_callback("confirm_write")
async def confirm_write(_: cl.Action) -> None:
    await _safe_call("POST", f"/sessions/{_session_id()}/confirm")


@cl.action_callback("continue_edit")
async def continue_edit(_: cl.Action) -> None:
    await _safe_call(
        "POST", f"/sessions/{_session_id()}/messages",
        json={"content": "继续修改"},
    )


@cl.action_callback("cancel_operation")
async def cancel_operation(_: cl.Action) -> None:
    await _safe_call("POST", f"/sessions/{_session_id()}/cancel")


@cl.action_callback("select_job")
async def select_job(action: cl.Action) -> None:
    index = int(action.payload["index"])
    await _safe_call("POST", f"/sessions/{_session_id()}/select/{index}")


@cl.action_callback("publish_job")
async def publish_job(_: cl.Action) -> None:
    await _safe_call("POST", f"/sessions/{_session_id()}/publish")


@cl.action_callback("close_job")
async def close_job(_: cl.Action) -> None:
    await _safe_call("POST", f"/sessions/{_session_id()}/close")
