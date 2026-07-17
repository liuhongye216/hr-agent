from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Protocol, TypedDict

from pydantic import ValidationError

from .llm import LLMConfigurationError, LLMServiceError
from .normalization import normalize_patch
from .patching import (
    apply_draft_patch, changed_fields, sanitize_model_patch, unapplied_fields,
)
from .repository import CsvJobRepository
from .schemas import (
    CREATE_CONTENT_FIELDS, FIELD_LABELS, JSON_FIELDS, REQUIRED_CREATE_FIELDS,
    DraftPatch, JobFields, Phase, QueryPlan, StructuredCommand,
)


class Interpreter(Protocol):
    def extract(self, text: str, context: dict[str, Any]) -> StructuredCommand: ...


class AgentState(TypedDict, total=False):
    phase: str
    draft: dict[str, Any]
    provenance: dict[str, Any]
    recent_messages: list[dict[str, str]]
    last_assistant_question: str | None
    pending_decision: dict[str, Any] | None
    pending_action: str | None
    target_id: str | None
    candidates: list[dict[str, str]]
    can_confirm: bool
    can_cancel: bool
    message: str
    missing_fields: list[str]
    state_version: int
    draft_revision: int
    preview_revision: int | None


def initial_state() -> AgentState:
    return {
        "phase": Phase.IDLE.value,
        "draft": {},
        "provenance": {},
        "recent_messages": [],
        "last_assistant_question": None,
        "pending_decision": None,
        "pending_action": None,
        "target_id": None,
        "candidates": [],
        "can_confirm": False,
        "can_cancel": False,
        "message": "你好，我可以帮你查询、统计、新增、修改或删除招聘岗位。",
        "missing_fields": [],
        "state_version": 0,
        "draft_revision": 0,
        "preview_revision": None,
    }


def _reset(message: str) -> AgentState:
    state = initial_state()
    state["message"] = message
    return state


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = []
    return [str(item) for item in value or []]


def _row_draft(row: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        if key not in JobFields.model_fields or value in (None, ""):
            continue
        result[key] = _as_list(value) if key in JSON_FIELDS else value
    return JobFields.model_validate(result).model_dump(exclude_none=True)


def missing_create_fields(draft: dict[str, Any]) -> list[str]:
    missing = [field for field in REQUIRED_CREATE_FIELDS if not draft.get(field)]
    if not any(draft.get(field) for field in CREATE_CONTENT_FIELDS):
        missing.append("job_content")
    return missing


_ENUM_LABELS = {
    "internship": "实习招聘", "campus": "校园招聘", "experienced": "社会招聘",
    "mixed": "不限", "unknown": "未明确", "full_time": "全职",
    "part_time": "兼职", "full_or_part_time": "全职或兼职", "contract": "合同制",
    "temporary": "临时用工", "onsite": "现场办公", "remote": "远程办公",
    "hybrid": "混合办公", "month": "月", "year": "年", "day": "日", "hour": "小时",
}


def _display_value(field: str, value: Any) -> str:
    if value is None or value == "":
        return "未填写"
    if field in {"experience_min_months", "experience_max_months"}:
        return f"{value}个月"
    if field == "education_min_level":
        return {4: "本科及以上", 5: "硕士及以上", 6: "博士及以上"}.get(int(value), str(value))
    return _ENUM_LABELS.get(str(value), str(value))


def render_preview(draft: dict[str, Any], provenance: dict[str, Any] | None = None) -> str:
    del provenance  # provenance is retained for audit, never exposed as internal JSON.
    scalar_order = (
        "company_name", "title", "recruitment", "employment", "city", "work_address",
        "work_mode", "education_min_level", "experience_min_months", "experience_max_months",
        "salary_min", "salary_max", "salary_currency", "salary_period", "source_url",
    )
    lines = ["### 岗位草稿"]
    for field in scalar_order:
        if field in draft:
            lines.append(f"- {FIELD_LABELS[field]}：{_display_value(field, draft[field])}")
    for field in ("responsibilities_json", "requirements_json", "skills_json", "certificates_json", "benefits_json"):
        values = _as_list(draft.get(field))
        if not values:
            continue
        lines.append(f"\n**{FIELD_LABELS[field]}**")
        lines.extend(f"- {value}" for value in values)
    return "\n".join(lines)


def build_context(state: AgentState) -> dict[str, Any]:
    draft = deepcopy(state.get("draft", {}))
    return {
        "phase": state.get("phase", Phase.IDLE.value),
        "draft": draft,
        "provenance": deepcopy(state.get("provenance", {})),
        "last_assistant_question": state.get("last_assistant_question"),
        "pending_decision": deepcopy(state.get("pending_decision")),
        "recent_messages": deepcopy(state.get("recent_messages", [])),
        "ready_to_save": not missing_create_fields(draft),
        "pending_action": state.get("pending_action"),
        "candidate_count": len(state.get("candidates", [])),
        "state_version": state.get("state_version", 0),
        "draft_revision": state.get("draft_revision", 0),
    }


def _append_recent(state: AgentState, user_text: str, assistant_text: str) -> None:
    messages = [
        *state.get("recent_messages", []),
        {"role": "user", "content": user_text[:2_000]},
        {"role": "assistant", "content": assistant_text[:2_000]},
    ][-10:]
    while sum(len(item["content"]) for item in messages) > 8_000 and len(messages) > 2:
        messages.pop(0)
    state["recent_messages"] = messages


_AFFIRMATIVE_RE = re.compile(
    r"^(?:确认|确认了|我确认|确认写入|确定|是|是的|对|对的|可以|没错|好|好的|同意|保存吧)$"
)
_NEGATIVE_RE = re.compile(r"^(?:不是|不对|否|不要|不同意|拒绝)$")


def _update_summary(patch: DraftPatch, mentioned_fields: list[str]) -> str:
    parts: list[str] = []
    for field in dict.fromkeys(mentioned_fields):
        if field in patch.set_fields:
            parts.append(f"{FIELD_LABELS.get(field, field)}＝{_display_value(field, patch.set_fields[field])}")
        elif field in JSON_FIELDS:
            count = len(patch.append_items.get(field, []))
            if count:
                parts.append(f"{FIELD_LABELS.get(field, field)}新增{count}条")
    return "；".join(parts)


def _render_query_detail(detail: dict[str, Any]) -> str:
    lines = ["### 岗位详情"]
    order = (
        "job_id", "company_name", "title", "city", "work_address", "recruitment",
        "employment", "work_mode", "education_min_level", "experience_min_months",
        "experience_max_months", "salary_min", "salary_max", "salary_currency",
        "salary_period", "responsibilities_json", "requirements_json", "skills_json",
        "certificates_json", "benefits_json", "source_url",
    )
    for field in order:
        if field not in detail:
            continue
        value = detail[field]
        if field in JSON_FIELDS:
            lines.append(f"\n**{FIELD_LABELS[field]}**")
            lines.extend(f"- {item}" for item in _as_list(value))
        else:
            lines.append(f"- {FIELD_LABELS[field]}：{_display_value(field, value)}")
    return "\n".join(lines)


class JobCsvAgent:
    def __init__(self, repository: CsvJobRepository, interpreter: Interpreter) -> None:
        self.repository = repository
        self.interpreter = interpreter

    def _finish(
        self, state: AgentState, user_text: str, message: str, *, question: str | None = None,
    ) -> AgentState:
        state["message"] = message
        state["last_assistant_question"] = question
        _append_recent(state, user_text, message)
        return state

    def _draft_response(
        self, state: AgentState, text: str, clarification: str | None = None,
        update_summary: str = "",
    ) -> AgentState:
        draft = state.get("draft", {})
        missing = missing_create_fields(draft)
        state["missing_fields"] = missing
        state["can_cancel"] = True
        pending = state.get("pending_decision")
        if pending:
            clarification = str(pending.get("prompt") or clarification or "请确认这项内容。")
        prefix = f"本轮已更新：{update_summary}。\n" if update_summary else ""
        preview = render_preview(draft, state.get("provenance", {}))
        state["preview_revision"] = state.get("draft_revision", 0)
        if missing or pending:
            state["phase"] = Phase.EDITING.value if state.get("pending_action") == "update" else Phase.CREATING.value
            state["can_confirm"] = False
            if clarification:
                question = clarification
            else:
                labels = "、".join(FIELD_LABELS.get(field, field) for field in missing)
                question = f"还缺少：{labels}。请直接补充原文信息。"
            return self._finish(state, text, f"{prefix}{preview}\n{question}", question=question)
        state["phase"] = Phase.CONFIRMING.value
        state["can_confirm"] = True
        return self._finish(
            state, text,
            f"{prefix}{preview}\n当前草稿已达到保存条件，请核对后使用当前版本确认一次。",
        )

    def _show_current(self, state: AgentState, text: str) -> AgentState:
        missing = missing_create_fields(state.get("draft", {}))
        state["missing_fields"] = missing
        state["can_confirm"] = (
            not missing and not state.get("pending_decision")
            and state.get("pending_action") in {"create", "update"}
            and state.get("preview_revision") == state.get("draft_revision")
        )
        return self._finish(state, text, "当前完整草稿：\n" + render_preview(state.get("draft", {})))

    def _select(self, state: AgentState, index: int, text: str) -> AgentState:
        candidates = state.get("candidates", [])
        if index < 1 or index > len(candidates):
            return self._finish(state, text, "选择序号超出范围，请重新选择。")
        row = candidates[index - 1]
        state["target_id"] = row["job_id"]
        state["candidates"] = []
        if state.get("pending_action") == "delete":
            state["phase"] = Phase.CONFIRMING.value
            state["can_confirm"] = True
            state["can_cancel"] = True
            state["preview_revision"] = state.get("draft_revision", 0)
            return self._finish(state, text, f"将删除 {row.get('company_name')} 的 {row.get('title')} 岗位。")
        state["draft"] = _row_draft(row)
        state["provenance"] = {}
        state["draft_revision"] = state.get("draft_revision", 0) + 1
        state["preview_revision"] = None
        state["phase"] = Phase.EDITING.value
        state["can_confirm"] = False
        state["can_cancel"] = True
        return self._finish(state, text, "已选择岗位，请说明要修改的字段或原文条目。")

    def _find_target(self, state: AgentState, query: str, action: str, text: str) -> AgentState:
        rows = self.repository.search(query)
        state["pending_action"] = action
        state["can_cancel"] = True
        if not rows:
            return self._finish(state, text, "没有找到对应岗位，请补充公司名称或完整岗位名称。")
        state["candidates"] = rows
        if len(rows) == 1:
            return self._select(state, 1, text)
        options = "\n".join(
            f"{index}. 公司：{row.get('company_name')}；岗位：{row.get('title')}；城市：{row.get('city')}"
            for index, row in enumerate(rows, 1)
        )
        return self._finish(state, text, f"找到多个岗位，请选择：\n{options}")

    def _confirm(self, state: AgentState, text: str) -> AgentState:
        if not state.get("can_confirm"):
            return self._finish(state, text, "当前没有等待确认的写入操作。")
        if state.get("preview_revision") != state.get("draft_revision"):
            state["can_confirm"] = False
            return self._finish(state, text, "草稿已变化，请先查看最新预览后再确认。")
        action = state.get("pending_action")
        if action == "create":
            missing = missing_create_fields(state.get("draft", {}))
            if missing:
                return self._draft_response(state, text)
            row = self.repository.create(state["draft"])
            message = f"已保存岗位：{row.get('company_name')} - {row.get('title')}。"
        elif action == "update" and state.get("target_id"):
            current = _row_draft(self.repository.get(state["target_id"]))
            clear_fields = [field for field in current if field not in state.get("draft", {})]
            _, row = self.repository.update(state["target_id"], state["draft"], clear_fields)
            message = f"已更新岗位：{row.get('company_name')} - {row.get('title')}。"
        elif action == "delete" and state.get("target_id"):
            row = self.repository.delete(state["target_id"])
            message = f"已删除岗位：{row.get('company_name')} - {row.get('title')}。"
        else:
            return self._finish(state, text, "当前没有可执行的确认操作。")
        return _reset(message)

    def _query(self, state: AgentState, text: str, plan: QueryPlan) -> AgentState:
        filters = plan.filters()
        if plan.mode == "count":
            count = self.repository.count(filters)
            return self._finish(state, text, f"当前共有 {count} 条岗位记录。")
        if plan.mode == "list":
            rows = self.repository.list_summaries(filters, plan.limit)
            if not rows:
                return self._finish(state, text, "没有找到符合条件的岗位记录。")
            lines = [
                f"{index}. 公司：{row.get('company_name')}；岗位：{row.get('title')}；城市：{row.get('city')}"
                for index, row in enumerate(rows, 1)
            ]
            return self._finish(state, text, "\n".join(lines))
        ids = self.repository.find_job_ids(filters, plan.limit)
        if not ids:
            return self._finish(state, text, "没有找到符合条件的岗位记录。")
        if len(ids) > 1:
            rows = self.repository.list_summaries(filters, plan.limit)
            lines = [
                f"{index}. 公司：{row.get('company_name')}；岗位：{row.get('title')}；城市：{row.get('city')}"
                for index, row in enumerate(rows, 1)
            ]
            return self._finish(state, text, "找到多条记录，请补充条件后查看详情：\n" + "\n".join(lines))
        return self._finish(state, text, _render_query_detail(self.repository.get_public_detail(ids[0])))

    def _evaluate_patch(
        self, command: StructuredCommand, text: str, before_draft: dict[str, Any],
        before_provenance: dict[str, Any], mentioned_fields: list[str] | None = None,
    ) -> tuple[DraftPatch, dict[str, Any], dict[str, Any], list[str]]:
        patch = normalize_patch(command.patch, text)
        declared = mentioned_fields if mentioned_fields is not None else command.mentioned_fields
        touched = (
            set(patch.set_fields) | set(patch.append_items) | set(patch.remove_items)
            | set(patch.clear_fields) | {item.field for item in patch.replace_items}
        )
        mentioned = list(dict.fromkeys([*declared, *sorted(touched)]))
        patch = sanitize_model_patch(
            patch, text, mentioned, command.evidence_spans, command.ignored_fragments,
        )
        after_draft, after_provenance = apply_draft_patch(before_draft, before_provenance, patch)
        failures = unapplied_fields(before_draft, after_draft, patch, mentioned)
        return patch, after_draft, after_provenance, failures

    def _apply_with_retry(
        self, state: AgentState, command: StructuredCommand, text: str,
    ) -> tuple[AgentState, StructuredCommand, DraftPatch] | AgentState:
        before_draft = deepcopy(state.get("draft", {}))
        before_provenance = deepcopy(state.get("provenance", {}))
        patch, after, provenance, failures = self._evaluate_patch(
            command, text, before_draft, before_provenance,
        )
        used_command = command
        if failures:
            retry_context = build_context(state)
            retry_context["retry"] = {
                "reason": "mentioned_fields_not_applied", "unapplied_fields": failures,
                "instruction": "重新抽取本轮原文，必须返回能落实这些字段的 patch 和 evidence。",
            }
            try:
                retry = self.interpreter.extract(text, retry_context)
                patch, after, provenance, failures = self._evaluate_patch(
                    retry, text, before_draft, before_provenance, command.mentioned_fields,
                )
                used_command = retry
            except (LLMConfigurationError, LLMServiceError, ValidationError):
                pass
        if failures:
            state["draft"] = before_draft
            state["provenance"] = before_provenance
            state["can_confirm"] = False
            state["preview_revision"] = None
            labels = "、".join(FIELD_LABELS.get(field, field) for field in failures)
            return self._finish(
                state, text, f"本轮字段未成功应用，尚未保存。未应用字段：{labels}。",
            )
        changes = changed_fields(before_draft, after)
        state["draft"] = after
        state["provenance"] = provenance
        if changes:
            state["draft_revision"] = state.get("draft_revision", 0) + 1
        state["preview_revision"] = None
        return state, used_command, patch

    def handle(
        self, state: AgentState, text: str, forced_command: dict[str, Any] | None = None,
    ) -> AgentState:
        current: AgentState = deepcopy(state)
        text = " ".join(text.split())

        if forced_command:
            if forced_command.get("intent") == "confirm":
                return self._confirm(current, text)
            if forced_command.get("intent") == "cancel":
                return _reset("已取消当前操作，未写入任何数据。")
            if forced_command.get("selection_index") is not None:
                return self._select(current, int(forced_command["selection_index"]), text)

        if _AFFIRMATIVE_RE.fullmatch(text):
            pending = current.get("pending_decision")
            if pending:
                patch = DraftPatch.model_validate(pending["patch"])
                before = deepcopy(current.get("draft", {}))
                draft, provenance = apply_draft_patch(
                    before, current.get("provenance", {}), patch,
                )
                current["draft"] = draft
                current["provenance"] = provenance
                if changed_fields(before, draft):
                    current["draft_revision"] = current.get("draft_revision", 0) + 1
                mentioned = list(pending.get("mentioned_fields", []))
                current["pending_decision"] = None
                current["preview_revision"] = None
                return self._draft_response(
                    current, text, update_summary=_update_summary(patch, mentioned),
                )
            return self._confirm(current, text)
        if _NEGATIVE_RE.fullmatch(text) and current.get("pending_decision"):
            current["pending_decision"] = None
            return self._draft_response(current, text)
        if text in {"取消", "取消操作", "算了"}:
            return _reset("已取消当前操作，未写入任何数据。")
        if re.search(r"(?:现在|当前).*(?:字段|草稿)|字段是什么", text):
            return self._show_current(current, text)

        context = build_context(current)
        try:
            command = self.interpreter.extract(text, context)
        except (LLMConfigurationError, LLMServiceError) as exc:
            return self._finish(current, text, str(exc))
        except ValidationError as exc:
            return self._finish(current, text, f"模型返回的结构化结果无效，草稿已保留：{exc}")

        if command.intent == "confirm":
            return self._confirm(current, text)
        if command.intent == "cancel":
            return _reset("已取消当前操作，未写入任何数据。")
        if command.intent in {"search", "count", "detail"}:
            mode = {"count": "count", "detail": "detail"}.get(command.intent, "list")
            plan = command.query_plan or QueryPlan(mode=mode)
            if command.intent != "search" and plan.mode != mode:
                plan = plan.model_copy(update={"mode": mode})
            return self._query(current, text, plan)
        if command.intent == "delete":
            return self._find_target(current, command.search_query or text, "delete", text)
        if command.intent == "update" and current.get("pending_action") not in {"create", "update"}:
            if command.search_query:
                return self._find_target(current, command.search_query, "update", text)
        if command.intent in {"help", "conversation", "unsupported", "unknown"}:
            return self._finish(
                current, text,
                command.natural_reply or "我可以继续补充当前草稿，或查询、修改已有岗位。",
            )

        if command.intent == "create":
            if current.get("pending_action") != "create" or command.task_relation == "start_new":
                current = initial_state()
            current["pending_action"] = "create"
            current["phase"] = Phase.CREATING.value
        elif command.intent != "update" or current.get("pending_action") not in {"create", "update"}:
            return self._finish(current, text, "请说明要新增、修改或查询的岗位。")

        current["pending_decision"] = None
        applied = self._apply_with_retry(current, command, text)
        if isinstance(applied, dict):
            return applied
        current, used_command, patch = applied
        if used_command.pending_decision is not None:
            decision_patch = normalize_patch(used_command.pending_decision.patch, text)
            decision_patch = sanitize_model_patch(
                decision_patch, text, used_command.mentioned_fields,
                used_command.evidence_spans, used_command.ignored_fragments,
            )
            if (
                decision_patch.set_fields or decision_patch.append_items
                or decision_patch.replace_items or decision_patch.remove_items
                or decision_patch.clear_fields
            ):
                current["pending_decision"] = {
                    **used_command.pending_decision.model_dump(mode="json"),
                    "patch": decision_patch.model_dump(mode="json"),
                    "mentioned_fields": used_command.mentioned_fields,
                }
        clarification = used_command.clarification_question if used_command.requires_clarification else None
        return self._draft_response(
            current, text, clarification,
            update_summary=_update_summary(patch, command.mentioned_fields),
        )
