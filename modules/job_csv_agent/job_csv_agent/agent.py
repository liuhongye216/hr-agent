from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Protocol, TypedDict

from pydantic import ValidationError

from .llm import LLMConfigurationError, LLMServiceError
from .normalization import infer_atomic_mentions, normalize_patch
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
    storage_valid: bool
    extraction_complete: bool
    unresolved_fragments: list[dict[str, Any]]
    requested_fields: list[str]
    source_messages: list[dict[str, str]]
    current_message_id: str | None


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
        "storage_valid": False,
        "extraction_complete": True,
        "unresolved_fragments": [],
        "requested_fields": [],
        "source_messages": [],
        "current_message_id": None,
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
    if field in {"experience_min_months", "experience_max_months", "internship_min_months"}:
        return f"{value}个月"
    if field == "onsite_days_per_week":
        return f"每周{value}天"
    if field == "education_min_level":
        return {4: "本科及以上", 5: "硕士及以上", 6: "博士及以上"}.get(int(value), str(value))
    return _ENUM_LABELS.get(str(value), str(value))


def render_preview(draft: dict[str, Any], provenance: dict[str, Any] | None = None) -> str:
    del provenance  # provenance is retained for audit, never exposed as internal JSON.
    scalar_order = (
        "company_name", "title", "recruitment", "employment", "city", "work_address",
        "work_mode", "education_min_level", "experience_min_months", "experience_max_months",
        "internship_min_months", "onsite_days_per_week", "salary_min", "salary_max",
        "salary_currency", "salary_period", "source_url",
    )
    lines = ["### 岗位草稿"]
    for field in scalar_order:
        if field in draft:
            lines.append(f"- {FIELD_LABELS[field]}：{_display_value(field, draft[field])}")
    for field in (
        "responsibilities_json", "requirements_json", "preferred_requirements_json",
        "not_required_requirements_json", "skills_json", "student_status_json",
        "certificates_json", "benefits_json",
    ):
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
        "storage_valid": not missing_create_fields(draft),
        "extraction_complete": state.get("extraction_complete", True),
        "unresolved_fragments": deepcopy(state.get("unresolved_fragments", [])),
        "requested_fields": list(state.get("requested_fields", [])),
        "source_messages": deepcopy(state.get("source_messages", [])),
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


def _append_source_message(state: AgentState, user_text: str) -> None:
    message_id = state.get("current_message_id") or f"user_{len(state.get('source_messages', [])) + 1}"
    messages = list(state.get("source_messages", []))
    if not messages or messages[-1].get("id") != message_id:
        messages.append({"id": message_id, "content": user_text[:10_000]})
    state["source_messages"] = messages[-20:]


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
        "experience_max_months", "internship_min_months", "onsite_days_per_week",
        "salary_min", "salary_max", "salary_currency", "salary_period",
        "responsibilities_json", "requirements_json", "preferred_requirements_json",
        "not_required_requirements_json", "skills_json", "student_status_json",
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


def _patch_for_fields(patch: DraftPatch, fields: set[str]) -> DraftPatch:
    data = patch.model_dump(mode="python")
    data["set_fields"] = {key: value for key, value in data["set_fields"].items() if key in fields}
    data["set_sources"] = {key: value for key, value in data["set_sources"].items() if key in fields}
    data["append_items"] = {key: value for key, value in data["append_items"].items() if key in fields}
    data["remove_items"] = {key: value for key, value in data["remove_items"].items() if key in fields}
    data["replace_items"] = [item for item in data["replace_items"] if item["field"] in fields]
    data["clear_fields"] = [field for field in data["clear_fields"] if field in fields]
    return DraftPatch.model_validate(data)


def _merge_patches(first: DraftPatch, second: DraftPatch) -> DraftPatch:
    data = first.model_dump(mode="python")
    other = second.model_dump(mode="python")
    data["set_fields"].update(other["set_fields"])
    data["set_sources"].update(other["set_sources"])
    for section in ("append_items", "remove_items"):
        for field, values in other[section].items():
            data[section].setdefault(field, []).extend(values)
    data["replace_items"].extend(other["replace_items"])
    data["clear_fields"] = list(dict.fromkeys([*data["clear_fields"], *other["clear_fields"]]))
    return DraftPatch.model_validate(data)


_REPAIR_RE = re.compile(r"(?:再|重新|仔细).*(?:看看|识别|提取)|(?:漏了|遗漏)|(?:学历|技能|经验|要求|职责)信息?呢")


def _repair_fields(text: str) -> list[str]:
    mapping = {
        "学历": ["education_min_level", "student_status_json"],
        "技能": ["skills_json"],
        "经验": ["experience_min_months", "experience_max_months"],
        "要求": ["requirements_json", "preferred_requirements_json", "not_required_requirements_json"],
        "职责": ["responsibilities_json"],
    }
    return list(dict.fromkeys(field for word, fields in mapping.items() if word in text for field in fields))


def _repair_sources(state: AgentState) -> list[dict[str, str]]:
    by_id = {item.get("id", ""): item for item in state.get("source_messages", [])}
    selected: list[dict[str, str]] = []
    for fragment in state.get("unresolved_fragments", []):
        message_id = str(fragment.get("source_message_id") or "")
        if message_id in by_id and by_id[message_id] not in selected:
            selected.append(by_id[message_id])
    if not selected:
        selected = [
            item for item in reversed(state.get("source_messages", []))
            if len(item.get("content", "")) >= 12 and not _REPAIR_RE.search(item.get("content", ""))
        ][:2]
        selected.reverse()
    return selected


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
        _append_source_message(state, user_text)
        return state

    def _draft_response(
        self, state: AgentState, text: str, clarification: str | None = None,
        update_summary: str = "", clarification_fields: list[str] | None = None,
    ) -> AgentState:
        draft = state.get("draft", {})
        missing = missing_create_fields(draft)
        storage_valid = not missing
        extraction_complete = state.get("extraction_complete", True)
        unresolved = state.get("unresolved_fragments", [])
        if unresolved:
            extraction_complete = False
        state["storage_valid"] = storage_valid
        state["extraction_complete"] = extraction_complete
        state["missing_fields"] = missing
        state["can_cancel"] = True
        pending = state.get("pending_decision")
        if pending:
            clarification = str(pending.get("prompt") or clarification or "请确认这项内容。")
        prefix = f"本轮已更新：{update_summary}。\n" if update_summary else ""
        preview = render_preview(draft, state.get("provenance", {}))
        state["preview_revision"] = state.get("draft_revision", 0)
        if missing or pending or not extraction_complete:
            state["phase"] = Phase.EDITING.value if state.get("pending_action") == "update" else Phase.CREATING.value
            state["can_confirm"] = False
            if clarification:
                question = clarification
            elif unresolved:
                fragments = "；".join(str(item.get("text", "")) for item in unresolved[:3])
                question = f"还有信息尚未完整归类：{fragments}。请补充说明，或指出需要我重新识别的字段。"
            else:
                labels = "、".join(FIELD_LABELS.get(field, field) for field in missing)
                question = f"还缺少：{labels}。请直接补充原文信息。"
            state["requested_fields"] = list(dict.fromkeys(clarification_fields or missing))
            status = ""
            if storage_valid and not extraction_complete:
                status = "\n\n草稿已具备最低保存条件，但抽取尚未完整。"
            return self._finish(
                state, text, f"{prefix}{preview}{status}\n\n{question}", question=question,
            )
        state["phase"] = Phase.CONFIRMING.value
        state["can_confirm"] = True
        state["requested_fields"] = []
        return self._finish(
            state, text,
            f"{prefix}{preview}\n\n当前草稿已完整抽取并达到保存条件，请核对后使用当前版本确认一次。",
        )

    def _show_current(self, state: AgentState, text: str) -> AgentState:
        missing = missing_create_fields(state.get("draft", {}))
        state["missing_fields"] = missing
        state["can_confirm"] = (
            not missing and not state.get("pending_decision")
            and state.get("extraction_complete", True)
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
        source_messages: dict[str, str] | None = None,
        allowed_fields: set[str] | None = None,
        source_message_id: str | None = None,
    ) -> tuple[DraftPatch, dict[str, Any], dict[str, Any], list[str]]:
        inferred = infer_atomic_mentions(text, source_message_id)
        patch = normalize_patch(command.patch, text, [*command.mentions, *inferred])
        if allowed_fields is not None:
            patch = _patch_for_fields(patch, allowed_fields)
        declared = mentioned_fields if mentioned_fields is not None else command.mentioned_fields
        touched = (
            set(patch.set_fields) | set(patch.append_items) | set(patch.remove_items)
            | set(patch.clear_fields) | {item.field for item in patch.replace_items}
        )
        mentioned = list(dict.fromkeys([*declared, *sorted(touched)]))
        if allowed_fields is not None:
            mentioned = [field for field in mentioned if field in allowed_fields]
        patch = sanitize_model_patch(
            patch, text, mentioned, command.evidence_spans, command.ignored_fragments,
            source_messages,
        )
        after_draft, after_provenance = apply_draft_patch(before_draft, before_provenance, patch)
        failures = unapplied_fields(before_draft, after_draft, patch, mentioned)
        return patch, after_draft, after_provenance, failures

    def _apply_with_retry(
        self, state: AgentState, command: StructuredCommand, text: str,
        source_messages: dict[str, str] | None = None,
    ) -> tuple[AgentState, StructuredCommand, DraftPatch, list[str]]:
        before_draft = deepcopy(state.get("draft", {}))
        before_provenance = deepcopy(state.get("provenance", {}))
        patch, after, provenance, failures = self._evaluate_patch(
            command, text, before_draft, before_provenance,
            source_messages=source_messages,
            source_message_id=state.get("current_message_id"),
        )
        used_command = command
        if failures:
            touched = (
                set(patch.set_fields) | set(patch.append_items) | set(patch.remove_items)
                | set(patch.clear_fields) | {item.field for item in patch.replace_items}
            )
            successful_patch = _patch_for_fields(patch, touched - set(failures))
            partial_draft, partial_provenance = apply_draft_patch(
                before_draft, before_provenance, successful_patch,
            )
            retry_context = build_context(state)
            retry_context["draft"] = deepcopy(partial_draft)
            retry_context["retry"] = {
                "reason": "mentioned_fields_not_applied", "unapplied_fields": failures,
                "instruction": "只重新抽取未应用字段；已成功字段会保留。证据可以引用 source_messages。",
            }
            try:
                retry = self.interpreter.extract(text, retry_context)
                retry_patch, retry_after, retry_provenance, retry_failures = self._evaluate_patch(
                    retry, text, partial_draft, partial_provenance, failures,
                    source_messages, set(failures), state.get("current_message_id"),
                )
                patch = _merge_patches(successful_patch, retry_patch)
                after, provenance, failures = retry_after, retry_provenance, retry_failures
                used_command = retry.model_copy(update={
                    "unresolved_fragments": [
                        *command.unresolved_fragments, *retry.unresolved_fragments,
                    ],
                    "extraction_complete": (
                        command.extraction_complete and retry.extraction_complete
                    ),
                })
            except (LLMConfigurationError, LLMServiceError, ValidationError):
                patch = successful_patch
                after, provenance = partial_draft, partial_provenance
        changes = changed_fields(before_draft, after)
        state["draft"] = after
        state["provenance"] = provenance
        if changes:
            state["draft_revision"] = state.get("draft_revision", 0) + 1
        state["preview_revision"] = None
        return state, used_command, patch, failures

    def handle(
        self, state: AgentState, text: str, forced_command: dict[str, Any] | None = None,
    ) -> AgentState:
        current: AgentState = deepcopy(state)
        text = " ".join(text.split())
        current["current_message_id"] = f"user_{len(current.get('source_messages', [])) + 1}"

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

        repair_requested = bool(_REPAIR_RE.search(text))
        repair_sources = _repair_sources(current) if repair_requested else []
        semantic_text = "\n".join(item["content"] for item in repair_sources) if repair_sources else text
        source_messages = {
            str(item.get("id")): str(item.get("content", ""))
            for item in current.get("source_messages", [])
        }
        source_messages[str(current["current_message_id"])] = text
        context = build_context(current)
        context["current_message_id"] = current["current_message_id"]
        if repair_requested:
            context["repair"] = {
                "requested_fields": _repair_fields(text),
                "source_message_ids": [item["id"] for item in repair_sources],
                "instruction": "重新分析这些历史用户原文；evidence_spans 使用对应 source_message_id。",
            }
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
            if re.search(r"招聘|招一名|招募", semantic_text) and (
                command.patch.set_fields or command.mentions or infer_atomic_mentions(semantic_text)
            ):
                command = command.model_copy(update={"intent": "create", "task_relation": "start_new"})
        if repair_requested and current.get("pending_action") in {"create", "update"} and command.intent in {
            "help", "conversation", "unsupported", "unknown",
        }:
            command = command.model_copy(update={"intent": "update", "task_relation": "continue_current"})
        if command.intent in {"help", "conversation", "unsupported", "unknown"} and re.search(
            r"招聘|招一名|招募", semantic_text,
        ) and {"company_name", "title"}.issubset(command.patch.set_fields):
            command = command.model_copy(update={"intent": "create", "task_relation": "start_new"})
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
        requested_before = set(current.get("requested_fields", []))
        previous_unresolved = list(current.get("unresolved_fragments", []))
        current, used_command, patch, failures = self._apply_with_retry(
            current, command, semantic_text, source_messages,
        )
        touched = (
            set(patch.set_fields) | set(patch.append_items) | set(patch.remove_items)
            | set(patch.clear_fields) | {item.field for item in patch.replace_items}
        )
        unresolved = [item.model_dump(mode="json") for item in used_command.unresolved_fragments]
        unresolved.extend({
            "text": f"字段“{FIELD_LABELS.get(field, field)}”未能应用",
            "reason": "本轮结构化结果没有产生可验证的字段更新",
            "suggested_fields": [field],
            "source_message_id": current.get("current_message_id"),
        } for field in failures)
        if not repair_requested:
            unresolved = [
                item for item in previous_unresolved
                if not (set(item.get("suggested_fields", [])) & touched)
            ] + unresolved
        deduplicated: list[dict[str, Any]] = []
        seen_unresolved: set[tuple[str, str]] = set()
        for item in unresolved:
            key = (str(item.get("text", "")), str(item.get("source_message_id", "")))
            if key not in seen_unresolved:
                seen_unresolved.add(key)
                deduplicated.append(item)
        unresolved = deduplicated
        current["unresolved_fragments"] = unresolved
        current["extraction_complete"] = used_command.extraction_complete and not unresolved
        if used_command.pending_decision is not None:
            decision_patch = normalize_patch(used_command.pending_decision.patch, text)
            decision_patch = sanitize_model_patch(
                decision_patch, semantic_text, used_command.mentioned_fields,
                used_command.evidence_spans, used_command.ignored_fragments,
                source_messages,
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
        clarification_fields = used_command.clarification_fields
        remaining_missing = set(missing_create_fields(current.get("draft", {})))
        remaining_requested = {
            field for field in requested_before
            if field in remaining_missing or (
                "job_content" in remaining_missing and field in CREATE_CONTENT_FIELDS
            )
        }
        if remaining_requested and not (requested_before & touched):
            still_needed = "、".join(FIELD_LABELS.get(field, field) for field in remaining_requested)
            clarification = f"本轮其他信息已记录；还需要补充：{still_needed}。"
            clarification_fields = list(remaining_requested)
        return self._draft_response(
            current, text, clarification,
            update_summary=_update_summary(
                patch, list(dict.fromkeys([*command.mentioned_fields, *sorted(touched)])),
            ),
            clarification_fields=clarification_fields,
        )
