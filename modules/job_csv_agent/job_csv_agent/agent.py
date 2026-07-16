from __future__ import annotations

import json
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from .jd_text2sql_adapter import JDText2SQLAdapter
from .schemas import (
    CREATE_CONTENT_FIELDS, FIELD_LABELS, JSON_FIELDS, REQUIRED_CREATE_FIELDS, JobFields, Phase,
    SemanticFact, StructuredCommand,
)
from .repository import CsvJobRepository
from .semantics import persisted_semantic_facts, reconcile_command_semantics


class Interpreter(Protocol):
    def extract(self, text: str, context: dict[str, Any]) -> StructuredCommand: ...


class AgentState(TypedDict, total=False):
    phase: str
    user_input: str
    forced_command: dict[str, Any] | None
    draft: dict[str, Any]
    semantic_facts: list[dict[str, Any]]
    clarification_questions: list[str]
    target_id: str | None
    candidates: list[dict[str, str]]
    pending_action: str | None
    pending_fields: dict[str, Any]
    pending_clear_fields: list[str]
    pending_semantic_facts: list[dict[str, Any]]
    message: str
    missing_fields: list[str]
    can_confirm: bool
    can_cancel: bool


def initial_state() -> AgentState:
    return {
        "phase": Phase.IDLE.value,
        "draft": {},
        "semantic_facts": [],
        "clarification_questions": [],
        "target_id": None,
        "candidates": [],
        "pending_action": None,
        "pending_fields": {},
        "pending_clear_fields": [],
        "pending_semantic_facts": [],
        "message": "请告诉我您要新建、修改、删除还是查询招聘岗位。",
        "missing_fields": [],
        "can_confirm": False,
        "can_cancel": False,
    }


def _clean_turn() -> dict[str, Any]:
    return {"user_input": "", "forced_command": None}


def _reset(message: str) -> AgentState:
    state = initial_state()
    state["message"] = message
    return state


def _display_value(field: str, value: Any) -> str:
    if value in (None, ""):
        return "（空）"
    if field in JSON_FIELDS and isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
    if isinstance(value, list):
        return "、".join(str(item) for item in value) or "（空列表）"
    return str(value)


def _field_lines(fields: dict[str, Any]) -> str:
    if not fields:
        return "（本轮未识别到字段）"
    return "\n".join(
        f"- {FIELD_LABELS.get(key, key)}：{_display_value(key, value)}" for key, value in fields.items()
    )


def _merge_facts(
    existing: list[dict[str, Any]], incoming: list[SemanticFact],
) -> list[dict[str, Any]]:
    result = list(existing)
    seen = {
        (item.get("value", "").casefold(), item.get("category"), item.get("source_type"))
        for item in result
    }
    for fact in incoming:
        item = fact.model_dump(mode="json")
        key = (fact.value.casefold(), fact.category, fact.source_type)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _semantic_notes(facts: list[dict[str, Any]], questions: list[str]) -> str:
    unknown = [item["value"] for item in facts if item.get("category") == "unknown"]
    inferred = [item["value"] for item in facts if item.get("source_type") == "inferred"]
    sections: list[str] = []
    if unknown:
        sections.append("⚠️ 待澄清（不会写入 CSV）：" + "、".join(unknown))
    if inferred:
        sections.append("💡 待确认能力建议（inferred，不会写入 CSV）：" + "、".join(inferred))
    if questions:
        sections.append("建议确认：\n" + "\n".join(f"- {item}" for item in questions))
    return "\n\n" + "\n\n".join(sections) if sections else ""


class JobCsvAgent:
    def __init__(
        self,
        repository: CsvJobRepository,
        interpreter: Interpreter,
        query_adapter: JDText2SQLAdapter | None = None,
    ) -> None:
        self.repository = repository
        self.interpreter = interpreter
        self.query_adapter = query_adapter
        builder = StateGraph(AgentState)
        builder.add_node("idle", self._idle)
        builder.add_node("creating", self._creating)
        builder.add_node("editing", self._editing)
        builder.add_node("confirming", self._confirming)
        builder.add_conditional_edges(START, self._route, {
            "idle": "idle", "creating": "creating", "editing": "editing", "confirming": "confirming",
        })
        for node in ("idle", "creating", "editing", "confirming"):
            builder.add_edge(node, END)
        self.graph = builder.compile()

    @staticmethod
    def _route(state: AgentState) -> str:
        return {
            Phase.CREATING.value: "creating",
            Phase.EDITING.value: "editing",
            Phase.CONFIRMING.value: "confirming",
        }.get(state.get("phase", Phase.IDLE.value), "idle")

    def handle(
        self, state: AgentState | None, text: str, forced_command: dict[str, Any] | None = None,
    ) -> AgentState:
        current = {**initial_state(), **(state or {})}
        current.update({"user_input": text.strip(), "forced_command": forced_command})
        return self.graph.invoke(current)

    def _command(self, state: AgentState) -> StructuredCommand:
        if state.get("forced_command") is not None:
            command = StructuredCommand.model_validate(state["forced_command"])
            return reconcile_command_semantics(command, state.get("user_input", ""))
        text = state.get("user_input", "").strip()
        normalized = text.casefold().replace(" ", "")
        if normalized in {"确认", "确认写入", "是", "好的", "yes", "ok"}:
            return StructuredCommand(intent="confirm")
        if normalized in {"取消", "放弃", "不要了", "cancel"}:
            return StructuredCommand(intent="cancel")
        if normalized in {"继续修改", "修改", "我还要改"}:
            return StructuredCommand(intent="update")
        if normalized.isdigit():
            return StructuredCommand(intent="unknown", selection_index=int(normalized))
        context = {
            "phase": state.get("phase"),
            "draft": state.get("draft", {}),
            "target_id": state.get("target_id"),
            "candidate_count": len(state.get("candidates", [])),
            "pending_action": state.get("pending_action"),
        }
        return reconcile_command_semantics(self.interpreter.extract(text, context), text)

    @staticmethod
    def _missing(draft: dict[str, Any]) -> list[str]:
        missing = [field for field in REQUIRED_CREATE_FIELDS if not draft.get(field)]
        if not any(draft.get(field) for field in CREATE_CONTENT_FIELDS):
            missing.append("job_content")
        return missing

    def _creation_result(
        self,
        draft: dict[str, Any],
        changed: dict[str, Any],
        semantic_facts: list[dict[str, Any]],
        clarification_questions: list[str],
    ) -> AgentState:
        missing = self._missing(draft)
        recognized = f"本轮识别到的字段：\n{_field_lines(changed)}"
        notes = _semantic_notes(semantic_facts, clarification_questions)
        if missing:
            labels = "、".join(FIELD_LABELS.get(field, field) for field in missing)
            return {
                **_clean_turn(), "phase": Phase.CREATING.value, "draft": draft,
                "semantic_facts": semantic_facts,
                "clarification_questions": clarification_questions,
                "missing_fields": missing, "can_confirm": False, "can_cancel": True,
                "message": f"{recognized}{notes}\n\n当前还缺少必填字段：{labels}。请继续补充。",
            }
        return {
            **_clean_turn(), "phase": Phase.CONFIRMING.value, "draft": draft,
            "pending_action": "create", "pending_fields": draft, "pending_clear_fields": [],
            "semantic_facts": semantic_facts,
            "clarification_questions": clarification_questions,
            "pending_semantic_facts": semantic_facts,
            "missing_fields": [], "can_confirm": True, "can_cancel": True,
            "message": f"{recognized}\n\n📝 请确认新建岗位：\n{_field_lines(draft)}{notes}",
        }

    def _idle(self, state: AgentState) -> AgentState:
        command = self._command(state)
        fields = command.fields.model_dump(exclude_none=True)
        if command.intent == "create":
            facts = _merge_facts([], command.semantic_facts)
            return self._creation_result(fields, fields, facts, command.clarification_questions)
        if command.intent in {"update", "delete"}:
            return self._locate_for_edit(command)
        if command.intent == "search":
            if self.query_adapter is None:
                return {**_clean_turn(), "message": "当前未配置 jd_text2sql 查询适配器。"}
            result = self.query_adapter.ask(state.get("user_input", ""))
            if result.get("needs_clarification"):
                return {**_clean_turn(), "message": result.get("message") or "请补充查询条件。"}
            rows = result.get("rows", [])
            body = json.dumps(rows, ensure_ascii=False, indent=2, default=str)
            return {**_clean_turn(), "message": f"查询结果（{len(rows)} 条）：\n```json\n{body}\n```"}
        return {
            **_clean_turn(),
            "message": "我还不能确定您的意图。请明确说“新建岗位”“修改岗位”“删除岗位”或提出岗位查询。",
        }

    def _creating(self, state: AgentState) -> AgentState:
        command = self._command(state)
        if command.intent == "cancel":
            return _reset("已取消新建，未写入 CSV。")
        changed = command.fields.model_dump(exclude_none=True)
        draft = dict(state.get("draft", {}))
        draft.update(changed)
        for field in command.clear_fields:
            draft.pop(field, None)
        facts = _merge_facts(state.get("semantic_facts", []), command.semantic_facts)
        questions = list(dict.fromkeys([
            *state.get("clarification_questions", []), *command.clarification_questions,
        ]))
        return self._creation_result(draft, changed, facts, questions)

    def _locate_for_edit(self, command: StructuredCommand) -> AgentState:
        fields = command.fields.model_dump(exclude_none=True)
        query = (command.search_query or "").strip()
        if not query:
            query = " ".join(str(fields.get(key, "")) for key in ("company_name", "title")).strip()
        if not query:
            return {
                **_clean_turn(), "phase": Phase.EDITING.value, "can_cancel": True,
                "message": "请提供要操作的公司名、岗位名或 job_id。",
            }
        candidates = self.repository.search(query)
        if not candidates:
            return {
                **_clean_turn(), "phase": Phase.EDITING.value, "can_cancel": True,
                "message": f"没有找到与“{query}”匹配的岗位，请换一个公司名、岗位名或 job_id。",
            }
        pending = {
            "pending_action": "delete" if command.intent == "delete" else "update",
            "pending_fields": fields,
            "pending_clear_fields": command.clear_fields,
            "pending_semantic_facts": [
                fact.model_dump(mode="json") for fact in command.semantic_facts
            ],
        }
        if len(candidates) == 1:
            return self._selected(candidates[0], pending)
        lines = "\n".join(
            f"[{index}] {row['company_name']} - {row['title']} (ID: {row['job_id']})"
            for index, row in enumerate(candidates, 1)
        )
        return {
            **_clean_turn(), **pending, "phase": Phase.EDITING.value,
            "candidates": candidates, "can_cancel": True,
            "message": f"找到多个候选岗位，请输入序号或点击按钮选择：\n{lines}",
        }

    def _selected(self, row: dict[str, str], pending: dict[str, Any]) -> AgentState:
        action = pending.get("pending_action", "update")
        fields = pending.get("pending_fields", {})
        clear_fields = pending.get("pending_clear_fields", [])
        semantic_facts = pending.get("pending_semantic_facts", [])
        base = {
            **_clean_turn(), "target_id": row["job_id"], "candidates": [],
            "pending_action": action, "pending_fields": fields,
            "pending_clear_fields": clear_fields, "can_cancel": True,
            "pending_semantic_facts": semantic_facts,
        }
        if action == "delete":
            return {
                **base, "phase": Phase.CONFIRMING.value, "can_confirm": True,
                "message": (
                    "📝 请确认删除以下岗位（此最小实现执行物理删除）：\n"
                    f"- {row['company_name']} - {row['title']} (ID: {row['job_id']})"
                ),
            }
        if not fields and not clear_fields:
            return {
                **base, "phase": Phase.EDITING.value, "pending_action": None,
                "message": f"已锁定 {row['company_name']} - {row['title']}。请告诉我要修改哪些字段。",
            }
        return self._update_preview(row, fields, clear_fields, semantic_facts)

    def _update_preview(
        self, row: dict[str, str], fields: dict[str, Any], clear_fields: list[str],
        semantic_facts: list[dict[str, Any]] | None = None,
    ) -> AgentState:
        lines: list[str] = []
        for field, value in fields.items():
            lines.append(
                f"- {FIELD_LABELS.get(field, field)}：{_display_value(field, row.get(field))} ➔ {_display_value(field, value)}"
            )
        for field in clear_fields:
            lines.append(f"- {FIELD_LABELS.get(field, field)}：{_display_value(field, row.get(field))} ➔ （清空）")
        return {
            **_clean_turn(), "phase": Phase.CONFIRMING.value, "target_id": row["job_id"],
            "candidates": [], "pending_action": "update", "pending_fields": fields,
            "pending_clear_fields": clear_fields, "can_confirm": True, "can_cancel": True,
            "pending_semantic_facts": semantic_facts or [],
            "message": (
                f"📝 请确认修改：{row['company_name']} - {row['title']} (ID: {row['job_id']})\n"
                + "\n".join(lines)
            ),
        }

    def _editing(self, state: AgentState) -> AgentState:
        command = self._command(state)
        if command.intent == "cancel":
            return _reset("已取消修改，未写入 CSV。")
        candidates = state.get("candidates", [])
        if candidates:
            index = command.selection_index
            if index is None or not 1 <= index <= len(candidates):
                return {**_clean_turn(), "message": f"请输入 1 到 {len(candidates)} 之间的序号。"}
            pending = {
                "pending_action": state.get("pending_action") or "update",
                "pending_fields": state.get("pending_fields", {}),
                "pending_clear_fields": state.get("pending_clear_fields", []),
                "pending_semantic_facts": state.get("pending_semantic_facts", []),
            }
            return self._selected(candidates[index - 1], pending)
        target_id = state.get("target_id")
        if not target_id:
            return self._locate_for_edit(command)
        row = self.repository.get(target_id)
        if command.intent == "delete":
            return self._selected(row, {
                "pending_action": "delete", "pending_fields": {}, "pending_clear_fields": [],
                "pending_semantic_facts": [],
            })
        changed = command.fields.model_dump(exclude_none=True)
        if not changed and not command.clear_fields:
            return {**_clean_turn(), "message": "没有识别到要修改的字段，请再具体描述一次。"}
        fields = dict(state.get("pending_fields", {})) if state.get("pending_action") == "update" else {}
        fields.update(changed)
        clears = (
            list(state.get("pending_clear_fields", []))
            if state.get("pending_action") == "update" else []
        )
        clears = list(dict.fromkeys([*clears, *command.clear_fields]))
        for key in changed:
            if key in clears:
                clears.remove(key)
        facts = _merge_facts(state.get("pending_semantic_facts", []), command.semantic_facts)
        return self._update_preview(row, fields, clears, facts)

    def _confirming(self, state: AgentState) -> AgentState:
        command = self._command(state)
        if command.intent == "cancel":
            return _reset("已取消操作，CSV 未发生变化。")
        action = state.get("pending_action")
        if command.intent == "confirm":
            if action == "create":
                row = self.repository.create(
                    state.get("pending_fields", {}),
                    persisted_semantic_facts(state.get("pending_semantic_facts", [])),
                )
                message = f"已写入新岗位：{row['company_name']} - {row['title']} (ID: {row['job_id']})。"
            elif action == "update":
                _, row = self.repository.update(
                    state.get("target_id") or "", state.get("pending_fields", {}),
                    state.get("pending_clear_fields", []),
                    persisted_semantic_facts(state.get("pending_semantic_facts", [])),
                )
                message = f"已更新岗位：{row['company_name']} - {row['title']} (ID: {row['job_id']})。"
            elif action == "delete":
                row = self.repository.delete(state.get("target_id") or "")
                message = f"已删除岗位：{row['company_name']} - {row['title']} (ID: {row['job_id']})。"
            else:
                return _reset("没有待确认的操作。")
            if self.query_adapter is not None:
                try:
                    counts = self.query_adapter.rebuild()
                    message += f" jd_text2sql 查询库已同步（{counts['jobs']} 条）。"
                except Exception as exc:  # CSV commit succeeded; never invite a duplicate retry.
                    message += f" 但查询库同步失败：{exc}。请稍后执行 `python -m jd_text2sql build-db`。"
            return _reset(message)
        if command.intent == "update":
            changed = command.fields.model_dump(exclude_none=True)
            if not changed and not command.clear_fields:
                phase = Phase.CREATING.value if action == "create" else Phase.EDITING.value
                return {
                    **_clean_turn(), "phase": phase, "pending_action": action,
                    "draft": state.get("pending_fields", {}) if action == "create" else state.get("draft", {}),
                    "semantic_facts": state.get("pending_semantic_facts", []),
                    "clarification_questions": state.get("clarification_questions", []),
                    "can_confirm": False, "message": "好的，请继续告诉我要调整哪些字段。",
                }
            if action == "create":
                draft = dict(state.get("pending_fields", {}))
                draft.update(changed)
                for field in command.clear_fields:
                    draft.pop(field, None)
                facts = _merge_facts(state.get("pending_semantic_facts", []), command.semantic_facts)
                questions = list(dict.fromkeys([
                    *state.get("clarification_questions", []), *command.clarification_questions,
                ]))
                return self._creation_result(draft, changed, facts, questions)
            if action == "update":
                fields = dict(state.get("pending_fields", {}))
                fields.update(changed)
                clears = list(dict.fromkeys([*state.get("pending_clear_fields", []), *command.clear_fields]))
                for key in changed:
                    if key in clears:
                        clears.remove(key)
                row = self.repository.get(state.get("target_id") or "")
                facts = _merge_facts(state.get("pending_semantic_facts", []), command.semantic_facts)
                return self._update_preview(row, fields, clears, facts)
        return {
            **_clean_turn(), "message": "当前正在等待确认。请选择“确认写入”“取消”或“继续修改”。",
        }
