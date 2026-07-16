from __future__ import annotations

import json
import re
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from .jd_text2sql_adapter import JDText2SQLAdapter
from .llm import deterministic_command
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
    unanswered_questions: list[str]
    asked_questions: list[str]
    pending_suggestions: list[str]
    missing_important_fields: list[str]
    incomplete_warning_acknowledged: bool
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
        "unanswered_questions": [],
        "asked_questions": [],
        "pending_suggestions": [],
        "missing_important_fields": [],
        "incomplete_warning_acknowledged": False,
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
    friendly_values = {
        "full_time": "全职", "part_time": "兼职", "full_or_part_time": "全职或兼职",
        "internship": "实习", "contract": "合同制", "temporary": "临时用工",
        "onsite": "现场办公", "remote": "远程办公", "hybrid": "混合办公",
        "campus": "校园招聘", "experienced": "社会招聘", "mixed": "不限",
        "month": "月", "year": "年", "day": "日", "hour": "小时", "per_order": "单",
    }
    if isinstance(value, str) and value in friendly_values:
        return friendly_values[value]
    return str(value)


def _field_lines(fields: dict[str, Any]) -> str:
    if not fields:
        return "（本轮没有新增信息）"
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
        sections.append("仍需您确认：" + "、".join(unknown))
    if inferred:
        sections.append("建议补充（确认前不会保存）：\n" + "\n".join(f"- {item}" for item in inferred))
    if questions:
        sections.append("接下来可优先确认：\n" + "\n".join(f"- {item}" for item in questions[:3]))
    return "\n\n" + "\n\n".join(sections) if sections else ""


def _important_missing(draft: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    checks = (
        ("responsibilities_json", "岗位职责"),
        ("requirements_json", "任职要求"),
        ("skills_json", "技能要求"),
        ("city", "工作城市"),
        ("work_mode", "办公模式"),
        ("employment", "用工类型"),
        ("education_min_level", "学历要求"),
    )
    for field, label in checks:
        if not draft.get(field):
            missing.append(label)
    if not draft.get("salary_min") and not draft.get("salary_max"):
        missing.append("薪资范围")
    if not draft.get("experience_min_months") and not draft.get("experience_max_months"):
        missing.append("经验要求")
    if draft.get("employment") == "internship" or draft.get("recruitment") == "internship":
        missing.append("实习时长与每周出勤要求")
    return missing


def _priority_questions(draft: dict[str, Any], suggestions: list[str]) -> list[str]:
    questions: list[str] = []
    if not draft.get("company_name") and not draft.get("title"):
        questions.append("请补充公司名称和岗位名称。")
    elif not draft.get("company_name"):
        questions.append("请补充公司名称。")
    elif not draft.get("title"):
        questions.append("请补充岗位名称。")
    if not any(draft.get(field) for field in CREATE_CONTENT_FIELDS):
        questions.append("请补充主要工作内容或任职要求。")
    questions.extend(suggestions)
    return list(dict.fromkeys(questions))


def _question_is_answered(question: str, draft: dict[str, Any], user_input: str) -> bool:
    if "公司名称" in question and draft.get("company_name"):
        if "岗位名称" not in question or draft.get("title"):
            return True
    if "岗位名称" in question and draft.get("title") and "公司名称" not in question:
        return True
    if ("工作内容" in question or "任职要求" in question) and any(
        draft.get(field) for field in CREATE_CONTENT_FIELDS
    ):
        return True
    if "后训练范围" in question and re.search(r"SFT|RLHF|DPO|奖励模型|模型评估|其他", user_input, re.IGNORECASE):
        return True
    if "Python" in question and re.search(r"Python|PyTorch|分布式", user_input, re.IGNORECASE):
        return True
    return False


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
        deterministic = deterministic_command(text, state.get("phase", Phase.IDLE.value))
        if deterministic is not None:
            return reconcile_command_semantics(deterministic, text)
        context = {
            "phase": state.get("phase"),
            "draft": state.get("draft", {}),
            "target_id": state.get("target_id"),
            "candidate_count": len(state.get("candidates", [])),
            "pending_action": state.get("pending_action"),
        }
        command = reconcile_command_semantics(self.interpreter.extract(text, context), text)
        if state.get("phase") == Phase.CREATING.value:
            # An active creation draft owns ordinary follow-up text. Only the explicit
            # deterministic operation patterns above may switch to an existing-position flow.
            command = command.model_copy(update={"intent": "update", "search_query": None})
        return command

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
        previous_state: AgentState | None = None,
        user_input: str = "",
    ) -> AgentState:
        previous_state = previous_state or initial_state()
        missing = self._missing(draft)
        important_missing = _important_missing(draft)
        all_questions = _priority_questions(draft, clarification_questions)
        unresolved = [
            question for question in dict.fromkeys([
                *previous_state.get("unanswered_questions", []), *all_questions,
            ])
            if not _question_is_answered(question, draft, user_input)
        ]
        asked = list(previous_state.get("asked_questions", []))
        display_questions = [question for question in unresolved if question not in asked][:3]
        asked = list(dict.fromkeys([*asked, *display_questions]))
        suggestions = [
            item["value"] for item in semantic_facts if item.get("source_type") == "inferred"
        ]
        recognized = f"本轮整理出的岗位信息：\n{_field_lines(changed)}"
        notes = _semantic_notes(semantic_facts, display_questions)
        question_state = {
            "clarification_questions": clarification_questions,
            "unanswered_questions": unresolved,
            "asked_questions": asked,
            "pending_suggestions": suggestions,
            "missing_important_fields": important_missing,
            "incomplete_warning_acknowledged": False,
        }
        if missing:
            labels = "、".join(FIELD_LABELS.get(field, field) for field in missing)
            return {
                **_clean_turn(), "phase": Phase.CREATING.value, "draft": draft,
                "semantic_facts": semantic_facts,
                **question_state,
                "missing_fields": missing, "can_confirm": False, "can_cancel": True,
                "message": f"{recognized}{notes}\n\n要生成岗位，当前还需要：{labels}。",
            }
        missing_section = (
            "\n\n仍未填写的重要信息：\n"
            + "\n".join(f"- {item}" for item in important_missing)
            + "\n\n您可以继续补充，也可以按当前内容保存；保存前我会再提醒一次。"
            if important_missing else ""
        )
        return {
            **_clean_turn(), "phase": Phase.CONFIRMING.value, "draft": draft,
            "pending_action": "create", "pending_fields": draft, "pending_clear_fields": [],
            "semantic_facts": semantic_facts,
            **question_state,
            "pending_semantic_facts": semantic_facts,
            "missing_fields": [], "can_confirm": True, "can_cancel": True,
            "message": (
                f"{recognized}\n\n已整理的岗位信息：\n{_field_lines(draft)}"
                f"{notes}{missing_section}"
            ),
        }

    def _idle(self, state: AgentState) -> AgentState:
        command = self._command(state)
        fields = command.fields.model_dump(exclude_none=True)
        if command.intent == "create":
            facts = _merge_facts([], command.semantic_facts)
            return self._creation_result(
                fields, fields, facts, command.clarification_questions, state,
                state.get("user_input", ""),
            )
        if command.intent in {"update", "delete"}:
            return self._locate_for_edit(command)
        if command.intent == "search":
            return self._run_search(state.get("user_input", ""))
        return {
            **_clean_turn(),
            "message": "我还不能确定您想进行哪项操作。请明确说“新建岗位”“修改岗位”“删除岗位”或提出岗位查询。",
        }

    def _run_search(self, text: str) -> AgentState:
        if self.query_adapter is None:
            return {**_clean_turn(), "message": "岗位查询暂时不可用，请稍后再试。"}
        result = self.query_adapter.ask(text)
        if result.get("needs_clarification"):
            return {**_clean_turn(), "message": result.get("message") or "请补充想查找的公司或岗位。"}
        rows = result.get("rows", [])
        if not rows:
            return {**_clean_turn(), "message": "没有找到对应岗位，请检查公司名称或岗位名称是否准确。"}
        sections: list[str] = []
        for index, row in enumerate(rows, 1):
            visible = {
                key: value for key, value in row.items()
                if key in FIELD_LABELS and key not in {"job_id"} and value not in (None, "", [], "[]")
            }
            sections.append(f"{index}.\n{_field_lines(visible)}")
        return {**_clean_turn(), "message": f"找到 {len(rows)} 个岗位：\n\n" + "\n\n".join(sections)}

    def _creating(self, state: AgentState) -> AgentState:
        command = self._command(state)
        if command.intent == "cancel":
            return _reset("已取消新建，内容未保存。")
        if command.intent == "search":
            return self._run_search(state.get("user_input", ""))
        if command.intent in {"update", "delete"} and command.search_query:
            return self._locate_for_edit(command)
        changed = command.fields.model_dump(exclude_none=True)
        draft = dict(state.get("draft", {}))
        draft.update(changed)
        for field in command.clear_fields:
            draft.pop(field, None)
        facts = _merge_facts(state.get("semantic_facts", []), command.semantic_facts)
        questions = list(dict.fromkeys([
            *state.get("clarification_questions", []), *command.clarification_questions,
        ]))
        return self._creation_result(
            draft, changed, facts, questions, state, state.get("user_input", ""),
        )

    def _locate_for_edit(self, command: StructuredCommand) -> AgentState:
        fields = command.fields.model_dump(exclude_none=True)
        query = (command.search_query or "").strip()
        if not query:
            query = " ".join(str(fields.get(key, "")) for key in ("company_name", "title")).strip()
        if not query:
            action_text = "删除" if command.intent == "delete" else "修改"
            return {
                **_clean_turn(), "phase": Phase.EDITING.value, "can_cancel": True,
                "message": f"请告诉我您想{action_text}的是哪家公司、哪个岗位。",
            }
        candidates = self.repository.search(query)
        if not candidates:
            return {
                **_clean_turn(), "phase": Phase.EDITING.value, "can_cancel": True,
                "message": "没有找到对应岗位，请检查公司名称或岗位名称是否准确。",
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
            f"[{index}] {row['company_name']} - {row['title']}"
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
                    "请确认删除以下岗位：\n"
                    f"- {row['company_name']} - {row['title']}"
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
                f"请确认修改：{row['company_name']} - {row['title']}\n"
                + "\n".join(lines)
            ),
        }

    def _editing(self, state: AgentState) -> AgentState:
        command = self._command(state)
        if command.intent == "cancel":
            return _reset("已取消修改，内容未保存。")
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
            return {**_clean_turn(), "message": "没有识别到要修改的内容，请再具体描述一次。"}
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
            return _reset("已取消操作，内容未保存。")
        action = state.get("pending_action")
        if command.intent == "confirm":
            important_missing = state.get("missing_important_fields", [])
            if action == "create" and important_missing and not state.get("incomplete_warning_acknowledged"):
                labels = "、".join(important_missing)
                return {
                    **state, **_clean_turn(), "phase": Phase.CONFIRMING.value,
                    "incomplete_warning_acknowledged": True,
                    "can_confirm": True, "can_cancel": True,
                    "message": (
                        f"保存前提醒：以下重要信息仍未填写：{labels}。\n\n"
                        "如果确定按当前内容保存，请再次确认；也可以继续补充。"
                    ),
                }
            if action == "create":
                row = self.repository.create(
                    state.get("pending_fields", {}),
                    persisted_semantic_facts(state.get("pending_semantic_facts", [])),
                )
                message = f"已保存新岗位：{row['company_name']} - {row['title']}。"
            elif action == "update":
                _, row = self.repository.update(
                    state.get("target_id") or "", state.get("pending_fields", {}),
                    state.get("pending_clear_fields", []),
                    persisted_semantic_facts(state.get("pending_semantic_facts", [])),
                )
                message = f"已更新岗位：{row['company_name']} - {row['title']}。"
            elif action == "delete":
                row = self.repository.delete(state.get("target_id") or "")
                message = f"已删除岗位：{row['company_name']} - {row['title']}。"
            else:
                return _reset("没有待确认的操作。")
            if self.query_adapter is not None:
                try:
                    self.query_adapter.rebuild()
                    message += " 岗位查询数据已同步。"
                except Exception:  # The save succeeded; never invite a duplicate retry.
                    message += " 岗位已保存，但查询服务暂时未同步，请联系管理员。"
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
                    "unanswered_questions": state.get("unanswered_questions", []),
                    "asked_questions": state.get("asked_questions", []),
                    "pending_suggestions": state.get("pending_suggestions", []),
                    "missing_important_fields": state.get("missing_important_fields", []),
                    "incomplete_warning_acknowledged": False,
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
                return self._creation_result(
                    draft, changed, facts, questions, state, state.get("user_input", ""),
                )
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
