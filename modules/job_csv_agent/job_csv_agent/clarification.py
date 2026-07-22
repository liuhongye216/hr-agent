from __future__ import annotations

import uuid
from typing import Any

from hr_agent_contracts import ClarificationQuestion


class ClarificationPolicy:
    """Cross-turn question budget and structured question history."""

    def __init__(self, max_turns: int = 5) -> None:
        self.max_turns = max(1, max_turns)

    def record(
        self, state: dict[str, Any], prompt: str, fields: list[str],
        *, reason: str, blocking: bool = True,
    ) -> ClarificationQuestion:
        history = list(state.get("clarification_history", []))
        sequence = int(state.get("clarification_count", 0)) + 1
        question = ClarificationQuestion(
            question_id=f"question_{uuid.uuid4().hex}",
            job_id=state.get("target_id"), fields=fields or ["job_content"],
            prompt=prompt, priority="blocking" if blocking else "normal",
            reason=reason, sequence=sequence,
        )
        state["clarification_count"] = sequence
        state["clarification_history"] = [*history, question.model_dump(mode="json")][-20:]
        state["active_question"] = question.model_dump(mode="json")
        state["clarification_budget_exhausted"] = sequence >= self.max_turns
        return question

    def answer_current(self, state: dict[str, Any], answer: str, source_message_id: str | None) -> None:
        current = state.get("active_question")
        if not current:
            return
        answers = list(state.get("clarification_answers", []))
        answers.append({
            "schema_version": "1.0.0", "question_id": current["question_id"],
            "answer": answer, "source_message_id": source_message_id,
        })
        state["clarification_answers"] = answers[-20:]
        state["active_question"] = None
