from __future__ import annotations

from .schemas import RouteCategory, RouteDecision, StructuredCommand


def fast_route(text: str) -> None:
    """Open-ended text routing is intentionally disabled.

    The symbol remains as a compatibility marker for callers that imported the old
    rule router.  Confirmation, cancellation and candidate selection are handled by
    the agent's strict task-control parser instead.
    """
    return None


def is_count_request(text: str) -> bool:
    """Count intent is selected by the LLM; execution remains deterministic."""
    return False


def resolve_route(command: StructuredCommand, text: str = "") -> RouteDecision:
    """Resolve typed model output or deterministic task-control commands."""
    if command.route is not None:
        return command.route
    if command.selection_index is not None:
        return RouteDecision(category=RouteCategory.TASK_CONTROL, action="select")
    if command.intent in {"create", "update", "delete"}:
        return RouteDecision(category=RouteCategory.JOB_WRITE, action=command.intent)
    if command.intent == "search":
        return RouteDecision(category=RouteCategory.JOB_READ, action="filter")
    if command.intent in {"confirm", "cancel"}:
        return RouteDecision(category=RouteCategory.TASK_CONTROL, action=command.intent)
    if command.intent == "help":
        return RouteDecision(category=RouteCategory.HELP, action="explain_capabilities")
    if command.intent == "conversation":
        return RouteDecision(category=RouteCategory.CONVERSATION, action="chat")
    return RouteDecision(category=RouteCategory.UNSUPPORTED, action="unsupported", confidence=0.5)
