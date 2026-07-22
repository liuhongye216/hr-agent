from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


END = "__end__"
Node = Callable[[dict[str, Any]], str | None]


@dataclass
class ExplicitStateGraph:
    """Small explicit graph with named nodes, conditional transitions and a step limit."""

    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[str, str] = field(default_factory=dict)
    entrypoint: str | None = None
    max_steps: int = 20

    def add_node(self, name: str, node: Node) -> None:
        if name in self.nodes:
            raise ValueError(f"duplicate workflow node: {name}")
        self.nodes[name] = node

    def add_edge(self, source: str, target: str) -> None:
        self.edges[source] = target

    def set_entrypoint(self, name: str) -> None:
        self.entrypoint = name

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        if self.entrypoint is None:
            raise RuntimeError("workflow has no entrypoint")
        current = self.entrypoint
        trace: list[str] = []
        for _ in range(self.max_steps):
            if current == END:
                state["workflow_trace"] = trace
                return state
            try:
                node = self.nodes[current]
            except KeyError as exc:
                raise RuntimeError(f"workflow points to unknown node: {current}") from exc
            trace.append(current)
            selected = node(state)
            current = selected or self.edges.get(current, END)
        raise RuntimeError(f"workflow exceeded {self.max_steps} steps: {' -> '.join(trace)}")
