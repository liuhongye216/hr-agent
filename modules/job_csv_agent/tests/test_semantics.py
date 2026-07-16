from __future__ import annotations

import pytest

from job_csv_agent.semantics import classify_atomic_text


def _core(text: str) -> list[tuple[str, str, str]]:
    return [
        (fact.value, fact.category, fact.importance)
        for fact in classify_atomic_text(text)
        if fact.category != "skill"
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("从0开始预训练大模型", [("从0开始预训练大模型", "responsibility", "neutral")]),
        ("具备从0开始预训练大模型的经验", [("具备从0开始预训练大模型的经验", "requirement", "must")]),
        ("负责训练数据清洗和模型预训练", [("负责训练数据清洗和模型预训练", "responsibility", "neutral")]),
        ("熟悉 Python 和 PyTorch", [("熟悉 Python 和 PyTorch", "requirement", "must")]),
        (
            "参与模型预训练，有分布式训练经验优先",
            [
                ("参与模型预训练", "responsibility", "neutral"),
                ("有分布式训练经验优先", "requirement", "preferred"),
            ],
        ),
        (
            "本科及以上学历，负责训练平台建设",
            [
                ("本科及以上学历", "requirement", "must"),
                ("负责训练平台建设", "responsibility", "neutral"),
            ],
        ),
    ],
)
def test_atomic_semantic_classification(text: str, expected: list[tuple[str, str, str]]) -> None:
    assert _core(text) == expected


def test_requirement_can_emit_explicit_skill_tags() -> None:
    facts = classify_atomic_text("熟悉 Python 和 PyTorch")
    assert [(fact.value, fact.category) for fact in facts] == [
        ("熟悉 Python 和 PyTorch", "requirement"),
        ("Python", "skill"),
        ("PyTorch", "skill"),
    ]
