from __future__ import annotations

import pytest

from job_csv_agent.normalization import (
    parse_education_level, parse_experience_months, parse_recruitment,
)


@pytest.mark.parametrize(("text", "months"), [
    ("三年", 36), ("3年", 36), ("三年以上", 36), ("至少三年", 36),
    ("两年半", 30), ("六个月", 6),
])
def test_experience_normalization(text: str, months: int) -> None:
    assert parse_experience_months(text) == months


@pytest.mark.parametrize(("text", "value"), [
    ("社招", "experienced"), ("社会招聘", "experienced"),
    ("校招", "campus"), ("校园招聘", "campus"), ("实习招聘", "internship"),
])
def test_recruitment_normalization(text: str, value: str) -> None:
    assert parse_recruitment(text) == value


@pytest.mark.parametrize(("text", "level"), [
    ("本科及以上", 4), ("硕士及以上学历", 5), ("博士及以上", 6),
])
def test_education_normalization(text: str, level: int) -> None:
    assert parse_education_level(text) == level
