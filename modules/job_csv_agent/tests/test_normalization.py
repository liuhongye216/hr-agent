from __future__ import annotations

import pytest

from job_csv_agent.normalization import (
    parse_education_level, parse_experience_months, parse_experience_range,
    parse_internship_min_months, parse_onsite_days_per_week, parse_recruitment,
    parse_student_statuses,
)


@pytest.mark.parametrize(("text", "months"), [
    ("三年", 36), ("3年", 36), ("三年以上", 36), ("至少三年", 36),
    ("两年半", 30), ("六个月", 6), ("5到8年", 60),
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


@pytest.mark.parametrize(("text", "value"), [
    ("5到8年后端开发经验", (60, 96)),
    ("3至5年经验", (36, 60)),
    ("2~4年工作经验", (24, 48)),
    ("5年以上经验", (60, None)),
    ("3年以下经验", (None, 36)),
])
def test_experience_range_normalization(
    text: str, value: tuple[int | None, int | None],
) -> None:
    assert parse_experience_range(text) == value


def test_internship_and_student_status_normalization() -> None:
    text = "本科或研究生在读，至少连续实习4个月，每周到岗4天"
    assert parse_education_level(text) == 4
    assert parse_student_statuses(text) == ["本科在读", "研究生在读"]
    assert parse_internship_min_months(text) == 4
    assert parse_onsite_days_per_week(text) == 4
