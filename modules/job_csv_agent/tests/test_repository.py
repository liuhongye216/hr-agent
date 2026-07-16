from __future__ import annotations

import csv
from pathlib import Path

import pytest

from job_csv_agent.repository import CsvJobRepository, SemanticValidationError
from job_csv_agent.config import is_valid_api_key
from job_csv_agent.schemas import BUSINESS_COLUMNS


def empty_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.DictWriter(handle, fieldnames=BUSINESS_COLUMNS).writeheader()


def test_repository_crud_and_fuzzy_search(tmp_path: Path) -> None:
    path = tmp_path / "jobs.csv"
    empty_csv(path)
    repository = CsvJobRepository(path)

    created = repository.create({
        "company_name": "示例科技", "title": "Python 后台开发",
        "requirements_json": ["三年 Python 经验"], "salary_min": 20_000,
    })
    assert created["job_id"].startswith("job_")
    assert created["skills_json"] == "[]"
    assert repository.search("示例 Python")[0]["job_id"] == created["job_id"]

    before, after = repository.update(created["job_id"], {"salary_min": 25_000})
    assert before["salary_min"] == "20000.0"
    assert after["salary_min"] == "25000.0"

    with pytest.raises(ValueError, match="必填字段"):
        repository.update(created["job_id"], {}, ["company_name"])
    assert repository.get(created["job_id"])["company_name"] == "示例科技"

    deleted = repository.delete(created["job_id"])
    assert deleted["job_id"] == created["job_id"]
    assert repository.search("示例 Python") == []


def test_api_key_validation_rejects_example_values() -> None:
    assert is_valid_api_key("replace-me") is False
    assert is_valid_api_key("your-api-key") is False
    assert is_valid_api_key("sk-12345678901234567890") is True


def test_repository_allows_responsibility_only_create(tmp_path: Path) -> None:
    path = tmp_path / "jobs.csv"
    empty_csv(path)
    row = CsvJobRepository(path).create({
        "company_name": "示例科技",
        "title": "大模型工程师",
        "responsibilities_json": ["从0开始预训练大模型"],
    })
    assert row["requirements_json"] == "[]"
    assert row["responsibilities_json"] == '["从0开始预训练大模型"]'


def test_repository_rejects_semantic_misclassification(tmp_path: Path) -> None:
    path = tmp_path / "jobs.csv"
    empty_csv(path)
    repository = CsvJobRepository(path)
    with pytest.raises(SemanticValidationError, match="动作型职责"):
        repository.create({
            "company_name": "示例科技", "title": "大模型工程师",
            "requirements_json": ["从0开始预训练大模型"],
        })
    with pytest.raises(SemanticValidationError, match="候选人资格"):
        repository.create({
            "company_name": "示例科技", "title": "大模型工程师",
            "responsibilities_json": ["本科及以上学历"],
        })


def test_inferred_fact_requires_user_confirmation_before_write(tmp_path: Path) -> None:
    path = tmp_path / "jobs.csv"
    empty_csv(path)
    repository = CsvJobRepository(path)
    fields = {
        "company_name": "示例科技", "title": "大模型工程师",
        "responsibilities_json": ["从0开始预训练大模型"],
        "requirements_json": ["具备分布式训练经验"],
    }
    inferred = [{
        "value": "具备分布式训练经验", "category": "requirement", "importance": "unknown",
        "source_type": "inferred", "evidence_text": "从0开始预训练大模型",
        "needs_confirmation": True,
    }]
    with pytest.raises(SemanticValidationError, match="推断信息未经用户确认"):
        repository.create(fields, inferred)

    confirmed = [{**inferred[0], "source_type": "user_confirmed", "importance": "must", "needs_confirmation": False}]
    row = repository.create(fields, confirmed)
    assert row["requirements_json"] == '["具备分布式训练经验"]'
