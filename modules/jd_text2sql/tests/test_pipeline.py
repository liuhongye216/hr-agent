from __future__ import annotations

import csv
import json

import pytest

from jd_text2sql.pipeline import BUSINESS_COLUMNS, build_business_csv, export_source_csv
from jd_text2sql.pipeline import _rule_long_text


def _sample_job() -> dict:
    return {
        "job_id": "job-1",
        "title": "数据实习生",
        "salary": "200-300元/天",
        "city": "深圳",
        "experience": "经验不限",
        "education": "本科",
        "company_name": "示例公司",
        "work_address": "深圳南山区",
        "jd_raw": "为符合岗位要求的同学提供机会。岗位职责：1、整理业务数据；2、维护分析报表。岗位要求：1、本科及以上；2、熟练使用Excel和SQL。",
        "job_tags": ["五险一金", "餐补"],
        "source_url": "https://example.com/job-1",
        "scraped_at": "2026-07-16T00:00:00+08:00",
        "nested": {"kept": True},
    }


def test_two_stage_rules_pipeline(tmp_path) -> None:
    source_jsonl = tmp_path / "jobs.jsonl"
    source_csv = tmp_path / "source.csv"
    business_csv = tmp_path / "business.csv"
    source_jsonl.write_text(json.dumps(_sample_job(), ensure_ascii=False) + "\n", encoding="utf-8")

    source_stats = export_source_csv(source_jsonl, source_csv)
    assert source_stats["source_rows"] == 1
    with source_csv.open(encoding="utf-8-sig", newline="") as handle:
        source_row = next(csv.DictReader(handle))
    assert json.loads(source_row["nested"]) == {"kept": True}

    business_stats = build_business_csv(source_csv, business_csv, mode="rules")
    assert business_stats["business_rows"] == 1
    with business_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)
        assert tuple(reader.fieldnames or []) == BUSINESS_COLUMNS
    assert row["salary_min"] == "200.0"
    assert row["salary_period"] == "day"
    assert row["education_min_level"] == "4"
    assert json.loads(row["responsibilities_json"]) == ["整理业务数据", "维护分析报表"]
    assert "Excel" in json.loads(row["skills_json"])
    assert json.loads(row["benefits_json"]) == ["五险一金", "餐补"]


def test_hybrid_requires_explicit_permission(tmp_path) -> None:
    source_jsonl = tmp_path / "jobs.jsonl"
    source_csv = tmp_path / "source.csv"
    source_jsonl.write_text(json.dumps(_sample_job(), ensure_ascii=False) + "\n", encoding="utf-8")
    export_source_csv(source_jsonl, source_csv)
    with pytest.raises(PermissionError):
        build_business_csv(source_csv, tmp_path / "business.csv", mode="hybrid")


def test_rule_extraction_splits_and_reclassifies_compound_semantics() -> None:
    extracted = _rule_long_text(
        "岗位职责：参与模型预训练，有分布式训练经验优先；"
        "本科及以上学历，负责训练平台建设。"
    )
    assert extracted["responsibilities"] == ["参与模型预训练", "负责训练平台建设"]
    assert extracted["requirements"] == ["有分布式训练经验优先", "本科及以上学历"]
