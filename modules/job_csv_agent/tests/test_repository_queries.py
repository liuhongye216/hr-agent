from __future__ import annotations

import csv
from pathlib import Path

from job_csv_agent.agent import JobCsvAgent, initial_state
from job_csv_agent.repository import CsvJobRepository
from job_csv_agent.schemas import BUSINESS_COLUMNS, StructuredCommand


def seeded_repository(path: Path) -> CsvJobRepository:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.DictWriter(handle, fieldnames=BUSINESS_COLUMNS).writeheader()
    repository = CsvJobRepository(path)
    repository.create({
        "company_name": "后浪澎湃有限公司", "title": "混元大语言模型后训练算法工程师",
        "city": "深圳/北京/上海", "recruitment": "experienced",
        "requirements_json": ["硕士及以上学历"], "responsibilities_json": ["负责后训练算法研发"],
    })
    repository.create({
        "company_name": "后浪澎湃有限公司", "title": "数据工程师", "city": "深圳",
        "responsibilities_json": ["负责数据平台建设"],
    })
    repository.create({
        "company_name": "远方科技", "title": "测试工程师", "city": "北京",
        "responsibilities_json": ["负责软件测试"],
    })
    return repository


class QueryInterpreter:
    def extract(self, text: str, context: dict) -> StructuredCommand:
        if text == "一共有多少条目":
            return StructuredCommand(intent="count", query_plan={"mode": "count"})
        if "一共有多少条" in text:
            return StructuredCommand(intent="count", query_plan={
                "mode": "count", "company_name": "后浪澎湃有限公司",
                "title": "混元大语言模型后训练算法工程师",
            })
        if "查看详情" in text:
            return StructuredCommand(intent="detail", query_plan={
                "mode": "detail", "company_name": "后浪澎湃有限公司",
                "title": "混元大语言模型后训练算法工程师",
            })
        return StructuredCommand(intent="search", query_plan={
            "mode": "list", "company_name": "后浪澎湃有限公司",
            "title": "混元大语言模型后训练算法工程师",
        })


def test_repository_query_methods_apply_projection_and_city_contains(tmp_path: Path) -> None:
    repository = seeded_repository(tmp_path / "jobs.csv")
    assert repository.count({}) == 3
    assert repository.count({"city": "北京"}) == 2
    summaries = repository.list_summaries({"company_name": "后浪澎湃"}, 10)
    assert len(summaries) == 2
    assert set(summaries[0]) == {"company_name", "title", "city"}
    detail = repository.get_public_detail(repository.find_job_ids({"title": "混元"})[0])
    assert "content_hash" not in detail
    assert "scraped_at" not in detail
    assert "extraction_mode" not in detail


def test_count_list_and_detail_are_rendered_without_internal_rows(tmp_path: Path) -> None:
    agent = JobCsvAgent(seeded_repository(tmp_path / "jobs.csv"), QueryInterpreter())

    total = agent.handle(initial_state(), "一共有多少条目")
    assert total["message"] == "当前共有 3 条岗位记录。"
    assert "{" not in total["message"] and "requirements_json" not in total["message"]

    filtered = agent.handle(
        initial_state(), "后浪澎湃有限公司的混元大语言模型后训练算法工程师一共有多少条",
    )
    assert filtered["message"] == "当前共有 1 条岗位记录。"

    listed = agent.handle(initial_state(), "查找后浪澎湃有限公司的混元大语言模型后训练算法工程师")
    assert "1. 公司：后浪澎湃有限公司" in listed["message"]
    assert "岗位：混元大语言模型后训练算法工程师" in listed["message"]
    assert "requirements_json" not in listed["message"] and "{" not in listed["message"]

    detail = agent.handle(initial_state(), "查看详情：后浪澎湃有限公司的混元大语言模型后训练算法工程师")
    assert "### 岗位详情" in detail["message"]
    assert "**任职要求**" in detail["message"]
    for internal in ("requirements_json", "responsibilities_json", "content_hash", "extraction_mode", "scraped_at"):
        assert internal not in detail["message"]
