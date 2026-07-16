from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import Error as PlaywrightError, Page, sync_playwright


BASE_URL = "https://www.zhipin.com/"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data"
DEFAULT_PROFILE = Path(__file__).resolve().parent / ".browser_profile"


@dataclass(frozen=True)
class Settings:
    count: int
    output_dir: Path
    profile_dir: Path
    executable_path: str | None
    headless: bool
    min_delay: float
    max_delay: float
    timeout_ms: int
    seed_url: str
    cdp_url: str | None


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def utc_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def job_id_from_url(url: str) -> str:
    match = re.search(r"/job_detail/([^/?#]+)\.html", url)
    if match:
        return match.group(1)
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", url).strip("_")[-80:]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def page_is_security(page: Page) -> bool:
    try:
        url = page.url.lower()
        title = (page.title() or "").strip()
        return (
            "security.html" in url
            or "/zp/verify.html" in url
            or title in {"请稍候 - BOSS直聘", "请稍候", "安全验证 - BOSS直聘", "安全验证"}
        )
    except PlaywrightError:
        return False


def index_items_from_html(html: str, seed_url: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    items: list[dict[str, Any]] = []
    for anchor in soup.select('a[href*="/job_detail/"]'):
        href = urljoin(seed_url, anchor.get("href") or "")
        if not re.search(r"/job_detail/[^/?#]+\.html", href):
            continue
        card = anchor.find_parent("li") or anchor.parent
        company_link = card.select_one('a[href*="/gongsi/"]') if card else None
        attrs = [clean_text(x.get_text(" ", strip=True)) for x in anchor.select(".job-text span")]
        company_meta = [
            clean_text(x.get_text(" ", strip=True))
            for x in (card.select(".sub-li-bottom-commany-info span") if card else [])
            if clean_text(x.get_text(" ", strip=True))
        ]
        items.append(
            {
                "source_url": href,
                "title_from_list": clean_text((anchor.select_one(".name") or anchor).get_text(" ", strip=True)),
                "salary_from_list": clean_text((anchor.select_one(".salary") or "").get_text(" ", strip=True))
                if anchor.select_one(".salary") else "",
                "list_attributes": [x for x in attrs if x],
                "company_from_list": clean_text(company_link.get_text(" ", strip=True)) if company_link else "",
                "company_url_from_list": urljoin(seed_url, company_link.get("href") or "") if company_link else "",
                "company_meta_from_list": company_meta,
                "list_card_text": clean_text(anchor.get_text(" ", strip=True)),
            }
        )

    deduped: dict[str, dict[str, Any]] = {}
    for item in items:
        url = item.get("source_url", "")
        if url and urlparse(url).netloc.endswith("zhipin.com"):
            deduped.setdefault(url, item)
    return list(deduped.values())


def collect_index(seed_url: str, request_context: Any | None = None) -> list[dict[str, Any]]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
        "Accept-Language": "zh-CN,zh;q=0.9",
    }
    if request_context is not None:
        response = request_context.get(seed_url, headers=headers, timeout=30_000)
        if not response.ok:
            raise RuntimeError(f"职位索引请求失败：HTTP {response.status}")
        return index_items_from_html(response.text(), seed_url)

    response = requests.get(seed_url, timeout=30, headers=headers)
    response.raise_for_status()
    return index_items_from_html(response.text, seed_url)


def detail_payload(page: Page) -> dict[str, Any]:
    return page.evaluate(
        r"""
        () => {
          const clean = (v) => (v || '').replace(/\s+/g, ' ').trim();
          const text = (selector) => clean(document.querySelector(selector)?.textContent);
          const href = (selector) => document.querySelector(selector)?.href || '';
          const unique = (items) => [...new Set(items.map(clean).filter(Boolean))];
          const companyFacts = Array.from(document.querySelectorAll('.sider-company > p'))
            .map((x) => clean(x.textContent)).filter(Boolean);
          const bossAttr = text('.job-boss-info .boss-info-attr');
          const legalItems = {};
          for (const li of document.querySelectorAll('.business-info-box li')) {
            const label = clean(li.querySelector('span')?.textContent);
            const clone = li.cloneNode(true);
            clone.querySelector('span')?.remove();
            if (label) legalItems[label] = clean(clone.textContent);
          }
          return {
            page_url: location.href,
            page_title: document.title,
            title: text('.job-primary h1') || text('h1'),
            salary: text('.job-primary .salary'),
            city: text('.job-primary .text-city'),
            experience: text('.job-primary .text-experiece'),
            education: text('.job-primary .text-degree'),
            jd_raw: text('.job-detail-section .job-sec-text:not(.fold-text)') || text('.job-sec-text'),
            job_tags: unique(Array.from(document.querySelectorAll('.job-banner .job-tags span')).map(x => x.textContent)),
            company_name: clean(document.querySelector('.sider-company .company-info a[title]')?.getAttribute('title')),
            company_url: href('.sider-company .company-info a[href*="/gongsi/"]'),
            company_facts: companyFacts,
            recruiter_name: text('.job-boss-info h2.name'),
            recruiter_meta: bossAttr,
            work_address: text('.company-address .location-address'),
            company_intro_raw: text('.company-info-box .job-sec-text'),
            legal_company_info: legalItems,
            page_updated_at: clean(Array.from(document.querySelectorAll('p')).find(p => /页面更新时间/.test(p.textContent || ''))?.textContent)
          };
        }
        """
    )


def detail_payload_from_html(html: str, page_url: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")

    def text(selector: str) -> str:
        node = soup.select_one(selector)
        return clean_text(node.get_text(" ", strip=True)) if node else ""

    def href(selector: str) -> str:
        node = soup.select_one(selector)
        return urljoin(page_url, node.get("href") or "") if node else ""

    legal_items: dict[str, str] = {}
    for li in soup.select(".business-info-box li"):
        label_node = li.select_one("span")
        label = clean_text(label_node.get_text(" ", strip=True)) if label_node else ""
        if not label:
            continue
        clone = BeautifulSoup(str(li), "html.parser")
        clone_label = clone.select_one("span")
        if clone_label:
            clone_label.decompose()
        legal_items[label] = clean_text(clone.get_text(" ", strip=True))

    jd_node = soup.select_one(".job-detail-section .job-sec-text:not(.fold-text)")
    if jd_node is None:
        jd_node = soup.select_one(".job-detail-section .job-sec-text")
    jd_raw = clean_text(jd_node.get_text(" ", strip=True)) if jd_node else ""
    gated_markers = ("登录查看完整内容", "登录后查看完整内容", "扫码登录查看完整内容")
    is_gated = any(marker in jd_raw for marker in gated_markers)

    title_node = soup.select_one("title")
    page_title = clean_text(title_node.get_text(" ", strip=True)) if title_node else ""
    updated_node = next(
        (p for p in soup.select("p") if "页面更新时间" in clean_text(p.get_text(" ", strip=True))),
        None,
    )
    company_title = soup.select_one('.sider-company .company-info a[title]')
    return {
        "page_url": page_url,
        "page_title": page_title,
        "title": text(".job-primary h1") or text("h1"),
        "salary": text(".job-primary .salary"),
        "city": text(".job-primary .text-city"),
        "experience": text(".job-primary .text-experiece"),
        "education": text(".job-primary .text-degree"),
        "jd_raw": jd_raw,
        "jd_char_count": len(jd_raw),
        "jd_source_mode": "authenticated_server_html",
        "jd_is_complete": bool(jd_node and jd_raw and not is_gated),
        "job_tags": list(dict.fromkeys(clean_text(x.get_text(" ", strip=True)) for x in soup.select(".job-banner .job-tags span") if clean_text(x.get_text(" ", strip=True)))),
        "company_name": clean_text(company_title.get("title")) if company_title else "",
        "company_url": href('.sider-company .company-info a[href*="/gongsi/"]'),
        "company_facts": [clean_text(x.get_text(" ", strip=True)) for x in soup.select(".sider-company > p") if clean_text(x.get_text(" ", strip=True))],
        "recruiter_name": text(".job-boss-info h2.name"),
        "recruiter_meta": text(".job-boss-info .boss-info-attr"),
        "work_address": text(".company-address .location-address"),
        "company_intro_raw": text(".company-info-box .job-sec-text"),
        "legal_company_info": legal_items,
        "page_updated_at": clean_text(updated_node.get_text(" ", strip=True)) if updated_node else "",
    }


def scrape_detail_request(
    request_context: Any,
    referer_url: str,
    item: dict[str, Any],
    timeout_ms: int,
) -> dict[str, Any]:
    url = item["source_url"]
    response = request_context.get(
        url,
        headers={"Referer": referer_url, "Accept-Language": "zh-CN,zh;q=0.9"},
        timeout=timeout_ms,
    )
    if not response.ok:
        raise RuntimeError(f"DETAIL_HTTP_{response.status}")
    if "security.html" in response.url or "/zp/verify.html" in response.url:
        raise RuntimeError("SECURITY_CHECK")

    payload = detail_payload_from_html(response.text(), response.url)
    if not payload.get("title") or not payload.get("jd_is_complete"):
        raise RuntimeError("JD_TEXT_INCOMPLETE_OR_GATED")

    list_attrs = item.get("list_attributes") or []
    payload.update(
        {
            "job_id": job_id_from_url(url),
            "source_url": url,
            "scraped_at": utc_timestamp(),
            "http_status": response.status,
            "list_snapshot": item,
        }
    )
    payload["title"] = payload.get("title") or item.get("title_from_list", "")
    payload["salary"] = payload.get("salary") or item.get("salary_from_list", "")
    payload["company_name"] = payload.get("company_name") or item.get("company_from_list", "")
    if not payload.get("city") and list_attrs:
        payload["city"] = list_attrs[0]
    if not payload.get("experience") and len(list_attrs) > 1:
        payload["experience"] = list_attrs[1]
    if not payload.get("education") and len(list_attrs) > 2:
        payload["education"] = list_attrs[2]
    payload["list_tags"] = list_attrs[3:]
    return payload


def scrape_detail(page: Page, item: dict[str, Any], timeout_ms: int) -> dict[str, Any]:
    url = item["source_url"]
    if page_is_security(page):
        raise RuntimeError("SECURITY_CHECK")
    response = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    deadline = time.monotonic() + min(timeout_ms / 1000, 20)
    while time.monotonic() < deadline:
        try:
            if page.locator(".job-primary h1").count() == 1 and page.locator(".job-sec-text").count() > 0:
                break
        except PlaywrightError:
            pass
        page.wait_for_timeout(250)

    if page_is_security(page):
        raise RuntimeError("SECURITY_CHECK")

    payload = detail_payload(page)
    if "/job_detail/" not in payload.get("page_url", "") or not payload.get("title"):
        raise RuntimeError(f"详情页未加载成功，当前页面：{payload.get('page_url') or page.url}")

    list_attrs = item.get("list_attributes") or []
    payload.update(
        {
            "job_id": job_id_from_url(url),
            "source_url": url,
            "scraped_at": utc_timestamp(),
            "http_status": response.status if response else None,
            "list_snapshot": item,
        }
    )

    # 列表页字段用于详情页字段缺失时兜底；不做语义抽取。
    payload["title"] = payload.get("title") or item.get("title_from_list", "")
    payload["salary"] = payload.get("salary") or item.get("salary_from_list", "")
    payload["company_name"] = payload.get("company_name") or item.get("company_from_list", "")
    if not payload.get("city") and list_attrs:
        payload["city"] = list_attrs[0]
    if not payload.get("experience") and len(list_attrs) > 1:
        payload["experience"] = list_attrs[1]
    if not payload.get("education") and len(list_attrs) > 2:
        payload["education"] = list_attrs[2]
    payload["list_tags"] = list_attrs[3:]
    return payload


def rebuild_aggregate(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((output_dir / "raw").glob("*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue

    jsonl_path = output_dir / "jobs.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_fields = [
        "job_id", "title", "salary", "city", "experience", "education",
        "company_name", "work_address", "recruiter_name", "recruiter_meta",
        "jd_raw", "source_url", "page_updated_at", "scraped_at",
    ]
    with (output_dir / "jobs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run(settings: Settings) -> int:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = settings.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    settings.profile_dir.mkdir(parents=True, exist_ok=True)

    errors_path = settings.output_dir / "errors.jsonl"
    existing_ids = {path.stem for path in raw_dir.glob("*.json")}

    with sync_playwright() as playwright:
        owns_context = not settings.cdp_url
        if settings.cdp_url:
            browser = playwright.chromium.connect_over_cdp(settings.cdp_url)
            if not browser.contexts:
                raise RuntimeError("已连接浏览器，但没有可用的浏览器上下文。")
            context = browser.contexts[0]
            zhipin_pages = [candidate for candidate in context.pages if "zhipin.com" in candidate.url]
            page = zhipin_pages[0] if zhipin_pages else context.new_page()
        else:
            launch_args: dict[str, Any] = {
                "user_data_dir": str(settings.profile_dir),
                "headless": settings.headless,
                "locale": "zh-CN",
                "viewport": {"width": 1440, "height": 1000},
            }
            if settings.executable_path:
                launch_args["executable_path"] = settings.executable_path
            else:
                launch_args["channel"] = "msedge"
            context = playwright.chromium.launch_persistent_context(**launch_args)
            page = context.pages[0] if context.pages else context.new_page()
        page.set_default_timeout(settings.timeout_ms)

        try:
            index = collect_index(settings.seed_url, context.request)
            write_json(settings.output_dir / "source_index.json", index)
            print(f"发现 {len(index)} 个去重后的公开职位链接。")
            if len(index) < settings.count:
                print(f"警告：当前页面只发现 {len(index)} 个链接，将尽量采集。", file=sys.stderr)

            random.shuffle(index)
            success = len(existing_ids)
            for item in index:
                if success >= settings.count:
                    break
                job_id = job_id_from_url(item["source_url"])
                if job_id in existing_ids:
                    continue

                try:
                    record = scrape_detail_request(context.request, settings.seed_url, item, settings.timeout_ms)
                    if len(record.get("jd_raw", "")) < 20:
                        raise RuntimeError("JD_TEXT_TOO_SHORT")
                    write_json(raw_dir / f"{job_id}.json", record)
                    existing_ids.add(job_id)
                    success += 1
                    print(f"[{success}/{settings.count}] {record.get('title')} | {record.get('company_name')}")
                except Exception as exc:  # 单条失败不影响断点续采
                    error = {
                        "job_id": job_id,
                        "source_url": item.get("source_url"),
                        "error": str(exc),
                        "failed_at": utc_timestamp(),
                    }
                    append_jsonl(errors_path, error)
                    print(f"[失败] {job_id}: {exc}", file=sys.stderr)
                    if str(exc) == "SECURITY_CHECK":
                        print("触发站点安全验证，采集已安全停止。请勿绕过验证；可稍后用有头模式继续。", file=sys.stderr)
                        break

                time.sleep(random.uniform(settings.min_delay, settings.max_delay))
        finally:
            if owns_context:
                context.close()

    rows = rebuild_aggregate(settings.output_dir)
    report = {
        "requested_count": settings.count,
        "saved_count": len(rows),
        "complete": len(rows) >= settings.count,
        "output_dir": str(settings.output_dir.resolve()),
        "generated_at": utc_timestamp(),
    }
    write_json(settings.output_dir / "collection_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["complete"] else 2


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(description="采集 BOSS 直聘公开职位详情并保存为原始 JD 数据集。")
    parser.add_argument("--count", type=int, default=100, help="目标 JD 数量，默认 100")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--seed-url", default=BASE_URL, help="职位链接来源页，默认首页")
    parser.add_argument("--browser-executable", default=None, help="Chrome/Edge 可执行文件路径")
    parser.add_argument("--cdp-url", default=None, help="连接已启动的普通浏览器，例如 http://127.0.0.1:9222")
    parser.add_argument("--headless", action="store_true", help="无头运行；遇验证时建议去掉此参数")
    parser.add_argument("--min-delay", type=float, default=2.0)
    parser.add_argument("--max-delay", type=float, default=4.0)
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    args = parser.parse_args()
    if not 1 <= args.count <= 500:
        parser.error("--count 必须在 1 到 500 之间")
    if args.min_delay < 0 or args.max_delay < args.min_delay:
        parser.error("延迟参数不合法")
    return Settings(
        count=args.count,
        output_dir=args.output_dir.resolve(),
        profile_dir=args.profile_dir.resolve(),
        executable_path=args.browser_executable,
        headless=args.headless,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
        timeout_ms=args.timeout_ms,
        seed_url=args.seed_url,
        cdp_url=args.cdp_url,
    )


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
