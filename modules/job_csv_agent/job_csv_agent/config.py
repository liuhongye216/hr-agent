from __future__ import annotations

import os
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = MODULE_ROOT.parents[1]


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.removeprefix("export ").strip(), value.strip().strip('"').strip("'"))


_load_env(AGENT_ROOT / ".env")
_load_env(MODULE_ROOT / ".env")
# Environment values are captured at import time; restart the API after changing .env.

BUSINESS_CSV = AGENT_ROOT / "data" / "business" / "jobs.csv"
SQLITE_DB = AGENT_ROOT / "runtime" / "jd_text2sql.sqlite"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")


def is_valid_api_key(value: str | None = None) -> bool:
    key = (DEEPSEEK_API_KEY if value is None else value) or ""
    normalized = key.strip().casefold()
    return (
        len(key.strip()) >= 20
        and normalized not in {"replace-me", "your-api-key", "changeme"}
        and not normalized.startswith("replace-")
    )
