from __future__ import annotations

import os
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = MODULE_ROOT.parents[1]


def _load_local_env(path: Path) -> None:
    """Load a git-ignored local .env without another runtime dependency."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_local_env(AGENT_ROOT / ".env")
_load_local_env(MODULE_ROOT / ".env")

# Data belongs to the complete company Agent, not to this implementation module.
DATA_DIR = AGENT_ROOT / "data"
SOURCE_DIR = DATA_DIR / "source"
BUSINESS_DIR = DATA_DIR / "business"
SOURCE_JSONL = SOURCE_DIR / "jobs.jsonl"
SOURCE_CSV = SOURCE_DIR / "jobs.csv"
BUSINESS_JOBS_CSV = BUSINESS_DIR / "jobs.csv"

RUNTIME_DIR = AGENT_ROOT / "runtime"
DEFAULT_DB_PATH = RUNTIME_DIR / "jd_text2sql.sqlite"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
EXTRACTION_MODEL = os.getenv("DEEPSEEK_EXTRACTION_MODEL", DEEPSEEK_MODEL)
TEXT2SQL_MODEL = os.getenv("DEEPSEEK_TEXT2SQL_MODEL", DEEPSEEK_MODEL)

# The lightweight semantic layer exposes one business table only.
INTERNAL_RELATIONS = frozenset({"jobs", "json_each"})
