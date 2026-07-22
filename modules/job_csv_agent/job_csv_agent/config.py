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
RUNTIME_DIR = AGENT_ROOT / "runtime"
_control_plane_value = Path(os.getenv("HR_AGENT_CONTROL_PLANE_DB", str(RUNTIME_DIR / "hr_agent.sqlite")))
CONTROL_PLANE_DB = (
    _control_plane_value if _control_plane_value.is_absolute()
    else AGENT_ROOT / _control_plane_value
)
DATABASE_URL = os.getenv("HR_AGENT_DATABASE_URL")
_skill_root_value = Path(os.getenv("HR_AGENT_SKILL_ROOT", str(AGENT_ROOT / "skills")))
SKILL_ROOT = _skill_root_value if _skill_root_value.is_absolute() else AGENT_ROOT / _skill_root_value
DEFAULT_COMPANY_ID = os.getenv("HR_AGENT_DEFAULT_COMPANY_ID", "company_demo")
DEFAULT_OPERATOR_ID = os.getenv("HR_AGENT_DEFAULT_OPERATOR_ID", "operator_demo")
DEFAULT_ROLES = tuple(
    item.strip() for item in os.getenv(
        "HR_AGENT_DEFAULT_ROLES", "recruiter,hr_admin",
    ).split(",") if item.strip()
)
ALLOW_DEMO_IDENTITY = os.getenv("HR_AGENT_ALLOW_DEMO_IDENTITY", "true").casefold() in {
    "1", "true", "yes", "on",
}
AUTH_SECRET = os.getenv("HR_AGENT_AUTH_SECRET")
MAX_CLARIFICATION_TURNS = int(os.getenv("HR_AGENT_MAX_CLARIFICATION_TURNS", "5"))
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
