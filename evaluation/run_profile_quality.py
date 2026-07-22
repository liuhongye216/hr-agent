from __future__ import annotations

import json
from pathlib import Path

from hr_agent_contracts import JobProfile
from job_csv_agent.critic import DeterministicProfileCritic


def run(cases_path: Path) -> dict[str, float | int]:
    critic = DeterministicProfileCritic()
    total = 0
    passed = 0
    for line in cases_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        review = critic.review(JobProfile.model_validate(case["profile"]))
        status_ok = review.status == case["expected_status"]
        issue_ok = not case.get("expected_issue") or case["expected_issue"] in {
            issue.code for issue in review.issues
        }
        total += 1
        passed += int(status_ok and issue_ok)
    return {"total": total, "passed": passed, "accuracy": passed / total if total else 0.0}


if __name__ == "__main__":
    path = Path(__file__).with_name("profile_quality_cases.jsonl")
    print(json.dumps(run(path), ensure_ascii=False))
