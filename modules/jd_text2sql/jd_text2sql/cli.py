from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import BUSINESS_JOBS_CSV, DEFAULT_DB_PATH, INTERNAL_RELATIONS, SOURCE_CSV, SOURCE_JSONL
from .db import build_database, execute_readonly
from .llm import text2sql_with_llm
from .pipeline import build_business_csv, export_source_csv
from .rule_text2sql import SQLDraft, generate_rule_sql
from .sql_guard import SQLGuardError, guard_sql


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _export_source(args: argparse.Namespace) -> int:
    _print_json(export_source_csv(args.input, args.output))
    return 0


def _build_business(args: argparse.Namespace) -> int:
    _print_json(
        build_business_csv(
            args.input,
            args.output,
            mode=args.mode,
            allow_external_llm=args.allow_external_llm,
            limit=args.limit,
        )
    )
    return 0


def _prepare(args: argparse.Namespace) -> int:
    source = export_source_csv(args.input, args.source_output)
    business = build_business_csv(
        args.source_output,
        args.business_output,
        mode=args.mode,
        allow_external_llm=args.allow_external_llm,
        limit=args.limit,
    )
    _print_json({"source": source, "business": business})
    return 0


def _build_db(args: argparse.Namespace) -> int:
    counts = build_database(business_csv=args.input, db_path=args.db)
    _print_json({"database": str(args.db), "row_counts": counts})
    return 0


def _generate_sql(args: argparse.Namespace) -> tuple[SQLDraft, str]:
    rule_draft = generate_rule_sql(args.question, args.db)
    if args.mode == "rules" or (args.mode == "hybrid" and rule_draft.matched_rules):
        return rule_draft, "rules"
    llm_draft = text2sql_with_llm(args.question)
    if llm_draft.needs_clarification:
        raise ValueError(llm_draft.clarification_question or "Question needs clarification")
    return (
        SQLDraft(
            sql=llm_draft.sql,
            parameters=tuple(llm_draft.parameters()),
            explanation=llm_draft.explanation,
            matched_rules=(),
        ),
        "llm",
    )


def _ask(args: argparse.Namespace) -> int:
    if not args.db.exists():
        build_database(db_path=args.db)
    draft, generator = _generate_sql(args)
    guarded = guard_sql(draft.sql, draft.parameters, INTERNAL_RELATIONS, max_rows=args.max_rows)
    payload: dict[str, Any] = {
        "question": args.question,
        "generator": generator,
        "sql": guarded.sql,
        "parameters": list(guarded.parameters),
        "explanation": draft.explanation,
    }
    if not args.no_execute:
        payload["rows"] = execute_readonly(args.db, guarded.sql, guarded.parameters)
        payload["row_count"] = len(payload["rows"])
    _print_json(payload)
    return 0


def _add_business_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=("rules", "hybrid"), default="hybrid")
    parser.add_argument("--allow-external-llm", action="store_true")
    parser.add_argument("--limit", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lightweight JD CSV pipeline and guarded Text2SQL")
    sub = parser.add_subparsers(dest="command", required=True)

    source_parser = sub.add_parser("export-source", help="Flatten source jobs.jsonl to a field-level CSV")
    source_parser.add_argument("--input", type=Path, default=SOURCE_JSONL)
    source_parser.add_argument("--output", type=Path, default=SOURCE_CSV)
    source_parser.set_defaults(func=_export_source)

    business_parser = sub.add_parser("build-business", help="Build the single rule + LLM business CSV")
    business_parser.add_argument("--input", type=Path, default=SOURCE_CSV)
    business_parser.add_argument("--output", type=Path, default=BUSINESS_JOBS_CSV)
    _add_business_options(business_parser)
    business_parser.set_defaults(func=_build_business)

    prepare_parser = sub.add_parser("prepare", help="Run both CSV stages")
    prepare_parser.add_argument("--input", type=Path, default=SOURCE_JSONL)
    prepare_parser.add_argument("--source-output", type=Path, default=SOURCE_CSV)
    prepare_parser.add_argument("--business-output", type=Path, default=BUSINESS_JOBS_CSV)
    _add_business_options(prepare_parser)
    prepare_parser.set_defaults(func=_prepare)

    db_parser = sub.add_parser("build-db", help="Load the one business CSV into disposable SQLite")
    db_parser.add_argument("--input", type=Path, default=BUSINESS_JOBS_CSV)
    db_parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    db_parser.set_defaults(func=_build_db)

    ask_parser = sub.add_parser("ask", help="Generate, guard, and optionally execute Text2SQL")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    ask_parser.add_argument("--mode", choices=("rules", "hybrid", "llm"), default="hybrid")
    ask_parser.add_argument("--max-rows", type=int, default=100)
    ask_parser.add_argument("--no-execute", action="store_true")
    ask_parser.set_defaults(func=_ask)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, ValueError, RuntimeError, PermissionError, SQLGuardError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
