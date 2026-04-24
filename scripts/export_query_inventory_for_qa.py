from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "source_eval_set",
    "doc_id",
    "year",
    "query_id",
    "base_query_id",
    "is_paraphrase",
    "paraphrase_suffix",
    "difficulty",
    "answer_type",
    "evidence_layout",
    "question",
    "expected_answer",
    "expected_pages",
    "expected_section",
    "expected_subsection",
    "filter_hints_json",
    "question_clear",
    "evidence_supported",
    "expected_pages_correct",
    "expected_answer_correct",
    "answer_type_correct",
    "difficulty_correct",
    "paraphrase_equivalent",
    "issue_found",
    "issue_type",
    "notes",
    "review_status",
    "reviewer",
    "review_date",
]

PARAPHRASE_RE = re.compile(r"^(?P<base>.+)_(?P<suffix>P\d+)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a QA-oriented query inventory CSV.")
    p.add_argument(
        "--input-root",
        default="data_processed_region_routing_2026-03-21",
        help="Root containing per-document eval_set.json files.",
    )
    p.add_argument(
        "--glob",
        default="Grampian-*/eval_set.json",
        help="Glob pattern under input-root for eval_set files.",
    )
    p.add_argument(
        "--output-csv",
        default="results/query_inventory/current_query_inventory_for_qa.csv",
        help="Path to write the QA-oriented query inventory CSV.",
    )
    return p.parse_args()


def _iter_queries(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("queries"), list):
        return [row for row in payload["queries"] if isinstance(row, dict)]
    raise ValueError("Expected eval_set payload to be a list or a dict with 'queries' list.")


def _paraphrase_parts(query_id: str) -> tuple[str, str, str]:
    qid = str(query_id or "").strip()
    m = PARAPHRASE_RE.match(qid)
    if not m:
        return qid, "no", ""
    return str(m.group("base")), "yes", str(m.group("suffix"))


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_csv = Path(args.output_csv)

    rows: list[dict[str, str]] = []
    for eval_path in sorted(input_root.glob(args.glob)):
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        queries = _iter_queries(payload)
        for q in queries:
            query_id = str(q.get("query_id") or "")
            base_query_id, is_paraphrase, paraphrase_suffix = _paraphrase_parts(query_id)
            rows.append(
                {
                    "source_eval_set": str(eval_path),
                    "doc_id": str(q.get("doc_id") or eval_path.parent.name),
                    "year": str(q.get("year") or ""),
                    "query_id": query_id,
                    "base_query_id": base_query_id,
                    "is_paraphrase": is_paraphrase,
                    "paraphrase_suffix": paraphrase_suffix,
                    "difficulty": str(q.get("difficulty") or ""),
                    "answer_type": str(q.get("answer_type") or ""),
                    "evidence_layout": str(q.get("evidence_layout") or ""),
                    "question": str(q.get("question") or ""),
                    "expected_answer": str(q.get("expected_answer") or ""),
                    "expected_pages": json.dumps(q.get("expected_pages") or [], ensure_ascii=False),
                    "expected_section": str(q.get("expected_section") or ""),
                    "expected_subsection": str(q.get("expected_subsection") or ""),
                    "filter_hints_json": json.dumps(q.get("filter_hints") or {}, ensure_ascii=False, sort_keys=True),
                    "question_clear": "",
                    "evidence_supported": "",
                    "expected_pages_correct": "",
                    "expected_answer_correct": "",
                    "answer_type_correct": "",
                    "difficulty_correct": "",
                    "paraphrase_equivalent": "",
                    "issue_found": "no",
                    "issue_type": "",
                    "notes": "",
                    "review_status": "pending",
                    "reviewer": "",
                    "review_date": "",
                }
            )

    rows.sort(
        key=lambda r: (
            r["doc_id"],
            0 if r["is_paraphrase"] == "yes" else 1,
            r["base_query_id"],
            r["query_id"],
        )
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} QA rows to {output_csv}")


if __name__ == "__main__":
    main()
