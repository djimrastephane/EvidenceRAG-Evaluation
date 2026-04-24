from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "source_eval_set",
    "query_id",
    "doc_id",
    "year",
    "difficulty",
    "answer_type",
    "evidence_layout",
    "question",
    "expected_answer",
    "expected_pages",
    "expected_section",
    "expected_subsection",
    "filter_hints_json",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a flat CSV inventory of eval_set queries.")
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
        default="results/query_inventory/current_query_inventory.csv",
        help="Path to write the flattened query inventory CSV.",
    )
    return p.parse_args()


def _iter_queries(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("queries"), list):
        return [row for row in payload["queries"] if isinstance(row, dict)]
    raise ValueError("Expected eval_set payload to be a list or a dict with 'queries' list.")


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_csv = Path(args.output_csv)

    rows: list[dict[str, str]] = []
    for eval_path in sorted(input_root.glob(args.glob)):
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        queries = _iter_queries(payload)
        for q in queries:
            rows.append(
                {
                    "source_eval_set": str(eval_path),
                    "query_id": str(q.get("query_id") or ""),
                    "doc_id": str(q.get("doc_id") or eval_path.parent.name),
                    "year": str(q.get("year") or ""),
                    "difficulty": str(q.get("difficulty") or ""),
                    "answer_type": str(q.get("answer_type") or ""),
                    "evidence_layout": str(q.get("evidence_layout") or ""),
                    "question": str(q.get("question") or ""),
                    "expected_answer": str(q.get("expected_answer") or ""),
                    "expected_pages": json.dumps(q.get("expected_pages") or [], ensure_ascii=False),
                    "expected_section": str(q.get("expected_section") or ""),
                    "expected_subsection": str(q.get("expected_subsection") or ""),
                    "filter_hints_json": json.dumps(q.get("filter_hints") or {}, ensure_ascii=False, sort_keys=True),
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} queries to {output_csv}")


if __name__ == "__main__":
    main()
