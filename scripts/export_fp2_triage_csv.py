from __future__ import annotations

import argparse
import csv
from pathlib import Path


OUTPUT_FIELDS = [
    "query_id",
    "document",
    "difficulty",
    "question",
    "expected_answer",
    "expected_pages",
    "evidence_layout",
    "answer_type",
    "top1_chunk_id",
    "top1_pages",
    "top3_chunk_ids",
    "gold_exists",
    "gold_pages_found",
    "fp2_subtype",
    "notes",
    "review_status",
    "reviewer",
    "review_date",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export FP2 cases into a triage CSV template.")
    p.add_argument(
        "--input-csv",
        default=(
            "results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/"
            "failure_analysis/llm_on_normalized/current_pipeline_fp1_fp7_per_query.csv"
        ),
        help="Per-query failure analysis CSV.",
    )
    p.add_argument(
        "--output-csv",
        default="results/fp2_triage/current_fp2_triage.csv",
        help="Path to write the populated FP2 triage CSV.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    rows: list[dict[str, str]] = []
    with input_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("failure_type") or "").strip() != "FP2_MISSED_TOP_RANK":
                continue
            rows.append(
                {
                    "query_id": str(row.get("query_id") or ""),
                    "document": str(row.get("document") or ""),
                    "difficulty": str(row.get("difficulty") or ""),
                    "question": str(row.get("question") or ""),
                    "expected_answer": str(row.get("expected_answer") or ""),
                    "expected_pages": str(row.get("expected_pages") or ""),
                    "evidence_layout": str(row.get("evidence_layout") or ""),
                    "answer_type": str(row.get("answer_type") or ""),
                    "top1_chunk_id": str(row.get("top1_chunk_id") or ""),
                    "top1_pages": str(row.get("top1_pages") or ""),
                    "top3_chunk_ids": str(row.get("top3_chunk_ids") or ""),
                    "gold_exists": str(row.get("gold_exists") or ""),
                    "gold_pages_found": str(row.get("gold_pages_found") or ""),
                    "fp2_subtype": "",
                    "notes": "",
                    "review_status": "pending",
                    "reviewer": "",
                    "review_date": "",
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} FP2 rows to {output_csv}")


if __name__ == "__main__":
    main()
