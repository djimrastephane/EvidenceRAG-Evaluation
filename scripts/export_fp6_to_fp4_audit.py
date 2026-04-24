from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an audit sheet for FP6 -> FP4 transitions after enabling LLM generation.")
    parser.add_argument(
        "--baseline-per-query",
        default="results/live_fp1_fp7_current_pipeline_2026-03-17/current_pipeline_fp1_fp7_per_query.csv",
    )
    parser.add_argument(
        "--candidate-per-query",
        default="results/live_fp1_fp7_current_pipeline_llm_2026-03-17/current_pipeline_fp1_fp7_per_query.csv",
    )
    parser.add_argument(
        "--comparison-per-query",
        default="results/live_fp1_fp7_compare_llm_vs_retrieval/fp1_fp7_per_query_comparison.csv",
    )
    parser.add_argument(
        "--out-csv",
        default="results/live_fp1_fp7_compare_llm_vs_retrieval/fp6_to_fp4_audit.csv",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    baseline_rows = load_rows((REPO_ROOT / args.baseline_per_query).resolve())
    candidate_rows = load_rows((REPO_ROOT / args.candidate_per_query).resolve())
    comparison_rows = load_rows((REPO_ROOT / args.comparison_per_query).resolve())
    out_csv = (REPO_ROOT / args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    baseline_map = {(r["document"], r["query_id"]): r for r in baseline_rows}
    candidate_map = {(r["document"], r["query_id"]): r for r in candidate_rows}

    filtered = [
        r
        for r in comparison_rows
        if str(r.get("baseline_failure_type") or "") == "FP6_INCORRECT_SPECIFICITY"
        and str(r.get("candidate_failure_type") or "") == "FP4_NOT_EXTRACTED"
    ]

    output_rows: list[dict[str, str]] = []
    for row in filtered:
        key = (str(row.get("document") or ""), str(row.get("query_id") or ""))
        base = baseline_map.get(key, {})
        cand = candidate_map.get(key, {})
        output_rows.append(
            {
                "document": key[0],
                "query_id": key[1],
                "difficulty": str(cand.get("difficulty") or base.get("difficulty") or ""),
                "question": str(row.get("question") or cand.get("question") or base.get("question") or ""),
                "expected_answer": str(cand.get("expected_answer") or base.get("expected_answer") or ""),
                "expected_pages": str(cand.get("expected_pages") or base.get("expected_pages") or ""),
                "expected_section": str(cand.get("expected_section") or base.get("expected_section") or ""),
                "expected_subsection": str(cand.get("expected_subsection") or base.get("expected_subsection") or ""),
                "evidence_layout": str(cand.get("evidence_layout") or base.get("evidence_layout") or ""),
                "answer_type": str(cand.get("answer_type") or base.get("answer_type") or ""),
                "baseline_failure_type": str(row.get("baseline_failure_type") or ""),
                "candidate_failure_type": str(row.get("candidate_failure_type") or ""),
                "baseline_extracted_answer": str(row.get("baseline_extracted_answer") or ""),
                "candidate_extracted_answer": str(row.get("candidate_extracted_answer") or ""),
                "candidate_generated_answer": str(cand.get("generated_answer") or ""),
                "candidate_generation_status": str(cand.get("generation_status") or ""),
                "candidate_generation_confidence": str(cand.get("generation_confidence") or ""),
                "page_hit": str(cand.get("page_hit") or ""),
                "gold_exists": str(cand.get("gold_exists") or ""),
                "gold_chunk_count": str(cand.get("gold_chunk_count") or ""),
                "gold_pages_found": str(cand.get("gold_pages_found") or ""),
                "top1_chunk_id": str(cand.get("top1_chunk_id") or ""),
                "top1_pages": str(cand.get("top1_pages") or ""),
                "top3_chunk_ids": str(cand.get("top3_chunk_ids") or ""),
                "top3_pages": str(cand.get("top3_pages") or ""),
            }
        )

    fieldnames = list(output_rows[0].keys()) if output_rows else [
        "document",
        "query_id",
        "difficulty",
        "question",
        "expected_answer",
        "expected_pages",
        "expected_section",
        "expected_subsection",
        "evidence_layout",
        "answer_type",
        "baseline_failure_type",
        "candidate_failure_type",
        "baseline_extracted_answer",
        "candidate_extracted_answer",
        "candidate_generated_answer",
        "candidate_generation_status",
        "candidate_generation_confidence",
        "page_hit",
        "gold_exists",
        "gold_chunk_count",
        "gold_pages_found",
        "top1_chunk_id",
        "top1_pages",
        "top3_chunk_ids",
        "top3_pages",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print({"out_csv": str(out_csv), "row_count": len(output_rows)})


if __name__ == "__main__":
    main()
