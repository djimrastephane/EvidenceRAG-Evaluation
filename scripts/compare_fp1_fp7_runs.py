from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two FP1-FP7 evaluation runs and export counts/per-query deltas."
    )
    parser.add_argument(
        "--baseline-dir",
        default="results/live_fp1_fp7_current_pipeline_2026-03-17",
        help="Directory containing baseline current_pipeline_fp1_fp7_{per_query,counts,summary} artifacts.",
    )
    parser.add_argument(
        "--candidate-dir",
        required=True,
        help="Directory containing candidate current_pipeline_fp1_fp7_{per_query,counts,summary} artifacts.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/live_fp1_fp7_compare",
        help="Directory to write comparison outputs.",
    )
    return parser.parse_args()


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fail_sort_key(code: str) -> tuple[int, str]:
    text = str(code or "").strip().upper()
    if text == "HIT":
        return (999, text)
    if text.startswith("FP"):
        digits = "".join(ch for ch in text[2:] if ch.isdigit())
        if digits:
            return (int(digits), text)
    return (998, text)


def main() -> None:
    args = parse_args()
    baseline_dir = (REPO_ROOT / args.baseline_dir).resolve()
    candidate_dir = (REPO_ROOT / args.candidate_dir).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_per_query = load_csv_rows(baseline_dir / "current_pipeline_fp1_fp7_per_query.csv")
    candidate_per_query = load_csv_rows(candidate_dir / "current_pipeline_fp1_fp7_per_query.csv")
    baseline_counts = load_csv_rows(baseline_dir / "current_pipeline_fp1_fp7_counts.csv")
    candidate_counts = load_csv_rows(candidate_dir / "current_pipeline_fp1_fp7_counts.csv")
    baseline_summary = load_summary(baseline_dir / "current_pipeline_fp1_fp7_summary.json")
    candidate_summary = load_summary(candidate_dir / "current_pipeline_fp1_fp7_summary.json")

    baseline_map = {
        (str(r.get("document") or "").strip(), str(r.get("query_id") or "").strip()): r for r in baseline_per_query
    }
    candidate_map = {
        (str(r.get("document") or "").strip(), str(r.get("query_id") or "").strip()): r for r in candidate_per_query
    }
    shared_keys = sorted(set(baseline_map) & set(candidate_map))

    per_query_rows: list[dict[str, object]] = []
    transition_counter: Counter[tuple[str, str]] = Counter()
    stage_change_counter: Counter[tuple[str, str]] = Counter()
    for key in shared_keys:
        base = baseline_map[key]
        cand = candidate_map[key]
        base_ft = str(base.get("failure_type") or "").strip()
        cand_ft = str(cand.get("failure_type") or "").strip()
        transition_counter[(base_ft, cand_ft)] += 1
        stage_change_counter[
            (str(base.get("failure_stage") or "").strip(), str(cand.get("failure_stage") or "").strip())
        ] += 1
        per_query_rows.append(
            {
                "document": key[0],
                "query_id": key[1],
                "question": str(base.get("question") or cand.get("question") or ""),
                "baseline_failure_type": base_ft,
                "candidate_failure_type": cand_ft,
                "baseline_failure_stage": str(base.get("failure_stage") or ""),
                "candidate_failure_stage": str(cand.get("failure_stage") or ""),
                "baseline_extracted_answer": str(base.get("extracted_answer") or ""),
                "candidate_extracted_answer": str(cand.get("extracted_answer") or ""),
                "baseline_generation_status": str(base.get("generation_status") or ""),
                "candidate_generation_status": str(cand.get("generation_status") or ""),
                "baseline_answer_status": str(base.get("answer_status") or ""),
                "candidate_answer_status": str(cand.get("answer_status") or ""),
                "changed": "yes" if base_ft != cand_ft else "no",
                "improved_to_hit": "yes" if base_ft != "HIT" and cand_ft == "HIT" else "no",
                "regressed_from_hit": "yes" if base_ft == "HIT" and cand_ft != "HIT" else "no",
            }
        )

    baseline_count_map = {
        (str(r.get("series") or "").strip(), str(r.get("fp_code") or "").strip()): int(r.get("count") or 0)
        for r in baseline_counts
    }
    candidate_count_map = {
        (str(r.get("series") or "").strip(), str(r.get("fp_code") or "").strip()): int(r.get("count") or 0)
        for r in candidate_counts
    }
    count_keys = sorted(set(baseline_count_map) | set(candidate_count_map))

    counts_delta_rows: list[dict[str, object]] = []
    totals_by_code: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for series, fp_code in count_keys:
        base_count = baseline_count_map.get((series, fp_code), 0)
        cand_count = candidate_count_map.get((series, fp_code), 0)
        totals_by_code[fp_code][0] += base_count
        totals_by_code[fp_code][1] += cand_count
        counts_delta_rows.append(
            {
                "document": series,
                "failure_type": fp_code,
                "baseline_count": base_count,
                "candidate_count": cand_count,
                "delta": cand_count - base_count,
            }
        )

    totals_rows = []
    for fp_code in sorted(totals_by_code, key=fail_sort_key):
        base_total, cand_total = totals_by_code[fp_code]
        totals_rows.append(
            {
                "document": "ALL",
                "failure_type": fp_code,
                "baseline_count": base_total,
                "candidate_count": cand_total,
                "delta": cand_total - base_total,
            }
        )
    counts_delta_rows.extend(totals_rows)

    transitions_rows = []
    for (base_ft, cand_ft), count in sorted(
        transition_counter.items(),
        key=lambda item: (fail_sort_key(item[0][0]), fail_sort_key(item[0][1])),
    ):
        transitions_rows.append(
            {
                "baseline_failure_type": base_ft,
                "candidate_failure_type": cand_ft,
                "count": count,
            }
        )

    summary = {
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "baseline_total_queries": int(baseline_summary.get("total_queries", len(baseline_per_query))),
        "candidate_total_queries": int(candidate_summary.get("total_queries", len(candidate_per_query))),
        "shared_queries": int(len(shared_keys)),
        "queries_changed_failure_type": int(sum(1 for row in per_query_rows if row["changed"] == "yes")),
        "improved_to_hit": int(sum(1 for row in per_query_rows if row["improved_to_hit"] == "yes")),
        "regressed_from_hit": int(sum(1 for row in per_query_rows if row["regressed_from_hit"] == "yes")),
        "baseline_failure_counts_total": baseline_summary.get("failure_counts_total", {}),
        "candidate_failure_counts_total": candidate_summary.get("failure_counts_total", {}),
        "stage_transitions": {
            f"{base_stage}->{cand_stage}": count
            for (base_stage, cand_stage), count in sorted(stage_change_counter.items())
        },
    }

    per_query_path = out_dir / "fp1_fp7_per_query_comparison.csv"
    counts_delta_path = out_dir / "fp1_fp7_counts_delta.csv"
    transitions_path = out_dir / "fp1_fp7_transition_matrix.csv"
    summary_path = out_dir / "fp1_fp7_comparison_summary.json"

    with per_query_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_query_rows[0].keys()) if per_query_rows else [])
        if per_query_rows:
            writer.writeheader()
            writer.writerows(per_query_rows)

    with counts_delta_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(counts_delta_rows[0].keys()) if counts_delta_rows else [])
        if counts_delta_rows:
            writer.writeheader()
            writer.writerows(counts_delta_rows)

    with transitions_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(transitions_rows[0].keys()) if transitions_rows else [])
        if transitions_rows:
            writer.writeheader()
            writer.writerows(transitions_rows)

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary_json": str(summary_path), "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
