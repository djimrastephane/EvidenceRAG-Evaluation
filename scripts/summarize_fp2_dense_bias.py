from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create compact summaries from the FP2 dense-bias comparison table.")
    parser.add_argument(
        "--input-csv",
        default="results/live_fp2_audit/fp2_before_after_dense_bias.csv",
        help="Comparison CSV created by compare_fp2_dense_bias.py",
    )
    parser.add_argument(
        "--recovered-csv",
        default="results/live_fp2_audit/fp2_dense_bias_recovered_only.csv",
        help="Output CSV with only recovered cases.",
    )
    parser.add_argument(
        "--remaining-csv",
        default="results/live_fp2_audit/fp2_dense_bias_worst_remaining.csv",
        help="Output CSV with worst remaining misses.",
    )
    parser.add_argument(
        "--remaining-limit",
        type=int,
        default=20,
        help="How many remaining misses to keep, sorted worst-first by variant gold hybrid rank.",
    )
    return parser.parse_args()


def _rank_key(value: str) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return 10**9


def main() -> None:
    args = parse_args()
    input_csv = (REPO_ROOT / args.input_csv).resolve()
    recovered_csv = (REPO_ROOT / args.recovered_csv).resolve()
    remaining_csv = (REPO_ROOT / args.remaining_csv).resolve()

    with input_csv.open() as f:
        rows = list(csv.DictReader(f))
        headers = rows[0].keys() if rows else []

    recovered = [row for row in rows if str(row.get("recovered_to_hit_at_1", "")).strip().lower() == "yes"]
    remaining = [row for row in rows if str(row.get("variant_failure_type", "")).strip() == "FP2_MISSED_TOP_RANK"]
    remaining.sort(
        key=lambda row: (
            -_rank_key(row.get("variant_gold_hybrid_rank", "")),
            -_rank_key(row.get("baseline_gold_hybrid_rank", "")),
            row.get("document", ""),
            row.get("query_id", ""),
        )
    )
    remaining = remaining[: max(0, int(args.remaining_limit))]

    recovered_csv.parent.mkdir(parents=True, exist_ok=True)
    with recovered_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(headers))
        writer.writeheader()
        writer.writerows(recovered)

    remaining_csv.parent.mkdir(parents=True, exist_ok=True)
    with remaining_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(headers))
        writer.writeheader()
        writer.writerows(remaining)

    print(
        {
            "recovered_csv": str(recovered_csv),
            "recovered_count": len(recovered),
            "remaining_csv": str(remaining_csv),
            "remaining_count": len(remaining),
        }
    )


if __name__ == "__main__":
    main()
