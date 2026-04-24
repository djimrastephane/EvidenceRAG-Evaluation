from __future__ import annotations

"""Compare legacy Table 4.1 results against the thesis_rag 224/56 benchmark.

This script packages a thesis-auditable comparison between the legacy Table 4.1
headline metrics and the equivalent aggregate metrics derived from the
refactored ``thesis_rag`` 5-document, 250-query benchmark outputs.

The current implementation reads:

- the legacy reference table from ``results/current_method_comparison_2026-04-07``
- the refactored pipeline outputs from
  ``results/thesis_ablations/chunk_size_ablation_2026-04-15/pipeline_outputs``

It then writes a dated comparison bundle under ``results/thesis_validations`` so
the values can be cited, reviewed, or reproduced by an examiner.
"""

import argparse
import csv
import json
from datetime import date
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEGACY_CSV = REPO_ROOT / "results" / "current_method_comparison_2026-04-07" / "current_method_comparison_aggregate.csv"
DEFAULT_THESIS_RAG_ROOT = REPO_ROOT / "results" / "thesis_ablations" / "chunk_size_ablation_2026-04-15" / "pipeline_outputs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "thesis_validations" / f"table_4_1_comparison_{date.today().isoformat()}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Table 4.1 against thesis_rag 224/56 results.")
    parser.add_argument("--legacy-csv", type=Path, default=DEFAULT_LEGACY_CSV, help="Legacy method-comparison aggregate CSV.")
    parser.add_argument(
        "--thesis-rag-root",
        type=Path,
        default=DEFAULT_THESIS_RAG_ROOT,
        help="Root directory containing thesis_rag 5-doc chunk ablation outputs.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Dated bundle directory.")
    return parser.parse_args()


def _read_legacy_table_4_1(path: Path) -> dict[str, float | int | str]:
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    wanted = {("hybrid_boost", "1"), ("hybrid_boost", "3"), ("hybrid_boost", "10")}
    selected = {
        (row["method"], row["k"]): row
        for row in rows
        if (row["method"], row["k"]) in wanted
    }
    missing = wanted - set(selected)
    if missing:
        raise RuntimeError(f"Missing legacy rows in {path}: {sorted(missing)}")
    k1 = selected[("hybrid_boost", "1")]
    k3 = selected[("hybrid_boost", "3")]
    k10 = selected[("hybrid_boost", "10")]
    return {
        "source": str(path),
        "method": "Hybrid + subsection boost",
        "chunking": "224 / 56",
        "queries_evaluated": int(k10["queries"]),
        "page_hit_at_1": float(k1["weighted_page_hit"]),
        "page_hit_at_3": float(k3["weighted_page_hit"]),
        "mrr_at_10": float(k10["weighted_page_mrr"]),
    }


def _read_thesis_rag_results(root: Path) -> dict[str, float | int | str]:
    result_files = sorted(root.glob("*chunk_224_56/*/per_query_results.json"))
    if len(result_files) != 5:
        raise RuntimeError(f"Expected 5 per_query_results.json files under {root}, found {len(result_files)}")
    all_rows: list[dict[str, object]] = []
    for path in result_files:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if len(rows) != 50:
            raise RuntimeError(f"Expected 50 queries in {path}, found {len(rows)}")
        all_rows.extend(rows)
    total = len(all_rows)
    if total != 250:
        raise RuntimeError(f"Expected 250 total queries, found {total}")
    hit1 = sum(1 for row in all_rows if bool(row["hit_at_1"])) / total
    hit3 = sum(1 for row in all_rows if bool(row["hit_at_3"])) / total
    mrr10 = sum(float(row["reciprocal_rank"]) for row in all_rows) / total
    return {
        "source": str(root),
        "method": "thesis_rag hybrid (legacy-style subsection boost path)",
        "chunking": "224 / 56",
        "queries_evaluated": total,
        "page_hit_at_1": hit1,
        "page_hit_at_3": hit3,
        "mrr_at_10": mrr10,
    }


def _write_bundle(output_dir: Path, legacy: dict[str, float | int | str], thesis_rag: dict[str, float | int | str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = {
        "legacy_table_4_1": legacy,
        "thesis_rag_equivalent": thesis_rag,
        "differences": {
            "page_hit_at_1": float(thesis_rag["page_hit_at_1"]) - float(legacy["page_hit_at_1"]),
            "page_hit_at_3": float(thesis_rag["page_hit_at_3"]) - float(legacy["page_hit_at_3"]),
            "mrr_at_10": float(thesis_rag["mrr_at_10"]) - float(legacy["mrr_at_10"]),
            "queries_evaluated": int(thesis_rag["queries_evaluated"]) - int(legacy["queries_evaluated"]),
        },
    }
    (output_dir / "table_4_1_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    csv_rows = [
        {
            "system": "legacy_table_4_1",
            **legacy,
        },
        {
            "system": "thesis_rag_equivalent",
            **thesis_rag,
        },
        {
            "system": "difference_new_minus_legacy",
            "source": "",
            "method": "",
            "chunking": "224 / 56",
            "queries_evaluated": comparison["differences"]["queries_evaluated"],
            "page_hit_at_1": comparison["differences"]["page_hit_at_1"],
            "page_hit_at_3": comparison["differences"]["page_hit_at_3"],
            "mrr_at_10": comparison["differences"]["mrr_at_10"],
        },
    ]
    with (output_dir / "table_4_1_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    md_lines = [
        "# Table 4.1 Comparison",
        "",
        "| System | Hit@1 | Hit@3 | MRR@10 | Queries |",
        "|---|---:|---:|---:|---:|",
        f"| Legacy Table 4.1 | {float(legacy['page_hit_at_1']):.4f} | {float(legacy['page_hit_at_3']):.4f} | {float(legacy['mrr_at_10']):.4f} | {int(legacy['queries_evaluated'])} |",
        f"| thesis_rag 224/56 | {float(thesis_rag['page_hit_at_1']):.4f} | {float(thesis_rag['page_hit_at_3']):.4f} | {float(thesis_rag['mrr_at_10']):.4f} | {int(thesis_rag['queries_evaluated'])} |",
        f"| Difference (new - legacy) | {comparison['differences']['page_hit_at_1']:+.4f} | {comparison['differences']['page_hit_at_3']:+.4f} | {comparison['differences']['mrr_at_10']:+.4f} | {comparison['differences']['queries_evaluated']:+d} |",
        "",
    ]
    (output_dir / "table_4_1_comparison.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    legacy = _read_legacy_table_4_1(args.legacy_csv)
    thesis_rag = _read_thesis_rag_results(args.thesis_rag_root)
    _write_bundle(args.output_dir, legacy, thesis_rag)
    print(args.output_dir)


if __name__ == "__main__":
    main()
