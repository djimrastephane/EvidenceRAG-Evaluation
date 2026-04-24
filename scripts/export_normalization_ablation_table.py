from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a compact CSV table from normalization_ablation_report.json."
    )
    parser.add_argument(
        "--report-json",
        required=True,
        help="Path to normalization_ablation_report.json produced by ablate_embedding_normalization_mode.py",
    )
    parser.add_argument(
        "--out-csv",
        default="",
        help="Output CSV path. Defaults to <report-dir>/normalization_ablation_table.csv",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return obj


def flatten_report(payload: dict[str, Any]) -> pd.DataFrame:
    run_info = payload.get("run_info", {})
    emb = payload.get("embedding_comparison", {})
    retrieval = payload.get("retrieval_comparison", {})
    baseline_metrics = payload.get("baseline_metrics_by_k", {})
    variant_metrics = payload.get("variant_metrics_by_k", {})
    deltas = retrieval.get("metric_deltas_by_k", {})

    all_ks = sorted(
        {
            int(k)
            for k in list(baseline_metrics.keys()) + list(variant_metrics.keys()) + list(deltas.keys())
            if str(k).isdigit()
        }
    )
    if not all_ks:
        raise ValueError("No k-indexed metrics found in report.")

    rows: list[dict[str, Any]] = []
    for k in all_ks:
        ks = str(k)
        base = baseline_metrics.get(ks, {})
        var = variant_metrics.get(ks, {})
        delta = deltas.get(ks, {})
        rows.append(
            {
                "data_dir": run_info.get("data_dir"),
                "embedding_model": run_info.get("embedding_model"),
                "k": int(k),
                "baseline_page_hit_rate": base.get("page_hit_rate_at_k"),
                "variant_page_hit_rate": var.get("page_hit_rate_at_k"),
                "delta_page_hit_rate": delta.get("page_hit_rate_delta"),
                "baseline_mean_page_mrr": base.get("mean_page_mrr_at_k"),
                "variant_mean_page_mrr": var.get("mean_page_mrr_at_k"),
                "delta_mean_page_mrr": delta.get("mean_page_mrr_delta"),
                "baseline_chunk_hit_rate": base.get("chunk_hit_rate_at_k"),
                "variant_chunk_hit_rate": var.get("chunk_hit_rate_at_k"),
                "delta_chunk_hit_rate": delta.get("chunk_hit_rate_delta"),
                "top1_identical_rate": retrieval.get("top1_identical_rate"),
                "topk_sequence_identical_rate": retrieval.get("topk_sequence_identical_rate"),
                "hit_at_1_identical_rate": retrieval.get("hit_at_1_identical_rate"),
                "differing_query_count": retrieval.get("differing_query_count"),
                "num_queries": retrieval.get("num_queries"),
                "doc_emb_max_abs_diff": emb.get("document_embeddings", {}).get("max_abs_diff"),
                "doc_emb_mean_abs_diff": emb.get("document_embeddings", {}).get("mean_abs_diff"),
                "doc_emb_allclose_1e_6": emb.get("document_embeddings", {}).get("allclose_atol_1e-6"),
                "query_emb_max_abs_diff": emb.get("query_embeddings", {}).get("max_abs_diff"),
                "query_emb_mean_abs_diff": emb.get("query_embeddings", {}).get("mean_abs_diff"),
                "query_emb_allclose_1e_6": emb.get("query_embeddings", {}).get("allclose_atol_1e-6"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    report_json = Path(args.report_json).resolve()
    if not report_json.exists():
        raise FileNotFoundError(f"Missing report JSON: {report_json}")
    out_csv = Path(args.out_csv).resolve() if args.out_csv else report_json.with_name("normalization_ablation_table.csv")

    payload = load_json(report_json)
    df = flatten_report(payload)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"Saved CSV: {out_csv}")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
