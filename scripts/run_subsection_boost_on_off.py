from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import pandas as pd


DEFAULT_DOCS = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]

DATA_ROOT = Path("data_processed")
PYTHON_EXE = Path(os.getenv("RAG_PIPELINE_PYTHON", "/opt/anaconda3/envs/rag-pipeline/bin/python"))
MODEL = "models/all-MiniLM-L6-v2"
OUT_ROOT = Path("results/subsection_boost_on_off_2026-04-07")
PREV_ROOT = Path("results/subsection_boost_on_off_2026-03-02")


def run_eval(doc_dir: Path, *, enabled: bool, model: str) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["ENABLE_SUBSECTION_BOOST"] = "1" if enabled else "0"
    cmd = [
        str(PYTHON_EXE),
        "scripts/retrieval_eval_hybrid.py",
        "--data-dir",
        str(doc_dir),
        "--model",
        model,
        "--device",
        "cpu",
    ]
    print("$", " ".join(cmd), f"[subsection_boost={'on' if enabled else 'off'}]")
    subprocess.run(cmd, check=True, env=env)
    return json.loads((doc_dir / "retrieval_metrics_hybrid.json").read_text(encoding="utf-8"))


def copy_outputs(doc_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("retrieval_metrics_hybrid.json", "retrieval_results_hybrid.json", "retrieval_summary_hybrid.csv"):
        src = doc_dir / name
        if src.exists():
            out = out_dir / name
            out.write_bytes(src.read_bytes())


def weighted_delta(df: pd.DataFrame, metric_before: str, metric_after: str, weight_col: str = "queries") -> float:
    weights = df[weight_col].astype(float)
    delta = df[metric_after].astype(float) - df[metric_before].astype(float)
    return float((delta * weights).sum() / weights.sum())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run subsection boost on/off ablation for the five Grampian reports.")
    parser.add_argument("--docs", nargs="*", default=DEFAULT_DOCS)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--previous-root", type=Path, default=PREV_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []

    for mode in ("off", "on"):
        enabled = mode == "on"
        for doc in args.docs:
            doc_dir = DATA_ROOT / doc
            metrics = run_eval(doc_dir, enabled=enabled, model=args.model)
            copy_outputs(doc_dir, args.out_root / mode / doc)
            q = int(metrics["metrics_by_k"]["1"]["num_queries"])
            for k in ("1", "3", "5", "10"):
                m = metrics["metrics_by_k"][k]
                rows.append(
                    {
                        "doc_id": doc,
                        "mode": mode,
                        "queries": q,
                        "k": int(k),
                        "page_hit_rate": float(m["page_hit_rate_at_k"]),
                        "page_mrr": float(m["mean_page_mrr_at_k"]),
                        "chunk_hit_rate": float(m["chunk_hit_rate_at_k"]),
                        "chunk_mrr": float(m["mean_chunk_mrr_at_k"]),
                    }
                )

    df = pd.DataFrame(rows)
    by_doc_k = []
    for doc in args.docs:
        for k in (1, 3, 5, 10):
            off = df[(df["doc_id"] == doc) & (df["mode"] == "off") & (df["k"] == k)].iloc[0]
            on = df[(df["doc_id"] == doc) & (df["mode"] == "on") & (df["k"] == k)].iloc[0]
            by_doc_k.append(
                {
                    "doc_id": doc,
                    "queries": int(off["queries"]),
                    "k": k,
                    "page_hit_off": off["page_hit_rate"],
                    "page_hit_on": on["page_hit_rate"],
                    "page_hit_delta": on["page_hit_rate"] - off["page_hit_rate"],
                    "page_mrr_off": off["page_mrr"],
                    "page_mrr_on": on["page_mrr"],
                    "page_mrr_delta": on["page_mrr"] - off["page_mrr"],
                    "chunk_hit_off": off["chunk_hit_rate"],
                    "chunk_hit_on": on["chunk_hit_rate"],
                    "chunk_hit_delta": on["chunk_hit_rate"] - off["chunk_hit_rate"],
                    "chunk_mrr_off": off["chunk_mrr"],
                    "chunk_mrr_on": on["chunk_mrr"],
                    "chunk_mrr_delta": on["chunk_mrr"] - off["chunk_mrr"],
                }
            )
    by_doc_k_df = pd.DataFrame(by_doc_k)
    by_doc_k_df.to_csv(args.out_root / "subsection_boost_on_off_by_doc_k.csv", index=False)

    aggregate_rows = []
    for k in (1, 3, 5, 10):
        sub = by_doc_k_df[by_doc_k_df["k"] == k]
        aggregate_rows.append(
            {
                "k": k,
                "weighted_page_hit_delta": weighted_delta(sub, "page_hit_off", "page_hit_on"),
                "weighted_page_mrr_delta": weighted_delta(sub, "page_mrr_off", "page_mrr_on"),
                "weighted_chunk_hit_delta": weighted_delta(sub, "chunk_hit_off", "chunk_hit_on"),
                "weighted_chunk_mrr_delta": weighted_delta(sub, "chunk_mrr_off", "chunk_mrr_on"),
            }
        )
    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_df.to_csv(args.out_root / "subsection_boost_on_off_aggregate.csv", index=False)

    comparison_lines = []
    prev_agg_path = args.previous_root / "subsection_boost_on_off_aggregate.csv"
    if prev_agg_path.exists():
        prev = pd.read_csv(prev_agg_path)
        cur = aggregate_df.merge(prev, on="k", suffixes=("_current", "_previous"))
        cur["page_hit_delta_change_vs_previous"] = cur["weighted_page_hit_delta_current"] - cur["weighted_page_hit_delta_previous"]
        cur["page_mrr_delta_change_vs_previous"] = cur["weighted_page_mrr_delta_current"] - cur["weighted_page_mrr_delta_previous"]
        cur.to_csv(args.out_root / "subsection_boost_on_off_vs_previous.csv", index=False)
        comparison_lines.append("## Change Vs Previous Run")
        for _, row in cur.iterrows():
            comparison_lines.append(
                f"- k={int(row['k'])}: hit delta {row['weighted_page_hit_delta_previous']:+.4f} -> {row['weighted_page_hit_delta_current']:+.4f}; "
                f"MRR delta {row['weighted_page_mrr_delta_previous']:+.4f} -> {row['weighted_page_mrr_delta_current']:+.4f}"
            )
        comparison_lines.append("")

    report = [
        "# Subsection Boost ON vs OFF (updated eval subsections)",
        "",
        f"- Runs OK: `{len(args.docs) * 2}` / `{len(args.docs) * 2}`",
        "",
        "## Weighted deltas (ON - OFF)",
    ]
    for _, row in aggregate_df.iterrows():
        report.append(
            f"- k={int(row['k'])}: "
            f"Hit delta `{row['weighted_page_hit_delta']:+.4f}`, "
            f"MRR delta `{row['weighted_page_mrr_delta']:+.4f}`, "
            f"Chunk hit delta `{row['weighted_chunk_hit_delta']:+.4f}`, "
            f"Chunk MRR delta `{row['weighted_chunk_mrr_delta']:+.4f}`"
        )
    report.append("")
    report.extend(comparison_lines)
    (args.out_root / "subsection_boost_on_off_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    summary = {
        "docs": args.docs,
        "model": args.model,
        "aggregate": aggregate_rows,
        "previous_root": str(args.previous_root),
    }
    (args.out_root / "subsection_boost_on_off_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote: {args.out_root / 'subsection_boost_on_off_report.md'}")


if __name__ == "__main__":
    main()
