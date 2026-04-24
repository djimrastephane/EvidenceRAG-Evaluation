from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from scripts._matplotlib_env import configure_matplotlib_env
except ImportError:
    from _matplotlib_env import configure_matplotlib_env

configure_matplotlib_env()

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a comparison chart and leakage table for doc-vs-global retrieval scope runs."
    )
    parser.add_argument("--doc-metrics", required=True, type=Path)
    parser.add_argument("--global-metrics", required=True, type=Path)
    parser.add_argument("--global-results", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=15, help="How many leaked queries to include in the table.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_metrics_frame(doc_metrics: dict, global_metrics: dict) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    k_values = sorted(int(k) for k in doc_metrics["metrics_by_k"].keys())
    for k in k_values:
        sk = str(k)
        d = doc_metrics["metrics_by_k"][sk]
        g = global_metrics["metrics_by_k"][sk]
        gl = global_metrics.get("leakage_counts_by_k", {}).get(sk, {})
        rows.append(
            {
                "k": k,
                "doc_hit_rate": float(d["page_hit_rate_at_k"]),
                "global_hit_rate": float(g["page_hit_rate_at_k"]),
                "doc_mrr": float(d["mean_page_mrr_at_k"]),
                "global_mrr": float(g["mean_page_mrr_at_k"]),
                "global_any_leakage_rate": float(gl.get("any_leakage_rate_at_k", 0.0)),
                "global_mean_leakage_rate": float(gl.get("mean_leakage_rate_at_k", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def build_leakage_frame(global_results: dict, top_n: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in global_results.get("results", []):
        k1 = item.get("per_k", {}).get("1", {})
        retrieved_doc_ids = list(k1.get("retrieved_doc_ids_top_k", []) or [])
        top_doc = str(retrieved_doc_ids[0]) if retrieved_doc_ids else ""
        expected_doc = str(item.get("doc_id") or "")
        if top_doc and expected_doc and top_doc != expected_doc:
            rows.append(
                {
                    "query_id": str(item.get("query_id") or ""),
                    "question": str(item.get("question") or ""),
                    "expected_doc_id": expected_doc,
                    "top1_doc_id": top_doc,
                    "top1_pages": ", ".join(str(x) for x in (k1.get("retrieved_pages_ranked", []) or [])[:3]),
                    "leakage_rate_top_10": float(item.get("per_k", {}).get("10", {}).get("leakage_rate_top_k", 0.0)),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["leakage_rate_top_10", "query_id"], ascending=[False, True]).head(top_n)


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        path.write_text("No leaked top-1 queries found.\n", encoding="utf-8")
        return
    path.write_text(df.to_markdown(index=False), encoding="utf-8")


def plot_metrics(df: pd.DataFrame, out_path: Path) -> None:
    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)
    fig.patch.set_facecolor("#F8FAFC")

    blue = "#355C8C"
    slate = "#94A3B8"
    vermilion = "#B45309"
    ink = "#334155"
    grid = "#E2E8F0"

    ax = axes[0]
    ax.plot(df["k"], df["doc_hit_rate"], marker="o", linewidth=2.2, color=blue, label="Document scope")
    ax.plot(df["k"], df["global_hit_rate"], marker="o", linewidth=2.2, color=slate, label="Global scope")
    ax.set_title("Page Hit Rate by Retrieval Scope", color=ink, fontsize=12, weight="bold")
    ax.set_xlabel("k", color=ink)
    ax.set_ylabel("Hit rate", color=ink)
    ax.set_xticks(df["k"].tolist())
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.tick_params(colors=ink)
    ax.legend(frameon=False, loc="lower right")

    ax2 = axes[1]
    ax2.bar(df["k"], df["global_any_leakage_rate"], width=1.2, color=vermilion, alpha=0.9)
    ax2.set_title("Wrong-Document Leakage in Global Scope", color=ink, fontsize=12, weight="bold")
    ax2.set_xlabel("k", color=ink)
    ax2.set_ylabel("Queries with leakage", color=ink)
    ax2.set_xticks(df["k"].tolist())
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis="y", color=grid, linewidth=0.8)
    ax2.tick_params(colors=ink)

    for axis in axes:
        axis.set_facecolor("white")
        for spine in axis.spines.values():
            spine.set_color("#CBD5E1")

    fig.suptitle(
        "Grampian-2020-2021 Queries: Document vs Global Retrieval Scope",
        fontsize=14,
        weight="bold",
        color="#1E293B",
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_metrics = load_json(args.doc_metrics.resolve())
    global_metrics = load_json(args.global_metrics.resolve())
    global_results = load_json(args.global_results.resolve())

    metrics_df = build_metrics_frame(doc_metrics, global_metrics)
    leakage_df = build_leakage_frame(global_results, top_n=int(args.top_n))

    metrics_csv = out_dir / "doc_vs_global_scope_metrics.csv"
    leakage_csv = out_dir / "global_scope_top1_leakage_examples.csv"
    leakage_md = out_dir / "global_scope_top1_leakage_examples.md"
    chart_png = out_dir / "doc_vs_global_scope_comparison.png"

    metrics_df.to_csv(metrics_csv, index=False)
    leakage_df.to_csv(leakage_csv, index=False)
    write_markdown_table(leakage_df, leakage_md)
    plot_metrics(metrics_df, chart_png)

    print(f"Saved: {metrics_csv}")
    print(f"Saved: {leakage_csv}")
    print(f"Saved: {leakage_md}")
    print(f"Saved: {chart_png}")


if __name__ == "__main__":
    main()
