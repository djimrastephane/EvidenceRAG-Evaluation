#!/usr/bin/env python3
"""Plot MiniLM-cap chunk-size ablation trends from the retrieval ablation summary."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import _matplotlib_env
import matplotlib.pyplot as plt
import pandas as pd


MINILM_CAP = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot chunk-size ablation trends against the MiniLM token cap.")
    parser.add_argument(
        "--summary-csv",
        default="results/ablations/ablation_minilm_cap_5docs/retrieval_ablation_summary.csv",
        help="Input ablation summary CSV.",
    )
    parser.add_argument(
        "--output",
        default="docs/figures/minilm_cap_ablation_trend.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--ablation-root",
        default="data_variants/ablation_minilm_cap_5docs",
        help="Fallback root containing per-experiment directories.",
    )
    return parser.parse_args()


def weighted_mean(df: pd.DataFrame, value_col: str, weight_col: str = "scored_queries") -> float:
    weights = df[weight_col].astype(float)
    values = df[value_col].astype(float)
    return float((values * weights).sum() / weights.sum())


def build_plot_frame(summary_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    if df.empty:
        raise RuntimeError(f"Summary CSV is empty: {summary_csv}")

    rows: list[dict[str, float | int]] = []
    for chunk_size, group in df.groupby("chunk_size_tokens"):
        k1 = group[group["k"] == 1].copy()
        k10 = group[group["k"] == 10].copy()
        if k1.empty or k10.empty:
            continue
        rows.append(
            {
                "chunk_size": int(chunk_size),
                "overlap": int(k1["chunk_overlap_tokens"].iloc[0]),
                "page_hit_at_1": weighted_mean(k1, "page_hit_rate"),
                "page_mrr_at_10": weighted_mean(k10, "page_mrr"),
                "queries": int(k1["scored_queries"].sum()),
                "chunks_indexed": int(group["data_dir"].nunique()),
            }
        )

    out = pd.DataFrame(rows).sort_values("chunk_size").reset_index(drop=True)
    if out.empty:
        raise RuntimeError("No valid chunk-size rows found in summary CSV.")
    return out


def build_plot_frame_from_ablation_root(ablation_root: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    pattern = re.compile(r"chunk_(\d+)_(\d+)$")

    for exp_dir in sorted(ablation_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        match = pattern.search(exp_dir.name)
        if not match:
            continue
        chunk_size = int(match.group(1))
        overlap = int(match.group(2))
        doc_dirs = [p for p in exp_dir.iterdir() if p.is_dir()]
        if not doc_dirs:
            continue
        metrics_path = doc_dirs[0] / "retrieval_metrics.json"
        if not metrics_path.exists():
            continue
        obj = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics_by_k = obj.get("metrics_by_k", {})
        if "1" not in metrics_by_k or "10" not in metrics_by_k:
            continue
        rows.append(
            {
                "chunk_size": chunk_size,
                "overlap": overlap,
                "page_hit_at_1": float(metrics_by_k["1"]["page_hit_rate_at_k"]),
                "page_mrr_at_10": float(metrics_by_k["10"]["mean_page_mrr_at_k"]),
                "queries": int(obj.get("answer_scoring", {}).get("scored_queries") or 50),
                "doc_id": doc_dirs[0].name,
            }
        )

    if not rows:
        raise RuntimeError(f"No valid retrieval metrics found under {ablation_root}")

    df = pd.DataFrame(rows)
    out_rows: list[dict[str, float | int]] = []
    for chunk_size, group in df.groupby("chunk_size"):
        weights = group["queries"].astype(float)
        out_rows.append(
            {
                "chunk_size": int(chunk_size),
                "overlap": int(group["overlap"].iloc[0]),
                "page_hit_at_1": float((group["page_hit_at_1"] * weights).sum() / weights.sum()),
                "page_mrr_at_10": float((group["page_mrr_at_10"] * weights).sum() / weights.sum()),
                "queries": int(weights.sum()),
            }
        )
    return pd.DataFrame(out_rows).sort_values("chunk_size").reset_index(drop=True)


def plot_trend(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2), sharex=True)
    fig.patch.set_facecolor("white")

    x = df["chunk_size"].astype(float).to_numpy()
    labels = [f"{int(r.chunk_size)}/{int(r.overlap)}" for r in df.itertuples()]

    # Shared visual cues: cap boundary and over-cap region.
    for ax in axes:
        ax.axvspan(MINILM_CAP, float(x.max()) + 16.0, color="#f2d7d5", alpha=0.35, zorder=0)
        ax.axvline(MINILM_CAP, linestyle="--", linewidth=1.4, color="#7f8c8d", zorder=1)
        ax.grid(True, axis="y", alpha=0.28, linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_xlabel("Chunk size / overlap")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            MINILM_CAP + 3,
            0.97,
            "MiniLM cap",
            transform=ax.get_xaxis_transform(),
            fontsize=9,
            color="#566573",
            ha="left",
            va="top",
        )

    axes[0].plot(
        x,
        df["page_hit_at_1"],
        marker="o",
        markersize=6.5,
        markerfacecolor="white",
        markeredgewidth=1.5,
        linewidth=2.2,
        color="#1f77b4",
    )
    axes[0].set_ylabel("Page Hit@1")
    axes[0].set_title("Top-Rank Retrieval")
    hit_vals = df["page_hit_at_1"].astype(float)
    axes[0].set_ylim(hit_vals.min() - 0.03, hit_vals.max() + 0.05)

    axes[1].plot(
        x,
        df["page_mrr_at_10"],
        marker="o",
        markersize=6.5,
        markerfacecolor="white",
        markeredgewidth=1.5,
        linewidth=2.2,
        color="#b9770e",
    )
    axes[1].set_ylabel("MRR@10")
    axes[1].set_title("Ranking Quality")
    mrr_vals = df["page_mrr_at_10"].astype(float)
    axes[1].set_ylim(mrr_vals.min() - 0.03, mrr_vals.max() + 0.05)

    fig.suptitle("Chunk-Size Ablation Relative to the MiniLM 256-Token Cap", fontsize=14, fontweight="bold", y=1.02)
    fig.text(
        0.5,
        0.98,
        "250-query weighted summary across the 5-document Grampian evaluation set; shaded region marks chunk sizes above the embedding limit.",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#4d4d4d",
    )

    axes[1].annotate(
        "Over-cap chunk sizes do not improve retrieval\nand the explicit 400/100 condition declines further.",
        xy=(400, float(df.loc[df["chunk_size"] == 400, "page_mrr_at_10"].iloc[0])),
        xytext=(287, float(df["page_mrr_at_10"].min()) + 0.025),
        fontsize=9.5,
        bbox={"facecolor": "white", "edgecolor": "0.75", "boxstyle": "round,pad=0.35"},
        arrowprops={"arrowstyle": "->", "linewidth": 0.9, "color": "#555555"},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=1.2)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv)
    output_path = Path(args.output)
    ablation_root = Path(args.ablation_root)

    df = build_plot_frame(summary_csv)
    if df["chunk_size"].nunique() < 2:
        df = build_plot_frame_from_ablation_root(ablation_root)
    plot_trend(df, output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
