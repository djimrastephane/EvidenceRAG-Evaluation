"""plot_fp1_fp7_side_by_side_norm.py

Generates a side-by-side normalised FP1-FP7 failure heatmap from two
counts CSV files (e.g. retrieval-only vs LLM-on runs).

Each cell shows: count\n(pct%)
Colourmap is normalised per-panel so that the darkest cell in each panel
is the same shade regardless of absolute counts.

Usage:
    python scripts/plot_fp1_fp7_side_by_side_norm.py \
        --left-csv  results/fp1_fp7_retrieval_boost_off_2026-04-20/current_pipeline_fp1_fp7_counts.csv \
        --right-csv results/fp1_fp7_llm_boost_off_2026-04-20/current_pipeline_fp1_fp7_counts.csv \
        --left-title "Retrieval only" \
        --right-title "LLM-assisted" \
        --queries-per-series 50 \
        --output results/figure_d1_fp1_fp7_postfix_2026-04-20/fp1_fp7_heatmaps_side_by_side_norm_labeled.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FP_ORDER = ["FP1", "FP2", "FP3", "FP4", "FP5", "FP6", "FP7"]

FP_LABELS = {
    "FP1": "Missing\ncontent",
    "FP2": "Missed top-\nranked result",
    "FP3": "Not in\ncontext",
    "FP4": "Not extracted\nfrom context",
    "FP5": "Incorrect\noutput format",
    "FP6": "Incorrect\nspecificity",
    "FP7": "Incomplete\nanswer",
}

DOC_ORDER_KEY = {
    "2020-2021": 0,
    "2021-2022": 1,
    "2022-2023": 2,
    "2023-2024": 3,
    "2024-2025": 4,
}


def _sort_series(s: str) -> int:
    return DOC_ORDER_KEY.get(str(s).strip(), 99)


def load_counts(csv_path: Path, queries_per_series: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["series"] = df["series"].astype(str).str.strip()
    df["fp_code"] = df["fp_code"].astype(str).str.strip()
    df["count"] = df["count"].astype(int)

    # Pivot: rows = series, cols = FP codes
    pivot = df.pivot(index="series", columns="fp_code", values="count").fillna(0)
    for col in FP_ORDER:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[FP_ORDER]

    # Sort rows by doc year
    pivot = pivot.loc[sorted(pivot.index, key=_sort_series)]
    pct = pivot / queries_per_series * 100
    return pivot, pct


def plot_panel(ax, counts: pd.DataFrame, pcts: pd.DataFrame, title: str) -> None:
    n_rows, n_cols = counts.shape
    data = counts.values.astype(float)
    vmax = max(1.0, data.max())

    im = ax.imshow(data, cmap="Reds", aspect="auto", vmin=0, vmax=vmax)

    # Axes labels — FP code + definition on x-axis
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(
        [f"{c}\n{FP_LABELS.get(c, c)}" for c in counts.columns],
        fontsize=8, ha="center",
    )
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(counts.index.tolist(), fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    # Grid
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="#D1D5DB", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Cell annotations
    threshold = vmax * 0.5
    for i in range(n_rows):
        for j in range(n_cols):
            cnt = int(counts.iat[i, j])
            pct = float(pcts.iat[i, j])
            text = f"{cnt}\n{pct:.1f}%"
            color = "white" if data[i, j] >= threshold else "#1F2937"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=8, color=color, fontweight="semibold",
                    linespacing=1.4)

    # Colorbar legend
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Count", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    return im


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Side-by-side FP1-FP7 normalised heatmap")
    p.add_argument("--left-csv", required=True)
    p.add_argument("--right-csv", required=True)
    p.add_argument("--left-title", default="Retrieval only")
    p.add_argument("--right-title", default="LLM-assisted")
    p.add_argument("--queries-per-series", type=int, default=50)
    p.add_argument("--output", required=True)
    p.add_argument("--suptitle", default="FP1-FP7 Failure Heatmap (count and % of {n} queries)")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    left_counts, left_pcts   = load_counts(Path(args.left_csv),  args.queries_per_series)
    right_counts, right_pcts = load_counts(Path(args.right_csv), args.queries_per_series)

    suptitle = args.suptitle.replace("{n}", str(args.queries_per_series))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")

    plot_panel(axes[0], left_counts,  left_pcts,  args.left_title)
    plot_panel(axes[1], right_counts, right_pcts, args.right_title)

    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
