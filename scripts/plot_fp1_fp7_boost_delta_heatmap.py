"""plot_fp1_fp7_boost_delta_heatmap.py

Generates Figure A.3: change in FP1-FP7 failure counts after subsection
boosting relative to the baseline (boost OFF) pipeline.

Reads from the April 7 boost comparison run.

Usage:
    python scripts/plot_fp1_fp7_boost_delta_heatmap.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

REPO_ROOT  = Path(__file__).resolve().parents[1]
DATA_CSV   = REPO_ROOT / "results/live_fp1_fp7_compare_subsection_on_vs_off_2026-04-07/fp1_fp7_counts_delta.csv"
OUT_PATH   = REPO_ROOT / "results/live_fp1_fp7_compare_subsection_on_vs_off_2026-04-07/current_pipeline_fp1_fp7_delta_heatmap.png"

FP_ORDER = ["FP1", "FP2", "FP3", "FP4", "FP5", "FP6", "FP7"]

FP_LABELS = {
    "FP1": "FP1\nMissing content",
    "FP2": "FP2\nMissed top-ranked",
    "FP3": "FP3\nNot in context",
    "FP4": "FP4\nNot extracted",
    "FP5": "FP5\nIncorrect format",
    "FP6": "FP6\nIncorrect specificity",
    "FP7": "FP7\nIncomplete answer",
}

DOC_ORDER = ["2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025"]


def load_deltas() -> dict[str, dict[str, int]]:
    rows = list(csv.DictReader(open(DATA_CSV)))
    data: dict[str, dict[str, int]] = {doc: {} for doc in DOC_ORDER}
    for r in rows:
        doc = r["document"]
        if doc not in DOC_ORDER:
            continue
        data[doc][r["failure_type"]] = int(r["delta"])
    return data


def main() -> None:
    deltas = load_deltas()

    # Build matrix: rows=docs, cols=FP codes
    matrix = np.array([
        [deltas[doc].get(fp, 0) for fp in FP_ORDER]
        for doc in DOC_ORDER
    ], dtype=float)

    abs_max = max(1.0, np.abs(matrix).max())
    norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    cmap = plt.cm.RdBu_r  # red=increase, blue=decrease

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("white")

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    # Axes
    ax.set_xticks(np.arange(len(FP_ORDER)))
    ax.set_xticklabels([FP_LABELS[fp] for fp in FP_ORDER], fontsize=8.5, ha="center")
    ax.set_yticks(np.arange(len(DOC_ORDER)))
    ax.set_yticklabels(DOC_ORDER, fontsize=9)

    # Grid
    ax.set_xticks(np.arange(-0.5, len(FP_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(DOC_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Cell annotations
    for i in range(len(DOC_ORDER)):
        for j in range(len(FP_ORDER)):
            val = int(matrix[i, j])
            label = f"+{val}" if val > 0 else str(val)
            norm_val = norm(matrix[i, j])
            text_color = "white" if abs(norm_val - 0.5) > 0.25 else "#1F2937"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=10, fontweight="bold", color=text_color)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Delta count (after − before)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title(
        "Change in FP1–FP7 failures after subsection boosting\n"
        r"(post-fix 224/56 pipeline, \texttt{enable\_subsection\_boost}: OFF → ON)",
        fontsize=11, fontweight="bold", pad=10,
    )

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
