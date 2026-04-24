#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import _matplotlib_env
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot FP1-FP7 delta heatmap for before/after failure counts.")
    p.add_argument("--counts-delta-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def _sort_doc_key(value: str) -> tuple[int, str]:
    text = str(value).strip()
    if text == "ALL":
        return (9999, text)
    digits = "".join(ch for ch in text if ch.isdigit())
    return (int(digits[:4]) if len(digits) >= 4 else 0, text)


def plot_delta_heatmap(counts_delta_csv: Path, output: Path, dpi: int = 300) -> None:
    df = pd.read_csv(counts_delta_csv)
    df = df[df["document"].astype(str) != "ALL"].copy()

    fp_order = ["FP1", "FP2", "FP3", "FP4", "FP5", "FP6", "FP7"]
    row_order = sorted(df["document"].astype(str).unique().tolist(), key=_sort_doc_key)
    heat = (
        df.pivot(index="document", columns="failure_type", values="delta")
        .reindex(index=row_order, columns=fp_order)
        .fillna(0.0)
    )

    vmax = max(1.0, float(np.abs(heat.values).max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    fig.patch.set_facecolor("white")
    im = ax.imshow(heat.values, cmap="RdBu_r", aspect="auto", norm=norm)

    ax.set_xticks(np.arange(len(fp_order)))
    ax.set_xticklabels(fp_order, fontsize=10, fontweight="semibold")
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_order, fontsize=10)
    ax.set_title(
        "Change in FP1-FP7 failures after subsection boosting",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    ax.set_xticks(np.arange(-0.5, len(fp_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_order), 1), minor=True)
    ax.grid(which="minor", color="#E5E7EB", linestyle="-", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = int(heat.iat[i, j])
            text = "0" if value == 0 else f"{value:+d}"
            text_color = "white" if abs(value) >= max(3, int(round(vmax * 0.45))) else "#1F2937"
            ax.text(j, i, text, ha="center", va="center", fontsize=10, color=text_color, fontweight="semibold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("Delta count (after - before)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


def main() -> None:
    args = parse_args()
    plot_delta_heatmap(
        counts_delta_csv=Path(args.counts_delta_csv),
        output=Path(args.output),
        dpi=int(args.dpi),
    )


if __name__ == "__main__":
    main()
