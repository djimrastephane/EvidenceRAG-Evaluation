"""Generate Figure 4.4: Page Hit@1 by query difficulty tier and retrieval method.

Data: frozen 224/56 boost-OFF artifacts + current eval_set (2026-04-24 rerun).
Wilson score 95% CIs computed from per-tier counts.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Data (rerun_main_tables_2026-04-24/results.json) ────────────────────────
DATA: dict[str, dict[str, tuple[float, int]]] = {
    "Dense (MiniLM)": {"LEX": (0.800, 125), "MOD": (0.760, 75), "STR": (0.700, 50)},
    "BM25":           {"LEX": (0.768, 125), "MOD": (0.733, 75), "STR": (0.500, 50)},
    "Hybrid (base)":  {"LEX": (0.784, 125), "MOD": (0.760, 75), "STR": (0.600, 50)},
}

# Wong colorblind-safe palette
COLORS = {
    "Dense (MiniLM)": "#0072B2",
    "BM25":           "#D55E00",
    "Hybrid (base)":  "#009E73",
}

MARKERS = {
    "Dense (MiniLM)": "o",
    "BM25":           "s",
    "Hybrid (base)":  "^",
}

END_LABELS = {
    "Dense (MiniLM)": "0.70 (-13%)",
    "Hybrid (base)":  "0.60 (-23%)",
    "BM25":           "0.50 (-35%)",
}

END_OFFSETS = {
    "Dense (MiniLM)": 0.015,
    "Hybrid (base)":  0.008,
    "BM25":          -0.010,
}

METHODS   = ["Dense (MiniLM)", "Hybrid (base)", "BM25"]
TIERS     = ["LEX", "MOD", "STR"]
X_LABELS  = ["LEX\n(n=125)", "MOD\n(n=75)", "STR\n(n=50)"]


def wilson_half_width(p: float, n: int, z: float = 1.96) -> float:
    denom  = 1.0 + z**2 / n
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return margin


# Figure
fig, ax = plt.subplots(figsize=(7.5, 3.45))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

x = np.arange(len(TIERS), dtype=float)

for method in METHODS:
    vals = np.array([DATA[method][tier][0] for tier in TIERS])
    ns = [DATA[method][tier][1] for tier in TIERS]
    errs = np.array([wilson_half_width(float(v), n) for v, n in zip(vals, ns)])
    color = COLORS[method]

    ax.plot(
        x, vals,
        color=color,
        marker=MARKERS[method],
        markersize=5.8,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.8,
        linewidth=2.05,
        label=method,
        zorder=3,
    )
    ax.errorbar(
        x, vals,
        yerr=errs,
        fmt="none",
        ecolor=color,
        alpha=0.32,
        elinewidth=1.0,
        capsize=2.8,
        capthick=1.0,
        zorder=2,
    )
    ax.text(
        2.065,
        vals[-1] + END_OFFSETS[method],
        END_LABELS[method],
        ha="left",
        va="center",
        fontsize=9.0,
        color=color,
        fontweight="semibold",
    )

# Axis styling
ax.set_xlim(-0.06, 2.70)
ax.set_ylim(0.40, 0.88)
ax.set_yticks(np.arange(0.4, 0.89, 0.1))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
ax.set_ylabel("Page Hit@1", fontsize=10.2, color="#444444")

ax.set_xticks(x)
ax.set_xticklabels(X_LABELS, fontsize=10.0)

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color("#aaaaaa")

ax.tick_params(axis="both", colors="#444444", length=3)

ax.yaxis.grid(True, linestyle="--", linewidth=0.55, color="#dddddd", zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

ax.legend(
    loc="lower left",
    fontsize=9.0,
    frameon=True,
    framealpha=0.92,
    edgecolor="#cccccc",
    handlelength=1.5,
    borderpad=0.6,
)

ax.set_title(
    "Retrieval performance by tier complexity",
    fontsize=11.0,
    fontweight="bold",
    loc="left",
    pad=10,
)

# Save
plt.tight_layout(pad=0.45)

OUTPUT_DIRS = [
    Path(
    "/Users/djimra/MSc Data Science Jan 2025/Thesis documents/"
    "Thesis/University_of_Aberdeen_thesis_template/figures"
    ),
    Path(
        "/Users/djimra/MSc Data Science Jan 2025/Thesis documents/"
        "EvidenceRAG-Evaluation/figures"
    ),
]

for out_dir in OUTPUT_DIRS:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out_dir / f"difficulty_tier_method_comparison.{ext}"
        fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
        print(f"Saved {path}")

plt.close(fig)
