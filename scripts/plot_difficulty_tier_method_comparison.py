"""Generate Figure 4.3: Page Hit@1 by query difficulty tier and retrieval method.

Data: frozen 224/56 boost-OFF artifacts + current eval_set (2026-04-24 rerun).
Wilson score 95% CIs computed from per-tier counts.
"""

from __future__ import annotations

import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Data (from rerun_main_tables_2026-04-24/results.json) ───────────────────
DATA: dict[str, dict[str, tuple[float, int]]] = {
    "Dense":  {"LEX": (0.800, 125), "MOD": (0.760, 75), "STR": (0.700, 50)},
    "BM25":   {"LEX": (0.768, 125), "MOD": (0.733, 75), "STR": (0.500, 50)},
    "Hybrid": {"LEX": (0.784, 125), "MOD": (0.760, 75), "STR": (0.600, 50)},
}

COLORS = {"Dense": "#0072B2", "BM25": "#D55E00", "Hybrid": "#009E73"}
TIERS   = ["LEX", "MOD", "STR"]
METHODS = ["Dense", "BM25", "Hybrid"]


def wilson_half_width(p: float, n: int, z: float = 1.96) -> float:
    denom  = 1.0 + z**2 / n
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return margin


# ── Figure ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 4.4))

n_methods  = len(METHODS)
bar_width  = 0.72 / n_methods
x          = np.arange(len(TIERS))

for i, method in enumerate(METHODS):
    vals    = [DATA[method][t][0] for t in TIERS]
    ns      = [DATA[method][t][1] for t in TIERS]
    errs    = [wilson_half_width(v, n) for v, n in zip(vals, ns)]
    offsets = x + (i - (n_methods - 1) / 2.0) * bar_width

    ax.bar(offsets, vals, bar_width,
           color=COLORS[method], label=method, zorder=3, linewidth=0)

    ax.errorbar(offsets, vals, yerr=errs,
                fmt="none", color="#333333",
                capsize=3, capthick=1.1, elinewidth=1.1, zorder=4)

    for xi, (v, err) in zip(offsets, zip(vals, errs)):
        ax.text(xi, v + err + 0.012, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="semibold")

# ── Axes style (Tufte) ───────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(TIERS, fontsize=11)
ax.set_ylabel("Hit@1", fontsize=11)
ax.set_ylim(0.35, 1.05)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color("#aaaaaa")

ax.tick_params(axis="both", colors="#444444", length=3)
ax.yaxis.label.set_color("#444444")

ax.yaxis.grid(True, linestyle="--", linewidth=0.55, color="#dddddd", zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

for xv in (0.5, 1.5):
    ax.axvline(xv, color="#cccccc", linewidth=0.8, linestyle=":", zorder=1)

# ── Title & legend ────────────────────────────────────────────────────────────
ax.set_title("Retrieval Performance by Query Complexity",
             fontsize=13, fontweight="bold", loc="left", pad=14)
ax.legend(fontsize=10, frameon=False, loc="upper right")

# ── Footer ───────────────────────────────────────────────────────────────────
fig.text(
    0.12, 0.005,
    r"LEX $\to$ STR relative drop:  "
    r"$\bf{Dense}$ $-$13%    $\bf{BM25}$ $-$36%    $\bf{Hybrid}$ $-$24%",
    ha="left", va="bottom",
    fontsize=8.5, style="italic", color="#555555",
)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0.04, 1, 1])

OUT = pathlib.Path(
    "/Users/djimra/MSc Data Science Jan 2025/Thesis documents/"
    "Thesis/University_of_Aberdeen_thesis_template/figures"
)
for ext in ("pdf", "png"):
    dest = OUT / f"difficulty_tier_method_comparison.{ext}"
    fig.savefig(dest, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved {dest}")

plt.close(fig)
