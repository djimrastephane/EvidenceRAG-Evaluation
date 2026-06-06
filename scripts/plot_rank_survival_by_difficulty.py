"""
Rank survival curves broken down by query difficulty (LEX / MOD / STR).
Style matches Figure 4.1: KM step functions with Wilson CI bands.

Four systems: Dense, BM25, Hybrid-base, Hybrid+boost.
Output: results/thesis_figures/rank_survival_by_difficulty/
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
DATA_PROC = REPO / "data_processed"
BOOST_CSV = (
    REPO
    / "results/thesis_figures/figure_4_2_rank_survival_2026-04-21"
    / "retrieval_rank_survival_compare.csv"
)
OUT_DIR = REPO / "results" / "thesis_figures" / "rank_survival_by_difficulty"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOCS = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]

METHOD_FILES = {
    "Dense":   "retrieval_results.json",
    "BM25":    "retrieval_results_bm25.json",
    "Hybrid":  "retrieval_results_hybrid.json",
}

# ── style — matched to Figure 4.1 ────────────────────────────────────────────
SYSTEM_STYLE: dict[str, dict] = {
    "Dense":        {"color": "#4C72B0", "ls": "--",  "lw": 1.8,
                     "label": "Dense (MiniLM)"},
    "BM25":         {"color": "#55A868", "ls": ":",   "lw": 1.8,
                     "label": "BM25-only"},
    "Hybrid":       {"color": "#64B5CD", "ls": ":",   "lw": 1.8,
                     "label": "Hybrid (base)"},
    "Hybrid+boost": {"color": "#FF7F0E", "ls": "-",   "lw": 2.2,
                     "label": "Hybrid + subsection boost"},
}

# ── helpers ──────────────────────────────────────────────────────────────────

def load_eval_set(doc_id: str) -> dict[str, str]:
    path = DATA_PROC / doc_id / "eval_set.json"
    data = json.loads(path.read_text())
    return {q["query_id"]: q["difficulty"] for q in data["queries"]}


def first_correct_rank(result: dict) -> float:
    expected = set(result.get("expected_pages") or [])
    top10 = result.get("per_k", {}).get("10", {}).get("retrieved_pages_ranked", [])
    for rank, page in enumerate(top10, start=1):
        if page in expected:
            return float(rank)
    return 11.0  # censored


def load_method_records(method_label: str, filename: str) -> list[dict]:
    rows = []
    for doc in DOCS:
        path = DATA_PROC / doc / filename
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        results = data.get("results", data) if isinstance(data, dict) else data
        for r in results:
            rows.append({
                "system": method_label,
                "query_id": r["query_id"],
                "first_correct_rank": first_correct_rank(r),
            })
    return rows


def load_boost_records() -> list[dict]:
    df = pd.read_csv(BOOST_CSV)
    boost = df[df["system"] == "hybrid_boost"].copy()
    boost["first_correct_rank"] = boost["first_correct_rank"].fillna(11.0).clip(upper=11.0)
    return [
        {"system": "Hybrid+boost", "query_id": row["query_id"],
         "first_correct_rank": row["first_correct_rank"]}
        for _, row in boost.iterrows()
    ]


def build_dataframe() -> pd.DataFrame:
    difficulty_map: dict[str, str] = {}
    for doc in DOCS:
        difficulty_map.update(load_eval_set(doc))
    rows: list[dict] = []
    for label, fname in METHOD_FILES.items():
        rows.extend(load_method_records(label, fname))
    rows.extend(load_boost_records())
    df = pd.DataFrame(rows)
    df["difficulty"] = df["query_id"].map(difficulty_map)
    df = df.dropna(subset=["difficulty"])
    return df


# ── KM step function with Wilson CI ─────────────────────────────────────────

def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def km_steps(ranks: pd.Series, max_k: int = 10
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (ks, surv, ci_lo, ci_hi) for a step-function plot.
    Prepends k=0 anchor (surv=1).
    """
    n = len(ranks)
    ks    = [0]
    surv  = [1.0]
    ci_lo = [1.0]
    ci_hi = [1.0]
    for k in range(1, max_k + 1):
        p = (ranks > k).mean()
        lo, hi = wilson_ci(p, n)
        ks.append(k)
        surv.append(p)
        ci_lo.append(lo)
        ci_hi.append(hi)
    return (np.array(ks), np.array(surv),
            np.array(ci_lo), np.array(ci_hi))


# ── plot ─────────────────────────────────────────────────────────────────────

def plot(df: pd.DataFrame) -> None:
    difficulties = ["LEX", "MOD", "STR"]
    n_counts = df[df["system"] == "Dense"].groupby("difficulty")["query_id"].count()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
    fig.subplots_adjust(wspace=0.28)

    for ax, diff in zip(axes, difficulties):
        sub = df[df["difficulty"] == diff]
        n   = n_counts.get(diff, 0)
        y_max = 0.0

        for system, style in SYSTEM_STYLE.items():
            ranks = sub[sub["system"] == system]["first_correct_rank"]
            if ranks.empty:
                continue
            ks, surv, ci_lo, ci_hi = km_steps(ranks)
            y_max = max(y_max, surv[1])   # max at k=1

            ax.step(ks, surv,
                    where="post",
                    color=style["color"],
                    ls=style["ls"],
                    lw=style["lw"],
                    label=style["label"])
            ax.fill_between(ks, ci_lo, ci_hi,
                            step="post",
                            color=style["color"],
                            alpha=0.12)

        ax.set_title(
            f"{diff}  —  "
            f"{'lexical' if diff=='LEX' else 'moderate' if diff=='MOD' else 'structural'}"
            f" queries\n(n={n})",
            fontsize=10, fontweight="bold", linespacing=1.5,
        )
        ax.set_xlabel("Rank", fontsize=10)
        ax.set_xlim(1, 10)
        ax.set_ylim(0, min(1.0, y_max * 1.35))
        ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda y, _: f"{y:.2f}"))
        ax.grid(axis="y", ls="--", lw=0.6, alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Survival probability", fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(
        "Rank-based survival comparison by query difficulty tier",
        fontsize=12, fontweight="bold", y=1.03,
    )

    for fmt in ("pdf", "png"):
        out = OUT_DIR / f"rank_survival_by_difficulty.{fmt}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    df = build_dataframe()
    print(f"Loaded {len(df)} records across {df['system'].nunique()} systems")
    print(df.groupby(["system", "difficulty"])["query_id"].count().unstack())
    plot(df)
