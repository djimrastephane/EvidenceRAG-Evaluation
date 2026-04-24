from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_ROOT = Path("results/subsection_boost_on_off_2026-04-07")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render subsection boost analysis charts.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def render_hit_delta_ci(root: Path) -> Path:
    df = pd.read_csv(root / "subsection_boost_query_bootstrap_summary.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = df["k"].astype(str).tolist()
    y = df["page_hit_delta_mean"].to_numpy()
    yerr = [
        y - df["page_hit_delta_ci_low"].to_numpy(),
        df["page_hit_delta_ci_high"].to_numpy() - y,
    ]
    ax.errorbar(x, y, yerr=yerr, fmt="o-", color="#2F855A", ecolor="#1F5E3D", capsize=4, linewidth=2.2)
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_title("Subsection boost effect on page hit rate")
    ax.set_xlabel("k")
    ax.set_ylabel("Delta (ON - OFF)")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    out = root / "subsection_boost_hit_delta_ci.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def render_wins_losses(root: Path) -> Path:
    df = pd.read_csv(root / "subsection_boost_query_bootstrap_summary.csv")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = range(len(df))
    wins = df["page_hit_wins"].to_numpy()
    losses = df["page_hit_losses"].to_numpy()
    unchanged = df["page_hit_unchanged"].to_numpy()
    ax.bar(x, wins, label="Wins", color="#2F855A")
    ax.bar(x, losses, bottom=wins, label="Losses", color="#C05621")
    ax.bar(x, unchanged, bottom=wins + losses, label="Unchanged", color="#B7B7B7")
    ax.set_xticks(list(x), [str(k) for k in df["k"].tolist()])
    ax.set_xlabel("k")
    ax.set_ylabel("Query count")
    ax.set_title("Per-query outcomes with subsection boost")
    ax.legend(frameon=False, ncols=3, loc="upper center")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    out = root / "subsection_boost_wins_losses.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def render_fp2_rate(root: Path) -> Path:
    df = pd.read_csv(root / "subsection_boost_fp2_summary.csv")
    off = float(df["fp2_off_rate"].iloc[0]) * 100.0
    on = float(df["fp2_on_rate"].iloc[0]) * 100.0
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    bars = ax.bar(["Boost OFF", "Boost ON"], [off, on], color=["#B7B7B7", "#2F855A"], edgecolor="#444444")
    ax.set_ylabel("FP2 rate (%)")
    ax.set_title("FP2 rate before and after subsection boost")
    ax.set_ylim(0, max(off, on) * 1.25)
    for bar, val in zip(bars, [off, on]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"{val:.1f}%", ha="center", va="bottom")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    out = root / "subsection_boost_fp2_rate.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    outputs = [
        render_hit_delta_ci(args.root),
        render_wins_losses(args.root),
        render_fp2_rate(args.root),
    ]
    for out in outputs:
        print(out)


if __name__ == "__main__":
    main()
