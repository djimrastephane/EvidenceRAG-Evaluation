from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "results" / "retrieval_pairwise_win_loss_tie_2026-03-26" / "all_win_loss_tie_summary.csv"
OUT_DIR = ROOT / "results" / "retrieval_pairwise_win_loss_tie_2026-03-26"
OUT_PNG = OUT_DIR / "per_query_comparison_publication.png"
OUT_PDF = OUT_DIR / "per_query_comparison_publication.pdf"

COMPARISON_ORDER = ["Hybrid vs Dense", "Hybrid vs BM25"]
PANEL_TITLES = {
    "Hybrid vs Dense": "Hybrid vs Dense (MiniLM)",
    "Hybrid vs BM25": "Hybrid vs BM25",
}
METRIC_ORDER = ["Hit@1", "Hit@3", "MRR@10"]

# Color-blind safe palette; ties are lighter so they visually recede.
COLORS = {
    "wins": "#0072B2",
    "losses": "#D55E00",
    "ties": "#D0D0D0",
}
HATCHES = {
    "wins": "//",
    "losses": "/",
    "ties": ".",
}


@dataclass(frozen=True)
class Row:
    comparison: str
    metric: str
    wins: int
    losses: int
    ties: int
    queries_compared: int


def load_rows(path: Path) -> list[Row]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            Row(
                comparison=row["comparison"],
                metric=row["metric"],
                wins=int(row["wins"]),
                losses=int(row["losses"]),
                ties=int(row["ties"]),
                queries_compared=int(row["queries_compared"]),
            )
            for row in reader
        ]


def index_rows(rows: list[Row]) -> dict[str, dict[str, Row]]:
    indexed: dict[str, dict[str, Row]] = {}
    for row in rows:
        indexed.setdefault(row.comparison, {})[row.metric] = row
    return indexed


def configure_style() -> None:
    # Keep typography consistent and readable at thesis print scale.
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 11.5,
            "hatch.linewidth": 0.65,
        }
    )


def add_segment_labels(ax: plt.Axes, row: Row, y: float) -> None:
    label_color = "#1F1F1F"
    ax.text(-row.losses - 1.6, y, f"{row.losses}", ha="right", va="center", color=label_color, fontsize=10.5)
    ax.text(row.ties / 2.0, y, f"{row.ties}", ha="center", va="center", color=label_color, fontsize=10.5)
    ax.text(row.ties + row.wins + 1.6, y, f"{row.wins}", ha="left", va="center", color=label_color, fontsize=10.5)


def draw_panel(ax: plt.Axes, title: str, rows_by_metric: dict[str, Row], x_limits: tuple[int, int]) -> None:
    y_positions = list(range(len(METRIC_ORDER)))
    bar_height = 0.48

    for y, metric in zip(y_positions, METRIC_ORDER):
        row = rows_by_metric[metric]

        ax.barh(
            y,
            -row.losses,
            height=bar_height,
            color=COLORS["losses"],
            hatch=HATCHES["losses"],
            edgecolor="#555555",
            linewidth=0.55,
            zorder=3,
        )
        ax.barh(
            y,
            row.ties,
            height=bar_height,
            color=COLORS["ties"],
            hatch=HATCHES["ties"],
            edgecolor="#777777",
            linewidth=0.45,
            alpha=0.6,
            zorder=2,
        )
        ax.barh(
            y,
            row.wins,
            left=row.ties,
            height=bar_height,
            color=COLORS["wins"],
            hatch=HATCHES["wins"],
            edgecolor="#555555",
            linewidth=0.55,
            zorder=3,
        )
        add_segment_labels(ax, row, y)

    ax.axvline(0, color="#333333", linewidth=0.7, zorder=4)
    ax.set_title(title, pad=10, fontweight="semibold")
    ax.set_yticks(y_positions, METRIC_ORDER)
    ax.set_xlabel("Number of queries")
    ax.set_xlim(*x_limits)
    ax.invert_yaxis()

    # No gridlines for a cleaner journal-style presentation.
    ax.grid(False)
    ax.set_axisbelow(True)

    # Remove heavy frame styling for a cleaner publication look.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#888888")
    ax.tick_params(axis="y", length=0, colors="#222222")
    ax.tick_params(axis="x", colors="#333333")


def build_figure(rows: list[Row]) -> plt.Figure:
    indexed = index_rows(rows)
    x_limits = (-100, 250)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 5.0),
        sharey=True,
        gridspec_kw={"wspace": 0.10},
    )

    for ax, comparison in zip(axes, COMPARISON_ORDER):
        draw_panel(ax, PANEL_TITLES[comparison], indexed[comparison], x_limits)

    fig.suptitle(
        "Per-query comparison of hybrid and baseline retrieval methods (n = 250)",
        y=0.97,
        fontsize=13.5,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.03,
        "Left = losses, centre = ties, right = wins (per query).",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#333333",
    )

    # Manual spacing leaves room for the footer and avoids label overlap.
    fig.subplots_adjust(left=0.082, right=0.985, top=0.86, bottom=0.20, wspace=0.10)
    return fig


def main() -> None:
    configure_style()
    rows = load_rows(INPUT_CSV)
    fig = build_figure(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=320, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
