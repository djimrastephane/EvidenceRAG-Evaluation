from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
BASE_OUT_DIR = ROOT / "results" / "retrieval_pairwise_win_loss_tie_2026-03-26"

COMPARISON_ORDER = ["Hybrid vs Dense", "Hybrid vs BM25"]
PANEL_TITLES = {
    "Hybrid vs Dense": "Hybrid vs Dense (MiniLM)",
    "Hybrid vs BM25": "Hybrid vs BM25",
}
METRIC_ORDER = ["Hit@1", "Hit@3", "MRR@10"]
METRICS = {
    "Hit@1": (1, "page_recall_at_k"),
    "Hit@3": (3, "page_recall_at_k"),
    "MRR@10": (10, "page_mrr_at_k"),
}

# Color-blind safe palette with lighter ties so they recede visually.
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
class SourceSpec:
    label: str
    root: Path
    summary_name: str


@dataclass(frozen=True)
class Row:
    comparison: str
    metric: str
    wins: int
    losses: int
    ties: int
    queries_compared: int


SOURCES = {
    "dense": SourceSpec(
        label="Dense-only (all-MiniLM-L6-v2)",
        root=ROOT / "results" / "dense_encoder_ablation" / "smoke_l6_only" / "artifacts" / "all-MiniLM-L6-v2" / "source_docs",
        summary_name="retrieval_summary.csv",
    ),
    "hybrid": SourceSpec(
        label="Hybrid (MiniLM + BM25 default tokenizer)",
        root=ROOT / "results" / "bm25_tokenizer_sensitivity" / "thesis_bm25_tokenizer_sensitivity_20260325" / "hybrid_default",
        summary_name="retrieval_summary_hybrid.csv",
    ),
    "bm25": SourceSpec(
        label="BM25-only (default tokenizer)",
        root=ROOT / "results" / "bm25_tokenizer_sensitivity" / "thesis_bm25_tokenizer_sensitivity_20260325" / "bm25_default",
        summary_name="retrieval_summary_bm25.csv",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pairwise win/loss/tie chart for a single document.")
    parser.add_argument("--doc-id", required=True, help="Document id such as Grampian-2024-2025")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to results/retrieval_pairwise_win_loss_tie_2026-03-26/<doc-id>",
    )
    return parser.parse_args()


def load_rows(spec: SourceSpec, doc_id: str) -> dict[str, dict[int, dict[str, str]]]:
    path = spec.root / doc_id / spec.summary_name
    rows: dict[str, dict[int, dict[str, str]]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.setdefault(row["query_id"], {})[int(row["k"])] = row
    return rows


def compare(
    comparison: str,
    left: dict[str, dict[int, dict[str, str]]],
    right: dict[str, dict[int, dict[str, str]]],
) -> list[Row]:
    query_ids = sorted(set(left).intersection(right))
    out: list[Row] = []
    for metric_name, (k, field) in METRICS.items():
        wins = losses = ties = 0
        for qid in query_ids:
            left_val = float(left[qid][k][field])
            right_val = float(right[qid][k][field])
            if abs(left_val - right_val) < 1e-12:
                ties += 1
            elif left_val > right_val:
                wins += 1
            else:
                losses += 1
        out.append(
            Row(
                comparison=comparison,
                metric=metric_name,
                wins=wins,
                losses=losses,
                ties=ties,
                queries_compared=len(query_ids),
            )
        )
    return out


def write_summary_csv(path: Path, rows: list[Row]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["comparison", "metric", "wins", "losses", "ties", "queries_compared"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def index_rows(rows: list[Row]) -> dict[str, dict[str, Row]]:
    indexed: dict[str, dict[str, Row]] = {}
    for row in rows:
        indexed.setdefault(row.comparison, {})[row.metric] = row
    return indexed


def configure_style() -> None:
    # Keep typography readable at thesis print scale.
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
    ax.text(-row.losses - 1.2, y, f"{row.losses}", ha="right", va="center", color=label_color, fontsize=10.5)
    ax.text(row.ties / 2.0, y, f"{row.ties}", ha="center", va="center", color=label_color, fontsize=10.5)
    ax.text(row.ties + row.wins + 1.2, y, f"{row.wins}", ha="left", va="center", color=label_color, fontsize=10.5)


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
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#888888")
    ax.tick_params(axis="y", length=0, colors="#222222")
    ax.tick_params(axis="x", colors="#333333")


def build_figure(rows: list[Row], query_count: int) -> plt.Figure:
    indexed = index_rows(rows)
    x_limits = (-25, max(row.wins + row.ties for row in rows) + 10)

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
        f"Per-query comparison of hybrid and baseline retrieval methods (n = {query_count})",
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
    fig.subplots_adjust(left=0.082, right=0.985, top=0.86, bottom=0.20, wspace=0.10)
    return fig


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (BASE_OUT_DIR / args.doc_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_style()

    loaded = {name: load_rows(spec, args.doc_id) for name, spec in SOURCES.items()}
    rows = [
        *compare("Hybrid vs Dense", loaded["hybrid"], loaded["dense"]),
        *compare("Hybrid vs BM25", loaded["hybrid"], loaded["bm25"]),
    ]

    query_count = rows[0].queries_compared if rows else 0
    write_summary_csv(out_dir / "all_win_loss_tie_summary.csv", rows)

    fig = build_figure(rows, query_count)
    fig.savefig(out_dir / "per_query_comparison_publication.png", dpi=320, bbox_inches="tight")
    fig.savefig(out_dir / "per_query_comparison_publication.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
