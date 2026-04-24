from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

try:
    from scripts._matplotlib_env import configure_matplotlib_env
except ImportError:
    from _matplotlib_env import configure_matplotlib_env

configure_matplotlib_env()

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


CELL_COLORS = {
    "absent": "#F7F7F7",
    "ubiquitous": "#009E73",
    "present": "#56B4E9",
    "deprecated": "#CC79A7",
    "debut": "#E69F00",
}
EDGE_COLOR = "#C7CCD4"
CELL_HATCHES = {
    "absent": "",
    "ubiquitous": "",
    "present": "///",
    "deprecated": "xx",
    "debut": "",
}
TEXT_COLOR = "#1F2937"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot acronym lifecycle heatmap from a 1/0 matrix CSV.")
    parser.add_argument(
        "--matrix-csv",
        type=Path,
        default=Path("results/query_inventory/acronym_document_presence_matrix_thesis12.csv"),
        help="Matrix CSV with acronym rows and document columns containing 1/0 presence.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/query_inventory/acronym_lifecycle_heatmap_from_matrix.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Optional title.",
    )
    return parser.parse_args()


def year_label(doc_name: str) -> str:
    years = re.findall(r"\d{4}", doc_name)
    if len(years) >= 2:
        return f"{years[0][2:]}-{years[1][2:]}"
    return doc_name


def milestone_label(doc_name: str) -> str | None:
    mapping = {
        "Grampian-2004-2005": "2004",
        "Grampian-2010-2011": "2010",
        "Grampian-2015-2016": "2016",
        "Grampian-2019-2020": "2019",
        "Grampian-2024-2025": "2025",
    }
    return mapping.get(doc_name)


def load_matrix(matrix_csv: Path) -> tuple[list[str], list[str], dict[str, list[int]]]:
    with matrix_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        docs = [f for f in fieldnames if f not in ("acronym", "doc_count")]
        acronyms: list[str] = []
        matrix: dict[str, list[int]] = {}
        for row in reader:
            acronym = row["acronym"]
            acronyms.append(acronym)
            matrix[acronym] = [int(row[doc]) for doc in docs]
    return acronyms, docs, matrix


def classify_row(presence: list[int]) -> str:
    if all(presence):
        return "ubiquitous"
    return "present"


def draw_heatmap(acronyms: list[str], docs: list[str], matrix: dict[str, list[int]], output_path: Path, title: str) -> None:
    n_rows = len(acronyms)
    n_cols = len(docs)
    highlight_rows = {"PGC", "A&E", "COVID-19"}
    fig_w = 13.2
    fig_h = max(8.5, 0.42 * n_rows + 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#FBFCFE")
    ax.set_facecolor("#FFFFFF")

    label_x = -0.55
    cell_size = 0.9
    gap = 0.08

    for row_idx, acronym in enumerate(acronyms):
        y = n_rows - 1 - row_idx
        presence = matrix[acronym]
        row_kind = classify_row(presence)
        first_seen_idx = next((i for i, flag in enumerate(presence) if flag == 1), None)
        last_seen_idx = next((i for i in range(len(presence) - 1, -1, -1) if presence[i] == 1), None)

        ax.text(
            label_x,
            y + 0.45,
            acronym,
            ha="right",
            va="center",
            fontsize=10,
            weight="bold" if acronym in highlight_rows else "semibold",
            color=TEXT_COLOR,
        )

        for col_idx in range(n_cols):
            x = col_idx
            is_present = presence[col_idx] == 1
            if first_seen_idx == col_idx:
                kind = "debut"
            elif is_present:
                kind = row_kind
            elif row_kind != "ubiquitous" and last_seen_idx is not None and col_idx > last_seen_idx:
                kind = "deprecated"
            else:
                kind = "absent"
            rect = Rectangle(
                (x + gap / 2, y + gap / 2),
                cell_size - gap,
                cell_size - gap,
                facecolor=CELL_COLORS[kind],
                edgecolor=EDGE_COLOR,
                linewidth=0.6,
                hatch=CELL_HATCHES[kind],
            )
            ax.add_patch(rect)
            if first_seen_idx == col_idx:
                ax.scatter(
                    x + 0.45,
                    y + 0.88,
                    marker="^",
                    s=38,
                    color="#FFFFFF",
                    edgecolors="#1F1F1F",
                    linewidths=0.8,
                    zorder=4,
                )

    for col_idx, doc in enumerate(docs):
        label = milestone_label(doc)
        if not label:
            continue
        ax.text(
            col_idx + 0.45,
            n_rows + 0.08,
            label,
            rotation=0,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#55657B",
        )

    split_idx = next((i for i, doc in enumerate(docs) if "2019-2020" in doc), None)
    if split_idx is not None:
        split_x = split_idx
        ax.axvline(
            split_x,
            color="#51627A",
            linestyle=(0, (1.4, 2.0)),
            linewidth=2.0,
            ymin=0.045,
            ymax=0.94,
            zorder=5,
        )
        ax.text(split_x - 0.38, n_rows + 0.42, "pre-2019", ha="right", va="bottom", fontsize=8, weight="bold", color="#6B7C93")
        ax.text(split_x + 0.38, n_rows + 0.42, "2019+", ha="left", va="bottom", fontsize=8, weight="bold", color="#6B7C93")

    if title:
        ax.text(-1.5, n_rows + 1.0, title, ha="left", va="bottom", fontsize=12, weight="bold", color=TEXT_COLOR)

    legend_items = [
        Patch(facecolor=CELL_COLORS["debut"], edgecolor=EDGE_COLOR, label="Debut year"),
        Patch(facecolor=CELL_COLORS["ubiquitous"], edgecolor=EDGE_COLOR, label=f"Always used ({n_cols}/{n_cols})"),
        Patch(facecolor=CELL_COLORS["present"], edgecolor=EDGE_COLOR, hatch=CELL_HATCHES["present"], label="Present"),
        Patch(facecolor=CELL_COLORS["deprecated"], edgecolor=EDGE_COLOR, hatch=CELL_HATCHES["deprecated"], label="No longer used"),
        Patch(facecolor=CELL_COLORS["absent"], edgecolor=EDGE_COLOR, label="Absent"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=5,
        frameon=False,
        fontsize=9,
        handlelength=1.2,
        handleheight=1.2,
    )

    ax.set_xlim(label_x - 1.6, n_cols + 0.2)
    ax.set_ylim(-0.2, n_rows + 1.25)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    acronyms, docs, matrix = load_matrix(args.matrix_csv)
    draw_heatmap(acronyms, docs, matrix, args.output, args.title)
    print(args.output)


if __name__ == "__main__":
    main()
