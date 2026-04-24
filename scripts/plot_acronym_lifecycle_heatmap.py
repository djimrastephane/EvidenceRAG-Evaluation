from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

try:
    from scripts._matplotlib_env import configure_matplotlib_env
except ImportError:
    from _matplotlib_env import configure_matplotlib_env

configure_matplotlib_env()

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D


DEFAULT_ACRONYMS = [
    "NHS",
    "PAO",
    "RRL",
    "CETV",
    "RICS",
    "CNORIS",
    "SPPA",
    "REPORT",
    "OF",
    "SPFM",
    "PGC",
    "A&E",
    "SGHSCD",
    "ESM",
    "ARI",
    "IJB",
    "CRL",
    "NEST",
    "COVID-19",
    "G-OPES",
]

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
MUTED_TEXT = "#6B7280"
GRID_BG = "#FFFFFF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a lighter acronym lifecycle heatmap.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data_processed"),
        help="Root directory containing Grampian-* processed document folders.",
    )
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=Path("results/query_inventory/acronym_glossary_candidates_grampian_full_2004_2025.csv"),
        help="CSV from the acronym miner containing high_value_score and first/last seen metadata.",
    )
    parser.add_argument(
        "--acronyms",
        type=str,
        default=",".join(DEFAULT_ACRONYMS),
        help="Comma-separated acronym order to plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/query_inventory/acronym_lifecycle_heatmap_light.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--hide-score",
        action="store_true",
        help="Hide the score column for a cleaner thesis figure.",
    )
    parser.add_argument(
        "--sort",
        choices=["input", "first_seen", "alpha"],
        default="input",
        help="Sort acronyms by input order, first appearance then alphabetically, or pure alphabetical order.",
    )
    return parser.parse_args()


def sorted_doc_dirs(data_root: Path) -> list[Path]:
    docs = [p for p in data_root.glob("Grampian-*") if p.is_dir()]
    docs.sort(key=lambda p: tuple(int(part) for part in re.findall(r"\d{4}", p.name)[:2]))
    return docs


def load_document_text(doc_dir: Path) -> str:
    chunks_path = doc_dir / "sections.csv"
    if not chunks_path.exists():
        return ""
    parts: list[str] = []
    with chunks_path.open(encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                parts.append(row[-2])
    return "\n".join(parts)


def acronym_pattern(acronym: str) -> re.Pattern[str]:
    escaped = re.escape(acronym)
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)


def load_presence_matrix(doc_dirs: list[Path], acronyms: list[str]) -> dict[str, list[bool]]:
    patterns = {acronym: acronym_pattern(acronym) for acronym in acronyms}
    matrix = {acronym: [] for acronym in acronyms}
    for doc_dir in doc_dirs:
        text = load_document_text(doc_dir)
        for acronym in acronyms:
            matrix[acronym].append(bool(patterns[acronym].search(text)))
    return matrix


def load_candidate_rows(csv_path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["acronym"]] = row
    return rows


def first_seen_sort_key(acronym: str, row: dict[str, str] | None) -> tuple[int, str]:
    if not row or not row.get("first_seen_doc"):
        return (9999, acronym.lower())
    match = re.search(r"(\d{4})-(\d{4})", row["first_seen_doc"])
    if not match:
        return (9999, acronym.lower())
    return (int(match.group(1)), acronym.lower())


def sort_acronyms(acronyms: list[str], candidate_rows: dict[str, dict[str, str]], mode: str) -> list[str]:
    if mode == "alpha":
        return sorted(acronyms, key=lambda x: x.lower())
    if mode == "first_seen":
        return sorted(acronyms, key=lambda x: first_seen_sort_key(x, candidate_rows.get(x)))
    return acronyms


def parse_score(value: str | None) -> int:
    if not value:
        return 0
    return int(round(float(value)))


def classify_row(presence: list[bool]) -> str:
    if all(presence):
        return "ubiquitous"
    return "present"


def year_label(doc_name: str) -> str:
    years = re.findall(r"\d{4}", doc_name)
    if len(years) >= 2:
        return f"{years[0][2:]}-{years[1][2:]}"
    return doc_name


def draw_heatmap(
    acronyms: list[str],
    doc_dirs: list[Path],
    presence_matrix: dict[str, list[bool]],
    candidate_rows: dict[str, dict[str, str]],
    output_path: Path,
    hide_score: bool,
) -> None:
    n_rows = len(acronyms)
    n_cols = len(doc_dirs)

    fig_w = 14
    fig_h = max(8.5, 0.42 * n_rows + 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#FBFCFE")
    ax.set_facecolor(GRID_BG)

    score_x = -3.6
    label_x = -0.55
    cell_size = 0.9
    gap = 0.08

    for row_idx, acronym in enumerate(acronyms):
        y = n_rows - 1 - row_idx
        presence = presence_matrix[acronym]
        row_kind = classify_row(presence)
        first_seen_idx = next((i for i, flag in enumerate(presence) if flag), None)
        last_seen_idx = next((i for i in range(len(presence) - 1, -1, -1) if presence[i]), None)

        score = parse_score(candidate_rows.get(acronym, {}).get("high_value_score"))
        if not hide_score:
            ax.text(score_x, y + 0.45, str(score), ha="right", va="center", fontsize=9, color=MUTED_TEXT)
        ax.text(label_x, y + 0.45, acronym, ha="right", va="center", fontsize=10, weight="bold", color=TEXT_COLOR)

        for col_idx in range(n_cols):
            x = col_idx
            if first_seen_idx == col_idx:
                kind = "debut"
            elif presence[col_idx]:
                kind = row_kind
            elif row_kind != "ubiquitous" and last_seen_idx is not None and col_idx > last_seen_idx:
                kind = "deprecated"
            else:
                kind = "absent"
            color = CELL_COLORS[kind]

            rect = Rectangle(
                (x + gap / 2, y + gap / 2),
                cell_size - gap,
                cell_size - gap,
                facecolor=color,
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

    doc_labels = [year_label(doc_dir.name) for doc_dir in doc_dirs]
    for col_idx, label in enumerate(doc_labels):
        ax.text(
            col_idx + 0.45,
            n_rows + 0.42,
            label,
            rotation=55,
            ha="left",
            va="bottom",
            fontsize=8,
            color="#55657B",
        )

    if not hide_score:
        ax.text(score_x, n_rows + 0.55, "score", ha="right", va="bottom", fontsize=9, color=MUTED_TEXT)

    split_idx = next((i for i, doc_dir in enumerate(doc_dirs) if "2019-2020" in doc_dir.name), None)
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
        ax.text(
            split_x - 0.18,
            n_rows + 0.2,
            "pre-2019",
            ha="right",
            va="bottom",
            fontsize=9,
            weight="bold",
            color="#6B7C93",
        )
        ax.text(
            split_x + 0.18,
            n_rows + 0.2,
            "2019+",
            ha="left",
            va="bottom",
            fontsize=9,
            weight="bold",
            color="#6B7C93",
        )

    legend_items = [
        Patch(facecolor=CELL_COLORS["debut"], edgecolor=EDGE_COLOR, label="Debut year"),
        Patch(facecolor=CELL_COLORS["ubiquitous"], edgecolor=EDGE_COLOR, label=f"Always used ({len(doc_dirs)}/{len(doc_dirs)})"),
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

    left_limit = (label_x - 2.0) if hide_score else (score_x - 0.2)
    ax.set_xlim(left_limit, n_cols + 0.2)
    ax.set_ylim(-0.2, n_rows + 1.0)
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
    candidate_rows = load_candidate_rows(args.candidate_csv)
    acronyms = [item.strip() for item in args.acronyms.split(",") if item.strip()]
    acronyms = sort_acronyms(acronyms, candidate_rows, args.sort)
    doc_dirs = sorted_doc_dirs(args.data_root)
    presence_matrix = load_presence_matrix(doc_dirs, acronyms)
    draw_heatmap(acronyms, doc_dirs, presence_matrix, candidate_rows, args.output, args.hide_score)
    metadata = {
        "acronyms": acronyms,
        "docs": [doc.name for doc in doc_dirs],
        "output": str(args.output),
    }
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
