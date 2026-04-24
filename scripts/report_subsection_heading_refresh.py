from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from rag_pdf.headings import looks_like_lettered_subsection, looks_like_numbered_heading
from rag_pdf.sections import (
    _detect_page_heading_signals,
    _infer_labels,
    build_sections_from_pages,
    find_section_for_page,
)


DEFAULT_DOCS = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]


@dataclass
class DocStats:
    doc_id: str
    total_chunks: int
    pages: int
    old_unknown_chunks: int
    new_unknown_chunks: int
    unknown_to_known_chunks: int
    changed_known_to_different_known_chunks: int
    unchanged_chunks: int
    old_unknown_pages: int
    new_unknown_pages: int
    unknown_to_known_pages: int

    @property
    def old_unknown_chunk_pct(self) -> float:
        return 100.0 * self.old_unknown_chunks / self.total_chunks if self.total_chunks else 0.0

    @property
    def new_unknown_chunk_pct(self) -> float:
        return 100.0 * self.new_unknown_chunks / self.total_chunks if self.total_chunks else 0.0

    @property
    def old_unknown_page_pct(self) -> float:
        return 100.0 * self.old_unknown_pages / self.pages if self.pages else 0.0

    @property
    def new_unknown_page_pct(self) -> float:
        return 100.0 * self.new_unknown_pages / self.pages if self.pages else 0.0


def build_sections_with_old_subsection_logic(pages_df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct section labels using the pre-patch subsection heuristic.

    The old behavior accepted lettered subsection markers but not numbered headings.
    We emulate that by temporarily wrapping `_detect_page_heading_signals`.
    """

    import rag_pdf.sections as sections_mod

    original = sections_mod._detect_page_heading_signals

    def old_detect(row: pd.Series):
        section_found, subsection_found, top_lines, lines = original(row)
        if subsection_found and looks_like_numbered_heading(str(subsection_found)) and not looks_like_lettered_subsection(
            str(subsection_found)
        ):
            subsection_found = None
        return section_found, subsection_found, top_lines, lines

    sections_mod._detect_page_heading_signals = old_detect
    try:
        old_sections = build_sections_from_pages(pages_df)
    finally:
        sections_mod._detect_page_heading_signals = original

    return old_sections


def map_chunk_subsections(chunks_df: pd.DataFrame, sections_df: pd.DataFrame) -> pd.Series:
    mapped: list[str] = []
    for _, row in chunks_df.iterrows():
        _, _, subsection = find_section_for_page(sections_df, int(row["page_start"]))
        mapped.append(str(subsection or "Unknown"))
    return pd.Series(mapped, index=chunks_df.index)


def compute_doc_stats(doc_dir: Path) -> tuple[DocStats, pd.DataFrame]:
    pages = pd.read_parquet(doc_dir / "pages.parquet")
    chunks = pd.read_parquet(doc_dir / "chunks.parquet")

    old_sections = build_sections_with_old_subsection_logic(pages)
    new_sections = build_sections_from_pages(pages)

    old_chunk_sub = map_chunk_subsections(chunks, old_sections)
    new_chunk_sub = map_chunk_subsections(chunks, new_sections)

    old_page_sub = {
        int(p): str(sub)
        for p, sub in zip(old_sections["page_start"], old_sections["subsection_title"])
    }
    new_page_sub = {
        int(p): str(sub)
        for p, sub in zip(new_sections["page_start"], new_sections["subsection_title"])
    }

    old_unknown_chunk_mask = old_chunk_sub.eq("Unknown")
    new_unknown_chunk_mask = new_chunk_sub.eq("Unknown")
    unknown_to_known_chunk_mask = old_unknown_chunk_mask & ~new_unknown_chunk_mask
    changed_known_chunk_mask = (
        ~old_unknown_chunk_mask
        & ~new_unknown_chunk_mask
        & old_chunk_sub.ne(new_chunk_sub)
    )

    page_labels = sorted(set(pages["page"].astype(int).tolist()))
    old_unknown_pages = sum(1 for p in page_labels if old_page_sub.get(p, "Unknown") == "Unknown")
    new_unknown_pages = sum(1 for p in page_labels if new_page_sub.get(p, "Unknown") == "Unknown")
    unknown_to_known_pages = sum(
        1
        for p in page_labels
        if old_page_sub.get(p, "Unknown") == "Unknown" and new_page_sub.get(p, "Unknown") != "Unknown"
    )

    changed_rows = chunks.loc[old_chunk_sub.ne(new_chunk_sub), ["chunk_id", "page_start", "page_end"]].copy()
    changed_rows["old_subsection_title"] = old_chunk_sub.loc[changed_rows.index]
    changed_rows["new_subsection_title"] = new_chunk_sub.loc[changed_rows.index]
    changed_rows.insert(0, "doc_id", doc_dir.name)

    stats = DocStats(
        doc_id=doc_dir.name,
        total_chunks=int(len(chunks)),
        pages=int(pages["page"].nunique()),
        old_unknown_chunks=int(old_unknown_chunk_mask.sum()),
        new_unknown_chunks=int(new_unknown_chunk_mask.sum()),
        unknown_to_known_chunks=int(unknown_to_known_chunk_mask.sum()),
        changed_known_to_different_known_chunks=int(changed_known_chunk_mask.sum()),
        unchanged_chunks=int(old_chunk_sub.eq(new_chunk_sub).sum()),
        old_unknown_pages=int(old_unknown_pages),
        new_unknown_pages=int(new_unknown_pages),
        unknown_to_known_pages=int(unknown_to_known_pages),
    )
    return stats, changed_rows


def render_chart(summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    docs = summary_df["doc_id"].tolist()
    x = list(range(len(docs)))
    width = 0.35

    axes[0].bar(
        [i - width / 2 for i in x],
        summary_df["old_unknown_chunk_pct"],
        width=width,
        label="Before",
        color="#B7B7B7",
        edgecolor="#4A4A4A",
    )
    axes[0].bar(
        [i + width / 2 for i in x],
        summary_df["new_unknown_chunk_pct"],
        width=width,
        label="After",
        color="#2F855A",
        edgecolor="#1F5E3D",
    )
    axes[0].set_title("Unknown subsection rate by chunk")
    axes[0].set_ylabel("Percent of chunks")
    axes[0].set_xticks(x, docs, rotation=25, ha="right")
    axes[0].legend(frameon=False)
    axes[0].set_ylim(0, max(summary_df["old_unknown_chunk_pct"].max(), summary_df["new_unknown_chunk_pct"].max()) * 1.18)

    axes[1].bar(
        x,
        summary_df["unknown_to_known_chunks"],
        color="#C05621",
        edgecolor="#7C3A12",
    )
    axes[1].set_title("Chunks improved from Unknown to known")
    axes[1].set_ylabel("Chunk count")
    axes[1].set_xticks(x, docs, rotation=25, ha="right")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.35)

    fig.suptitle("Effect of numbered-heading subsection detection", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_markdown(summary_df: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# Subsection Heading Refresh Summary",
        "",
        "The table compares the old subsection heuristic (lettered subsections only) against the new default behavior that also accepts numbered headings.",
        "",
        "| Report | Unknown chunks before | Unknown chunks after | Unknown→known chunks | Other known-label changes |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['doc_id']} | {int(row['old_unknown_chunks'])} ({row['old_unknown_chunk_pct']:.1f}%) | "
            f"{int(row['new_unknown_chunks'])} ({row['new_unknown_chunk_pct']:.1f}%) | "
            f"{int(row['unknown_to_known_chunks'])} | {int(row['changed_known_to_different_known_chunks'])} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize subsection-heading refresh effects for the canonical Grampian corpora.")
    parser.add_argument("--data-root", type=Path, default=Path("data_processed"))
    parser.add_argument("--docs", nargs="*", default=DEFAULT_DOCS)
    parser.add_argument("--out-dir", type=Path, default=Path("results/subsection_heading_refresh"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    changed_frames: list[pd.DataFrame] = []

    for doc in args.docs:
        doc_dir = args.data_root / doc
        stats, changed_rows = compute_doc_stats(doc_dir)
        summary_rows.append(
            {
                "doc_id": stats.doc_id,
                "pages": stats.pages,
                "total_chunks": stats.total_chunks,
                "old_unknown_chunks": stats.old_unknown_chunks,
                "new_unknown_chunks": stats.new_unknown_chunks,
                "old_unknown_chunk_pct": stats.old_unknown_chunk_pct,
                "new_unknown_chunk_pct": stats.new_unknown_chunk_pct,
                "unknown_to_known_chunks": stats.unknown_to_known_chunks,
                "changed_known_to_different_known_chunks": stats.changed_known_to_different_known_chunks,
                "unchanged_chunks": stats.unchanged_chunks,
                "old_unknown_pages": stats.old_unknown_pages,
                "new_unknown_pages": stats.new_unknown_pages,
                "old_unknown_page_pct": stats.old_unknown_page_pct,
                "new_unknown_page_pct": stats.new_unknown_page_pct,
                "unknown_to_known_pages": stats.unknown_to_known_pages,
            }
        )
        if not changed_rows.empty:
            changed_frames.append(changed_rows)

    summary_df = pd.DataFrame(summary_rows).sort_values("doc_id")
    summary_csv = out_dir / "subsection_heading_refresh_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    changed_csv = out_dir / "subsection_heading_refresh_changed_chunks.csv"
    pd.concat(changed_frames, ignore_index=True).to_csv(changed_csv, index=False)

    chart_png = out_dir / "subsection_heading_refresh_summary.png"
    render_chart(summary_df, chart_png)

    summary_md = out_dir / "subsection_heading_refresh_summary.md"
    write_markdown(summary_df, summary_md)

    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {changed_csv}")
    print(f"Wrote: {chart_png}")
    print(f"Wrote: {summary_md}")
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
