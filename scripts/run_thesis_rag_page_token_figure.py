from __future__ import annotations

"""Generate the 21-report page-token figure directly from thesis_rag page artifacts.

This script intentionally stops after page extraction and cleaning because the
figure depends only on page text and OCR flags, not on chunk construction or
indexing. It therefore provides a faster and cleaner validation path when
checking whether the thesis token-distribution figure still holds under the
refactored pipeline.
"""

import argparse
from pathlib import Path
import sys

import _matplotlib_env
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from thesis_rag.artifacts import save_pages
from thesis_rag.chunking import count_tokens, get_encoder
from thesis_rag.config import load_config
from thesis_rag.loader import discover_documents, extract_page_structures
from thesis_rag.preprocessing import build_page_records
from thesis_rag.utils import write_json


def parse_args() -> argparse.Namespace:
    """Parse config and output directory for the page-token rerun."""
    parser = argparse.ArgumentParser(description="Generate the 21-report page-token figure from thesis_rag page artifacts.")
    parser.add_argument("--config", required=True, help="Path to the thesis_rag YAML config.")
    parser.add_argument("--output-dir", required=True, help="Directory to write page artifacts and figure outputs.")
    return parser.parse_args()


def _build_stats(values: list[int], pages_total: int, ocr_pages_total: int, doc_count: int) -> dict[str, float | int]:
    """Aggregate the summary numbers shown in the figure."""
    tokens = pd.Series(values, dtype="float64")
    return {
        "doc_count": doc_count,
        "pages_total": pages_total,
        "ocr_pages_total": ocr_pages_total,
        "ocr_pct": float((ocr_pages_total / pages_total) * 100.0) if pages_total else 0.0,
        "median": float(tokens.median()),
        "q75": float(tokens.quantile(0.75)),
        "q90": float(tokens.quantile(0.90)),
        "pct_lt_150": float((tokens < 150).mean() * 100.0),
        "pct_gt_500": float((tokens > 500).mean() * 100.0),
    }


def _plot_distribution(values: list[int], stats: dict[str, float | int], out_path: Path) -> None:
    """Render the token-distribution histogram in the same style as the thesis figure."""
    tokens = pd.Series(values, dtype="float64")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    bins = list(range(0, 1251, 50))
    ax.hist(tokens, bins=bins, color="#88a7b1", edgecolor="white", linewidth=0.8, alpha=0.95)
    ax.axvline(float(stats["median"]), color="#1f4e5f", linewidth=2.0, label=f"Median: {float(stats['median']):.0f}")
    ax.axvline(float(stats["q75"]), color="#ba6f3b", linewidth=1.8, linestyle="--", label=f"75th percentile: {float(stats['q75']):.0f}")
    ax.axvline(float(stats["q90"]), color="#8c3d3d", linewidth=1.8, linestyle=":", label=f"90th percentile: {float(stats['q90']):.0f}")
    ax.axvspan(0, 150, color="#dceeed", alpha=0.55)
    ax.axvspan(500, 1250, color="#f4e1d7", alpha=0.5)
    ax.set_title("Page Token Distribution Across 21 NHS Grampian Reports", fontsize=14, fontweight="bold")
    ax.set_xlabel("Page length (tiktoken tokens)", fontsize=11)
    ax.set_ylabel("Number of pages", fontsize=11)
    ax.set_xlim(0, 1250)
    ax.set_ylim(bottom=0)
    ax.text(
        76,
        ax.get_ylim()[1] * 0.80,
        f"<150 tokens\n{float(stats['pct_lt_150']):.1f}% of pages",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#284b53",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#b8d8d6"),
    )
    ax.text(
        1010,
        ax.get_ylim()[1] * 0.93,
        f">500 tokens\n{float(stats['pct_gt_500']):.1f}% of pages",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#6e3d2c",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#e5c8b8"),
    )
    summary_text = (
        f"{int(stats['doc_count'])} documents | {int(stats['pages_total'])} pages\n"
        f"OCR-processed pages: {int(stats['ocr_pages_total'])} ({float(stats['ocr_pct']):.1f}%)"
    )
    ax.text(
        0.985,
        0.80,
        summary_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cfcfcf"),
    )
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), frameon=True, fontsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run page extraction/cleaning over the 21-report corpus and write figure artifacts."""
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    page_root = output_dir / "page_artifacts"
    page_root.mkdir(parents=True, exist_ok=True)
    enc = get_encoder()
    token_values: list[int] = []
    pages_total = 0
    ocr_pages_total = 0
    doc_count = 0

    for document in discover_documents(config.paths.data_dir):
        page_structs = extract_page_structures(document)
        pages = build_page_records(document.doc_id, page_structs, config.ocr)
        save_pages(pages, page_root / document.doc_id)
        token_values.extend(count_tokens(str(page.clean_text or ""), enc) for page in pages)
        pages_total += len(pages)
        ocr_pages_total += sum(page.ocr_used for page in pages)
        doc_count += 1

    stats = _build_stats(token_values, pages_total, ocr_pages_total, doc_count)
    write_json(output_dir / "thesis_rag_page_token_stats.json", stats)
    _plot_distribution(token_values, stats, output_dir / "grampian_token_distribution_21docs_thesis_rag.png")
    print(output_dir)


if __name__ == "__main__":
    main()
