from __future__ import annotations

"""Compare legacy and thesis_rag page-token distributions for the 21 Grampian reports.

The script reads page artifacts from a thesis_rag preprocess run, recomputes the
same tiktoken-based summary statistics used in the original figure, regenerates
the histogram, and writes a side-by-side comparison bundle for thesis audit
purposes.
"""

import argparse
import json
from pathlib import Path
import sys

import _matplotlib_env
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from thesis_rag.chunking import count_tokens, get_encoder
from thesis_rag.utils import write_json

LEGACY_ROOT = REPO_ROOT / "data_variants" / "tiktoken_all_docs_224_56"


def parse_args() -> argparse.Namespace:
    """Parse input/output paths for the token-distribution comparison."""
    parser = argparse.ArgumentParser(description="Compare legacy and thesis_rag page-token distributions.")
    parser.add_argument("--run-dir", required=True, help="Path to a thesis_rag preprocess run directory.")
    parser.add_argument("--output-dir", required=True, help="Directory to write the comparison bundle.")
    return parser.parse_args()


def _collect_stats_from_thesis_run(run_dir: Path) -> dict[str, float | int]:
    """Compute token-distribution statistics from thesis_rag page artifacts."""
    enc = get_encoder()
    values: list[int] = []
    pages_total = 0
    ocr_pages_total = 0
    doc_count = 0

    for doc_dir in sorted(path for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("Grampian-")):
        pages_path = doc_dir / "pages.parquet"
        if not pages_path.exists():
            continue
        pages_df = pd.read_parquet(pages_path, columns=["clean_text", "ocr_used"])
        values.extend(count_tokens(str(text or ""), enc) for text in pages_df["clean_text"])
        pages_total += int(len(pages_df))
        ocr_pages_total += int(pages_df["ocr_used"].sum())
        doc_count += 1

    return _build_stats(values, pages_total, ocr_pages_total, doc_count)


def _collect_stats_from_legacy_artifacts() -> dict[str, float | int]:
    """Compute the published legacy statistics from the archived artifacts."""
    enc = get_encoder()
    values: list[int] = []
    pages_total = 0
    ocr_pages_total = 0
    doc_count = 0

    for doc_dir in sorted(path for path in LEGACY_ROOT.iterdir() if path.is_dir() and path.name.startswith("Grampian-")):
        pages_path = doc_dir / "pages.parquet"
        metrics_path = doc_dir / "metrics.json"
        if not pages_path.exists() or not metrics_path.exists():
            continue
        pages_df = pd.read_parquet(pages_path, columns=["clean_text"])
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        counts = metrics.get("counts", {})
        values.extend(count_tokens(str(text or ""), enc) for text in pages_df["clean_text"])
        pages_total += int(counts.get("pages_total", len(pages_df)))
        ocr_pages_total += int(counts.get("ocr_raw_pages_accepted", 0))
        ocr_pages_total += int(counts.get("ocr_short_pages_accepted", 0))
        doc_count += 1

    return _build_stats(values, pages_total, ocr_pages_total, doc_count)


def _build_stats(values: list[int], pages_total: int, ocr_pages_total: int, doc_count: int) -> dict[str, float | int]:
    """Aggregate the summary numbers shown in the thesis figure."""
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


def _plot_distribution(run_dir: Path, stats: dict[str, float | int], out_path: Path) -> None:
    """Regenerate the page-token histogram using thesis_rag page artifacts."""
    enc = get_encoder()
    values: list[int] = []
    for doc_dir in sorted(path for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("Grampian-")):
        pages_path = doc_dir / "pages.parquet"
        if not pages_path.exists():
            continue
        pages_df = pd.read_parquet(pages_path, columns=["clean_text"])
        values.extend(count_tokens(str(text or ""), enc) for text in pages_df["clean_text"])
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


def _write_comparison(legacy: dict[str, float | int], thesis_rag: dict[str, float | int], out_dir: Path) -> None:
    """Write side-by-side comparison files for thesis traceability."""
    rows = []
    for key in ["doc_count", "pages_total", "ocr_pages_total", "ocr_pct", "median", "q75", "q90", "pct_lt_150", "pct_gt_500"]:
        legacy_value = legacy[key]
        thesis_value = thesis_rag[key]
        rows.append(
            {
                "metric": key,
                "legacy": legacy_value,
                "thesis_rag": thesis_value,
                "delta": float(thesis_value) - float(legacy_value),
                "match_rounded": round(float(legacy_value), 3) == round(float(thesis_value), 3),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "page_token_distribution_comparison.csv", index=False)
    write_json(out_dir / "legacy_stats.json", legacy)
    write_json(out_dir / "thesis_rag_stats.json", thesis_rag)


def main() -> None:
    """Generate thesis_rag token-distribution stats and compare them to legacy."""
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    legacy = _collect_stats_from_legacy_artifacts()
    thesis_rag = _collect_stats_from_thesis_run(run_dir)
    _write_comparison(legacy, thesis_rag, out_dir)
    _plot_distribution(run_dir, thesis_rag, out_dir / "grampian_token_distribution_21docs_thesis_rag.png")
    print(out_dir)


if __name__ == "__main__":
    main()
