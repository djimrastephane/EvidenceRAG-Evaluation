from __future__ import annotations

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

from rag_pdf.chunking import count_tokens, get_encoder

DATA_ROOT = REPO_ROOT / "data_variants" / "tiktoken_all_docs_224_56"
OUT_PATH = REPO_ROOT / "docs" / "figures" / "grampian_token_distribution_21docs.png"


def load_page_tokens() -> tuple[pd.Series, int, int, int]:
    enc = get_encoder()
    values: list[int] = []
    pages_total = 0
    ocr_pages_total = 0
    doc_count = 0

    for doc_dir in sorted(DATA_ROOT.glob("Grampian-*")):
        metrics_path = doc_dir / "metrics.json"
        pages_path = doc_dir / "pages.parquet"
        if not metrics_path.exists() or not pages_path.exists():
            continue

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        counts = metrics.get("counts", {})
        pages_df = pd.read_parquet(pages_path, columns=["clean_text"])

        values.extend(count_tokens(str(text or ""), enc) for text in pages_df["clean_text"])
        pages_total += int(counts.get("pages_total", len(pages_df)))
        ocr_pages_total += int(counts.get("ocr_raw_pages_accepted", 0))
        ocr_pages_total += int(counts.get("ocr_short_pages_accepted", 0))
        doc_count += 1

    return pd.Series(values, dtype="float64"), pages_total, ocr_pages_total, doc_count


def main() -> None:
    tokens, pages_total, ocr_pages_total, doc_count = load_page_tokens()

    median = float(tokens.median())
    q75 = float(tokens.quantile(0.75))
    q90 = float(tokens.quantile(0.90))
    pct_lt_150 = float((tokens < 150).mean() * 100.0)
    pct_gt_500 = float((tokens > 500).mean() * 100.0)
    ocr_pct = float((ocr_pages_total / pages_total) * 100.0)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    bins = list(range(0, 1251, 50))
    ax.hist(
        tokens,
        bins=bins,
        color="#88a7b1",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.95,
    )

    ax.axvline(median, color="#1f4e5f", linewidth=2.0, label=f"Median: {median:.0f}")
    ax.axvline(q75, color="#ba6f3b", linewidth=1.8, linestyle="--", label=f"75th percentile: {q75:.0f}")
    ax.axvline(q90, color="#8c3d3d", linewidth=1.8, linestyle=":", label=f"90th percentile: {q90:.0f}")

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
        f"<150 tokens\n{pct_lt_150:.1f}% of pages",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#284b53",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#b8d8d6"),
    )
    ax.text(
        1010,
        ax.get_ylim()[1] * 0.93,
        f">500 tokens\n{pct_gt_500:.1f}% of pages",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#6e3d2c",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#e5c8b8"),
    )

    summary_text = (
        f"{doc_count} documents | {pages_total} pages\n"
        f"OCR-processed pages: {ocr_pages_total} ({ocr_pct:.1f}%)"
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
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
