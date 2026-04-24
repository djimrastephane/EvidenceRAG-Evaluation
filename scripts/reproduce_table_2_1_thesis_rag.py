from __future__ import annotations

"""Reproduce thesis Table 2.1 with the refactored ``thesis_rag`` pipeline.

The original era summary table was derived from legacy processed artifacts.
This script regenerates the same era-level statistics directly from the raw
Grampian PDFs using the refactored page extraction and preprocessing logic.

For each era, it reports:

- average per-document word count
- unique word types across the era
- average sentence length in words
- top recurring themes inferred from section/subsection metadata
- percentage of OCR-processed pages

It also writes a comparison against the currently committed thesis table values
so differences can be inspected before updating the manuscript.
"""

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from thesis_rag.config import load_config
from thesis_rag.loader import extract_page_structures
from thesis_rag.preprocessing import build_chunk_records, build_page_records
from thesis_rag.schemas import DocumentRecord, PageRecord
from thesis_rag.utils import now_utc_iso, write_json


DOC_ID_RE = re.compile(r"^(?P<prefix>.+)-(?P<start>\d{4})-(?P<end>\d{4})$")
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

ERAS = [
    ("2004--2009", 2004, 2009),
    ("2010--2018", 2010, 2018),
    ("2019--2021", 2019, 2021),
    ("2022--2025", 2022, 2025),
]

TABLE_2_1_REFERENCE = {
    "2004--2009": {
        "avg_word_count": 15474,
        "unique_tokens": 3173,
        "avg_sent_length": 29.0,
        "themes": ["Internal control", "Sustainability", "Clinical services"],
        "ocr_pct": 1.7,
    },
    "2010--2018": {
        "avg_word_count": 28777,
        "unique_tokens": 6495,
        "avg_sent_length": 28.3,
        "themes": ["Risk management", "Strategic reporting", "Special payments"],
        "ocr_pct": 77.0,
    },
    "2019--2021": {
        "avg_word_count": 42536,
        "unique_tokens": 4206,
        "avg_sent_length": 28.7,
        "themes": ["Corporate governance", "Staff remuneration", "Integration boards"],
        "ocr_pct": 0.5,
    },
    "2022--2025": {
        "avg_word_count": 45283,
        "unique_tokens": 4381,
        "avg_sent_length": 27.2,
        "themes": ["Corporate governance", "Performance analysis", "Staff remuneration"],
        "ocr_pct": 0.0,
    },
}

THEME_EXCLUDE = {
    "unknown",
    "performance report",
    "accountability report",
    "financial statements",
    "overview",
    "a overview",
    "b performance analysis",
    "performance analysis",
}

THEME_REMAP = {
    "statement on internal control": "Internal control",
    "sustainability and environmental reporting": "Sustainability",
    "clinical services costs": "Clinical services",
    "financial risk factors": "Risk management",
    "strategic report": "Strategic reporting",
    "losses and special payments": "Special payments",
    "corporate governance report": "Corporate governance",
    "remuneration and staff report": "Staff remuneration",
    "integration joint boards": "Integration boards",
    "b performance analysis": "Performance analysis",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce thesis Table 2.1 from thesis_rag outputs.")
    parser.add_argument(
        "--pipeline-config",
        default="configs/thesis_rag.yaml",
        help="Base thesis_rag YAML config.",
    )
    parser.add_argument(
        "--pdf-root",
        default="Data/Annual Accounts NHS Grampian/Preliminary_Test",
        help="Directory containing Grampian PDF files.",
    )
    parser.add_argument(
        "--bundle-dir",
        default="",
        help="Optional explicit output bundle directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing bundle directory if it already exists.",
    )
    return parser.parse_args()


def _default_bundle_dir() -> Path:
    return REPO_ROOT / "results" / "thesis_validations" / f"table_2_1_era_summary_{date.today().isoformat()}"


def _doc_start_year(doc_id: str) -> int:
    match = DOC_ID_RE.match(doc_id)
    if not match:
        raise ValueError(f"Unexpected doc id format: {doc_id}")
    return int(match.group("start"))


def _discover_grampian_pdfs(pdf_root: Path) -> list[DocumentRecord]:
    docs = [DocumentRecord(doc_id=path.stem, pdf_path=str(path)) for path in sorted(pdf_root.glob("Grampian-*.pdf"))]
    if len(docs) != 21:
        raise ValueError(f"Expected 21 Grampian PDFs, found {len(docs)} in {pdf_root}")
    return docs


def _tokenize_words(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text or "")]


def _sentence_lengths(text: str) -> list[int]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return []
    parts = [part.strip() for part in SENT_SPLIT_RE.split(cleaned) if part.strip()]
    lengths: list[int] = []
    for part in parts:
        count = len(_tokenize_words(part))
        if count > 0:
            lengths.append(count)
    return lengths


def _normalize_theme(value: str) -> str | None:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    text = re.sub(r"^[ivx]+\)\s*", "", text)
    text = re.sub(r"^[a-z]\)\s*", "", text)
    text = text.strip()
    if not text or text in THEME_EXCLUDE:
        return None
    mapped = THEME_REMAP.get(text)
    return mapped or text.title()


def _collect_doc_metrics(document: DocumentRecord, config) -> dict[str, object]:
    page_structs = extract_page_structures(document)
    pages = build_page_records(document.doc_id, page_structs, config.ocr)
    chunks = build_chunk_records(document.doc_id, pages, config.chunking, source_pdf_path=document.pdf_path)
    unique_themes: set[tuple[int, str]] = set()
    theme_counter: Counter[str] = Counter()
    for chunk in chunks:
        page_num = int(chunk.page_start or chunk.page_number)
        for raw in (chunk.subsection_title, chunk.section_title):
            theme = _normalize_theme(str(raw or ""))
            if not theme:
                continue
            key = (page_num, theme)
            if key not in unique_themes:
                unique_themes.add(key)
                theme_counter[theme] += 1
    all_text = " ".join(page.clean_text for page in pages if page.clean_text)
    return {
        "doc_id": document.doc_id,
        "start_year": _doc_start_year(document.doc_id),
        "word_count": len(_tokenize_words(all_text)),
        "unique_tokens": set(_tokenize_words(all_text)),
        "sentence_lengths": [length for page in pages for length in _sentence_lengths(page.clean_text)],
        "ocr_pages": sum(1 for page in pages if page.ocr_used),
        "page_count": len(pages),
        "theme_counter": theme_counter,
    }


def _summarize_era(label: str, doc_rows: list[dict[str, object]]) -> dict[str, object]:
    word_counts = [int(row["word_count"]) for row in doc_rows]
    unique_tokens: set[str] = set()
    sentence_lengths: list[int] = []
    total_pages = 0
    total_ocr_pages = 0
    theme_counter: Counter[str] = Counter()
    for row in doc_rows:
        unique_tokens.update(row["unique_tokens"])  # type: ignore[arg-type]
        sentence_lengths.extend(row["sentence_lengths"])  # type: ignore[arg-type]
        total_pages += int(row["page_count"])
        total_ocr_pages += int(row["ocr_pages"])
        theme_counter.update(row["theme_counter"])  # type: ignore[arg-type]
    top_themes = [theme for theme, _ in theme_counter.most_common(3)]
    return {
        "Era": label,
        "Avg. Word Count": round(sum(word_counts) / max(len(word_counts), 1)),
        "Unique Tokens": len(unique_tokens),
        "Avg. Sent. Length": round(sum(sentence_lengths) / max(len(sentence_lengths), 1), 1),
        "Recurring Themes": " | ".join(top_themes),
        "% OCR-processed pages": round((total_ocr_pages / max(total_pages, 1)) * 100.0, 1),
    }


def _write_markdown(path: Path, df: pd.DataFrame) -> None:
    path.write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")


def _comparison_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in df.to_dict(orient="records"):
        era = str(row["Era"])
        ref = TABLE_2_1_REFERENCE[era]
        rows.append(
            {
                "Era": era,
                "avg_word_count_old": ref["avg_word_count"],
                "avg_word_count_new": row["Avg. Word Count"],
                "avg_word_count_diff": int(row["Avg. Word Count"]) - int(ref["avg_word_count"]),
                "unique_tokens_old": ref["unique_tokens"],
                "unique_tokens_new": row["Unique Tokens"],
                "unique_tokens_diff": int(row["Unique Tokens"]) - int(ref["unique_tokens"]),
                "avg_sent_length_old": ref["avg_sent_length"],
                "avg_sent_length_new": row["Avg. Sent. Length"],
                "avg_sent_length_diff": round(float(row["Avg. Sent. Length"]) - float(ref["avg_sent_length"]), 1),
                "ocr_pct_old": ref["ocr_pct"],
                "ocr_pct_new": row["% OCR-processed pages"],
                "ocr_pct_diff": round(float(row["% OCR-processed pages"]) - float(ref["ocr_pct"]), 1),
                "themes_old": " | ".join(ref["themes"]),
                "themes_new": row["Recurring Themes"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Regenerate Table 2.1 era summary statistics from raw PDFs and write CSV/JSON/Markdown outputs."""
    args = parse_args()
    config = load_config(Path(args.pipeline_config))
    pdf_root = (REPO_ROOT / args.pdf_root).resolve()
    bundle_dir = Path(args.bundle_dir).resolve() if args.bundle_dir else _default_bundle_dir()
    if bundle_dir.exists():
        if not args.force:
            raise FileExistsError(f"{bundle_dir} already exists; pass --force to overwrite it.")
        shutil.rmtree(bundle_dir)
    (bundle_dir / "tables").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "comparison").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "configs").mkdir(parents=True, exist_ok=True)

    docs = _discover_grampian_pdfs(pdf_root)
    doc_metrics = [_collect_doc_metrics(doc, config) for doc in docs]

    rows: list[dict[str, object]] = []
    for label, start_year, end_year in ERAS:
        era_docs = [row for row in doc_metrics if start_year <= int(row["start_year"]) <= end_year]
        rows.append(_summarize_era(label, era_docs))
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(bundle_dir / "tables" / "table_2_1_thesis_rag.csv", index=False)
    _write_markdown(bundle_dir / "tables" / "table_2_1_thesis_rag.md", summary_df)

    comparison_df = _comparison_rows(summary_df)
    comparison_df.to_csv(bundle_dir / "comparison" / "table_2_1_comparison.csv", index=False)
    _write_markdown(bundle_dir / "comparison" / "table_2_1_comparison.md", comparison_df)

    write_json(
        bundle_dir / "summary.json",
        {
            "generated_utc": now_utc_iso(),
            "pdf_root": str(pdf_root),
            "doc_count": len(docs),
            "eras": ERAS,
            "table": rows,
        },
    )
    shutil.copy2(Path(args.pipeline_config), bundle_dir / "configs" / Path(args.pipeline_config).name)
    print(bundle_dir)


if __name__ == "__main__":
    main()
