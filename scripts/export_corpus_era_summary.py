#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd


TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
DOC_ID_RE = re.compile(r"^(?P<prefix>.+)-(?P<start>\d{4})-(?P<end>\d{4})$")

DEFAULT_ERAS = (
    "2004-2009:2004:2009",
    "2010-2018:2010:2018",
    "2019-2025:2019:2025",
)

THEME_EXCLUDE = {
    "performance report",
    "accountability report",
    "financial statements",
    "unknown",
    "risk title",
    "directors report",
    "director s report",
    "overview",
    "a overview",
    "scottish parliament",
    "directions by the scottish ministers",
    "i director s report",
    "independent auditor s report",
    "statement of changes in taxpayers equity",
    "accounting policies",
    "notes to the accounts",
    "sub total",
    "intra group",
    "corresponding amounts",
    "useful life",
    "operating leases",
    "charitable endowment funds",
    "remuneration report",
    "iii statement of health board members responsibilities in respect of the accounts",
    "statement of the chief executive s responsibilities as the accountable",
    "statement of the chief executive s responsibilities as the accountable officer of the health board",
    "ec carbon",
    "hch income",
    "assets under",
    "by provider",
    "cash equivalent",
    "mr richard carey",
}

THEME_BANNED_WORDS = {
    "account",
    "accounts",
    "asset",
    "assets",
    "audit",
    "auditor",
    "balance",
    "cash",
    "employee",
    "employees",
    "equity",
    "financial",
    "flow",
    "lease",
    "leases",
    "liabilities",
    "liability",
    "payables",
    "receivables",
    "remuneration",
    "sheet",
    "statement",
    "statements",
    "taxpayers",
}

THEME_PREFIX_EXCLUDE = (
    "statement of financial position",
    "consolidated statement",
    "cash flow statement",
    "statement of comprehensive net expenditure",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an era-level corpus summary table from processed report artifacts."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data_processed"),
        help="Directory containing processed report folders.",
    )
    parser.add_argument(
        "--doc-prefix",
        default="Grampian",
        help="Only include doc IDs matching '<doc-prefix>-YYYY-YYYY'.",
    )
    parser.add_argument(
        "--era",
        action="append",
        dest="eras",
        help="Era definition in the form LABEL:START_YEAR:END_YEAR. Can be passed multiple times.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/corpus_era_summary_grampian.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("results/corpus_era_summary_grampian.md"),
        help="Markdown output path.",
    )
    return parser.parse_args()


def parse_era_specs(raw_eras: list[str] | None) -> list[tuple[str, int, int]]:
    specs = raw_eras or list(DEFAULT_ERAS)
    eras: list[tuple[str, int, int]] = []
    for spec in specs:
        try:
            label, start_raw, end_raw = spec.split(":")
            start_year = int(start_raw)
            end_year = int(end_raw)
        except ValueError as exc:
            raise ValueError(f"Invalid era spec '{spec}'. Expected LABEL:START_YEAR:END_YEAR.") from exc
        if end_year < start_year:
            raise ValueError(f"Invalid era spec '{spec}'. END_YEAR must be >= START_YEAR.")
        eras.append((label, start_year, end_year))
    return eras


def iter_doc_dirs(data_root: Path, doc_prefix: str) -> list[Path]:
    matches: list[Path] = []
    for doc_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        match = DOC_ID_RE.match(doc_dir.name)
        if not match:
            continue
        if match.group("prefix") != doc_prefix:
            continue
        required = ("pages.parquet", "sections.csv")
        if all((doc_dir / name).exists() for name in required):
            matches.append(doc_dir)
    return matches


def doc_start_year(doc_dir: Path) -> int:
    match = DOC_ID_RE.match(doc_dir.name)
    if not match:
        raise ValueError(f"Unexpected doc directory format: {doc_dir.name}")
    return int(match.group("start"))


def normalize_theme(value: str) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    substitutions = (
        (r"^[A-Z]\)\s*", ""),
        (r"^[ivx]+\)\s*", ""),
        (r"^note\s+\d+[a-z]?\.?\s*", ""),
        (r"^\(?[a-z]\)\s*", ""),
        (r"\([^)]*\)", " "),
        (r"[^a-z0-9\s]", " "),
        (r"\s+", " "),
    )
    for pattern, replacement in substitutions:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = text.strip()
    if not text or text in THEME_EXCLUDE:
        return None
    if any(ch.isdigit() for ch in text):
        return None
    if text.startswith(THEME_PREFIX_EXCLUDE):
        return None

    words = text.split()
    if len(words) < 2 or len(words) > 6:
        return None
    banned_count = sum(1 for word in words if word in THEME_BANNED_WORDS)
    if banned_count >= max(2, len(words) - 1):
        return None
    return text


def count_words(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def summarize_era(doc_dirs: list[Path], label: str) -> dict[str, object]:
    doc_word_counts: list[int] = []
    unique_tokens: set[str] = set()
    theme_counts: Counter[str] = Counter()
    total_pages = 0
    image_based_pages = 0

    for doc_dir in doc_dirs:
        pages_df = pd.read_parquet(doc_dir / "pages.parquet", columns=["clean_text", "extractor"])
        page_tokens: list[str] = []
        for text in pages_df["clean_text"].fillna("").astype(str):
            tokens = count_words(text)
            page_tokens.extend(tokens)
            unique_tokens.update(tokens)
        doc_word_counts.append(len(page_tokens))
        total_pages += len(pages_df)
        image_based_pages += int((pages_df["extractor"] == "ocr").sum())

        sections_df = pd.read_csv(doc_dir / "sections.csv", usecols=["section_title", "subsection_title"])
        for column in ("section_title", "subsection_title"):
            for value in sections_df[column].dropna():
                theme = normalize_theme(str(value))
                if theme:
                    theme_counts[theme] += 1

    if not doc_word_counts:
        raise ValueError(f"No documents found for era '{label}'.")

    top_themes = [theme.title() for theme, _ in theme_counts.most_common(3)]
    return {
        "Era": label,
        "Avg. Word Count": round(sum(doc_word_counts) / len(doc_word_counts)),
        "Unique Tokens": len(unique_tokens),
        "Top 3 Recurring Themes": "; ".join(top_themes) if top_themes else "",
        "% Image-based pages": round((image_based_pages / total_pages) * 100.0, 1) if total_pages else 0.0,
    }


def build_summary(data_root: Path, doc_prefix: str, eras: list[tuple[str, int, int]]) -> pd.DataFrame:
    doc_dirs = iter_doc_dirs(data_root, doc_prefix)
    rows: list[dict[str, object]] = []
    for label, start_year, end_year in eras:
        era_docs = [doc_dir for doc_dir in doc_dirs if start_year <= doc_start_year(doc_dir) <= end_year]
        rows.append(summarize_era(era_docs, label))
    return pd.DataFrame(rows)


def write_markdown(df: pd.DataFrame, out_path: Path) -> None:
    lines = [df.to_markdown(index=False)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    eras = parse_era_specs(args.eras)
    summary_df = build_summary(args.data_root, args.doc_prefix, eras)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.out_csv, index=False)
    write_markdown(summary_df, args.out_md)

    print(summary_df.to_markdown(index=False))
    print(f"\nWrote {args.out_csv}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
