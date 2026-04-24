from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an acronym-by-document 1/0 presence matrix with summary counts."
    )
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=Path("results/query_inventory/acronym_glossary_candidates_grampian_full_2004_2025.csv"),
        help="CSV containing the acronym list to export.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data_processed"),
        help="Root directory containing Grampian-* processed documents.",
    )
    parser.add_argument(
        "--doc-glob",
        type=str,
        default="Grampian-*",
        help="Glob for document folders under --data-root.",
    )
    parser.add_argument(
        "--output-matrix",
        type=Path,
        default=Path("results/query_inventory/acronym_document_presence_matrix.csv"),
        help="Output CSV for acronym x document 1/0 matrix.",
    )
    parser.add_argument(
        "--output-doc-counts",
        type=Path,
        default=Path("results/query_inventory/acronym_document_presence_counts_by_doc.csv"),
        help="Output CSV with acronym counts per document.",
    )
    parser.add_argument(
        "--output-acronym-counts",
        type=Path,
        default=Path("results/query_inventory/acronym_document_presence_counts_by_acronym.csv"),
        help="Output CSV with document counts per acronym.",
    )
    return parser.parse_args()


def sorted_doc_dirs(data_root: Path, doc_glob: str) -> list[Path]:
    docs = [p for p in data_root.glob(doc_glob) if p.is_dir()]
    docs.sort(key=lambda p: tuple(int(part) for part in re.findall(r"\d{4}", p.name)[:2]))
    return docs


def load_acronyms(candidate_csv: Path) -> list[str]:
    acronyms: list[str] = []
    with candidate_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            acronym = (row.get("acronym") or "").strip()
            if acronym:
                acronyms.append(acronym)
    return acronyms


def load_document_text(doc_dir: Path) -> str:
    sections_path = doc_dir / "sections.csv"
    if not sections_path.exists():
        return ""
    parts: list[str] = []
    with sections_path.open(encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            text = (row.get("section_text") or "").strip()
            if text:
                parts.append(text)
    return "\n".join(parts)


def build_patterns(acronyms: list[str]) -> dict[str, re.Pattern[str]]:
    return {
        acronym: re.compile(rf"(?<![A-Za-z0-9]){re.escape(acronym)}(?![A-Za-z0-9])", re.IGNORECASE)
        for acronym in acronyms
    }


def main() -> None:
    args = parse_args()
    acronyms = load_acronyms(args.candidate_csv)
    doc_dirs = sorted_doc_dirs(args.data_root, args.doc_glob)
    patterns = build_patterns(acronyms)

    matrix_rows: list[dict[str, int | str]] = []
    doc_counts = {doc_dir.name: 0 for doc_dir in doc_dirs}
    acronym_counts = {acronym: 0 for acronym in acronyms}

    texts = {doc_dir.name: load_document_text(doc_dir) for doc_dir in doc_dirs}

    for acronym in acronyms:
        row: dict[str, int | str] = {"acronym": acronym}
        pattern = patterns[acronym]
        total = 0
        for doc_dir in doc_dirs:
            present = 1 if pattern.search(texts[doc_dir.name]) else 0
            row[doc_dir.name] = present
            total += present
            doc_counts[doc_dir.name] += present
        row["doc_count"] = total
        acronym_counts[acronym] = total
        matrix_rows.append(row)

    args.output_matrix.parent.mkdir(parents=True, exist_ok=True)
    with args.output_matrix.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["acronym"] + [doc_dir.name for doc_dir in doc_dirs] + ["doc_count"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matrix_rows)

    with args.output_doc_counts.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["document", "acronym_count"])
        writer.writeheader()
        for doc_dir in doc_dirs:
            writer.writerow({"document": doc_dir.name, "acronym_count": doc_counts[doc_dir.name]})

    with args.output_acronym_counts.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["acronym", "document_count"])
        writer.writeheader()
        for acronym in acronyms:
            writer.writerow({"acronym": acronym, "document_count": acronym_counts[acronym]})

    print(args.output_matrix)
    print(args.output_doc_counts)
    print(args.output_acronym_counts)


if __name__ == "__main__":
    main()
