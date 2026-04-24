from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9/&-]{1,11}\b")
LONGFORM_BEFORE_RE = re.compile(
    r"\b([A-Za-z][A-Za-z/&,'-]*(?:\s+[A-Za-z][A-Za-z/&,'-]*){1,11})\s+\(([A-Z][A-Z0-9/&-]{1,11})\)"
)
LONGFORM_AFTER_RE = re.compile(
    r"\b([A-Z][A-Z0-9/&-]{1,11})\s+\(([A-Za-z][A-Za-z/&,'-]*(?:\s+[A-Za-z][A-Za-z/&,'-]*){1,11})\)"
)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")

BLACKLIST = {
    "AND",
    "THE",
    "FOR",
    "WITH",
    "FROM",
    "YEAR",
    "YEARS",
    "PAGE",
    "PAGES",
    "NOTE",
    "NOTES",
    "TABLE",
    "FIGURE",
    "N",
    "NA",
    "UK",
    "USA",
    "EUR",
    "GBP",
    "VAT",
}

CONNECTOR_WORDS = {
    "and",
    "of",
    "for",
    "the",
    "to",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "a",
    "an",
    "&",
}

MOJIBAKE_REPLACEMENTS = {
    "‚Äô": "'",
    "‚Äú": '"',
    "‚Äù": '"',
    "‚Äì": "-",
    "‚Äî": "-",
    "‚Äú": '"',
    "‚Ä¶": "...",
    "‚Ä¢": "-",
    "Â£": "£",
    "Â": "",
}

CSV_FIELDS = [
    "acronym",
    "high_value_score",
    "corpus_count",
    "doc_count",
    "first_seen_doc",
    "last_seen_doc",
    "question_count",
    "question_doc_count",
    "pattern_count",
    "longform_count",
    "best_longform",
    "longforms_json",
    "question_matches_longform_count",
    "example_doc_ids",
    "example_questions_json",
    "example_contexts_json",
    "status",
    "notes",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Mine acronym glossary candidates from processed corpus chunks and eval questions, "
            "then rank them for manual glossary curation."
        )
    )
    p.add_argument("--data-root", default="data_processed", help="Root containing per-document corpora.")
    p.add_argument(
        "--doc-glob",
        default="Grampian-20*-20*",
        help="Glob for document directories under --data-root.",
    )
    p.add_argument(
        "--docs",
        default="",
        help="Optional comma-separated explicit document ids to scan instead of relying only on --doc-glob.",
    )
    p.add_argument(
        "--require-eval-set",
        action="store_true",
        help="Only include documents that contain eval_set.json.",
    )
    p.add_argument(
        "--min-corpus-count",
        type=int,
        default=2,
        help="Minimum corpus token count to keep a candidate acronym.",
    )
    p.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Drop candidates below this computed high-value score.",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="Optional maximum number of rows to write after ranking (0 = all).",
    )
    p.add_argument(
        "--output-csv",
        default="results/query_inventory/acronym_glossary_candidates.csv",
        help="Path to write the ranked glossary candidate CSV.",
    )
    p.add_argument(
        "--output-json",
        default="results/query_inventory/acronym_glossary_candidates_summary.json",
        help="Path to write the mining summary JSON.",
    )
    return p.parse_args()


def _iter_queries(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("queries"), list):
        return [row for row in payload["queries"] if isinstance(row, dict)]
    return []


def _is_valid_acronym(token: str) -> bool:
    token = str(token or "").strip()
    if len(token) < 2 or len(token) > 12:
        return False
    if token in BLACKLIST:
        return False
    if token.isdigit():
        return False
    alpha = sum(ch.isalpha() for ch in token)
    if alpha < 2:
        return False
    if token.replace("-", "").isdigit():
        return False
    return True


def _normalize_longform(text: str) -> str:
    words = [w.strip("-&'/,") for w in WORD_RE.findall(str(text or ""))]
    return " ".join(words).strip()


def _clean_display_text(text: str) -> str:
    value = str(text or "")
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        value = value.replace(bad, good)
    return value


def _initials(longform: str) -> str:
    parts = []
    for raw in _normalize_longform(longform).split():
        token = raw.lower()
        if token in CONNECTOR_WORDS:
            continue
        if token:
            parts.append(token[0].upper())
    return "".join(parts)


def _plausible_pair(acronym: str, longform: str) -> bool:
    acronym_letters = "".join(ch for ch in acronym.upper() if ch.isalpha())
    initials = _initials(longform)
    if not acronym_letters or not initials:
        return False
    return initials.startswith(acronym_letters) or acronym_letters.startswith(initials)


def _truncate_context(text: str, acronym: str, window: int = 90) -> str:
    raw = " ".join(_clean_display_text(text).split())
    idx = raw.find(acronym)
    if idx < 0:
        return raw[: 2 * window].strip()
    start = max(0, idx - window)
    end = min(len(raw), idx + len(acronym) + window)
    snippet = raw[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(raw):
        snippet = snippet + "..."
    return snippet


def _load_doc_texts(doc_dir: Path) -> list[str]:
    chunks_path = doc_dir / "chunks.parquet"
    if chunks_path.exists():
        df = pd.read_parquet(chunks_path, columns=["chunk_text"])
        if "chunk_text" in df.columns:
            return [str(x).strip() for x in df["chunk_text"].tolist() if str(x).strip()]
    sections_path = doc_dir / "sections.csv"
    if sections_path.exists():
        df = pd.read_csv(sections_path, usecols=["section_text"])
        if "section_text" in df.columns:
            return [str(x).strip() for x in df["section_text"].tolist() if str(x).strip()]
    return []


def _load_questions(doc_dir: Path) -> list[str]:
    eval_path = doc_dir / "eval_set.json"
    if not eval_path.exists():
        return []
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    return [str(q.get("question") or "").strip() for q in _iter_queries(payload) if str(q.get("question") or "").strip()]


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    explicit_docs = [x.strip() for x in str(args.docs).split(",") if x.strip()]
    doc_dirs: list[Path]
    if explicit_docs:
        doc_dirs = []
        missing: list[str] = []
        for doc_id in explicit_docs:
            doc_dir = data_root / doc_id
            if doc_dir.is_dir():
                doc_dirs.append(doc_dir)
            else:
                missing.append(doc_id)
        if missing:
            raise FileNotFoundError(f"Missing requested document directories under {data_root}: {missing}")
        doc_dirs = sorted(doc_dirs)
    else:
        doc_dirs = sorted([p for p in data_root.glob(args.doc_glob) if p.is_dir()])
    if args.require_eval_set:
        doc_dirs = [p for p in doc_dirs if (p / "eval_set.json").exists()]
    if not doc_dirs:
        raise FileNotFoundError(f"No document directories found under {data_root} matching {args.doc_glob}")

    corpus_count: Counter[str] = Counter()
    corpus_docs: dict[str, set[str]] = defaultdict(set)
    pattern_count: Counter[str] = Counter()
    longform_counter: dict[str, Counter[str]] = defaultdict(Counter)
    example_contexts: dict[str, list[str]] = defaultdict(list)
    example_doc_ids: dict[str, list[str]] = defaultdict(list)
    question_count: Counter[str] = Counter()
    question_docs: dict[str, set[str]] = defaultdict(set)
    question_examples: dict[str, list[str]] = defaultdict(list)
    question_longform_matches: Counter[str] = Counter()

    total_chunks = 0
    total_questions = 0

    for doc_dir in doc_dirs:
        doc_id = doc_dir.name
        texts = _load_doc_texts(doc_dir)
        total_chunks += len(texts)
        for text in texts:
            seen_in_doc: set[str] = set()
            for acronym in ACRONYM_RE.findall(text):
                if not _is_valid_acronym(acronym):
                    continue
                corpus_count[acronym] += 1
                seen_in_doc.add(acronym)
                if len(example_contexts[acronym]) < 3:
                    example_contexts[acronym].append(_truncate_context(text, acronym))
            for acronym in seen_in_doc:
                corpus_docs[acronym].add(doc_id)
                if len(example_doc_ids[acronym]) < 5 and doc_id not in example_doc_ids[acronym]:
                    example_doc_ids[acronym].append(doc_id)
            for match in LONGFORM_BEFORE_RE.finditer(text):
                longform, acronym = match.group(1), match.group(2)
                longform = _normalize_longform(longform)
                if _is_valid_acronym(acronym) and longform and _plausible_pair(acronym, longform):
                    pattern_count[acronym] += 1
                    longform_counter[acronym][longform] += 1
            for match in LONGFORM_AFTER_RE.finditer(text):
                acronym, longform = match.group(1), match.group(2)
                longform = _normalize_longform(longform)
                if _is_valid_acronym(acronym) and longform and _plausible_pair(acronym, longform):
                    pattern_count[acronym] += 1
                    longform_counter[acronym][longform] += 1

        questions = _load_questions(doc_dir)
        total_questions += len(questions)
        for question in questions:
            seen_q: set[str] = set()
            for acronym in ACRONYM_RE.findall(question):
                if not _is_valid_acronym(acronym):
                    continue
                question_count[acronym] += 1
                seen_q.add(acronym)
                if len(question_examples[acronym]) < 3:
                    question_examples[acronym].append(_clean_display_text(question))
            for acronym in seen_q:
                question_docs[acronym].add(doc_id)

    all_acronyms = sorted(set(corpus_count) | set(question_count))
    rows: list[dict[str, Any]] = []
    for acronym in all_acronyms:
        c_count = int(corpus_count.get(acronym, 0))
        if c_count < int(args.min_corpus_count) and int(question_count.get(acronym, 0)) == 0:
            continue
        doc_ids_sorted = sorted(corpus_docs.get(acronym, set()))
        longforms = longform_counter.get(acronym, Counter())
        best_longform = longforms.most_common(1)[0][0] if longforms else ""
        q_longform_hits = 0
        if longforms:
            candidate_longforms = [lf.lower() for lf, _ in longforms.most_common(5)]
            for q in question_examples.get(acronym, []):
                ql = q.lower()
                if any(lf in ql for lf in candidate_longforms):
                    q_longform_hits += 1
        score = (
            3.0 * int(question_count.get(acronym, 0))
            + 2.0 * int(len(question_docs.get(acronym, set())))
            + 1.5 * int(pattern_count.get(acronym, 0))
            + 1.0 * int(len(corpus_docs.get(acronym, set())))
            + math.log1p(c_count)
        )
        if best_longform:
            score += 2.0
        if q_longform_hits:
            score += 2.0 * q_longform_hits
        if score < float(args.min_score):
            continue
        rows.append(
            {
                "acronym": acronym,
                "high_value_score": round(score, 3),
                "corpus_count": c_count,
                "doc_count": int(len(doc_ids_sorted)),
                "first_seen_doc": (doc_ids_sorted[0] if doc_ids_sorted else ""),
                "last_seen_doc": (doc_ids_sorted[-1] if doc_ids_sorted else ""),
                "question_count": int(question_count.get(acronym, 0)),
                "question_doc_count": int(len(question_docs.get(acronym, set()))),
                "pattern_count": int(pattern_count.get(acronym, 0)),
                "longform_count": int(sum(longforms.values())),
                "best_longform": best_longform,
                "longforms_json": json.dumps(longforms.most_common(8), ensure_ascii=False),
                "question_matches_longform_count": q_longform_hits,
                "example_doc_ids": ",".join(example_doc_ids.get(acronym, [])[:5]),
                "example_questions_json": json.dumps(question_examples.get(acronym, [])[:3], ensure_ascii=False),
                "example_contexts_json": json.dumps(example_contexts.get(acronym, [])[:3], ensure_ascii=False),
                "status": "",
                "notes": "",
            }
        )

    rows.sort(
        key=lambda r: (
            -float(r["high_value_score"]),
            -int(r["question_count"]),
            -int(r["pattern_count"]),
            str(r["acronym"]),
        )
    )
    if int(args.top_n) > 0:
        rows = rows[: int(args.top_n)]

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "data_root": str(data_root),
        "doc_glob": str(args.doc_glob),
        "explicit_docs": explicit_docs,
        "require_eval_set": bool(args.require_eval_set),
        "documents_scanned": [p.name for p in doc_dirs],
        "document_count": len(doc_dirs),
        "total_chunk_texts_scanned": total_chunks,
        "total_questions_scanned": total_questions,
        "candidates_written": len(rows),
        "output_csv": str(output_csv),
        "top_10": rows[:10],
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
