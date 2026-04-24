from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


DOC_IDS = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit canonical eval_set.json files for likely annotation issues.")
    parser.add_argument(
        "--data-root",
        default="data_processed",
        help="Root containing canonical per-document data directories.",
    )
    parser.add_argument(
        "--doc-ids",
        nargs="*",
        default=DOC_IDS,
        help="Document IDs to audit.",
    )
    parser.add_argument(
        "--out-csv",
        default="results/eval_audit/query_audit.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-md",
        default="results/eval_audit/query_audit_summary.md",
        help="Output markdown summary path.",
    )
    return parser.parse_args()


def load_eval_items(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return list(raw.get("queries", []))
    if isinstance(raw, list):
        return list(raw)
    raise ValueError(f"Unsupported eval_set structure: {path}")


def normalize_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip().lower()


def normalize_answer_for_fuzzy_find(text: str) -> str:
    text = normalize_text(text)
    text = text.replace("£", "")
    text = text.replace(",", "")
    text = text.replace("(", " ").replace(")", " ")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def pages_lookup(pages_df: pd.DataFrame) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for row in pages_df.itertuples(index=False):
        page_num = int(getattr(row, "page"))
        lookup[page_num] = str(getattr(row, "clean_text", "") or "")
    return lookup


def chunk_page_map(chunks_df: pd.DataFrame) -> dict[int, list[dict[str, str]]]:
    out: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in chunks_df.itertuples(index=False):
        pages = getattr(row, "pages", None)
        if pages is None:
            start = getattr(row, "page_start", None)
            end = getattr(row, "page_end", None)
            if start is None:
                page_list: list[int] = []
            elif end is None:
                page_list = [int(start)]
            else:
                page_list = list(range(int(start), int(end) + 1))
        else:
            page_list = [int(x) for x in list(pages)]
        item = {
            "section_title": str(getattr(row, "section_title", "") or ""),
            "subsection_title": str(getattr(row, "subsection_title", "") or ""),
            "chunk_text": str(getattr(row, "chunk_text", "") or ""),
        }
        for page in page_list:
            out[page].append(item)
    return out


def contains_answer(expected_answer: str, haystacks: list[str]) -> tuple[bool, bool]:
    target_exact = normalize_text(expected_answer)
    target_fuzzy = normalize_answer_for_fuzzy_find(expected_answer)
    exact = False
    fuzzy = False
    for hay in haystacks:
        hay_exact = normalize_text(hay)
        hay_fuzzy = normalize_answer_for_fuzzy_find(hay)
        if target_exact and target_exact in hay_exact:
            exact = True
        if target_fuzzy and target_fuzzy in hay_fuzzy:
            fuzzy = True
        if exact and fuzzy:
            return exact, fuzzy
    return exact, fuzzy


def normalize_numeric_token(token: str) -> str:
    s = str(token or "").strip().lower()
    if not s:
        return ""
    is_percent = s.endswith("%")
    s = s.replace("%", "")
    s = s.replace("£", "")
    s = s.replace(",", "")
    s = s.strip()
    if not s:
        return ""
    try:
        val = float(s)
    except Exception:
        return ""
    if val.is_integer():
        out = str(int(val))
    else:
        out = f"{val:.6f}".rstrip("0").rstrip(".")
    return out + ("%" if is_percent else "")


def extract_numeric_tokens(text: str) -> list[str]:
    raw = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?%?", str(text or ""))
    out: list[str] = []
    for tok in raw:
        norm = normalize_numeric_token(tok)
        if norm:
            out.append(norm)
    return out


def numeric_support(expected_answer: str, haystacks: list[str]) -> tuple[bool, bool]:
    expected_nums = set(extract_numeric_tokens(expected_answer))
    if not expected_nums:
        return False, False
    found: set[str] = set()
    for hay in haystacks:
        found.update(extract_numeric_tokens(hay))
    if not found:
        return False, False
    all_found = expected_nums.issubset(found)
    any_found = bool(expected_nums.intersection(found))
    return all_found, any_found


STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "than", "that", "this", "were", "was",
    "over", "under", "more", "less", "than", "what", "which", "when", "where", "year",
    "march", "april", "june", "july", "august", "september", "october", "november",
    "december", "january", "february", "million", "thousand", "increase", "decrease",
    "compared", "during", "related", "activities", "costs", "cost", "amount", "total",
    "support", "funding", "board", "nhs", "grampian", "scottish", "government",
}


def extract_content_terms(text: str) -> set[str]:
    terms = re.findall(r"[a-z][a-z\-]{2,}", normalize_text(text))
    out = {
        t for t in terms
        if t not in STOPWORDS and not t.isdigit() and len(t) >= 4
    }
    return out


def keyword_support(expected_answer: str, haystacks: list[str]) -> tuple[bool, float, list[str]]:
    expected_terms = extract_content_terms(expected_answer)
    if not expected_terms:
        return False, 0.0, []
    hay_terms: set[str] = set()
    for hay in haystacks:
        hay_terms.update(extract_content_terms(hay))
    matched = sorted(expected_terms.intersection(hay_terms))
    coverage = float(len(matched)) / float(len(expected_terms)) if expected_terms else 0.0
    supported = len(matched) >= 2 and coverage >= 0.6
    return supported, coverage, matched


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    all_items: list[dict[str, Any]] = []
    per_doc_context: dict[str, dict[str, Any]] = {}
    qid_counts: Counter[str] = Counter()

    for doc_id in args.doc_ids:
        doc_dir = data_root / doc_id
        eval_path = doc_dir / "eval_set.json"
        pages_path = doc_dir / "pages.parquet"
        chunks_path = doc_dir / "chunks.parquet"
        if not (eval_path.exists() and pages_path.exists() and chunks_path.exists()):
            raise FileNotFoundError(f"Missing required files for {doc_id} in {doc_dir}")

        items = load_eval_items(eval_path)
        pages_df = pd.read_parquet(pages_path)
        chunks_df = pd.read_parquet(chunks_path)
        page_text = pages_lookup(pages_df)
        chunk_map = chunk_page_map(chunks_df)
        max_page = max(page_text) if page_text else 0
        per_doc_context[doc_id] = {
            "items": items,
            "page_text": page_text,
            "chunk_map": chunk_map,
            "max_page": max_page,
        }
        all_items.extend(items)
        for item in items:
            qid_counts[str(item.get("query_id", "")).strip()] += 1

    rows: list[dict[str, Any]] = []
    for doc_id in args.doc_ids:
        ctx = per_doc_context[doc_id]
        max_page = int(ctx["max_page"])
        page_text = ctx["page_text"]
        chunk_map = ctx["chunk_map"]
        for item in ctx["items"]:
            query_id = str(item.get("query_id", "")).strip()
            question = str(item.get("question", "")).strip()
            expected_pages_raw = item.get("expected_pages", [])
            expected_pages = [int(x) for x in expected_pages_raw if str(x).isdigit()] if isinstance(expected_pages_raw, list) else []
            expected_answer = str(item.get("expected_answer", "") or "").strip()
            expected_section = str(item.get("expected_section", "") or "").strip()
            expected_subsection = str(item.get("expected_subsection", "") or "").strip()
            answer_type = str(item.get("answer_type", "") or "").strip()
            evidence_layout = str(item.get("evidence_layout", "") or "").strip()

            invalid_pages = [p for p in expected_pages if p < 1 or p > max_page]
            missing_expected_pages = len(expected_pages) == 0
            page_texts = [page_text.get(p, "") for p in expected_pages]
            page_text_present = all(bool(t.strip()) for t in page_texts) if expected_pages else False
            chunks_for_pages = [chunk for p in expected_pages for chunk in chunk_map.get(p, [])]
            chunk_texts = [c["chunk_text"] for c in chunks_for_pages if c["chunk_text"].strip()]
            hay_expected_pages = page_texts + chunk_texts
            exact_on_expected_pages, fuzzy_on_expected_pages = contains_answer(expected_answer, hay_expected_pages)
            numeric_all_on_expected_pages, numeric_any_on_expected_pages = numeric_support(expected_answer, hay_expected_pages)
            keyword_on_expected_pages, keyword_cov_expected_pages, keyword_terms_expected_pages = keyword_support(
                expected_answer, hay_expected_pages
            )
            answer_supported_on_expected_pages = bool(
                fuzzy_on_expected_pages
                or numeric_all_on_expected_pages
                or (answer_type != "number" and keyword_on_expected_pages)
            )

            all_doc_page_texts = list(page_text.values())
            all_doc_chunk_texts = [c["chunk_text"] for chunks in chunk_map.values() for c in chunks if c["chunk_text"].strip()]
            hay_anywhere = all_doc_page_texts + all_doc_chunk_texts
            exact_anywhere, fuzzy_anywhere = contains_answer(expected_answer, hay_anywhere)
            numeric_all_anywhere, numeric_any_anywhere = numeric_support(expected_answer, hay_anywhere)
            keyword_anywhere, keyword_cov_anywhere, keyword_terms_anywhere = keyword_support(
                expected_answer, hay_anywhere
            )
            answer_supported_anywhere = bool(
                fuzzy_anywhere
                or numeric_all_anywhere
                or (answer_type != "number" and keyword_anywhere)
            )

            section_match = False
            subsection_match = False
            if expected_pages:
                section_match = any(normalize_text(expected_section) == normalize_text(c["section_title"]) for c in chunks_for_pages if expected_section)
                subsection_match = any(
                    normalize_text(expected_subsection) == normalize_text(c["subsection_title"])
                    for c in chunks_for_pages
                    if expected_subsection
                )

            flags: list[str] = []
            if qid_counts[query_id] > 1:
                flags.append("duplicate_query_id")
            if not question:
                flags.append("empty_question")
            if missing_expected_pages:
                flags.append("missing_expected_pages")
            if invalid_pages:
                flags.append("expected_page_out_of_range")
            if not expected_answer:
                flags.append("missing_expected_answer")
            if not answer_type:
                flags.append("missing_answer_type")
            if expected_pages and not page_text_present:
                flags.append("expected_page_text_missing")
            if expected_answer and expected_pages and not answer_supported_on_expected_pages:
                flags.append("answer_not_found_on_expected_pages")
            if expected_answer and not answer_supported_anywhere:
                flags.append("answer_not_found_anywhere_in_doc")
            if expected_section and expected_pages and not section_match:
                flags.append("expected_section_not_matched")
            if expected_subsection and expected_pages and not subsection_match:
                flags.append("expected_subsection_not_matched")
            if evidence_layout == "table" and expected_answer and expected_pages and not exact_on_expected_pages and fuzzy_on_expected_pages:
                flags.append("table_answer_format_mismatch")

            rows.append(
                {
                    "doc_id": doc_id,
                    "query_id": query_id,
                    "difficulty": str(item.get("difficulty", "") or ""),
                    "answer_type": answer_type,
                    "evidence_layout": evidence_layout,
                    "expected_pages": json.dumps(expected_pages),
                    "invalid_pages": json.dumps(invalid_pages),
                    "expected_section": expected_section,
                    "expected_subsection": expected_subsection,
                    "question": question,
                    "expected_answer": expected_answer,
                    "page_text_present": int(page_text_present),
                    "answer_exact_on_expected_pages": int(exact_on_expected_pages),
                    "answer_fuzzy_on_expected_pages": int(fuzzy_on_expected_pages),
                    "answer_numeric_all_on_expected_pages": int(numeric_all_on_expected_pages),
                    "answer_numeric_any_on_expected_pages": int(numeric_any_on_expected_pages),
                    "answer_keyword_supported_on_expected_pages": int(keyword_on_expected_pages),
                    "answer_keyword_coverage_on_expected_pages": round(float(keyword_cov_expected_pages), 4),
                    "answer_keyword_terms_on_expected_pages": "|".join(keyword_terms_expected_pages),
                    "answer_supported_on_expected_pages": int(answer_supported_on_expected_pages),
                    "answer_exact_anywhere_in_doc": int(exact_anywhere),
                    "answer_fuzzy_anywhere_in_doc": int(fuzzy_anywhere),
                    "answer_numeric_all_anywhere_in_doc": int(numeric_all_anywhere),
                    "answer_numeric_any_anywhere_in_doc": int(numeric_any_anywhere),
                    "answer_keyword_supported_anywhere_in_doc": int(keyword_anywhere),
                    "answer_keyword_coverage_anywhere_in_doc": round(float(keyword_cov_anywhere), 4),
                    "answer_keyword_terms_anywhere_in_doc": "|".join(keyword_terms_anywhere),
                    "answer_supported_anywhere_in_doc": int(answer_supported_anywhere),
                    "expected_section_match": int(section_match),
                    "expected_subsection_match": int(subsection_match),
                    "flag_count": len(flags),
                    "flags": "|".join(flags),
                    "review_priority": (
                        "high"
                        if any(
                            f in flags
                            for f in [
                                "missing_expected_pages",
                                "expected_page_out_of_range",
                                "answer_not_found_on_expected_pages",
                                "answer_not_found_anywhere_in_doc",
                            ]
                        )
                        else "medium"
                        if flags
                        else "low"
                    ),
                }
            )

    rows.sort(key=lambda r: (-int(r["flag_count"]), str(r["doc_id"]), str(r["query_id"])))
    fieldnames = list(rows[0].keys()) if rows else []
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    flag_counter: Counter[str] = Counter()
    for row in rows:
        for flag in str(row["flags"]).split("|"):
            if flag:
                flag_counter[flag] += 1
    high_priority = [row for row in rows if row["review_priority"] == "high"]
    medium_priority = [row for row in rows if row["review_priority"] == "medium"]

    lines = [
        "# Eval Set Audit Summary",
        "",
        f"- Documents audited: {len(args.doc_ids)}",
        f"- Queries audited: {len(rows)}",
        f"- High-priority reviews: {len(high_priority)}",
        f"- Medium-priority reviews: {len(medium_priority)}",
        "",
        "## Flag Counts",
        "",
    ]
    if flag_counter:
        for flag, count in sorted(flag_counter.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- `{flag}`: {count}")
    else:
        lines.append("- No flags raised.")

    lines.extend(["", "## Highest-Priority Queries", ""])
    if high_priority:
        for row in high_priority[:25]:
            lines.append(
                f"- `{row['query_id']}` ({row['doc_id']}): {row['flags']}"
            )
    else:
        lines.append("- None.")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
