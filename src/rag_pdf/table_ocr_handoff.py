"""Build chunk records for table pages that went through the OCR fallback path."""

from __future__ import annotations

from typing import Optional

from rag_pdf.chunking import count_tokens
from rag_pdf.schemas import build_page_list_struct, make_chunk_id_global


def build_ocr_table_chunk_record(
    *,
    doc_id: str,
    corpus_id: str,
    report_year,
    report_year_source: Optional[str],
    period_end_date,
    run_date_utc: str,
    page_no: int,
    ocr_chunk_text: str,
    ocr_table_type: Optional[str],
    debug: dict,
    enc,
) -> dict:
    """Assemble a complete chunk record dict for a table page that was processed via the OCR fallback path.

    The record follows the same schema as pdfplumber/camelot-extracted table chunks so that
    downstream indexing and retrieval treat OCR-sourced table chunks identically to
    parser-extracted ones.  Debug statistics (digit_ratio, currency_hits, etc.) are stored
    as dedicated fields for post-hoc quality analysis.
    """
    chunk_id_local = f"table_ocr_p{page_no:04d}"
    pages = [page_no]
    return {
        "doc_id": doc_id,
        "corpus_id": corpus_id,
        "report_year": report_year,
        "report_year_source": report_year_source,
        "period_end_date": period_end_date,
        "run_date_utc": run_date_utc,
        "chunk_id": chunk_id_local,
        "chunk_id_global": make_chunk_id_global(doc_id, chunk_id_local),
        "part": "Unknown",
        "section_title": "Financial Tables",
        "subsection_title": None,
        "page_start": page_no,
        "page_end": page_no,
        "pages": pages,
        "page_list": build_page_list_struct(pages),
        "chunk_text": ocr_chunk_text,
        "chunk_tokens": count_tokens(ocr_chunk_text, enc),
        "word_count": len(ocr_chunk_text.split()),
        "is_table_like": True,
        "many_numbers": True,
        "is_table": True,
        "table_type": ocr_table_type,
        "table_ref": None,
        "table_source": "ocr_fallback",
        "parsing_report_accuracy": None,
        "parsing_report_whitespace": None,
        "ocr_fallback_digit_ratio": float(debug.get("digit_ratio", 0.0)),
        "ocr_fallback_currency_hits": int(debug.get("currency_hits", 0)),
        "ocr_fallback_num_lines_with_2plus_nums": int(debug.get("num_lines_with_2plus_nums", 0)),
        "ocr_fallback_corrupted_flag": bool(debug.get("corrupted_flag", False)),
        "ocr_fallback_matched_keywords": "|".join(debug.get("matched_keywords", [])),
    }


def build_rejected_ocr_table_page(*, page_no: int, page_text: str) -> dict:
    """Return a minimal record for a table page whose OCR output was rejected as unusable.

    Rejected pages are collected separately so they can be inspected during QA without
    polluting the accepted chunk list.
    """
    return {
        "page": page_no,
        "text": page_text,
    }
