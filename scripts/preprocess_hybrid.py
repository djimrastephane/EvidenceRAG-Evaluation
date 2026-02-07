from __future__ import annotations

# This script orchestrates the hybrid preprocessing pipeline.
# Core logic lives in src/rag_pdf/ modules for clarity and testability.

import sys
from collections import defaultdict
from pathlib import Path

try:
    import fitz  # PyMuPDF
except Exception as e:
    raise RuntimeError(
        "Failed to import PyMuPDF.\n"
        "Fix: pip uninstall -y fitz frontend && pip install -U pymupdf\n"
    ) from e

import pdfplumber
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from rag_pdf.boilerplate import remove_repeated_header_footer_lines, strip_by_coordinates
from rag_pdf.chunking import chunk_text_by_tokens, count_tokens, get_encoder
from rag_pdf.config import PreprocessConfig
from rag_pdf.extract_page import OCR_AVAILABLE, extract_page_struct_hybrid, extract_page_with_ocr
from rag_pdf.headings import select_heading_candidates
from rag_pdf.metrics import StepTimer, safe_json_dump
from rag_pdf.schemas import build_page_list_struct, make_chunk_id_global
from rag_pdf.sections import build_sections_from_pages, find_section_for_page
from rag_pdf.table_detect import classify_page_content, contains_many_numbers, detect_table_type, is_table_like_from_raw_lines
from rag_pdf.table_extract import process_table_pages
from rag_pdf.text_normalize import (
    extract_report_metadata_from_pdf,
    extract_report_year_from_filename,
    normalize_page_text,
    now_utc_iso,
)


def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(c.isalpha() for c in text)
    return alpha / max(len(text), 1)


def _digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    digits = sum(c.isdigit() for c in text)
    return digits / max(len(text), 1)


def _apply_config_overrides(cfg: PreprocessConfig) -> None:
    import rag_pdf.boilerplate as boilerplate_mod
    import rag_pdf.extract_page as extract_page_mod
    import rag_pdf.headings as headings_mod
    import rag_pdf.table_detect as table_detect_mod
    import rag_pdf.table_extract as table_extract_mod

    boilerplate_mod.TOP_STRIP_FRAC = cfg.TOP_STRIP_FRAC
    boilerplate_mod.BOTTOM_STRIP_FRAC = cfg.BOTTOM_STRIP_FRAC
    boilerplate_mod.LEFT_STRIP_FRAC = cfg.LEFT_STRIP_FRAC
    boilerplate_mod.RIGHT_STRIP_FRAC = cfg.RIGHT_STRIP_FRAC
    boilerplate_mod.HEADER_FOOTER_REPEAT_FRAC = cfg.HEADER_FOOTER_REPEAT_FRAC
    boilerplate_mod.TOP_LINE_K = cfg.TOP_LINE_K
    boilerplate_mod.BOT_LINE_K = cfg.BOT_LINE_K

    headings_mod.HEADING_MAX_CHARS = cfg.HEADING_MAX_CHARS
    headings_mod.HEADING_MIN_CHARS = cfg.HEADING_MIN_CHARS
    headings_mod.HEADING_FONT_BOOST_FRAC = cfg.HEADING_FONT_BOOST_FRAC

    extract_page_mod.PRIMARY_EXTRACTOR = cfg.PRIMARY_EXTRACTOR
    extract_page_mod.FALLBACK_MIN_CHARS = cfg.FALLBACK_MIN_CHARS
    extract_page_mod.FALLBACK_ON_BAD_TEXT = cfg.FALLBACK_ON_BAD_TEXT
    extract_page_mod.FALLBACK_ON_EXCEPTION = cfg.FALLBACK_ON_EXCEPTION

    table_detect_mod.TABLE_DIGIT_RATIO = cfg.TABLE_DIGIT_RATIO
    table_detect_mod.TABLE_SPACE_RATIO = cfg.TABLE_SPACE_RATIO
    table_detect_mod.TABLE_MIN_LINES = cfg.TABLE_MIN_LINES

    table_extract_mod.CAMELOT_LATTICE_ACCURACY_THRESHOLD = cfg.CAMELOT_LATTICE_ACCURACY_THRESHOLD
    table_extract_mod.TABLE_SUMMARY_MAX_ROWS = cfg.TABLE_SUMMARY_MAX_ROWS


def main() -> None:
    """
    Execute hybrid text + table preprocessing pipeline.

    Pipeline stages:
    0. Extract metadata from cover pages
    1. Page extraction with hybrid loader (PyMuPDF + pdfplumber fallback)
    2. Coordinate-based boilerplate stripping (orientation-aware)
    3. Repetition-based header/footer removal
    4. Page classification (text vs. table)
    5. Section inference from headings
    6. Fork processing:
       - Text pages → standard chunking
       - Table pages → dual representation (summary + structured)
    7. Write outputs (parquet + CSV)
    """
    cfg = PreprocessConfig(
        PDF_PATH=Path(
            "/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/Data/Annual Accounts NHS Grampian/Preliminary_Test/Grampian-2022-2023.pdf"
        ),
        OUT_ROOT=Path(
            "/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/data_processed"
        ),
        CORPUS_ID=None,
        CHUNK_SIZE_TOKENS=320,
        CHUNK_OVERLAP_TOKENS=90,
        TOP_STRIP_FRAC=0.08,
        BOTTOM_STRIP_FRAC=0.08,
        LEFT_STRIP_FRAC=0.08,
        RIGHT_STRIP_FRAC=0.08,
        HEADER_FOOTER_REPEAT_FRAC=0.40,
        TOP_LINE_K=5,
        BOT_LINE_K=5,
        HEADING_MAX_CHARS=110,
        HEADING_MIN_CHARS=4,
        HEADING_FONT_BOOST_FRAC=0.85,
        MIN_CHUNK_WORDS=20,
        PRIMARY_EXTRACTOR="pymupdf",
        FALLBACK_MIN_CHARS=80,
        FALLBACK_ON_BAD_TEXT=True,
        FALLBACK_ON_EXCEPTION=True,
        TABLE_DIGIT_RATIO=0.15,
        TABLE_SPACE_RATIO=0.3,
        TABLE_MIN_LINES=1,
        CAMELOT_LATTICE_ACCURACY_THRESHOLD=70,
        TABLE_SUMMARY_MAX_ROWS=5,
        OCR_MIN_ALPHA_RATIO=0.3,
        OCR_MIN_DIGIT_RATIO=0.6,
    )

    _apply_config_overrides(cfg)

    timer = StepTimer()

    if not cfg.PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {cfg.PDF_PATH}")

    doc_id = cfg.PDF_PATH.stem
    run_date_utc = now_utc_iso()
    enc = get_encoder()
    corpus_id = cfg.CORPUS_ID or doc_id

    print(f"\n{'=' * 60}")
    print(f"Processing: {doc_id}")
    print(f"{'=' * 60}\n")

    doc = fitz.open(cfg.PDF_PATH)
    timer.mark("Open PDF (PyMuPDF)")

    with pdfplumber.open(str(cfg.PDF_PATH)) as pdf_plumber:
        timer.mark("Open PDF (PDFPlumber)")

        # Extract cover metadata
        pdf_meta = extract_report_metadata_from_pdf(doc, max_pages=2)
        report_year_from_pdf = pdf_meta.get("report_year_from_pdf")
        report_year_from_filename = extract_report_year_from_filename(doc_id)

        report_year = report_year_from_pdf or report_year_from_filename
        report_year_source = "pdf_cover" if report_year_from_pdf else "filename"
        period_end_date = pdf_meta.get("period_end_date")

        print(f"Report Year: {report_year} (source: {report_year_source})")
        print(f"Period End: {period_end_date or 'Not detected'}\n")

        timer.mark("Step 0: cover metadata extraction")

        # Extract all pages
        pages_text_lines = {}
        page_heading_candidates = {}
        page_extractor_used = {}
        page_extractor_notes = {}

        qa_removed_top = defaultdict(list)
        qa_removed_bottom = defaultdict(list)

        print("Extracting pages...")
        page_structs = {}  # ← ADD THIS LINE AT THE TOP

        for i in range(doc.page_count):
            if (i + 1) % 20 == 0:
                print(f"  Page {i + 1}/{doc.page_count}")

            page_no = i + 1

            s, used, note = extract_page_struct_hybrid(
                doc,
                pdf_plumber,
                i,
                pdf_path=str(cfg.PDF_PATH),
            )
            page_structs[page_no] = s  # ← ADD THIS LINE
            page_extractor_used[page_no] = used
            page_extractor_notes[page_no] = note

            # Check if raw lines look like a table (before cleanup)
            raw_lines = [ln["text"] for ln in s.get("lines_all", [])]
            is_raw_table = is_table_like_from_raw_lines(raw_lines)

            kept, rem_a, rem_b = strip_by_coordinates(
                s["lines_all"],
                page_height=s["page_height"],
                page_width=s["page_width"],
                rotation=s["rotation"],
            )

            pages_text_lines[page_no] = kept
            page_heading_candidates[page_no] = select_heading_candidates(
                s["lines_all"], s["p95_font"]
            )

            # Store raw table flag for later use
            pages_text_lines[page_no] = (kept, is_raw_table)

            qa_removed_top[page_no] = rem_a
            qa_removed_bottom[page_no] = rem_b

        timer.mark("Step 1: page extraction + coord strip")

        # Remove repeated headers/footers
        pages_text_only = {pno: lines if isinstance(lines, list) else lines[0]
                           for pno, lines in pages_text_lines.items()}
        pages_text_lines2, common_header, common_footer = remove_repeated_header_footer_lines(
            pages_text_only
        )
        timer.mark("Step 2: repeated header/footer strip")

        # Build pages dataframe with classification
        print("\nClassifying pages...")
        pages_records = []
        text_pages = []
        table_pages = []
        short_clean_pages = 0
        ocr_used_pages = 0
        ocr_attempts = 0
        ocr_too_short = 0
        ocr_debug_logged = 0
        ocr_force_table = {}

        for i in range(doc.page_count):
            page_no = i + 1
            raw = "\n".join(pages_text_lines2.get(page_no, [])).strip()
            clean_text = normalize_page_text(raw)

            ocr_clean_len = None
            ocr_text_len = None
            if OCR_AVAILABLE and len(clean_text) < 50:
                short_clean_pages += 1
                ocr_attempts += 1
                ocr_text = extract_page_with_ocr(str(cfg.PDF_PATH), page_no - 1)
                ocr_clean = normalize_page_text(ocr_text)
                ocr_text_len = len(ocr_text)
                ocr_clean_len = len(ocr_clean)
                ocr_alpha = _alpha_ratio(ocr_clean)
                ocr_digits = _digit_ratio(ocr_clean)
                accept_ocr = len(ocr_clean) >= 50 and (
                    ocr_alpha >= cfg.OCR_MIN_ALPHA_RATIO or ocr_digits > cfg.OCR_MIN_DIGIT_RATIO
                )
                if accept_ocr:
                    clean_text = ocr_clean
                    page_extractor_used[page_no] = "ocr"
                    note = "clean_text_short_used_ocr"
                    if ocr_digits > cfg.OCR_MIN_DIGIT_RATIO:
                        ocr_force_table[page_no] = True
                        note = f"{note};table_like"
                    page_extractor_notes[page_no] = note
                    ocr_used_pages += 1
                    print(f"[OCR] page {page_no} used (clean_text_short)")
                else:
                    ocr_too_short += 1
                    if ocr_debug_logged < 3:
                        print(
                            f"[OCR] page {page_no} too short: "
                            f"ocr_len={len(ocr_text)} ocr_clean_len={len(ocr_clean)}"
                        )
                        ocr_debug_logged += 1

            # Get raw table flag from earlier detection
            page_data = pages_text_lines.get(page_no)
            is_raw_table = False
            if isinstance(page_data, tuple):
                is_raw_table = page_data[1]

            # Classify page content (combine raw check + post-cleanup check)
            classification = classify_page_content(clean_text)

            # Override if raw structure indicated table
            if is_raw_table and not classification["is_table"]:
                classification["is_table"] = True
                classification["confidence"] = "medium"
                if not classification["table_type"]:
                    classification["table_type"] = detect_table_type(clean_text)

            if ocr_force_table.get(page_no) and not classification["is_table"]:
                classification["is_table"] = True
                classification["confidence"] = "medium"
                if not classification["table_type"]:
                    classification["table_type"] = detect_table_type(clean_text)

            s = page_structs.get(page_no, {})  # ← ADD THIS LINE

            pages_records.append({
                "doc_id": doc_id,
                "corpus_id": corpus_id,
                "report_year": report_year,
                "report_year_source": report_year_source,
                "period_end_date": period_end_date,
                "run_date_utc": run_date_utc,
                "page": page_no,
                "clean_text": clean_text,
                "heading_candidates": page_heading_candidates.get(page_no, []),
                "extractor": page_extractor_used.get(page_no, "unknown"),
                "extractor_notes": page_extractor_notes.get(page_no, ""),
                "ocr_text_len": ocr_text_len,
                "ocr_clean_text_len": ocr_clean_len,
                "is_table": classification["is_table"],
                "table_type": classification["table_type"],
                "classification_confidence": classification["confidence"],
                "rotation": s.get("rotation", 0),  # ← ADD THIS
                "page_width": s.get("page_width", 0.0),  # ← ADD THIS
                "page_height": s.get("page_height", 0.0),  # ← ADD THIS
            })

            # Split into text vs. table pages
            if classification["is_table"]:
                table_pages.append({
                    "page": page_no,
                    "text": clean_text,
                    "table_type": classification["table_type"],
                })
            else:
                text_pages.append({
                    "page": page_no,
                    "text": clean_text,
                })

        pages_df = pd.DataFrame(pages_records)

        print(f"  Text pages: {len(text_pages)}")
        print(f"  Table pages: {len(table_pages)}")
        print(f"  OCR short pages: {short_clean_pages}")
        print(f"  OCR used pages: {ocr_used_pages}")
        print(f"  OCR attempts: {ocr_attempts}")
        print(f"  OCR too short: {ocr_too_short}")

        timer.mark("Step 3: pages dataframe + classification")

        # Build sections (from all pages for context)
        sections_df = build_sections_from_pages(pages_df)
        timer.mark("Step 4: section inference")

        # Process TEXT pages → standard chunking
        print("\nChunking text pages...")
        text_chunks = []

        for tpage in text_pages:
            page_no = tpage["page"]
            text = tpage["text"]
            if not text:
                continue

            part, section = find_section_for_page(sections_df, page_no)
            page_chunks = chunk_text_by_tokens(
                text,
                cfg.CHUNK_SIZE_TOKENS,
                cfg.CHUNK_OVERLAP_TOKENS,
                enc,
            )

            for j, ctext in enumerate(page_chunks):
                wc = len(ctext.split())
                if wc < cfg.MIN_CHUNK_WORDS:
                    continue

                chunk_id_local = f"p{page_no:04d}_{j:03d}"
                pages = [page_no]
                page_list_struct = build_page_list_struct(pages)

                text_chunks.append({
                    "doc_id": doc_id,
                    "corpus_id": corpus_id,
                    "report_year": report_year,
                    "report_year_source": report_year_source,
                    "period_end_date": period_end_date,
                    "run_date_utc": run_date_utc,
                    "chunk_id": chunk_id_local,
                    "chunk_id_global": make_chunk_id_global(doc_id, chunk_id_local),
                    "part": part,
                    "section_title": section,
                    "page_start": page_no,
                    "page_end": page_no,
                    "pages": pages,
                    "page_list": page_list_struct,
                    "chunk_text": ctext,
                    "chunk_tokens": count_tokens(ctext, enc),
                    "word_count": wc,
                    "is_table_like": False,
                    "many_numbers": contains_many_numbers(ctext),
                    "is_table": False,
                    "table_type": None,
                    "table_ref": None,
                })

        text_chunks_df = pd.DataFrame(text_chunks)
        print(f"  Created {len(text_chunks_df)} text chunks")

        timer.mark("Step 5: text chunking")

        # Process TABLE pages → dual representation
        print("\nExtracting tables...")
        table_chunks_df, structured_tables_df = process_table_pages(
            table_pages,
            cfg.PDF_PATH,
            pdf_plumber,
            doc_id,
            corpus_id,
            report_year,
            period_end_date,
            report_year_source,
            run_date_utc,
            enc,
        )

        print(f"  Extracted {len(structured_tables_df)} tables")
        print(f"  Created {len(table_chunks_df)} table summary chunks")

        timer.mark("Step 6: table extraction + summarization")

        # Merge text and table chunks
        all_chunks_df = pd.concat([text_chunks_df, table_chunks_df], ignore_index=True)
        all_chunks_df = all_chunks_df.sort_values(["page_start", "chunk_id"]).reset_index(drop=True)

        print(
            f"\nTotal chunks: {len(all_chunks_df)} "
            f"({len(text_chunks_df)} text + {len(table_chunks_df)} table)"
        )

        # Validate page-bounded chunks
        if len(all_chunks_df) > 0:
            bad_span = all_chunks_df[all_chunks_df["page_start"] != all_chunks_df["page_end"]]
            if len(bad_span) > 0:
                raise ValueError(
                    f"Found {len(bad_span)} chunks spanning multiple pages. "
                    "Pipeline requires page-bounded chunks for accurate citations."
                )

        timer.mark("Step 7: chunk merging + validation")

        # Write outputs
        out_dir = cfg.OUT_ROOT / doc_id
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nWriting outputs to: {out_dir}")

        pages_df.to_parquet(out_dir / "pages.parquet", index=False)
        sections_df.to_parquet(out_dir / "sections.parquet", index=False)
        all_chunks_df.to_parquet(out_dir / "chunks.parquet", index=False)
        ocr_pages_df = pages_df.loc[
            pages_df["extractor"] == "ocr",
            ["page", "extractor_notes", "ocr_text_len", "ocr_clean_text_len", "clean_text"],
        ].copy()
        ocr_pages_df = ocr_pages_df.rename(
            columns={"ocr_clean_text_len": "clean_text_len"}
        )
        ocr_pages_df["clean_text_len"] = ocr_pages_df["clean_text"].fillna("").str.len()
        ocr_pages_df = ocr_pages_df.drop(columns=["clean_text"])
        ocr_pages_df.to_csv(out_dir / "ocr_pages.csv", index=False)

        # Write structured tables
        if len(structured_tables_df) > 0:
            structured_tables_df.to_parquet(out_dir / "tables_structured.parquet", index=False)

        timer.mark("Step 8: parquet writes")

        # Generate metrics
        metrics = {
            "schema_version": "3.0_hybrid",
            "doc_id": doc_id,
            "corpus_id": corpus_id,
            "report_year": report_year,
            "period_end_date": period_end_date,
            "counts": {
                "pages_total": len(pages_df),
                "pages_text": len(text_pages),
                "pages_table": len(table_pages),
                "sections": len(sections_df),
                "chunks_total": len(all_chunks_df),
                "chunks_text": len(text_chunks_df),
                "chunks_table": len(table_chunks_df),
                "tables_extracted": len(structured_tables_df),
            },
            "params": {
                "chunk_size_tokens": cfg.CHUNK_SIZE_TOKENS,
                "chunk_overlap_tokens": cfg.CHUNK_OVERLAP_TOKENS,
                "top_strip_frac": cfg.TOP_STRIP_FRAC,
                "bottom_strip_frac": cfg.BOTTOM_STRIP_FRAC,
                "left_strip_frac": cfg.LEFT_STRIP_FRAC,
                "right_strip_frac": cfg.RIGHT_STRIP_FRAC,
                "header_footer_repeat_frac": cfg.HEADER_FOOTER_REPEAT_FRAC,
                "min_chunk_words": cfg.MIN_CHUNK_WORDS,
                "primary_extractor": cfg.PRIMARY_EXTRACTOR,
            },
            "table_types_detected": (
                structured_tables_df["table_type"].value_counts().to_dict()
                if len(structured_tables_df) > 0
                else {}
            ),
        }

        safe_json_dump(metrics, out_dir / "metrics.json")

        print(f"\n{'=' * 60}")
        print("PROCESSING COMPLETE")
        print(f"{'=' * 60}")
        print(f"\nOutputs written to: {out_dir}")
        print(f"  - pages.parquet: {len(pages_df)} pages")
        print(f"  - sections.parquet: {len(sections_df)} sections")
        print(f"  - chunks.parquet: {len(all_chunks_df)} chunks (text + table summaries)")
        if len(structured_tables_df) > 0:
            print(f"  - tables_structured.parquet: {len(structured_tables_df)} tables")
        print("  - metrics.json: Pipeline statistics")

        timer.mark("Step 9: metrics + completion")

    doc.close()
    timer.mark("Close documents")
    timer.report()


if __name__ == "__main__":
    main()
