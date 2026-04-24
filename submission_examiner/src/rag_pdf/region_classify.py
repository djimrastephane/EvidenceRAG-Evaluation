"""Classify page regions as text or table using layout, drawing, and content signals."""

from __future__ import annotations

from rag_pdf.region_segment import PageRegion
from rag_pdf.table_detect import (
    classify_page_content,
    detect_table_type,
    is_column_alignment_table_like,
    is_graphics_table_like,
    is_table_like_from_raw_lines,
)


def classify_region(
    region: PageRegion,
    *,
    drawings: list[dict] | None = None,
) -> dict:
    """Classify a single page region as text or table; return a classification dict with is_table, table_type, and confidence."""
    raw_lines = [str(ln.get("text", "")).strip() for ln in region.lines if str(ln.get("text", "")).strip()]
    is_raw_table = is_table_like_from_raw_lines(raw_lines)
    if is_column_alignment_table_like(region.lines):
        is_raw_table = True
    if drawings and is_graphics_table_like(drawings):
        is_raw_table = True

    classification = classify_page_content(region.text)
    if is_raw_table and not classification["is_table"]:
        classification["is_table"] = True
        classification["is_text"] = False
        classification["confidence"] = "medium"
        if not classification["table_type"]:
            classification["table_type"] = detect_table_type(region.text)

    classification["is_raw_table"] = bool(is_raw_table)
    classification["region_id"] = region.region_id
    classification["page"] = region.page
    return classification
