from __future__ import annotations

import pandas as pd

from rag_pdf.headings import is_part_label, looks_like_heading_text_only
from rag_pdf.text_normalize import normalize_line


def build_sections_from_pages(pages_df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer document sections from page-level headings.

    Sections are bounded by heading detections. Each section spans
    from its heading page to the page before the next heading.
    """
    sections = []
    current_part = None
    current_section = "Unknown"
    current_pages: list[int] = []
    current_texts: list[str] = []

    def flush():
        if not current_pages:
            return
        sections.append({
            "doc_id": pages_df["doc_id"].iloc[0],
            "report_year": pages_df["report_year"].iloc[0],
            "period_end_date": pages_df["period_end_date"].iloc[0],
            "report_year_source": pages_df["report_year_source"].iloc[0],
            "run_date_utc": pages_df["run_date_utc"].iloc[0],
            "part": current_part or "Unknown",
            "section_title": current_section or "Unknown",
            "page_start": int(min(current_pages)),
            "page_end": int(max(current_pages)),
            "section_text": "\n".join(current_texts).strip(),
        })
        current_pages.clear()
        current_texts.clear()

    for _, row in pages_df.iterrows():
        page_no = int(row["page"])
        text = str(row["clean_text"] or "")
        lines = [normalize_line(x) for x in text.splitlines() if normalize_line(x)]

        # Check for part labels
        for l in lines[:25]:
            p = is_part_label(l)
            if p:
                current_part = p
                break

        # Check for headings
        heading_candidates = row.get("heading_candidates", [])
        heading_found = None

        if isinstance(heading_candidates, list) and heading_candidates:
            heading_found = heading_candidates[0]
        else:
            for l in lines[:25]:
                if looks_like_heading_text_only(l) and not is_part_label(l):
                    heading_found = l
                    break

        if heading_found and current_texts:
            flush()
            current_section = heading_found

        current_pages.append(page_no)
        current_texts.append(text)

    flush()
    df = pd.DataFrame(sections)
    if len(df) > 0:
        df["word_count"] = df["section_text"].str.split().str.len()
    else:
        df["word_count"] = []
    return df


def find_section_for_page(sections_df: pd.DataFrame, page_no: int) -> tuple[str, str]:
    """
    Find which section a page belongs to.

    Returns:
        (part, section_title)
    """
    if len(sections_df) == 0:
        return "Unknown", "Unknown"

    m = sections_df[(sections_df["page_start"] <= page_no) & (sections_df["page_end"] >= page_no)]
    if len(m) == 0:
        return "Unknown", "Unknown"
    r = m.iloc[-1]
    return str(r["part"]), str(r["section_title"])
