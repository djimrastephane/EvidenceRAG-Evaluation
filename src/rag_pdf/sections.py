from __future__ import annotations

import re

import pandas as pd

from rag_pdf.headings import (
    is_part_label,
    is_section_anchor_line,
    is_global_boilerplate_heading,
    looks_like_heading_text_only,
    looks_like_lettered_subsection,
)
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
    current_subsection = None
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
            "subsection_title": current_subsection or "Unknown",
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

        raw_top = row.get("top_lines", [])
        top_lines: list[str] = []
        if isinstance(raw_top, (list, tuple)):
            for item in raw_top:
                if isinstance(item, dict):
                    txt = str(item.get("text", ""))
                else:
                    txt = str(item)
                norm = normalize_line(txt)
                if norm:
                    top_lines.append(norm)

        # Check for part labels
        for l in top_lines or lines[:25]:
            p = is_part_label(l)
            if p:
                current_part = p
                break

        # Check for headings
        raw_candidates = row.get("heading_candidates", [])
        if raw_candidates is None or raw_candidates is False:
            heading_candidates = []
        elif isinstance(raw_candidates, str):
            heading_candidates = []
        elif isinstance(raw_candidates, (list, tuple)):
            heading_candidates = list(raw_candidates)
        elif hasattr(raw_candidates, "__iter__"):
            heading_candidates = list(raw_candidates)
        else:
            heading_candidates = []
        section_found = None
        subsection_found = None

        if top_lines:
            for i, line in enumerate(top_lines):
                if is_part_label(line):
                    continue
                if (
                    re.match(r"^[A-Z][.)]?$", line)
                    and i + 1 < len(top_lines)
                    and subsection_found is None
                ):
                    combined = f"{line[0]} {top_lines[i + 1]}"
                    if looks_like_lettered_subsection(combined):
                        subsection_found = combined
                        continue
                if subsection_found is None and looks_like_lettered_subsection(line):
                    subsection_found = line
                    continue
                if section_found is None and (
                    is_section_anchor_line(line)
                    or (looks_like_heading_text_only(line) and not is_global_boilerplate_heading(line))
                ):
                    section_found = line
                if section_found and subsection_found:
                    break
        elif heading_candidates:
            for cand in heading_candidates:
                if subsection_found is None and looks_like_lettered_subsection(cand):
                    subsection_found = cand
                    continue
                if section_found is None and looks_like_heading_text_only(cand):
                    section_found = cand
                if section_found and subsection_found:
                    break
        else:
            for l in lines[:25]:
                if not is_part_label(l):
                    if subsection_found is None and looks_like_lettered_subsection(l):
                        subsection_found = l
                        continue
                    if section_found is None and looks_like_heading_text_only(l):
                        section_found = l
                if section_found and subsection_found:
                    break

        if subsection_found is None and text:
            m = re.search(r"\b([A-Z])[.)]?\s+([A-Z][A-Z ]{3,})\b", text)
            if m:
                candidate = f"{m.group(1)} {normalize_line(m.group(2))}"
                if looks_like_lettered_subsection(candidate):
                    subsection_found = candidate

        if (section_found or subsection_found) and current_texts:
            flush()
            if section_found:
                current_section = section_found
                current_subsection = None
            if subsection_found:
                current_subsection = subsection_found

        current_pages.append(page_no)
        current_texts.append(text)

    flush()
    df = pd.DataFrame(sections)
    if len(df) > 0:
        df["word_count"] = df["section_text"].str.split().str.len()
    else:
        df["word_count"] = []
    return df


def find_section_for_page(sections_df: pd.DataFrame, page_no: int) -> tuple[str, str, str]:
    """
    Find which section a page belongs to.

    Returns:
        (part, section_title)
    """
    if len(sections_df) == 0:
        return "Unknown", "Unknown", "Unknown"

    m = sections_df[(sections_df["page_start"] <= page_no) & (sections_df["page_end"] >= page_no)]
    if len(m) == 0:
        return "Unknown", "Unknown", "Unknown"
    r = m.iloc[-1]
    return str(r["part"]), str(r["section_title"]), str(r.get("subsection_title", "Unknown"))
