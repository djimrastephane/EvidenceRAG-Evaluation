"""
check_ocr_coordinate_fix.py

Verifies the impact of the OCR y-coordinate fix (extract_page.py).

Before the fix, every OCR line was assigned:
    y0 = y1 = page_height * 0.5

This placed ALL lines at the page midpoint, so boilerplate stripping
(which removes lines in the top 8% and bottom 8% of the page) never
removed any OCR header/footer lines.

After the fix, lines are evenly distributed top-to-bottom, so actual
headers and footers land in the correct coordinate zones and get stripped.

Usage:
    # Synthetic demo (no PDF needed)
    python scripts/check_ocr_coordinate_fix.py

    # Against a real PDF page that triggers OCR fallback
    python scripts/check_ocr_coordinate_fix.py --pdf path/to/file.pdf --page 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_pdf.boilerplate import strip_by_coordinates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PAGE_HEIGHT = 841.0  # A4 in points
PAGE_WIDTH  = 595.0


def make_lines_before(text_lines: list[str]) -> list[dict]:
    """Old behaviour: every line at page midpoint."""
    y = PAGE_HEIGHT * 0.5
    return [
        {"text": ln, "x0": PAGE_WIDTH * 0.1, "x1": PAGE_WIDTH * 0.9,
         "y0": y, "y1": y, "max_size": 0.0}
        for ln in text_lines
    ]


def make_lines_after(text_lines: list[str]) -> list[dict]:
    """New behaviour: lines evenly distributed across page body (5%–95%)."""
    n = max(len(text_lines), 1)
    line_height = (PAGE_HEIGHT * 0.9) / n
    y_start     = PAGE_HEIGHT * 0.05
    lines = []
    for i, ln in enumerate(text_lines):
        y0 = y_start + i * line_height
        y1 = y0 + line_height
        lines.append({"text": ln, "x0": PAGE_WIDTH * 0.1, "x1": PAGE_WIDTH * 0.9,
                       "y0": y0, "y1": y1, "max_size": 0.0})
    return lines


def run_stripping(lines: list[dict], label: str) -> None:
    kept, removed_top, removed_bot = strip_by_coordinates(
        lines, page_height=PAGE_HEIGHT, page_width=PAGE_WIDTH, rotation=0
    )
    top_zone  = PAGE_HEIGHT * 0.08
    bot_zone  = PAGE_HEIGHT * 0.92

    print(f"\n  [{label}]")
    print(f"  Strip zones: y < {top_zone:.0f} (top 8%) or y > {bot_zone:.0f} (bottom 8%)")
    print(f"  {'Line':<45}  {'y_mid':>7}  {'result'}")
    print(f"  {'─'*45}  {'─'*7}  {'─'*10}")

    for ln in lines:
        y_mid = (ln["y0"] + ln["y1"]) / 2.0
        if ln["text"] in removed_top:
            result = "STRIPPED (header)"
        elif ln["text"] in removed_bot:
            result = "STRIPPED (footer)"
        else:
            result = "kept"
        flag = " ✗" if "STRIPPED" in result else ""
        print(f"  {ln['text'][:45]:<45}  {y_mid:>7.1f}  {result}{flag}")

    print(f"\n  Summary: {len(kept)} kept, "
          f"{len(removed_top)} stripped as header, "
          f"{len(removed_bot)} stripped as footer")


# ---------------------------------------------------------------------------
# Synthetic demo
# ---------------------------------------------------------------------------

SYNTHETIC_PAGE = [
    "Grampian NHS Board — Annual Accounts 2022/23",   # ← real header
    "Page 47 of 120",                                  # ← real header/page number
    "",
    "Note 4: Staff Costs",
    "The following table sets out staff costs for the year.",
    "Permanently employed staff costs amounted to £142.3m",
    "in 2022/23, compared to £138.1m in 2021/22.",
    "Agency staff costs increased to £12.4m.",
    "Total staff costs including social security costs",
    "and pension contributions are shown in Table 4.1.",
    "Further details are provided in the Remuneration Report.",
    "",
    "Grampian NHS Board",                              # ← real footer
    "47",                                              # ← page number footer
]

# Remove blank lines (OCR output already filtered via normalize_line)
OCR_LINES = [ln for ln in SYNTHETIC_PAGE if ln.strip()]


def demo_ordering(lines_before: list[dict], lines_after: list[dict]) -> None:
    """Show that sort() was a no-op before the fix."""
    print("\n  [Line ordering after sort()]")
    print(f"  {'Line':<45}  {'before y_mid':>12}  {'after y_mid':>11}")
    print(f"  {'─'*45}  {'─'*12}  {'─'*11}")
    for b, a in zip(lines_before, lines_after):
        yb = (b["y0"] + b["y1"]) / 2.0
        ya = (a["y0"] + a["y1"]) / 2.0
        print(f"  {b['text'][:45]:<45}  {yb:>12.1f}  {ya:>11.1f}")
    print()
    print("  Before: every line at y=420 — sort by (y0, x0) is a no-op.")
    print("  Tesseract usually outputs top-to-bottom, but any out-of-order")
    print("  span cannot be corrected. After: sort restores visual order.")


# Sparse-page example: few lines → first/last fall inside 8% strip zone
SPARSE_PAGE = [
    "Grampian NHS Board — Annual Accounts 2022/23",   # header
    "Note 4: Staff Costs",
    "Total staff costs were £154.7m (2021/22: £142.1m).",
    "47",                                              # footer
]
SPARSE_LINES = [ln for ln in SPARSE_PAGE if ln.strip()]


def demo_synthetic() -> None:
    print("=" * 65)
    print("  OCR coordinate fix — synthetic page demo")
    print("  (Simulates a page where PyMuPDF yielded no text and OCR")
    print("   fallback was used.  Page height = 841pt, width = 595pt)")
    print("=" * 65)

    # ── Dense page (12 lines) ──────────────────────────────────────────
    print("\n  ── Dense page (12 OCR lines) ──")
    print("  With many lines the first/last y_mid lands close to but inside")
    print("  the 8% keep zone, so stripping behaviour is identical.")
    run_stripping(make_lines_before(OCR_LINES), "BEFORE fix  (all lines at y=420)")
    run_stripping(make_lines_after(OCR_LINES),  "AFTER fix   (lines evenly spaced)")

    # ── Sparse page (4 lines) ─────────────────────────────────────────
    print("\n  ── Sparse page (4 OCR lines) ──")
    print("  Fewer lines → larger line_height → first/last y_mid falls in")
    print("  the 8% header/footer strip zone. This is where the fix bites.")
    run_stripping(make_lines_before(SPARSE_LINES), "BEFORE fix  (all lines at y=420)")
    run_stripping(make_lines_after(SPARSE_LINES),  "AFTER fix   (lines evenly spaced)")

    # ── Ordering issue ────────────────────────────────────────────────
    demo_ordering(make_lines_before(OCR_LINES), make_lines_after(OCR_LINES))


# ---------------------------------------------------------------------------
# Real PDF mode
# ---------------------------------------------------------------------------

def demo_real_pdf(pdf_path: str, page_index: int) -> None:
    try:
        import pymupdf as fitz
        import pdfplumber
    except ImportError as e:
        print(f"Missing dependency: {e}")
        sys.exit(1)

    from rag_pdf.extract_page import extract_page_struct_hybrid

    pdf_path_str = str(Path(pdf_path).resolve())
    doc = fitz.open(pdf_path_str)
    pl  = pdfplumber.open(pdf_path_str)

    struct, extractor, note = extract_page_struct_hybrid(
        doc, pl, page_index, pdf_path=pdf_path_str
    )

    lines = struct.get("lines_all", [])
    ph    = struct.get("page_height", PAGE_HEIGHT)
    pw    = struct.get("page_width",  PAGE_WIDTH)
    rot   = struct.get("rotation", 0)

    print("=" * 65)
    print(f"  Real PDF: {Path(pdf_path).name}  page {page_index + 1}")
    print(f"  Extractor used: {extractor}   note: {note}")
    print(f"  Lines extracted: {len(lines)}   page size: {pw:.0f}×{ph:.0f}")
    print("=" * 65)

    if extractor != "ocr":
        print("\n  This page did NOT trigger OCR fallback.")
        print("  The coordinate fix only applies to OCR pages.")
        print("  Try a page that is image-only or has very little text.")
        return

    texts = [ln["text"] for ln in lines]

    run_stripping(make_lines_before(texts), "BEFORE fix  (all lines at y=midpoint)")
    run_stripping(lines,                    "AFTER fix   (real evenly-spaced coords)")

    doc.close()
    pl.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Check OCR coordinate fix impact.")
    parser.add_argument("--pdf",  default=None, help="Path to a PDF file (optional).")
    parser.add_argument("--page", default=1,    type=int,
                        help="1-based page number to test (default: 1).")
    args = parser.parse_args()

    if args.pdf:
        demo_real_pdf(args.pdf, args.page - 1)
    else:
        demo_synthetic()


if __name__ == "__main__":
    main()
