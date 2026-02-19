from __future__ import annotations

import re

from rag_pdf.config import DEFAULT_CONFIG
from rag_pdf.text_normalize import normalize_line

TABLE_DIGIT_RATIO = DEFAULT_CONFIG.TABLE_DIGIT_RATIO
TABLE_SPACE_RATIO = DEFAULT_CONFIG.TABLE_SPACE_RATIO
TABLE_MIN_LINES = DEFAULT_CONFIG.TABLE_MIN_LINES


def is_table_like(text: str) -> bool:
    """
    Heuristic check if text content represents a table.

    Criteria:
    - At least 4 lines
    - High digit ratio (>12%)
    - Many double-spaces (column separation)
    """
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < TABLE_MIN_LINES:
        return False
    non_space_chars = sum(1 for ch in text if not ch.isspace())
    digit_ratio = sum(ch.isdigit() for ch in text) / max(1, non_space_chars)
    many_spaces = sum(l.count("  ") for l in lines) / max(1, len(lines))
    base = digit_ratio > TABLE_DIGIT_RATIO and many_spaces > TABLE_SPACE_RATIO
    if base:
        return True
    return is_small_financial_table(text)


def count_numeric_tokens(line: str) -> int:
    patterns = [
        r"\(\d[\d,]*\)",  # parentheses negatives
        r"\d{1,3}(?:,\d{3})+(?:\.\d+)?",  # comma numbers
        r"\d+(?:\.\d+)?%",  # percents
        r"\d+(?:\.\d+)?",  # plain numbers
        r"\b-\b",  # dash placeholder
    ]
    count = 0
    for pat in patterns:
        count += len(re.findall(pat, line))
    return count


def is_small_financial_table(text: str) -> bool:
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 4:
        return False

    currency_pattern = re.compile(r"£\s*'?000'?s?|\(£\s*'?000'?\)", re.IGNORECASE)
    if not currency_pattern.search(text):
        return False

    header_keywords = {
        "limit", "actual", "variance", "reported", "consolidated",
        "assets", "liabilities", "outturn", "surplus", "deficit", "net",
    }

    body_like = []
    header_like_indices = []
    for i, line in enumerate(lines):
        tokens = count_numeric_tokens(line)
        has_alpha = bool(re.search(r"[A-Za-z]", line))
        if has_alpha and tokens >= 2:
            body_like.append(i)
        words = {w.strip(".,;:()").lower() for w in line.split()}
        if sum(1 for w in words if w in header_keywords) >= 2:
            header_like_indices.append(i)
        if line.count("  ") >= 3 and tokens >= 2:
            return True

    def _window_has_body(start_idx: int, need: int) -> bool:
        end = min(start_idx + 12, len(lines))
        return sum(1 for i in body_like if start_idx <= i < end) >= need

    for i in range(len(lines)):
        if _window_has_body(i, 3):
            return True

    for idx in header_like_indices:
        if _window_has_body(idx + 1, 2):
            return True

    return False


def is_graphics_table_like(drawings: list[dict]) -> bool:
    """
    Detect table-like grids from vector drawings (lines/rectangles).

    The detector is intentionally conservative:
    - ignores filled rectangles (common in charts/bars)
    - prefers explicit horizontal/vertical line evidence
    - downweights pages dominated by curve paths (common in plots)
    """
    if not drawings:
        return False
    def _coords_from_item(item):
        if len(item) >= 5:
            return item[1], item[2], item[3], item[4]
        if len(item) == 3:
            p0, p1 = item[1], item[2]
            if isinstance(p0, (tuple, list)) and isinstance(p1, (tuple, list)) and len(p0) == 2 and len(p1) == 2:
                return p0[0], p0[1], p1[0], p1[1]
        if len(item) == 2:
            rect = item[1]
            if isinstance(rect, (tuple, list)) and len(rect) == 4:
                return rect[0], rect[1], rect[2], rect[3]
        return None
    h_lines = 0
    v_lines = 0
    rects = 0
    curve_ops = 0
    for d in drawings:
        drawing_type = str(d.get("type") or "").lower()
        has_fill = d.get("fill") is not None and drawing_type == "f"
        for item in d.get("items", []):
            if not item:
                continue
            op = item[0]
            if op == "c":
                curve_ops += 1
            if op == "l":
                coords = _coords_from_item(item)
                if not coords:
                    continue
                x0, y0, x1, y1 = coords
                dx = abs(x1 - x0)
                dy = abs(y1 - y0)
                if max(dx, dy) < 30:
                    continue
                if dy <= 2 and dx >= 30:
                    h_lines += 1
                elif dx <= 2 and dy >= 30:
                    v_lines += 1
            elif op == "re":
                # Filled rectangles are often chart bars/background blocks.
                if has_fill:
                    continue
                rects += 1
                coords = _coords_from_item(item)
                if not coords:
                    continue
                x0, y0, x1, y1 = coords
                dx = abs(x1 - x0)
                dy = abs(y1 - y0)
                if dx >= 30 and dy >= 12:
                    h_lines += 2
                    v_lines += 2
    # Plot-heavy pages are curve-dominant and should not be marked as table.
    if curve_ops > (h_lines + v_lines) * 3 and rects < 2:
        return False

    if rects >= 4 and h_lines >= 8 and v_lines >= 4:
        return True
    return h_lines >= 10 and v_lines >= 5


def is_column_alignment_table_like(lines_all: list[dict]) -> bool:
    """
    Detect table-like columns by clustered x positions of raw lines.
    """
    if not lines_all or len(lines_all) < 12:
        return False
    buckets = {}
    digit_lines = 0
    for ln in lines_all:
        x0 = ln.get("x0")
        if x0 is None:
            continue
        bucket = int(round(float(x0) / 12.0) * 12)
        buckets[bucket] = buckets.get(bucket, 0) + 1
        text = str(ln.get("text", ""))
        if any(ch.isdigit() for ch in text):
            digit_lines += 1
    dense_cols = [b for b, c in buckets.items() if c >= 6]
    if digit_lines < 6:
        return False
    return len(dense_cols) >= 5


if __name__ == "__main__":
    small_table = """\
Consolidated Net Assets (£000's)
Item 2024/25 2023/24
Total assets 1,559,285 1,512,300
Total liabilities (64,947) (60,100)
Net assets 1,494,338 1,452,200
"""
    narrative = """\
This report includes a single reference to £000 in the notes. The narrative
explains performance but does not present tabular figures or columns.
"""
    big_table = """\
Header  2024  2023  2022
Line A  1,000  2,000  3,000
Line B  4,000  5,000  6,000
Line C  7,000  8,000  9,000
Line D  10,000  11,000  12,000
"""
    print("small_table:", is_table_like(small_table))
    print("narrative:", is_table_like(narrative))
    print("big_table:", is_table_like(big_table))


def is_table_like_from_raw_lines(lines: list[str]) -> bool:
    """
    Check if raw lines (before cleanup) look like a table.

    More lenient than is_table_like() because it checks the structure
    before boilerplate removal may have collapsed the layout.

    Criteria:
    - High digit content (>15%)
    - Contains common table keywords
    - Multiple lines with consistent spacing patterns
    """
    if not lines or len(lines) < 2:
        return False

    text = "\n".join(lines)

    # Check for financial table keywords
    text_lower = text.lower()
    table_keywords = [
        "note", "£", "£000", "£'000",
        "total", "balance", "expenditure", "income", "assets",
        "liabilities", "depreciation", "impairment",
    ]
    table_keyword_patterns = [
        re.compile(r"\b20\d{2}/\d{2}\b"),  # Year pattern like 2022/23
        re.compile(r"£\s?[\d,]+(?:\.\d+)?"),  # Currency amounts like £1,000
    ]
    keyword_hits = sum(1 for kw in table_keywords if kw in text_lower)
    keyword_hits += sum(1 for pat in table_keyword_patterns if pat.search(text_lower))

    # Small-table detector: line has 2+ numeric bands + table keyword signal.
    range_pattern = re.compile(r"\b\d+\s*[-–]\s*\d+\b")
    for line in lines:
        if len(range_pattern.findall(line)) >= 2 and keyword_hits >= 1:
            return True

    # Check digit ratio
    non_space_chars = sum(1 for ch in text if not ch.isspace())
    digit_ratio = sum(ch.isdigit() for ch in text) / max(1, non_space_chars)
    if digit_ratio < 0.15:
        return False

    # Strong signal: multiple keywords + high digits
    if keyword_hits >= 2 and digit_ratio > 0.15:
        return True

    # Check for tabular spacing (multiple aligned columns)
    lines_with_content = [l for l in lines if l.strip()]
    if len(lines_with_content) >= 3:
        # Look for consistent spacing patterns (tabs or multiple spaces)
        multi_space_lines = sum(1 for l in lines_with_content if "  " in l or "\t" in l)
        if multi_space_lines / len(lines_with_content) > 0.5:
            return True

    return False


def contains_many_numbers(text: str) -> bool:
    """Check if text has high numeric content (>10% digits)."""
    digits = sum(ch.isdigit() for ch in text)
    non_space_chars = sum(1 for ch in text if not ch.isspace())
    return digits / max(1, non_space_chars) > 0.10


# =============================================================================
# TABLE CLASSIFICATION
# =============================================================================

def detect_table_type(text: str) -> str | None:
    """
    Classify financial table type from text content.

    Uses keyword matching on normalized text.

    Recognized types:
    - cash_flow: Cash flow statements and non-cash adjustments
    - balance_sheet: Statement of financial position
    - income_statement: Income/expenditure statements (SOCNE)
    - staff_costs: Employee benefits and staff costs
    - property: Property, plant, and equipment (PPE)
    - provisions: Provisions and liabilities
    - unknown: Unrecognized table type

    Args:
        text: Page text content

    Returns:
        Table type string or None if not a table
    """
    text_norm = normalize_line(text.lower())

    patterns = {
        "cash_flow": [
            "cash flow",
            "non-cash transaction",
            "note 2a",
            "note 2b",
            "reconciliation of net cash",
        ],
        "balance_sheet": [
            "balance sheet",
            "statement of financial position",
            "net assets",
            "total assets",
        ],
        "income_statement": [
            "statement of comprehensive net expenditure",
            "socne",
            "income and expenditure",
            "operating costs",
        ],
        "staff_costs": [
            "staff costs",
            "employee benefit",
            "remuneration",
            "pension costs",
        ],
        "property": [
            "property, plant and equipment",
            "ppe",
            "intangible assets",
            "additions to assets",
        ],
        "provisions": [
            "provisions",
            "contingent liabilities",
            "clinical negligence",
        ],
        "financial_instruments": [
            "financial instruments",
            "financial assets",
            "financial liabilities",
        ],
    }

    # Score each table type by keyword matches
    scores = {}
    for table_type, keywords in patterns.items():
        score = sum(1 for kw in keywords if kw in text_norm)
        if score > 0:
            scores[table_type] = score

    if not scores:
        return "unknown"

    # Return highest scoring type
    return max(scores.items(), key=lambda x: x[1])[0]


def classify_page_content(text: str) -> dict:
    """
    Enhanced page classification with table subtyping.

    Returns:
        {
            "is_table": bool,
            "table_type": str or None,
            "is_text": bool,
            "has_numbers": bool,
            "confidence": str  # "high", "medium", "low"
        }
    """
    is_tbl = is_table_like(text)
    table_type = detect_table_type(text) if is_tbl else None

    # Confidence scoring
    confidence = "low"
    if is_tbl and table_type and table_type != "unknown":
        confidence = "high"
    elif is_tbl:
        confidence = "medium"

    return {
        "is_table": is_tbl,
        "table_type": table_type if table_type != "unknown" else None,
        "is_text": not is_tbl,
        "has_numbers": contains_many_numbers(text),
        "confidence": confidence,
    }
