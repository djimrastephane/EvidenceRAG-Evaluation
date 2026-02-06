from __future__ import annotations

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
    digit_ratio = sum(ch.isdigit() for ch in text) / max(1, len(text))
    many_spaces = sum(l.count("  ") for l in lines) / max(1, len(lines))
    return digit_ratio > TABLE_DIGIT_RATIO and many_spaces > TABLE_SPACE_RATIO


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

    # Check digit ratio
    digit_ratio = sum(ch.isdigit() for ch in text) / max(1, len(text))
    if digit_ratio < 0.15:
        return False

    # Check for financial table keywords
    text_lower = text.lower()
    table_keywords = [
        "note", "£", "£000", "£'000", "2022/23", "2021/22",
        "total", "balance", "expenditure", "income", "assets",
        "liabilities", "depreciation", "impairment",
    ]
    keyword_hits = sum(1 for kw in table_keywords if kw in text_lower)

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
    return digits / max(1, len(text)) > 0.10


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
