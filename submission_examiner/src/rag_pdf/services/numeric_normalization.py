"""Numeric value normalisation utilities for the answer extraction pipeline.

Provides functions to detect the semantic dimension (percent, currency, count) and
scale multiplier (thousands, millions) of numeric strings found in NHS annual report
text, and to canonicalise them into a consistent formatted form.
"""

from __future__ import annotations

import re
from typing import Optional


def detect_numeric_dimension(val: str) -> str:
    """Classify a numeric string as 'percent', 'currency', or 'count' based on surrounding symbols."""
    text = str(val or "").lower()
    if any(tok in text for tok in ["%", "percent", "percentage"]):
        return "percent"
    if "£" in text or "$" in text or "eur" in text or "usd" in text:
        return "currency"
    return "count"


def detect_numeric_multiplier(val: str) -> float:
    """Return the scale multiplier implied by the string (1000 for £000/thousands, 1e6 for millions, etc.)."""
    text = str(val or "").lower()
    compact = re.sub(r"\s+", "", text)
    if "£000" in compact or "($000)" in compact or "usd000" in compact or "eur000" in compact:
        return 1_000.0
    if "(000)" in compact or "[000]" in compact:
        return 1_000.0
    if re.search(r"\b(thousand|thousands)\b", text):
        return 1_000.0
    if re.search(r"(?<![a-z])[-+]?\d+(?:\.\d+)?\s*k\b", text):
        return 1_000.0
    if re.search(r"(?<![a-z])[-+]?\d+(?:\.\d+)?\s*m\b", text):
        return 1_000_000.0
    if re.search(r"\b(mn|million|millions)\b", text):
        return 1_000_000.0
    if re.search(r"(?<![a-z])[-+]?\d+(?:\.\d+)?\s*bn\b", text):
        return 1_000_000_000.0
    if re.search(r"\b(billion|billions)\b", text):
        return 1_000_000_000.0
    return 1.0


def normalize_numeric_value(val: str) -> Optional[dict[str, float | str]]:
    """Parse a numeric string and return a dict with 'value', 'dimension', and 'multiplier'; return None if unparseable."""
    text = str(val or "").strip()
    if not text:
        return None
    lowered = text.lower()
    negative = bool(re.search(r"\(\s*[-+]?\d[\d,]*(?:\.\d+)?\s*\)", lowered))
    cleaned = lowered.replace(",", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    try:
        number = float(match.group())
    except Exception:
        return None
    if negative and number > 0:
        number = -number
    multiplier = detect_numeric_multiplier(lowered)
    dimension = detect_numeric_dimension(lowered)
    return {
        "value": float(number * multiplier),
        "dimension": dimension,
        "multiplier": multiplier,
    }


def looks_like_standalone_numeric_answer(text: str) -> bool:
    """Return True if the text is a short, self-contained numeric expression (value only, no prose)."""
    s = str(text or "").strip()
    if not s:
        return False
    if len(s) > 64:
        return False
    alpha_tokens = re.findall(r"[A-Za-z]+", s)
    allowed_words = {
        "m", "mn", "bn", "million", "millions", "billion", "billions",
        "thousand", "thousands", "percent", "percentage", "eur", "usd",
    }
    if any(tok.lower() not in allowed_words for tok in alpha_tokens):
        return False
    cleaned = re.sub(r"[\d\s,.\-+£$()%/%a-zA-Z]", "", s)
    if cleaned.strip():
        return False
    return normalize_numeric_value(s) is not None


def canonicalize_numeric_text(text: str) -> Optional[str]:
    """Normalise a numeric string and format it as a human-readable canonical value (e.g. '£1,234' or '12.5%')."""
    normalized = normalize_numeric_value(text)
    if normalized is None:
        return None

    value = float(normalized["value"])
    dimension = str(normalized["dimension"])
    if dimension == "percent":
        if float(value).is_integer():
            return f"{int(value)}%"
        return f"{value:.2f}".rstrip("0").rstrip(".") + "%"

    sign = "-" if value < 0 else ""
    abs_value = abs(value)
    if abs_value.is_integer():
        body = f"{int(abs_value):,}"
    else:
        body = f"{abs_value:,.2f}".rstrip("0").rstrip(".")
    if dimension == "currency":
        return f"{sign}£{body}"
    return f"{sign}{body}"
