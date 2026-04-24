from __future__ import annotations

"""OCR fallback heuristics.

OCR is only triggered when extracted text is clearly poor. The heuristics here
are intentionally lightweight because the thesis goal is traceable behaviour,
not a learned OCR-quality classifier.
"""

from .schemas import OCRConfig


def page_needs_ocr(text: str, config: OCRConfig) -> bool:
    """Return whether a page should fall back to OCR based on text quality."""
    stripped = (text or "").strip()
    if len(stripped) < config.min_chars_before_fallback:
        return True
    alpha = sum(char.isalpha() for char in stripped) / max(len(stripped), 1)
    digit = sum(char.isdigit() for char in stripped) / max(len(stripped), 1)
    return alpha < config.min_alpha_ratio and digit < config.min_digit_ratio
