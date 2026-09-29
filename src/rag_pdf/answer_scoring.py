"""Answer-correctness scoring and FP1–FP7 failure attribution for the evaluation scripts.

Single source of truth for:
- ``score_answer_correctness``: is an extracted/generated answer correct, partial or wrong?
- ``answer_in_context``: does the retrieved context contain the gold answer? (FP3)
- ``categorize_failure_type``: FP1–FP7 / HIT / UNSCORED, following the README taxonomy.

Numeric policy
--------------
- Every number mention is parsed with its sign, currency, percent flag and scale
  (k / m / million / bn, and a ``(£000)`` / ``£000s`` table-unit marker that scales bare
  currency amounts by 1,000).
- Two values are equivalent when they agree at the coarser of their stated precisions
  (difference ≤ half a unit of the last stated digit) **and** differ by at most 2%.
  So 12% ≠ 13%, 3 ≠ 4, 25.0% ≠ 25.9%, −£3.2m ≠ £3.2m, while £1.28 billion = £1,278,771 (£000).
- Percentages never match non-percentages (0.25 ≠ 25%).
- The gold answer's *primary* value is its first number mention; later mentions are
  breakdowns or context. An extracted answer is correct if any of its mentions equals
  the primary value. A scale-only mismatch (e.g. £000 omitted) or a match on a secondary
  gold value is scored as partial.
- Year-like integers (1900–2100, fiscal years such as 2024/25) and alphanumeric codes
  (COVID-19, Q3) are ignored unless the gold value is itself a year.

Text policy (heuristic; claim-level human labels are still required for validity)
- Correct: normalised gold text is a substring of the answer, or ≥ 80% of gold content
  tokens appear in it. Partial: ≥ 40%. Otherwise incorrect. If the gold text contains
  numbers and none of them appear in the answer, it is incorrect.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

FAILURE_TYPES = {
    "FP1_MISSING_CONTENT": "retrieval",
    "FP2_MISSED_TOP_RANK": "retrieval",
    "FP3_NOT_IN_CONTEXT": "retrieval",
    "FP4_NOT_EXTRACTED": "generation",
    "FP5_WRONG_FORMAT": "generation",
    "FP6_INCORRECT_SPECIFICITY": "generation",
    "FP7_INCOMPLETE": "generation",
    "HIT": "none",
    "UNSCORED": "none",
}

MISSING_EXTRACTION_MARKERS = {"(no chunk text available)", "(no extraction rule matched)"}
REL_TOLERANCE = 0.02
TEXT_CORRECT_RECALL = 0.8
TEXT_PARTIAL_RECALL = 0.4

_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were with "
    "which who what when where how nhs grampian".split()
)

_TABLE_UNIT_RE = re.compile(r"\(\s*£\s*000\s*'?s?\s*\)|£\s*000'?s\b|£\s*'000", re.IGNORECASE)
_CODE_TOKEN_RE = re.compile(r"\b[A-Za-z]+[-‐‑]?\d+[A-Za-z]*\b")
_FISCAL_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\s*[/\-–]\s*\d{2,4}\b")
_PAREN_NEG_RE = re.compile(r"\(\s*([£$€]?)\s*(\d[\d,]*(?:\.\d+)?)\s*(bn|billion|mn|million|m|k|thousand)?\s*\)", re.IGNORECASE)
_CUR_PAREN_NEG_RE = re.compile(r"([£$€])\s*\(\s*(\d[\d,]*(?:\.\d+)?)\s*\)", re.IGNORECASE)
_MENTION_RE = re.compile(
    r"(?<![\w.])"
    r"(?P<sign>[-−–])?\s*(?P<cur>[£$€])?\s*(?P<sign2>[-−–])?\s*"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s*(?P<scale>bn|billion|mn|million|m|k|thousand)(?![a-z]))?"
    r"(?:\s*(?P<pct>%|per\s*cent|percent))?",
    re.IGNORECASE,
)
_SCALES = {
    "k": 1e3, "thousand": 1e3,
    "m": 1e6, "mn": 1e6, "million": 1e6,
    "bn": 1e9, "billion": 1e9,
}


@dataclass(frozen=True)
class NumberMention:
    value: float
    resolution: float  # size of one unit in the last stated digit, after scaling
    is_percent: bool
    is_currency: bool
    is_year: bool


def _prepare(text: str) -> tuple[str, bool]:
    s = str(text or "")
    table_unit = bool(_TABLE_UNIT_RE.search(s))
    s = _TABLE_UNIT_RE.sub(" ", s)
    s = _CODE_TOKEN_RE.sub(" ", s)
    # Accounting negatives: "(£3.2m)" / "£(3.2)m" -> "-£3.2m". A parenthesis that encloses
    # more than an amount ("(£3.2m on repairs)") is a breakdown, not a sign.
    s = _CUR_PAREN_NEG_RE.sub(lambda m: f"-{m.group(1)}{m.group(2)}", s)
    s = _PAREN_NEG_RE.sub(lambda m: f"-{m.group(1)}{m.group(2)}{m.group(3) or ''}", s)
    return s, table_unit


def parse_numbers(text: str, *, include_years: bool = False) -> list[NumberMention]:
    """Return every number mention in ``text`` in order of appearance."""
    s, table_unit = _prepare(text)
    fiscal_spans = [m.span() for m in _FISCAL_YEAR_RE.finditer(s)]
    mentions: list[NumberMention] = []
    for m in _MENTION_RE.finditer(s):
        raw = m.group("num")
        in_fiscal = any(a <= m.start("num") < b for a, b in fiscal_spans)
        number = float(raw.replace(",", ""))
        decimals = len(raw.split(".", 1)[1]) if "." in raw else 0
        scale_word = (m.group("scale") or "").lower()
        is_percent = bool(m.group("pct"))
        is_currency = bool(m.group("cur"))
        scale = _SCALES.get(scale_word, 1.0)
        if scale == 1.0 and table_unit and is_currency and not is_percent:
            scale = 1e3
        is_year = (
            in_fiscal
            or (not is_currency and not is_percent and not scale_word and decimals == 0
                and "," not in raw and 1900 <= number <= 2100)
        )
        if is_year and not include_years:
            continue
        negative = bool(m.group("sign") or m.group("sign2"))
        value = -number * scale if negative else number * scale
        mentions.append(
            NumberMention(
                value=value,
                resolution=(10.0 ** -decimals) * scale,
                is_percent=is_percent,
                is_currency=is_currency,
                is_year=is_year,
            )
        )
    return mentions


def values_equivalent(a: NumberMention, b: NumberMention) -> bool:
    """True when two mentions state the same quantity at the coarser stated precision."""
    if a.is_percent != b.is_percent:
        return False
    diff = abs(a.value - b.value)
    magnitude = max(abs(a.value), abs(b.value))
    if magnitude == 0.0:
        return diff == 0.0
    half_unit = max(a.resolution, b.resolution) / 2.0
    return diff <= half_unit * (1 + 1e-9) and diff <= REL_TOLERANCE * magnitude


def _scale_only_mismatch(expected: NumberMention, got: NumberMention) -> bool:
    for factor in (1e3, 1e6, 1e9):
        for f in (factor, 1.0 / factor):
            scaled = NumberMention(got.value * f, got.resolution * f, got.is_percent, got.is_currency, got.is_year)
            if values_equivalent(expected, scaled):
                return True
    return False


def _normalize_text(val: Any) -> str:
    text = re.sub(r"[£$€]", "", str(val or "").lower())
    text = re.sub(r"[^a-z0-9%\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _content_tokens(val: Any) -> list[str]:
    return [t for t in _normalize_text(val).split() if t not in _STOPWORDS and (len(t) > 1 or t.isdigit())]


def _text_recall(expected: Any, answer: str) -> float:
    exp = _content_tokens(expected)
    if not exp:
        return 0.0
    got = set(_content_tokens(answer))
    return sum(1 for t in exp if t in got) / len(exp)


def _expected_items(expected_answer: Any) -> list[str]:
    if isinstance(expected_answer, (list, tuple)):
        return [str(v) for v in expected_answer if str(v).strip()]
    return [str(expected_answer)] if str(expected_answer or "").strip() else []


def is_missing_extraction(extracted_answer: Optional[str]) -> bool:
    text = str(extracted_answer or "").strip().lower()
    return not text or text in MISSING_EXTRACTION_MARKERS


def _score_numeric_item(expected: str, answer: str) -> Optional[str]:
    """Return 'correct' / 'partial' / 'incorrect', or None if ``expected`` has no number."""
    exp = parse_numbers(expected)
    include_years = False
    if not exp:
        exp = parse_numbers(expected, include_years=True)
        include_years = True
    if not exp:
        return None
    primary, secondary = exp[0], exp[1:]
    got = parse_numbers(answer, include_years=include_years)
    if any(values_equivalent(primary, g) for g in got):
        return "correct"
    if any(_scale_only_mismatch(primary, g) for g in got):
        return "partial"
    if any(values_equivalent(s, g) for s in secondary for g in got):
        return "partial"
    return "incorrect"


def _score_text_item(expected: str, answer: str) -> str:
    exp_norm = _normalize_text(expected)
    if exp_norm and exp_norm in _normalize_text(answer):
        return "correct"
    exp_digits = {t for t in _content_tokens(expected) if t.isdigit()}
    if exp_digits and not exp_digits & set(_content_tokens(answer)):
        return "incorrect"  # the numbers in a text answer are essential ("Stage 3" != "stage 4")
    recall = _text_recall(expected, answer)
    if recall >= TEXT_CORRECT_RECALL:
        return "correct"
    if recall >= TEXT_PARTIAL_RECALL:
        return "partial"
    return "incorrect"


def score_answer_correctness(
    expected_answer: Any,
    answer_type: str,
    extracted_answer: Optional[str],
) -> tuple[Optional[bool], str]:
    """Return (is_correct, status); status is correct / partial / incorrect / not_extracted / not_scored."""
    items = _expected_items(expected_answer)
    if not items:
        return None, "not_scored"
    if is_missing_extraction(extracted_answer):
        return False, "not_extracted"
    answer = str(extracted_answer)
    statuses = []
    for item in items:
        status = _score_numeric_item(item, answer) if answer_type == "number" else None
        statuses.append(status or _score_text_item(item, answer))
    if all(s == "correct" for s in statuses):
        return True, "correct"
    if any(s in {"correct", "partial"} for s in statuses):
        return False, "partial"
    return False, "incorrect"


def answer_in_context(expected_answer: Any, context_text: str, answer_type: str) -> bool:
    """True when the retrieved context contains every gold answer item (scale-only differences allowed)."""
    items = _expected_items(expected_answer)
    if not items or not str(context_text or "").strip():
        return False
    for item in items:
        exp = (parse_numbers(item) or parse_numbers(item, include_years=True)) if answer_type == "number" else []
        if exp:
            primary = exp[0]
            got = parse_numbers(context_text, include_years=primary.is_year)
            if not any(values_equivalent(primary, g) or _scale_only_mismatch(primary, g) for g in got):
                return False
        elif _score_text_item(item, context_text) != "correct":
            return False
    return True


def format_matches(answer_type: str, extracted_answer: str) -> bool:
    if answer_type == "number":
        return bool(parse_numbers(extracted_answer, include_years=True))
    if answer_type == "date":
        patterns = (r"\b\d{4}-\d{2}-\d{2}\b", r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", r"\b\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b")
        return any(re.search(p, extracted_answer) for p in patterns)
    if answer_type == "list":
        return any(token in extracted_answer for token in (",", ";", "\n", " and "))
    return True


def categorize_failure_type(
    page_hit: int,
    gold_exists: bool,
    expected_answer: Any,
    answer_type: str,
    context_text: str,
    extracted_answer: Optional[str],
) -> str:
    """Attribute one query to FP1–FP7, HIT or UNSCORED (README failure taxonomy).

    FP1: gold page not in the index. FP2: gold page not retrieved at rank 1.
    FP3: gold page retrieved but the answer is not in the retrieved context.
    FP4: nothing extracted. FP5: wrong answer format. FP6: wrong value.
    FP7: partially correct. UNSCORED: no gold answer to score against.
    """
    if not gold_exists:
        return "FP1_MISSING_CONTENT"
    if not page_hit:
        return "FP2_MISSED_TOP_RANK"
    if not _expected_items(expected_answer):
        return "UNSCORED"
    if not answer_in_context(expected_answer, context_text, answer_type):
        return "FP3_NOT_IN_CONTEXT"
    if is_missing_extraction(extracted_answer):
        return "FP4_NOT_EXTRACTED"
    if not format_matches(answer_type, str(extracted_answer)):
        return "FP5_WRONG_FORMAT"
    correct, status = score_answer_correctness(expected_answer, answer_type, extracted_answer)
    if correct:
        return "HIT"
    if status == "partial":
        return "FP7_INCOMPLETE"
    return "FP6_INCORRECT_SPECIFICITY"


__all__ = [
    "FAILURE_TYPES",
    "NumberMention",
    "answer_in_context",
    "categorize_failure_type",
    "format_matches",
    "is_missing_extraction",
    "parse_numbers",
    "score_answer_correctness",
    "values_equivalent",
]
