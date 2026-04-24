"""Candidate selection logic for extracting numeric answers from retrieved chunks.

Scans retrieval results for numeric expressions matching a question's expected dimension
(percent, currency, count), scores candidates using keyword overlap, cue terms, and
structural signals, and returns the highest-confidence candidate with a debug trace.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from rag_pdf.services.numeric_normalization import (
    detect_numeric_dimension,
    normalize_numeric_value,
)

NUMERIC_EXPR_RE = re.compile(
    r"(?P<expr>"
    r"(?:£\s*)?"
    r"(?:\(\s*)?[-+]?\d[\d,]*(?:\.\d+)?(?:\s*\))?"
    r"(?:\s*(?:\((?:£)?000\)|%|percent(?:age)?|k|m|bn|million|millions|billion|billions|thousand|thousands))?"
    r")",
    flags=re.IGNORECASE,
)

QUESTION_STOPWORDS = {
    "what",
    "which",
    "who",
    "where",
    "when",
    "why",
    "how",
    "much",
    "many",
    "did",
    "does",
    "do",
    "was",
    "were",
    "is",
    "are",
    "the",
    "for",
    "from",
    "with",
    "during",
    "year",
    "report",
    "reported",
    "amount",
    "number",
    "value",
    "total",
    "of",
    "in",
    "on",
    "to",
    "by",
    "and",
    "difference",
}


def _question_keywords(question: str) -> list[str]:
    toks = re.findall(r"[a-z][a-z0-9\-]{2,}", str(question or "").lower())
    out: list[str] = []
    seen: set[str] = set()
    for tok in toks:
        if tok in QUESTION_STOPWORDS:
            continue
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def _question_expected_dimension(question: str) -> Optional[str]:
    q = str(question or "").lower()
    if any(tok in q for tok in ("%", "percent", "percentage", "proportion", "share", "rate", "ratio")):
        return "percent"
    if any(tok in q for tok in ("£", "cost", "spend", "spending", "budget", "deficit", "surplus", "overspend", "shortfall", "income", "expenditure", "funding")):
        return "currency"
    return None


def _question_cue_terms(question: str) -> list[str]:
    q = str(question or "").lower()
    cues: list[str] = []
    if any(tok in q for tok in ("budget", "spending", "spend", "differ", "difference", "underspend", "overspend")):
        cues.extend(["variance", "surplus", "deficit", "overspend", "underspend", "budget"])
    if any(tok in q for tok in ("deficit", "surplus", "overspend", "shortfall")):
        cues.extend(["deficit", "surplus", "overspend", "shortfall", "variance"])
    if any(tok in q for tok in ("total", "overall total", "sum")):
        cues.extend(["total"])
    if any(tok in q for tok in ("cash requirement",)):
        cues.extend(["cash requirement"])
    out: list[str] = []
    seen: set[str] = set()
    for cue in cues:
        if cue not in seen:
            seen.add(cue)
            out.append(cue)
    return out


def _context_window(text: str, start: int, end: int, width: int = 100) -> str:
    lo = max(0, int(start) - int(width))
    hi = min(len(text), int(end) + int(width))
    return str(text[lo:hi])


def _is_scale_label_only(raw: str) -> bool:
    compact = re.sub(r"\s+", "", str(raw or "").lower())
    return compact in {"£000", "(000)", "(£000)", "000"}


def _attach_nearby_scale(raw: str, context: str) -> str:
    value = str(raw or "").strip()
    if not value:
        return value
    lower_value = value.lower()
    lower_context = str(context or "").lower()
    if "£000" in lower_value or "(£000)" in lower_value or "(000)" in lower_value:
        return value
    if re.search(r"\b(million|millions|billion|billions|thousand|thousands|bn|m|k)\b", lower_value):
        return value
    if "%" in lower_value or "percent" in lower_value:
        return value
    if "£000" in lower_context or "(£000)" in lower_context or "(000)" in lower_context:
        return f"{value} (£000)"
    return value


def _is_year_fragment(text: str, start: int, end: int, raw: str) -> bool:
    candidate = str(raw or "").strip()
    if not candidate:
        return False
    window = str(text[max(0, start - 6): min(len(text), end + 6)])
    compact = re.sub(r"\s+", "", window)
    if re.search(r"(?:19|20)\d{2}/\d{1,2}", compact):
        return True
    if re.search(r"\d{1,2}/(?:19|20)\d{2}", compact):
        return True
    return False


def _candidate_debug(
    *,
    raw: str,
    canonical: Optional[str],
    score: float,
    score_breakdown: dict[str, float],
    context: str,
    chunk_id: str,
    pages: list[int],
    rank: int,
) -> dict[str, Any]:
    return {
        "raw": raw,
        "canonical": canonical,
        "score": round(float(score), 4),
        "score_breakdown": {k: round(float(v), 4) for k, v in score_breakdown.items()},
        "context": context,
        "chunk_id": chunk_id,
        "pages": pages,
        "rank": int(rank),
    }


def pick_best_numeric_candidate(
    *,
    question: str,
    results: list[dict[str, Any]],
) -> tuple[Optional[dict[str, Any]], dict[str, Any]]:
    """Score all numeric expressions in the retrieval results and return (best_candidate, debug_dict)."""
    keywords = _question_keywords(question)
    expected_dimension = _question_expected_dimension(question)
    q_lower = str(question or "").lower()
    cue_terms = _question_cue_terms(question)

    candidates: list[dict[str, Any]] = []
    for result in results:
        text = str(result.get("chunk_text") or "")
        if not text:
            continue
        chunk_id = str(result.get("chunk_id") or "")
        pages = [int(p) for p in (result.get("pages") or []) if str(p).strip().isdigit()]
        rank = int(result.get("rank") or 9999)
        for match in NUMERIC_EXPR_RE.finditer(text):
            raw = str(match.group("expr") or "").strip()
            if not raw or not re.search(r"\d", raw):
                continue
            if _is_year_fragment(text, match.start(), match.end(), raw):
                continue
            context = _context_window(text, match.start(), match.end(), width=120)
            if _is_scale_label_only(raw):
                continue
            raw = _attach_nearby_scale(raw, context)
            norm = normalize_numeric_value(raw)
            if norm is None:
                continue
            context_lower = context.lower()
            overlap = sum(1 for kw in keywords if kw in context_lower)
            dimension = str(norm.get("dimension") or detect_numeric_dimension(raw))
            score_breakdown: dict[str, float] = {
                "retrieval_rank": max(0.0, 3.0 - 0.2 * float(max(rank - 1, 0))),
                "keyword_overlap": 0.9 * float(overlap),
                "currency_hint": 0.0,
                "percent_hint": 0.0,
                "cue_hint": 0.0,
                "row_pattern_boost": 0.0,
                "narrative_penalty": 0.0,
                "structural_penalty": 0.0,
                "table_penalty": -0.15 if bool(result.get("is_table", False)) else 0.0,
                "plain_year_penalty": 0.0,
                "small_value_penalty": 0.0,
            }
            if expected_dimension == dimension:
                if dimension == "currency":
                    score_breakdown["currency_hint"] = 1.5
                elif dimension == "percent":
                    score_breakdown["percent_hint"] = 1.5
                else:
                    score_breakdown["currency_hint"] = 0.25
            elif expected_dimension is not None:
                score_breakdown["currency_hint"] = -1.0
            cue_hits = sum(1 for cue in cue_terms if cue in context_lower)
            if cue_hits:
                score_breakdown["cue_hint"] = 1.1 * float(cue_hits)
            elif any(cue in context_lower for cue in ("deficit", "surplus", "overspend", "shortfall", "budget", "spend", "spending", "income", "cost", "expenditure")):
                score_breakdown["cue_hint"] = 0.6
            # Prefer compact row-like evidence for budget/variance style questions.
            if any(tok in q_lower for tok in ("budget", "spending", "spend", "differ", "difference", "overspend", "underspend")):
                if "variance" in context_lower and any(tok in context_lower for tok in ("surplus", "deficit", "overspend", "underspend")):
                    score_breakdown["row_pattern_boost"] += 2.2
                if any(tok in context_lower for tok in ("memorandum", "in year out-turn", "core revenue resource")):
                    score_breakdown["row_pattern_boost"] += 0.8
                if any(tok in context_lower for tok in ("reported a surplus", "reported a deficit", "retained reserves", "general reserves", "earmarked reserves")):
                    score_breakdown["narrative_penalty"] -= 2.4
                if "underspend against the revenue budget" in context_lower:
                    score_breakdown["narrative_penalty"] -= 1.2
            if (
                not any(tok in q_lower for tok in ("total", "cash requirement", "resource limit"))
                and any(tok in context_lower for tok in ("total", "cash requirement", "resource limit"))
            ):
                score_breakdown["structural_penalty"] = -1.8
            if re.fullmatch(r"(?:19|20)\d{2}", raw.replace(",", "").strip()):
                score_breakdown["plain_year_penalty"] = -2.0
            digits_only = re.sub(r"[^\d]", "", raw)
            if re.search(r"\d{4}\s*/\s*\d{1,2}", context_lower) and digits_only and len(digits_only) <= 2:
                score_breakdown["plain_year_penalty"] = -2.5
            if expected_dimension == "currency":
                try:
                    abs_value = abs(float(norm.get("value") or 0.0))
                except Exception:
                    abs_value = 0.0
                if 0.0 < abs_value < 100000.0:
                    score_breakdown["small_value_penalty"] = -0.8
            score = sum(score_breakdown.values())
            candidates.append(
                {
                    "raw": raw,
                    "normalized": norm,
                    "canonical": None,
                    "score": score,
                    "score_breakdown": score_breakdown,
                    "context": context,
                    "chunk_id": chunk_id,
                    "pages": pages,
                    "rank": rank,
                }
            )

    candidates.sort(key=lambda c: (float(c["score"]), -int(c["rank"])), reverse=True)
    for candidate in candidates:
        norm = candidate.get("normalized") or {}
        dim = str(norm.get("dimension") or "")
        value = float(norm.get("value") or 0.0)
        sign = "-" if value < 0 else ""
        abs_value = abs(value)
        if dim == "percent":
            rendered = f"{int(abs_value)}%" if abs_value.is_integer() else f"{abs_value:.2f}".rstrip("0").rstrip(".") + "%"
            candidate["canonical"] = f"{sign}{rendered}" if sign else rendered
        elif dim == "currency":
            body = f"{int(abs_value):,}" if abs_value.is_integer() else f"{abs_value:,.2f}".rstrip("0").rstrip(".")
            candidate["canonical"] = f"{sign}£{body}"
        else:
            candidate["canonical"] = f"{int(abs_value):,}" if abs_value.is_integer() else f"{abs_value:,.2f}".rstrip("0").rstrip(".")
            if sign:
                candidate["canonical"] = f"{sign}{candidate['canonical']}"

    best = candidates[0] if candidates else None
    debug = {
        "applied": True,
        "candidate_count": len(candidates),
        "expected_dimension": expected_dimension,
        "top_candidates": [
            _candidate_debug(
                raw=str(c["raw"]),
                canonical=str(c.get("canonical") or ""),
                score=float(c["score"]),
                score_breakdown=dict(c["score_breakdown"]),
                context=str(c["context"]),
                chunk_id=str(c["chunk_id"]),
                pages=list(c["pages"]),
                rank=int(c["rank"]),
            )
            for c in candidates[:5]
        ],
    }
    if best is None:
        debug["reason"] = "no_numeric_candidates"
        return None, debug

    best_score = float(best["score"])
    if best_score < 1.5:
        debug["reason"] = "low_confidence"
        debug["selected_score"] = round(best_score, 4)
        return None, debug

    if expected_dimension == "currency" and not re.search(r"£|\b(?:m|bn|million|billion|thousand)\b|\(£?000\)", str(best["raw"]), flags=re.IGNORECASE):
        # Avoid picking bare counts for obviously monetary questions unless context strongly indicates money.
        if "£" not in str(best.get("context") or ""):
            debug["reason"] = "currency_unit_not_supported"
            debug["selected_score"] = round(best_score, 4)
            return None, debug

    debug["selected"] = _candidate_debug(
        raw=str(best["raw"]),
        canonical=str(best.get("canonical") or ""),
        score=best_score,
        score_breakdown=dict(best["score_breakdown"]),
        context=str(best["context"]),
        chunk_id=str(best["chunk_id"]),
        pages=list(best["pages"]),
        rank=int(best["rank"]),
    )
    return best, debug
