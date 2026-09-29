"""Hand-checked cases for answer scoring and FP attribution (audit findings F06, T4–T6)."""

from __future__ import annotations

import pytest

from rag_pdf.answer_scoring import (
    answer_in_context,
    categorize_failure_type,
    parse_numbers,
    score_answer_correctness,
)


@pytest.mark.parametrize(
    "expected, got, status",
    [
        # Materially different numbers must not match (previously all "correct").
        ("12%", "13%", "incorrect"),
        ("3", "4", "incorrect"),
        ("25.0%", "25.9%", "incorrect"),
        ("-£3.2m", "£3.2m", "incorrect"),
        ("£1,278,771 (£000)", "£1.29 billion", "incorrect"),
        ("£1,278,771 (£000)", "£1 billion", "incorrect"),
        ("150", "153", "incorrect"),
        # Equivalent forms must match.
        ("£5 million", "£5,000,000", "correct"),
        ("10.0", "10", "correct"),
        ("£1,278,771 (£000)", "£1,278,771k", "correct"),
        ("£1,278,771 (£000)", "£1.28 billion", "correct"),
        ("£6.7 million", "£6,700,000", "correct"),
        ("45%", "In 2024/25 the rate was 45%", "correct"),
        ("£(3.2)m", "-£3.2m", "correct"),
        ("(£3.2m)", "−£3.2 million", "correct"),
        ("Underspend of £769,000", "an underspend of £0.769m", "correct"),
        ("£769 (£000) surplus", "Surplus: £0.769m", "correct"),
        ("17,700 directly employed staff", "about 17,700 staff", "correct"),
        # Primary value is the first gold number; breakdowns are context.
        ("£8.6 million (£3.2m on COVID-19 related improvements)", "£8.6m", "correct"),
        ("£8.6 million (£3.2m on COVID-19 related improvements)", "£3.2m", "partial"),
        # Unit omitted (bare £000 table value) is only partially correct.
        ("£1,278,771 (£000)", "1,278,771", "partial"),
        # Fraction vs percent is not silently equated.
        ("25%", "0.25", "incorrect"),
    ],
)
def test_numeric_scoring(expected: str, got: str, status: str) -> None:
    assert score_answer_correctness(expected, "number", got)[1] == status


def test_years_and_codes_are_not_values() -> None:
    values = [m.value for m in parse_numbers("In 2024/25 COVID-19 costs in Q3 were £4.1m, up 5% since 2019")]
    assert values == pytest.approx([4.1e6, 5.0])


def test_breakdown_parenthesis_is_not_negative() -> None:
    assert parse_numbers("£8.6 million (£3.2m on repairs)")[1].value == 3.2e6


def test_empty_gold_is_unscored_not_hit() -> None:
    assert score_answer_correctness("", "text", "anything") == (None, "not_scored")
    assert categorize_failure_type(1, True, "", "text", "ctx", "anything") == "UNSCORED"


def test_answer_in_context_ignores_digit_fragments() -> None:
    # Previously True: comma-split tokens like "1" matched almost any text.
    assert not answer_in_context("£1,278,771 (£000)", "Section 1 overview of 278 posts", "number")
    assert answer_in_context("£1,278,771 (£000)", "Core RRL (£000) 1,278,771 1,278,002", "number")


@pytest.mark.parametrize(
    "expected, got, status",
    [
        ("Audit and Risk Committee", "The Audit and Risk Committee", "correct"),
        ("Audit and Risk Committee", "The Staff Governance Committee", "incorrect"),
        ("Revenue Resource Limit; Capital Resource Limit; Cash Requirement",
         "Revenue Resource Limit and Capital Resource Limit", "partial"),
        # Single-digit tokens count: "Stage 3" is not matched by any text mentioning "stage".
        ("Stage 3", "The board moved to stage 4 escalation", "incorrect"),
        ("Stage 3", "G-OPES remained at Stage 3 throughout", "correct"),
    ],
)
def test_text_scoring(expected: str, got: str, status: str) -> None:
    assert score_answer_correctness(expected, "text", got)[1] == status


def test_failure_attribution_order() -> None:
    ctx = "The Core Revenue Resource Limit was £1,278,771 (£000)."
    exp = "£1,278,771 (£000)"
    assert categorize_failure_type(0, False, exp, "number", ctx, "x") == "FP1_MISSING_CONTENT"
    assert categorize_failure_type(0, True, exp, "number", ctx, "x") == "FP2_MISSED_TOP_RANK"
    assert categorize_failure_type(1, True, exp, "number", "unrelated text", "x") == "FP3_NOT_IN_CONTEXT"
    assert categorize_failure_type(1, True, exp, "number", ctx, "") == "FP4_NOT_EXTRACTED"
    assert categorize_failure_type(1, True, exp, "number", ctx, "no figure given") == "FP5_WRONG_FORMAT"
    assert categorize_failure_type(1, True, exp, "number", ctx, "£1,300,000k") == "FP6_INCORRECT_SPECIFICITY"
    assert categorize_failure_type(1, True, exp, "number", ctx, "1,278,771") == "FP7_INCOMPLETE"
    assert categorize_failure_type(1, True, exp, "number", ctx, "£1,278,771k") == "HIT"
