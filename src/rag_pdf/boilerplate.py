from __future__ import annotations

from collections import Counter

from rag_pdf.config import DEFAULT_CONFIG
from rag_pdf.text_normalize import normalize_line

TOP_STRIP_FRAC = DEFAULT_CONFIG.TOP_STRIP_FRAC
BOTTOM_STRIP_FRAC = DEFAULT_CONFIG.BOTTOM_STRIP_FRAC
LEFT_STRIP_FRAC = DEFAULT_CONFIG.LEFT_STRIP_FRAC
RIGHT_STRIP_FRAC = DEFAULT_CONFIG.RIGHT_STRIP_FRAC

HEADER_FOOTER_REPEAT_FRAC = DEFAULT_CONFIG.HEADER_FOOTER_REPEAT_FRAC
TOP_LINE_K = DEFAULT_CONFIG.TOP_LINE_K
BOT_LINE_K = DEFAULT_CONFIG.BOT_LINE_K


def strip_by_coordinates(
    lines_all: list[dict],
    *,
    page_height: float,
    page_width: float,
    rotation: int,
) -> tuple[list[str], list[str], list[str]]:
    """
    Strip boilerplate using page coordinates (orientation-aware).

    Portrait pages: Strip top and bottom
    Rotated/landscape pages: Strip left and right

    Returns:
        (kept_lines, removed_primary, removed_secondary)
    """
    rot = rotation % 360
    is_rotated = rot in (90, 270)
    is_landscape = page_width / max(page_height, 1.0) > 1.2
    use_side_strips = is_rotated or is_landscape

    kept: list[str] = []
    removed_a: list[str] = []
    removed_b: list[str] = []

    if use_side_strips:
        left_x = page_width * LEFT_STRIP_FRAC
        right_x = page_width * (1.0 - RIGHT_STRIP_FRAC)

        for ln in lines_all:
            x_mid = (ln["x0"] + ln["x1"]) / 2.0
            txt = ln["text"]
            if x_mid <= left_x:
                removed_a.append(txt)
                continue
            if x_mid >= right_x:
                removed_b.append(txt)
                continue
            kept.append(txt)

        return kept, removed_a, removed_b

    top_y = page_height * TOP_STRIP_FRAC
    bot_y = page_height * (1.0 - BOTTOM_STRIP_FRAC)

    for ln in lines_all:
        y_mid = (ln["y0"] + ln["y1"]) / 2.0
        txt = ln["text"]
        if y_mid <= top_y:
            removed_a.append(txt)
            continue
        if y_mid >= bot_y:
            removed_b.append(txt)
            continue
        kept.append(txt)

    return kept, removed_a, removed_b


def remove_repeated_header_footer_lines(
    pages_text_lines: dict[int, list[str]]
) -> tuple[dict[int, list[str]], set[str], set[str]]:
    """
    Remove repeated header/footer lines across pages.

    Lines appearing in top/bottom K positions on ≥40% of pages
    are considered boilerplate and removed.
    """
    top_lines: list[str] = []
    bot_lines: list[str] = []

    for _, ls in pages_text_lines.items():
        norm = [normalize_line(x) for x in ls if normalize_line(x)]
        top_lines.extend(norm[:TOP_LINE_K])
        bot_lines.extend(norm[-BOT_LINE_K:])

    top_counts = Counter(top_lines)
    bot_counts = Counter(bot_lines)
    threshold = int(HEADER_FOOTER_REPEAT_FRAC * len(pages_text_lines))

    common_header = {l for l, c in top_counts.items() if c >= threshold}
    common_footer = {l for l, c in bot_counts.items() if c >= threshold}

    cleaned: dict[int, list[str]] = {}
    for pno, ls in pages_text_lines.items():
        norm = [normalize_line(x) for x in ls if normalize_line(x)]
        out: list[str] = []
        for i, l in enumerate(norm):
            if i < TOP_LINE_K and l in common_header:
                continue
            if i >= len(norm) - BOT_LINE_K and l in common_footer:
                continue
            out.append(l)
        cleaned[pno] = out

    return cleaned, common_header, common_footer
