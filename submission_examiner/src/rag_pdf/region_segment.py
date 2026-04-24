"""Split a PDF page into spatially coherent text regions based on line-height gaps."""

from __future__ import annotations

from dataclasses import dataclass

from rag_pdf.config import DEFAULT_CONFIG

REGION_CFG = DEFAULT_CONFIG.REGION


@dataclass
class PageRegion:
    page: int
    region_id: str
    region_index: int
    x0: float
    y0: float
    x1: float
    y1: float
    width: float
    height: float
    line_count: int
    text: str
    lines: list[dict]


def _line_height(line: dict) -> float:
    try:
        y0 = float(line.get("y0", 0.0) or 0.0)
        y1 = float(line.get("y1", 0.0) or 0.0)
    except Exception:
        return 0.0
    return max(0.0, y1 - y0)


def _median(values: list[float], default: float) -> float:
    vals = sorted(v for v in values if v > 0)
    if not vals:
        return default
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def _build_region(page_no: int, region_index: int, lines: list[dict]) -> PageRegion:
    xs0 = [float(ln.get("x0", 0.0) or 0.0) for ln in lines]
    xs1 = [float(ln.get("x1", 0.0) or 0.0) for ln in lines]
    ys0 = [float(ln.get("y0", 0.0) or 0.0) for ln in lines]
    ys1 = [float(ln.get("y1", 0.0) or 0.0) for ln in lines]
    x0 = min(xs0) if xs0 else 0.0
    x1 = max(xs1) if xs1 else 0.0
    y0 = min(ys0) if ys0 else 0.0
    y1 = max(ys1) if ys1 else 0.0
    texts = [str(ln.get("text", "")).strip() for ln in lines if str(ln.get("text", "")).strip()]
    return PageRegion(
        page=page_no,
        region_id=f"p{page_no:04d}_r{region_index:02d}",
        region_index=region_index,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        width=max(0.0, x1 - x0),
        height=max(0.0, y1 - y0),
        line_count=len(lines),
        text="\n".join(texts).strip(),
        lines=lines,
    )


def segment_page_into_regions(
    *,
    page_no: int,
    lines_all: list[dict],
) -> list[PageRegion]:
    """Group page text lines into vertically contiguous regions by detecting whitespace gaps larger than the median line height."""
    cfg = REGION_CFG
    if not lines_all:
        return []
    lines = sorted(lines_all, key=lambda ln: (float(ln.get("y0", 0.0) or 0.0), float(ln.get("x0", 0.0) or 0.0)))
    heights = [_line_height(ln) for ln in lines]
    median_height = max(_median(heights, default=10.0), 1.0)
    gap_threshold = max(cfg.MIN_REGION_HEIGHT, median_height * float(cfg.Y_GAP_MULTIPLIER))

    groups: list[list[dict]] = []
    current: list[dict] = []
    prev_y1 = None
    for line in lines:
        y0 = float(line.get("y0", 0.0) or 0.0)
        y1 = float(line.get("y1", 0.0) or 0.0)
        if not current:
            current = [line]
            prev_y1 = y1
            continue
        gap = y0 - float(prev_y1 or y0)
        if gap > gap_threshold:
            groups.append(current)
            current = [line]
        else:
            current.append(line)
        prev_y1 = y1
    if current:
        groups.append(current)

    merged: list[list[dict]] = []
    for group in groups:
        if not merged:
            merged.append(group)
            continue
        prev = merged[-1]
        prev_y1 = max(float(ln.get("y1", 0.0) or 0.0) for ln in prev)
        cur_y0 = min(float(ln.get("y0", 0.0) or 0.0) for ln in group)
        if len(group) < int(cfg.MIN_LINES_PER_REGION) and (cur_y0 - prev_y1) <= float(cfg.MERGE_GAP_TOLERANCE):
            prev.extend(group)
        else:
            merged.append(group)

    regions: list[PageRegion] = []
    for idx, group in enumerate(merged):
        region = _build_region(page_no, idx, group)
        if region.line_count < int(cfg.MIN_LINES_PER_REGION) and regions:
            regions[-1].lines.extend(region.lines)
            merged_region = _build_region(page_no, regions[-1].region_index, regions[-1].lines)
            regions[-1] = merged_region
            continue
        regions.append(region)
    return regions
