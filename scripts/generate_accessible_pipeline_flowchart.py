from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "figures"
PNG_PATH = OUT_DIR / "retrieval_pipeline_accessible_bw.png"
SVG_PATH = OUT_DIR / "retrieval_pipeline_accessible_bw.svg"

WIDTH = 1800
HEIGHT = 1120
BG = "#ffffff"
TEXT = "#111111"
LINE = "#333333"
PANEL_FILL = "#f2f2f2"
GROUP_FILL = "#e8e8e8"
BOX_FILL = "#fcfcfc"
EMPH_FILL = "#ece2b6"
ALERT_FILL = "#e2d2c8"
SIDE_FILL = "#e7e1ef"


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/Library/Fonts/Arial Bold.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/Library/Fonts/Arial.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]
        )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


TITLE_FONT = get_font(38, bold=True)
HEADER_FONT = get_font(26, bold=True)
BOX_FONT = get_font(20, bold=True)
SMALL_FONT = get_font(16)
TINY_FONT = get_font(13)
CAPTION_FONT = get_font(24, bold=True)
CAPTION_TEXT_FONT = get_font(22)


def draw_centered_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font, fill=TEXT):
    x0, y0, x1, y1 = box
    lines = text.split("\n")
    bboxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    line_heights = [bbox[3] - bbox[1] for bbox in bboxes]
    total_h = sum(line_heights) + max(0, len(lines) - 1) * 4
    y = y0 + ((y1 - y0) - total_h) / 2
    for line, bbox, h in zip(lines, bboxes, line_heights):
        w = bbox[2] - bbox[0]
        x = x0 + ((x1 - x0) - w) / 2
        draw.text((x, y), line, font=font, fill=fill)
        y += h + 4


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font,
    max_width: int,
    fill=TEXT,
    line_gap: int = 3,
):
    x, y = xy
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((0, 0), line, font=font)
        y += (bbox[3] - bbox[1]) + line_gap


def dashed_rounded_rect(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], radius: int, dash: int = 10):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=radius, outline=LINE, width=2, fill=PANEL_FILL)
    for x in range(x0 + 20, x1 - 20, dash * 2):
        draw.line((x, y0, min(x + dash, x1 - 20), y0), fill=LINE, width=2)
        draw.line((x, y1, min(x + dash, x1 - 20), y1), fill=LINE, width=2)
    for y in range(y0 + 20, y1 - 20, dash * 2):
        draw.line((x0, y, x0, min(y + dash, y1 - 20)), fill=LINE, width=2)
        draw.line((x1, y, x1, min(y + dash, y1 - 20)), fill=LINE, width=2)


def hatched_rect(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, spacing: int = 12):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=14, fill=fill, outline=LINE, width=3)
    start = x0 - (y1 - y0)
    end = x1 + (y1 - y0)
    for x in range(start, end, spacing):
        draw.line((x, y1, x + (y1 - y0), y0), fill="#8c8c8c", width=1)


def standard_box(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], label: str, sublabel: str | None = None):
    draw.rounded_rectangle(box, radius=14, fill=BOX_FILL, outline="#7a7a7a", width=2)
    x0, y0, x1, y1 = box
    if sublabel:
        draw_centered_text(draw, (x0, y0 + 6, x1, y1 - 12), f"{label}\n{sublabel}", BOX_FONT)
    else:
        draw_centered_text(draw, box, label, BOX_FONT)


def arrow(draw: ImageDraw.ImageDraw, pts: Iterable[tuple[int, int]], width: int = 3):
    pts = list(pts)
    draw.line(pts, fill=LINE, width=width)
    x1, y1 = pts[-2]
    x2, y2 = pts[-1]
    if abs(x2 - x1) > abs(y2 - y1):
        direction = 1 if x2 > x1 else -1
        tip = [(x2, y2), (x2 - 12 * direction, y2 - 6), (x2 - 12 * direction, y2 + 6)]
    else:
        direction = 1 if y2 > y1 else -1
        tip = [(x2, y2), (x2 - 6, y2 - 12 * direction), (x2 + 6, y2 - 12 * direction)]
    draw.polygon(tip, fill=LINE)


def build_png() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)

    draw.text((130, 55), "Offline Indexing Phase", font=HEADER_FONT, fill=TEXT)
    draw.text((1170, 55), "Online Query Phase", font=HEADER_FONT, fill=TEXT)

    offline = (55, 95, 475, 735)
    online = (920, 95, 1640, 955)
    dashed_rounded_rect(draw, offline, radius=24)
    dashed_rounded_rect(draw, online, radius=24)

    # Offline boxes
    off_boxes = [
        (95, 125, 435, 180, "PDF Documents", None),
        (120, 225, 410, 275, "Text Extraction", None),
        (120, 335, 410, 405, "Cleaning / Normalisation", "header/footer, OCR fallback, layout"),
        (95, 450, 435, 525, "Chunking", None),
        (120, 580, 410, 635, "Chunk Embedding", None),
        (95, 680, 435, 775, "FAISS Vector Index", None),
    ]
    for x0, y0, x1, y1, label, sublabel in off_boxes:
        standard_box(draw, (x0, y0, x1, y1), label, sublabel)
    for upper, lower in zip(off_boxes, off_boxes[1:]):
        cx = (upper[0] + upper[2]) // 2
        arrow(draw, [(cx, upper[3]), (cx, lower[1])])

    # Bridge box
    side = (535, 500, 870, 675)
    draw.rounded_rectangle(side, radius=18, fill=SIDE_FILL, outline="#7b6f8d", width=2)
    bm25_build = (565, 535, 825, 590)
    bm25_index = (565, 635, 825, 690)
    standard_box(draw, bm25_build, "BM25 Build (on load)")
    standard_box(draw, bm25_index, "BM25 Index", "(in-memory, cached)")
    draw.text((635, 460), "built on document load", font=TINY_FONT, fill="#666666")
    draw.text((455, 560), "chunk text", font=TINY_FONT, fill="#666666")
    draw.text((455, 610), "dense vectors", font=TINY_FONT, fill="#666666")

    # Online sections
    query_group = (935, 110, 1625, 320)
    retrieval_group = (935, 340, 1625, 555)
    generation_group = (1040, 565, 1585, 930)
    for group, fill in [(query_group, "#ebebeb"), (retrieval_group, GROUP_FILL), (generation_group, "#e6e6e6")]:
        draw.rounded_rectangle(group, radius=18, fill=fill, outline=None)

    query_box = (1040, 130, 1535, 190)
    standard_box(draw, query_box, "User Query")
    query_embed = (965, 235, 1245, 290)
    bm25_score = (1300, 235, 1575, 290)
    standard_box(draw, query_embed, "Query Embedding", "term scores")
    standard_box(draw, bm25_score, "BM25 Query Scoring")

    dense_ret = (970, 370, 1255, 445)
    bm25_ret = (1310, 370, 1585, 445)
    standard_box(draw, dense_ret, "Dense Retrieval", "(FAISS lookup)")
    standard_box(draw, bm25_ret, "BM25 Retrieval")

    rrf = (1085, 475, 1545, 540)
    hatched_rect(draw, rrf, EMPH_FILL)
    draw_centered_text(draw, rrf, "RRF Fusion", BOX_FONT)

    topk = (1100, 590, 1525, 645)
    llm = (1080, 705, 1545, 780)
    cite_gate = (1100, 815, 1525, 870)
    final = (1080, 905, 1545, 980)
    standard_box(draw, topk, "Top-k Chunks + Page IDs")
    standard_box(draw, llm, "LLM JSON Answer + Citations")
    hatched_rect(draw, cite_gate, ALERT_FILL)
    draw_centered_text(draw, cite_gate, "Citation Validation Gate", BOX_FONT)
    standard_box(draw, final, "Final Grounded Answer")

    # Online arrows
    qcx = (query_box[0] + query_box[2]) // 2
    arrow(draw, [(qcx, query_box[3]), (qcx, 210), ((query_embed[0] + query_embed[2]) // 2, 210), ((query_embed[0] + query_embed[2]) // 2, query_embed[1])])
    arrow(draw, [(qcx, query_box[3]), (qcx, 210), ((bm25_score[0] + bm25_score[2]) // 2, 210), ((bm25_score[0] + bm25_score[2]) // 2, bm25_score[1])])
    arrow(draw, [((query_embed[0] + query_embed[2]) // 2, query_embed[3]), ((query_embed[0] + query_embed[2]) // 2, dense_ret[1])])
    arrow(draw, [((bm25_score[0] + bm25_score[2]) // 2, bm25_score[3]), ((bm25_score[0] + bm25_score[2]) // 2, bm25_ret[1])])
    arrow(draw, [((bm25_score[0] + bm25_score[2]) // 2, bm25_score[3] - 5), ((bm25_score[0] + bm25_score[2]) // 2 - 35, bm25_ret[1])], width=2)
    arrow(draw, [((dense_ret[0] + dense_ret[2]) // 2, dense_ret[3]), ((dense_ret[0] + dense_ret[2]) // 2, rrf[1])])
    arrow(draw, [((bm25_ret[0] + bm25_ret[2]) // 2, bm25_ret[3]), ((bm25_ret[0] + bm25_ret[2]) // 2, rrf[1])])
    center_x = (rrf[0] + rrf[2]) // 2
    arrow(draw, [(center_x, rrf[3]), (center_x, topk[1])])
    arrow(draw, [((topk[0] + topk[2]) // 2, topk[3]), ((topk[0] + topk[2]) // 2, llm[1])])
    arrow(draw, [((llm[0] + llm[2]) // 2, llm[3]), ((llm[0] + llm[2]) // 2, cite_gate[1])])
    arrow(draw, [((cite_gate[0] + cite_gate[2]) // 2, cite_gate[3]), ((cite_gate[0] + cite_gate[2]) // 2, final[1])])

    # Cross-panel arrows and labels
    arrow(draw, [(435, 488), (520, 488), (520, 562), (565, 562)])
    arrow(draw, [(435, 728), (500, 728), (500, 628), (970, 628), (970, 405)])
    arrow(draw, [(825, 662), (920, 662), (920, 408), (1310, 408)])
    draw.text((886, 180), "Query Prep", font=TINY_FONT, fill="#666666")
    draw.text((878, 468), "Retrieval", font=TINY_FONT, fill="#666666")
    draw.text((1000, 760), "Generation", font=TINY_FONT, fill="#666666")
    draw_centered_text(
        draw,
        (1555, 815, 1715, 875),
        "insufficient evidence\nor no valid citation",
        TINY_FONT,
        fill="#666666",
    )

    # Side labels for non-color cues
    draw.text((80, 790), "Standard boxes: plain fill", font=SMALL_FONT, fill="#444444")
    draw.text((80, 820), "Critical checks: hatched fill + thicker border", font=SMALL_FONT, fill="#444444")

    # Caption
    draw.text((300, 1030), "Figure 3.1:", font=CAPTION_FONT, fill=TEXT)
    draw.text((510, 1030), "Overview of the retrieval pipeline used in this study.", font=CAPTION_TEXT_FONT, fill=TEXT)

    image.save(PNG_PATH)


def build_svg() -> None:
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
  <defs>
    <pattern id="hatch" width="12" height="12" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
      <line x1="0" y1="0" x2="0" y2="12" stroke="#8c8c8c" stroke-width="2"/>
    </pattern>
    <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
      <polygon points="0,0 12,6 0,12" fill="{LINE}"/>
    </marker>
    <style>
      .title {{ fill: {TEXT}; font-family: Arial, Helvetica, sans-serif; font-size: 38px; font-weight: 700; }}
      .header {{ fill: {TEXT}; font-family: Arial, Helvetica, sans-serif; font-size: 26px; font-weight: 700; }}
      .boxtext {{ fill: {TEXT}; font-family: Arial, Helvetica, sans-serif; font-size: 20px; font-weight: 700; }}
      .small {{ fill: #555555; font-family: Arial, Helvetica, sans-serif; font-size: 16px; }}
      .tiny {{ fill: #666666; font-family: Arial, Helvetica, sans-serif; font-size: 13px; }}
      .panel {{ fill: {PANEL_FILL}; stroke: {LINE}; stroke-width: 2; stroke-dasharray: 10 8; }}
      .group {{ fill: {GROUP_FILL}; stroke: none; }}
      .plain {{ fill: {BOX_FILL}; stroke: #7a7a7a; stroke-width: 2; }}
      .rrf {{ fill: {EMPH_FILL}; stroke: {LINE}; stroke-width: 3; }}
      .gate {{ fill: {ALERT_FILL}; stroke: {LINE}; stroke-width: 3; }}
      .wire {{ fill: none; stroke: {LINE}; stroke-width: 3; marker-end: url(#arrow); }}
    </style>
  </defs>
  <rect width="{WIDTH}" height="{HEIGHT}" fill="{BG}"/>
  <text class="header" x="130" y="55">Offline Indexing Phase</text>
  <text class="header" x="1170" y="55">Online Query Phase</text>

  <rect class="panel" x="55" y="95" width="420" height="640" rx="24"/>
  <rect class="panel" x="920" y="95" width="720" height="860" rx="24"/>

  <rect class="plain" x="95" y="125" width="340" height="55" rx="14"/>
  <text class="boxtext" x="265" y="158" text-anchor="middle">PDF Documents</text>
  <rect class="plain" x="120" y="225" width="290" height="50" rx="14"/>
  <text class="boxtext" x="265" y="255" text-anchor="middle">Text Extraction</text>
  <rect class="plain" x="120" y="335" width="290" height="70" rx="14"/>
  <text class="boxtext" x="265" y="364" text-anchor="middle">Cleaning / Normalisation</text>
  <text class="tiny" x="265" y="384" text-anchor="middle">header/footer, OCR fallback, layout</text>
  <rect class="plain" x="95" y="450" width="340" height="75" rx="14"/>
  <text class="boxtext" x="265" y="495" text-anchor="middle">Chunking</text>
  <rect class="plain" x="120" y="580" width="290" height="55" rx="14"/>
  <text class="boxtext" x="265" y="613" text-anchor="middle">Chunk Embedding</text>
  <rect class="plain" x="95" y="680" width="340" height="95" rx="14"/>
  <text class="boxtext" x="265" y="736" text-anchor="middle">FAISS Vector Index</text>

  <path class="wire" d="M265 180 L265 225"/>
  <path class="wire" d="M265 275 L265 335"/>
  <path class="wire" d="M265 405 L265 450"/>
  <path class="wire" d="M265 525 L265 580"/>
  <path class="wire" d="M265 635 L265 680"/>

  <rect x="535" y="500" width="335" height="175" rx="18" fill="{SIDE_FILL}" stroke="#7b6f8d" stroke-width="2"/>
  <rect class="plain" x="565" y="535" width="260" height="55" rx="14"/>
  <text class="boxtext" x="695" y="568" text-anchor="middle">BM25 Build (on load)</text>
  <rect class="plain" x="565" y="635" width="260" height="55" rx="14"/>
  <text class="boxtext" x="695" y="661" text-anchor="middle">BM25 Index</text>
  <text class="tiny" x="695" y="679" text-anchor="middle">(in-memory, cached)</text>
  <text class="tiny" x="635" y="460">built on document load</text>
  <text class="tiny" x="455" y="560">chunk text</text>
  <text class="tiny" x="455" y="610">dense vectors</text>

  <rect class="group" x="935" y="110" width="690" height="210" rx="18"/>
  <rect class="group" x="935" y="340" width="690" height="215" rx="18"/>
  <rect class="group" x="1040" y="565" width="545" height="365" rx="18"/>

  <rect class="plain" x="1040" y="130" width="495" height="60" rx="14"/>
  <text class="boxtext" x="1287" y="166" text-anchor="middle">User Query</text>
  <rect class="plain" x="965" y="235" width="280" height="55" rx="14"/>
  <text class="boxtext" x="1105" y="262" text-anchor="middle">Query Embedding</text>
  <text class="tiny" x="1105" y="280" text-anchor="middle">term scores</text>
  <rect class="plain" x="1300" y="235" width="275" height="55" rx="14"/>
  <text class="boxtext" x="1437" y="267" text-anchor="middle">BM25 Query Scoring</text>

  <rect class="plain" x="970" y="370" width="285" height="75" rx="14"/>
  <text class="boxtext" x="1112" y="398" text-anchor="middle">Dense Retrieval</text>
  <text class="tiny" x="1112" y="418" text-anchor="middle">(FAISS lookup)</text>
  <rect class="plain" x="1310" y="370" width="275" height="75" rx="14"/>
  <text class="boxtext" x="1447" y="415" text-anchor="middle">BM25 Retrieval</text>

  <rect class="rrf" x="1085" y="475" width="460" height="65" rx="14"/>
  <rect x="1085" y="475" width="460" height="65" rx="14" fill="url(#hatch)" opacity="0.35"/>
  <text class="boxtext" x="1315" y="514" text-anchor="middle">RRF Fusion</text>

  <rect class="plain" x="1100" y="590" width="425" height="55" rx="14"/>
  <text class="boxtext" x="1312" y="623" text-anchor="middle">Top-k Chunks + Page IDs</text>
  <rect class="plain" x="1080" y="705" width="465" height="75" rx="14"/>
  <text class="boxtext" x="1312" y="748" text-anchor="middle">LLM JSON Answer + Citations</text>
  <rect class="gate" x="1100" y="815" width="425" height="55" rx="14"/>
  <rect x="1100" y="815" width="425" height="55" rx="14" fill="url(#hatch)" opacity="0.35"/>
  <text class="boxtext" x="1312" y="848" text-anchor="middle">Citation Validation Gate</text>
  <rect class="plain" x="1080" y="905" width="465" height="75" rx="14"/>
  <text class="boxtext" x="1312" y="948" text-anchor="middle">Final Grounded Answer</text>

  <path class="wire" d="M1287 190 L1287 210 L1105 210 L1105 235"/>
  <path class="wire" d="M1287 190 L1287 210 L1437 210 L1437 235"/>
  <path class="wire" d="M1105 290 L1105 370"/>
  <path class="wire" d="M1437 290 L1437 370"/>
  <path class="wire" d="M1402 285 L1367 370"/>
  <path class="wire" d="M1112 445 L1112 475"/>
  <path class="wire" d="M1447 445 L1447 475"/>
  <path class="wire" d="M1315 540 L1315 590"/>
  <path class="wire" d="M1312 645 L1312 705"/>
  <path class="wire" d="M1312 780 L1312 815"/>
  <path class="wire" d="M1312 870 L1312 905"/>

  <path class="wire" d="M435 488 L520 488 L520 562 L565 562"/>
  <path class="wire" d="M435 728 L500 728 L500 628 L970 628 L970 405"/>
  <path class="wire" d="M825 662 L920 662 L920 408 L1310 408"/>

  <text class="tiny" x="886" y="180">Query Prep</text>
  <text class="tiny" x="878" y="468">Retrieval</text>
  <text class="tiny" x="1000" y="760">Generation</text>
  <text class="tiny" x="1635" y="838" text-anchor="middle">insufficient evidence</text>
  <text class="tiny" x="1635" y="856" text-anchor="middle">or no valid citation</text>

  <text class="small" x="80" y="790">Standard boxes: plain fill</text>
  <text class="small" x="80" y="820">Critical checks: hatched fill + thicker border</text>

  <text class="title" x="300" y="1030">Figure 3.1:</text>
  <text class="title" x="510" y="1030" font-size="22">Overview of the retrieval pipeline used in this study.</text>
</svg>
"""
    SVG_PATH.write_text(svg, encoding="utf-8")


def main() -> None:
    build_png()
    build_svg()
    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {SVG_PATH}")


if __name__ == "__main__":
    main()
