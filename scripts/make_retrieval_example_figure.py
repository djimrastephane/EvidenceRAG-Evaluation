from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from pathlib import Path

import _matplotlib_env
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a thesis-ready retrieval example figure.")
    parser.add_argument("--eval-set", required=True, help="Path to eval_set.json")
    parser.add_argument("--query-id", required=True, help="Query ID to render")
    parser.add_argument("--summary-224", required=True, help="Path to 224/56 retrieval_summary.csv")
    parser.add_argument("--summary-256", required=True, help="Path to 256/64 retrieval_summary.csv")
    parser.add_argument("--summary-280", required=True, help="Path to 280/90 retrieval_summary.csv")
    parser.add_argument("--out-png", required=True, help="Output figure PNG")
    parser.add_argument("--document-label", default="NHS Grampian Annual Report 2022-2023")
    parser.add_argument("--query-text", default="")
    parser.add_argument("--caption", default="")
    return parser.parse_args()


def load_query(eval_set_path: Path, query_id: str) -> dict:
    raw = json.loads(eval_set_path.read_text(encoding="utf-8"))
    items = raw if isinstance(raw, list) else raw.get("queries", [])
    for item in items:
        if str(item.get("query_id")) == query_id:
            return item
    raise KeyError(f"Query ID not found in eval set: {query_id}")


def load_summary_row(summary_path: Path, query_id: str, k: int) -> dict:
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if str(row[4]).strip().lower() == "k":
                continue
            if str(row[0]).strip() == query_id and str(row[4]).strip() == str(k):
                return {
                    "failure_type": str(row[14]).strip(),
                    "top_pages": [int(part.strip()) for part in str(row[24]).strip().strip("[]").split(",") if part.strip()],
                }
    raise KeyError(f"Query {query_id} with k={k} not found in {summary_path}")


def rank_of_gold(pages: list[int], gold_page: int) -> int | None:
    for idx, page in enumerate(pages, start=1):
        if page == gold_page:
            return idx
    return None


def reciprocal_rank(rank: int | None) -> float:
    if rank is None:
        return 0.0
    return 1.0 / float(rank)


def add_card(ax, x: float, y: float, w: float, h: float, face: str, edge: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            linewidth=0.9,
            edgecolor=edge,
            facecolor=face,
        )
    )


def draw_top_card(ax, query_id: str, document_label: str, query_text: str, gold_page: int) -> None:
    add_card(ax, 0.0, 0.84, 1.0, 0.16, face="#f7f5ef", edge="#d8d1c2")
    ax.text(0.02, 0.965, f"Query ID: {query_id}", fontsize=11, color="#5a544b", va="top")
    ax.text(0.25, 0.965, f"Source: {document_label}", fontsize=11, color="#5a544b", va="top")
    wrapped = textwrap.fill(query_text, width=80)
    ax.text(0.02, 0.915, f"“{wrapped}”", fontsize=14, color="#2e2a24", fontstyle="italic", va="top")
    ax.text(0.02, 0.862, f"Ground truth: page {gold_page}", fontsize=12, color="#3d8b40", va="top")


def draw_rank_summary(ax, summary_data: list[dict]) -> None:
    add_card(ax, 0.0, 0.70, 1.0, 0.10, face="#f7f5ef", edge="#d8d1c2")
    ax.text(0.02, 0.745, "Ground-truth page rank", fontsize=12, color="#4f4a42", va="center")
    xs = [0.40, 0.52, 0.64]
    labels = ["Config A", "Config B", "Config C"]
    for idx, (x, item, label) in enumerate(zip(xs, summary_data, labels)):
        rr = item["rank"]
        fill = "#b8841e" if rr and rr > 1 else "#4b972d"
        ax.text(
            x,
            0.752,
            "?" if rr is None else str(rr),
            fontsize=12,
            color="white",
            ha="center",
            va="center",
            bbox=dict(boxstyle="circle,pad=0.35", facecolor=fill, edgecolor="none"),
        )
        ax.text(x, 0.715, label, fontsize=10, color="#5a544b", ha="center", va="center")
        if idx < 2:
            ax.text(x + 0.06, 0.748, "->", fontsize=16, color="#7a746b", ha="center", va="center")


def draw_panel(ax, x0: float, w: float, tag: str, title: str, chunk_text: str, item: dict, gold_page: int) -> None:
    add_card(ax, x0, 0.25, w, 0.40, face="#f7f5ef", edge="#d8d1c2")
    ax.text(x0 + 0.02, 0.62, tag, fontsize=11, color="#5a544b")
    ax.text(x0 + 0.02, 0.585, title, fontsize=15, fontweight="bold", color="#2e2a24")
    ax.text(x0 + 0.02, 0.555, chunk_text, fontsize=10.5, color="#5a544b")

    table_top = 0.525
    row_h = 0.048
    col_x = [x0 + 0.02, x0 + 0.10, x0 + 0.22]
    col_labels = ["Rank", "Page", "Match"]
    ax.add_patch(plt.Rectangle((x0, table_top - row_h), w, row_h, facecolor="#ece7da", edgecolor="none"))
    for cx, label in zip(col_x, col_labels):
        ax.text(cx, table_top - row_h / 2, label, fontsize=10.5, color="#4f4a42", va="center", fontweight="bold")

    pages = item["pages"]
    for i in range(5):
        y = table_top - row_h * (i + 2)
        if i < len(pages):
            page = pages[i]
            is_gold = page == gold_page
            if is_gold:
                ax.add_patch(plt.Rectangle((x0, y), w, row_h, facecolor="#e2f0dc", edgecolor="none"))
            ax.text(col_x[0], y + row_h / 2, str(i + 1), fontsize=11, color="#2e2a24", va="center")
            ax.text(col_x[1], y + row_h / 2, str(page), fontsize=11, color="#2e2a24", va="center")
            ax.text(col_x[2], y + row_h / 2, "✓" if is_gold else "−", fontsize=12, color="#3d8b40" if is_gold else "#6d675f", va="center")
        ax.plot([x0, x0 + w], [y, y], color="#ddd6c7", linewidth=0.8)

    ax.text(x0 + 0.02, 0.30, f"Hit@1 = {item['hit1']}", fontsize=11.5, color="#c04b2f" if item["hit1"] == 0 else "#4b972d", fontweight="bold")
    ax.text(x0 + w - 0.02, 0.30, f"RR = {item['rr']:.2f}", fontsize=11.5, color="#4f4a42", ha="right", fontweight="bold")


def draw_bottom_table(ax, summary_data: list[dict]) -> None:
    add_card(ax, 0.0, 0.02, 1.0, 0.18, face="#f7f5ef", edge="#d8d1c2")
    headers = ["Panel", "Chunk (tok)", "Overlap (tok)", "Pages shown", "Rank of p.23", "Hit@1", "RR"]
    xs = [0.02, 0.11, 0.26, 0.45, 0.62, 0.79, 0.90]
    ax.add_patch(plt.Rectangle((0.0, 0.145), 1.0, 0.04, facecolor="#ece7da", edgecolor="none"))
    for x, header in zip(xs, headers):
        ax.text(x, 0.165, header, fontsize=10, color="#4f4a42", va="center", fontweight="bold")

    rows = [
        ("(a)", "224", "56", str(len(summary_data[0]["pages"])), summary_data[0]["rank"], summary_data[0]["hit1"], summary_data[0]["rr"]),
        ("(b)", "256", "64", str(len(summary_data[1]["pages"])), summary_data[1]["rank"], summary_data[1]["hit1"], summary_data[1]["rr"]),
        ("(c)", "280", "90", str(len(summary_data[2]["pages"])), summary_data[2]["rank"], summary_data[2]["hit1"], summary_data[2]["rr"]),
    ]
    y0 = 0.112
    row_h = 0.04
    for idx, row in enumerate(rows):
        y = y0 - idx * row_h
        ax.plot([0.0, 1.0], [y, y], color="#ddd6c7", linewidth=0.8)
        for col_idx, (x, value) in enumerate(zip(xs, row)):
            color = "#2e2a24"
            if col_idx in {4, 5, 6}:
                color = "#4b972d" if value in {1, "1", 1.0} or (isinstance(value, float) and value >= 1.0) else "#b8841e" if col_idx == 4 and value == 2 else "#c04b2f" if value == 0 else "#4f4a42"
            text = f"{value:.2f}" if isinstance(value, float) else str(value)
            ax.text(x, y - row_h / 2 + 0.02, text, fontsize=10.5, color=color, va="center", fontweight="bold" if col_idx >= 4 else None)


def main() -> None:
    args = parse_args()
    query = load_query(Path(args.eval_set), args.query_id)
    gold_page = int(query.get("expected_pages", [None])[0])

    paths = {
        "A": Path(args.summary_224),
        "B": Path(args.summary_256),
        "C": Path(args.summary_280),
    }
    chunk_meta = {
        "A": "chunk = 224 tok  •  overlap = 56 tok",
        "B": "chunk = 256 tok  •  overlap = 64 tok",
        "C": "chunk = 280 tok  •  overlap = 90 tok",
    }

    summary_data = []
    for key in ["A", "B", "C"]:
        row_k1 = load_summary_row(paths[key], args.query_id, 1)
        row_k5 = load_summary_row(paths[key], args.query_id, 5)
        rank = rank_of_gold(row_k5["top_pages"], gold_page)
        summary_data.append(
            {
                "pages": row_k5["top_pages"],
                "rank": rank,
                "hit1": 1 if row_k1["failure_type"] == "hit" else 0,
                "rr": reciprocal_rank(rank),
            }
        )

    query_text = args.query_text.strip() or str(query.get("question", "")).strip()
    caption = args.caption.strip()

    fig = plt.figure(figsize=(14, 9.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#fcfbf7")

    draw_top_card(ax, args.query_id, args.document_label, query_text, gold_page)
    draw_rank_summary(ax, summary_data)

    panel_w = 0.32
    panel_x = [0.0, 0.34, 0.68]
    tags = ["(a)", "(b)", "(c)"]
    titles = ["Configuration A", "Configuration B", "Configuration C"]
    for idx in range(3):
        draw_panel(ax, panel_x[idx], panel_w, tags[idx], titles[idx], chunk_meta[["A", "B", "C"][idx]], summary_data[idx], gold_page)

    draw_bottom_table(ax, summary_data)

    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
