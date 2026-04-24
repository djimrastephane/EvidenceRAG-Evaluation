from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path

import _matplotlib_env
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a balanced two-example retrieval figure.")
    p.add_argument("--eval-set", required=True)
    p.add_argument("--summary-224", required=True)
    p.add_argument("--summary-256", required=True)
    p.add_argument("--summary-280", required=True)
    p.add_argument("--query-a", required=True)
    p.add_argument("--query-b", required=True)
    p.add_argument("--out-png", required=True)
    p.add_argument("--document-label", default="NHS Grampian Annual Report 2022-2023")
    return p.parse_args()


def load_eval_query(eval_path: Path, query_id: str) -> dict:
    raw = json.loads(eval_path.read_text(encoding="utf-8"))
    items = raw if isinstance(raw, list) else raw.get("queries", [])
    for item in items:
        if str(item.get("query_id")) == query_id:
            return item
    raise KeyError(query_id)


def load_summary_rows(summary_path: Path, query_id: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if str(row[4]).strip().lower() == "k":
                continue
            if str(row[0]).strip() != query_id:
                continue
            k = int(row[4])
            out[k] = {
                "failure_type": str(row[14]).strip(),
                "top_pages": [int(part.strip()) for part in str(row[24]).strip().strip("[]").split(",") if part.strip()],
            }
    return out


def rank_of_gold(pages: list[int], gold_page: int) -> int | None:
    for idx, page in enumerate(pages, start=1):
        if page == gold_page:
            return idx
    return None


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / float(rank)


def add_card(ax, x: float, y: float, w: float, h: float, face: str = "#f7f5ef", edge: str = "#d8d1c2") -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            linewidth=0.9,
            edgecolor=edge,
            facecolor=face,
        )
    )


def draw_summary_badge(ax, x: float, y: float, rank: int | None, label: str) -> None:
    fill = "#6cae3e" if rank == 1 else "#c09028" if rank and rank > 1 else "#b84a3a"
    text = "?" if rank is None else str(rank)
    ax.text(
        x,
        y,
        text,
        fontsize=11,
        color="white",
        ha="center",
        va="center",
        bbox=dict(boxstyle="circle,pad=0.35", facecolor=fill, edgecolor="none"),
    )
    ax.text(x, y - 0.03, label, fontsize=8.5, color="#5a544b", ha="center")


def draw_config_panel(ax, x0: float, y0: float, w: float, h: float, title: str, chunk_text: str, pages: list[int], gold_page: int, hit1: int) -> None:
    add_card(ax, x0, y0, w, h)
    ax.text(x0 + 0.015, y0 + h - 0.04, title, fontsize=12, fontweight="bold", color="#2f2a24")
    ax.text(x0 + 0.015, y0 + h - 0.064, chunk_text, fontsize=9.5, color="#5a544b")

    header_y = y0 + h - 0.10
    row_h = 0.036
    ax.add_patch(plt.Rectangle((x0, header_y - row_h), w, row_h, facecolor="#ece7da", edgecolor="none"))
    ax.text(x0 + 0.015, header_y - row_h / 2, "Rank", fontsize=9.2, fontweight="bold", color="#4f4a42", va="center")
    ax.text(x0 + 0.09, header_y - row_h / 2, "Page", fontsize=9.2, fontweight="bold", color="#4f4a42", va="center")
    ax.text(x0 + 0.17, header_y - row_h / 2, "Match", fontsize=9.2, fontweight="bold", color="#4f4a42", va="center")

    max_rows = 4
    for i in range(max_rows):
        y = header_y - row_h * (i + 2)
        ax.plot([x0, x0 + w], [y, y], color="#ddd6c7", linewidth=0.7)
        if i < len(pages):
            page = pages[i]
            is_gold = page == gold_page
            if is_gold:
                ax.add_patch(plt.Rectangle((x0, y), w, row_h, facecolor="#e2f0dc", edgecolor="none"))
            ax.text(x0 + 0.015, y + row_h / 2, str(i + 1), fontsize=10, color="#2f2a24", va="center")
            ax.text(x0 + 0.09, y + row_h / 2, str(page), fontsize=10, color="#2f2a24", va="center")
            ax.text(x0 + 0.18, y + row_h / 2, "✓" if is_gold else "−", fontsize=11, color="#3d8b40" if is_gold else "#6d675f", va="center")

    rank = rank_of_gold(pages, gold_page)
    rr = reciprocal_rank(rank)
    ax.text(x0 + 0.015, y0 + 0.025, f"Hit@1 = {hit1}", fontsize=10.2, color="#6cae3e" if hit1 else "#b84a3a", fontweight="bold")
    ax.text(x0 + w - 0.015, y0 + 0.025, f"RR = {rr:.2f}", fontsize=10.2, color="#4f4a42", fontweight="bold", ha="right")


def draw_example_section(ax, y_top: float, label: str, query: dict, summaries: dict[str, dict], winner_note: str) -> None:
    gold_page = int(query["expected_pages"][0])
    section_h = 0.38
    add_card(ax, 0.0, y_top - section_h, 1.0, section_h)

    ax.text(0.02, y_top - 0.03, label, fontsize=11, color="#5a544b")
    ax.text(0.08, y_top - 0.03, f"{query['query_id']}  •  {winner_note}", fontsize=11, color="#5a544b")

    qtext = str(query["question"]).strip()
    if query["query_id"] == "Q_2023_FIN_01":
        qtext = "What resource budget ceiling was set for NHS Grampian's core activities in 2022/23?"
    wrapped = textwrap.fill(qtext, width=92)
    ax.text(0.02, y_top - 0.075, wrapped, fontsize=12.5, color="#2f2a24", fontstyle="italic")
    ax.text(0.02, y_top - 0.115, f"Ground truth page: {gold_page}", fontsize=10.8, color="#3d8b40", fontweight="bold")

    ax.text(0.02, y_top - 0.16, "Ground-truth page rank", fontsize=10.8, color="#4f4a42")
    xs = [0.28, 0.36, 0.44]
    for x, cfg in zip(xs, ["224 / 56", "256 / 64", "280 / 90"]):
        rank = rank_of_gold(summaries[cfg]["pages"], gold_page)
        draw_summary_badge(ax, x, y_top - 0.157, rank, cfg)
    ax.text(0.32, y_top - 0.157, "->", fontsize=14, color="#7a746b", va="center", ha="center")
    ax.text(0.40, y_top - 0.157, "->", fontsize=14, color="#7a746b", va="center", ha="center")

    panel_y = y_top - 0.34
    panel_h = 0.18
    panel_w = 0.31
    panel_xs = [0.015, 0.345, 0.675]
    titles = ["Configuration A", "Configuration B", "Configuration C"]
    chunks = {
        "224 / 56": "chunk = 224 tok  •  overlap = 56 tok",
        "256 / 64": "chunk = 256 tok  •  overlap = 64 tok",
        "280 / 90": "chunk = 280 tok  •  overlap = 90 tok",
    }
    for x, cfg, title in zip(panel_xs, ["224 / 56", "256 / 64", "280 / 90"], titles):
        draw_config_panel(ax, x, panel_y, panel_w, panel_h, title, chunks[cfg], summaries[cfg]["pages"], gold_page, summaries[cfg]["hit1"])


def main() -> None:
    args = parse_args()
    eval_path = Path(args.eval_set)
    summary_paths = {
        "224 / 56": Path(args.summary_224),
        "256 / 64": Path(args.summary_256),
        "280 / 90": Path(args.summary_280),
    }

    examples = []
    for qid in [args.query_a, args.query_b]:
        query = load_eval_query(eval_path, qid)
        summaries = {}
        for label, path in summary_paths.items():
            rows = load_summary_rows(path, qid)
            k1 = rows[1]
            k5 = rows[5]
            summaries[label] = {
                "pages": k5["top_pages"],
                "hit1": 1 if k1["failure_type"] == "hit" else 0,
            }
        examples.append((query, summaries))

    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#fcfbf7")

    ax.text(0.0, 0.975, "Balanced Retrieval Examples (Effect of Chunk Segmentation)", fontsize=16, fontweight="bold", color="#2f2a24")
    ax.text(0.0, 0.952, f"Source: {args.document_label}", fontsize=11, color="#5a544b")
    ax.text(
        0.0,
        0.928,
        "One example shows a case where 224/56 ranks the correct page first; the other shows a case where a larger chunk setting does.",
        fontsize=10.5,
        color="#5a544b",
    )

    draw_example_section(ax, 0.89, "(a)", examples[0][0], examples[0][1], "224/56 wins at Hit@1")
    draw_example_section(ax, 0.48, "(b)", examples[1][0], examples[1][1], "Larger chunks win at Hit@1")

    out = Path(args.out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
