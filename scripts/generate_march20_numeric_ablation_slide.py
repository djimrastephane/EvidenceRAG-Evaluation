from __future__ import annotations

import csv
import json
from pathlib import Path

import _matplotlib_env
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    summary_rows = load_csv(
        REPO_ROOT / "results/llm_numeric_model_ablation/fp6_to_fp4_model_ablation_summary.csv"
    )
    detail_rows = load_csv(
        REPO_ROOT / "results/llm_numeric_model_ablation/fp6_to_fp4_model_ablation.csv"
    )

    focus_query = "Q_2021_FIN_03"
    focus_rows = [r for r in detail_rows if str(r.get("query_id") or "") == focus_query]

    plt.rcParams.update({
        "font.size": 12,
        "font.family": "DejaVu Sans",
    })

    fig = plt.figure(figsize=(16, 9), facecolor="#f7f4ed")
    gs = fig.add_gridspec(
        12,
        12,
        left=0.04,
        right=0.98,
        top=0.94,
        bottom=0.06,
        wspace=0.55,
        hspace=0.6,
    )

    ax_title = fig.add_subplot(gs[0:2, :])
    ax_title.axis("off")
    ax_title.text(0.0, 0.78, "March 20: Which LLM Handles Numbers Best?", fontsize=24, fontweight="bold", color="#102a43")
    ax_title.text(
        0.0,
        0.3,
        "21 difficult number questions, same evidence, same number-focused prompt.",
        fontsize=13,
        color="#334e68",
    )

    models = [r["model"] for r in summary_rows]
    ok_rates = [float(r["ok_rate"]) * 100.0 for r in summary_rows]
    insuff_rates = [float(r["insufficient_evidence_rate"]) * 100.0 for r in summary_rows]
    scale_changed = [int(r["scale_changed_count"]) for r in summary_rows]
    sign_mismatch = [int(r["sign_mismatch_count"]) for r in summary_rows]

    x = np.arange(len(models))
    width = 0.36

    ax_rates = fig.add_subplot(gs[2:7, 0:6])
    ax_rates.set_title("How Often Did It Answer?", loc="left", fontsize=16, fontweight="bold", color="#102a43")
    ax_rates.bar(x - width / 2, ok_rates, width, label="Returned an answer", color="#2f855a")
    ax_rates.bar(x + width / 2, insuff_rates, width, label="Refused / said not enough evidence", color="#c05621")
    ax_rates.set_xticks(x)
    ax_rates.set_xticklabels(models, rotation=0)
    ax_rates.set_ylim(0, 110)
    ax_rates.set_ylabel("% of 21 questions")
    ax_rates.grid(axis="y", alpha=0.2)
    ax_rates.legend(frameon=False, loc="upper right")

    ax_errors = fig.add_subplot(gs[2:7, 6:12])
    ax_errors.set_title("When It Answered, Was the Number Still Right?", loc="left", fontsize=16, fontweight="bold", color="#102a43")
    ax_errors.bar(x - width / 2, scale_changed, width, label="Wrong scale", color="#805ad5")
    ax_errors.bar(x + width / 2, sign_mismatch, width, label="Wrong sign", color="#d53f8c")
    ax_errors.set_xticks(x)
    ax_errors.set_xticklabels(models, rotation=0)
    ax_errors.set_ylabel("Count of questions")
    ax_errors.grid(axis="y", alpha=0.2)
    ax_errors.legend(frameon=False, loc="upper right")

    ax_table = fig.add_subplot(gs[7:12, 0:7])
    ax_table.axis("off")
    ax_table.text(0.0, 1.03, "Summary", fontsize=16, fontweight="bold", color="#102a43", va="bottom")
    headers = ["Model", "Answered", "Refused", "Scale errors", "Sign errors"]
    table_data = [
        [
            r["model"],
            f"{float(r['ok_rate']) * 100:.1f}%",
            f"{float(r['insufficient_evidence_rate']) * 100:.1f}%",
            r["scale_changed_count"],
            r["sign_mismatch_count"],
        ]
        for r in summary_rows
    ]
    table = ax_table.table(
        cellText=table_data,
        colLabels=headers,
        loc="upper left",
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, 0.22, 0.95, 0.58],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#d9e2ec")
            cell.set_text_props(weight="bold", color="#102a43")
        else:
            cell.set_facecolor("#ffffff" if r % 2 else "#f8fbff")
            cell.set_text_props(color="#243b53")
        cell.set_edgecolor("#bcccdc")

    ax_examples = fig.add_subplot(gs[7:12, 7:12])
    ax_examples.axis("off")
    ax_examples.text(0.0, 1.03, "Example", fontsize=16, fontweight="bold", color="#102a43", va="bottom")
    ax_examples.text(0.0, 0.92, "Question: how much did spending differ from budget in 2020/21?", fontsize=11.5, color="#243b53", va="top")
    ax_examples.text(0.0, 0.84, "Correct answer: £769,000 surplus", fontsize=12.5, color="#102a43", va="top", fontweight="bold")
    y = 0.72
    for row in focus_rows:
        model = str(row.get("model") or "")
        gen = str(row.get("generated_answer") or "").strip() or "Insufficient evidence"
        status = str(row.get("generation_status") or "")
        label = "returned answer" if status == "ok" else "refused"
        lines = [f"{model}", f"{label}: {gen}"]
        ax_examples.text(0.0, y, "\n".join(lines), fontsize=11.5, color="#243b53", va="top")
        y -= 0.21

    ax_examples.text(
        0.0,
        0.05,
        "Do not rely on the LLM alone for numeric answers.\nA deterministic numeric extractor is still safer.",
        fontsize=11.5,
        color="#334e68",
        va="bottom",
    )

    ax_footer = fig.add_subplot(gs[11:12, :])
    ax_footer.axis("off")
    ax_footer.text(
        0.0,
        0.15,
        "Bottom line: a different LLM helps a little, but none solves numeric reliability on its own.",
        fontsize=12,
        color="#334e68",
    )

    out_dir = REPO_ROOT / "artifacts/march20_weekly_update"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "March20_numeric_ablation_slide.png"
    pdf_path = out_dir / "March20_numeric_ablation_slide.pdf"
    fig.savefig(png_path, dpi=180, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)

    combined_pdf = out_dir / "March20_weekly_update_two_slide.pdf"
    with PdfPages(combined_pdf) as pdf:
        for single in [out_dir / "March20_weekly_update_slide.png", png_path]:
            img = plt.imread(single)
            fig_page = plt.figure(figsize=(16, 9), facecolor="white")
            ax = fig_page.add_axes([0, 0, 1, 1])
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig_page, dpi=180)
            plt.close(fig_page)

    print(json.dumps({
        "png": str(png_path),
        "pdf": str(pdf_path),
        "combined_pdf": str(combined_pdf),
    }, indent=2))


if __name__ == "__main__":
    main()
