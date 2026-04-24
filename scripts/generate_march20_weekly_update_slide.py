from __future__ import annotations

import csv
import json
from pathlib import Path

import _matplotlib_env
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_model_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    comparison = load_json(
        REPO_ROOT / "results/live_fp1_fp7_compare_llm_vs_retrieval_norm/fp1_fp7_comparison_summary.json"
    )
    model_rows = load_model_rows(
        REPO_ROOT / "results/llm_numeric_model_ablation/fp6_to_fp4_model_ablation_summary.csv"
    )
    heatmap_path = REPO_ROOT / "results/live_fp1_fp7_compare_llm_vs_retrieval_norm/fp1_fp7_heatmaps_side_by_side_norm_labeled.png"
    heatmap = mpimg.imread(heatmap_path)

    baseline = comparison["baseline_failure_counts_total"]
    candidate = comparison["candidate_failure_counts_total"]
    improved = int(comparison["improved_to_hit"])
    regressed = int(comparison["regressed_from_hit"])

    plt.rcParams.update({
        "font.size": 12,
        "font.family": "DejaVu Sans",
    })

    fig = plt.figure(figsize=(16, 9), facecolor="#f6f2ea")
    gs = fig.add_gridspec(
        12,
        12,
        left=0.035,
        right=0.985,
        top=0.95,
        bottom=0.05,
        wspace=0.5,
        hspace=0.45,
    )

    ax_title = fig.add_subplot(gs[0:2, :])
    ax_title.axis("off")
    ax_title.text(0.0, 0.82, "March 20: What Improved, What Still Breaks", fontsize=24, fontweight="bold", color="#13293d")
    ax_title.text(
        0.0,
        0.34,
        "250 test questions across 5 Grampian reports. Numeric answers are scored with the cleaned matching rule.",
        fontsize=13,
        color="#334e68",
    )

    ax_heatmap = fig.add_subplot(gs[2:8, 0:8])
    ax_heatmap.imshow(heatmap)
    ax_heatmap.set_xticks([])
    ax_heatmap.set_yticks([])
    ax_heatmap.set_title("Where the System Still Fails", loc="left", pad=12, fontsize=16, fontweight="bold", color="#13293d")

    ax_findings = fig.add_subplot(gs[2:8, 8:12])
    ax_findings.axis("off")
    findings = [
        f"Numeric cleanup helps a little: correct answers rise from 30 to 36 without the LLM, and from {baseline['HIT']} to {candidate['HIT']} with it.",
        f"The LLM mainly fixes vague or over-broad answers: that error falls from {baseline['FP6_INCORRECT_SPECIFICITY']} to {candidate['FP6_INCORRECT_SPECIFICITY']}.",
        f"Ranking errors do not move: wrong-page-first stays at {candidate['FP2_MISSED_TOP_RANK']}, and missing-evidence-in-context stays at {candidate['FP3_NOT_IN_CONTEXT']}.",
        f"The trade-off is more rejected answers: FP4 rises to {candidate['FP4_NOT_EXTRACTED']}. Improved: {improved}; regressed: {regressed}.",
    ]
    ax_findings.text(0.0, 1.0, "Takeaways", fontsize=16, fontweight="bold", color="#13293d", va="top")
    y = 0.88
    for line in findings:
        ax_findings.text(0.0, y, "\u2022 " + line, fontsize=12.5, color="#243b53", va="top", wrap=True)
        y -= 0.185

    ax_models = fig.add_subplot(gs[8:12, :])
    ax_models.axis("off")
    ax_models.text(0.0, 1.02, "Quick Check: Does a Different LLM Fix the Number Problem?", fontsize=16, fontweight="bold", color="#13293d", va="bottom")

    headers = ["Model", "Answered", "Refused", "Scale errors", "Sign errors"]
    table_data = []
    for row in model_rows:
        table_data.append([
            row["model"],
            f"{float(row['ok_rate'])*100:.1f}%",
            f"{float(row['insufficient_evidence_rate'])*100:.1f}%",
            row["scale_changed_count"],
            row["sign_mismatch_count"],
        ])

    table = ax_models.table(
        cellText=table_data,
        colLabels=headers,
        loc="upper left",
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, 0.18, 0.86, 0.62],
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

    ax_models.text(
        0.88,
        0.78,
        "Bottom line",
        fontsize=14,
        fontweight="bold",
        color="#13293d",
        va="top",
    )
    interp = (
        "Changing the LLM helps only a little.\n"
        "Qwen answers most often, but still makes number mistakes.\n"
        "Mistral refuses more often.\n"
        "Llama3 answers more often than Mistral, but still changes sign or scale."
    )
    ax_models.text(0.88, 0.70, interp, fontsize=11.5, color="#243b53", va="top")

    out_dir = REPO_ROOT / "artifacts/march20_weekly_update"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "March20_weekly_update_slide.png"
    pdf_path = out_dir / "March20_weekly_update_slide.pdf"
    fig.savefig(png_path, dpi=180, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(json.dumps({"png": str(png_path), "pdf": str(pdf_path)}, indent=2))


if __name__ == "__main__":
    main()
