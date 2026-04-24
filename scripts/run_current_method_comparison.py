from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "current_method_comparison_2026-04-07"
DOC_IDS = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]
METHODS = {
    "dense": {
        "label": "Dense (MiniLM)",
        "cmd": ["scripts/retrieval_eval.py"],
        "metrics_name": "retrieval_metrics.json",
        "summary_name": "retrieval_summary.csv",
        "extra_env": {},
    },
    "bm25": {
        "label": "BM25-only",
        "cmd": ["scripts/retrieval_eval_bm25.py"],
        "metrics_name": "retrieval_metrics_bm25.json",
        "summary_name": "retrieval_summary_bm25.csv",
        "extra_env": {},
    },
    "hybrid_base": {
        "label": "Hybrid (base)",
        "cmd": ["scripts/retrieval_eval_hybrid.py"],
        "metrics_name": "retrieval_metrics_hybrid.json",
        "summary_name": "retrieval_summary_hybrid.csv",
        "extra_env": {"ENABLE_SUBSECTION_BOOST": "0", "SUBSECTION_BOOST": "0.0"},
    },
    "hybrid_boost": {
        "label": "Hybrid + subsection boost",
        "cmd": ["scripts/retrieval_eval_hybrid.py"],
        "metrics_name": "retrieval_metrics_hybrid.json",
        "summary_name": "retrieval_summary_hybrid.csv",
        "extra_env": {"ENABLE_SUBSECTION_BOOST": "1", "SUBSECTION_BOOST": "0.05"},
    },
}
WIN_LOSS_COMPARISONS = [
    ("Hybrid + subsection boost vs Hybrid (base)", "hybrid_boost", "hybrid_base"),
    ("Hybrid + subsection boost vs Dense", "hybrid_boost", "dense"),
    ("Hybrid + subsection boost vs BM25", "hybrid_boost", "bm25"),
]
METRIC_FIELDS = {
    "Hit@1": (1, "page_recall_at_k"),
    "Hit@3": (3, "page_recall_at_k"),
    "MRR@10": (10, "page_mrr_at_k"),
}
K_TABLE = [1, 3, 10]


@dataclass(frozen=True)
class MethodResult:
    method: str
    doc_id: str
    metrics_path: Path
    summary_path: Path


def run_cmd(cmd: list[str], env: dict[str, str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT / "src"))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("ST_MODEL_DEVICE", "cpu")
    env.setdefault("CROSS_ENCODER_DEVICE", "cpu")
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    return env


def run_method(doc_id: str, method_key: str) -> MethodResult:
    cfg = METHODS[method_key]
    data_dir = ROOT / "data_processed" / doc_id
    env = base_env()
    env.update(cfg["extra_env"])
    cmd = [sys.executable, *cfg["cmd"], "--data-dir", str(data_dir)]
    if method_key in {"dense", "hybrid_base", "hybrid_boost"}:
        cmd += ["--model", "models/all-MiniLM-L6-v2", "--device", "cpu"]
    run_cmd(cmd, env)

    out_dir = RESULTS_DIR / method_key / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = data_dir / cfg["metrics_name"]
    summary_path = data_dir / cfg["summary_name"]
    shutil.copy2(metrics_path, out_dir / cfg["metrics_name"])
    shutil.copy2(summary_path, out_dir / cfg["summary_name"])
    return MethodResult(
        method=method_key,
        doc_id=doc_id,
        metrics_path=out_dir / cfg["metrics_name"],
        summary_path=out_dir / cfg["summary_name"],
    )


def aggregate_metrics(results: list[MethodResult]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method_key, cfg in METHODS.items():
        method_results = [r for r in results if r.method == method_key]
        for k in [1, 3, 5, 10]:
            hit_values: list[float] = []
            mrr_values: list[float] = []
            query_total = 0
            for res in method_results:
                obj = json.loads(res.metrics_path.read_text(encoding="utf-8"))
                metrics = obj["metrics_by_k"][str(k)]
                hit_values.append(float(metrics["page_hit_rate_at_k"]))
                mrr_values.append(float(metrics["mean_page_mrr_at_k"]))
                query_total += int(metrics.get("num_queries", 0))
            rows.append(
                {
                    "method": method_key,
                    "label": cfg["label"],
                    "k": k,
                    "weighted_page_hit": sum(hit_values) / len(hit_values),
                    "weighted_page_mrr": sum(mrr_values) / len(mrr_values),
                    "queries": query_total,
                }
            )
    return pd.DataFrame(rows)


def load_summary_rows(results: list[MethodResult], method_key: str) -> dict[str, dict[int, dict[str, str]]]:
    out: dict[str, dict[int, dict[str, str]]] = {}
    for res in results:
        if res.method != method_key:
            continue
        with res.summary_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                out.setdefault(row["query_id"], {})[int(row["k"])] = row
    return out


def compare_rows(
    left_rows: dict[str, dict[int, dict[str, str]]],
    right_rows: dict[str, dict[int, dict[str, str]]],
) -> list[dict[str, object]]:
    qids = sorted(set(left_rows).intersection(right_rows))
    out: list[dict[str, object]] = []
    for metric_name, (k, field) in METRIC_FIELDS.items():
        wins = losses = ties = 0
        for qid in qids:
            left_val = float(left_rows[qid][k][field])
            right_val = float(right_rows[qid][k][field])
            if abs(left_val - right_val) < 1e-12:
                ties += 1
            elif left_val > right_val:
                wins += 1
            else:
                losses += 1
        out.append(
            {
                "metric": metric_name,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "queries_compared": len(qids),
            }
        )
    return out


def build_win_loss_summary(results: list[MethodResult]) -> pd.DataFrame:
    loaded = {method: load_summary_rows(results, method) for method in METHODS}
    rows: list[dict[str, object]] = []
    for title, left_method, right_method in WIN_LOSS_COMPARISONS:
        compared = compare_rows(loaded[left_method], loaded[right_method])
        for row in compared:
            rows.append({"comparison": title, **row})
    return pd.DataFrame(rows)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 10.0,
            "hatch.linewidth": 0.6,
        }
    )


def plot_win_loss(summary_df: pd.DataFrame) -> None:
    configure_plot_style()
    titles = [x[0] for x in WIN_LOSS_COMPARISONS]
    metric_order = ["Hit@1", "Hit@3", "MRR@10"]
    colors = {"wins": "#0072B2", "losses": "#D55E00", "ties": "#D0D0D0"}
    hatches = {"wins": "//", "losses": "/", "ties": "."}

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.9), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, title in zip(axes, titles):
        panel = summary_df[summary_df["comparison"] == title].set_index("metric")
        for y, metric in enumerate(metric_order):
            row = panel.loc[metric]
            ax.barh(y, -row["losses"], height=0.48, color=colors["losses"], hatch=hatches["losses"], edgecolor="#555555", linewidth=0.5, zorder=3)
            ax.barh(y, row["ties"], height=0.48, color=colors["ties"], hatch=hatches["ties"], edgecolor="#777777", linewidth=0.4, alpha=0.6, zorder=2)
            ax.barh(y, row["wins"], left=row["ties"], height=0.48, color=colors["wins"], hatch=hatches["wins"], edgecolor="#555555", linewidth=0.5, zorder=3)
            ax.text(-row["losses"] - 1.4, y, f"{int(row['losses'])}", ha="right", va="center", fontsize=9.5)
            ax.text(row["ties"] / 2.0, y, f"{int(row['ties'])}", ha="center", va="center", fontsize=9.5)
            ax.text(row["ties"] + row["wins"] + 1.4, y, f"{int(row['wins'])}", ha="left", va="center", fontsize=9.5)
        ax.axvline(0, color="#333333", linewidth=0.7)
        ax.set_title(title, pad=8, fontweight="semibold")
        ax.set_yticks(range(len(metric_order)), metric_order)
        ax.set_xlabel("Queries")
        ax.set_xlim(-110, 250)
        ax.invert_yaxis()
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#888888")
        ax.tick_params(axis="y", length=0, colors="#222222")
        ax.tick_params(axis="x", colors="#333333")

    fig.suptitle("Per-query comparison of the final hybrid system against current baselines (n = 250)", y=0.97, fontsize=13, fontweight="semibold")
    fig.text(0.5, 0.03, "Left = losses, centre = ties, right = wins.", ha="center", va="bottom", fontsize=9.5, color="#333333")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.19, wspace=0.08)
    png_path = RESULTS_DIR / "per_query_comparison_current_publication.png"
    pdf_path = RESULTS_DIR / "per_query_comparison_current_publication.pdf"
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def write_markdown(agg_df: pd.DataFrame, win_loss_df: pd.DataFrame) -> None:
    table42 = agg_df[agg_df["k"].isin(K_TABLE)].copy()
    pivot_hit = table42.pivot(index="label", columns="k", values="weighted_page_hit")
    pivot_mrr = table42[table42["k"] == 10].set_index("label")["weighted_page_mrr"]
    lines = [
        "# Current Method Comparison",
        "",
        "## Table 4.1 headline",
        "",
    ]
    headline = agg_df[(agg_df["method"] == "hybrid_boost") & (agg_df["k"].isin(K_TABLE))]
    for _, row in headline.sort_values("k").iterrows():
        if row["k"] in {1, 3}:
            lines.append(f"- Hit@{int(row['k'])}: {row['weighted_page_hit']:.4f}")
        if row["k"] == 10:
            lines.append(f"- MRR@10: {row['weighted_page_mrr']:.4f}")
    lines += [
        "",
        "## Table 4.2 comparison",
        "",
        "| Method | Hit@1 | Hit@3 | MRR@10 |",
        "|---|---:|---:|---:|",
    ]
    for label in [METHODS[k]["label"] for k in METHODS]:
        lines.append(
            f"| {label} | {pivot_hit.loc[label,1]:.4f} | {pivot_hit.loc[label,3]:.4f} | {pivot_mrr.loc[label]:.4f} |"
        )
    lines += [
        "",
        "## Win/loss summary",
        "",
        "| Comparison | Metric | Wins | Losses | Ties |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in win_loss_df.iterrows():
        lines.append(f"| {row['comparison']} | {row['metric']} | {int(row['wins'])} | {int(row['losses'])} | {int(row['ties'])} |")
    (RESULTS_DIR / "current_method_comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or summarize the current 4-way retrieval comparison.")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse already-copied per-document outputs under results/current_method_comparison_2026-04-07.",
    )
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: list[MethodResult] = []
    for doc_id in DOC_IDS:
        for method_key, cfg in METHODS.items():
            out_dir = RESULTS_DIR / method_key / doc_id
            metrics_path = out_dir / cfg["metrics_name"]
            summary_path = out_dir / cfg["summary_name"]
            if args.reuse_existing and metrics_path.exists() and summary_path.exists():
                results.append(
                    MethodResult(
                        method=method_key,
                        doc_id=doc_id,
                        metrics_path=metrics_path,
                        summary_path=summary_path,
                    )
                )
            else:
                results.append(run_method(doc_id, method_key))

    agg_df = aggregate_metrics(results)
    agg_df.to_csv(RESULTS_DIR / "current_method_comparison_aggregate.csv", index=False)

    win_loss_df = build_win_loss_summary(results)
    win_loss_df.to_csv(RESULTS_DIR / "current_method_comparison_win_loss.csv", index=False)
    plot_win_loss(win_loss_df)
    write_markdown(agg_df, win_loss_df)


if __name__ == "__main__":
    main()
