from __future__ import annotations

"""Reproduce thesis Figure 4.1 from the refactored thesis_rag pipeline.

The original Figure 4.1 is a per-query win/loss/tie comparison between the
subsection-boosted hybrid system and three baselines:

- Hybrid (base)
- Dense (MiniLM)
- BM25

This script rebuilds the same figure structure using the saved 5-document
``thesis_rag`` 224/56 benchmark artifacts. Dense, BM25, and boosted hybrid are
read directly from the saved page-hit files. The unboosted hybrid baseline is
reconstructed from the same saved chunk metadata, FAISS indexes, and eval sets
with subsection boosting disabled so the comparison stays aligned to the exact
same artifact set.
"""

import csv
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from thesis_rag.artifacts import load_chunks, load_queries
from thesis_rag.config import load_config
from thesis_rag.embedding import embed_queries
from thesis_rag.evaluator import evaluate_page_hits
from thesis_rag.ranking import chunk_hits_to_page_hits
from thesis_rag.retrieval_dense import search_faiss_stably
from thesis_rag.retrieval_hybrid import hybrid_retrieve_legacy_style
from thesis_rag.retrieval_sparse import build_bm25
from thesis_rag.schemas import EvaluationResult, RetrievalHit
from thesis_rag.utils import resolve_device, set_global_determinism


DOC_IDS = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]
PIPELINE_CONFIG = REPO_ROOT / "configs" / "thesis_rag.yaml"
ARTIFACT_ROOT = REPO_ROOT / "results" / "thesis_ablations" / "chunk_size_ablation_2026-04-15" / "pipeline_outputs"
OUT_DIR = REPO_ROOT / "results" / "thesis_figures" / f"figure_4_1_current_method_comparison_{date.today().isoformat()}"
OUT_PNG = OUT_DIR / "per_query_comparison_current_publication_thesis_rag.png"
OUT_PDF = OUT_DIR / "per_query_comparison_current_publication_thesis_rag.pdf"
OUT_CSV = OUT_DIR / "current_method_comparison_win_loss.csv"
OUT_MD = OUT_DIR / "current_method_comparison_summary.md"
OUT_JSON = OUT_DIR / "current_method_comparison_summary.json"

COMPARISONS = [
    ("Hybrid + subsection boost vs Hybrid (base)", "hybrid_boost", "hybrid_base"),
    ("Hybrid + subsection boost vs Dense", "hybrid_boost", "dense"),
    ("Hybrid + subsection boost vs BM25", "hybrid_boost", "bm25"),
]
METRICS = {
    "Hit@1": "hit_at_1",
    "Hit@3": "hit_at_3",
    "MRR@10": "reciprocal_rank",
}
METHOD_LABELS = {
    "dense": "Dense (MiniLM)",
    "bm25": "BM25-only",
    "hybrid_base": "Hybrid (base)",
    "hybrid_boost": "Hybrid + subsection boost",
}
COLORS = {"wins": "#0072B2", "losses": "#D55E00", "ties": "#D0D0D0"}
HATCHES = {"wins": "//", "losses": "/", "ties": "."}


@dataclass(frozen=True)
class SummaryRow:
    comparison: str
    metric: str
    wins: int
    losses: int
    ties: int
    queries_compared: int


def _load_hits(path: Path) -> list[RetrievalHit]:
    import pandas as pd

    frame = pd.read_csv(path)
    return [RetrievalHit(**row) for row in frame.to_dict(orient="records")]


def _load_saved_method_results(method: str) -> dict[str, EvaluationResult]:
    rows: dict[str, EvaluationResult] = {}
    filename = {
        "dense": "dense_page_hits.csv",
        "bm25": "bm25_page_hits.csv",
        "hybrid_boost": "hybrid_page_hits.csv",
    }[method]
    for doc_id in DOC_IDS:
        artifact_dir = ARTIFACT_ROOT / f"minilmcap_{doc_id}_chunk_224_56" / doc_id
        hits = _load_hits(artifact_dir / filename)
        queries = load_queries(REPO_ROOT / "data_processed" / doc_id / "eval_set.json")
        for result in evaluate_page_hits(queries, hits):
            rows[result.query_id] = result
    if len(rows) != 250:
        raise RuntimeError(f"Expected 250 query results for {method}, found {len(rows)}")
    return rows


def _load_hybrid_base_results() -> dict[str, EvaluationResult]:
    import faiss

    config = load_config(PIPELINE_CONFIG)
    config.embedding.model_name = str(REPO_ROOT / "models" / "all-MiniLM-L6-v2")
    config.retrieval.dense_top_k = 10
    config.retrieval.sparse_top_k = 10
    config.retrieval.hybrid_top_k = 10
    config.retrieval.rrf_k = 20
    config.retrieval.dense_weight = 0.5
    config.retrieval.sparse_weight = 2.0
    set_global_determinism(config.runtime.random_seed, config.runtime.deterministic_torch)
    device = resolve_device(config.runtime.device)

    rows: dict[str, EvaluationResult] = {}
    for doc_id in DOC_IDS:
        artifact_dir = ARTIFACT_ROOT / f"minilmcap_{doc_id}_chunk_224_56" / doc_id
        chunks = load_chunks(artifact_dir / "chunk_metadata.parquet")
        queries = load_queries(REPO_ROOT / "data_processed" / doc_id / "eval_set.json")
        index = faiss.read_index(str(artifact_dir / "faiss.index"))
        query_vectors = embed_queries(
            [query.query_text for query in queries],
            config.embedding,
            device=device,
            cache_dir=str(config.paths.model_cache_dir),
        )
        bm25 = build_bm25(chunks, config.bm25)
        raw_dense_scores, raw_dense_indices = search_faiss_stably(
            index,
            query_vectors,
            min(max(100, config.retrieval.hybrid_top_k), len(chunks)),
        )
        _dense, _bm25, hybrid_chunk_hits = hybrid_retrieve_legacy_style(
            chunks=chunks,
            queries=queries,
            dense_scores=raw_dense_scores,
            dense_indices=raw_dense_indices,
            bm25=bm25,
            max_k_search=max(100, config.retrieval.hybrid_top_k),
            dense_weight=config.retrieval.dense_weight,
            bm25_weight=config.retrieval.sparse_weight,
            rrf_k=config.retrieval.rrf_k,
            enable_subsection_boost=False,
            subsection_boost=0.0,
        )
        page_hits = chunk_hits_to_page_hits(
            hybrid_chunk_hits,
            "hybrid_pages_base",
            chunk_limit=config.retrieval.hybrid_top_k,
        )
        for result in evaluate_page_hits(queries, page_hits):
            rows[result.query_id] = result
    if len(rows) != 250:
        raise RuntimeError(f"Expected 250 query results for hybrid base, found {len(rows)}")
    return rows


def _compare(left: dict[str, EvaluationResult], right: dict[str, EvaluationResult], comparison: str) -> list[SummaryRow]:
    query_ids = sorted(set(left).intersection(right))
    out: list[SummaryRow] = []
    for metric_name, field in METRICS.items():
        wins = losses = ties = 0
        for query_id in query_ids:
            left_val = getattr(left[query_id], field)
            right_val = getattr(right[query_id], field)
            if isinstance(left_val, bool):
                left_num = float(left_val)
                right_num = float(right_val)
            else:
                left_num = float(left_val)
                right_num = float(right_val)
            if abs(left_num - right_num) < 1e-12:
                ties += 1
            elif left_num > right_num:
                wins += 1
            else:
                losses += 1
        out.append(SummaryRow(comparison, metric_name, wins, losses, ties, len(query_ids)))
    return out


def _configure_style() -> None:
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


def _plot(rows: list[SummaryRow]) -> None:
    _configure_style()
    by_comparison: dict[str, dict[str, SummaryRow]] = {}
    for row in rows:
        by_comparison.setdefault(row.comparison, {})[row.metric] = row

    metric_order = ["Hit@1", "Hit@3", "MRR@10"]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.9), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (title, _, _) in zip(axes, COMPARISONS):
        panel = by_comparison[title]
        for y, metric in enumerate(metric_order):
            row = panel[metric]
            ax.barh(y, -row.losses, height=0.48, color=COLORS["losses"], hatch=HATCHES["losses"], edgecolor="#555555", linewidth=0.5, zorder=3)
            ax.barh(y, row.ties, height=0.48, color=COLORS["ties"], hatch=HATCHES["ties"], edgecolor="#777777", linewidth=0.4, alpha=0.6, zorder=2)
            ax.barh(y, row.wins, left=row.ties, height=0.48, color=COLORS["wins"], hatch=HATCHES["wins"], edgecolor="#555555", linewidth=0.5, zorder=3)
            ax.text(-row.losses - 1.4, y, f"{row.losses}", ha="right", va="center", fontsize=9.5)
            ax.text(row.ties / 2.0, y, f"{row.ties}", ha="center", va="center", fontsize=9.5)
            ax.text(row.ties + row.wins + 1.4, y, f"{row.wins}", ha="left", va="center", fontsize=9.5)
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

    fig.suptitle("Per-query comparison of the final hybrid system against thesis_rag baselines (n = 250)", y=0.97, fontsize=13, fontweight="semibold")
    fig.text(0.5, 0.03, "Left = losses, centre = ties, right = wins.", ha="center", va="bottom", fontsize=9.5, color="#333333")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.19, wspace=0.08)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=320, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)


def _write_outputs(rows: list[SummaryRow]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["comparison", "metric", "wins", "losses", "ties", "queries_compared"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "comparison": row.comparison,
                    "metric": row.metric,
                    "wins": row.wins,
                    "losses": row.losses,
                    "ties": row.ties,
                    "queries_compared": row.queries_compared,
                }
            )

    summary = {
        "artifact_root": str(ARTIFACT_ROOT),
        "doc_ids": DOC_IDS,
        "comparisons": [
            {
                "comparison": row.comparison,
                "metric": row.metric,
                "wins": row.wins,
                "losses": row.losses,
                "ties": row.ties,
                "queries_compared": row.queries_compared,
            }
            for row in rows
        ],
    }
    OUT_JSON.write_text(__import__("json").dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Figure 4.1 Reproduction (thesis_rag)",
        "",
        "| Comparison | Metric | Wins | Losses | Ties | Queries |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.comparison} | {row.metric} | {row.wins} | {row.losses} | {row.ties} | {row.queries_compared} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Regenerate Figure 4.1 retrieval performance comparison across dense, BM25, hybrid, and boosted-hybrid."""
    dense = _load_saved_method_results("dense")
    bm25 = _load_saved_method_results("bm25")
    hybrid_boost = _load_saved_method_results("hybrid_boost")
    hybrid_base = _load_hybrid_base_results()

    rows: list[SummaryRow] = []
    method_map = {
        "dense": dense,
        "bm25": bm25,
        "hybrid_base": hybrid_base,
        "hybrid_boost": hybrid_boost,
    }
    for title, left_name, right_name in COMPARISONS:
        rows.extend(_compare(method_map[left_name], method_map[right_name], title))
    _write_outputs(rows)
    _plot(rows)
    print(OUT_DIR)


if __name__ == "__main__":
    main()
