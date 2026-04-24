"""ablate_bm25_params.py

Grid search over BM25 k1 × b parameters using BM25-only (sparse) retrieval
against the existing post-fix 224/56 eval sets.

Metrics reported per cell: hit@1, hit@5, MRR@10.
Produces:
  - heatmap_mrr10.png/pdf    — MRR@10 across the k1×b grid
  - heatmap_hit1.png/pdf     — hit@1 across the k1×b grid
  - heatmap_hit5.png/pdf     — hit@5 across the k1×b grid
  - results.json             — full per-cell scores
  - summary.txt              — ranked table + recommended params

Usage:
    python scripts/ablate_bm25_params.py [--top-k 10]
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH  = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from thesis_rag.artifacts import load_queries
from thesis_rag.evaluator import hit_at_k
from thesis_rag.ranking import chunk_hits_to_page_hits
from thesis_rag.retrieval_sparse import build_bm25, sparse_retrieve_legacy_style
from thesis_rag.schemas import BM25Config, ChunkRecord

ARTIFACT_ROOT = REPO_ROOT / "results/thesis_ablations/post_fix_rerun_2026-04-19/pipeline_outputs"
EVAL_ROOT     = REPO_ROOT / "data_processed"
DOCS = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]

K1_VALUES = [0.5, 1.0, 1.2, 1.5, 2.0, 3.0]
B_VALUES  = [0.0, 0.25, 0.5, 0.75, 1.0]

TOP_K_RETRIEVE = 10   # max rank needed for MRR@10 / hit@10


# ---------------------------------------------------------------------------
# Chunk loader (reuses same parquet layout as other scripts)
# ---------------------------------------------------------------------------

def _load_chunks(exp_dir: Path) -> list[ChunkRecord]:
    df = pd.read_parquet(exp_dir / "chunks.parquet")
    chunks = []
    for row in df.to_dict(orient="records"):
        pages_raw = row.get("pages")
        pages = list(pages_raw) if isinstance(pages_raw, list) else [
            int(row.get("page_start") or row.get("page_number") or 0)
        ]
        chunks.append(ChunkRecord(
            chunk_id=str(row["chunk_id"]),
            doc_id=str(row["doc_id"]),
            page_number=int(row.get("page_number") or row.get("page_start") or 0),
            chunk_index=int(row.get("chunk_index", 0)),
            text=str(row.get("text", "")),
            token_count=int(row.get("token_count", 0)),
            word_count=int(row.get("word_count", 0)),
            chunk_id_global=str(row.get("chunk_id_global", "")),
            page_start=int(row.get("page_start") or row.get("page_number") or 0),
            page_end=int(row.get("page_end") or row.get("page_number") or 0),
            pages=pages,
            part=str(row.get("part") or ""),
            section_title=str(row.get("section_title") or ""),
            subsection_title=str(row.get("subsection_title") or ""),
            is_table=bool(row.get("is_table", False)),
            table_type=str(row.get("table_type")) if row.get("table_type") else None,
            table_chunk_kind=str(row.get("table_chunk_kind")) if row.get("table_chunk_kind") else None,
            segment_boundary_type=str(row.get("segment_boundary_type")) if row.get("segment_boundary_type") else None,
            segment_has_search_hit=bool(row.get("segment_has_search_hit", False)),
        ))
    return chunks


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _compute_metrics(chunks: list[ChunkRecord], queries, bm25_index, top_k: int) -> dict:
    hits = sparse_retrieve_legacy_style(bm25_index, chunks, queries, top_k=top_k)
    page_hits = chunk_hits_to_page_hits(hits, "bm25_pages", chunk_limit=top_k)

    from collections import defaultdict
    grouped: dict[str, list[int]] = defaultdict(list)
    for h in page_hits:
        grouped[h.query_id].append(h.page_number)

    h1 = h5 = h10 = mrr = 0
    for q in queries:
        pages = grouped.get(q.query_id, [])
        h1  += int(hit_at_k(pages, q.gold_pages, 1))
        h5  += int(hit_at_k(pages, q.gold_pages, 5))
        h10 += int(hit_at_k(pages, q.gold_pages, 10))
        rr   = next((1.0 / (i + 1) for i, p in enumerate(pages[:10])
                     if p in set(q.gold_pages)), 0.0)
        mrr += rr

    n = max(len(queries), 1)
    return {"hit@1": h1 / n, "hit@5": h5 / n, "hit@10": h10 / n, "mrr@10": mrr / n, "n": n}


# ---------------------------------------------------------------------------
# Run grid
# ---------------------------------------------------------------------------

def run_grid(top_k: int) -> dict:
    """Returns nested dict: results[k1][b] = {hit@1, hit@5, mrr@10, ...}"""

    # Pre-load chunks and queries once per doc
    print("Loading chunks and queries...", flush=True)
    doc_data: list[tuple] = []
    for doc_id in DOCS:
        exp_dir = ARTIFACT_ROOT / f"minilmcap_{doc_id}_chunk_224_56" / doc_id
        chunks  = _load_chunks(exp_dir)
        queries = load_queries(EVAL_ROOT / doc_id / "eval_set.json")
        doc_data.append((doc_id, chunks, queries))
        print(f"  {doc_id}: {len(chunks)} chunks, {len(queries)} queries")

    total_cells = len(K1_VALUES) * len(B_VALUES)
    cell = 0
    results: dict[float, dict[float, dict]] = {}

    for k1 in K1_VALUES:
        results[k1] = {}
        for b in B_VALUES:
            cell += 1
            bm25_cfg = BM25Config(k1=k1, b=b)
            agg = {"hit@1": 0.0, "hit@5": 0.0, "hit@10": 0.0, "mrr@10": 0.0, "n": 0}

            for doc_id, chunks, queries in doc_data:
                bm25 = build_bm25(chunks, bm25_cfg)
                m    = _compute_metrics(chunks, queries, bm25, top_k)
                for key in ("hit@1", "hit@5", "hit@10", "mrr@10"):
                    agg[key] += m[key] * m["n"]
                agg["n"] += m["n"]

            n = agg["n"]
            results[k1][b] = {
                "hit@1":  agg["hit@1"]  / n,
                "hit@5":  agg["hit@5"]  / n,
                "hit@10": agg["hit@10"] / n,
                "mrr@10": agg["mrr@10"] / n,
                "n": n,
            }
            r = results[k1][b]
            print(f"  [{cell:2d}/{total_cells}] k1={k1:.1f} b={b:.2f}  "
                  f"hit@1={r['hit@1']:.4f}  hit@5={r['hit@5']:.4f}  mrr@10={r['mrr@10']:.4f}",
                  flush=True)

    return results


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def _build_matrix(results: dict, metric: str) -> np.ndarray:
    mat = np.zeros((len(K1_VALUES), len(B_VALUES)))
    for i, k1 in enumerate(K1_VALUES):
        for j, b in enumerate(B_VALUES):
            mat[i, j] = results[k1][b][metric]
    return mat


def _plot_heatmap(mat: np.ndarray, metric: str, title: str, out_path: Path,
                  current_k1: float = 1.5, current_b: float = 0.75) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    vmin, vmax = mat.min(), mat.max()
    im = ax.imshow(mat, cmap="YlGn", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(B_VALUES)))
    ax.set_xticklabels([str(b) for b in B_VALUES], fontsize=10)
    ax.set_yticks(range(len(K1_VALUES)))
    ax.set_yticklabels([str(k1) for k1 in K1_VALUES], fontsize=10)
    ax.set_xlabel("b  (length normalisation)", fontsize=11)
    ax.set_ylabel("k1  (term frequency saturation)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    best_val = mat.max()
    for i in range(len(K1_VALUES)):
        for j in range(len(B_VALUES)):
            val = mat[i, j]
            color = "white" if val > (vmin + 0.6 * (vmax - vmin)) else "black"
            weight = "bold" if abs(val - best_val) < 1e-6 else "normal"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight=weight)

    # Mark current pipeline setting
    cur_i = K1_VALUES.index(current_k1) if current_k1 in K1_VALUES else None
    cur_j = B_VALUES.index(current_b)   if current_b  in B_VALUES  else None
    if cur_i is not None and cur_j is not None:
        ax.add_patch(plt.Rectangle(
            (cur_j - 0.5, cur_i - 0.5), 1, 1,
            fill=False, edgecolor="#E45756", linewidth=2.5, label="current pipeline"
        ))
        ax.legend(handles=[
            plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#E45756", linewidth=2.5)
        ], labels=["current pipeline (k1=1.5, b=0.75)"],
            loc="lower right", fontsize=9, frameon=True)

    plt.colorbar(im, ax=ax, label=metric, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.with_suffix('.png')}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_and_save_summary(results: dict, out_path: Path) -> None:
    rows = []
    for k1 in K1_VALUES:
        for b in B_VALUES:
            r = results[k1][b]
            rows.append({"k1": k1, "b": b, **r})

    df = pd.DataFrame(rows).sort_values("mrr@10", ascending=False)

    lines = []
    lines.append("BM25 parameter grid search — averaged over 5 Grampian docs (250 queries)")
    lines.append(f"Corpus: post_fix_rerun_2026-04-19  chunk=224/56")
    lines.append("=" * 72)
    lines.append(f"  {'k1':>5} {'b':>5}  {'hit@1':>8} {'hit@5':>8} {'hit@10':>8} {'mrr@10':>9}")
    lines.append("  " + "-" * 58)
    for _, row in df.iterrows():
        marker = "  ← current" if (row["k1"] == 1.5 and row["b"] == 0.75) else ""
        lines.append(f"  {row['k1']:>5.1f} {row['b']:>5.2f}  "
                     f"{row['hit@1']:>8.4f} {row['hit@5']:>8.4f} "
                     f"{row['hit@10']:>8.4f} {row['mrr@10']:>9.4f}{marker}")
    lines.append("=" * 72)

    best = df.iloc[0]
    lines.append(f"\n  Best MRR@10: k1={best['k1']:.1f}  b={best['b']:.2f}  "
                 f"mrr@10={best['mrr@10']:.4f}  hit@1={best['hit@1']:.4f}")
    current = df[(df["k1"] == 1.5) & (df["b"] == 0.75)].iloc[0]
    delta = best["mrr@10"] - current["mrr@10"]
    lines.append(f"  Current:     k1=1.5  b=0.75    "
                 f"mrr@10={current['mrr@10']:.4f}  hit@1={current['hit@1']:.4f}")
    lines.append(f"  Δmrr@10 from tuning: {delta:+.4f}")

    summary = "\n".join(lines)
    print("\n" + summary)
    out_path.write_text(summary)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--top-k", type=int, default=10)
    args = p.parse_args()

    out_dir = REPO_ROOT / "results" / f"bm25_param_ablation_{date.today().isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"BM25 grid search — k1={K1_VALUES}  b={B_VALUES}")
    print(f"top_k={args.top_k}  docs={len(DOCS)}  queries=250")
    print("=" * 65, flush=True)

    results = run_grid(args.top_k)

    # Save raw results
    serialisable = {str(k1): {str(b): v for b, v in b_dict.items()}
                    for k1, b_dict in results.items()}
    (out_dir / "results.json").write_text(json.dumps(serialisable, indent=2))

    # Heatmaps
    print("\nGenerating heatmaps...", flush=True)
    for metric, title in [
        ("mrr@10", "BM25 parameter ablation — MRR@10 (250 queries, 5 docs)"),
        ("hit@1",  "BM25 parameter ablation — Hit@1 (250 queries, 5 docs)"),
        ("hit@5",  "BM25 parameter ablation — Hit@5 (250 queries, 5 docs)"),
    ]:
        mat = _build_matrix(results, metric)
        _plot_heatmap(mat, metric, title, out_dir / f"heatmap_{metric.replace('@','')}")

    _print_and_save_summary(results, out_dir / "summary.txt")
    print(f"\n  Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
