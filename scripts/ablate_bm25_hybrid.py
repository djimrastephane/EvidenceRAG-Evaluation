"""ablate_bm25_hybrid.py

Stage 2 of BM25 parameter ablation: re-evaluates the full hybrid pipeline
(dense + BM25 + RRF + subsection_boost=True) with the top BM25 k1/b
candidates identified in Stage 1.

Compares hit@1, hit@5, MRR@10 for each k1/b combo against the current
pipeline defaults (k1=1.5, b=0.75).

Usage:
    python scripts/ablate_bm25_hybrid.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from dataclasses import replace
from datetime import date
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import faiss

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH  = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from thesis_rag.artifacts import load_queries
from thesis_rag.evaluator import hit_at_k
from thesis_rag.ranking import chunk_hits_to_page_hits
from thesis_rag.retrieval_hybrid import hybrid_retrieve_legacy_style
from thesis_rag.retrieval_sparse import build_bm25
from thesis_rag.retrieval_dense import search_faiss_stably
from thesis_rag.schemas import BM25Config, ChunkRecord
from thesis_rag.utils import l2_normalize

ARTIFACT_ROOT = REPO_ROOT / "results/thesis_ablations/post_fix_rerun_2026-04-19/pipeline_outputs"
EVAL_ROOT     = REPO_ROOT / "data_processed"
DOCS = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]

RRF_K, DENSE_W, BM25_W = 20, 0.5, 2.0
TOP_K = 10

# Candidates: (label, k1, b)
CANDIDATES = [
    ("k1=1.5 b=0.75 (current)", 1.5, 0.75),
    ("k1=2.0 b=1.00 (best sparse MRR)", 2.0, 1.00),
    ("k1=1.2 b=0.75 (best sparse hit@1)", 1.2, 0.75),
    ("k1=1.0 b=0.75", 1.0, 0.75),
    ("k1=1.5 b=1.00", 1.5, 1.00),
]


# ---------------------------------------------------------------------------
# Chunk loader
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
# Metrics
# ---------------------------------------------------------------------------

def _metrics(chunks, queries, hybrid_hits) -> dict:
    page_hits = chunk_hits_to_page_hits(hybrid_hits, "hybrid_pages", chunk_limit=TOP_K)
    grouped: dict[str, list[int]] = defaultdict(list)
    for h in page_hits:
        grouped[h.query_id].append(h.page_number)

    h1 = h5 = h10 = mrr = 0
    for q in queries:
        pages = grouped.get(q.query_id, [])
        h1  += int(hit_at_k(pages, q.gold_pages, 1))
        h5  += int(hit_at_k(pages, q.gold_pages, 5))
        h10 += int(hit_at_k(pages, q.gold_pages, 10))
        mrr += next((1.0 / (i + 1) for i, p in enumerate(pages[:10])
                     if p in set(q.gold_pages)), 0.0)
    n = max(len(queries), 1)
    return {"hit@1": h1/n, "hit@5": h5/n, "hit@10": h10/n, "mrr@10": mrr/n, "n": len(queries)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from sentence_transformers import SentenceTransformer
    from thesis_rag.config import load_config
    from thesis_rag.utils import resolve_device

    config   = load_config(REPO_ROOT / "configs/thesis_rag.yaml")
    device   = resolve_device(config.runtime.device)
    apply_l2 = config.embedding.apply_l2_normalization

    print("Loading embedding model...", flush=True)
    model = SentenceTransformer(str(REPO_ROOT / "models/all-MiniLM-L6-v2"), device=device)

    # Pre-compute dense scores once per doc (shared across all BM25 configs)
    print("Pre-computing dense scores for all docs...", flush=True)
    doc_cache: list[dict] = []
    for doc_id in DOCS:
        exp_dir = ARTIFACT_ROOT / f"minilmcap_{doc_id}_chunk_224_56" / doc_id
        chunks  = _load_chunks(exp_dir)
        queries = load_queries(EVAL_ROOT / doc_id / "eval_set.json")
        index   = faiss.read_index(str(exp_dir / "faiss.index"))

        q_vecs = model.encode(
            [q.query_text for q in queries],
            batch_size=32, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=False,
        ).astype("float32")
        if apply_l2:
            q_vecs = l2_normalize(q_vecs)

        raw_scores, raw_indices = search_faiss_stably(index, q_vecs, min(100, len(chunks)))
        doc_cache.append({
            "doc_id": doc_id, "chunks": chunks, "queries": queries,
            "raw_scores": raw_scores, "raw_indices": raw_indices,
        })
        print(f"  {doc_id}: dense scores ready", flush=True)

    # Evaluate each candidate
    print()
    all_results: list[dict] = []

    for label, k1, b in CANDIDATES:
        bm25_cfg = BM25Config(k1=k1, b=b)
        agg = {"hit@1": 0.0, "hit@5": 0.0, "hit@10": 0.0, "mrr@10": 0.0, "n": 0}

        for dc in doc_cache:
            bm25 = build_bm25(dc["chunks"], bm25_cfg)
            _, _, hybrid_hits = hybrid_retrieve_legacy_style(
                chunks=dc["chunks"],
                queries=dc["queries"],
                dense_scores=dc["raw_scores"],
                dense_indices=dc["raw_indices"],
                bm25=bm25,
                max_k_search=100,
                dense_weight=DENSE_W,
                bm25_weight=BM25_W,
                rrf_k=RRF_K,
                enable_subsection_boost=True,
                enable_lexical_rerank=True,
            )
            m = _metrics(dc["chunks"], dc["queries"], hybrid_hits)
            for key in ("hit@1", "hit@5", "hit@10", "mrr@10"):
                agg[key] += m[key] * m["n"]
            agg["n"] += m["n"]

        n = agg["n"]
        row = {
            "label": label, "k1": k1, "b": b,
            "hit@1":  agg["hit@1"]  / n,
            "hit@5":  agg["hit@5"]  / n,
            "hit@10": agg["hit@10"] / n,
            "mrr@10": agg["mrr@10"] / n,
        }
        all_results.append(row)
        print(f"  {label:<38}  hit@1={row['hit@1']:.4f}  hit@5={row['hit@5']:.4f}  "
              f"hit@10={row['hit@10']:.4f}  mrr@10={row['mrr@10']:.4f}", flush=True)

    # Summary table
    current = next(r for r in all_results if r["k1"] == 1.5 and r["b"] == 0.75)
    best_mrr = max(all_results, key=lambda r: r["mrr@10"])
    best_h1  = max(all_results, key=lambda r: r["hit@1"])

    print()
    print("=" * 78)
    print(f"  {'Config':<38}  {'hit@1':>7} {'Δhit@1':>7}  {'hit@5':>7}  {'mrr@10':>8} {'Δmrr':>7}")
    print("  " + "-" * 72)
    for r in all_results:
        d1  = r["hit@1"]  - current["hit@1"]
        dmr = r["mrr@10"] - current["mrr@10"]
        marker = " ← current" if (r["k1"] == 1.5 and r["b"] == 0.75) else ""
        print(f"  {r['label']:<38}  {r['hit@1']:>7.4f} {d1:>+7.4f}  "
              f"{r['hit@5']:>7.4f}  {r['mrr@10']:>8.4f} {dmr:>+7.4f}{marker}")
    print("=" * 78)
    print(f"\n  Best hybrid MRR@10 : {best_mrr['label']}  → {best_mrr['mrr@10']:.4f}")
    print(f"  Best hybrid hit@1  : {best_h1['label']}  → {best_h1['hit@1']:.4f}")
    print(f"  Δmrr@10 best vs current: {best_mrr['mrr@10'] - current['mrr@10']:+.4f}")

    # Bar chart
    out_dir = REPO_ROOT / "results" / f"bm25_hybrid_ablation_{date.today().isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels   = [r["label"].split(" (")[0] for r in all_results]
    mrr_vals = [r["mrr@10"] for r in all_results]
    h1_vals  = [r["hit@1"]  for r in all_results]
    colors   = ["#E45756" if (r["k1"] == 1.5 and r["b"] == 0.75) else "#4C78A8"
                for r in all_results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, vals, metric in [(axes[0], mrr_vals, "MRR@10"), (axes[1], h1_vals, "Hit@1")]:
        bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_ylim(min(vals) * 0.97, max(vals) * 1.02)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f"Hybrid {metric} by BM25 k1/b", fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8, rotation=20)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    from matplotlib.patches import Patch
    fig.legend(handles=[
        Patch(facecolor="#E45756", label="current pipeline"),
        Patch(facecolor="#4C78A8", label="candidate"),
    ], loc="upper right", fontsize=9, frameon=False)

    fig.suptitle("Stage 2: Hybrid pipeline re-evaluation with varied BM25 k1/b\n"
                 "RRF k=20  dense_w=0.5  bm25_w=2.0  subsection_boost=True  chunk=224/56",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "hybrid_bm25_comparison.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "hybrid_bm25_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    (out_dir / "results.json").write_text(json.dumps(all_results, indent=2))
    print(f"\n  Saved to: {out_dir}")


if __name__ == "__main__":
    main()
