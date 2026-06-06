"""ablate_rrf_weights.py

Grid search over RRF fusion weights (w_dense × w_bm25) to validate the
thesis claim that dense_weight=0.5 / bm25_weight=2.0 was selected from
180 tested configurations.

Grid: 10 dense_weight values × 18 bm25_weight values = 180 combinations.

Dense scores are computed once per document; BM25 is built once per document
(k1=1.5, b=0.75 fixed at pipeline defaults). All 180 weight combinations are
applied in-memory — no re-embedding or re-indexing required.

Usage:
    python scripts/ablate_rrf_weights.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
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

RRF_K = 20
TOP_K = 10

# Promoted configuration (thesis claim)
PROMOTED_DENSE = 0.5
PROMOTED_BM25  = 2.0

# 10 × 18 = 180 combinations
DENSE_WEIGHTS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
BM25_WEIGHTS  = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0,
                 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]


# ---------------------------------------------------------------------------
# Chunk loader (identical to ablate_bm25_hybrid.py)
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

def _metrics(queries, hybrid_hits) -> dict:
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
    bm25_cfg = BM25Config(k1=1.5, b=0.75)

    n_combos = len(DENSE_WEIGHTS) * len(BM25_WEIGHTS)
    print(f"RRF weight grid search (subsection_boost=OFF): {len(DENSE_WEIGHTS)} dense × {len(BM25_WEIGHTS)} bm25 = {n_combos} combinations", flush=True)

    print("Loading embedding model...", flush=True)
    model = SentenceTransformer(str(REPO_ROOT / "models/all-MiniLM-L6-v2"), device=device)

    print("Pre-computing dense scores and BM25 indexes for all docs...", flush=True)
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
        bm25 = build_bm25(chunks, bm25_cfg)

        doc_cache.append({
            "doc_id": doc_id,
            "chunks": chunks,
            "queries": queries,
            "raw_scores": raw_scores,
            "raw_indices": raw_indices,
            "bm25": bm25,
        })
        print(f"  {doc_id}: ready ({len(chunks)} chunks, {len(queries)} queries)", flush=True)

    total_queries = sum(len(dc["queries"]) for dc in doc_cache)
    print(f"\nTotal queries across all docs: {total_queries}")
    print(f"Sweeping {n_combos} weight combinations...\n", flush=True)

    all_results: list[dict] = []

    for i, dw in enumerate(DENSE_WEIGHTS):
        for j, bw in enumerate(BM25_WEIGHTS):
            combo_num = i * len(BM25_WEIGHTS) + j + 1
            agg = {"hit@1": 0.0, "hit@5": 0.0, "hit@10": 0.0, "mrr@10": 0.0, "n": 0}

            for dc in doc_cache:
                _, _, hybrid_hits = hybrid_retrieve_legacy_style(
                    chunks=dc["chunks"],
                    queries=dc["queries"],
                    dense_scores=dc["raw_scores"],
                    dense_indices=dc["raw_indices"],
                    bm25=dc["bm25"],
                    max_k_search=100,
                    dense_weight=dw,
                    bm25_weight=bw,
                    rrf_k=RRF_K,
                    enable_subsection_boost=False,
                    enable_lexical_rerank=True,
                )
                m = _metrics(dc["queries"], hybrid_hits)
                for key in ("hit@1", "hit@5", "hit@10", "mrr@10"):
                    agg[key] += m[key] * m["n"]
                agg["n"] += m["n"]

            n = agg["n"]
            row = {
                "dense_weight": dw,
                "bm25_weight": bw,
                "hit@1":  agg["hit@1"]  / n,
                "hit@5":  agg["hit@5"]  / n,
                "hit@10": agg["hit@10"] / n,
                "mrr@10": agg["mrr@10"] / n,
                "promoted": (dw == PROMOTED_DENSE and bw == PROMOTED_BM25),
            }
            all_results.append(row)

            marker = " *** PROMOTED ***" if row["promoted"] else ""
            print(f"  [{combo_num:3d}/{n_combos}]  dw={dw:<5}  bw={bw:<6}  "
                  f"hit@1={row['hit@1']:.4f}  hit@5={row['hit@5']:.4f}  "
                  f"mrr@10={row['mrr@10']:.4f}{marker}", flush=True)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    promoted = next(r for r in all_results if r["promoted"])
    best_mrr = max(all_results, key=lambda r: r["mrr@10"])
    best_h1  = max(all_results, key=lambda r: r["hit@1"])

    print()
    print("=" * 72)
    print(f"  Promoted config  dw={PROMOTED_DENSE}  bw={PROMOTED_BM25}:")
    print(f"    hit@1={promoted['hit@1']:.4f}  hit@5={promoted['hit@5']:.4f}  "
          f"hit@10={promoted['hit@10']:.4f}  mrr@10={promoted['mrr@10']:.4f}")
    print()
    print(f"  Best MRR@10  dw={best_mrr['dense_weight']}  bw={best_mrr['bm25_weight']}:")
    print(f"    hit@1={best_mrr['hit@1']:.4f}  hit@5={best_mrr['hit@5']:.4f}  "
          f"mrr@10={best_mrr['mrr@10']:.4f}")
    print()
    print(f"  Best Hit@1   dw={best_h1['dense_weight']}  bw={best_h1['bm25_weight']}:")
    print(f"    hit@1={best_h1['hit@1']:.4f}  hit@5={best_h1['hit@5']:.4f}  "
          f"mrr@10={best_h1['mrr@10']:.4f}")
    print()
    print(f"  Δ MRR@10 (best − promoted): {best_mrr['mrr@10'] - promoted['mrr@10']:+.4f}")
    print(f"  Δ hit@1  (best − promoted): {best_h1['hit@1']  - promoted['hit@1']:+.4f}")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # Heatmaps
    # -----------------------------------------------------------------------
    out_dir = REPO_ROOT / "results" / f"rrf_weight_ablation_no_boost_{date.today().isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _make_grid(metric: str) -> np.ndarray:
        grid = np.zeros((len(DENSE_WEIGHTS), len(BM25_WEIGHTS)))
        for r in all_results:
            i = DENSE_WEIGHTS.index(r["dense_weight"])
            j = BM25_WEIGHTS.index(r["bm25_weight"])
            grid[i, j] = r[metric]
        return grid

    for metric, title in [("mrr@10", "MRR@10"), ("hit@1", "Hit@1")]:
        grid = _make_grid(metric)

        fig, ax = plt.subplots(figsize=(14, 5))
        im = ax.imshow(grid, aspect="auto", cmap="YlOrRd",
                       vmin=grid.min() * 0.995, vmax=grid.max() * 1.005)
        plt.colorbar(im, ax=ax, label=title)

        ax.set_xticks(range(len(BM25_WEIGHTS)))
        ax.set_xticklabels([str(w) for w in BM25_WEIGHTS], fontsize=8)
        ax.set_yticks(range(len(DENSE_WEIGHTS)))
        ax.set_yticklabels([str(w) for w in DENSE_WEIGHTS], fontsize=8)
        ax.set_xlabel("BM25 weight", fontsize=10)
        ax.set_ylabel("Dense weight", fontsize=10)
        ax.set_title(
            f"RRF fusion weight grid search — {title}  (180 combinations)\n"
            f"RRF k={RRF_K}  BM25 k1=1.5 b=0.75  subsection_boost=True  chunk=224/56",
            fontsize=10,
        )

        # Mark the promoted configuration
        pi = DENSE_WEIGHTS.index(PROMOTED_DENSE)
        pj = BM25_WEIGHTS.index(PROMOTED_BM25)
        ax.add_patch(plt.Rectangle((pj - 0.5, pi - 0.5), 1, 1,
                                   fill=False, edgecolor="#1a6faf", linewidth=2.5,
                                   label=f"promoted (dw={PROMOTED_DENSE}, bw={PROMOTED_BM25})"))
        ax.plot(pj, pi, "x", color="#1a6faf", markersize=10, markeredgewidth=2)

        # Mark the best cell
        bi = DENSE_WEIGHTS.index(best_mrr["dense_weight"] if metric == "mrr@10" else best_h1["dense_weight"])
        bj = BM25_WEIGHTS.index(best_mrr["bm25_weight"]  if metric == "mrr@10" else best_h1["bm25_weight"])
        ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1,
                                   fill=False, edgecolor="#2ca02c", linewidth=2.0,
                                   linestyle="--",
                                   label=f"best {title}"))

        # Annotate cells with values
        for i in range(len(DENSE_WEIGHTS)):
            for j in range(len(BM25_WEIGHTS)):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
                        fontsize=6, color="black")

        ax.legend(loc="lower right", fontsize=8, frameon=True)
        fig.tight_layout()
        fig.savefig(out_dir / f"heatmap_{metric.replace('@', '')}.png", dpi=200, bbox_inches="tight")
        fig.savefig(out_dir / f"heatmap_{metric.replace('@', '')}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {metric} heatmap", flush=True)

    # -----------------------------------------------------------------------
    # JSON output
    # -----------------------------------------------------------------------
    summary = {
        "grid": {
            "dense_weights": DENSE_WEIGHTS,
            "bm25_weights": BM25_WEIGHTS,
            "n_combinations": n_combos,
            "rrf_k": RRF_K,
        },
        "promoted": {
            "dense_weight": PROMOTED_DENSE,
            "bm25_weight": PROMOTED_BM25,
            "hit@1":  promoted["hit@1"],
            "hit@5":  promoted["hit@5"],
            "hit@10": promoted["hit@10"],
            "mrr@10": promoted["mrr@10"],
        },
        "best_mrr10": {
            "dense_weight": best_mrr["dense_weight"],
            "bm25_weight":  best_mrr["bm25_weight"],
            "mrr@10":       best_mrr["mrr@10"],
        },
        "best_hit1": {
            "dense_weight": best_h1["dense_weight"],
            "bm25_weight":  best_h1["bm25_weight"],
            "hit@1":        best_h1["hit@1"],
        },
        "results": all_results,
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
