#!/usr/bin/env python3
"""Rerun CE ablation using frozen 224/56 artifacts + current eval_set.json.

Replaces the discussion.tex CE table (previously run on data_processed with
baseline 0.676). Now uses frozen artifacts for a consistent baseline with
Table 4.1.

Configurations mirror the original CE ablation:
  baseline       : hybrid, no CE
  ce_topn5_w02   : CE top-5,  weight=0.2  (best config from original run)
  ce_topn10_w02  : CE top-10, weight=0.2
  ce_topn20_w02  : CE top-20, weight=0.2
  ce_topn20_w01  : CE top-20, weight=0.1
  ce_topn20_w03  : CE top-20, weight=0.3

Output: results/rerun_ce_ablation_2026-04-24/results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thesis_rag.artifacts import load_queries
from thesis_rag.evaluator import aggregate_metrics, evaluate_page_hits
from thesis_rag.ranking import chunk_hits_to_page_hits
from thesis_rag.retrieval_dense import search_faiss_stably
from thesis_rag.retrieval_hybrid import hybrid_retrieve_legacy_style
from thesis_rag.retrieval_sparse import build_bm25
from thesis_rag.schemas import BM25Config, ChunkRecord, QueryRecord, RetrievalConfig, RetrievalHit

DOCS = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]
OFF_ROOT   = ROOT / "results" / "thesis_ablations" / "chunk_size_ablation_boost_off_2026-04-20" / "pipeline_outputs"
EVAL_ROOT  = ROOT / "data_processed"
MODEL_DIR  = ROOT / "models"
CE_MODEL_PATH = MODEL_DIR / "cross-encoder-ms-marco-MiniLM-L-6-v2"
OUTPUT_DIR = ROOT / "results" / "rerun_ce_ablation_2026-04-24"

RETRIEVAL_CFG = RetrievalConfig(
    dense_top_k=20, sparse_top_k=20, hybrid_top_k=20,
    rrf_k=20, dense_weight=0.5, sparse_weight=2.0,
)
BM25_CFG = BM25Config(k1=1.5, b=0.75)

CE_CONFIGS = [
    ("baseline",     False, 20, 0.2),
    ("ce_topn5_w02",  True,  5, 0.2),
    ("ce_topn10_w02", True, 10, 0.2),
    ("ce_topn20_w02", True, 20, 0.2),
    ("ce_topn20_w01", True, 20, 0.1),
    ("ce_topn20_w03", True, 20, 0.3),
]

LABELS = {
    "baseline":      "Hybrid (base)",
    "ce_topn5_w02":  "CE top-5, w=0.2",
    "ce_topn10_w02": "CE top-10, w=0.2",
    "ce_topn20_w02": "CE top-20, w=0.2",
    "ce_topn20_w01": "CE top-20, w=0.1",
    "ce_topn20_w03": "CE top-20, w=0.3",
}


def load_chunks(path: Path) -> list[ChunkRecord]:
    df = pd.read_parquet(path)
    chunks = []
    for idx, row in enumerate(df.itertuples(index=False)):
        raw_pages = getattr(row, "pages", None)
        pages = (
            [int(p) for p in raw_pages.tolist()] if hasattr(raw_pages, "tolist")
            else [int(p) for p in raw_pages] if raw_pages and len(raw_pages)
            else [int(row.page_start)]
        )
        chunks.append(ChunkRecord(
            chunk_id=str(row.chunk_id),
            doc_id=str(row.doc_id),
            page_number=int(row.page_start),
            chunk_index=idx,
            text=str(getattr(row, "text", "") or ""),
            token_count=int(row.token_count),
            word_count=int(row.word_count),
            chunk_id_global=str(getattr(row, "chunk_id_global", "") or ""),
            page_start=int(row.page_start),
            page_end=int(row.page_end),
            pages=pages,
            part=str(getattr(row, "part", "") or ""),
            section_title=str(getattr(row, "section_title", "") or ""),
            subsection_title=str(getattr(row, "subsection_title", "") or "") or None,
            is_table=bool(row.is_table),
            table_type=str(getattr(row, "table_type", "") or ""),
            table_chunk_kind=str(getattr(row, "table_chunk_kind", "") or ""),
            segment_boundary_type=None,
            segment_has_search_hit=False,
        ))
    return chunks


def apply_ce_rerank(
    hits: list[RetrievalHit],
    ce_model: CrossEncoder,
    topn: int,
    weight: float,
) -> list[RetrievalHit]:
    """Re-rank top-N hits per query using cross-encoder scores."""
    from collections import defaultdict
    by_query: dict[str, list[RetrievalHit]] = defaultdict(list)
    for h in hits:
        by_query[h.query_id].append(h)

    reranked = []
    for qid, qhits in by_query.items():
        qhits.sort(key=lambda h: h.rank)
        top = qhits[:topn]
        rest = qhits[topn:]

        pairs = [[h.query_text, h.text] for h in top]
        ce_raw = ce_model.predict(pairs, convert_to_numpy=True)
        ce_scores = torch.sigmoid(torch.tensor(ce_raw)).numpy()

        # Normalise RRF scores to [0,1]
        rrf_scores = np.array([h.score for h in top])
        rrf_min, rrf_max = rrf_scores.min(), rrf_scores.max()
        rrf_norm = (rrf_scores - rrf_min) / (rrf_max - rrf_min + 1e-12)

        blended = (1 - weight) * rrf_norm + weight * ce_scores
        order = np.argsort(-blended)

        for new_rank, idx in enumerate(order, start=1):
            h = top[idx]
            reranked.append(RetrievalHit(
                query_id=h.query_id, query_text=h.query_text,
                rank=new_rank, score=float(blended[idx]),
                retrieval_method="ce_hybrid",
                doc_id=h.doc_id, page_number=h.page_number,
                chunk_id=h.chunk_id, pages=h.pages, text=h.text,
            ))
        for i, h in enumerate(rest, start=topn + 1):
            reranked.append(RetrievalHit(
                query_id=h.query_id, query_text=h.query_text,
                rank=i, score=h.score,
                retrieval_method="ce_hybrid",
                doc_id=h.doc_id, page_number=h.page_number,
                chunk_id=h.chunk_id, pages=h.pages, text=h.text,
            ))
    return reranked


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading cross-encoder from {CE_MODEL_PATH}...")
    ce_model = CrossEncoder(str(CE_MODEL_PATH))
    print("Loaded.\n")

    from sentence_transformers import SentenceTransformer
    print("Loading sentence-transformer for query embedding...")
    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Pre-load artifacts and pre-compute query vectors once per doc
    doc_data = []
    for doc in DOCS:
        art_dir = OFF_ROOT / f"minilmcap_{doc}_chunk_224_56" / doc
        chunks  = load_chunks(art_dir / "chunk_metadata.parquet")
        queries = load_queries(EVAL_ROOT / doc / "eval_set.json")
        idx     = faiss.read_index(str(art_dir / "faiss.index"))
        q_vecs  = st_model.encode(
            [q.query_text for q in queries],
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
        ).astype(np.float32)
        doc_data.append((chunks, queries, q_vecs, idx))
        print(f"  {doc}: {len(chunks)} chunks, {len(queries)} queries")

    results = {}
    baseline_h1 = None

    for cfg_name, use_ce, topn, weight in CE_CONFIGS:
        print(f"\nRunning {cfg_name}...")
        all_queries, all_hits = [], []

        for chunks, queries, embs, faiss_idx in doc_data:
            all_queries.extend(queries)
            max_k = min(RETRIEVAL_CFG.hybrid_top_k, len(chunks))

            # Embed queries with the stored embeddings' normalisation
            # Use stored FAISS index; embed queries via sentence-transformers
            from sentence_transformers import SentenceTransformer
            _model_name = "sentence-transformers/all-MiniLM-L6-v2"
            _model = SentenceTransformer(_model_name)
            q_vecs = _model.encode(
                [q.query_text for q in queries],
                normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
            ).astype(np.float32)

            d_scores, d_indices = search_faiss_stably(faiss_idx, q_vecs, max_k)
            bm25 = build_bm25(chunks, BM25_CFG)
            _, _, chunk_hits = hybrid_retrieve_legacy_style(
                chunks=chunks, queries=queries,
                dense_scores=d_scores, dense_indices=d_indices,
                bm25=bm25, max_k_search=max_k,
                dense_weight=RETRIEVAL_CFG.dense_weight,
                bm25_weight=RETRIEVAL_CFG.sparse_weight,
                rrf_k=RETRIEVAL_CFG.rrf_k,
                enable_lexical_rerank=True,
                enable_subsection_boost=False,
                subsection_boost=0.05,
                cross_page_out_of_section_penalty=0.08,
            )

            if use_ce:
                chunk_hits = apply_ce_rerank(chunk_hits, ce_model, topn=topn, weight=weight)

            page_hits = chunk_hits_to_page_hits(chunk_hits, "hybrid_pages")
            all_hits.extend(page_hits)

        eval_res = evaluate_page_hits(all_queries, all_hits)
        m = aggregate_metrics(eval_res, ks=[1, 3, 10])
        h1  = round(m["hit@1"], 4)
        mrr = round(m["mrr"],   4)

        if cfg_name == "baseline":
            baseline_h1 = h1
            delta_str = "--"
        else:
            delta_str = f"+{h1 - baseline_h1:.3f}" if baseline_h1 is not None else "--"

        print(f"  Hit@1={h1:.4f}  MRR@10={mrr:.4f}  ΔHit@1={delta_str}")
        results[cfg_name] = {
            "label": LABELS[cfg_name],
            "hit@1": h1,
            "mrr@10": mrr,
            "delta_hit1": round(h1 - baseline_h1, 4) if baseline_h1 is not None else None,
        }

    print("\n=== CE ABLATION TABLE (frozen artifacts + current eval_set) ===")
    print(f"{'Configuration':<25} {'Hit@1':>8} {'MRR@10':>8} {'ΔHit@1':>9}")
    print("-" * 54)
    for cfg_name, r in results.items():
        d = f"+{r['delta_hit1']:.3f}" if r["delta_hit1"] is not None else "--"
        print(f"{r['label']:<25} {r['hit@1']:>8.4f} {r['mrr@10']:>8.4f} {d:>9}")

    out_path = OUTPUT_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
