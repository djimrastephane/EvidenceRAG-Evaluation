#!/usr/bin/env python3
"""FinBERT encoder ablation: dense-only and hybrid, subsection boost OFF.

Compares ProsusAI/finbert (768-dim, mean pooling, L2 normalised) against the
encoders already reported in Table C.3. Uses the same data_processed/ chunks
and eval_set.json used in the original dense encoder ablation (no subsection
boosting, same RRF/BM25 settings).
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thesis_rag.artifacts import load_queries
from thesis_rag.evaluator import aggregate_metrics, evaluate_page_hits
from thesis_rag.retrieval_dense import search_faiss_stably
from thesis_rag.retrieval_hybrid import hybrid_retrieve_legacy_style
from thesis_rag.retrieval_sparse import build_bm25
from thesis_rag.schemas import BM25Config, ChunkRecord, QueryRecord, RetrievalConfig, RetrievalHit

DATA_ROOT = ROOT / "data_processed"
OUTPUT_DIR = ROOT / "results" / "finbert_encoder_ablation_2026-04-24"

DOC_IDS = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]

RETRIEVAL_CFG = RetrievalConfig(
    dense_top_k=20, sparse_top_k=20, hybrid_top_k=20,
    rrf_k=20, dense_weight=0.5, sparse_weight=2.0,
)
BM25_CFG = BM25Config(k1=1.5, b=0.75)

FINBERT_ID = "ProsusAI/finbert"
BATCH_SIZE = 16
MAX_SEQ_LEN = 512


def load_finbert(device: str = "cpu"):
    tok = AutoTokenizer.from_pretrained(FINBERT_ID)
    mdl = AutoModel.from_pretrained(FINBERT_ID, use_safetensors=True)
    mdl.eval().to(device)
    return tok, mdl


def embed_texts(
    texts: list[str], tokenizer, model, device: str = "cpu", batch_size: int = 16
) -> np.ndarray:
    """Mean-pool BERT token embeddings and L2-normalise."""
    parts: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_SEQ_LEN,
        ).to(device)
        with torch.no_grad():
            out = model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        parts.append(pooled.cpu().numpy())
    mat = np.vstack(parts).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / (norms + 1e-12)


def load_chunks(path: Path) -> list[ChunkRecord]:
    df = pd.read_parquet(path)
    chunks: list[ChunkRecord] = []
    for idx, row in enumerate(df.itertuples(index=False)):
        raw_pages = getattr(row, "pages", None)
        if raw_pages is None or not len(raw_pages):
            pages = [int(row.page_start)]
        elif hasattr(raw_pages, "tolist"):
            pages = [int(p) for p in raw_pages.tolist()]
        else:
            pages = [int(p) for p in raw_pages]
        chunks.append(ChunkRecord(
            chunk_id=str(row.chunk_id),
            doc_id=str(row.doc_id),
            page_number=int(row.page_start),
            chunk_index=idx,
            text=str(row.chunk_text or ""),
            token_count=int(row.chunk_tokens),
            word_count=int(row.word_count),
            chunk_id_global=str(getattr(row, "chunk_id_global", "") or ""),
            page_start=int(row.page_start),
            page_end=int(row.page_end),
            pages=pages,
            part=str(getattr(row, "part", "") or ""),
            section_title=str(getattr(row, "section_title", "") or ""),
            subsection_title=None,
            is_table=bool(row.is_table),
            table_type=str(getattr(row, "table_type", "") or ""),
            table_chunk_kind=str(getattr(row, "table_chunk_kind", "") or ""),
            segment_boundary_type=None,
            segment_has_search_hit=False,
        ))
    return chunks


def dense_hits_from_faiss(
    chunks: list[ChunkRecord],
    queries: list[QueryRecord],
    scores: np.ndarray,
    indices: np.ndarray,
    top_k: int,
) -> list[RetrievalHit]:
    hits: list[RetrievalHit] = []
    for q_idx, query in enumerate(queries):
        for rank, (sc, ci) in enumerate(
            zip(scores[q_idx].tolist(), indices[q_idx].tolist()), start=1
        ):
            if ci < 0 or rank > top_k:
                continue
            chunk = chunks[ci]
            hits.append(RetrievalHit(
                query_id=query.query_id,
                query_text=query.query_text,
                rank=rank,
                score=float(sc),
                retrieval_method="dense",
                doc_id=chunk.doc_id,
                page_number=chunk.page_number,
                chunk_id=chunk.chunk_id,
                pages=chunk.pages,
                text=chunk.text,
            ))
    return hits


def group_hits(hits: list[RetrievalHit]) -> dict[str, list[RetrievalHit]]:
    out: dict[str, list[RetrievalHit]] = defaultdict(list)
    for h in hits:
        out[h.query_id].append(h)
    for qid in out:
        out[qid].sort(key=lambda h: h.rank)
    return dict(out)


def count_fp2(eval_results, hits_by_query: dict) -> int:
    n = 0
    for er in eval_results:
        if er.hit_at_1:
            continue
        gold = set(er.gold_pages)
        top10 = {h.page_number for h in hits_by_query.get(er.query_id, [])[:10]}
        if gold & top10:
            n += 1
    return n


def mean_margin(scores: np.ndarray) -> float:
    margins = [float(row[0] - row[1]) for row in scores if len(row) >= 2]
    return float(np.mean(margins)) if margins else 0.0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {FINBERT_ID}...")
    tokenizer, finbert = load_finbert(device="cpu")
    print("Loaded (768-dim, mean pooling, L2 normalised)\n")

    all_queries: list[QueryRecord] = []
    all_dense_hits: list[RetrievalHit] = []
    all_hybrid_hits: list[RetrievalHit] = []
    all_dense_scores: list[np.ndarray] = []

    t_embed = 0.0
    t_dense = 0.0
    t_hybrid = 0.0

    for doc_id in DOC_IDS:
        doc_dir = DATA_ROOT / doc_id
        print(f"--- {doc_id} ---")
        chunks = load_chunks(doc_dir / "chunks.parquet")
        queries = load_queries(doc_dir / "eval_set.json")
        all_queries.extend(queries)
        print(f"  chunks={len(chunks)}  queries={len(queries)}")

        t0 = time.perf_counter()
        chunk_vecs = embed_texts([c.text for c in chunks], tokenizer, finbert, batch_size=BATCH_SIZE)
        faiss_idx = faiss.IndexFlatIP(chunk_vecs.shape[1])
        faiss_idx.add(chunk_vecs)
        q_vecs = embed_texts([q.query_text for q in queries], tokenizer, finbert, batch_size=BATCH_SIZE)
        t_embed += time.perf_counter() - t0

        max_k = min(RETRIEVAL_CFG.hybrid_top_k, len(chunks))

        t1 = time.perf_counter()
        d_scores, d_indices = search_faiss_stably(faiss_idx, q_vecs, max_k)
        t_dense += time.perf_counter() - t1
        all_dense_scores.append(d_scores)
        all_dense_hits.extend(dense_hits_from_faiss(chunks, queries, d_scores, d_indices, max_k))

        t2 = time.perf_counter()
        bm25 = build_bm25(chunks, BM25_CFG)
        _, _, hyb_hits = hybrid_retrieve_legacy_style(
            chunks=chunks,
            queries=queries,
            dense_scores=d_scores,
            dense_indices=d_indices,
            bm25=bm25,
            max_k_search=max_k,
            dense_weight=RETRIEVAL_CFG.dense_weight,
            bm25_weight=RETRIEVAL_CFG.sparse_weight,
            rrf_k=RETRIEVAL_CFG.rrf_k,
            enable_lexical_rerank=True,
            enable_subsection_boost=False,
            subsection_boost=0.05,
            cross_page_out_of_section_penalty=0.08,
        )
        t_hybrid += time.perf_counter() - t2
        all_hybrid_hits.extend(hyb_hits)

    dense_eval = evaluate_page_hits(all_queries, all_dense_hits)
    dense_by_q = group_hits(all_dense_hits)
    dense_m = aggregate_metrics(dense_eval, ks=[1, 3, 10])
    dense_fp2 = count_fp2(dense_eval, dense_by_q)
    dense_margin = mean_margin(np.vstack(all_dense_scores))

    hybrid_eval = evaluate_page_hits(all_queries, all_hybrid_hits)
    hybrid_by_q = group_hits(all_hybrid_hits)
    hybrid_m = aggregate_metrics(hybrid_eval, ks=[1, 3, 10])
    hybrid_fp2 = count_fp2(hybrid_eval, hybrid_by_q)

    rt_dense = t_embed + t_dense
    rt_hybrid = t_embed + t_dense + t_hybrid

    print(f"\n=== FinBERT Encoder Ablation (n=250 queries) ===")
    hdr = f"{'Setup':<10} {'Hit@1':>7} {'Hit@3':>7} {'MRR@10':>8} {'FP2':>5} {'Margin':>9} {'Runtime(s)':>11}"
    print(hdr)
    print(f"{'dense':<10} {dense_m['hit@1']:>7.3f} {dense_m['hit@3']:>7.3f} {dense_m['mrr']:>8.3f} {dense_fp2:>5} {dense_margin:>9.4f} {rt_dense:>11.1f}")
    print(f"{'hybrid':<10} {hybrid_m['hit@1']:>7.3f} {hybrid_m['hit@3']:>7.3f} {hybrid_m['mrr']:>8.3f} {hybrid_fp2:>5} {'--':>9} {rt_hybrid:>11.1f}")

    results = {
        "model": FINBERT_ID,
        "embedding_dim": 768,
        "pooling": "mean",
        "l2_normalized": True,
        "enable_subsection_boost": False,
        "dense": {
            "hit_at_1": round(dense_m["hit@1"], 4),
            "hit_at_3": round(dense_m["hit@3"], 4),
            "mrr_at_10": round(dense_m["mrr"], 4),
            "fp2_count": dense_fp2,
            "mean_margin": round(dense_margin, 6),
            "runtime_s": round(rt_dense, 1),
        },
        "hybrid": {
            "hit_at_1": round(hybrid_m["hit@1"], 4),
            "hit_at_3": round(hybrid_m["hit@3"], 4),
            "mrr_at_10": round(hybrid_m["mrr"], 4),
            "fp2_count": hybrid_fp2,
            "runtime_s": round(rt_hybrid, 1),
        },
    }
    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
