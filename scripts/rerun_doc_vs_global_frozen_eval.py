#!/usr/bin/env python3
"""Recompute doc-vs-global ablation using frozen artifacts + current eval_set.json.

Doc-constrained (boost OFF): re-evaluates hybrid_page_hits.csv from boost-OFF
  frozen artifacts against current gold pages.

Doc-constrained (boost ON): re-evaluates hybrid_page_hits.csv from boost-ON
  frozen artifacts against current gold pages.

Global (boost ON): loads all five docs simultaneously and runs cross-document
  hybrid retrieval using all five frozen FAISS indices merged. Requires
  re-running retrieval because the global CSV was not pre-computed.

Output: results/rerun_doc_vs_global_2026-04-24/results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

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
ON_ROOT    = ROOT / "results" / "thesis_ablations" / "chunk_size_ablation_2026-04-15"           / "pipeline_outputs"
EVAL_ROOT  = ROOT / "data_processed"
OUTPUT_DIR = ROOT / "results" / "rerun_doc_vs_global_2026-04-24"

RETRIEVAL_CFG = RetrievalConfig(
    dense_top_k=20, sparse_top_k=20, hybrid_top_k=20,
    rrf_k=20, dense_weight=0.5, sparse_weight=2.0,
)
BM25_CFG = BM25Config(k1=1.5, b=0.75)


def load_gold_map() -> dict[str, dict]:
    gold: dict[str, dict] = {}
    for doc in DOCS:
        data = json.loads((EVAL_ROOT / doc / "eval_set.json").read_text())
        for q in data["queries"]:
            gold[q["query_id"]] = {
                "gold_pages": set(q["expected_pages"]),
                "doc_id": q["doc_id"],
            }
    return gold


def hits_from_csv(path: Path, gold_map: dict) -> list[dict]:
    df = pd.read_csv(path, usecols=["query_id", "rank", "page_number"])
    rows = []
    for qid, grp in df.groupby("query_id"):
        if qid not in gold_map:
            continue
        gold = gold_map[qid]["gold_pages"]
        ranked = grp.sort_values("rank")["page_number"].tolist()
        first_rel = next((i + 1 for i, pg in enumerate(ranked) if pg in gold), None)
        rows.append({
            "query_id": qid,
            "hit@1":  1.0 if set(ranked[:1]) & gold else 0.0,
            "hit@3":  1.0 if set(ranked[:3]) & gold else 0.0,
            "hit@5":  1.0 if set(ranked[:5]) & gold else 0.0,
            "hit@10": 1.0 if set(ranked[:10]) & gold else 0.0,
            "mrr@10": (1.0 / first_rel) if first_rel and first_rel <= 10 else 0.0,
        })
    return rows


def load_chunks(path: Path, doc_id: str) -> list[ChunkRecord]:
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
            doc_id=doc_id,
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


def run_global_boost_on(gold_map: dict) -> dict:
    """Run cross-document retrieval (all 5 docs in one index) with boost ON."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Build global FAISS index and chunk list
    all_chunks: list[ChunkRecord] = []
    for doc in DOCS:
        art_dir = ON_ROOT / f"minilmcap_{doc}_chunk_224_56" / doc
        chunks  = load_chunks(art_dir / "chunk_metadata.parquet", doc)
        all_chunks.extend(chunks)

    # Embed all chunks
    emb_dim = 384
    global_idx = faiss.IndexFlatIP(emb_dim)
    for doc in DOCS:
        art_dir = ON_ROOT / f"minilmcap_{doc}_chunk_224_56" / doc
        embs = np.load(art_dir / "embeddings.npy").astype(np.float32)
        global_idx.add(embs)

    # All queries (all 5 docs)
    all_queries: list[QueryRecord] = []
    for doc in DOCS:
        all_queries.extend(load_queries(EVAL_ROOT / doc / "eval_set.json"))

    # Embed queries
    q_vecs = model.encode(
        [q.query_text for q in all_queries],
        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True,
    ).astype(np.float32)

    max_k = min(RETRIEVAL_CFG.hybrid_top_k, len(all_chunks))
    d_scores, d_indices = search_faiss_stably(global_idx, q_vecs, max_k)
    bm25 = build_bm25(all_chunks, BM25_CFG)

    _, _, chunk_hits = hybrid_retrieve_legacy_style(
        chunks=all_chunks, queries=all_queries,
        dense_scores=d_scores, dense_indices=d_indices,
        bm25=bm25, max_k_search=max_k,
        dense_weight=RETRIEVAL_CFG.dense_weight,
        bm25_weight=RETRIEVAL_CFG.sparse_weight,
        rrf_k=RETRIEVAL_CFG.rrf_k,
        enable_lexical_rerank=True,
        enable_subsection_boost=True,
        subsection_boost=0.05,
        cross_page_out_of_section_penalty=0.08,
    )
    page_hits = chunk_hits_to_page_hits(chunk_hits, "global_hybrid")

    # Evaluate: compare retrieved page to gold page (cross-doc: all pages visible)
    eval_res = evaluate_page_hits(all_queries, page_hits)
    m = aggregate_metrics(eval_res, ks=[1, 3, 5, 10])
    return {k: round(v, 4) for k, v in m.items()}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gold_map = load_gold_map()

    # Doc-constrained, boost OFF
    print("Computing doc-constrained boost-OFF...")
    off_frames = []
    for doc in DOCS:
        p = OFF_ROOT / f"minilmcap_{doc}_chunk_224_56" / doc / "hybrid_page_hits.csv"
        off_frames.append(pd.DataFrame(hits_from_csv(p, gold_map)))
    off_df = pd.concat(off_frames, ignore_index=True)
    off_m = {
        "hit@1":  round(off_df["hit@1"].mean(), 4),
        "hit@3":  round(off_df["hit@3"].mean(), 4),
        "hit@5":  round(off_df["hit@5"].mean(), 4),
        "hit@10": round(off_df["hit@10"].mean(), 4),
        "mrr@10": round(off_df["mrr@10"].mean(), 4),
    }
    print(f"  Doc-constrained boost-OFF: H@1={off_m['hit@1']:.4f}  MRR={off_m['mrr@10']:.4f}")

    # Doc-constrained, boost ON
    print("Computing doc-constrained boost-ON...")
    on_frames = []
    for doc in DOCS:
        p = ON_ROOT / f"minilmcap_{doc}_chunk_224_56" / doc / "hybrid_page_hits.csv"
        on_frames.append(pd.DataFrame(hits_from_csv(p, gold_map)))
    on_df = pd.concat(on_frames, ignore_index=True)
    on_m = {
        "hit@1":  round(on_df["hit@1"].mean(), 4),
        "hit@3":  round(on_df["hit@3"].mean(), 4),
        "hit@5":  round(on_df["hit@5"].mean(), 4),
        "hit@10": round(on_df["hit@10"].mean(), 4),
        "mrr@10": round(on_df["mrr@10"].mean(), 4),
    }
    print(f"  Doc-constrained boost-ON:  H@1={on_m['hit@1']:.4f}  MRR={on_m['mrr@10']:.4f}")

    # Global, boost ON
    print("Computing global boost-ON (requires embedding all queries)...")
    global_m = run_global_boost_on(gold_map)
    print(f"  Global boost-ON:           H@1={global_m['hit@1']:.4f}  MRR={global_m['mrr']:.4f}")

    print("\n=== DOC vs GLOBAL TABLE ===")
    print(f"{'Configuration':<35} {'H@1':>6} {'H@3':>6} {'H@5':>6} {'H@10':>6} {'MRR@10':>8}")
    print("-" * 68)
    print(f"{'Doc-constrained, boost OFF':<35} {off_m['hit@1']:>6.3f} {off_m['hit@3']:>6.3f} {off_m['hit@5']:>6.3f} {off_m['hit@10']:>6.3f} {off_m['mrr@10']:>8.4f}")
    print(f"{'Doc-constrained, boost ON':<35} {on_m['hit@1']:>6.3f} {on_m['hit@3']:>6.3f} {on_m['hit@5']:>6.3f} {on_m['hit@10']:>6.3f} {on_m['mrr@10']:>8.4f}")
    print(f"{'Global, boost ON':<35} {global_m['hit@1']:>6.3f} {global_m['hit@3']:>6.3f} {global_m['hit@5']:>6.3f} {global_m['hit@10']:>6.3f} {global_m['mrr']:>8.4f}")

    results = {
        "doc_constrained_boost_off": off_m,
        "doc_constrained_boost_on":  on_m,
        "global_boost_on": global_m,
    }
    out_path = OUTPUT_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
