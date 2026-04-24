#!/usr/bin/env python3
"""Compute chunk-level (pre-dedup) Hit@k for chunk-vs-page table using frozen 224/56 artifacts."""
from __future__ import annotations
import json, sys, numpy as np, pandas as pd, faiss
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thesis_rag.artifacts import load_queries
from thesis_rag.retrieval_sparse import build_bm25, sparse_retrieve_legacy_style
from thesis_rag.retrieval_hybrid import hybrid_retrieve_legacy_style
from thesis_rag.retrieval_dense import search_faiss_stably
from thesis_rag.schemas import BM25Config, ChunkRecord, RetrievalConfig

DOCS = ["Grampian-2020-2021","Grampian-2021-2022","Grampian-2022-2023","Grampian-2023-2024","Grampian-2024-2025"]
OFF_ROOT  = ROOT / "results/thesis_ablations/chunk_size_ablation_boost_off_2026-04-20/pipeline_outputs"
ON_ROOT   = ROOT / "results/thesis_ablations/chunk_size_ablation_2026-04-15/pipeline_outputs"
EVAL_ROOT = ROOT / "data_processed"
BM25_CFG  = BM25Config(k1=1.5, b=0.75)
RET_CFG   = RetrievalConfig(dense_top_k=20, sparse_top_k=20, hybrid_top_k=20,
                             rrf_k=20, dense_weight=0.5, sparse_weight=2.0)


def load_chunks(path: Path) -> list[ChunkRecord]:
    df = pd.read_parquet(path)
    chunks = []
    for idx, row in enumerate(df.itertuples(index=False)):
        raw_pages = getattr(row, "pages", None)
        pages = ([int(p) for p in raw_pages.tolist()] if hasattr(raw_pages, "tolist")
                 else [int(p) for p in raw_pages] if raw_pages and len(raw_pages)
                 else [int(row.page_start)])
        chunks.append(ChunkRecord(
            chunk_id=str(row.chunk_id), doc_id=str(row.doc_id),
            page_number=int(row.page_start), chunk_index=idx,
            text=str(getattr(row, "text", "") or ""),
            token_count=int(row.token_count), word_count=int(row.word_count),
            chunk_id_global=str(getattr(row, "chunk_id_global", "") or ""),
            page_start=int(row.page_start), page_end=int(row.page_end), pages=pages,
            part=str(getattr(row, "part", "") or ""),
            section_title=str(getattr(row, "section_title", "") or ""),
            subsection_title=str(getattr(row, "subsection_title", "") or "") or None,
            is_table=bool(row.is_table),
            table_type=str(getattr(row, "table_type", "") or ""),
            table_chunk_kind=str(getattr(row, "table_chunk_kind", "") or ""),
            segment_boundary_type=None, segment_has_search_hit=False,
        ))
    return chunks


def chunk_hit_at_k(hits_by_query: dict, gold_map: dict, k: int) -> float:
    hits = total = 0
    for qid, ranked in hits_by_query.items():
        if qid not in gold_map:
            continue
        gold = gold_map[qid]
        top_k_pages = [p for _, p in sorted(ranked)[:k]]
        if any(p in gold for p in top_k_pages):
            hits += 1
        total += 1
    return hits / total if total > 0 else 0.0


def main() -> None:
    gold_map: dict[str, set] = {}
    for doc in DOCS:
        data = json.loads((EVAL_ROOT / doc / "eval_set.json").read_text())
        for q in data["queries"]:
            gold_map[q["query_id"]] = set(q["expected_pages"])

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    results: dict[str, dict] = {}
    for config_name, art_root, use_boost in [
        ("Dense (MiniLM)",             OFF_ROOT, False),
        ("BM25-only",                  OFF_ROOT, False),
        ("Hybrid (base)",              OFF_ROOT, False),
        ("Hybrid + subsection boost",  ON_ROOT,  True),
    ]:
        print(f"\nComputing chunk hits: {config_name}")
        dense_by_query: dict[str, list] = {}
        bm25_by_query:  dict[str, list] = {}
        hybrid_by_query: dict[str, list] = {}

        for doc in DOCS:
            art_dir = art_root / f"minilmcap_{doc}_chunk_224_56" / doc
            chunks  = load_chunks(art_dir / "chunk_metadata.parquet")
            queries = load_queries(EVAL_ROOT / doc / "eval_set.json")
            faiss_idx = faiss.read_index(str(art_dir / "faiss.index"))
            q_vecs = model.encode(
                [q.query_text for q in queries],
                normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
            ).astype(np.float32)
            max_k = min(20, len(chunks))
            d_scores, d_indices = search_faiss_stably(faiss_idx, q_vecs, max_k)
            bm25 = build_bm25(chunks, BM25_CFG)

            for qi, q in enumerate(queries):
                dense_by_query[q.query_id] = [
                    (r + 1, chunks[ci].page_number)
                    for r, ci in enumerate(d_indices[qi]) if ci < len(chunks)
                ]

            for h in sparse_retrieve_legacy_style(bm25, chunks, queries, top_k=max_k):
                bm25_by_query.setdefault(h.query_id, []).append((h.rank, h.page_number))

            _, _, hyb_hits = hybrid_retrieve_legacy_style(
                chunks=chunks, queries=queries,
                dense_scores=d_scores, dense_indices=d_indices,
                bm25=bm25, max_k_search=max_k,
                dense_weight=RET_CFG.dense_weight, bm25_weight=RET_CFG.sparse_weight,
                rrf_k=RET_CFG.rrf_k, enable_lexical_rerank=True,
                enable_subsection_boost=use_boost, subsection_boost=0.05,
                cross_page_out_of_section_penalty=0.08,
            )
            for h in hyb_hits:
                hybrid_by_query.setdefault(h.query_id, []).append((h.rank, h.page_number))

        if config_name == "Dense (MiniLM)":
            by_q = dense_by_query
        elif config_name == "BM25-only":
            by_q = bm25_by_query
        else:
            by_q = hybrid_by_query

        h1 = chunk_hit_at_k(by_q, gold_map, 1)
        h3 = chunk_hit_at_k(by_q, gold_map, 3)
        results[config_name] = {"chunk_h1": round(h1, 3), "chunk_h3": round(h3, 3)}
        print(f"  Chunk H@1={h1:.4f}  Chunk H@3={h3:.4f}")

    print("\n=== CHUNK HIT SUMMARY ===")
    for name, r in results.items():
        print(f"  {name:<30}: Chunk H@1={r['chunk_h1']:.3f}  Chunk H@3={r['chunk_h3']:.3f}")

    out = ROOT / "results" / "rerun_chunk_hits_2026-04-24" / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
