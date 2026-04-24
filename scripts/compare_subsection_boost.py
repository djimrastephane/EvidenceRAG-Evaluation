"""compare_subsection_boost.py

Re-runs retrieval on the existing post-fix 224/56 pipeline outputs with
enable_subsection_boost=True and enable_subsection_boost=False and prints
a side-by-side metric comparison.

Usage:
    python scripts/compare_subsection_boost.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from thesis_rag.artifacts import load_queries
from thesis_rag.evaluator import evaluate_page_hits, hit_at_k
from thesis_rag.ranking import chunk_hits_to_page_hits
from thesis_rag.retrieval_hybrid import hybrid_retrieve_legacy_style
from thesis_rag.retrieval_sparse import build_bm25, sparse_retrieve_legacy_style
from thesis_rag.retrieval_dense import dense_retrieve_legacy_style, search_faiss_stably
from thesis_rag.utils import l2_normalize
import faiss
import pandas as pd

PIPELINE_OUTPUTS = REPO_ROOT / "results/thesis_ablations/post_fix_rerun_2026-04-19/pipeline_outputs"
EVAL_ROOT = REPO_ROOT / "data_processed"
DOCS = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]
RRF_K = 20
DENSE_WEIGHT = 0.5
BM25_WEIGHT = 2.0


def _load_chunks(exp_dir: Path):
    from thesis_rag.schemas import ChunkRecord
    df = pd.read_parquet(exp_dir / "chunks.parquet")
    chunks = []
    for row in df.to_dict(orient="records"):
        pages_raw = row.get("pages")
        pages = list(pages_raw) if isinstance(pages_raw, list) else [int(row.get("page_start") or row.get("page_number") or 0)]
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


def _metrics(queries, hybrid_chunk_hits) -> dict:
    page_hits_10 = chunk_hits_to_page_hits(hybrid_chunk_hits, "hybrid_pages", chunk_limit=10)

    grouped: dict = defaultdict(list)
    for h in page_hits_10:
        grouped[h.query_id].append(h.page_number)

    h1 = h3 = h5 = h10 = mrr = 0
    for q in queries:
        pages = grouped.get(q.query_id, [])
        h1  += int(hit_at_k(pages, q.gold_pages, 1))
        h3  += int(hit_at_k(pages, q.gold_pages, 3))
        h5  += int(hit_at_k(pages, q.gold_pages, 5))
        h10 += int(hit_at_k(pages, q.gold_pages, 10))
        rr = next((1/(i+1) for i, p in enumerate(pages[:10]) if p in set(q.gold_pages)), 0.0)
        mrr += rr

    n = max(len(queries), 1)
    return {"hit@1": h1/n, "hit@3": h3/n, "hit@5": h5/n, "hit@10": h10/n, "mrr@10": mrr/n, "n": len(queries)}


def run_doc(doc_id: str, model, apply_l2: bool, config) -> tuple[dict, dict]:
    exp_dir = PIPELINE_OUTPUTS / f"minilmcap_{doc_id}_chunk_224_56" / doc_id

    chunks = _load_chunks(exp_dir)
    queries = load_queries(EVAL_ROOT / doc_id / "eval_set.json")
    index = faiss.read_index(str(exp_dir / "faiss.index"))

    q_texts = [q.query_text for q in queries]
    q_vecs = model.encode(q_texts, batch_size=32, show_progress_bar=False,
                          convert_to_numpy=True, normalize_embeddings=False).astype("float32")
    if apply_l2:
        q_vecs = l2_normalize(q_vecs)

    bm25 = build_bm25(chunks, config.bm25)
    raw_scores, raw_indices = search_faiss_stably(index, q_vecs, min(100, len(chunks)))

    results = {}
    for boost in (True, False):
        _, _, hybrid_hits = hybrid_retrieve_legacy_style(
            chunks=chunks,
            queries=queries,
            dense_scores=raw_scores,
            dense_indices=raw_indices,
            bm25=bm25,
            max_k_search=100,
            dense_weight=DENSE_WEIGHT,
            bm25_weight=BM25_WEIGHT,
            rrf_k=RRF_K,
            enable_subsection_boost=boost,
            enable_lexical_rerank=True,
        )
        results[boost] = _metrics(queries, hybrid_hits)

    return results[True], results[False]


def main() -> None:
    from sentence_transformers import SentenceTransformer
    from thesis_rag.config import load_config
    from thesis_rag.utils import resolve_device
    config = load_config(REPO_ROOT / "configs/thesis_rag.yaml")
    device = resolve_device(config.runtime.device)
    print("Loading embedding model...", flush=True)
    model = SentenceTransformer(str(REPO_ROOT / "models/all-MiniLM-L6-v2"), device=device)
    apply_l2 = config.embedding.apply_l2_normalization

    rows = []
    for doc_id in DOCS:
        print(f"  Running {doc_id}...", flush=True)
        with_boost, without_boost = run_doc(doc_id, model, apply_l2, config)
        rows.append({
            "document": doc_id,
            "boost=True  hit@1": with_boost["hit@1"],
            "boost=False hit@1": without_boost["hit@1"],
            "Δhit@1": with_boost["hit@1"] - without_boost["hit@1"],
            "boost=True  mrr@10": with_boost["mrr@10"],
            "boost=False mrr@10": without_boost["mrr@10"],
            "Δmrr@10": with_boost["mrr@10"] - without_boost["mrr@10"],
            "boost=True  hit@5": with_boost["hit@5"],
            "boost=False hit@5": without_boost["hit@5"],
            "Δhit@5": with_boost["hit@5"] - without_boost["hit@5"],
        })

    print()
    print("=" * 90)
    print("  Subsection boost impact  (chunk=224/56, rrf_k=20, dense=0.5, bm25=2.0)")
    print("=" * 90)
    hdr = f"  {'Document':<22} {'B=T hit@1':>9} {'B=F hit@1':>9} {'Δhit@1':>7}  {'B=T hit@5':>9} {'B=F hit@5':>9} {'Δhit@5':>7}  {'B=T mrr@10':>10} {'B=F mrr@10':>10} {'Δmrr@10':>8}"
    print(hdr)
    print("  " + "-" * 87)
    for r in rows:
        print(
            f"  {r['document']:<22}"
            f" {r['boost=True  hit@1']:>9.3f}"
            f" {r['boost=False hit@1']:>9.3f}"
            f" {r['Δhit@1']:>+7.3f}"
            f"  {r['boost=True  hit@5']:>9.3f}"
            f" {r['boost=False hit@5']:>9.3f}"
            f" {r['Δhit@5']:>+7.3f}"
            f"  {r['boost=True  mrr@10']:>10.3f}"
            f" {r['boost=False mrr@10']:>10.3f}"
            f" {r['Δmrr@10']:>+8.3f}"
        )
    print("  " + "-" * 87)
    avg_d1  = sum(r["Δhit@1"]  for r in rows) / len(rows)
    avg_d5  = sum(r["Δhit@5"]  for r in rows) / len(rows)
    avg_dmr = sum(r["Δmrr@10"] for r in rows) / len(rows)
    print(f"  {'Average delta':<22} {'':>9} {'':>9} {avg_d1:>+7.3f}  {'':>9} {'':>9} {avg_d5:>+7.3f}  {'':>10} {'':>10} {avg_dmr:>+8.3f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
