"""find_balanced_retrieval_examples.py

Re-runs hybrid retrieval (enable_subsection_boost=False) for all three chunk
configurations (224/56, 256/64, 280/90) on the Grampian-2022-2023 document,
then finds the best pair of exemplar queries:
  - One where 224/56 hits rank 1 but B and C do not
  - One where B or C hits rank 1 but 224/56 does not

Outputs a ranked candidate list to stdout and saves results JSON.

Usage:
    python scripts/find_balanced_retrieval_examples.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH  = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import faiss
import numpy as np
import pandas as pd

from thesis_rag.artifacts import load_queries
from thesis_rag.ranking import chunk_hits_to_page_hits
from thesis_rag.retrieval_hybrid import hybrid_retrieve_legacy_style
from thesis_rag.retrieval_sparse import build_bm25
from thesis_rag.retrieval_dense import search_faiss_stably
from thesis_rag.schemas import BM25Config, ChunkRecord
from thesis_rag.utils import l2_normalize

ARTIFACT_ROOT = REPO_ROOT / "results/thesis_ablations/post_fix_rerun_2026-04-19/pipeline_outputs"
EVAL_ROOT     = REPO_ROOT / "data_processed"
DOC_ID        = "Grampian-2022-2023"

CONFIGS = {
    "A_224_56":  "minilmcap_Grampian-2022-2023_chunk_224_56",
    "B_256_64":  "minilmcap_Grampian-2022-2023_chunk_256_64",
    "C_280_90":  "minilmcap_Grampian-2022-2023_chunk_280_90",
}

RRF_K, DENSE_W, BM25_W = 20, 0.5, 2.0
TOP_K = 10


def load_chunks(exp_dir: Path) -> list[ChunkRecord]:
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


def run_hybrid(chunks, queries, raw_scores, raw_indices):
    bm25 = build_bm25(chunks, BM25Config(k1=1.5, b=0.75))
    _, _, hits = hybrid_retrieve_legacy_style(
        chunks=chunks,
        queries=queries,
        dense_scores=raw_scores,
        dense_indices=raw_indices,
        bm25=bm25,
        max_k_search=100,
        dense_weight=DENSE_W,
        bm25_weight=BM25_W,
        rrf_k=RRF_K,
        enable_subsection_boost=False,
        enable_lexical_rerank=True,
    )
    page_hits = chunk_hits_to_page_hits(hits, "hybrid_pages", chunk_limit=TOP_K)
    # Build per-query ranked page list
    ranked: dict[str, list[int]] = defaultdict(list)
    for h in sorted(page_hits, key=lambda x: x.rank):
        ranked[h.query_id].append(h.page_number)
    return ranked


def gold_rank(ranked_pages: list[int], gold_pages: list[int]) -> int | None:
    gold_set = set(gold_pages)
    for i, p in enumerate(ranked_pages, 1):
        if p in gold_set:
            return i
    return None


def rr(rank: int | None) -> float:
    return 1.0 / rank if rank and rank <= TOP_K else 0.0


def main():
    from sentence_transformers import SentenceTransformer
    from thesis_rag.config import load_config
    from thesis_rag.utils import resolve_device

    config   = load_config(REPO_ROOT / "configs/thesis_rag.yaml")
    device   = resolve_device(config.runtime.device)
    apply_l2 = config.embedding.apply_l2_normalization

    print("Loading embedding model...", flush=True)
    model = SentenceTransformer(str(REPO_ROOT / "models/all-MiniLM-L6-v2"), device=device)

    queries = load_queries(EVAL_ROOT / DOC_ID / "eval_set.json")
    q_vecs  = model.encode(
        [q.query_text for q in queries], batch_size=32,
        show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False,
    ).astype("float32")
    if apply_l2:
        q_vecs = l2_normalize(q_vecs)

    results_by_config: dict[str, dict[str, list[int]]] = {}

    for cfg_name, cfg_dir in CONFIGS.items():
        exp_dir = ARTIFACT_ROOT / cfg_dir / DOC_ID
        print(f"Loading {cfg_name}...", flush=True)
        chunks = load_chunks(exp_dir)
        index  = faiss.read_index(str(exp_dir / "faiss.index"))
        raw_scores, raw_indices = search_faiss_stably(index, q_vecs, min(100, len(chunks)))
        ranked = run_hybrid(chunks, queries, raw_scores, raw_indices)
        results_by_config[cfg_name] = ranked
        print(f"  {cfg_name}: done", flush=True)

    # Build per-query comparison
    query_map = {q.query_id: q for q in queries}
    rows = []
    for q in queries:
        qid = q.query_id
        gold = list(q.gold_pages)
        ranks = {}
        rrs   = {}
        top1s = {}
        for cfg, ranked in results_by_config.items():
            pages = ranked.get(qid, [])
            r = gold_rank(pages, gold)
            ranks[cfg]  = r
            rrs[cfg]    = rr(r)
            top1s[cfg]  = pages[0] if pages else None
        rows.append({
            "query_id": qid,
            "question": q.query_text,
            "gold_pages": gold,
            **{f"rank_{k}": v for k, v in ranks.items()},
            **{f"rr_{k}": round(v, 4) for k, v in rrs.items()},
            **{f"top1_{k}": v for k, v in top1s.items()},
        })

    df = pd.DataFrame(rows)

    # --- Find panel (a): 224/56 wins, at least one other doesn't ---
    # 224/56 at rank 1, at least one of B/C NOT at rank 1
    panel_a = df[
        (df["rank_A_224_56"] == 1) &
        ((df["rank_B_256_64"] != 1) | (df["rank_C_280_90"] != 1))
    ].copy()
    # Sort by biggest advantage: max(rank_B, rank_C) descending
    panel_a["max_other_rank"] = panel_a[["rank_B_256_64", "rank_C_280_90"]].max(axis=1)
    panel_a = panel_a.sort_values("max_other_rank", ascending=False)

    # --- Find panel (b): B or C wins, 224/56 does NOT ---
    # At least one of B/C at rank 1, 224/56 NOT at rank 1
    panel_b = df[
        ((df["rank_B_256_64"] == 1) | (df["rank_C_280_90"] == 1)) &
        (df["rank_A_224_56"] != 1)
    ].copy()
    # Sort by 224/56 gold rank descending (bigger miss = more illustrative)
    panel_b = panel_b.sort_values("rank_A_224_56", ascending=False, na_position="first")

    print("\n=== PANEL (a) candidates — 224/56 wins, others miss ===")
    cols = ["query_id", "gold_pages", "rank_A_224_56", "rank_B_256_64", "rank_C_280_90",
            "top1_A_224_56", "top1_B_256_64", "top1_C_280_90"]
    print(panel_a[cols].head(10).to_string(index=False))

    print("\n=== PANEL (b) candidates — B or C wins, 224/56 misses ===")
    print(panel_b[cols].head(10).to_string(index=False))

    # Save full results
    out_dir = REPO_ROOT / "results/balanced_retrieval_regen_2026-04-20"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "all_query_ranks.csv", index=False)

    # Save top candidates
    candidates = {
        "panel_a_candidates": panel_a[cols].head(5).to_dict(orient="records"),
        "panel_b_candidates": panel_b[cols].head(5).to_dict(orient="records"),
    }
    (out_dir / "candidates.json").write_text(json.dumps(candidates, indent=2))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
