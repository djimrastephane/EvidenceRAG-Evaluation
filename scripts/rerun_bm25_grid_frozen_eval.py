#!/usr/bin/env python3
"""Recompute BM25 parameter grid using frozen 224/56 chunks + current eval_set.json.

Runs BM25-only retrieval for 30 k1×b configurations on the frozen chunk text
from chunk_size_ablation_boost_off_2026-04-20, evaluated against the current
data_processed/eval_set.json.

Output: results/rerun_bm25_grid_2026-04-24/results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thesis_rag.artifacts import load_queries
from thesis_rag.retrieval_sparse import build_bm25, sparse_retrieve_legacy_style
from thesis_rag.ranking import chunk_hits_to_page_hits
from thesis_rag.evaluator import evaluate_page_hits, aggregate_metrics
from thesis_rag.schemas import BM25Config, ChunkRecord

DOCS = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]
OFF_ROOT   = ROOT / "results" / "thesis_ablations" / "chunk_size_ablation_boost_off_2026-04-20" / "pipeline_outputs"
EVAL_ROOT  = ROOT / "data_processed"
OUTPUT_DIR = ROOT / "results" / "rerun_bm25_grid_2026-04-24"

K1_VALUES = [0.5, 1.0, 1.2, 1.5, 2.0, 3.0]
B_VALUES  = [0.0, 0.25, 0.5, 0.75, 1.0]
TOP_K     = 20


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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-load all chunks and queries once
    all_data = []
    for doc in DOCS:
        art_dir = OFF_ROOT / f"minilmcap_{doc}_chunk_224_56" / doc
        chunks  = load_chunks(art_dir / "chunk_metadata.parquet")
        queries = load_queries(EVAL_ROOT / doc / "eval_set.json")
        all_data.append((chunks, queries))
    total_q = sum(len(q) for _, q in all_data)
    print(f"Loaded {total_q} queries, running 30 BM25 configurations...\n")

    configs = [(k1, b) for k1 in K1_VALUES for b in B_VALUES]
    cell_results = []

    for k1, b in configs:
        cfg = BM25Config(k1=k1, b=b)
        all_queries, all_hits = [], []
        for chunks, queries in all_data:
            all_queries.extend(queries)
            bm25 = build_bm25(chunks, cfg)
            chunk_hits = sparse_retrieve_legacy_style(bm25, chunks, queries, top_k=TOP_K)
            page_hits  = chunk_hits_to_page_hits(chunk_hits, "bm25_pages")
            all_hits.extend(page_hits)
        eval_res = evaluate_page_hits(all_queries, all_hits)
        m = aggregate_metrics(eval_res, ks=[1, 5, 10])
        cell_results.append({
            "k1": k1, "b": b,
            "hit@1":  round(m["hit@1"], 4),
            "hit@5":  round(m["hit@5"], 4),
            "hit@10": round(m["hit@10"], 4),
            "mrr@10": round(m["mrr"], 4),
        })
        print(f"k1={k1:<4} b={b:<5} H@1={m['hit@1']:.4f}  H@5={m['hit@5']:.4f}  MRR@10={m['mrr']:.4f}")

    # Sort by MRR@10 desc, print ranked
    cell_results.sort(key=lambda r: r["mrr@10"], reverse=True)
    print("\n=== RANKED BM25 GRID (by MRR@10) ===")
    print(f"{'Rank':>4}  {'k1':>4}  {'b':>5}  {'H@1':>8}  {'H@5':>8}  {'MRR@10':>8}")
    print("-" * 50)
    for rank, r in enumerate(cell_results, 1):
        marker = " ← promoted" if r["k1"] == 1.5 and r["b"] == 0.75 else ""
        print(f"{rank:>4}  {r['k1']:>4}  {r['b']:>5}  {r['hit@1']:>8.4f}  {r['hit@5']:>8.4f}  {r['mrr@10']:>8.4f}{marker}")

    out_path = OUTPUT_DIR / "results.json"
    out_path.write_text(json.dumps({"configs": cell_results}, indent=2), encoding="utf-8")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
