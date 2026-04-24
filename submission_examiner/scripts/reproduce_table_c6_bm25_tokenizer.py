"""reproduce_table_c6_bm25_tokenizer.py

Reproduces Table C.6 (BM25 tokenization sensitivity check) against the
post-fix 2026-04-19 pipeline artifacts (chunk=224/56).

Compares default tokenizer (keeps hyphens) vs no-hyphen tokenizer for
both BM25-only and hybrid retrieval.

Thesis values to match:
  BM25-only | default   | Hit@1=0.692 Hit@3=0.832 MRR@10=0.772 FP2=77
  BM25-only | no hyphen | Hit@1=0.708 Hit@3=0.836 MRR@10=0.781 FP2=73
  Hybrid    | default   | Hit@1=0.688 Hit@3=0.840 MRR@10=0.776 FP2=78
  Hybrid    | no hyphen | Hit@1=0.688 Hit@3=0.840 MRR@10=0.776 FP2=78

Usage:
    python scripts/reproduce_table_c6_bm25_tokenizer.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import replace
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH  = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import faiss
import pandas as pd

from thesis_rag.artifacts import load_queries
from thesis_rag.evaluator import hit_at_k
from thesis_rag.ranking import chunk_hits_to_page_hits
from thesis_rag.retrieval_hybrid import hybrid_retrieve_legacy_style
from thesis_rag.retrieval_sparse import build_bm25, sparse_retrieve_legacy_style
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

# Thesis table C.6 reference values
THESIS_VALUES = {
    ("bm25",   "default"):   {"hit@1": 0.692, "hit@3": 0.832, "mrr@10": 0.772, "fp2": 77},
    ("bm25",   "no_hyphen"): {"hit@1": 0.708, "hit@3": 0.836, "mrr@10": 0.781, "fp2": 73},
    ("hybrid", "default"):   {"hit@1": 0.688, "hit@3": 0.840, "mrr@10": 0.776, "fp2": 78},
    ("hybrid", "no_hyphen"): {"hit@1": 0.688, "hit@3": 0.840, "mrr@10": 0.776, "fp2": 78},
}


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------

def _tokenize_default(text: str) -> list[str]:
    """Current pipeline tokenizer — keeps hyphens."""
    return re.findall(r"[a-z0-9][a-z0-9\-]{1,}", str(text or "").lower())


def _tokenize_no_hyphen(text: str) -> list[str]:
    """Split on hyphens before tokenizing."""
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


TOKENIZERS = {
    "default":   _tokenize_default,
    "no_hyphen": _tokenize_no_hyphen,
}


# ---------------------------------------------------------------------------
# Custom BM25 builder with pluggable tokenizer
# ---------------------------------------------------------------------------

def _build_bm25_custom(chunks: list[ChunkRecord], tokenize_fn, k1: float = 1.5, b: float = 0.75):
    corpus = [tokenize_fn(c.text) for c in chunks]
    try:
        from rank_bm25 import BM25Okapi
        return BM25Okapi(corpus, k1=k1, b=b), "rank_bm25"
    except Exception:
        pass
    # Fallback: use the internal _FallbackBM25 via build_bm25 monkeypatch
    import thesis_rag.retrieval_sparse as _rs
    orig = _rs._tokenize_bm25
    _rs._tokenize_bm25 = tokenize_fn
    bm25 = build_bm25(chunks, BM25Config(k1=k1, b=b))
    _rs._tokenize_bm25 = orig
    return bm25, "fallback"


def _get_scores_custom(bm25, query: str, tokenize_fn) -> list[float]:
    tokens = tokenize_fn(query)
    if hasattr(bm25, "get_scores"):
        return bm25.get_scores(tokens)
    if hasattr(bm25, "score_query"):
        return bm25.score_query(tokens)
    raise TypeError(f"Unsupported BM25 type: {type(bm25)}")


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

def _compute_metrics(page_hits, queries) -> dict:
    grouped: dict[str, list[int]] = defaultdict(list)
    for h in page_hits:
        grouped[h.query_id].append(h.page_number)

    h1 = h3 = h10 = mrr = fp2 = 0
    for q in queries:
        pages = grouped.get(q.query_id, [])
        h1  += int(hit_at_k(pages, q.gold_pages, 1))
        h3  += int(hit_at_k(pages, q.gold_pages, 3))
        h10 += int(hit_at_k(pages, q.gold_pages, 10))
        rr   = next((1.0 / (i + 1) for i, p in enumerate(pages[:10])
                     if p in set(q.gold_pages)), 0.0)
        mrr += rr
        # FP2: correct page in top-10 but not at rank 1
        hit1 = hit_at_k(pages, q.gold_pages, 1)
        hit10 = hit_at_k(pages, q.gold_pages, 10)
        if not hit1 and hit10:
            fp2 += 1

    n = max(len(queries), 1)
    return {"hit@1": h1/n, "hit@3": h3/n, "hit@10": h10/n, "mrr@10": mrr/n,
            "fp2": fp2, "n": len(queries)}


# ---------------------------------------------------------------------------
# BM25-only retrieval with custom tokenizer
# ---------------------------------------------------------------------------

def run_bm25_only(doc_cache, tokenize_fn) -> dict:
    """Run BM25-only retrieval across all cached documents using the given tokeniser; return aggregated metrics."""
    agg = {"hit@1": 0.0, "hit@3": 0.0, "hit@10": 0.0, "mrr@10": 0.0, "fp2": 0, "n": 0}
    for dc in doc_cache:
        bm25, _ = _build_bm25_custom(dc["chunks"], tokenize_fn)
        # Custom scoring
        hits = []
        from thesis_rag.schemas import RetrievalHit
        for q in dc["queries"]:
            scores = _get_scores_custom(bm25, q.query_text, tokenize_fn)
            ranking = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:TOP_K]
            for rank, (idx, score) in enumerate(ranking, 1):
                chunk = dc["chunks"][idx]
                hits.append(RetrievalHit(
                    query_id=q.query_id, query_text=q.query_text,
                    rank=rank, score=float(score), retrieval_method="bm25",
                    doc_id=chunk.doc_id, page_number=chunk.page_number,
                    chunk_id=chunk.chunk_id,
                    pages=chunk.pages or [chunk.page_number],
                    text=chunk.text,
                ))
        page_hits = chunk_hits_to_page_hits(hits, "bm25_pages", chunk_limit=TOP_K)
        m = _compute_metrics(page_hits, dc["queries"])
        for key in ("hit@1", "hit@3", "hit@10", "mrr@10"):
            agg[key] += m[key] * m["n"]
        agg["fp2"] += m["fp2"]
        agg["n"]   += m["n"]
    n = agg["n"]
    return {k: agg[k] / n if k not in ("fp2", "n") else agg[k] for k in agg}


# ---------------------------------------------------------------------------
# Hybrid retrieval with custom tokenizer
# ---------------------------------------------------------------------------

def run_hybrid(doc_cache, tokenize_fn) -> dict:
    """Run hybrid dense+BM25 RRF retrieval with the given tokeniser substituted into the sparse path; return aggregated metrics."""
    agg = {"hit@1": 0.0, "hit@3": 0.0, "hit@10": 0.0, "mrr@10": 0.0, "fp2": 0, "n": 0}
    for dc in doc_cache:
        bm25, _ = _build_bm25_custom(dc["chunks"], tokenize_fn)

        import thesis_rag.retrieval_sparse as _rs
        orig_tok = _rs._tokenize_bm25
        _rs._tokenize_bm25 = tokenize_fn

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
            enable_subsection_boost=False,
            enable_lexical_rerank=True,
        )
        _rs._tokenize_bm25 = orig_tok

        page_hits = chunk_hits_to_page_hits(hybrid_hits, "hybrid_pages", chunk_limit=TOP_K)
        m = _compute_metrics(page_hits, dc["queries"])
        for key in ("hit@1", "hit@3", "hit@10", "mrr@10"):
            agg[key] += m[key] * m["n"]
        agg["fp2"] += m["fp2"]
        agg["n"]   += m["n"]
    n = agg["n"]
    return {k: agg[k] / n if k not in ("fp2", "n") else agg[k] for k in agg}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Reproduce Appendix Table C.6 comparing default vs hyphen-stripped BM25 tokenisation on the 5-doc Grampian corpus."""
    from sentence_transformers import SentenceTransformer
    from thesis_rag.config import load_config
    from thesis_rag.utils import resolve_device

    config   = load_config(REPO_ROOT / "configs/thesis_rag.yaml")
    device   = resolve_device(config.runtime.device)
    apply_l2 = config.embedding.apply_l2_normalization

    print("Loading embedding model...", flush=True)
    model = SentenceTransformer(str(REPO_ROOT / "models/all-MiniLM-L6-v2"), device=device)

    print("Pre-computing dense scores...", flush=True)
    doc_cache = []
    for doc_id in DOCS:
        exp_dir = ARTIFACT_ROOT / f"minilmcap_{doc_id}_chunk_224_56" / doc_id
        chunks  = _load_chunks(exp_dir)
        queries = load_queries(EVAL_ROOT / doc_id / "eval_set.json")
        index   = faiss.read_index(str(exp_dir / "faiss.index"))
        q_vecs  = model.encode(
            [q.query_text for q in queries], batch_size=32,
            show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False,
        ).astype("float32")
        if apply_l2:
            q_vecs = l2_normalize(q_vecs)
        raw_scores, raw_indices = search_faiss_stably(index, q_vecs, min(100, len(chunks)))
        doc_cache.append({"doc_id": doc_id, "chunks": chunks, "queries": queries,
                          "raw_scores": raw_scores, "raw_indices": raw_indices})
        print(f"  {doc_id}: ready", flush=True)

    # Run all 4 combinations
    results = []
    for tok_name, tokenize_fn in TOKENIZERS.items():
        for setup_name, run_fn in [("bm25", run_bm25_only), ("hybrid", run_hybrid)]:
            print(f"  Running {setup_name} / {tok_name}...", flush=True)
            t0 = time.perf_counter()
            m  = run_fn(doc_cache, tokenize_fn)
            rt = time.perf_counter() - t0
            results.append({
                "setup": setup_name, "tokenizer": tok_name,
                "hit@1": m["hit@1"], "hit@3": m["hit@3"],
                "mrr@10": m["mrr@10"], "fp2": m["fp2"],
                "runtime_s": round(rt, 2),
            })

    # Print comparison table
    print()
    print("=" * 90)
    print(f"  Table C.6 reproduction — post-fix pipeline  (chunk=224/56, 250 queries)")
    print(f"  {'Setup':<8} {'Tokenizer':<11}  {'Hit@1':>7} {'ΔHit@1':>8}  {'Hit@3':>7}  "
          f"{'MRR@10':>8} {'ΔMRR':>7}  {'FP2':>4} {'ΔFP2':>6}  {'Match?':>7}")
    print("  " + "-" * 84)
    all_match = True
    for r in results:
        key = (r["setup"], r["tokenizer"])
        ref = THESIS_VALUES[key]
        d1   = r["hit@1"]  - ref["hit@1"]
        dmrr = r["mrr@10"] - ref["mrr@10"]
        dfp2 = r["fp2"]    - ref["fp2"]
        tol  = 0.005  # allow ±0.5pp rounding tolerance
        match = abs(d1) <= tol and abs(dmrr) <= tol and abs(dfp2) <= 5
        all_match = all_match and match
        flag = "✓" if match else "DIFF"
        print(f"  {r['setup']:<8} {r['tokenizer']:<11}  "
              f"{r['hit@1']:>7.4f} {d1:>+8.4f}  {r['hit@3']:>7.4f}  "
              f"{r['mrr@10']:>8.4f} {dmrr:>+7.4f}  {r['fp2']:>4} {dfp2:>+6}  {flag:>7}")
    print("  " + "-" * 84)
    print(f"\n  {'OVERALL: REPRODUCED ✓' if all_match else 'OVERALL: DIFFERENCES FOUND — review above'}")
    print("=" * 90)

    # Save
    out_dir = REPO_ROOT / "results" / f"table_c6_reproduction_{date.today().isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps({
        "thesis_values": {str(k): v for k, v in THESIS_VALUES.items()},
        "reproduced": results,
    }, indent=2))
    print(f"\n  Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
