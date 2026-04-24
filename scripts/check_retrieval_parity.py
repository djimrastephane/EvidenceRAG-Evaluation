"""Verify that the SearchService and the standalone retrieval evaluator return identical ranked results.

For a single specified query, runs retrieval through both the production SearchService and
the evaluator-side FAISS+BM25 stack, then compares the top-k chunk ID sequences. A mismatch
indicates that the two code paths have diverged, which would invalidate reproducibility claims.
Writes a parity payload JSON to stdout or an optional output file.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    from scripts.retrieval_eval_bm25 import BM25Index, tokenize
except ModuleNotFoundError:
    from retrieval_eval_bm25 import BM25Index, tokenize

from rag_pdf.retrieval.canonical_hybrid import apply_post_fusion_rerank, fuse_ranked_lists
from rag_pdf.retrieval.rerank import RerankConfig
from rag_pdf.retrieval.hybrid_utils import l2_normalize
from rag_pdf.services.search_service import (
    ENABLE_LEXICAL_RERANK,
    ENABLE_SUBSECTION_BOOST,
    ENTITY_MATCH_BOOST,
    FUSION_STRATEGY,
    MAX_ENTITY_MATCHES,
    NUMERIC_DENSITY_BOOST,
    RRF_BM25_WEIGHT,
    RRF_DENSE_WEIGHT,
    RRF_K,
    SEGMENT_SEARCH_HIT_BOOST,
    SUBSECTION_BOOST,
    TABLE_CHUNK_BOOST,
    SearchService,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check evaluator/service retrieval parity for one query.")
    p.add_argument("--data-dir", required=True, help="Processed doc directory.")
    p.add_argument("--query-id", required=True, help="Query id from eval_set.json.")
    p.add_argument("--model-path", default="models/all-MiniLM-L6-v2")
    p.add_argument("--k", type=int, default=10)
    return p.parse_args()


def _load_eval_items(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict) and isinstance(obj.get("queries"), list):
        return [x for x in obj["queries"] if isinstance(x, dict)]
    return []


def _chunk_text_map(chunks: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    if "chunk_id_global" in chunks.columns:
        for _, row in chunks.iterrows():
            cid = str(row.get("chunk_id_global") or "")
            if cid:
                out[cid] = str(row.get("chunk_text") or "")
    if "chunk_id" in chunks.columns:
        for _, row in chunks.iterrows():
            cid = str(row.get("chunk_id") or "")
            if cid and cid not in out:
                out[cid] = str(row.get("chunk_text") or "")
    return out


def _evaluator_ranked_chunk_ids(data_dir: Path, model_path: Path, query_item: dict[str, Any], k: int) -> list[str]:
    meta = pd.read_parquet(data_dir / "chunk_meta.parquet")
    chunks = pd.read_parquet(data_dir / "chunks.parquet")
    index = faiss.read_index(str(data_dir / "faiss.index"))
    model = SentenceTransformer(str(model_path))
    question = str(query_item.get("question") or "").strip()
    expected_section = str(query_item.get("expected_section") or "").strip()
    expected_subsection = str(query_item.get("expected_subsection") or "").strip()
    text_by_id = _chunk_text_map(chunks)
    corpus_texts = []
    for _, row in meta.iterrows():
        cid = str(row.get("chunk_id_global") or row.get("chunk_id") or "")
        corpus_texts.append(text_by_id.get(cid, ""))
    bm25 = BM25Index([tokenize(t) for t in corpus_texts], k1=1.5, b=0.75)
    emb = model.encode([question], convert_to_numpy=True, normalize_embeddings=False).astype("float32")
    emb = l2_normalize(emb).astype("float32")
    dense_scores, dense_idxs = index.search(emb, len(meta))
    dense_ranked = [int(idx) for idx in dense_idxs[0].tolist()]
    dense_score_map = {int(idx): float(score) for idx, score in zip(dense_idxs[0].tolist(), dense_scores[0].tolist())}
    bm25_scores = bm25.score_query(tokenize(question))
    bm25_ranked = [idx for idx, _ in sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)]
    bm25_score_map = {int(idx): float(score) for idx, score in enumerate(bm25_scores)}
    rerank_cfg = RerankConfig(
        table_chunk_boost=TABLE_CHUNK_BOOST,
        entity_match_boost=ENTITY_MATCH_BOOST,
        numeric_density_boost=NUMERIC_DENSITY_BOOST,
        segment_search_hit_boost=SEGMENT_SEARCH_HIT_BOOST,
        max_entity_matches=MAX_ENTITY_MATCHES,
    )
    fused_ranked, scores_map = fuse_ranked_lists(
        fusion_strategy=FUSION_STRATEGY,
        dense_ranked=dense_ranked,
        bm25_ranked=bm25_ranked,
        dense_score_map=dense_score_map,
        bm25_score_map=bm25_score_map,
        rrf_k=RRF_K,
        dense_weight=RRF_DENSE_WEIGHT,
        bm25_weight=RRF_BM25_WEIGHT,
    )
    fused_ranked, _ = apply_post_fusion_rerank(
        question=question,
        fused_ranked=fused_ranked,
        scores_map=scores_map,
        meta=meta,
        chunk_text_by_id=text_by_id,
        rerank_cfg=rerank_cfg,
        enable_lexical_rerank=bool(ENABLE_LEXICAL_RERANK),
        expected_section=expected_section,
        expected_subsection=expected_subsection,
        enable_subsection_boost=bool(ENABLE_SUBSECTION_BOOST),
        subsection_boost=SUBSECTION_BOOST,
        cross_page_out_of_section_penalty=0.08,
    )
    out: list[str] = []
    for idx in fused_ranked[:k]:
        row = meta.iloc[idx]
        out.append(str(row.get("chunk_id_global") or row.get("chunk_id") or ""))
    return out


def available_query_ids(eval_items: list[dict[str, Any]]) -> list[str]:
    """Return all non-empty query_id strings from the eval set."""
    return [str(item.get("query_id") or "").strip() for item in eval_items if str(item.get("query_id") or "").strip()]


def resolve_query_item(eval_items: list[dict[str, Any]], requested_query_id: str) -> dict[str, Any]:
    """Find and return the eval item matching the requested query_id; raise ValueError with suggestions if not found."""
    requested_query_id = str(requested_query_id).strip()
    query_item = next((item for item in eval_items if str(item.get("query_id") or "").strip() == requested_query_id), None)
    if query_item is not None:
        return query_item
    query_ids = available_query_ids(eval_items)
    suggestions = difflib.get_close_matches(requested_query_id, query_ids, n=5, cutoff=0.0)
    sample_ids = query_ids[:10]
    detail = [
        f"Query id not found in eval_set.json: {requested_query_id}",
        f"Available query count: {len(query_ids)}",
    ]
    if suggestions:
        detail.append("Closest matches: " + ", ".join(suggestions))
    if sample_ids:
        detail.append("Example valid ids: " + ", ".join(sample_ids))
    raise ValueError("\n".join(detail))


def parity_payload(
    *,
    repo_root: Path,
    data_dir: Path,
    model_path: Path,
    query_item: dict[str, Any],
    k: int,
    service: SearchService | None = None,
) -> dict[str, Any]:
    """Run one query through both retrieval paths and return a parity dict with status 'pass' or 'fail'."""
    parity_service = service or SearchService(repo_root=repo_root, model_path=model_path)
    query_id = str(query_item.get("query_id") or "").strip()
    service_out = parity_service.search(
        data_dir=data_dir,
        question=str(query_item.get("question") or "").strip(),
        k=int(k),
        query_id=query_id,
    )
    service_chunk_ids = [str(row.get("chunk_id") or "") for row in list(service_out.get("results") or [])[: int(k)]]
    evaluator_chunk_ids = _evaluator_ranked_chunk_ids(
        data_dir=data_dir,
        model_path=model_path,
        query_item=query_item,
        k=int(k),
    )
    return {
        "status": "pass" if service_chunk_ids == evaluator_chunk_ids else "fail",
        "query_id": query_id,
        "data_dir": str(data_dir),
        "service_chunk_ids": service_chunk_ids,
        "evaluator_chunk_ids": evaluator_chunk_ids,
    }


def main() -> None:
    args = parse_args()
    data_dir = (REPO_ROOT / args.data_dir).resolve()
    model_path = (REPO_ROOT / args.model_path).resolve()
    eval_items = _load_eval_items(data_dir / "eval_set.json")
    query_item = resolve_query_item(eval_items, str(args.query_id))
    service = SearchService(repo_root=REPO_ROOT, model_path=model_path)
    payload = parity_payload(
        repo_root=REPO_ROOT,
        data_dir=data_dir,
        model_path=model_path,
        query_item=query_item,
        k=int(args.k),
        service=service,
    )
    print(json.dumps(payload, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
