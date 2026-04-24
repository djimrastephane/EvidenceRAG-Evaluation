#!/usr/bin/env python3
"""Ablation: CE reranking + subsection boost on frozen thesis artifacts, per-difficulty breakdown.

Runs 4 conditions (baseline, boost, ce, ce+boost) on the canonical chunk_224_56 artifacts
from results/thesis_ablations/chunk_size_ablation_boost_off_2026-04-20/ and reports
hit@1 and MRR@10 broken down by query difficulty tier (LEX / MOD / STR / ALL),
plus FP2/FP3 error counts per condition.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thesis_rag.artifacts import load_chunks, load_queries
from thesis_rag.embedding import embed_queries
from thesis_rag.evaluator import evaluate_page_hits
from thesis_rag.retrieval_dense import search_faiss_stably
from thesis_rag.retrieval_hybrid import hybrid_retrieve_legacy_style
from thesis_rag.retrieval_sparse import build_bm25
from thesis_rag.schemas import (
    BM25Config,
    EmbeddingConfig,
    QueryRecord,
    RetrievalConfig,
    RetrievalHit,
)

ARTIFACT_ROOT = (
    ROOT
    / "results"
    / "thesis_ablations"
    / "chunk_size_ablation_boost_off_2026-04-20"
    / "pipeline_outputs"
)
EVAL_ROOT = ROOT / "data_processed"
MODEL_DIR = ROOT / "models"
CE_MODEL_PATH = MODEL_DIR / "cross-encoder-ms-marco-MiniLM-L-6-v2"
OUTPUT_DIR = ROOT / "results" / "ce_boost_difficulty_ablation_2026-04-23"

DOC_IDS = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]

EMBED_CFG = EmbeddingConfig(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    apply_l2_normalization=True,
    batch_size=32,
    expected_dimension=384,
)
RETRIEVAL_CFG = RetrievalConfig(
    dense_top_k=20,
    sparse_top_k=20,
    hybrid_top_k=20,
    rrf_k=20,
    dense_weight=0.5,
    sparse_weight=2.0,
)
BM25_CFG = BM25Config(k1=1.5, b=0.75)

CE_TOPN = 5
CE_WEIGHT = 0.2
EVAL_AT_K = 10


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def group_hits(hits: list[RetrievalHit]) -> dict[str, list[RetrievalHit]]:
    out: dict[str, list[RetrievalHit]] = defaultdict(list)
    for h in hits:
        out[h.query_id].append(h)
    for qid in out:
        out[qid].sort(key=lambda h: h.rank)
    return dict(out)


def apply_ce_rerank(
    hits: list[RetrievalHit],
    ce_model,
    topn: int,
    weight: float,
) -> list[RetrievalHit]:
    by_q = group_hits(hits)
    out: list[RetrievalHit] = []
    for qid, q_hits in by_q.items():
        top = q_hits[:topn]
        rest = q_hits[topn:]
        query_text = q_hits[0].query_text
        pairs = [(query_text, h.text or "") for h in top]
        ce_raw = ce_model.predict(pairs)
        ce_norm = _sigmoid(np.array(ce_raw, dtype=np.float32))
        rrf_scores = np.array([h.score for h in top], dtype=np.float32)
        rrf_norm = rrf_scores / (rrf_scores.max() + 1e-10)
        blended = (1.0 - weight) * rrf_norm + weight * ce_norm
        order = np.argsort(-blended)
        for new_rank, orig_idx in enumerate(order, start=1):
            h = top[orig_idx]
            out.append(RetrievalHit(
                query_id=h.query_id,
                query_text=h.query_text,
                rank=new_rank,
                score=float(blended[orig_idx]),
                retrieval_method="hybrid_ce",
                doc_id=h.doc_id,
                page_number=h.page_number,
                chunk_id=h.chunk_id,
                pages=h.pages,
                text=h.text,
            ))
        for new_rank, h in enumerate(rest, start=topn + 1):
            out.append(RetrievalHit(
                query_id=h.query_id,
                query_text=h.query_text,
                rank=new_rank,
                score=h.score,
                retrieval_method=h.retrieval_method,
                doc_id=h.doc_id,
                page_number=h.page_number,
                chunk_id=h.chunk_id,
                pages=h.pages,
                text=h.text,
            ))
    return out


def classify_fp(eval_result, all_predicted: list[int], at_k: int = 10) -> str:
    gold = set(eval_result.gold_pages)
    if eval_result.hit_at_1:
        return "TP"
    if any(p in gold for p in all_predicted[:at_k]):
        return "FP2"
    return "FP3"


def build_metrics(
    queries: list[QueryRecord],
    hits: list[RetrievalHit],
    at_k: int = 10,
) -> dict[str, dict]:
    eval_results = evaluate_page_hits(queries, hits)
    by_q_hits = group_hits(hits)
    q_by_id = {q.query_id: q for q in queries}

    per_query = []
    for er in eval_results:
        q = q_by_id[er.query_id]
        ordered_hits = by_q_hits.get(er.query_id, [])
        predicted = [h.page_number for h in ordered_hits]
        fp_class = classify_fp(er, predicted, at_k=at_k)
        per_query.append({
            "difficulty": q.difficulty or "UNKNOWN",
            "hit_at_1": er.hit_at_1,
            "rr": er.reciprocal_rank,
            "fp": fp_class,
        })

    groups: dict[str, list] = defaultdict(list)
    for rec in per_query:
        groups[rec["difficulty"]].append(rec)
        groups["ALL"].append(rec)

    result = {}
    for diff in ["LEX", "MOD", "STR", "ALL"]:
        recs = groups.get(diff, [])
        n = len(recs)
        if n == 0:
            result[diff] = {"n": 0, "hit_at_1": 0.0, "mrr_at_10": 0.0, "TP": 0, "FP2": 0, "FP3": 0}
            continue
        result[diff] = {
            "n": n,
            "hit_at_1": round(sum(r["hit_at_1"] for r in recs) / n, 4),
            "mrr_at_10": round(sum(r["rr"] for r in recs) / n, 4),
            "TP": sum(1 for r in recs if r["fp"] == "TP"),
            "FP2": sum(1 for r in recs if r["fp"] == "FP2"),
            "FP3": sum(1 for r in recs if r["fp"] == "FP3"),
        }
    return result


def print_results(results: dict[str, dict[str, dict]]) -> None:
    conditions = list(results.keys())
    diffs = ["LEX", "MOD", "STR", "ALL"]

    print("\n=== Page Hit@1 by condition and difficulty ===")
    header = f"{'Condition':<22}" + "".join(f"  {d:>7}" for d in diffs)
    print(header)
    for cond in conditions:
        row = f"{cond:<22}"
        for d in diffs:
            v = results[cond].get(d, {}).get("hit_at_1", float("nan"))
            row += f"  {v:>7.3f}"
        print(row)

    print("\n=== MRR@10 by condition and difficulty ===")
    print(header)
    for cond in conditions:
        row = f"{cond:<22}"
        for d in diffs:
            v = results[cond].get(d, {}).get("mrr_at_10", float("nan"))
            row += f"  {v:>7.3f}"
        print(row)

    print("\n=== FP2 / FP3 counts (ALL queries) ===")
    all_n = results[conditions[0]].get("ALL", {}).get("n", 0)
    print(f"n={all_n} total queries")
    print(f"{'Condition':<22}  {'TP':>5}  {'FP2':>5}  {'FP3':>5}")
    for cond in conditions:
        a = results[cond].get("ALL", {})
        print(f"{cond:<22}  {a.get('TP', 0):>5}  {a.get('FP2', 0):>5}  {a.get('FP3', 0):>5}")

    print("\n=== Hit@1 delta vs baseline (ALL) ===")
    base_h1 = results.get("baseline", {}).get("ALL", {}).get("hit_at_1", 0.0)
    for cond in conditions:
        h1 = results[cond].get("ALL", {}).get("hit_at_1", 0.0)
        delta = h1 - base_h1
        print(f"  {cond:<22}  {h1:.3f}  ({delta:+.3f})")


def main() -> None:
    import faiss
    from sentence_transformers import CrossEncoder

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading CE model from {CE_MODEL_PATH}...")
    ce_model = CrossEncoder(str(CE_MODEL_PATH), device="cpu")

    all_queries: list[QueryRecord] = []
    cond_hits: dict[str, list[RetrievalHit]] = {
        "baseline": [],
        "boost": [],
        "ce": [],
        "ce+boost": [],
    }

    for doc_id in DOC_IDS:
        art_dir = ARTIFACT_ROOT / f"minilmcap_{doc_id}_chunk_224_56" / doc_id
        eval_path = EVAL_ROOT / doc_id / "eval_set.json"
        print(f"\n--- {doc_id} ---")

        chunks = load_chunks(art_dir / "chunk_metadata.parquet")
        embeddings = np.load(art_dir / "embeddings.npy").astype(np.float32)
        faiss_index = faiss.read_index(str(art_dir / "faiss.index"))

        queries = load_queries(eval_path)
        all_queries.extend(queries)
        print(f"  chunks={len(chunks)}  queries={len(queries)}")

        q_vecs = embed_queries(
            [q.query_text for q in queries],
            EMBED_CFG,
            device="cpu",
            cache_dir=str(MODEL_DIR),
        )
        max_k = min(RETRIEVAL_CFG.hybrid_top_k, len(chunks))
        dense_scores, dense_indices = search_faiss_stably(faiss_index, q_vecs, max_k)
        bm25 = build_bm25(chunks, BM25_CFG)

        for enable_boost, base_label, ce_label in [
            (False, "baseline", "ce"),
            (True, "boost", "ce+boost"),
        ]:
            _, _, hyb_hits = hybrid_retrieve_legacy_style(
                chunks=chunks,
                queries=queries,
                dense_scores=dense_scores,
                dense_indices=dense_indices,
                bm25=bm25,
                max_k_search=max_k,
                dense_weight=RETRIEVAL_CFG.dense_weight,
                bm25_weight=RETRIEVAL_CFG.sparse_weight,
                rrf_k=RETRIEVAL_CFG.rrf_k,
                enable_lexical_rerank=True,
                enable_subsection_boost=enable_boost,
                subsection_boost=0.05,
                cross_page_out_of_section_penalty=0.08,
            )
            cond_hits[base_label].extend(hyb_hits)

            ce_hits = apply_ce_rerank(hyb_hits, ce_model, topn=CE_TOPN, weight=CE_WEIGHT)
            cond_hits[ce_label].extend(ce_hits)

        print(f"  done")

    print(f"\nEvaluating across {len(all_queries)} queries...")
    results: dict[str, dict[str, dict]] = {}
    for cond, hits in cond_hits.items():
        results[cond] = build_metrics(all_queries, hits, at_k=EVAL_AT_K)

    print_results(results)

    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
