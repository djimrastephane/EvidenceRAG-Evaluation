from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    from scripts.retrieval_eval_bm25 import BM25Index, get_retrieved_pages, set_bm25_tokenizer_variant, tokenize
except ModuleNotFoundError:
    from retrieval_eval_bm25 import BM25Index, get_retrieved_pages, set_bm25_tokenizer_variant, tokenize

from rag_pdf.retrieval.canonical_hybrid import (
    apply_post_fusion_rerank,
    fuse_ranked_lists,
    normalize_cross_encoder_scores,
)
from rag_pdf.retrieval.rerank import RerankConfig
from rag_pdf.retrieval.hybrid_utils import l2_normalize
from runtime_env import collect_runtime_provenance, critical_environment_checks

try:
    from scripts.retrieval_eval_hybrid import (
        ENABLE_LEXICAL_RERANK,
        ENABLE_SUBSECTION_BOOST,
        ENTITY_MATCH_BOOST,
        MAX_ENTITY_MATCHES,
        MAX_K_SEARCH,
        NUMERIC_DENSITY_BOOST,
        SEGMENT_SEARCH_HIT_BOOST,
        SUBSECTION_BOOST,
        TABLE_CHUNK_BOOST,
        chunk_hit_at_k,
        chunk_hit_flags,
        chunk_mrr,
        chunk_precision_at_k,
        compute_leakage,
        get_chunk_ids,
        get_doc_ids,
        get_expected_doc_id,
        mrr_for_pages,
        parse_k_list,
        parse_query_id,
        precision_at_k,
        recall_at_k,
        resolve_torch_device,
        unique_preserve_order,
        utc_now_iso,
        validate_query_id,
        write_json,
        _collect_eval_set_info,
        _collect_pipeline_settings,
        _validate_eval_items,
    )
except ModuleNotFoundError:
    from retrieval_eval_hybrid import (
        ENABLE_LEXICAL_RERANK,
        ENABLE_SUBSECTION_BOOST,
        ENTITY_MATCH_BOOST,
        MAX_ENTITY_MATCHES,
        MAX_K_SEARCH,
        NUMERIC_DENSITY_BOOST,
        SEGMENT_SEARCH_HIT_BOOST,
        SUBSECTION_BOOST,
        TABLE_CHUNK_BOOST,
        chunk_hit_at_k,
        chunk_hit_flags,
        chunk_mrr,
        chunk_precision_at_k,
        compute_leakage,
        get_chunk_ids,
        get_doc_ids,
        get_expected_doc_id,
        mrr_for_pages,
        parse_k_list,
        parse_query_id,
        precision_at_k,
        recall_at_k,
        resolve_torch_device,
        unique_preserve_order,
        utc_now_iso,
        validate_query_id,
        write_json,
        _collect_eval_set_info,
        _collect_pipeline_settings,
        _validate_eval_items,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare explicit manual L2 normalization against SentenceTransformers "
            "normalize_embeddings=True under the same hybrid retrieval pipeline."
        )
    )
    parser.add_argument("--data-dir", required=True, help="Processed doc directory containing chunks/meta/eval_set.")
    parser.add_argument("--model", default="models/all-MiniLM-L6-v2", help="Sentence-transformers model name or local path.")
    parser.add_argument("--device", default="cpu", help="Embedding device: cpu, mps, cuda, or auto.")
    parser.add_argument("--k-list", default="1,3,5,10", help="Comma-separated list of k values.")
    parser.add_argument("--rrf-k", type=int, default=20, help="RRF constant k.")
    parser.add_argument("--dense-weight", type=float, default=0.5, help="Dense rank contribution weight.")
    parser.add_argument("--bm25-weight", type=float, default=2.0, help="BM25 rank contribution weight.")
    parser.add_argument("--bm25-k1", type=float, default=1.5, help="BM25 k1 parameter.")
    parser.add_argument("--bm25-b", type=float, default=0.75, help="BM25 b parameter.")
    parser.add_argument(
        "--bm25-tokenizer",
        choices=("default", "no_hyphen"),
        default="default",
        help="Lexical tokenizer variant.",
    )
    parser.add_argument("--no-lexical-rerank", action="store_true", help="Disable lexical rerank boosts.")
    parser.add_argument("--no-subsection-boost", action="store_true", help="Disable subsection boost.")
    parser.add_argument(
        "--enable-cross-encoder-rerank",
        action="store_true",
        help="Enable local cross-encoder reranking on top fused candidates.",
    )
    parser.add_argument("--cross-encoder-model", default="models/bge-reranker-v2-m3")
    parser.add_argument("--cross-encoder-device", default="cpu")
    parser.add_argument("--cross-encoder-topn", type=int, default=50)
    parser.add_argument("--cross-encoder-weight", type=float, default=0.2)
    parser.add_argument(
        "--out-json",
        default="",
        help="Output JSON path. Defaults to <data-dir>/normalization_ablation_report.json.",
    )
    parser.add_argument(
        "--diff-limit",
        type=int,
        default=25,
        help="Maximum number of differing queries to include in the report payload.",
    )
    return parser.parse_args()


def load_eval_items(path: Path) -> list[dict[str, Any]]:
    eval_obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(eval_obj, list):
        items = eval_obj
    elif isinstance(eval_obj, dict) and isinstance(eval_obj.get("queries"), list):
        items = eval_obj.get("queries", [])
    else:
        items = []
    if not items:
        raise ValueError(f"eval_set.json must be a non-empty list (or {{'queries': [...]}}): {path}")
    _validate_eval_items(items)
    return items


def chunk_text_map(chunks: pd.DataFrame) -> dict[str, str]:
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


def build_corpus_texts(meta: pd.DataFrame, text_by_id: dict[str, str]) -> list[str]:
    out: list[str] = []
    for _, row in meta.iterrows():
        cid = str(row.get("chunk_id_global") or row.get("chunk_id") or "")
        out.append(text_by_id.get(cid, ""))
    return out


def build_embedding_texts(chunks: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for _, row in chunks.iterrows():
        section = str(row.get("section_title") or "").strip()
        subsection = str(row.get("subsection_title") or "").strip()
        is_table = bool(row.get("is_table", False))
        chunk_text = str(row.get("chunk_text") or "")
        parts: list[str] = []
        if section and section.lower() != "unknown":
            parts.append(section)
        if subsection and subsection.lower() != "unknown" and not is_table:
            parts.append(subsection)
        parts.append(chunk_text)
        out.append("\n".join(parts))
    return out


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    *,
    mode: str,
    batch_size: int = 32,
    show_progress_bar: bool = False,
) -> np.ndarray:
    if mode == "manual_l2":
        emb = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype("float32")
        return l2_normalize(emb).astype("float32")
    if mode == "st_builtin":
        emb = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        return emb
    raise ValueError(f"Unsupported normalization mode: {mode}")


def build_index(embeddings: np.ndarray) -> faiss.Index:
    d = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype("float32"))
    return index


def embedding_diff_summary(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    if a.shape != b.shape:
        return {
            "same_shape": False,
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
        }
    diff = np.abs(a - b)
    return {
        "same_shape": True,
        "shape": list(a.shape),
        "max_abs_diff": float(np.max(diff)) if diff.size else 0.0,
        "mean_abs_diff": float(np.mean(diff)) if diff.size else 0.0,
        "allclose_atol_1e-6": bool(np.allclose(a, b, atol=1e-6)),
        "allclose_atol_1e-5": bool(np.allclose(a, b, atol=1e-5)),
    }


def evaluate_mode(
    *,
    mode: str,
    meta: pd.DataFrame,
    chunks: pd.DataFrame,
    eval_items: list[dict[str, Any]],
    model: SentenceTransformer,
    cross_encoder: CrossEncoder | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    embedding_texts = build_embedding_texts(chunks)
    doc_emb = encode_texts(model, embedding_texts, mode=mode, batch_size=32, show_progress_bar=False)
    query_texts = [str(x.get("question", "")).strip() for x in eval_items]
    query_emb = encode_texts(model, query_texts, mode=mode, batch_size=32, show_progress_bar=False)
    index = build_index(doc_emb)

    text_by_id = chunk_text_map(chunks)
    corpus_texts = build_corpus_texts(meta, text_by_id)
    bm25 = BM25Index([tokenize(t) for t in corpus_texts], k1=float(args.bm25_k1), b=float(args.bm25_b))

    rerank_cfg = RerankConfig(
        table_chunk_boost=TABLE_CHUNK_BOOST,
        entity_match_boost=ENTITY_MATCH_BOOST,
        numeric_density_boost=NUMERIC_DENSITY_BOOST,
        segment_search_hit_boost=SEGMENT_SEARCH_HIT_BOOST,
        max_entity_matches=MAX_ENTITY_MATCHES,
    )
    k_list = parse_k_list(args.k_list)
    max_k = max(k_list)
    max_k_search = min(max(MAX_K_SEARCH, max_k), len(meta))
    enable_lexical_rerank = ENABLE_LEXICAL_RERANK and (not args.no_lexical_rerank)
    enable_subsection_boost = ENABLE_SUBSECTION_BOOST and (not args.no_subsection_boost)
    meta_doc_ids = set(str(x) for x in meta["doc_id"].dropna().unique()) if "doc_id" in meta.columns else set()

    summary_rows: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []

    for qi, item in enumerate(eval_items):
        query_id = str(item.get("query_id", "")).strip()
        validate_query_id(query_id)
        qid_parts = parse_query_id(query_id)
        question = query_texts[qi]
        if not question:
            continue

        expected_raw = item.get("expected_pages", [])
        expected_pages = set(int(x) for x in expected_raw) if isinstance(expected_raw, list) else set()
        expected_doc_id = get_expected_doc_id(item)
        expected_section = str(item.get("expected_section", "")).strip()
        expected_subsection = str(item.get("expected_subsection", "")).strip()
        evidence_layout = str(item.get("evidence_layout", "")).strip()
        answer_type = str(item.get("answer_type", "unknown"))

        if expected_doc_id and meta_doc_ids and expected_doc_id not in meta_doc_ids:
            raise ValueError(
                f"Query {query_id} expects doc_id={expected_doc_id}, "
                f"but meta has doc_id values like: {sorted(list(meta_doc_ids))[:5]}"
            )

        dense_scores, dense_idxs = index.search(query_emb[qi : qi + 1], max_k_search)
        dense_ranked = [int(idx) for idx in dense_idxs[0].tolist()]
        dense_score_map = {int(idx): float(score) for idx, score in zip(dense_idxs[0].tolist(), dense_scores[0].tolist())}

        bm25_scores = bm25.score_query(tokenize(question))
        bm25_ranked = [idx for idx, _ in sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:max_k_search]]
        bm25_score_map = {int(idx): float(score) for idx, score in enumerate(bm25_scores)}

        fused_ranked, scores_map = fuse_ranked_lists(
            fusion_strategy="rrf",
            dense_ranked=dense_ranked,
            bm25_ranked=bm25_ranked,
            dense_score_map=dense_score_map,
            bm25_score_map=bm25_score_map,
            rrf_k=int(args.rrf_k),
            dense_weight=float(args.dense_weight),
            bm25_weight=float(args.bm25_weight),
        )

        if cross_encoder is not None and fused_ranked:
            ce_topn = max(1, min(int(args.cross_encoder_topn), len(fused_ranked)))
            cand = fused_ranked[:ce_topn]
            pairs: list[tuple[str, str]] = []
            for idx in cand:
                row = meta.iloc[idx]
                cid = str(row.get("chunk_id_global") or row.get("chunk_id") or "")
                pairs.append((question, text_by_id.get(cid, "")))
            ce_scores_raw = np.asarray(cross_encoder.predict(pairs), dtype=np.float32)
            ce_scores = normalize_cross_encoder_scores(ce_scores_raw)
            for idx, ce_s in zip(cand, ce_scores.tolist()):
                scores_map[idx] = float(scores_map.get(idx, 0.0)) + float(args.cross_encoder_weight) * float(ce_s)

        fused_ranked, scores_map = apply_post_fusion_rerank(
            question=question,
            fused_ranked=fused_ranked,
            scores_map=scores_map,
            meta=meta,
            chunk_text_by_id=text_by_id,
            rerank_cfg=rerank_cfg,
            enable_lexical_rerank=enable_lexical_rerank,
            expected_section=expected_section,
            expected_subsection=expected_subsection,
            enable_subsection_boost=enable_subsection_boost,
            subsection_boost=SUBSECTION_BOOST,
            cross_page_out_of_section_penalty=0.08,
        )

        ranked_chunk_ids: list[str] = []
        ranked_scores: list[float] = []
        for idx in fused_ranked[:max_k]:
            row = meta.iloc[idx]
            ranked_chunk_ids.append(str(row.get("chunk_id_global") or row.get("chunk_id") or ""))
            ranked_scores.append(float(scores_map.get(idx, 0.0)))

        per_k: dict[str, Any] = {}
        for k in k_list:
            top_idxs = fused_ranked[:k]
            retrieved_chunks = meta.iloc[top_idxs].copy()
            top_scores = [float(scores_map.get(i, 0.0)) for i in top_idxs]
            retrieved_chunks["score"] = top_scores

            retrieved_chunk_ids = get_chunk_ids(retrieved_chunks)
            retrieved_doc_ids = get_doc_ids(retrieved_chunks)
            leakage = compute_leakage(expected_doc_id, retrieved_doc_ids)

            ranked_pages: list[int] = []
            for _, row in retrieved_chunks.iterrows():
                ranked_pages.extend(get_retrieved_pages(row))
            ranked_pages_unique = unique_preserve_order(ranked_pages)

            page_recall = recall_at_k(expected_pages, ranked_pages_unique)
            page_precision = precision_at_k(expected_pages, ranked_pages_unique)
            page_mrr = mrr_for_pages(expected_pages, ranked_pages_unique)

            flags = chunk_hit_flags(expected_pages, retrieved_chunks)
            c_hit = chunk_hit_at_k(flags)
            c_prec = chunk_precision_at_k(flags)
            c_mrr = chunk_mrr(flags)
            failure_stage = "hit" if page_recall >= 1.0 else "missed_top_ranked"

            per_k[str(k)] = {
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "retrieved_doc_ids_top_k": retrieved_doc_ids,
                "retrieved_pages_ranked": ranked_pages_unique,
                "retrieved_scores": top_scores,
                "page_recall_at_k": float(page_recall),
                "page_precision_at_k": float(page_precision),
                "page_mrr_at_k": float(page_mrr),
                "chunk_hit_at_k": float(c_hit),
                "chunk_precision_at_k": float(c_prec),
                "chunk_mrr_at_k": float(c_mrr),
                "failure_stage": failure_stage,
                **leakage,
            }

            summary_rows.append(
                {
                    "query_id": query_id,
                    "topic": qid_parts["topic"],
                    "year": qid_parts["year"],
                    "sequence": qid_parts["sequence"],
                    "k": int(k),
                    "answer_type": answer_type,
                    "doc_id": expected_doc_id,
                    "expected_section": expected_section,
                    "expected_subsection": expected_subsection,
                    "expected_pages": sorted(list(expected_pages)),
                    "evidence_layout": evidence_layout,
                    "failure_stage": failure_stage,
                    "leakage_count_top_k": leakage["leakage_count_top_k"],
                    "leakage_rate_top_k": leakage["leakage_rate_top_k"],
                    "page_recall_at_k": page_recall,
                    "page_precision_at_k": page_precision,
                    "page_mrr_at_k": page_mrr,
                    "chunk_hit_at_k": c_hit,
                    "chunk_precision_at_k": c_prec,
                    "chunk_mrr_at_k": c_mrr,
                }
            )

        k1_data = per_k.get("1", {})
        page_hit = 1 if k1_data.get("page_recall_at_k", 0.0) > 0 else 0
        per_query.append(
            {
                "query_id": query_id,
                "question": question,
                "doc_id": expected_doc_id,
                "expected_pages": sorted(list(expected_pages)),
                "page_hit_at_1": page_hit,
                "failure_type": "HIT" if page_hit else "FP2_MISSED_TOP_RANK",
                "top_chunk_ids_at_max_k": ranked_chunk_ids,
                "top_scores_at_max_k": ranked_scores,
                "per_k": per_k,
            }
        )

    df_sum = pd.DataFrame(summary_rows)
    metrics_by_k: dict[str, Any] = {}
    for k in k_list:
        dfk = df_sum[df_sum["k"] == k]
        metrics_by_k[str(k)] = {
            "num_queries": int(len(dfk)),
            "page_hit_rate_at_k": float((dfk["page_recall_at_k"] > 0).mean()) if len(dfk) else 0.0,
            "mean_page_recall_at_k": float(dfk["page_recall_at_k"].mean()) if len(dfk) else 0.0,
            "mean_page_precision_at_k": float(dfk["page_precision_at_k"].mean()) if len(dfk) else 0.0,
            "mean_page_mrr_at_k": float(dfk["page_mrr_at_k"].mean()) if len(dfk) else 0.0,
            "chunk_hit_rate_at_k": float((dfk["chunk_hit_at_k"] > 0).mean()) if len(dfk) else 0.0,
            "mean_chunk_precision_at_k": float(dfk["chunk_precision_at_k"].mean()) if len(dfk) else 0.0,
            "mean_chunk_mrr_at_k": float(dfk["chunk_mrr_at_k"].mean()) if len(dfk) else 0.0,
        }

    return {
        "mode": mode,
        "doc_embeddings": doc_emb,
        "query_embeddings": query_emb,
        "per_query": per_query,
        "metrics_by_k": metrics_by_k,
    }


def compare_modes(
    *,
    baseline: dict[str, Any],
    variant: dict[str, Any],
    k_list: list[int],
    diff_limit: int,
) -> dict[str, Any]:
    base_queries = {str(row["query_id"]): row for row in baseline["per_query"]}
    variant_queries = {str(row["query_id"]): row for row in variant["per_query"]}
    max_k = max(k_list)
    differing_queries: list[dict[str, Any]] = []
    same_top1 = 0
    same_topk_sequence = 0
    same_hit1 = 0

    for query_id in sorted(base_queries.keys()):
        left = base_queries[query_id]
        right = variant_queries.get(query_id, {})
        left_top = list(left.get("top_chunk_ids_at_max_k") or [])[:max_k]
        right_top = list(right.get("top_chunk_ids_at_max_k") or [])[:max_k]
        if left_top[:1] == right_top[:1]:
            same_top1 += 1
        if left_top == right_top:
            same_topk_sequence += 1
        if int(left.get("page_hit_at_1", 0)) == int(right.get("page_hit_at_1", 0)):
            same_hit1 += 1
        if left_top != right_top or int(left.get("page_hit_at_1", 0)) != int(right.get("page_hit_at_1", 0)):
            if len(differing_queries) < diff_limit:
                differing_queries.append(
                    {
                        "query_id": query_id,
                        "baseline_top_chunk_ids": left_top,
                        "variant_top_chunk_ids": right_top,
                        "baseline_hit_at_1": int(left.get("page_hit_at_1", 0)),
                        "variant_hit_at_1": int(right.get("page_hit_at_1", 0)),
                    }
                )

    total = max(1, len(base_queries))
    metric_deltas: dict[str, Any] = {}
    for k in k_list:
        kb = baseline["metrics_by_k"].get(str(k), {})
        kv = variant["metrics_by_k"].get(str(k), {})
        metric_deltas[str(k)] = {
            "page_hit_rate_delta": float(kv.get("page_hit_rate_at_k", 0.0) - kb.get("page_hit_rate_at_k", 0.0)),
            "mean_page_mrr_delta": float(kv.get("mean_page_mrr_at_k", 0.0) - kb.get("mean_page_mrr_at_k", 0.0)),
            "chunk_hit_rate_delta": float(kv.get("chunk_hit_rate_at_k", 0.0) - kb.get("chunk_hit_rate_at_k", 0.0)),
        }

    return {
        "num_queries": int(len(base_queries)),
        "top1_identical_rate": float(same_top1 / total),
        "topk_sequence_identical_rate": float(same_topk_sequence / total),
        "hit_at_1_identical_rate": float(same_hit1 / total),
        "metric_deltas_by_k": metric_deltas,
        "differing_query_count": int(
            len([qid for qid in base_queries.keys() if (base_queries[qid].get("top_chunk_ids_at_max_k") or [])[:max_k] != (variant_queries.get(qid, {}).get("top_chunk_ids_at_max_k") or [])[:max_k]])
        ),
        "differing_queries_sample": differing_queries,
    }


def main() -> None:
    args = parse_args()
    set_bm25_tokenizer_variant(str(args.bm25_tokenizer))
    data_dir = Path(args.data_dir).resolve()
    out_json = Path(args.out_json).resolve() if args.out_json else data_dir / "normalization_ablation_report.json"
    chunks_path = data_dir / "chunks.parquet"
    meta_path = data_dir / "chunk_meta.parquet"
    eval_path = data_dir / "eval_set.json"

    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing file: {chunks_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing file: {meta_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing file: {eval_path}")

    chunks = pd.read_parquet(chunks_path)
    meta = pd.read_parquet(meta_path)
    eval_items = load_eval_items(eval_path)
    eval_obj = json.loads(eval_path.read_text(encoding="utf-8"))
    model = SentenceTransformer(str(args.model), device=resolve_torch_device(args.device))
    cross_encoder = (
        CrossEncoder(str(args.cross_encoder_model), device=resolve_torch_device(args.cross_encoder_device))
        if args.enable_cross_encoder_rerank
        else None
    )

    baseline = evaluate_mode(
        mode="manual_l2",
        meta=meta,
        chunks=chunks,
        eval_items=eval_items,
        model=model,
        cross_encoder=cross_encoder,
        args=args,
    )
    variant = evaluate_mode(
        mode="st_builtin",
        meta=meta,
        chunks=chunks,
        eval_items=eval_items,
        model=model,
        cross_encoder=cross_encoder,
        args=args,
    )

    k_list = parse_k_list(args.k_list)
    payload = {
        "run_info": {
            "run_utc": utc_now_iso(),
            "data_dir": str(data_dir),
            "out_json": str(out_json),
            "embedding_model": str(args.model),
            "runtime": collect_runtime_provenance(),
            "critical_environment_checks": critical_environment_checks(),
            "comparison": {
                "baseline_mode": "encode(normalize_embeddings=False) + manual_l2_normalize",
                "variant_mode": "encode(normalize_embeddings=True) with no extra manual normalization",
            },
            "hybrid_settings": {
                "rrf_k": int(args.rrf_k),
                "dense_weight": float(args.dense_weight),
                "bm25_weight": float(args.bm25_weight),
                "bm25_k1": float(args.bm25_k1),
                "bm25_b": float(args.bm25_b),
                "bm25_tokenizer": str(args.bm25_tokenizer),
                "enable_lexical_rerank": bool(ENABLE_LEXICAL_RERANK and (not args.no_lexical_rerank)),
                "enable_subsection_boost": bool(ENABLE_SUBSECTION_BOOST and (not args.no_subsection_boost)),
                "enable_cross_encoder_rerank": bool(args.enable_cross_encoder_rerank),
                "cross_encoder_model": str(args.cross_encoder_model) if args.enable_cross_encoder_rerank else None,
            },
            "pipeline_settings": _collect_pipeline_settings(data_dir),
            "eval_set": _collect_eval_set_info(eval_path, eval_obj, len(eval_items)),
        },
        "embedding_comparison": {
            "document_embeddings": embedding_diff_summary(baseline["doc_embeddings"], variant["doc_embeddings"]),
            "query_embeddings": embedding_diff_summary(baseline["query_embeddings"], variant["query_embeddings"]),
        },
        "baseline_metrics_by_k": baseline["metrics_by_k"],
        "variant_metrics_by_k": variant["metrics_by_k"],
        "retrieval_comparison": compare_modes(
            baseline=baseline,
            variant=variant,
            k_list=k_list,
            diff_limit=max(1, int(args.diff_limit)),
        ),
    }
    write_json(out_json, payload)
    print(json.dumps(payload["retrieval_comparison"], indent=2))


if __name__ == "__main__":
    main()
