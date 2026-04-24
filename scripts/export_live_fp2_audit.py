from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rag_pdf.services.search_helpers import read_eval_items
from rag_pdf.services.search_service import (
    BM25Index,
    CROSS_ENCODER_MODEL_NAME,
    CROSS_ENCODER_TOPN,
    CROSS_ENCODER_WEIGHT,
    ENABLE_CROSS_ENCODER_RERANK,
    FUSION_STRATEGY,
    RRF_BM25_WEIGHT,
    RRF_DENSE_WEIGHT,
    RRF_K,
    SearchService,
    l2_normalize,
    rrf_fuse,
    score_fuse,
    tokenize,
    to_pages_list,
)
from corpus_guard import list_eval_ready_doc_dirs, print_skipped_eval_ready_docs


CSV_HEADERS = [
    "query_id",
    "document",
    "difficulty",
    "question",
    "expected_answer",
    "expected_pages",
    "expected_section",
    "expected_subsection",
    "evidence_layout",
    "answer_type",
    "failure_type",
    "gold_chunk_match_basis",
    "gold_chunk_ids",
    "selected_gold_chunk_id",
    "selected_gold_chunk_pages",
    "gold_hybrid_rank",
    "gold_dense_rank",
    "gold_bm25_rank",
    "top1_chunk_id",
    "top1_pages",
    "top1_dense_rank",
    "top1_bm25_rank",
    "top3_chunk_ids",
    "top3_pages",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export current live FP2_MISSED_TOP_RANK cases across the 5-doc corpus using SearchService."
    )
    parser.add_argument(
        "--data-root",
        default="data_processed",
        help="Root containing processed document folders.",
    )
    parser.add_argument(
        "--doc-pattern",
        default="Grampian-20*-20*",
        help="Glob used to select documents under data-root.",
    )
    parser.add_argument(
        "--allow-incomplete-corpora",
        action="store_true",
        help="Include matching doc folders even if they are missing canonical evaluation artifacts.",
    )
    parser.add_argument(
        "--model-path",
        default="models/all-MiniLM-L6-v2",
        help="SentenceTransformer model path used by the live pipeline.",
    )
    parser.add_argument(
        "--out-csv",
        default="results/live_fp2_audit/current_pipeline_fp2_5docs.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-meta",
        default="results/live_fp2_audit/current_pipeline_fp2_5docs.meta.json",
        help="Output JSON metadata path.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Search k to request from SearchService while evaluating top-1 failure.",
    )
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def pages_to_text(value: Any) -> str:
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=True)
    if value is None:
        return ""
    return str(value)


def _row_pages(row: Any) -> list[int]:
    pages = to_pages_list(row.get("pages"))
    if pages:
        return pages
    out: list[int] = []
    for key in ("page_start", "page_end"):
        raw = row.get(key)
        if raw is None:
            continue
        try:
            page = int(raw)
        except Exception:
            continue
        if page not in out:
            out.append(page)
    return out


def _build_live_rankings(service: SearchService, data_dir: Path, question: str) -> tuple[Any, dict[int, int], dict[int, int], list[int]]:
    loaded_doc = service._load_doc(data_dir)
    scope_meta = loaded_doc.meta.reset_index(drop=True)
    scope_index = loaded_doc.index
    chunk_text_by_id = loaded_doc.chunk_text_by_id

    emb = service.model.encode([question], convert_to_numpy=True, normalize_embeddings=False).astype("float32")
    emb = l2_normalize(emb).astype("float32")

    dense_scores, dense_idxs = scope_index.search(emb, len(scope_meta))
    dense_ranked = [int(idx) for idx in dense_idxs[0].tolist()]
    dense_score_map = {
        int(idx): float(score)
        for idx, score in zip(dense_idxs[0].tolist(), dense_scores[0].tolist())
    }
    dense_rank_map = {idx: rank for rank, idx in enumerate(dense_ranked, start=1)}

    bm25_corpus: list[str] = []
    for idx in range(len(scope_meta)):
        row = scope_meta.iloc[idx]
        cid = str(row.get("chunk_id_global") or row.get("chunk_id") or "")
        bm25_corpus.append(chunk_text_by_id.get(cid, ""))
    bm25 = BM25Index([tokenize(t) for t in bm25_corpus], k1=1.5, b=0.75)
    bm25_scores = bm25.score_query(tokenize(question))
    bm25_ranked_pairs = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
    bm25_ranked = [int(idx) for idx, _ in bm25_ranked_pairs]
    bm25_score_map = {int(idx): float(score) for idx, score in bm25_ranked_pairs}
    bm25_rank_map = {idx: rank for rank, idx in enumerate(bm25_ranked, start=1)}

    fusion_strategy = FUSION_STRATEGY if FUSION_STRATEGY in {"rrf", "score_fusion"} else "rrf"
    if fusion_strategy == "score_fusion":
        fused_ranked, fused_scores = score_fuse(
            dense_score_map=dense_score_map,
            bm25_score_map=bm25_score_map,
            dense_weight=RRF_DENSE_WEIGHT,
            bm25_weight=RRF_BM25_WEIGHT,
        )
    else:
        fused_ranked, fused_scores = rrf_fuse(
            dense_ranked=dense_ranked,
            bm25_ranked=bm25_ranked,
            rrf_k=RRF_K,
            dense_weight=RRF_DENSE_WEIGHT,
            bm25_weight=RRF_BM25_WEIGHT,
        )
    scores_map: dict[int, float] = dict(fused_scores)
    if service.cross_encoder is not None and fused_ranked:
        ce_topn = min(len(fused_ranked), service.cross_encoder_topn)
        cand = fused_ranked[:ce_topn]
        pairs: list[tuple[str, str]] = []
        for idx in cand:
            row = scope_meta.iloc[idx]
            cid = str(row.get("chunk_id_global") or row.get("chunk_id") or "")
            pairs.append((question, chunk_text_by_id.get(cid, "")))
        ce_scores_raw = service.cross_encoder.predict(pairs)
        lo = float(min(ce_scores_raw)) if len(ce_scores_raw) else 0.0
        hi = float(max(ce_scores_raw)) if len(ce_scores_raw) else 0.0
        if hi > lo:
            ce_scores = [(float(v) - lo) / (hi - lo) for v in ce_scores_raw]
        else:
            ce_scores = [0.0 for _ in ce_scores_raw]
        for idx, ce_s in zip(cand, ce_scores):
            scores_map[idx] = float(scores_map.get(idx, 0.0)) + service.cross_encoder_weight * float(ce_s)
        fused_ranked = sorted(fused_ranked, key=lambda i: scores_map.get(i, 0.0), reverse=True)

    hybrid_rank_map = {idx: rank for rank, idx in enumerate(fused_ranked, start=1)}
    return scope_meta, dense_rank_map, bm25_rank_map, fused_ranked


def _gold_chunk_info(
    loaded_doc: Any,
    scope_meta: Any,
    expected_pages: list[int],
    expected_answer: str,
    dense_rank_map: dict[int, int],
    bm25_rank_map: dict[int, int],
    hybrid_rank_map: dict[int, int],
) -> dict[str, str]:
    chunk_text_by_id = loaded_doc.chunk_text_by_id
    candidate_rows: list[tuple[int, str, list[int], str]] = []
    expected_pages_set = set(expected_pages)
    for idx in range(len(scope_meta)):
        row = scope_meta.iloc[idx]
        pages = _row_pages(row)
        if expected_pages_set and expected_pages_set.intersection(pages):
            chunk_id = str(row.get("chunk_id_global") or row.get("chunk_id") or "")
            chunk_text = str(chunk_text_by_id.get(chunk_id, ""))
            candidate_rows.append((idx, chunk_id, pages, chunk_text))

    selected_rows = candidate_rows
    match_basis = "expected_pages"
    expected_answer_norm = str(expected_answer or "").strip().lower()
    if expected_answer_norm:
        answer_rows = [row for row in candidate_rows if expected_answer_norm in row[3].lower()]
        if answer_rows:
            selected_rows = answer_rows
            match_basis = "expected_pages+expected_answer"

    if not selected_rows:
        return {
            "gold_chunk_match_basis": "no_gold_chunk_found",
            "gold_chunk_ids": "",
            "selected_gold_chunk_id": "",
            "selected_gold_chunk_pages": "",
            "gold_hybrid_rank": "",
            "gold_dense_rank": "",
            "gold_bm25_rank": "",
        }

    def _rank_key(item: tuple[int, str, list[int], str]) -> tuple[int, int, int]:
        idx = item[0]
        return (
            int(hybrid_rank_map.get(idx, 10**9)),
            int(dense_rank_map.get(idx, 10**9)),
            int(bm25_rank_map.get(idx, 10**9)),
        )

    best = min(selected_rows, key=_rank_key)
    best_idx, best_chunk_id, best_pages, _ = best
    return {
        "gold_chunk_match_basis": match_basis,
        "gold_chunk_ids": json.dumps([row[1] for row in selected_rows], ensure_ascii=True),
        "selected_gold_chunk_id": best_chunk_id,
        "selected_gold_chunk_pages": json.dumps(best_pages, ensure_ascii=True),
        "gold_hybrid_rank": str(hybrid_rank_map.get(best_idx, "")),
        "gold_dense_rank": str(dense_rank_map.get(best_idx, "")),
        "gold_bm25_rank": str(bm25_rank_map.get(best_idx, "")),
    }


def _top_return_info(
    scope_meta: Any,
    fused_ranked: list[int],
    dense_rank_map: dict[int, int],
    bm25_rank_map: dict[int, int],
) -> dict[str, str]:
    if not fused_ranked:
        return {
            "top1_chunk_id": "",
            "top1_pages": "",
            "top1_dense_rank": "",
            "top1_bm25_rank": "",
            "top3_chunk_ids": "",
            "top3_pages": "",
        }

    def _chunk_id(idx: int) -> str:
        row = scope_meta.iloc[idx]
        return str(row.get("chunk_id_global") or row.get("chunk_id") or "")

    top1_idx = fused_ranked[0]
    top1_pages = _row_pages(scope_meta.iloc[top1_idx])
    top3_idxs = fused_ranked[:3]
    top3_chunk_ids = [_chunk_id(idx) for idx in top3_idxs]
    top3_pages = [_row_pages(scope_meta.iloc[idx]) for idx in top3_idxs]

    return {
        "top1_chunk_id": _chunk_id(top1_idx),
        "top1_pages": json.dumps(top1_pages, ensure_ascii=True),
        "top1_dense_rank": str(dense_rank_map.get(top1_idx, "")),
        "top1_bm25_rank": str(bm25_rank_map.get(top1_idx, "")),
        "top3_chunk_ids": json.dumps(top3_chunk_ids, ensure_ascii=True),
        "top3_pages": json.dumps(top3_pages, ensure_ascii=True),
    }


def main() -> None:
    args = parse_args()
    data_root = (REPO_ROOT / args.data_root).resolve()
    model_path = (REPO_ROOT / args.model_path).resolve()
    out_csv = (REPO_ROOT / args.out_csv).resolve()
    out_meta = (REPO_ROOT / args.out_meta).resolve()

    if args.allow_incomplete_corpora:
        candidate_docs = sorted([p for p in data_root.glob(args.doc_pattern) if p.is_dir()])
    else:
        candidate_docs, skipped = list_eval_ready_doc_dirs(data_root, str(args.doc_pattern))
        print_skipped_eval_ready_docs(skipped)
    if not candidate_docs:
        raise FileNotFoundError(f"No document directories matched under {data_root} with pattern {args.doc_pattern}")

    docs_with_items: list[tuple[Path, list[dict[str, Any]]]] = []
    per_doc_counts: dict[str, int] = {}
    for doc_dir in candidate_docs:
        eval_items = read_eval_items(doc_dir / "eval_set.json")
        if not eval_items:
            continue
        docs_with_items.append((doc_dir, eval_items))
        per_doc_counts[doc_dir.name] = len(eval_items)

    if not docs_with_items:
        raise FileNotFoundError(f"No eval_set queries found under {data_root} for pattern {args.doc_pattern}")

    service = SearchService(repo_root=REPO_ROOT, model_path=model_path)

    rows: list[dict[str, str]] = []
    total_queries = 0

    for doc_dir, eval_items in docs_with_items:
        for item in eval_items:
            total_queries += 1
            query_id = str(item.get("query_id") or "").strip()
            question = str(item.get("question") or "").strip()
            expected_pages = [int(x) for x in item.get("expected_pages", []) if str(x).isdigit()]
            scope_meta, dense_rank_map, bm25_rank_map, fused_ranked = _build_live_rankings(
                service=service,
                data_dir=doc_dir,
                question=question,
            )
            hybrid_rank_map = {idx: rank for rank, idx in enumerate(fused_ranked, start=1)}
            rank1_pages = _row_pages(scope_meta.iloc[fused_ranked[0]]) if fused_ranked else []
            page_hit = 1 if (expected_pages and any(p in expected_pages for p in rank1_pages)) else 0
            failure_type = "FP2_MISSED_TOP_RANK" if (expected_pages and page_hit == 0) else "HIT"

            if failure_type != "FP2_MISSED_TOP_RANK":
                continue

            gold_info = _gold_chunk_info(
                loaded_doc=service._load_doc(doc_dir),
                scope_meta=scope_meta,
                expected_pages=expected_pages,
                expected_answer=str(item.get("expected_answer") or ""),
                dense_rank_map=dense_rank_map,
                bm25_rank_map=bm25_rank_map,
                hybrid_rank_map=hybrid_rank_map,
            )
            top_return_info = _top_return_info(
                scope_meta=scope_meta,
                fused_ranked=fused_ranked,
                dense_rank_map=dense_rank_map,
                bm25_rank_map=bm25_rank_map,
            )

            rows.append(
                {
                    "query_id": query_id,
                    "document": str(item.get("doc_id") or doc_dir.name),
                    "difficulty": str(item.get("difficulty") or ""),
                    "question": question,
                    "expected_answer": str(item.get("expected_answer") or ""),
                    "expected_pages": pages_to_text(item.get("expected_pages")),
                    "expected_section": str(item.get("expected_section") or ""),
                    "expected_subsection": str(item.get("expected_subsection") or ""),
                    "evidence_layout": str(item.get("evidence_layout") or ""),
                    "answer_type": str(item.get("answer_type") or ""),
                    "failure_type": failure_type,
                    **gold_info,
                    **top_return_info,
                }
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "generated_utc": utc_now_iso(),
        "repo_root": str(REPO_ROOT),
        "data_root": str(data_root),
        "model_path": str(model_path),
        "doc_pattern": str(args.doc_pattern),
        "documents": [d.name for d, _ in docs_with_items],
        "queries_per_doc": per_doc_counts,
        "total_queries": int(total_queries),
        "fp2_count": int(len(rows)),
        "search_request_k": int(args.k),
        "live_search_settings": {
            "fusion_strategy": str(FUSION_STRATEGY),
            "rrf_k": int(RRF_K),
            "dense_weight": float(RRF_DENSE_WEIGHT),
            "bm25_weight": float(RRF_BM25_WEIGHT),
            "enable_cross_encoder_rerank": bool(ENABLE_CROSS_ENCODER_RERANK),
            "cross_encoder_model": str(CROSS_ENCODER_MODEL_NAME),
            "cross_encoder_topn": int(CROSS_ENCODER_TOPN),
            "cross_encoder_weight": float(CROSS_ENCODER_WEIGHT),
        },
        "environment_overrides": {
            key: os.getenv(key)
            for key in [
                "FUSION_STRATEGY",
                "ENABLE_CROSS_ENCODER_RERANK",
                "CROSS_ENCODER_MODEL_NAME",
                "CROSS_ENCODER_TOPN",
                "CROSS_ENCODER_WEIGHT",
            ]
        },
        "csv_headers": CSV_HEADERS,
        "output_csv": str(out_csv),
    }
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps({"output_csv": str(out_csv), "output_meta": str(out_meta), "fp2_count": len(rows), "total_queries": total_queries}, indent=2))


if __name__ == "__main__":
    main()
