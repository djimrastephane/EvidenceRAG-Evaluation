from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

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
    "baseline_failure_type",
    "variant_failure_type",
    "recovered_to_hit_at_1",
    "baseline_gold_chunk_id",
    "variant_gold_chunk_id",
    "baseline_gold_hybrid_rank",
    "variant_gold_hybrid_rank",
    "baseline_gold_dense_rank",
    "variant_gold_dense_rank",
    "baseline_gold_bm25_rank",
    "variant_gold_bm25_rank",
    "baseline_top1_chunk_id",
    "variant_top1_chunk_id",
    "baseline_top1_pages",
    "variant_top1_pages",
    "baseline_top3_chunk_ids",
    "variant_top3_chunk_ids",
    "baseline_top3_pages",
    "variant_top3_pages",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare current FP2 cases against a denser-biased fusion rerun over the 250-query live corpus."
    )
    parser.add_argument("--data-root", default="data_processed")
    parser.add_argument("--doc-pattern", default="Grampian-20*-20*")
    parser.add_argument(
        "--allow-incomplete-corpora",
        action="store_true",
        help="Include matching doc folders even if they are missing canonical evaluation artifacts.",
    )
    parser.add_argument("--model-path", default="models/all-MiniLM-L6-v2")
    parser.add_argument(
        "--baseline-fp2-csv",
        default="results/live_fp2_audit/current_pipeline_fp2_5docs.csv",
        help="Current live FP2 export used to define the baseline 64 cases.",
    )
    parser.add_argument(
        "--out-csv",
        default="results/live_fp2_audit/fp2_before_after_dense_bias.csv",
    )
    parser.add_argument(
        "--out-meta",
        default="results/live_fp2_audit/fp2_before_after_dense_bias.meta.json",
    )
    parser.add_argument("--fusion-strategy", default=FUSION_STRATEGY)
    parser.add_argument("--rrf-k", type=int, default=RRF_K)
    parser.add_argument("--baseline-dense-weight", type=float, default=RRF_DENSE_WEIGHT)
    parser.add_argument("--baseline-bm25-weight", type=float, default=RRF_BM25_WEIGHT)
    parser.add_argument("--variant-dense-weight", type=float, default=2.0)
    parser.add_argument("--variant-bm25-weight", type=float, default=0.5)
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def pages_to_text(value: Any) -> str:
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=True)
    if value is None:
        return ""
    return str(value)


def row_pages(row: Any) -> list[int]:
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


def load_baseline_qids(path: Path) -> set[str]:
    with path.open() as f:
        return {str(row["query_id"]).strip() for row in csv.DictReader(f)}


def build_rankings(
    service: SearchService,
    data_dir: Path,
    question: str,
    fusion_strategy: str,
    rrf_k: int,
    dense_weight: float,
    bm25_weight: float,
) -> dict[str, Any]:
    loaded_doc = service._load_doc(data_dir)
    scope_meta = loaded_doc.meta.reset_index(drop=True)
    scope_index = loaded_doc.index
    chunk_text_by_id = loaded_doc.chunk_text_by_id

    emb = service.model.encode([question], convert_to_numpy=True, normalize_embeddings=False).astype("float32")
    emb = l2_normalize(emb).astype("float32")

    dense_scores, dense_idxs = scope_index.search(emb, len(scope_meta))
    dense_ranked = [int(idx) for idx in dense_idxs[0].tolist()]
    dense_score_map = {int(idx): float(score) for idx, score in zip(dense_idxs[0].tolist(), dense_scores[0].tolist())}
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

    strategy = fusion_strategy if fusion_strategy in {"rrf", "score_fusion"} else "rrf"
    if strategy == "score_fusion":
        fused_ranked, fused_scores = score_fuse(
            dense_score_map=dense_score_map,
            bm25_score_map=bm25_score_map,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
        )
    else:
        fused_ranked, fused_scores = rrf_fuse(
            dense_ranked=dense_ranked,
            bm25_ranked=bm25_ranked,
            rrf_k=rrf_k,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
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
        ce_scores_raw = np.asarray(service.cross_encoder.predict(pairs), dtype=np.float32)
        if ce_scores_raw.size:
            lo = float(np.min(ce_scores_raw))
            hi = float(np.max(ce_scores_raw))
            if hi > lo:
                ce_scores = ((ce_scores_raw - lo) / (hi - lo)).astype(np.float32)
            else:
                ce_scores = np.zeros_like(ce_scores_raw, dtype=np.float32)
            for idx, ce_s in zip(cand, ce_scores.tolist()):
                scores_map[idx] = float(scores_map.get(idx, 0.0)) + service.cross_encoder_weight * float(ce_s)
            fused_ranked = sorted(fused_ranked, key=lambda i: scores_map.get(i, 0.0), reverse=True)

    hybrid_rank_map = {idx: rank for rank, idx in enumerate(fused_ranked, start=1)}
    return {
        "loaded_doc": loaded_doc,
        "scope_meta": scope_meta,
        "dense_rank_map": dense_rank_map,
        "bm25_rank_map": bm25_rank_map,
        "hybrid_rank_map": hybrid_rank_map,
        "fused_ranked": fused_ranked,
    }


def extract_gold_info(
    loaded_doc: Any,
    scope_meta: Any,
    expected_pages: list[int],
    expected_answer: str,
    dense_rank_map: dict[int, int],
    bm25_rank_map: dict[int, int],
    hybrid_rank_map: dict[int, int],
) -> dict[str, str]:
    chunk_text_by_id = loaded_doc.chunk_text_by_id
    candidates: list[tuple[int, str, list[int], str]] = []
    expected_pages_set = set(expected_pages)
    for idx in range(len(scope_meta)):
        row = scope_meta.iloc[idx]
        pages = row_pages(row)
        if expected_pages_set and expected_pages_set.intersection(pages):
            chunk_id = str(row.get("chunk_id_global") or row.get("chunk_id") or "")
            candidates.append((idx, chunk_id, pages, str(chunk_text_by_id.get(chunk_id, ""))))

    if not candidates:
        return {
            "gold_chunk_id": "",
            "gold_hybrid_rank": "",
            "gold_dense_rank": "",
            "gold_bm25_rank": "",
        }

    expected_answer_norm = str(expected_answer or "").strip().lower()
    if expected_answer_norm:
        filtered = [row for row in candidates if expected_answer_norm in row[3].lower()]
        if filtered:
            candidates = filtered

    best = min(
        candidates,
        key=lambda item: (
            int(hybrid_rank_map.get(item[0], 10**9)),
            int(dense_rank_map.get(item[0], 10**9)),
            int(bm25_rank_map.get(item[0], 10**9)),
        ),
    )
    best_idx, best_chunk_id, _, _ = best
    return {
        "gold_chunk_id": best_chunk_id,
        "gold_hybrid_rank": str(hybrid_rank_map.get(best_idx, "")),
        "gold_dense_rank": str(dense_rank_map.get(best_idx, "")),
        "gold_bm25_rank": str(bm25_rank_map.get(best_idx, "")),
    }


def extract_top_info(scope_meta: Any, fused_ranked: list[int], dense_rank_map: dict[int, int], bm25_rank_map: dict[int, int]) -> dict[str, str]:
    if not fused_ranked:
        return {
            "top1_chunk_id": "",
            "top1_pages": "",
            "top1_dense_rank": "",
            "top1_bm25_rank": "",
            "top3_chunk_ids": "",
            "top3_pages": "",
        }

    def chunk_id(idx: int) -> str:
        row = scope_meta.iloc[idx]
        return str(row.get("chunk_id_global") or row.get("chunk_id") or "")

    top1_idx = fused_ranked[0]
    top3_idxs = fused_ranked[:3]
    return {
        "top1_chunk_id": chunk_id(top1_idx),
        "top1_pages": json.dumps(row_pages(scope_meta.iloc[top1_idx]), ensure_ascii=True),
        "top1_dense_rank": str(dense_rank_map.get(top1_idx, "")),
        "top1_bm25_rank": str(bm25_rank_map.get(top1_idx, "")),
        "top3_chunk_ids": json.dumps([chunk_id(idx) for idx in top3_idxs], ensure_ascii=True),
        "top3_pages": json.dumps([row_pages(scope_meta.iloc[idx]) for idx in top3_idxs], ensure_ascii=True),
    }


def page_hit_at_1(expected_pages: list[int], fused_ranked: list[int], scope_meta: Any) -> bool:
    if not expected_pages or not fused_ranked:
        return False
    return bool(set(expected_pages).intersection(row_pages(scope_meta.iloc[fused_ranked[0]])))


def main() -> None:
    args = parse_args()
    data_root = (REPO_ROOT / args.data_root).resolve()
    model_path = (REPO_ROOT / args.model_path).resolve()
    baseline_fp2_csv = (REPO_ROOT / args.baseline_fp2_csv).resolve()
    out_csv = (REPO_ROOT / args.out_csv).resolve()
    out_meta = (REPO_ROOT / args.out_meta).resolve()

    baseline_qids = load_baseline_qids(baseline_fp2_csv)
    if args.allow_incomplete_corpora:
        docs = sorted([p for p in data_root.glob(args.doc_pattern) if p.is_dir() and read_eval_items(p / "eval_set.json")])
    else:
        docs, skipped = list_eval_ready_doc_dirs(data_root, str(args.doc_pattern))
        print_skipped_eval_ready_docs(skipped)
        docs = [p for p in docs if read_eval_items(p / "eval_set.json")]
    service = SearchService(repo_root=REPO_ROOT, model_path=model_path)

    rows: list[dict[str, str]] = []
    total_queries = 0
    variant_hit1_total = 0

    for doc_dir in docs:
        eval_items = read_eval_items(doc_dir / "eval_set.json")
        for item in eval_items:
            total_queries += 1
            qid = str(item.get("query_id") or "").strip()
            question = str(item.get("question") or "").strip()
            expected_pages = [int(x) for x in item.get("expected_pages", []) if str(x).isdigit()]
            expected_answer = str(item.get("expected_answer") or "")

            baseline = build_rankings(
                service=service,
                data_dir=doc_dir,
                question=question,
                fusion_strategy=str(args.fusion_strategy),
                rrf_k=int(args.rrf_k),
                dense_weight=float(args.baseline_dense_weight),
                bm25_weight=float(args.baseline_bm25_weight),
            )
            variant = build_rankings(
                service=service,
                data_dir=doc_dir,
                question=question,
                fusion_strategy=str(args.fusion_strategy),
                rrf_k=int(args.rrf_k),
                dense_weight=float(args.variant_dense_weight),
                bm25_weight=float(args.variant_bm25_weight),
            )

            baseline_hit1 = page_hit_at_1(expected_pages, baseline["fused_ranked"], baseline["scope_meta"])
            variant_hit1 = page_hit_at_1(expected_pages, variant["fused_ranked"], variant["scope_meta"])
            if variant_hit1:
                variant_hit1_total += 1

            if qid not in baseline_qids:
                continue

            baseline_gold = extract_gold_info(
                loaded_doc=baseline["loaded_doc"],
                scope_meta=baseline["scope_meta"],
                expected_pages=expected_pages,
                expected_answer=expected_answer,
                dense_rank_map=baseline["dense_rank_map"],
                bm25_rank_map=baseline["bm25_rank_map"],
                hybrid_rank_map=baseline["hybrid_rank_map"],
            )
            variant_gold = extract_gold_info(
                loaded_doc=variant["loaded_doc"],
                scope_meta=variant["scope_meta"],
                expected_pages=expected_pages,
                expected_answer=expected_answer,
                dense_rank_map=variant["dense_rank_map"],
                bm25_rank_map=variant["bm25_rank_map"],
                hybrid_rank_map=variant["hybrid_rank_map"],
            )
            baseline_top = extract_top_info(
                scope_meta=baseline["scope_meta"],
                fused_ranked=baseline["fused_ranked"],
                dense_rank_map=baseline["dense_rank_map"],
                bm25_rank_map=baseline["bm25_rank_map"],
            )
            variant_top = extract_top_info(
                scope_meta=variant["scope_meta"],
                fused_ranked=variant["fused_ranked"],
                dense_rank_map=variant["dense_rank_map"],
                bm25_rank_map=variant["bm25_rank_map"],
            )

            rows.append(
                {
                    "query_id": qid,
                    "document": str(item.get("doc_id") or doc_dir.name),
                    "difficulty": str(item.get("difficulty") or ""),
                    "question": question,
                    "expected_answer": expected_answer,
                    "expected_pages": pages_to_text(item.get("expected_pages")),
                    "expected_section": str(item.get("expected_section") or ""),
                    "expected_subsection": str(item.get("expected_subsection") or ""),
                    "evidence_layout": str(item.get("evidence_layout") or ""),
                    "answer_type": str(item.get("answer_type") or ""),
                    "baseline_failure_type": "FP2_MISSED_TOP_RANK" if not baseline_hit1 else "HIT",
                    "variant_failure_type": "FP2_MISSED_TOP_RANK" if not variant_hit1 else "HIT",
                    "recovered_to_hit_at_1": "yes" if (not baseline_hit1 and variant_hit1) else "no",
                    "baseline_gold_chunk_id": baseline_gold["gold_chunk_id"],
                    "variant_gold_chunk_id": variant_gold["gold_chunk_id"],
                    "baseline_gold_hybrid_rank": baseline_gold["gold_hybrid_rank"],
                    "variant_gold_hybrid_rank": variant_gold["gold_hybrid_rank"],
                    "baseline_gold_dense_rank": baseline_gold["gold_dense_rank"],
                    "variant_gold_dense_rank": variant_gold["gold_dense_rank"],
                    "baseline_gold_bm25_rank": baseline_gold["gold_bm25_rank"],
                    "variant_gold_bm25_rank": variant_gold["gold_bm25_rank"],
                    "baseline_top1_chunk_id": baseline_top["top1_chunk_id"],
                    "variant_top1_chunk_id": variant_top["top1_chunk_id"],
                    "baseline_top1_pages": baseline_top["top1_pages"],
                    "variant_top1_pages": variant_top["top1_pages"],
                    "baseline_top3_chunk_ids": baseline_top["top3_chunk_ids"],
                    "variant_top3_chunk_ids": variant_top["top3_chunk_ids"],
                    "baseline_top3_pages": baseline_top["top3_pages"],
                    "variant_top3_pages": variant_top["top3_pages"],
                }
            )

    rows.sort(key=lambda r: (r["recovered_to_hit_at_1"] != "yes", r["document"], r["query_id"]))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    recovered = sum(1 for row in rows if row["recovered_to_hit_at_1"] == "yes")
    meta = {
        "generated_utc": utc_now_iso(),
        "repo_root": str(REPO_ROOT),
        "data_root": str(data_root),
        "documents": [d.name for d in docs],
        "total_queries_rerun": int(total_queries),
        "baseline_fp2_case_count": int(len(rows)),
        "recovered_from_baseline_fp2": int(recovered),
        "variant_hit_at_1_total_all_queries": int(variant_hit1_total),
        "baseline_settings": {
            "fusion_strategy": str(args.fusion_strategy),
            "rrf_k": int(args.rrf_k),
            "dense_weight": float(args.baseline_dense_weight),
            "bm25_weight": float(args.baseline_bm25_weight),
            "enable_cross_encoder_rerank": bool(ENABLE_CROSS_ENCODER_RERANK),
            "cross_encoder_model": str(CROSS_ENCODER_MODEL_NAME),
            "cross_encoder_topn": int(CROSS_ENCODER_TOPN),
            "cross_encoder_weight": float(CROSS_ENCODER_WEIGHT),
        },
        "variant_settings": {
            "fusion_strategy": str(args.fusion_strategy),
            "rrf_k": int(args.rrf_k),
            "dense_weight": float(args.variant_dense_weight),
            "bm25_weight": float(args.variant_bm25_weight),
            "enable_cross_encoder_rerank": bool(ENABLE_CROSS_ENCODER_RERANK),
            "cross_encoder_model": str(CROSS_ENCODER_MODEL_NAME),
            "cross_encoder_topn": int(CROSS_ENCODER_TOPN),
            "cross_encoder_weight": float(CROSS_ENCODER_WEIGHT),
        },
        "baseline_fp2_csv": str(baseline_fp2_csv),
        "output_csv": str(out_csv),
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"output_csv": str(out_csv), "output_meta": str(out_meta), "baseline_fp2_case_count": len(rows), "recovered_from_baseline_fp2": recovered}, indent=2))


if __name__ == "__main__":
    main()
