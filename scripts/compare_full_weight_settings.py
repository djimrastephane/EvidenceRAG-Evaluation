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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare full 250-query live retrieval metrics for baseline vs candidate fusion weights."
    )
    parser.add_argument("--data-root", default="data_processed")
    parser.add_argument("--doc-pattern", default="Grampian-20*-20*")
    parser.add_argument("--model-path", default="models/all-MiniLM-L6-v2")
    parser.add_argument("--fusion-strategy", default=FUSION_STRATEGY)
    parser.add_argument("--rrf-k", type=int, default=RRF_K)
    parser.add_argument("--baseline-dense-weight", type=float, default=RRF_DENSE_WEIGHT)
    parser.add_argument("--baseline-bm25-weight", type=float, default=RRF_BM25_WEIGHT)
    parser.add_argument("--candidate-dense-weight", type=float, default=2.0)
    parser.add_argument("--candidate-bm25-weight", type=float, default=0.5)
    parser.add_argument(
        "--out-csv",
        default="results/live_fp2_audit/full_weight_comparison_per_query.csv",
    )
    parser.add_argument(
        "--out-json",
        default="results/live_fp2_audit/full_weight_comparison_summary.json",
    )
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def build_rankings(
    service: SearchService,
    data_dir: Path,
    question: str,
    fusion_strategy: str,
    rrf_k: int,
    dense_weight: float,
    bm25_weight: float,
) -> tuple[Any, list[int]]:
    loaded_doc = service._load_doc(data_dir)
    scope_meta = loaded_doc.meta.reset_index(drop=True)
    scope_index = loaded_doc.index
    chunk_text_by_id = loaded_doc.chunk_text_by_id

    emb = service.model.encode([question], convert_to_numpy=True, normalize_embeddings=False).astype("float32")
    emb = l2_normalize(emb).astype("float32")

    dense_scores, dense_idxs = scope_index.search(emb, len(scope_meta))
    dense_ranked = [int(idx) for idx in dense_idxs[0].tolist()]
    dense_score_map = {int(idx): float(score) for idx, score in zip(dense_idxs[0].tolist(), dense_scores[0].tolist())}

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
    return scope_meta, fused_ranked


def metrics_for_query(expected_pages: list[int], scope_meta: Any, fused_ranked: list[int]) -> dict[str, Any]:
    ranked_pages: list[int] = []
    for idx in fused_ranked[:10]:
        ranked_pages.extend(row_pages(scope_meta.iloc[idx]))
    exp = set(expected_pages)
    top_pages = [row_pages(scope_meta.iloc[idx]) for idx in fused_ranked[:10]]

    def hit_at(k: int) -> int:
        if not exp:
            return 0
        seen: list[int] = []
        for idx in fused_ranked[:k]:
            seen.extend(row_pages(scope_meta.iloc[idx]))
        return int(bool(exp.intersection(seen)))

    mrr = 0.0
    if exp:
        for rank, idx in enumerate(fused_ranked, start=1):
            if exp.intersection(row_pages(scope_meta.iloc[idx])):
                mrr = 1.0 / float(rank)
                break
    top1_chunk_id = ""
    top1_pages = ""
    if fused_ranked:
        row = scope_meta.iloc[fused_ranked[0]]
        top1_chunk_id = str(row.get("chunk_id_global") or row.get("chunk_id") or "")
        top1_pages = json.dumps(row_pages(row), ensure_ascii=True)
    return {
        "hit@1": hit_at(1),
        "hit@3": hit_at(3),
        "hit@5": hit_at(5),
        "hit@10": hit_at(10),
        "mrr@10": mrr,
        "top1_chunk_id": top1_chunk_id,
        "top1_pages": top1_pages,
        "top10_pages": json.dumps(top_pages, ensure_ascii=True),
    }


def main() -> None:
    args = parse_args()
    data_root = (REPO_ROOT / args.data_root).resolve()
    model_path = (REPO_ROOT / args.model_path).resolve()
    out_csv = (REPO_ROOT / args.out_csv).resolve()
    out_json = (REPO_ROOT / args.out_json).resolve()

    docs = sorted([p for p in data_root.glob(args.doc_pattern) if p.is_dir() and read_eval_items(p / "eval_set.json")])
    service = SearchService(repo_root=REPO_ROOT, model_path=model_path)

    rows: list[dict[str, Any]] = []
    totals = {
        "baseline": {"hit@1": 0, "hit@3": 0, "hit@5": 0, "hit@10": 0, "mrr@10": 0.0},
        "candidate": {"hit@1": 0, "hit@3": 0, "hit@5": 0, "hit@10": 0, "mrr@10": 0.0},
    }
    improved_hit1 = 0
    regressed_hit1 = 0
    same_hit1 = 0
    total_queries = 0

    for doc_dir in docs:
        eval_items = read_eval_items(doc_dir / "eval_set.json")
        for item in eval_items:
            total_queries += 1
            question = str(item.get("question") or "").strip()
            expected_pages = [int(x) for x in item.get("expected_pages", []) if str(x).isdigit()]

            base_meta, base_ranked = build_rankings(
                service=service,
                data_dir=doc_dir,
                question=question,
                fusion_strategy=str(args.fusion_strategy),
                rrf_k=int(args.rrf_k),
                dense_weight=float(args.baseline_dense_weight),
                bm25_weight=float(args.baseline_bm25_weight),
            )
            cand_meta, cand_ranked = build_rankings(
                service=service,
                data_dir=doc_dir,
                question=question,
                fusion_strategy=str(args.fusion_strategy),
                rrf_k=int(args.rrf_k),
                dense_weight=float(args.candidate_dense_weight),
                bm25_weight=float(args.candidate_bm25_weight),
            )

            base = metrics_for_query(expected_pages, base_meta, base_ranked)
            cand = metrics_for_query(expected_pages, cand_meta, cand_ranked)

            for key in totals["baseline"]:
                totals["baseline"][key] += base[key]
                totals["candidate"][key] += cand[key]

            if cand["hit@1"] > base["hit@1"]:
                improved_hit1 += 1
            elif cand["hit@1"] < base["hit@1"]:
                regressed_hit1 += 1
            else:
                same_hit1 += 1

            rows.append(
                {
                    "query_id": str(item.get("query_id") or ""),
                    "document": str(item.get("doc_id") or doc_dir.name),
                    "difficulty": str(item.get("difficulty") or ""),
                    "question": question,
                    "expected_pages": json.dumps(expected_pages, ensure_ascii=True),
                    "baseline_hit@1": base["hit@1"],
                    "candidate_hit@1": cand["hit@1"],
                    "baseline_hit@3": base["hit@3"],
                    "candidate_hit@3": cand["hit@3"],
                    "baseline_hit@5": base["hit@5"],
                    "candidate_hit@5": cand["hit@5"],
                    "baseline_hit@10": base["hit@10"],
                    "candidate_hit@10": cand["hit@10"],
                    "baseline_mrr@10": round(float(base["mrr@10"]), 6),
                    "candidate_mrr@10": round(float(cand["mrr@10"]), 6),
                    "delta_hit@1": int(cand["hit@1"] - base["hit@1"]),
                    "delta_mrr@10": round(float(cand["mrr@10"] - base["mrr@10"]), 6),
                    "baseline_top1_chunk_id": base["top1_chunk_id"],
                    "candidate_top1_chunk_id": cand["top1_chunk_id"],
                    "baseline_top1_pages": base["top1_pages"],
                    "candidate_top1_pages": cand["top1_pages"],
                    "baseline_top10_pages": base["top10_pages"],
                    "candidate_top10_pages": cand["top10_pages"],
                }
            )

    rows.sort(key=lambda r: (r["delta_hit@1"], r["delta_mrr@10"]), reverse=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "generated_utc": utc_now_iso(),
        "documents": [d.name for d in docs],
        "total_queries": int(total_queries),
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
        "candidate_settings": {
            "fusion_strategy": str(args.fusion_strategy),
            "rrf_k": int(args.rrf_k),
            "dense_weight": float(args.candidate_dense_weight),
            "bm25_weight": float(args.candidate_bm25_weight),
            "enable_cross_encoder_rerank": bool(ENABLE_CROSS_ENCODER_RERANK),
            "cross_encoder_model": str(CROSS_ENCODER_MODEL_NAME),
            "cross_encoder_topn": int(CROSS_ENCODER_TOPN),
            "cross_encoder_weight": float(CROSS_ENCODER_WEIGHT),
        },
        "baseline_metrics": {
            "hit@1": totals["baseline"]["hit@1"] / float(total_queries),
            "hit@3": totals["baseline"]["hit@3"] / float(total_queries),
            "hit@5": totals["baseline"]["hit@5"] / float(total_queries),
            "hit@10": totals["baseline"]["hit@10"] / float(total_queries),
            "mrr@10": totals["baseline"]["mrr@10"] / float(total_queries),
        },
        "candidate_metrics": {
            "hit@1": totals["candidate"]["hit@1"] / float(total_queries),
            "hit@3": totals["candidate"]["hit@3"] / float(total_queries),
            "hit@5": totals["candidate"]["hit@5"] / float(total_queries),
            "hit@10": totals["candidate"]["hit@10"] / float(total_queries),
            "mrr@10": totals["candidate"]["mrr@10"] / float(total_queries),
        },
        "per_query_hit@1_changes": {
            "improved": int(improved_hit1),
            "regressed": int(regressed_hit1),
            "unchanged": int(same_hit1),
        },
        "output_csv": str(out_csv),
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
