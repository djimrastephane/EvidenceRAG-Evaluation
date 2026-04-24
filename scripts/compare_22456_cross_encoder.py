from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sentence_transformers import CrossEncoder

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rag_pdf.services.search_helpers import read_eval_items
from rag_pdf.services.search_service import (
    BM25Index,
    SearchService,
    l2_normalize,
    rrf_fuse,
    tokenize,
    to_pages_list,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare 224/56 retrieval with cross-encoder reranker off vs on.")
    parser.add_argument("--run-root", default="data_variants/ablation_224_56_5docs")
    parser.add_argument("--run-filter", default="sanity224_")
    parser.add_argument("--model-path", default="models/all-MiniLM-L6-v2")
    parser.add_argument("--rrf-k", type=int, default=20)
    parser.add_argument("--dense-weight", type=float, default=0.5)
    parser.add_argument("--bm25-weight", type=float, default=2.0)
    parser.add_argument("--ce-model", default="models/bge-reranker-v2-m3")
    parser.add_argument("--ce-topn", type=int, default=50)
    parser.add_argument("--ce-weight", type=float, default=0.2)
    parser.add_argument(
        "--out-csv",
        default="results/live_fp2_audit/compare_22456_cross_encoder_per_query.csv",
    )
    parser.add_argument(
        "--out-json",
        default="results/live_fp2_audit/compare_22456_cross_encoder_summary.json",
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
    rrf_k: int,
    dense_weight: float,
    bm25_weight: float,
    cross_encoder: Optional[CrossEncoder],
    ce_topn: int,
    ce_weight: float,
) -> tuple[Any, list[int]]:
    loaded_doc = service._load_doc(data_dir)
    scope_meta = loaded_doc.meta.reset_index(drop=True)
    scope_index = loaded_doc.index
    chunk_text_by_id = loaded_doc.chunk_text_by_id

    emb = service.model.encode([question], convert_to_numpy=True, normalize_embeddings=False).astype("float32")
    emb = l2_normalize(emb).astype("float32")

    dense_scores, dense_idxs = scope_index.search(emb, len(scope_meta))
    dense_ranked = [int(idx) for idx in dense_idxs[0].tolist()]
    bm25_corpus: list[str] = []
    for idx in range(len(scope_meta)):
        row = scope_meta.iloc[idx]
        cid = str(row.get("chunk_id_global") or row.get("chunk_id") or "")
        bm25_corpus.append(chunk_text_by_id.get(cid, ""))
    bm25 = BM25Index([tokenize(t) for t in bm25_corpus], k1=1.5, b=0.75)
    bm25_scores = bm25.score_query(tokenize(question))
    bm25_ranked = [int(idx) for idx, _ in sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)]

    fused_ranked, fused_scores = rrf_fuse(
        dense_ranked=dense_ranked,
        bm25_ranked=bm25_ranked,
        rrf_k=rrf_k,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
    )
    score_map = dict(fused_scores)

    if cross_encoder is not None and fused_ranked:
        cand = fused_ranked[: min(len(fused_ranked), int(ce_topn))]
        pairs: list[tuple[str, str]] = []
        for idx in cand:
            row = scope_meta.iloc[idx]
            cid = str(row.get("chunk_id_global") or row.get("chunk_id") or "")
            pairs.append((question, chunk_text_by_id.get(cid, "")))
        ce_scores_raw = np.asarray(
            cross_encoder.predict(pairs, batch_size=4, show_progress_bar=False),
            dtype=np.float32,
        )
        if ce_scores_raw.size:
            lo = float(np.min(ce_scores_raw))
            hi = float(np.max(ce_scores_raw))
            if hi > lo:
                ce_scores = ((ce_scores_raw - lo) / (hi - lo)).astype(np.float32)
            else:
                ce_scores = np.zeros_like(ce_scores_raw, dtype=np.float32)
            for idx, ce_s in zip(cand, ce_scores.tolist()):
                score_map[idx] = float(score_map.get(idx, 0.0)) + float(ce_weight) * float(ce_s)
            fused_ranked = sorted(fused_ranked, key=lambda i: score_map.get(i, 0.0), reverse=True)

    return scope_meta, fused_ranked


def metrics_for_query(expected_pages: list[int], scope_meta: Any, fused_ranked: list[int]) -> dict[str, Any]:
    exp = set(expected_pages)

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
    }


def main() -> None:
    args = parse_args()
    run_root = (REPO_ROOT / args.run_root).resolve()
    model_path = (REPO_ROOT / args.model_path).resolve()
    ce_model = (REPO_ROOT / args.ce_model).resolve()
    out_csv = (REPO_ROOT / args.out_csv).resolve()
    out_json = (REPO_ROOT / args.out_json).resolve()

    run_dirs = sorted([d for d in run_root.iterdir() if d.is_dir() and args.run_filter in d.name])
    if not run_dirs:
        raise FileNotFoundError(f"No run directories matching {args.run_filter} under {run_root}")

    service = SearchService(repo_root=REPO_ROOT, model_path=model_path)
    cross_encoder = CrossEncoder(str(ce_model), device="cpu")

    rows: list[dict[str, Any]] = []
    totals = {
        "off": {"hit@1": 0, "hit@3": 0, "hit@5": 0, "hit@10": 0, "mrr@10": 0.0, "fp2": 0},
        "on": {"hit@1": 0, "hit@3": 0, "hit@5": 0, "hit@10": 0, "mrr@10": 0.0, "fp2": 0},
    }
    improved_hit1 = 0
    regressed_hit1 = 0
    total_queries = 0
    documents: list[str] = []

    for run_dir in run_dirs:
        doc_dirs = [d for d in run_dir.iterdir() if d.is_dir()]
        if not doc_dirs:
            continue
        data_dir = doc_dirs[0]
        documents.append(data_dir.name)
        eval_items = read_eval_items(data_dir / "eval_set.json")
        for item in eval_items:
            total_queries += 1
            question = str(item.get("question") or "").strip()
            expected_pages = [int(x) for x in item.get("expected_pages", []) if str(x).isdigit()]

            off_meta, off_ranked = build_rankings(
                service=service,
                data_dir=data_dir,
                question=question,
                rrf_k=int(args.rrf_k),
                dense_weight=float(args.dense_weight),
                bm25_weight=float(args.bm25_weight),
                cross_encoder=None,
                ce_topn=int(args.ce_topn),
                ce_weight=float(args.ce_weight),
            )
            on_meta, on_ranked = build_rankings(
                service=service,
                data_dir=data_dir,
                question=question,
                rrf_k=int(args.rrf_k),
                dense_weight=float(args.dense_weight),
                bm25_weight=float(args.bm25_weight),
                cross_encoder=cross_encoder,
                ce_topn=int(args.ce_topn),
                ce_weight=float(args.ce_weight),
            )

            off = metrics_for_query(expected_pages, off_meta, off_ranked)
            on = metrics_for_query(expected_pages, on_meta, on_ranked)

            for key in ("hit@1", "hit@3", "hit@5", "hit@10", "mrr@10"):
                totals["off"][key] += off[key]
                totals["on"][key] += on[key]
            totals["off"]["fp2"] += int(1 - off["hit@1"])
            totals["on"]["fp2"] += int(1 - on["hit@1"])

            if on["hit@1"] > off["hit@1"]:
                improved_hit1 += 1
            elif on["hit@1"] < off["hit@1"]:
                regressed_hit1 += 1

            rows.append(
                {
                    "query_id": str(item.get("query_id") or ""),
                    "document": str(item.get("doc_id") or data_dir.name),
                    "difficulty": str(item.get("difficulty") or ""),
                    "question": question,
                    "expected_pages": json.dumps(expected_pages, ensure_ascii=True),
                    "off_hit@1": off["hit@1"],
                    "on_hit@1": on["hit@1"],
                    "off_hit@3": off["hit@3"],
                    "on_hit@3": on["hit@3"],
                    "off_hit@5": off["hit@5"],
                    "on_hit@5": on["hit@5"],
                    "off_hit@10": off["hit@10"],
                    "on_hit@10": on["hit@10"],
                    "off_mrr@10": round(float(off["mrr@10"]), 6),
                    "on_mrr@10": round(float(on["mrr@10"]), 6),
                    "delta_hit@1": int(on["hit@1"] - off["hit@1"]),
                    "delta_mrr@10": round(float(on["mrr@10"] - off["mrr@10"]), 6),
                    "off_top1_chunk_id": off["top1_chunk_id"],
                    "on_top1_chunk_id": on["top1_chunk_id"],
                    "off_top1_pages": off["top1_pages"],
                    "on_top1_pages": on["top1_pages"],
                    "off_failure_type": "FP2_MISSED_TOP_RANK" if not off["hit@1"] else "HIT",
                    "on_failure_type": "FP2_MISSED_TOP_RANK" if not on["hit@1"] else "HIT",
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
        "run_root": str(run_root),
        "run_filter": str(args.run_filter),
        "documents": documents,
        "total_queries": int(total_queries),
        "fixed_hybrid_settings": {
            "rrf_k": int(args.rrf_k),
            "dense_weight": float(args.dense_weight),
            "bm25_weight": float(args.bm25_weight),
        },
        "cross_encoder_settings": {
            "model": str(ce_model),
            "topn": int(args.ce_topn),
            "weight": float(args.ce_weight),
        },
        "off_metrics": {
            "hit@1": totals["off"]["hit@1"] / float(total_queries),
            "hit@3": totals["off"]["hit@3"] / float(total_queries),
            "hit@5": totals["off"]["hit@5"] / float(total_queries),
            "hit@10": totals["off"]["hit@10"] / float(total_queries),
            "mrr@10": totals["off"]["mrr@10"] / float(total_queries),
            "fp2_count": int(totals["off"]["fp2"]),
        },
        "on_metrics": {
            "hit@1": totals["on"]["hit@1"] / float(total_queries),
            "hit@3": totals["on"]["hit@3"] / float(total_queries),
            "hit@5": totals["on"]["hit@5"] / float(total_queries),
            "hit@10": totals["on"]["hit@10"] / float(total_queries),
            "mrr@10": totals["on"]["mrr@10"] / float(total_queries),
            "fp2_count": int(totals["on"]["fp2"]),
        },
        "per_query_hit@1_changes": {
            "improved": int(improved_hit1),
            "regressed": int(regressed_hit1),
            "unchanged": int(total_queries - improved_hit1 - regressed_hit1),
        },
        "output_csv": str(out_csv),
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
