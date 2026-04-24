from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PARENT_PATH = REPO_ROOT.parent
SRC_PATH = REPO_ROOT / "src"
SCRIPTS_PATH = REPO_ROOT / "scripts"
for path in [PARENT_PATH, REPO_ROOT, SRC_PATH, SCRIPTS_PATH]:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rag_pdf.services.search_helpers import read_eval_items
from rag_pdf.services.search_service import (
    CROSS_ENCODER_MODEL_NAME,
    CROSS_ENCODER_TOPN,
    CROSS_ENCODER_WEIGHT,
    ENABLE_CROSS_ENCODER_RERANK,
    FUSION_STRATEGY,
    RRF_BM25_WEIGHT,
    RRF_DENSE_WEIGHT,
    RRF_K,
    SearchService,
)

try:
    from scripts.retrieval_eval import categorize_failure_type, compute_gold_presence, score_answer_correctness
except ModuleNotFoundError:
    from retrieval_eval import categorize_failure_type, compute_gold_presence, score_answer_correctness


FP_STAGE = {
    "FP1_MISSING_CONTENT": "retrieval",
    "FP2_MISSED_TOP_RANK": "retrieval",
    "FP3_NOT_IN_CONTEXT": "retrieval",
    "FP4_NOT_EXTRACTED": "generation",
    "FP5_WRONG_FORMAT": "generation",
    "FP6_INCORRECT_SPECIFICITY": "generation",
    "FP7_INCOMPLETE": "generation",
    "HIT": "none",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rescore FP1-FP7 using saved generated answers and current correctness rules.")
    parser.add_argument("--data-root", default="data_processed")
    parser.add_argument("--doc-pattern", default="Grampian-20*-20*")
    parser.add_argument("--model-path", default="models/all-MiniLM-L6-v2")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--saved-generation-csv",
        default="results/live_fp1_fp7_current_pipeline_llm_2026-03-17/current_pipeline_fp1_fp7_per_query.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="results/live_fp1_fp7_current_pipeline_llm_norm_2026-03-17",
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


def pretty_series(doc_id: str) -> str:
    return str(doc_id).replace("Grampian-", "")


def load_saved_generation(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {
        (str(row.get("document") or "").strip(), str(row.get("query_id") or "").strip()): row
        for row in rows
    }


def main() -> None:
    args = parse_args()
    data_root = (REPO_ROOT / args.data_root).resolve()
    model_path = (REPO_ROOT / args.model_path).resolve()
    saved_generation_csv = (REPO_ROOT / args.saved_generation_csv).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_map = load_saved_generation(saved_generation_csv)
    docs = sorted([p for p in data_root.glob(args.doc_pattern) if p.is_dir() and read_eval_items(p / "eval_set.json")])
    if not docs:
        raise FileNotFoundError(f"No docs with eval sets found under {data_root} matching {args.doc_pattern}")

    service = SearchService(repo_root=REPO_ROOT, model_path=model_path)
    query_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []

    for doc_dir in docs:
        doc_id = doc_dir.name
        eval_items = read_eval_items(doc_dir / "eval_set.json")
        loaded_doc = service._load_doc(doc_dir)
        meta = loaded_doc.meta
        doc_counter: Counter[str] = Counter()

        for item in eval_items:
            query_id = str(item.get("query_id") or "").strip()
            question = str(item.get("question") or "").strip()
            expected_pages = [int(x) for x in item.get("expected_pages", []) if str(x).isdigit()]
            expected_pages_set = set(expected_pages)
            expected_answer = item.get("expected_answer")
            answer_type = str(item.get("answer_type") or "unknown")

            out = service.search(
                data_dir=doc_dir,
                question=question,
                k=int(args.k),
                query_id=query_id or None,
                include_generated_answer=False,
            )

            saved = saved_map.get((doc_id, query_id), {})
            saved_generated_answer = str(saved.get("generated_answer") or "").strip()
            extracted_answer = saved_generated_answer if saved_generated_answer else None

            gold_presence = compute_gold_presence(meta, doc_id, expected_pages_set)
            results = out.get("results") or []
            top1_pages = list(results[0].get("pages") or []) if results else []
            page_hit = 1 if (expected_pages_set and expected_pages_set.intersection(top1_pages)) else 0
            context_results = results[:3] if len(results) >= 3 else results[:1]
            context_text = "\n".join(str(r.get("chunk_text") or "") for r in context_results if str(r.get("chunk_text") or "").strip())

            failure_type = categorize_failure_type(
                page_hit=page_hit,
                gold_exists=bool(gold_presence.get("gold_exists", False)),
                expected_answer=expected_answer,
                answer_type=answer_type,
                context_text=context_text,
                extracted_answer=extracted_answer,
            )
            answer_correct, answer_status = score_answer_correctness(
                expected_answer=expected_answer,
                answer_type=answer_type,
                extracted_answer=extracted_answer,
            )

            top3_chunk_ids = [str(r.get("chunk_id") or "") for r in results[:3]]
            top3_pages = [list(r.get("pages") or []) for r in results[:3]]

            query_rows.append(
                {
                    "query_id": query_id,
                    "document": doc_id,
                    "difficulty": str(item.get("difficulty") or ""),
                    "question": question,
                    "expected_answer": str(expected_answer or ""),
                    "expected_pages": pages_to_text(expected_pages),
                    "expected_section": str(item.get("expected_section") or ""),
                    "expected_subsection": str(item.get("expected_subsection") or ""),
                    "evidence_layout": str(item.get("evidence_layout") or ""),
                    "answer_type": answer_type,
                    "failure_type": failure_type,
                    "failure_stage": FP_STAGE.get(failure_type, "unknown"),
                    "page_hit": int(page_hit),
                    "gold_exists": bool(gold_presence.get("gold_exists", False)),
                    "gold_chunk_count": int(gold_presence.get("gold_chunk_count", 0)),
                    "gold_pages_found": pages_to_text(gold_presence.get("gold_pages_found", [])),
                    "extracted_answer": (str(extracted_answer) if extracted_answer is not None else ""),
                    "answer_correct": answer_correct,
                    "answer_status": answer_status,
                    "include_generated_answer": True,
                    "generation_status": str(saved.get("generation_status") or ""),
                    "generation_confidence": saved.get("generation_confidence"),
                    "generated_answer": saved_generated_answer,
                    "retrieval_mode": str(out.get("retrieval_mode") or ""),
                    "top1_chunk_id": (str(results[0].get("chunk_id") or "") if results else ""),
                    "top1_pages": pages_to_text(top1_pages),
                    "top3_chunk_ids": pages_to_text(top3_chunk_ids),
                    "top3_pages": pages_to_text(top3_pages),
                }
            )
            doc_counter[failure_type] += 1

        for code in ["FP1", "FP2", "FP3", "FP4", "FP5", "FP6", "FP7"]:
            full = {
                "FP1": "FP1_MISSING_CONTENT",
                "FP2": "FP2_MISSED_TOP_RANK",
                "FP3": "FP3_NOT_IN_CONTEXT",
                "FP4": "FP4_NOT_EXTRACTED",
                "FP5": "FP5_WRONG_FORMAT",
                "FP6": "FP6_INCORRECT_SPECIFICITY",
                "FP7": "FP7_INCOMPLETE",
            }[code]
            count_rows.append(
                {
                    "series": pretty_series(doc_id),
                    "doc_id": doc_id,
                    "fp_code": code,
                    "failure_type": full,
                    "count": int(doc_counter.get(full, 0)),
                }
            )

    per_query_csv = out_dir / "current_pipeline_fp1_fp7_per_query.csv"
    counts_csv = out_dir / "current_pipeline_fp1_fp7_counts.csv"
    summary_json = out_dir / "current_pipeline_fp1_fp7_summary.json"

    pd.DataFrame(query_rows).to_csv(per_query_csv, index=False)
    pd.DataFrame(count_rows).to_csv(counts_csv, index=False)

    total_counter = Counter(row["failure_type"] for row in query_rows)
    summary = {
        "generated_utc": utc_now_iso(),
        "data_root": str(data_root),
        "documents": [d.name for d in docs],
        "queries_per_doc": {d.name: len(read_eval_items(d / "eval_set.json")) for d in docs},
        "total_queries": int(len(query_rows)),
        "failure_counts_total": dict(total_counter),
        "current_pipeline_settings": {
            "fusion_strategy": str(FUSION_STRATEGY),
            "rrf_k": int(RRF_K),
            "dense_weight": float(RRF_DENSE_WEIGHT),
            "bm25_weight": float(RRF_BM25_WEIGHT),
            "enable_cross_encoder_rerank": bool(ENABLE_CROSS_ENCODER_RERANK),
            "cross_encoder_model": str(CROSS_ENCODER_MODEL_NAME),
            "cross_encoder_topn": int(CROSS_ENCODER_TOPN),
            "cross_encoder_weight": float(CROSS_ENCODER_WEIGHT),
            "k": int(args.k),
            "include_generated_answer": True,
            "rescore_mode": "saved_generation_reused",
            "saved_generation_csv": str(saved_generation_csv),
        },
        "artifacts": {
            "per_query_csv": str(per_query_csv),
            "counts_csv": str(counts_csv),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
