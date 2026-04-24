"""Run the FP1-FP7 failure-point analysis across all Grampian cohorts.

Iterates over each eval-ready document folder, runs the hybrid SearchService pipeline,
and classifies each query result into one of seven failure categories: FP1 (content
missing from corpus), FP2 (gold page retrieved but not top-ranked), FP3 (gold page
absent from top-10), FP4 (answer not extracted), FP5 (wrong format), FP6 (incorrect
specificity), FP7 (incomplete answer), or HIT. Writes per-query CSVs, aggregate counts,
and a summary JSON to the specified output directory.
"""

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
if str(PARENT_PATH) not in sys.path:
    sys.path.insert(0, str(PARENT_PATH))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if SCRIPTS_PATH.exists() and str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))

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
from corpus_guard import list_eval_ready_doc_dirs, print_skipped_eval_ready_docs

try:
    from scripts.retrieval_eval import compute_gold_presence
except (ModuleNotFoundError, ImportError):
    try:
        from retrieval_eval import compute_gold_presence
    except (ModuleNotFoundError, ImportError):
        compute_gold_presence = None  # defined inline below

import re as _re


def _extract_nums(text: str) -> set[str]:
    return set(_re.findall(r"\d[\d,\.]*", str(text or "")))


def categorize_failure_type(
    page_hit: int,
    gold_exists: bool,
    expected_answer,
    answer_type: str,
    context_text: str,
    extracted_answer,
    gold_in_top10: bool = False,
    gold_in_top3: bool = False,
) -> str:
    """Classify a single query result into HIT or one of the FP1-FP7 failure categories.

    Retrieval failures (page_hit=0) map to FP1 (content absent), FP2 (gold retrieved but
    not top-ranked), or FP3 (gold absent from top-10). Generation failures (page_hit=1)
    map to FP4 (not extracted), FP5 (wrong format), FP6 (incorrect specificity), or
    FP7 (incomplete answer).
    """
    ea   = str(expected_answer  or "").strip().lower()
    extr = str(extracted_answer or "").strip().lower()

    if page_hit:
        # Retrieval succeeded — evaluate answer quality
        if not ea:
            return "HIT"
        if ea in extr:
            return "HIT"
        if not extr:
            return "FP4_NOT_EXTRACTED"
        if answer_type in ("number", "date", "currency"):
            ea_nums   = _extract_nums(ea)
            extr_nums = _extract_nums(extr)
            if ea_nums and (ea_nums & extr_nums):
                return "HIT"
            if extr_nums:
                return "FP5_WRONG_FORMAT"
            return "FP4_NOT_EXTRACTED"
        # Text answer
        ea_words   = set(ea.split())
        extr_words = set(extr.split())
        overlap    = ea_words & extr_words
        if len(overlap) >= max(1, len(ea_words) * 0.5):
            return "HIT"
        if len(extr) < len(ea) * 0.4:
            return "FP7_INCOMPLETE"
        return "FP6_INCORRECT_SPECIFICITY"
    else:
        # Retrieval failed — use page-rank presence for FP2 vs FP3
        if not gold_exists:
            return "FP1_MISSING_CONTENT"
        # FP2: gold page retrieved (in top-10) but not at rank 1
        if gold_in_top10:
            return "FP2_MISSED_TOP_RANK"
        # FP3: gold page not retrieved at all in top-10
        return "FP3_NOT_IN_CONTEXT"


def score_answer_correctness(expected_answer, answer_type: str, extracted_answer):
    """Return (correct: bool, status: str) for an extracted answer against the expected value."""
    ea   = str(expected_answer  or "").strip().lower()
    extr = str(extracted_answer or "").strip().lower()
    if not ea:
        correct = True
        status  = "no_expected_answer"
    elif ea in extr:
        correct = True
        status  = "exact_match"
    elif answer_type in ("number", "date", "currency"):
        ea_nums   = _extract_nums(ea)
        extr_nums = _extract_nums(extr)
        correct   = bool(ea_nums and (ea_nums & extr_nums))
        status    = "numeric_match" if correct else "numeric_mismatch"
    else:
        correct = False
        status  = "no_match"
    return correct, status


if compute_gold_presence is None:
    def compute_gold_presence(meta, expected_doc_id: str, expected_pages: set):
        import pandas as pd
        df = meta
        if expected_doc_id and "doc_id" in df.columns:
            df = df[df["doc_id"].astype(str) == expected_doc_id]
        if len(df) == 0:
            return {"gold_exists": False, "gold_chunk_count": 0, "gold_pages_found": []}
        pages_found: set = set()
        gold_chunk_count = 0
        for _, row in df.iterrows():
            pages_raw = row.get("pages") or []
            if isinstance(pages_raw, str):
                import json
                try:
                    pages_raw = json.loads(pages_raw)
                except Exception:
                    pages_raw = []
            pages = set(int(p) for p in pages_raw if str(p).isdigit())
            if not pages:
                pn = row.get("page_number") or row.get("page_start")
                if pn is not None:
                    pages = {int(pn)}
            if expected_pages & pages:
                gold_chunk_count += 1
                pages_found.update(expected_pages & pages)
        return {
            "gold_exists": bool(gold_chunk_count > 0),
            "gold_chunk_count": int(gold_chunk_count),
            "gold_pages_found": sorted(list(pages_found)),
        }


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
    parser = argparse.ArgumentParser(description="Run full FP1-FP7 analysis on the current hybrid SearchService pipeline.")
    parser.add_argument("--data-root", default="data_processed")
    parser.add_argument("--doc-pattern", default="Grampian-20*-20*")
    parser.add_argument(
        "--allow-incomplete-corpora",
        action="store_true",
        help="Include matching doc folders even if they are missing canonical evaluation artifacts.",
    )
    parser.add_argument("--model-path", default="models/all-MiniLM-L6-v2")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--include-generated-answer",
        action="store_true",
        help="Enable local LLM answer generation via SearchService for FP4-FP7 evaluation.",
    )
    parser.add_argument("--gen-max-context-chunks", type=int, default=None)
    parser.add_argument("--gen-max-context-chars", type=int, default=None)
    parser.add_argument("--gen-max-chunk-chars", type=int, default=None)
    parser.add_argument("--gen-timeout-seconds", type=float, default=None)
    parser.add_argument(
        "--out-dir",
        default="results/live_fp1_fp7_current_pipeline_2026-03-17",
        help="Directory to write per-query results, counts, and heatmap.",
    )
    return parser.parse_args()


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def pages_to_text(value: Any) -> str:
    """Serialise a page list (or any value) to a plain string for CSV output."""
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=True)
    if value is None:
        return ""
    return str(value)


def pretty_series(doc_id: str) -> str:
    """Strip the 'Grampian-' prefix for compact display in heatmap axis labels."""
    return str(doc_id).replace("Grampian-", "")


def main() -> None:
    args = parse_args()
    data_root = (REPO_ROOT / args.data_root).resolve()
    model_path = (REPO_ROOT / args.model_path).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.allow_incomplete_corpora:
        docs = sorted([p for p in data_root.glob(args.doc_pattern) if p.is_dir() and read_eval_items(p / "eval_set.json")])
    else:
        docs, skipped = list_eval_ready_doc_dirs(data_root, str(args.doc_pattern))
        print_skipped_eval_ready_docs(skipped)
        docs = [p for p in docs if read_eval_items(p / "eval_set.json")]
    if not docs:
        raise FileNotFoundError(f"No docs with eval sets found under {data_root} matching {args.doc_pattern}")

    service = SearchService(repo_root=REPO_ROOT, model_path=model_path)
    query_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    generation_overrides = {
        "max_context_chunks": args.gen_max_context_chunks,
        "max_context_chars": args.gen_max_context_chars,
        "max_chunk_chars": args.gen_max_chunk_chars,
        "timeout_seconds": args.gen_timeout_seconds,
    }
    generation_overrides = {k: v for k, v in generation_overrides.items() if v is not None}

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
                include_generated_answer=bool(args.include_generated_answer),
                generation_overrides=generation_overrides,
            )

            gold_presence = compute_gold_presence(meta, doc_id, expected_pages_set)
            results = out.get("results") or []
            top1_pages = list(results[0].get("pages") or []) if results else []
            page_hit = 1 if (expected_pages_set and expected_pages_set.intersection(top1_pages)) else 0

            # Pre-compute gold page presence in top-3 and top-10
            def _result_pages(r) -> set:
                return set(int(p) for p in (r.get("pages") or []) if str(p).isdigit())
            all_top10_pages = set().union(*[_result_pages(r) for r in results[:10]]) if results else set()
            all_top3_pages  = set().union(*[_result_pages(r) for r in results[:3]])  if results else set()
            gold_in_top10   = bool(expected_pages_set and expected_pages_set & all_top10_pages)
            gold_in_top3    = bool(expected_pages_set and expected_pages_set & all_top3_pages)

            context_results = results[:3] if len(results) >= 3 else results[:1]
            context_text = "\n".join(str(r.get("chunk_text") or "") for r in context_results if str(r.get("chunk_text") or "").strip())
            extracted_answer = out.get("generated_answer") if args.include_generated_answer else out.get("predicted_answer")
            failure_type = categorize_failure_type(
                page_hit=page_hit,
                gold_exists=bool(gold_presence.get("gold_exists", False)),
                expected_answer=expected_answer,
                answer_type=answer_type,
                context_text=context_text,
                extracted_answer=(str(extracted_answer) if extracted_answer is not None else None),
                gold_in_top10=gold_in_top10,
                gold_in_top3=gold_in_top3,
            )
            answer_correct, answer_status = score_answer_correctness(
                expected_answer=expected_answer,
                answer_type=answer_type,
                extracted_answer=(str(extracted_answer) if extracted_answer is not None else None),
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
                    "include_generated_answer": bool(args.include_generated_answer),
                    "generation_status": str(out.get("generation_status") or ""),
                    "generation_confidence": out.get("generation_confidence"),
                    "generated_answer": (str(out.get("generated_answer") or "") if out.get("generated_answer") is not None else ""),
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
    heatmap_png = out_dir / "current_pipeline_fp1_fp7_heatmap.png"

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
            "include_generated_answer": bool(args.include_generated_answer),
            "generation_overrides": generation_overrides,
            "local_llm_enabled": bool(service.local_llm.enabled),
            "local_llm_base_url": str(service.local_llm.base_url),
            "local_llm_model": str(service.local_llm.model),
        },
        "artifacts": {
            "per_query_csv": str(per_query_csv),
            "counts_csv": str(counts_csv),
            "heatmap_png": str(heatmap_png),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Reuse existing plotting surface for the heatmap.
    import subprocess

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "plot_fp_failure_heatmap.py"),
        "--counts-csv",
        str(counts_csv),
        "--output",
        str(heatmap_png),
        "--queries-per-series",
        "50",
    ]
    heatmap_status = "ok"
    try:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    except Exception as exc:
        heatmap_status = f"plot_failed: {exc}"

    summary["artifacts"]["heatmap_status"] = heatmap_status
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
