"""Run retrieval parity checks across a small batch of documents and queries.

Selects up to --max-docs documents and --queries-per-doc queries per document, then calls
check_retrieval_parity.parity_payload for each combination to confirm that SearchService
and the standalone evaluator return identical ranked results. Writes a JSON report with a
top-level pass/fail status and per-query payloads. Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from check_retrieval_parity import _load_eval_items, parity_payload
from corpus_guard import list_eval_ready_doc_dirs, print_skipped_eval_ready_docs
from rag_pdf.services.search_service import SearchService


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check evaluator/service retrieval parity across a small batch of queries.")
    p.add_argument("--data-root", default="data_processed", help="Root containing processed doc directories.")
    p.add_argument("--doc-pattern", default="Grampian-20*-20*", help="Glob used to select documents under data-root.")
    p.add_argument("--model-path", default="models/all-MiniLM-L6-v2")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--max-docs", type=int, default=3, help="Maximum number of documents to check.")
    p.add_argument("--queries-per-doc", type=int, default=3, help="Maximum number of queries to check per document.")
    p.add_argument(
        "--out-json",
        default="results/reproducibility/retrieval_parity_batch_smoke.json",
        help="JSON report path.",
    )
    p.add_argument(
        "--allow-incomplete-corpora",
        action="store_true",
        help="Include matching doc folders even if they are missing canonical evaluation artifacts.",
    )
    return p.parse_args()


def main() -> None:
    """Run batch parity checks and write a JSON report; exit non-zero if any check fails."""
    args = parse_args()
    data_root = (REPO_ROOT / args.data_root).resolve()
    model_path = (REPO_ROOT / args.model_path).resolve()

    if args.allow_incomplete_corpora:
        doc_dirs = sorted([p for p in data_root.glob(str(args.doc_pattern)) if p.is_dir()])
    else:
        doc_dirs, skipped = list_eval_ready_doc_dirs(data_root, str(args.doc_pattern))
        print_skipped_eval_ready_docs(skipped)
    doc_dirs = doc_dirs[: max(1, int(args.max_docs))]
    if not doc_dirs:
        raise FileNotFoundError(f"No matching document directories found under {data_root}")

    service = SearchService(repo_root=REPO_ROOT, model_path=model_path)
    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for doc_dir in doc_dirs:
        eval_items = _load_eval_items(doc_dir / "eval_set.json")
        for query_item in eval_items[: max(1, int(args.queries_per_doc))]:
            payload = parity_payload(
                repo_root=REPO_ROOT,
                data_dir=doc_dir,
                model_path=model_path,
                query_item=query_item,
                k=int(args.k),
                service=service,
            )
            checks.append(payload)
            if payload["status"] != "pass":
                failures.append(payload)

    report = {
        "status": "pass" if not failures else "fail",
        "data_root": str(data_root),
        "doc_pattern": str(args.doc_pattern),
        "model_path": str(model_path),
        "k": int(args.k),
        "max_docs": int(args.max_docs),
        "queries_per_doc": int(args.queries_per_doc),
        "num_checks": len(checks),
        "num_failures": len(failures),
        "checks": checks,
    }
    if str(args.out_json).strip():
        out_path = (REPO_ROOT / args.out_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
