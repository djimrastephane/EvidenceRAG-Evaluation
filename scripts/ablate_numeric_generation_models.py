from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rag_pdf.services.search_service import SearchService


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablate local LLM model choice on numeric generation queries.")
    p.add_argument(
        "--input-csv",
        default="results/live_fp1_fp7_compare_llm_vs_retrieval/fp6_to_fp4_audit.csv",
        help="CSV of numeric problem queries to replay.",
    )
    p.add_argument(
        "--models",
        default="qwen2.5:7b-instruct,mistral:latest,llama3:latest",
        help="Comma-separated Ollama model names.",
    )
    p.add_argument("--model-path", default="models/all-MiniLM-L6-v2")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--gen-timeout-seconds", type=float, default=30.0)
    p.add_argument(
        "--out-csv",
        default="results/llm_numeric_model_ablation/fp6_to_fp4_model_ablation.csv",
    )
    return p.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    input_csv = (REPO_ROOT / args.input_csv).resolve()
    out_csv = (REPO_ROOT / args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = load_rows(input_csv)
    model_names = [m.strip() for m in str(args.models).split(",") if m.strip()]
    model_path = (REPO_ROOT / args.model_path).resolve()

    output_rows: list[dict[str, Any]] = []
    for model_name in model_names:
        os.environ["LOCAL_LLM_ENABLED"] = "1"
        os.environ["LOCAL_LLM_BASE_URL"] = "http://127.0.0.1:11434"
        os.environ["LOCAL_LLM_MODEL"] = model_name
        svc = SearchService(repo_root=REPO_ROOT, model_path=model_path)

        for row in rows:
            doc_id = str(row.get("document") or "").strip()
            question = str(row.get("question") or "").strip()
            query_id = str(row.get("query_id") or "").strip()
            if not doc_id or not question:
                continue
            out = svc.search(
                data_dir=(REPO_ROOT / "data_processed" / doc_id).resolve(),
                question=question,
                k=int(args.k),
                query_id=query_id or None,
                include_generated_answer=True,
                generation_overrides={"timeout_seconds": float(args.gen_timeout_seconds)},
            )
            gdbg = out.get("generation_debug") or {}
            output_rows.append(
                {
                    "model": model_name,
                    "document": doc_id,
                    "query_id": query_id,
                    "question": question,
                    "expected_answer": str(row.get("expected_answer") or ""),
                    "expected_pages": str(row.get("expected_pages") or ""),
                    "answer_type": str(out.get("answer_type") or row.get("answer_type") or ""),
                    "predicted_answer": str(out.get("predicted_answer") or ""),
                    "predicted_answer_raw": str(out.get("predicted_answer_raw") or ""),
                    "generated_answer": str(out.get("generated_answer") or ""),
                    "generated_answer_raw": str(out.get("generated_answer_raw") or ""),
                    "generation_status": str(out.get("generation_status") or ""),
                    "generation_confidence": out.get("generation_confidence"),
                    "prompt_mode": str(gdbg.get("prompt_mode") or ""),
                    "citations_valid": int(gdbg.get("citations_valid") or 0),
                    "citations_rejected": int(gdbg.get("citations_rejected") or 0),
                    "parse_mode": str(gdbg.get("parse_mode") or ""),
                    "top1_chunk_id": str((out.get("results") or [{}])[0].get("chunk_id") or "") if out.get("results") else "",
                    "top1_pages": json.dumps((out.get("results") or [{}])[0].get("pages") or []) if out.get("results") else "[]",
                }
            )

    pd.DataFrame(output_rows).to_csv(out_csv, index=False)
    print(json.dumps({"out_csv": str(out_csv), "rows": len(output_rows), "models": model_names}, indent=2))


if __name__ == "__main__":
    main()
