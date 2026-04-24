"""run_ragas_context_eval.py

Runs RAGAS context_precision and context_recall on the post-fix 224/56
pipeline outputs using Ollama (qwen2.5:7b-instruct) as the LLM judge.

No generated answers required — only:
  - question
  - retrieved contexts (top-k chunk texts from hybrid_page_hits)
  - ground_truths (expected_answer from eval_set)

Usage:
    python scripts/run_ragas_context_eval.py [--top-k 5] [--model qwen2.5:7b-instruct]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH  = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from ragas.run_config import RunConfig
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from ragas.llms   import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from thesis_rag.artifacts import load_queries

ARTIFACT_ROOT = REPO_ROOT / "results/thesis_ablations/post_fix_rerun_2026-04-19/pipeline_outputs"
EVAL_ROOT     = REPO_ROOT / "data_processed"
DOCS = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--top-k",  type=int, default=5,
                   help="Number of retrieved chunks per query (default 5)")
    p.add_argument("--model",  default="mistral:latest",
                   help="Ollama model to use as RAGAS judge (mistral:latest recommended; qwen2.5 fails context_precision)")
    p.add_argument("--base-url", default="http://127.0.0.1:11434",
                   help="Ollama base URL")
    p.add_argument("--doc", default=None,
                   help="Run on a single doc_id only (e.g. Grampian-2023-2024)")
    return p.parse_args()


def build_ragas_rows(doc_id: str, top_k: int) -> list[dict]:
    exp_dir = ARTIFACT_ROOT / f"minilmcap_{doc_id}_chunk_224_56" / doc_id
    queries = load_queries(EVAL_ROOT / doc_id / "eval_set.json")
    hits_df = pd.read_csv(exp_dir / "hybrid_page_hits.csv")

    # Group chunk texts by query_id, preserving rank order
    grouped: dict[str, list[str]] = defaultdict(list)
    for _, row in hits_df.sort_values("rank").iterrows():
        qid  = str(row["query_id"])
        text = str(row.get("text", "")).strip()
        if text:
            grouped[qid].append(text)

    rows = []
    for q in queries:
        contexts = grouped.get(q.query_id, [])[:top_k]
        if not contexts:
            continue
        rows.append({
            "question":      q.query_text,
            "contexts":      contexts,
            "ground_truths": [str(q.expected_answer or "")],
            "doc_id":        doc_id,
            "query_id":      q.query_id,
        })
    return rows


def run_ragas(rows: list[dict], model: str, base_url: str) -> dict:
    llm = LangchainLLMWrapper(
        ChatOllama(model=model, base_url=base_url, temperature=0.0,
                   timeout=120, num_ctx=4096)
    )
    emb = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=model, base_url=base_url)
    )

    context_precision.llm = llm
    context_recall.llm    = llm
    context_precision.embeddings = emb
    context_recall.embeddings    = emb

    dataset = Dataset.from_dict({
        "user_input":          [r["question"]         for r in rows],
        "retrieved_contexts":  [r["contexts"]         for r in rows],
        "reference":           [r["ground_truths"][0] for r in rows],
    })

    run_cfg = RunConfig(timeout=180, max_workers=4, max_retries=3)
    result = evaluate(dataset, metrics=[context_precision, context_recall],
                      run_config=run_cfg, raise_exceptions=False)
    return result


def main() -> None:
    args = parse_args()
    docs = [args.doc] if args.doc else DOCS

    print(f"RAGAS evaluation — top_k={args.top_k}  model={args.model}")
    print(f"Metrics: context_precision, context_recall")
    print("=" * 65)

    all_rows: list[dict] = []
    per_doc_rows: dict[str, list[dict]] = {}

    for doc_id in docs:
        rows = build_ragas_rows(doc_id, args.top_k)
        per_doc_rows[doc_id] = rows
        all_rows.extend(rows)
        print(f"  {doc_id}: {len(rows)} queries loaded")

    # Per-document results
    print()
    doc_results = {}
    for doc_id in docs:
        print(f"  Evaluating {doc_id}...", flush=True)
        rows = per_doc_rows[doc_id]
        result = run_ragas(rows, args.model, args.base_url)
        def _valid(s) -> bool:
            return s is not None and not (isinstance(s, float) and math.isnan(s))

        scores = result["context_precision"]
        cp = sum(s for s in scores if _valid(s)) / max(sum(1 for s in scores if _valid(s)), 1)
        scores_cr = result["context_recall"]
        cr = sum(s for s in scores_cr if _valid(s)) / max(sum(1 for s in scores_cr if _valid(s)), 1)
        n_valid_cp = sum(1 for s in scores if _valid(s))
        n_valid_cr = sum(1 for s in scores_cr if _valid(s))
        doc_results[doc_id] = {"context_precision": cp, "context_recall": cr,
                                "n": len(rows), "n_valid_cp": n_valid_cp, "n_valid_cr": n_valid_cr}
        print(f"    context_precision={cp:.4f} ({n_valid_cp}/{len(rows)} valid)  context_recall={cr:.4f} ({n_valid_cr}/{len(rows)} valid)")

    # Summary table
    print()
    print("=" * 65)
    print(f"  {'Document':<22} {'Ctx Precision':>14} {'Ctx Recall':>12}  {'n':>4}")
    print("  " + "-" * 55)
    for doc_id in docs:
        r = doc_results[doc_id]
        yr = doc_id.replace("Grampian-", "")
        print(f"  {yr:<22} {r['context_precision']:>14.4f} {r['context_recall']:>12.4f}  {r['n']:>4}")
    avg_cp = sum(r["context_precision"] for r in doc_results.values()) / len(doc_results)
    avg_cr = sum(r["context_recall"]    for r in doc_results.values()) / len(doc_results)
    print("  " + "-" * 55)
    print(f"  {'Average':<22} {avg_cp:>14.4f} {avg_cr:>12.4f}")
    print("=" * 65)
    print(f"\n  Settings: top_k={args.top_k}  model={args.model}")
    print(f"  Pipeline: post_fix_rerun_2026-04-19  chunk=224/56")

    # Save results
    out_path = REPO_ROOT / "results" / "ragas_context_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "top_k": args.top_k,
        "model": args.model,
        "per_doc": doc_results,
        "avg_context_precision": avg_cp,
        "avg_context_recall":    avg_cr,
    }, indent=2))
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
