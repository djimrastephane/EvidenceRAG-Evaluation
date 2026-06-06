"""
Fine-tune sentence-transformers/all-MiniLM-L6-v2 on NHS Grampian
query-chunk pairs using MultipleNegativesRankingLoss.

Train split : Grampian-2020-2021 → 2023-2024  (200 pairs)
Val   split : Grampian-2024-2025               ( 50 pairs)

Usage
-----
    python scripts/finetune_biencoder.py [--epochs 5] [--batch-size 16]

Output
------
    models/miniLM-finetuned/          (fine-tuned model ready for indexing)
    results/finetune_biencoder_<date>.json   (before/after metrics)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_BASE = ROOT / "data_processed"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_MODEL = ROOT / "models" / "miniLM-finetuned"

# doc split — keep 2024-2025 strictly out of training
TRAIN_DOC_IDS = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
]
VAL_DOC_ID = "Grampian-2024-2025"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_queries(doc_id: str) -> list[dict]:
    path = DATA_BASE / doc_id / "eval_set.json"
    with open(path) as f:
        d = json.load(f)
    return d["queries"] if isinstance(d, dict) else d


def load_chunks(doc_id: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_BASE / doc_id / "chunks.parquet")


def best_positive_chunk(query: dict, chunks_df: pd.DataFrame) -> str | None:
    """Return the text of the longest chunk whose page_start is in expected_pages."""
    expected = set(query["expected_pages"])
    matches = chunks_df[chunks_df["page_start"].isin(expected)]
    if matches.empty:
        return None
    # prefer the chunk with the most tokens
    return matches.loc[matches["chunk_tokens"].idxmax(), "chunk_text"]


def hard_negative_from_retrieval(query: dict, chunks_df: pd.DataFrame) -> str | None:
    """
    Mine a hard negative from the existing retrieval results.
    Takes the top-ranked retrieved chunk that is NOT on an expected page.
    Falls back to a random non-matching chunk if no retrieval file exists.
    """
    doc_id = query["doc_id"]
    retrieval_path = DATA_BASE / doc_id / "retrieval_results_hybrid.json"
    expected = set(query["expected_pages"])

    if retrieval_path.exists():
        with open(retrieval_path) as f:
            results = json.load(f)["results"]
        # find this query's result
        qid = query["query_id"]
        for r in results:
            if r["query_id"] != qid:
                continue
            # per_k["20"] has the full top-20 ranked list
            pk = r.get("per_k", {})
            ranked_ids = pk.get("20", pk.get("10", {})).get("retrieved_chunk_ids", [])
            flags = pk.get("20", pk.get("10", {})).get("chunk_hit_flags", [])
            for chunk_id, flag in zip(ranked_ids, flags):
                if flag == 0:  # non-relevant → hard negative
                    row = chunks_df[chunks_df["chunk_id_global"] == chunk_id]
                    if not row.empty:
                        return row.iloc[0]["chunk_text"]

    # fallback: random non-matching chunk from same doc
    non_match = chunks_df[~chunks_df["page_start"].isin(expected)]
    if non_match.empty:
        return None
    return non_match.sample(1, random_state=42).iloc[0]["chunk_text"]


def build_pairs(doc_ids: list[str]) -> list[dict]:
    """Return list of {anchor, positive, negative} dicts."""
    pairs = []
    for doc_id in doc_ids:
        queries = load_queries(doc_id)
        chunks_df = load_chunks(doc_id)
        for q in queries:
            positive = best_positive_chunk(q, chunks_df)
            if positive is None:
                log.warning("No positive chunk for %s — skipping", q["query_id"])
                continue
            negative = hard_negative_from_retrieval(q, chunks_df)
            row = {"anchor": q["question"], "positive": positive}
            if negative:
                row["negative"] = negative
            pairs.append(row)
    return pairs


# ---------------------------------------------------------------------------
# Evaluator helpers  (InformationRetrievalEvaluator format)
# ---------------------------------------------------------------------------

def build_ir_evaluator(doc_id: str):
    """Build an InformationRetrievalEvaluator for the given doc."""
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    queries_raw = load_queries(doc_id)
    chunks_df = load_chunks(doc_id)

    queries: dict[str, str] = {}
    relevant_docs: dict[str, set[str]] = {}
    corpus: dict[str, str] = {}

    # corpus = all chunks from this doc
    for _, row in chunks_df.iterrows():
        corpus[row["chunk_id_global"]] = row["chunk_text"]

    for q in queries_raw:
        qid = q["query_id"]
        queries[qid] = q["question"]
        expected = set(q["expected_pages"])
        relevant_ids = set(
            chunks_df.loc[chunks_df["page_start"].isin(expected), "chunk_id_global"]
        )
        if relevant_ids:
            relevant_docs[qid] = relevant_ids

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=doc_id,
        show_progress_bar=False,
        mrr_at_k=[1, 3, 5, 10],
        ndcg_at_k=[1, 3, 5, 10],
        accuracy_at_k=[1, 3, 5, 10],
        precision_recall_at_k=[1, 3, 5, 10],
        map_at_k=[10],
    )


# ---------------------------------------------------------------------------
# Quick metric extraction (run evaluator and return dict)
# ---------------------------------------------------------------------------

def evaluate(model, doc_id: str) -> dict:
    evaluator = build_ir_evaluator(doc_id)
    scores = evaluator(model)
    # scores is a dict keyed like "InformationRetrievalEvaluator_<name>_accuracy@1" etc.
    out = {}
    for key, val in scores.items():
        short = key.split("_", 2)[-1] if "_" in key else key
        out[short] = round(float(val), 4)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_MODEL)
    return p.parse_args()


def main():
    args = parse_args()

    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
    from sentence_transformers import SentenceTransformerTrainingArguments
    from sentence_transformers.losses import MultipleNegativesRankingLoss

    # ---- build training data ----
    log.info("Building training pairs from %s", TRAIN_DOC_IDS)
    train_pairs = build_pairs(TRAIN_DOC_IDS)
    log.info("Training pairs: %d", len(train_pairs))

    has_negatives = all("negative" in p for p in train_pairs)
    log.info("Hard negatives available: %s", has_negatives)

    train_dataset = Dataset.from_list(train_pairs)

    # ---- load model ----
    log.info("Loading base model: %s", args.base_model)
    model = SentenceTransformer(args.base_model)

    # ---- baseline eval before training ----
    log.info("Running baseline eval on %s …", VAL_DOC_ID)
    baseline_scores = evaluate(model, VAL_DOC_ID)
    log.info("Baseline: %s", {k: v for k, v in baseline_scores.items() if "accuracy" in k or "mrr" in k})

    # ---- loss ----
    loss = MultipleNegativesRankingLoss(model)

    # ---- training arguments ----
    total_steps = (len(train_pairs) // args.batch_size) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        fp16=False,          # CPU-safe
        bf16=False,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=f"{VAL_DOC_ID}_cosine_accuracy@1",
        greater_is_better=True,
        logging_steps=5,
        seed=13,
    )

    evaluator = build_ir_evaluator(VAL_DOC_ID)

    # ---- train ----
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    log.info("Starting training: %d epochs, batch=%d, lr=%g", args.epochs, args.batch_size, args.lr)
    trainer.train()

    # ---- save final model ----
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output_dir))
    log.info("Model saved to %s", args.output_dir)

    # ---- final eval ----
    log.info("Running final eval on %s …", VAL_DOC_ID)
    final_scores = evaluate(model, VAL_DOC_ID)

    # ---- summarise ----
    result = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "output_model": str(args.output_dir),
        "train_doc_ids": TRAIN_DOC_IDS,
        "val_doc_id": VAL_DOC_ID,
        "n_train_pairs": len(train_pairs),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "baseline": baseline_scores,
        "finetuned": final_scores,
        "delta": {k: round(final_scores.get(k, 0) - baseline_scores.get(k, 0), 4)
                  for k in final_scores},
    }

    stamp = datetime.now().strftime("%Y-%m-%d")
    out_path = RESULTS_DIR / f"finetune_biencoder_{stamp}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Results written to %s", out_path)

    # print summary table
    print("\n=== Fine-tuning Results ===")
    print(f"{'Metric':<30} {'Baseline':>10} {'Fine-tuned':>12} {'Delta':>8}")
    print("-" * 62)
    key_metrics = [k for k in final_scores if any(x in k for x in ["accuracy@1", "accuracy@3", "accuracy@5", "accuracy@10", "mrr@10"])]
    for k in sorted(key_metrics):
        b = baseline_scores.get(k, 0)
        ft = final_scores.get(k, 0)
        print(f"{k:<30} {b:>10.3f} {ft:>12.3f} {ft-b:>+8.3f}")

    return result


if __name__ == "__main__":
    main()
