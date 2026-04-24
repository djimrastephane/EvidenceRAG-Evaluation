"""run_ragas_full_eval.py

Runs all four RAGAS metrics on the post-fix 224/56 pipeline with
subsection_boost=False, comparing LLM=OFF vs LLM=ON side-by-side.

  LLM=OFF panel:  context_precision  +  context_recall   (no generated answer)
  LLM=ON  panel:  all four metrics   +  faithfulness  +  answer_relevancy

Retrieval is re-run on the fly from saved indexes with subsection_boost=False.
Answers (LLM=ON) are generated via Ollama using a simple grounded prompt.

Usage:
    python scripts/run_ragas_full_eval.py [--top-k 5] [--judge mistral:latest]
                                          [--gen-model qwen2.5:7b-instruct]
                                          [--base-url http://127.0.0.1:11434]
                                          [--doc Grampian-2023-2024]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import faiss

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH  = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, answer_relevancy, faithfulness
from ragas.run_config import RunConfig
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from thesis_rag.artifacts import load_queries
from thesis_rag.ranking import chunk_hits_to_page_hits
from thesis_rag.retrieval_hybrid import hybrid_retrieve_legacy_style
from thesis_rag.retrieval_sparse import build_bm25
from thesis_rag.retrieval_dense import search_faiss_stably
from thesis_rag.schemas import ChunkRecord
from thesis_rag.utils import l2_normalize

ARTIFACT_ROOT = REPO_ROOT / "results/thesis_ablations/post_fix_rerun_2026-04-19/pipeline_outputs"
EVAL_ROOT     = REPO_ROOT / "data_processed"
DOCS = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]
RRF_K, DENSE_W, BM25_W = 20, 0.5, 2.0

METRIC_LABELS = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
DOC_LABELS    = [d.replace("Grampian-", "") for d in DOCS]


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--top-k",    type=int, default=5)
    p.add_argument("--judge",    default="mistral:latest",
                   help="Ollama model used as RAGAS judge")
    p.add_argument("--gen-model", default="qwen2.5:7b-instruct",
                   help="Ollama model used to generate answers (LLM=ON)")
    p.add_argument("--base-url", default="http://127.0.0.1:11434")
    p.add_argument("--doc",      default=None,
                   help="Restrict to a single doc_id for smoke-testing")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Chunk loader
# ---------------------------------------------------------------------------

def _load_chunks(exp_dir: Path) -> list[ChunkRecord]:
    df = pd.read_parquet(exp_dir / "chunks.parquet")
    chunks = []
    for row in df.to_dict(orient="records"):
        pages_raw = row.get("pages")
        pages = list(pages_raw) if isinstance(pages_raw, list) else [
            int(row.get("page_start") or row.get("page_number") or 0)
        ]
        chunks.append(ChunkRecord(
            chunk_id=str(row["chunk_id"]),
            doc_id=str(row["doc_id"]),
            page_number=int(row.get("page_number") or row.get("page_start") or 0),
            chunk_index=int(row.get("chunk_index", 0)),
            text=str(row.get("text", "")),
            token_count=int(row.get("token_count", 0)),
            word_count=int(row.get("word_count", 0)),
            chunk_id_global=str(row.get("chunk_id_global", "")),
            page_start=int(row.get("page_start") or row.get("page_number") or 0),
            page_end=int(row.get("page_end") or row.get("page_number") or 0),
            pages=pages,
            part=str(row.get("part") or ""),
            section_title=str(row.get("section_title") or ""),
            subsection_title=str(row.get("subsection_title") or ""),
            is_table=bool(row.get("is_table", False)),
            table_type=str(row.get("table_type")) if row.get("table_type") else None,
            table_chunk_kind=str(row.get("table_chunk_kind")) if row.get("table_chunk_kind") else None,
            segment_boundary_type=str(row.get("segment_boundary_type")) if row.get("segment_boundary_type") else None,
            segment_has_search_hit=bool(row.get("segment_has_search_hit", False)),
        ))
    return chunks


# ---------------------------------------------------------------------------
# Retrieval (subsection_boost=False)
# ---------------------------------------------------------------------------

def retrieve_boost_off(doc_id: str, model, apply_l2: bool, config, top_k: int) -> list[dict]:
    """Return list of {query_id, question, contexts, ground_truth} dicts."""
    exp_dir = ARTIFACT_ROOT / f"minilmcap_{doc_id}_chunk_224_56" / doc_id
    chunks  = _load_chunks(exp_dir)
    queries = load_queries(EVAL_ROOT / doc_id / "eval_set.json")
    index   = faiss.read_index(str(exp_dir / "faiss.index"))

    q_vecs = model.encode(
        [q.query_text for q in queries],
        batch_size=32, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=False,
    ).astype("float32")
    if apply_l2:
        q_vecs = l2_normalize(q_vecs)

    raw_scores, raw_indices = search_faiss_stably(index, q_vecs, min(100, len(chunks)))
    bm25 = build_bm25(chunks, config.bm25)

    _, _, hybrid_hits = hybrid_retrieve_legacy_style(
        chunks=chunks,
        queries=queries,
        dense_scores=raw_scores,
        dense_indices=raw_indices,
        bm25=bm25,
        max_k_search=100,
        dense_weight=DENSE_W,
        bm25_weight=BM25_W,
        rrf_k=RRF_K,
        enable_subsection_boost=False,
        enable_lexical_rerank=True,
    )

    grouped: dict[str, list[str]] = defaultdict(list)
    for h in sorted(hybrid_hits, key=lambda x: x.rank):
        text = str(getattr(h, "text", "") or "").strip()
        if text:
            grouped[str(h.query_id)].append(text)

    rows = []
    for q in queries:
        contexts = grouped.get(str(q.query_id), [])[:top_k]
        if not contexts:
            continue
        rows.append({
            "query_id":     str(q.query_id),
            "question":     q.query_text,
            "contexts":     contexts,
            "ground_truth": str(q.expected_answer or ""),
            "doc_id":       doc_id,
        })
    return rows


# ---------------------------------------------------------------------------
# Answer generation (LLM=ON)
# ---------------------------------------------------------------------------

_ANSWER_PROMPT_TMPL = """\
You are a financial analyst assistant. Answer the question using ONLY the context below.
Be concise — 1-3 sentences. Do not add information not present in the context.

Context:
{context}

Question: {question}

Answer:"""


def generate_answer(question: str, contexts: list[str], base_url: str, model: str,
                    timeout: int = 60) -> str:
    context_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    prompt = _ANSWER_PROMPT_TMPL.format(context=context_text, question=question)
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 200},
    }).encode()
    req = urllib.request.Request(
        url=f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read()).get("response", "").strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# RAGAS helpers
# ---------------------------------------------------------------------------

def _valid(s) -> bool:
    return s is not None and not (isinstance(s, float) and math.isnan(s))


def _mean(scores) -> float:
    valid = [s for s in scores if _valid(s)]
    return sum(valid) / len(valid) if valid else float("nan")


def _n_valid(scores) -> int:
    return sum(1 for s in scores if _valid(s))


def setup_ragas(judge_model: str, base_url: str):
    llm = LangchainLLMWrapper(
        ChatOllama(model=judge_model, base_url=base_url, temperature=0.0,
                   timeout=180, num_ctx=4096)
    )
    emb = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=judge_model, base_url=base_url)
    )
    for metric in [context_precision, context_recall, faithfulness, answer_relevancy]:
        metric.llm = llm
        metric.embeddings = emb


def run_ragas_off(rows: list[dict]) -> dict[str, float]:
    """Context-only metrics (LLM=OFF — no generated answer needed)."""
    dataset = Dataset.from_dict({
        "user_input":         [r["question"]     for r in rows],
        "retrieved_contexts": [r["contexts"]     for r in rows],
        "reference":          [r["ground_truth"] for r in rows],
    })
    run_cfg = RunConfig(timeout=180, max_workers=4, max_retries=3)
    result = evaluate(dataset, metrics=[context_precision, context_recall],
                      run_config=run_cfg, raise_exceptions=False)
    return {
        "context_precision": _mean(result["context_precision"]),
        "context_recall":    _mean(result["context_recall"]),
        "faithfulness":      float("nan"),
        "answer_relevancy":  float("nan"),
        "n_cp": _n_valid(result["context_precision"]),
        "n_cr": _n_valid(result["context_recall"]),
        "n":    len(rows),
    }


def run_ragas_on(rows: list[dict]) -> dict[str, float]:
    """All four metrics (LLM=ON — rows must have 'response' key)."""
    dataset = Dataset.from_dict({
        "user_input":         [r["question"]     for r in rows],
        "retrieved_contexts": [r["contexts"]     for r in rows],
        "reference":          [r["ground_truth"] for r in rows],
        "response":           [r["response"]     for r in rows],
    })
    run_cfg = RunConfig(timeout=180, max_workers=4, max_retries=3)
    result = evaluate(dataset,
                      metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
                      run_config=run_cfg, raise_exceptions=False)
    return {
        "context_precision": _mean(result["context_precision"]),
        "context_recall":    _mean(result["context_recall"]),
        "faithfulness":      _mean(result["faithfulness"]),
        "answer_relevancy":  _mean(result["answer_relevancy"]),
        "n_cp":  _n_valid(result["context_precision"]),
        "n_cr":  _n_valid(result["context_recall"]),
        "n_fai": _n_valid(result["faithfulness"]),
        "n_ar":  _n_valid(result["answer_relevancy"]),
        "n":     len(rows),
    }


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------

def plot_bar_charts(off_scores: dict, on_scores: dict, docs: list[str], out_path: Path) -> None:
    doc_labels = [d.replace("Grampian-", "") for d in docs]
    x = np.arange(len(docs))
    width = 0.13

    # Colours per metric
    COLOURS = {
        "context_precision": "#4C78A8",
        "context_recall":    "#72B7B2",
        "faithfulness":      "#F58518",
        "answer_relevancy":  "#E45756",
    }
    DISPLAY = {
        "context_precision": "Ctx Precision",
        "context_recall":    "Ctx Recall",
        "faithfulness":      "Faithfulness",
        "answer_relevancy":  "Ans Relevancy",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    panels = [
        (axes[0], off_scores, "LLM = OFF  (context metrics only)",
         ["context_precision", "context_recall"]),
        (axes[1], on_scores,  "LLM = ON  (all 4 RAGAS metrics)",
         ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]),
    ]

    for ax, scores, title, metrics in panels:
        n_metrics = len(metrics)
        offsets = np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * width

        for offset, metric in zip(offsets, metrics):
            vals = [scores.get(doc_id, {}).get(metric, np.nan) for doc_id in docs]
            bars = ax.bar(x + offset, vals, width=width * 0.9,
                          label=DISPLAY[metric], color=COLOURS[metric],
                          edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.005,
                            f"{val:.2f}", ha="center", va="bottom",
                            fontsize=7, rotation=90, color="#333333")

        ax.set_xticks(x)
        ax.set_xticklabels(doc_labels, fontsize=10)
        ax.set_ylim(0.0, 1.12)
        ax.set_ylabel("Score (0–1)", fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.legend(fontsize=9, frameon=False, loc="lower right")
        ax.grid(axis="y", color="#E0E0E0", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "RAGAS evaluation — subsection_boost=False  |  chunk=224/56  |  top_k=5\n"
        "Post-fix run: 2026-04-19   Judge: mistral:latest   Gen model: qwen2.5:7b-instruct",
        fontsize=10,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Bar chart saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    docs = [args.doc] if args.doc else DOCS

    from sentence_transformers import SentenceTransformer
    from thesis_rag.config import load_config
    from thesis_rag.utils import resolve_device

    config   = load_config(REPO_ROOT / "configs/thesis_rag.yaml")
    device   = resolve_device(config.runtime.device)
    apply_l2 = config.embedding.apply_l2_normalization

    print("Loading embedding model...", flush=True)
    model = SentenceTransformer(str(REPO_ROOT / "models/all-MiniLM-L6-v2"), device=device)

    print(f"Setting up RAGAS judge: {args.judge}", flush=True)
    setup_ragas(args.judge, args.base_url)

    # ------------------------------------------------------------------
    # Step 1: retrieve (boost=False) + generate answers
    # ------------------------------------------------------------------
    all_rows: dict[str, list[dict]] = {}
    for doc_id in docs:
        print(f"  Retrieving {doc_id} (boost=False)...", flush=True)
        rows = retrieve_boost_off(doc_id, model, apply_l2, config, args.top_k)
        print(f"    {len(rows)} queries — generating answers (LLM=ON)...", flush=True)
        for i, row in enumerate(rows):
            row["response"] = generate_answer(
                row["question"], row["contexts"],
                args.base_url, args.gen_model,
            )
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(rows)} answers generated", flush=True)
        all_rows[doc_id] = rows

    # ------------------------------------------------------------------
    # Step 2: RAGAS — LLM=OFF (context metrics)
    # ------------------------------------------------------------------
    print("\n--- RAGAS LLM=OFF ---", flush=True)
    off_scores: dict[str, dict] = {}
    for doc_id in docs:
        print(f"  {doc_id}...", flush=True)
        off_scores[doc_id] = run_ragas_off(all_rows[doc_id])
        s = off_scores[doc_id]
        print(f"    cp={s['context_precision']:.4f} ({s['n_cp']}/{s['n']})  "
              f"cr={s['context_recall']:.4f} ({s['n_cr']}/{s['n']})", flush=True)

    # ------------------------------------------------------------------
    # Step 3: RAGAS — LLM=ON (all 4 metrics)
    # ------------------------------------------------------------------
    print("\n--- RAGAS LLM=ON ---", flush=True)
    on_scores: dict[str, dict] = {}
    for doc_id in docs:
        print(f"  {doc_id}...", flush=True)
        on_scores[doc_id] = run_ragas_on(all_rows[doc_id])
        s = on_scores[doc_id]
        print(f"    cp={s['context_precision']:.4f}  cr={s['context_recall']:.4f}  "
              f"fai={s['faithfulness']:.4f}  ar={s['answer_relevancy']:.4f}", flush=True)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print(f"  {'Doc':<12}  {'OFF cp':>8} {'OFF cr':>8}  {'ON cp':>8} {'ON cr':>8} {'ON fai':>8} {'ON ar':>8}")
    print("  " + "-" * 74)
    for doc_id in docs:
        yr  = doc_id.replace("Grampian-", "")
        off = off_scores[doc_id]
        on  = on_scores[doc_id]
        print(f"  {yr:<12}  {off['context_precision']:>8.4f} {off['context_recall']:>8.4f}  "
              f"{on['context_precision']:>8.4f} {on['context_recall']:>8.4f} "
              f"{on['faithfulness']:>8.4f} {on['answer_relevancy']:>8.4f}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Heatmaps
    # ------------------------------------------------------------------
    tag = f"ragas_full_eval_boost_off_{date.today().isoformat()}"
    out_dir = REPO_ROOT / "results" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_bar_charts(off_scores, on_scores, docs, out_dir / "ragas_llm_off_vs_on.png")
    plot_bar_charts(off_scores, on_scores, docs, out_dir / "ragas_llm_off_vs_on.pdf")

    # Save raw scores
    (out_dir / "scores.json").write_text(json.dumps({
        "settings": {
            "top_k": args.top_k, "judge": args.judge,
            "gen_model": args.gen_model, "subsection_boost": False,
        },
        "llm_off": off_scores,
        "llm_on":  on_scores,
    }, indent=2))

    print(f"\n  Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
