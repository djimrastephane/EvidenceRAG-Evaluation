"""reproduce_figure_4_2_postfix.py

Regenerates Figure 4.2 from the post-fix 2026-04-19 run with three lines:
  - Dense (MiniLM)
  - Hybrid baseline  (subsection_boost=False)
  - Hybrid + subsection boost  (subsection_boost=True)

Dense and boosted-hybrid hits are loaded from the existing pipeline_outputs.
Baseline-hybrid hits are re-run on the fly using the saved indexes.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from random import Random

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import faiss

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from thesis_rag.artifacts import load_queries
from thesis_rag.evaluator import evaluate_page_hits
from thesis_rag.ranking import chunk_hits_to_page_hits
from thesis_rag.retrieval_hybrid import hybrid_retrieve_legacy_style
from thesis_rag.retrieval_sparse import build_bm25, sparse_retrieve_legacy_style
from thesis_rag.retrieval_dense import search_faiss_stably
from thesis_rag.schemas import ChunkRecord, RetrievalHit
from thesis_rag.utils import l2_normalize

ARTIFACT_ROOT = REPO_ROOT / "results" / "thesis_ablations" / "post_fix_rerun_2026-04-19" / "pipeline_outputs"
EVAL_ROOT     = REPO_ROOT / "data_processed"
OUT_DIR       = REPO_ROOT / "results" / "thesis_figures" / f"figure_4_2_postfix_{date.today().isoformat()}"
BOOTSTRAP_SAMPLES = 5000
MAX_RANK = 10
RRF_K, DENSE_W, BM25_W = 20, 0.5, 2.0
DOC_IDS = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QueryOutcome:
    system: str
    doc_id: str
    query_id: str
    gold_pages: list[int]
    first_correct_rank: float | None
    event: int
    time_rank: int
    censored: int


# ---------------------------------------------------------------------------
# Chunk loader (mirrors compare_subsection_boost.py)
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
# Load hits from saved CSV
# ---------------------------------------------------------------------------

def _hits_from_csv(path: Path) -> list[RetrievalHit]:
    frame = pd.read_csv(path)
    hits = []
    for row in frame.to_dict(orient="records"):
        pages_raw = row.get("pages", "")
        try:
            pages = json.loads(pages_raw) if isinstance(pages_raw, str) else list(pages_raw)
        except Exception:
            pages = [int(row.get("page_number", 0))]
        hits.append(RetrievalHit(
            query_id=str(row["query_id"]),
            query_text=str(row.get("query_text", "")),
            rank=int(row["rank"]),
            score=float(row.get("score", 0.0)),
            retrieval_method=str(row.get("retrieval_method", "")),
            doc_id=str(row["doc_id"]),
            page_number=int(row["page_number"]),
            chunk_id=str(row.get("chunk_id", "")),
            pages=pages,
            text=str(row.get("text", "")),
        ))
    return hits


# ---------------------------------------------------------------------------
# Build baseline-hybrid hits (boost=False) on the fly
# ---------------------------------------------------------------------------

def _build_baseline_hybrid_hits(model, apply_l2: bool, config) -> list[RetrievalHit]:
    all_hits: list[RetrievalHit] = []
    for doc_id in DOC_IDS:
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
        page_hits = chunk_hits_to_page_hits(hybrid_hits, "hybrid_pages", chunk_limit=MAX_RANK)
        all_hits.extend(page_hits)
    return all_hits


# ---------------------------------------------------------------------------
# Convert raw page hits to QueryOutcome list
# ---------------------------------------------------------------------------

def _outcomes_from_hits(system: str, page_hits: list[RetrievalHit]) -> list[QueryOutcome]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for h in page_hits:
        grouped[h.query_id].append(h.page_number)

    outcomes: list[QueryOutcome] = []
    for doc_id in DOC_IDS:
        queries = load_queries(EVAL_ROOT / doc_id / "eval_set.json")
        results = evaluate_page_hits(queries, [h for h in page_hits if h.doc_id == doc_id])
        for q, r in zip(queries, results):
            time_rank = int(r.first_relevant_rank) if r.first_relevant_rank else MAX_RANK
            censored  = 0 if r.first_relevant_rank else 1
            outcomes.append(QueryOutcome(
                system=system,
                doc_id=doc_id,
                query_id=q.query_id,
                gold_pages=list(q.gold_pages),
                first_correct_rank=float(r.first_relevant_rank) if r.first_relevant_rank else None,
                event=1 - censored,
                time_rank=time_rank,
                censored=censored,
            ))
    if len(outcomes) != 250:
        raise RuntimeError(f"Expected 250 outcomes for {system}, got {len(outcomes)}")
    return outcomes


def _outcomes_from_csv_hits(system: str, filename: str) -> list[QueryOutcome]:
    all_hits: list[RetrievalHit] = []
    for doc_id in DOC_IDS:
        exp_dir = ARTIFACT_ROOT / f"minilmcap_{doc_id}_chunk_224_56" / doc_id
        all_hits.extend(_hits_from_csv(exp_dir / filename))
    return _outcomes_from_hits(system, all_hits)


# ---------------------------------------------------------------------------
# KM survival curve with bootstrap CI
# ---------------------------------------------------------------------------

def _km_curve(outcomes: list[QueryOutcome], rng: Random) -> list[dict]:
    times  = [o.time_rank for o in outcomes]
    events = [o.event     for o in outcomes]
    rows   = []
    for rank in range(1, MAX_RANK + 1):
        surv_prob = sum(
            1 for t, e in zip(times, events) if e == 0 or t > rank
        ) / len(outcomes)
        boot = []
        for _ in range(BOOTSTRAP_SAMPLES):
            idx = [rng.randrange(len(outcomes)) for _ in range(len(outcomes))]
            s = sum(1 for i in idx if outcomes[i].censored == 1 or outcomes[i].time_rank > rank)
            boot.append(s / len(outcomes))
        boot.sort()
        rows.append({
            "rank": rank,
            "survival_probability": surv_prob,
            "ci_lower": boot[int(0.025 * BOOTSTRAP_SAMPLES)],
            "ci_upper": boot[int(0.975 * BOOTSTRAP_SAMPLES)],
        })
    return rows


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot(curves: dict[str, list[dict]]) -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "savefig.facecolor":"white",
        "font.family":      "DejaVu Sans",
        "font.size":        11,
        "axes.titlesize":   12.5,
        "axes.labelsize":   11.5,
        "xtick.labelsize":  10.5,
        "ytick.labelsize":  10.5,
    })

    styles = {
        "dense":          {"label": "Dense (MiniLM)",             "color": "#4C78A8", "ls": "-"},
        "hybrid_base":    {"label": "Hybrid (no subsection boost)","color": "#72B7B2", "ls": "--"},
        "hybrid_boosted": {"label": "Hybrid + subsection boost",  "color": "#F58518", "ls": "-"},
    }

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for key, rows in curves.items():
        s = styles[key]
        xs = [r["rank"] for r in rows]
        ys = [r["survival_probability"] for r in rows]
        lo = [r["ci_lower"]  for r in rows]
        hi = [r["ci_upper"]  for r in rows]
        ax.step(xs, ys, where="post", label=s["label"],
                color=s["color"], linewidth=2.2, linestyle=s["ls"])
        ax.fill_between(xs, lo, hi, step="post", color=s["color"], alpha=0.13)

    ax.set_xlim(1, MAX_RANK)
    ax.set_ylim(0.0, 0.30)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Survival probability")
    ax.set_title("Rank-based comparison: dense, hybrid baseline, and boosted-hybrid retrieval")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_4_2_postfix.png", dpi=320, bbox_inches="tight")
    fig.savefig(OUT_DIR / "figure_4_2_postfix.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Regenerate Figure 4.2 post-fix variant by re-running the hybrid baseline from raw chunks and saved dense hits."""
    from sentence_transformers import SentenceTransformer
    from thesis_rag.config import load_config
    from thesis_rag.utils import resolve_device

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config    = load_config(REPO_ROOT / "configs/thesis_rag.yaml")
    device    = resolve_device(config.runtime.device)
    apply_l2  = config.embedding.apply_l2_normalization

    print("Loading embedding model...", flush=True)
    model = SentenceTransformer(str(REPO_ROOT / "models/all-MiniLM-L6-v2"), device=device)

    print("Loading dense hits from saved CSV...", flush=True)
    dense_outcomes = _outcomes_from_csv_hits("dense", "dense_page_hits.csv")

    print("Loading boosted-hybrid hits from saved CSV...", flush=True)
    boosted_outcomes = _outcomes_from_csv_hits("hybrid_boosted", "hybrid_page_hits.csv")

    print("Running baseline hybrid (boost=False)...", flush=True)
    base_hits = _build_baseline_hybrid_hits(model, apply_l2, config)
    base_outcomes = _outcomes_from_hits("hybrid_base", base_hits)

    rng = Random(13)
    print("Computing KM curves with bootstrap CI...", flush=True)
    curves = {
        "dense":          _km_curve(dense_outcomes,   rng),
        "hybrid_base":    _km_curve(base_outcomes,    rng),
        "hybrid_boosted": _km_curve(boosted_outcomes, rng),
    }

    _plot(curves)

    summary = {
        "dense_hit@1":          round(1 - curves["dense"][0]["survival_probability"], 4),
        "hybrid_base_hit@1":    round(1 - curves["hybrid_base"][0]["survival_probability"], 4),
        "hybrid_boosted_hit@1": round(1 - curves["hybrid_boosted"][0]["survival_probability"], 4),
        "source_run": "post_fix_rerun_2026-04-19",
        "output_dir": str(OUT_DIR),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nFigure saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
