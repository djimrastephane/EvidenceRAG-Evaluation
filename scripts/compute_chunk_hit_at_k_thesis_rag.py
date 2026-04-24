"""compute_chunk_hit_at_k_thesis_rag.py

Computes true Chunk Hit@k (raw chunk level, before page deduplication) for
dense, BM25, and hybrid (boost-OFF and boost-ON) using the saved thesis_rag
224/56 pipeline artifacts.

Chunk Hit@k = at least one of the top-k *chunks* (raw, not deduped to pages)
has a page_number that matches expected_pages.

Output: JSON and Markdown summary tables saved to results/chunk_hit_at_k_2026-04-21/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import faiss

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from thesis_rag.artifacts import load_chunks, load_queries
from thesis_rag.config import load_config
from thesis_rag.embedding import embed_queries
from thesis_rag.retrieval_hybrid import hybrid_retrieve_legacy_style
from thesis_rag.retrieval_dense import search_faiss_stably
from thesis_rag.retrieval_sparse import build_bm25
from thesis_rag.utils import resolve_device, set_global_determinism

DOCS = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]
BOOST_OFF_ROOT = REPO_ROOT / "results/thesis_ablations/chunk_size_ablation_boost_off_2026-04-20/pipeline_outputs"
BOOST_ON_ROOT  = REPO_ROOT / "results/thesis_ablations/chunk_size_ablation_2026-04-15/pipeline_outputs"
PIPELINE_CONFIG = REPO_ROOT / "configs" / "thesis_rag.yaml"
OUT_DIR = REPO_ROOT / "results" / "chunk_hit_at_k_2026-04-21"
KS = [1, 3, 5, 10]


def _load_config():
    cfg = load_config(PIPELINE_CONFIG)
    cfg.embedding.model_name = str(REPO_ROOT / "models" / "all-MiniLM-L6-v2")
    cfg.retrieval.dense_top_k = 10
    cfg.retrieval.sparse_top_k = 10
    cfg.retrieval.hybrid_top_k = 10
    cfg.retrieval.rrf_k = 20
    cfg.retrieval.dense_weight = 0.5
    cfg.retrieval.sparse_weight = 2.0
    return cfg


def compute_chunk_hits(artifact_root: Path, enable_boost: bool) -> dict:
    cfg = _load_config()
    set_global_determinism(cfg.runtime.random_seed, cfg.runtime.deterministic_torch)
    device = resolve_device(cfg.runtime.device)

    per_doc: dict[str, dict] = {}
    all_dense = {k: [] for k in KS}
    all_bm25  = {k: [] for k in KS}
    all_hybrid = {k: [] for k in KS}

    for doc_id in DOCS:
        art = artifact_root / f"minilmcap_{doc_id}_chunk_224_56" / doc_id
        chunks  = load_chunks(art / "chunk_metadata.parquet")
        queries = load_queries(REPO_ROOT / "data_processed" / doc_id / "eval_set.json")
        index   = faiss.read_index(str(art / "faiss.index"))

        gold: dict[str, set[int]] = {q.query_id: set(q.gold_pages) for q in queries}

        query_vectors = embed_queries(
            [q.query_text for q in queries],
            cfg.embedding,
            device=device,
            cache_dir=str(cfg.paths.model_cache_dir),
        )
        bm25 = build_bm25(chunks, cfg.bm25)
        max_k = max(100, cfg.retrieval.hybrid_top_k)
        dense_scores, dense_indices = search_faiss_stably(index, query_vectors, min(max_k, len(chunks)))

        dense_hits, bm25_hits, hybrid_hits = hybrid_retrieve_legacy_style(
            chunks=chunks,
            queries=queries,
            dense_scores=dense_scores,
            dense_indices=dense_indices,
            bm25=bm25,
            max_k_search=max_k,
            dense_weight=cfg.retrieval.dense_weight,
            bm25_weight=cfg.retrieval.sparse_weight,
            rrf_k=cfg.retrieval.rrf_k,
            enable_subsection_boost=enable_boost,
            subsection_boost=0.05,
        )

        def hits_to_chunk_hit_at_k(hits, ks):
            from collections import defaultdict
            per_q: dict[str, list] = defaultdict(list)
            for h in hits:
                per_q[h.query_id].append(h)
            results = {k: [] for k in ks}
            for qid, qhits in per_q.items():
                sorted_hits = sorted(qhits, key=lambda h: h.rank)
                gp = gold.get(qid, set())
                for k in ks:
                    top_k = sorted_hits[:k]
                    results[k].append(1 if any(h.page_number in gp for h in top_k) else 0)
            return {k: sum(v) / len(v) for k, v in results.items()}

        doc_dense  = hits_to_chunk_hit_at_k(
            [h for h in dense_hits if h.query_id in gold], KS
        )
        doc_bm25   = hits_to_chunk_hit_at_k(
            [h for h in bm25_hits if h.query_id in gold], KS
        )
        doc_hybrid = hits_to_chunk_hit_at_k(
            [h for h in hybrid_hits if h.query_id in gold], KS
        )

        per_doc[doc_id] = {
            "dense": doc_dense,
            "bm25": doc_bm25,
            "hybrid": doc_hybrid,
        }
        for k in KS:
            all_dense[k].append(doc_dense[k])
            all_bm25[k].append(doc_bm25[k])
            all_hybrid[k].append(doc_hybrid[k])

        print(f"  {doc_id}: dense@1={doc_dense[1]:.4f} bm25@1={doc_bm25[1]:.4f} hybrid@1={doc_hybrid[1]:.4f}")

    aggregate = {
        "dense":  {k: sum(all_dense[k]) / len(all_dense[k]) for k in KS},
        "bm25":   {k: sum(all_bm25[k]) / len(all_bm25[k]) for k in KS},
        "hybrid": {k: sum(all_hybrid[k]) / len(all_hybrid[k]) for k in KS},
    }
    return {"per_doc": per_doc, "aggregate": aggregate}


def print_table(label: str, agg: dict) -> None:
    print(f"\n{label}")
    print(f"{'Method':<10} " + " ".join(f"Hit@{k}" for k in KS))
    for method in ["dense", "bm25", "hybrid"]:
        vals = " ".join(f"{agg[method][k]:.4f}" for k in KS)
        print(f"{method:<10} {vals}")


def write_outputs(boost_off: dict, boost_on: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "boost_off": boost_off,
        "boost_on": boost_on,
        "ks": KS,
        "description": (
            "Chunk Hit@k: fraction of queries where at least one of the top-k "
            "retrieved chunks (raw, before page deduplication) has a page_number "
            "matching expected_pages."
        ),
    }
    (OUT_DIR / "chunk_hit_at_k.json").write_text(json.dumps(data, indent=2))

    lines = [
        "# Chunk Hit@k (raw chunk level, before page deduplication)",
        "",
        "## Boost-OFF (primary pipeline)",
        "",
        "| Method | Chunk Hit@1 | Chunk Hit@3 | Chunk Hit@5 | Chunk Hit@10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, label in [("dense", "Dense (MiniLM)"), ("bm25", "BM25-only"), ("hybrid", "Hybrid (boost OFF)")]:
        row = boost_off["aggregate"][method]
        lines.append(f"| {label} | {row[1]:.4f} | {row[3]:.4f} | {row[5]:.4f} | {row[10]:.4f} |")

    lines += [
        "",
        "## Boost-ON",
        "",
        "| Method | Chunk Hit@1 | Chunk Hit@3 | Chunk Hit@5 | Chunk Hit@10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, label in [("dense", "Dense (MiniLM)"), ("bm25", "BM25-only"), ("hybrid", "Hybrid (boost ON)")]:
        row = boost_on["aggregate"][method]
        lines.append(f"| {label} | {row[1]:.4f} | {row[3]:.4f} | {row[5]:.4f} | {row[10]:.4f} |")

    lines += ["", "## Per-document Hybrid Chunk Hit@k", "", "| Document | Boost | Hit@1 | Hit@3 | Hit@5 | Hit@10 |", "|---|---|---:|---:|---:|---:|"]
    for doc_id in DOCS:
        for run, run_data in [("OFF", boost_off), ("ON", boost_on)]:
            row = run_data["per_doc"][doc_id]["hybrid"]
            lines.append(f"| {doc_id} | {run} | {row[1]:.4f} | {row[3]:.4f} | {row[5]:.4f} | {row[10]:.4f} |")

    (OUT_DIR / "chunk_hit_at_k.md").write_text("\n".join(lines) + "\n")
    print(f"\nSaved: {OUT_DIR / 'chunk_hit_at_k.json'}")
    print(f"Saved: {OUT_DIR / 'chunk_hit_at_k.md'}")


def main() -> None:
    print("Computing Chunk Hit@k — boost OFF:")
    boost_off = compute_chunk_hits(BOOST_OFF_ROOT, enable_boost=False)
    print_table("Boost-OFF aggregate:", boost_off["aggregate"])

    print("\nComputing Chunk Hit@k — boost ON:")
    boost_on = compute_chunk_hits(BOOST_ON_ROOT, enable_boost=True)
    print_table("Boost-ON aggregate:", boost_on["aggregate"])

    write_outputs(boost_off, boost_on)


if __name__ == "__main__":
    main()
