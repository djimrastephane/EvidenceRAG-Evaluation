from __future__ import annotations

"""Reproduce Table 4.5 under the refactored ``thesis_rag`` pipeline.

This script compares document-constrained retrieval against global retrieval
for the saved 224/56 ``thesis_rag`` artifacts from the 5-document chunk-size
ablation bundle. By default it evaluates the full 250-query 5-document set,
which is the like-for-like scope for the updated thesis results. The
document-scoped side reuses the stored hybrid retrieval outputs for each report,
while the global side rebuilds retrieval over the concatenated 5-document chunk
and embedding artifacts so the comparison stays within a single
refactored-pipeline state.

Outputs are written to a dated thesis validation bundle containing:

- copied config and source references
- document-constrained metrics
- global-scope metrics
- leakage summaries
- a side-by-side CSV/Markdown comparison suitable for the thesis table update
"""

import argparse
import json
import shutil
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from thesis_rag.artifacts import load_chunks, load_queries
from thesis_rag.config import load_config
from thesis_rag.embedding import embed_queries
from thesis_rag.evaluator import evaluate_page_hits
from thesis_rag.indexing import build_faiss_index
from thesis_rag.ranking import chunk_hits_to_page_hits
from thesis_rag.retrieval_dense import search_faiss_stably
from thesis_rag.retrieval_hybrid import hybrid_retrieve_legacy_style
from thesis_rag.retrieval_sparse import build_bm25
from thesis_rag.schemas import ChunkRecord, PipelineConfig, RetrievalHit
from thesis_rag.utils import now_utc_iso, resolve_device, write_json


DOC_IDS = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce Table 4.5 with thesis_rag outputs.")
    parser.add_argument(
        "--pipeline-config",
        default="configs/thesis_rag.yaml",
        help="Base thesis_rag YAML config used for query embedding and retrieval parameters.",
    )
    parser.add_argument(
        "--ablation-bundle",
        default="results/thesis_ablations/chunk_size_ablation_2026-04-15",
        help="Existing thesis_rag chunk-size ablation bundle containing 224/56 saved outputs.",
    )
    parser.add_argument(
        "--bundle-dir",
        default="",
        help="Optional explicit output directory. Defaults to results/thesis_validations/table_4_5_doc_vs_global_<YYYY-MM-DD>.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output bundle if it already exists.",
    )
    parser.add_argument(
        "--single-doc",
        default="",
        help="Optional single document id to reproduce only one 50-query slice instead of the full 250-query set.",
    )
    return parser.parse_args()


def _default_bundle_dir() -> Path:
    return REPO_ROOT / "results" / "thesis_validations" / f"table_4_5_doc_vs_global_{date.today().isoformat()}"


def _doc_run_dir(ablation_bundle: Path, doc_id: str) -> Path:
    return ablation_bundle / "pipeline_outputs" / f"minilmcap_{doc_id}_chunk_224_56" / doc_id


def _load_doc_retrieval_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "metrics_by_k" not in payload:
        raise ValueError(f"metrics_by_k missing from {path}")
    return payload


def _combine_doc_scope_metrics(ablation_bundle: Path, doc_ids: list[str]) -> dict[str, Any]:
    metrics_by_doc: dict[str, dict[str, Any]] = {}
    for doc_id in doc_ids:
        metrics = _load_doc_retrieval_metrics(_doc_run_dir(ablation_bundle, doc_id) / "retrieval_metrics.json")
        metrics_by_doc[doc_id] = metrics
    ks = sorted(int(k) for k in next(iter(metrics_by_doc.values()))["metrics_by_k"].keys())
    combined: dict[str, Any] = {}
    total_queries = 0
    for doc_metrics in metrics_by_doc.values():
        total_queries += int(doc_metrics["metrics_by_k"][str(ks[0])]["num_queries"])
    for k in ks:
        weighted_hit = 0.0
        weighted_mrr = 0.0
        weighted_queries = 0
        for doc_id, metrics in metrics_by_doc.items():
            row = metrics["metrics_by_k"][str(k)]
            q = int(row["num_queries"])
            weighted_queries += q
            weighted_hit += float(row["page_hit_rate_at_k"]) * q
            weighted_mrr += float(row["mean_page_mrr_at_k"]) * q
        combined[str(k)] = {
            "num_queries": weighted_queries,
            "page_hit_rate_at_k": weighted_hit / max(weighted_queries, 1),
            "mean_page_recall_at_k": weighted_hit / max(weighted_queries, 1),
            "mean_page_precision_at_k": 0.0,
            "mean_page_mrr_at_k": weighted_mrr / max(weighted_queries, 1),
            "chunk_hit_rate_at_k": 0.0,
            "mean_chunk_precision_at_k": 0.0,
            "mean_chunk_mrr_at_k": 0.0,
        }
    return {
        "run_info": {
            "system": "thesis_rag",
            "experiment": "table_4_5_doc_scope_224_56",
            "doc_ids": doc_ids,
            "generated_utc": now_utc_iso(),
        },
        "metrics_by_k": combined,
    }


def _combine_saved_artifacts(ablation_bundle: Path) -> tuple[list[ChunkRecord], np.ndarray]:
    chunks_all: list[ChunkRecord] = []
    vectors_all: list[np.ndarray] = []
    for doc_id in DOC_IDS:
        run_dir = _doc_run_dir(ablation_bundle, doc_id)
        chunks = load_chunks(run_dir / "chunk_metadata.parquet")
        vectors = np.load(run_dir / "embeddings.npy")
        if len(chunks) != len(vectors):
            raise ValueError(f"Chunk/vector count mismatch for {doc_id}: {len(chunks)} vs {len(vectors)}")
        chunks_all.extend(chunks)
        vectors_all.append(vectors.astype(np.float32))
    combined = np.vstack(vectors_all).astype(np.float32)
    return chunks_all, combined


def _build_global_metrics(
    *,
    config: PipelineConfig,
    chunks: list[ChunkRecord],
    vectors: np.ndarray,
    out_dir: Path,
    doc_ids: list[str],
) -> dict[str, Any]:
    queries = []
    for doc_id in doc_ids:
        query_path = REPO_ROOT / "data_processed" / doc_id / "eval_set.json"
        queries.extend(load_queries(query_path))
    device = resolve_device(config.runtime.device)
    query_vectors = embed_queries(
        [query.query_text for query in queries],
        config.embedding,
        device=device,
        cache_dir=str(config.paths.model_cache_dir),
    )
    index = build_faiss_index(vectors, config.faiss)
    bm25 = build_bm25(chunks, config.bm25)
    max_k_search = max(100, config.retrieval.hybrid_top_k)
    dense_scores, dense_indices = search_faiss_stably(index, query_vectors, min(max_k_search, len(chunks)))
    _dense_hits, _bm25_hits, hybrid_chunk_hits = hybrid_retrieve_legacy_style(
        chunks=chunks,
        queries=queries,
        dense_scores=dense_scores,
        dense_indices=dense_indices,
        bm25=bm25,
        max_k_search=max_k_search,
        dense_weight=config.retrieval.dense_weight,
        bm25_weight=config.retrieval.sparse_weight,
        rrf_k=config.retrieval.rrf_k,
    )
    hybrid_page_hits = chunk_hits_to_page_hits(
        hybrid_chunk_hits,
        "hybrid_pages",
        chunk_limit=config.retrieval.hybrid_top_k,
    )
    metrics_by_k: dict[str, Any] = {}
    leakage_counts_by_k: dict[str, Any] = {}
    per_query_rows: list[dict[str, Any]] = []
    for k in config.evaluation.ks:
        k = int(k)
        page_hits_k = [hit for hit in hybrid_page_hits if int(hit.rank) <= k]
        results = evaluate_page_hits(queries, page_hits_k)
        hit_rate = sum(1 for result in results if result.first_relevant_rank is not None) / max(len(results), 1)
        mrr = sum(result.reciprocal_rank for result in results) / max(len(results), 1)
        leakage_details = _leakage_for_k(queries, hybrid_page_hits, k)
        metrics_by_k[str(k)] = {
            "num_queries": len(results),
            "page_hit_rate_at_k": hit_rate,
            "mean_page_recall_at_k": hit_rate,
            "mean_page_precision_at_k": 0.0,
            "mean_page_mrr_at_k": mrr,
            "chunk_hit_rate_at_k": 0.0,
            "mean_chunk_precision_at_k": 0.0,
            "mean_chunk_mrr_at_k": 0.0,
        }
        leakage_counts_by_k[str(k)] = {
            "queries_with_any_leakage": leakage_details["queries_with_any_leakage"],
            "any_leakage_rate_at_k": leakage_details["any_leakage_rate_at_k"],
            "mean_leakage_rate_at_k": leakage_details["mean_leakage_rate_at_k"],
        }
        if k == max(config.evaluation.ks):
            per_query_rows = leakage_details["per_query_rows"]
    payload = {
        "run_info": {
            "system": "thesis_rag",
            "experiment": "table_4_5_global_scope_224_56",
            "query_doc_ids": doc_ids,
            "corpus_doc_ids": DOC_IDS,
            "chunk_size_tokens": config.chunking.chunk_size_tokens,
            "chunk_overlap_tokens": config.chunking.chunk_overlap_tokens,
            "embedding_model": config.embedding.model_name,
            "apply_l2_normalization": config.embedding.apply_l2_normalization,
            "rrf_k": config.retrieval.rrf_k,
            "dense_weight": config.retrieval.dense_weight,
            "bm25_weight": config.retrieval.sparse_weight,
            "generated_utc": now_utc_iso(),
        },
        "metrics_by_k": metrics_by_k,
        "leakage_counts_by_k": leakage_counts_by_k,
        "results": per_query_rows,
    }
    write_json(out_dir / "global_scope_metrics.json", payload)
    pd.DataFrame(per_query_rows).to_csv(out_dir / "global_scope_results.csv", index=False)
    return payload


def _leakage_for_k(queries, page_hits: list[RetrievalHit], k: int) -> dict[str, Any]:
    by_query: dict[str, list[RetrievalHit]] = {}
    for hit in page_hits:
        if int(hit.rank) > k:
            continue
        by_query.setdefault(hit.query_id, []).append(hit)
    queries_with_any_leakage = 0
    total_leakage_rate = 0.0
    rows: list[dict[str, Any]] = []
    for query in queries:
        ranked = sorted(by_query.get(query.query_id, []), key=lambda hit: int(hit.rank))
        retrieved_doc_ids = [str(hit.doc_id) for hit in ranked]
        wrong = [doc_id for doc_id in retrieved_doc_ids if doc_id != query.doc_id]
        leakage_rate = len(wrong) / max(len(retrieved_doc_ids), 1)
        any_leakage = bool(wrong)
        if any_leakage:
            queries_with_any_leakage += 1
        total_leakage_rate += leakage_rate
        rows.append(
            {
                "query_id": query.query_id,
                "question": query.query_text,
                "doc_id": query.doc_id,
                "k": k,
                "retrieved_doc_ids_top_k": retrieved_doc_ids,
                "retrieved_pages_ranked": [int(hit.page_number) for hit in ranked],
                "top1_doc_id": retrieved_doc_ids[0] if retrieved_doc_ids else "",
                "leakage_rate_top_k": leakage_rate,
                "any_leakage_top_k": any_leakage,
            }
        )
    return {
        "queries_with_any_leakage": queries_with_any_leakage,
        "any_leakage_rate_at_k": queries_with_any_leakage / max(len(queries), 1),
        "mean_leakage_rate_at_k": total_leakage_rate / max(len(queries), 1),
        "per_query_rows": rows,
    }


def _build_comparison_frame(doc_metrics: dict[str, Any], global_metrics: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ks = sorted(int(k) for k in doc_metrics["metrics_by_k"].keys())
    for k in ks:
        sk = str(k)
        d = doc_metrics["metrics_by_k"][sk]
        g = global_metrics["metrics_by_k"][sk]
        gl = global_metrics["leakage_counts_by_k"][sk]
        rows.append(
            {
                "k": k,
                "doc_hit_rate": float(d["page_hit_rate_at_k"]),
                "global_hit_rate": float(g["page_hit_rate_at_k"]),
                "doc_mrr": float(d["mean_page_mrr_at_k"]),
                "global_mrr": float(g["mean_page_mrr_at_k"]),
                "global_any_leakage_rate": float(gl["any_leakage_rate_at_k"]),
                "global_mean_leakage_rate": float(gl["mean_leakage_rate_at_k"]),
            }
        )
    return pd.DataFrame(rows)


def _write_markdown(path: Path, df: pd.DataFrame) -> None:
    lines = [
        "# Table 4.5 Reproduction (thesis_rag)",
        "",
        f"Generated: {now_utc_iso()}",
        "",
        df.to_markdown(index=False),
        "",
    ]
    k1 = df.loc[df["k"] == 1].iloc[0]
    k3 = df.loc[df["k"] == 3].iloc[0]
    lines.extend(
        [
            f"- Document-constrained Hit@1: {k1['doc_hit_rate']:.4f}",
            f"- Global Hit@1: {k1['global_hit_rate']:.4f}",
            f"- Document-constrained MRR@10: {float(df.loc[df['k']==10, 'doc_mrr'].iloc[0]):.4f}",
            f"- Global MRR@10: {float(df.loc[df['k']==10, 'global_mrr'].iloc[0]):.4f}",
            f"- Wrong-document rate at rank 1: {k1['global_any_leakage_rate']:.4f}",
            f"- Queries with any cross-document leakage by k=3: {k3['global_any_leakage_rate']:.4f}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_reference(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    """Regenerate Table 4.5 per-document vs global retrieval metrics and write CSV/Markdown outputs."""
    args = parse_args()
    pipeline_config_path = (REPO_ROOT / args.pipeline_config).resolve()
    ablation_bundle = (REPO_ROOT / args.ablation_bundle).resolve()
    bundle_dir = Path(args.bundle_dir).resolve() if args.bundle_dir else _default_bundle_dir()
    if bundle_dir.exists():
        if not args.force:
            raise FileExistsError(f"{bundle_dir} already exists; pass --force to overwrite it.")
        shutil.rmtree(bundle_dir)
    (bundle_dir / "comparison").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "reference").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "configs").mkdir(parents=True, exist_ok=True)

    _copy_reference(pipeline_config_path, bundle_dir / "configs" / pipeline_config_path.name)
    legacy_compare = REPO_ROOT / "results" / "ablations" / "global_scope_grampian_2020_2021_5docs_compare" / "doc_vs_global_scope_metrics.csv"
    _copy_reference(legacy_compare, bundle_dir / "reference" / legacy_compare.name)

    config = load_config(pipeline_config_path)
    config = replace(config, chunking=replace(config.chunking, chunk_size_tokens=224, chunk_overlap_tokens=56))
    model_path = Path(config.embedding.model_name)
    if model_path.is_absolute() and not model_path.exists():
        fallback_model = REPO_ROOT / "models" / model_path.name
        if fallback_model.exists():
            config = replace(
                config,
                embedding=replace(config.embedding, model_name=str(fallback_model.resolve())),
            )

    selected_docs = [args.single_doc] if args.single_doc else list(DOC_IDS)
    doc_metrics = _combine_doc_scope_metrics(ablation_bundle, selected_docs)
    write_json(bundle_dir / "doc_scope_metrics.json", doc_metrics)

    chunks, vectors = _combine_saved_artifacts(ablation_bundle)
    global_metrics = _build_global_metrics(
        config=config,
        chunks=chunks,
        vectors=vectors,
        out_dir=bundle_dir,
        doc_ids=selected_docs,
    )

    comparison = _build_comparison_frame(doc_metrics, global_metrics)
    comparison.to_csv(bundle_dir / "comparison" / "doc_vs_global_scope_metrics.csv", index=False)
    _write_markdown(bundle_dir / "comparison" / "doc_vs_global_scope_metrics.md", comparison)

    leakage_rows = pd.read_csv(bundle_dir / "global_scope_results.csv")
    top1_leakage = leakage_rows[(leakage_rows["k"] == 1) & (leakage_rows["top1_doc_id"] != leakage_rows["doc_id"])].copy()
    top1_leakage.to_csv(bundle_dir / "comparison" / "global_scope_top1_leakage_examples.csv", index=False)

    summary = {
        "generated_utc": now_utc_iso(),
        "query_doc_ids": selected_docs,
        "corpus_doc_ids": DOC_IDS,
        "doc_scope_hit@1": float(comparison.loc[comparison["k"] == 1, "doc_hit_rate"].iloc[0]),
        "global_scope_hit@1": float(comparison.loc[comparison["k"] == 1, "global_hit_rate"].iloc[0]),
        "doc_scope_mrr@10": float(comparison.loc[comparison["k"] == 10, "doc_mrr"].iloc[0]),
        "global_scope_mrr@10": float(comparison.loc[comparison["k"] == 10, "global_mrr"].iloc[0]),
        "wrong_document_rate_at_1": float(comparison.loc[comparison["k"] == 1, "global_any_leakage_rate"].iloc[0]),
        "any_leakage_rate_at_3": float(comparison.loc[comparison["k"] == 3, "global_any_leakage_rate"].iloc[0]),
    }
    write_json(bundle_dir / "comparison" / "summary.json", summary)
    print(bundle_dir)


if __name__ == "__main__":
    main()
