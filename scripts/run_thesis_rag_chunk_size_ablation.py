from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections import defaultdict
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from thesis_rag.artifacts import load_queries, save_chunks, save_hits, save_pages
from thesis_rag.config import load_config
from thesis_rag.diagnostics import build_query_diagnostics, save_diagnostics_csv
from thesis_rag.embedding import build_embedding_text, load_embedding_model
from thesis_rag.evaluator import evaluate_page_hits
from thesis_rag.fusion import reciprocal_rank_fusion
from thesis_rag.indexing import build_faiss_index, save_chunk_metadata, save_embeddings, save_faiss_index
from thesis_rag.loader import extract_page_structures
from thesis_rag.preprocessing import build_chunk_records, build_page_records
from thesis_rag.ranking import chunk_hits_to_page_hits
from thesis_rag.retrieval_dense import dense_retrieve_legacy_style, search_faiss_stably
from thesis_rag.retrieval_hybrid import hybrid_retrieve_legacy_style
from thesis_rag.retrieval_sparse import build_bm25, sparse_retrieve_legacy_style
from thesis_rag.schemas import ChunkRecord, DocumentRecord, PipelineConfig, QueryRecord
from thesis_rag.utils import configure_logging, git_commit_hash, l2_normalize, now_utc_iso, resolve_device, set_global_determinism


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the thesis chunk-size ablation with thesis_rag and compare it against the legacy reference table."
    )
    parser.add_argument(
        "--pipeline-config",
        default="configs/thesis_rag.yaml",
        help="Base thesis_rag YAML config.",
    )
    parser.add_argument(
        "--legacy-config",
        default="configs/retrieval_tuning_minilm_cap_5docs_rerun_2026-03-18.yaml",
        help="Legacy ablation YAML used to define the thesis chunk-size experiment set.",
    )
    parser.add_argument(
        "--legacy-reference-csv",
        default="results/thesis_rebuild_freeze_smoke_exports/chunk_ablation_table.csv",
        help="Legacy aggregate chunk-ablation table used as the parity reference.",
    )
    parser.add_argument(
        "--bundle-dir",
        default="",
        help="Optional explicit output bundle directory. Defaults to results/thesis_ablations/chunk_size_ablation_<YYYY-MM-DD>.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing bundle directory if it already exists.",
    )
    parser.add_argument(
        "--no-subsection-boost",
        action="store_true",
        help="Disable subsection boost in hybrid retrieval (enable_subsection_boost=False).",
    )
    return parser.parse_args()


def _default_bundle_dir() -> Path:
    return REPO_ROOT / "results" / "thesis_ablations" / f"chunk_size_ablation_{date.today().isoformat()}"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _group_legacy_experiments(legacy_cfg: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for experiment in legacy_cfg.get("experiments", []):
        doc_id = str(experiment["doc_id"])
        grouped[doc_id].append(experiment)
    for doc_id in grouped:
        grouped[doc_id].sort(
            key=lambda item: (
                int(item["chunking"]["size_tokens"]),
                int(item["chunking"]["overlap_tokens"]),
                str(item["name"]),
            )
        )
    return dict(sorted(grouped.items()))


def _prepare_bundle(bundle_dir: Path, pipeline_config_path: Path, legacy_config_path: Path, legacy_reference_csv: Path) -> None:
    for rel in ("configs", "manifests", "tables", "comparison", "pipeline_outputs", "logs", "notes", "reference"):
        (bundle_dir / rel).mkdir(parents=True, exist_ok=True)
    _copy_if_exists(pipeline_config_path, bundle_dir / "configs" / pipeline_config_path.name)
    _copy_if_exists(legacy_config_path, bundle_dir / "configs" / legacy_config_path.name)
    _copy_if_exists(legacy_reference_csv, bundle_dir / "reference" / legacy_reference_csv.name)


def _load_model(config: PipelineConfig):
    device = resolve_device(config.runtime.device)
    model = load_embedding_model(config.embedding, device=device, cache_dir=str(config.paths.model_cache_dir))
    return model, device


def _encode_texts(texts: list[str], *, model, config: PipelineConfig) -> Any:
    vectors = model.encode(
        texts,
        batch_size=config.embedding.batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    if config.embedding.apply_l2_normalization:
        vectors = l2_normalize(vectors)
    return vectors


def _metrics_payload(
    *,
    queries: list[QueryRecord],
    hybrid_chunk_hits: list[Any],
    ks: list[int],
    run_info: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics_by_k: dict[str, Any] = {}
    per_query_rows: list[dict[str, Any]] = []
    for k in ks:
        page_hits = chunk_hits_to_page_hits(hybrid_chunk_hits, "hybrid_pages", chunk_limit=int(k))
        results = evaluate_page_hits(queries, page_hits)
        hit_rate = sum(1 for result in results if result.first_relevant_rank is not None) / max(len(results), 1)
        mrr = sum(result.reciprocal_rank for result in results) / max(len(results), 1)
        metrics_by_k[str(k)] = {
            "num_queries": int(len(results)),
            "page_hit_rate_at_k": float(hit_rate),
            "mean_page_recall_at_k": float(hit_rate),
            "mean_page_precision_at_k": 0.0,
            "mean_page_mrr_at_k": float(mrr),
            "chunk_hit_rate_at_k": 0.0,
            "mean_chunk_precision_at_k": 0.0,
            "mean_chunk_mrr_at_k": 0.0,
        }
        if int(k) == max(ks):
            for result in results:
                per_query_rows.append(result.to_dict())
    return {"run_info": run_info, "metrics_by_k": metrics_by_k}, per_query_rows


def _prep_metrics_payload(*, doc_id: str, experiment_name: str, pages: list[Any], chunks: list[ChunkRecord], config: PipelineConfig) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "experiment": experiment_name,
        "created_utc": now_utc_iso(),
        "counts": {
            "pages_total": len(pages),
            "chunks_total": len(chunks),
            "ocr_pages_total": sum(1 for page in pages if page.ocr_used),
            "table_pages_total": sum(1 for page in pages if page.is_table),
        },
        "params": {
            "chunk_size_tokens": int(config.chunking.chunk_size_tokens),
            "chunk_overlap_tokens": int(config.chunking.chunk_overlap_tokens),
            "embedding_model": config.embedding.model_name,
            "apply_l2_normalization": bool(config.embedding.apply_l2_normalization),
            "expected_dimension": int(config.embedding.expected_dimension),
            "faiss_index_type": config.faiss.index_type,
            "bm25_k1": float(config.bm25.k1),
            "bm25_b": float(config.bm25.b),
            "dense_weight": float(config.retrieval.dense_weight),
            "sparse_weight": float(config.retrieval.sparse_weight),
            "rrf_k": int(config.retrieval.rrf_k),
            "dense_top_k": int(config.retrieval.dense_top_k),
            "sparse_top_k": int(config.retrieval.sparse_top_k),
            "hybrid_top_k": int(config.retrieval.hybrid_top_k),
            "evaluation_ks": [int(k) for k in config.evaluation.ks],
            "device": config.runtime.device,
            "random_seed": int(config.runtime.random_seed),
        },
    }


def _build_aggregate_table(per_doc_rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(per_doc_rows)
    aggregate = (
        df.groupby(["chunk_size_tokens", "chunk_overlap_tokens"], as_index=False)
        .agg(
            page_hit1=("page_hit1", "mean"),
            mrr10=("mrr10", "mean"),
            queries=("queries", "sum"),
            chunks_indexed=("chunks_indexed", "sum"),
        )
        .sort_values(["chunk_size_tokens", "chunk_overlap_tokens"])
        .reset_index(drop=True)
    )
    baseline_mask = (aggregate["chunk_size_tokens"] == 224) & (aggregate["chunk_overlap_tokens"] == 56)
    if not baseline_mask.any():
        raise RuntimeError("224/56 baseline not found in thesis_rag ablation aggregate.")
    baseline_hit = float(aggregate.loc[baseline_mask, "page_hit1"].iloc[0])
    baseline_mrr = float(aggregate.loc[baseline_mask, "mrr10"].iloc[0])
    aggregate["delta_hit1"] = aggregate["page_hit1"] - baseline_hit
    aggregate["delta_mrr10"] = aggregate["mrr10"] - baseline_mrr
    aggregate["configuration"] = (
        aggregate["chunk_size_tokens"].astype(int).astype(str)
        + " / "
        + aggregate["chunk_overlap_tokens"].astype(int).astype(str)
    )
    return aggregate[
        ["configuration", "page_hit1", "delta_hit1", "mrr10", "delta_mrr10", "queries", "chunks_indexed"]
    ]


def _write_markdown_report(path: Path, comparison: pd.DataFrame) -> None:
    mismatches = comparison[
        (comparison["page_hit1_diff"].abs() > 1e-12)
        | (comparison["mrr10_diff"].abs() > 1e-12)
        | (comparison["queries_diff"] != 0)
        | (comparison["chunks_indexed_diff"] != 0)
    ].copy()
    lines = [
        "# Chunk Size Ablation Comparison",
        "",
        f"Generated: {now_utc_iso()}",
        "",
    ]
    if mismatches.empty:
        lines.extend(["Status: exact match against the legacy aggregate reference.", ""])
    else:
        lines.extend(
            [
                f"Status: {len(mismatches)} configuration row(s) differ from the legacy aggregate reference.",
                "",
                "| Configuration | Legacy Hit@1 | New Hit@1 | Hit@1 Diff | Legacy MRR@10 | New MRR@10 | MRR@10 Diff | Legacy Chunks | New Chunks | Chunk Diff |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in mismatches.itertuples(index=False):
            lines.append(
                f"| {row.configuration} | {row.page_hit1_legacy:.6f} | {row.page_hit1_new:.6f} | {row.page_hit1_diff:+.6f} | "
                f"{row.mrr10_legacy:.6f} | {row.mrr10_new:.6f} | {row.mrr10_diff:+.6f} | "
                f"{int(row.chunks_indexed_legacy)} | {int(row.chunks_indexed_new)} | {int(row.chunks_indexed_diff):+d} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _comparison_frame(legacy_csv: Path, new_table: pd.DataFrame) -> pd.DataFrame:
    legacy = pd.read_csv(legacy_csv).rename(
        columns={
            "page_hit1": "page_hit1_legacy",
            "delta_hit1": "delta_hit1_legacy",
            "mrr10": "mrr10_legacy",
            "delta_mrr10": "delta_mrr10_legacy",
            "queries": "queries_legacy",
            "chunks_indexed": "chunks_indexed_legacy",
        }
    )
    new = new_table.rename(
        columns={
            "page_hit1": "page_hit1_new",
            "delta_hit1": "delta_hit1_new",
            "mrr10": "mrr10_new",
            "delta_mrr10": "delta_mrr10_new",
            "queries": "queries_new",
            "chunks_indexed": "chunks_indexed_new",
        }
    )
    comparison = legacy.merge(new, on="configuration", how="outer", validate="one_to_one").sort_values("configuration")
    comparison["page_hit1_diff"] = comparison["page_hit1_new"] - comparison["page_hit1_legacy"]
    comparison["mrr10_diff"] = comparison["mrr10_new"] - comparison["mrr10_legacy"]
    comparison["queries_diff"] = comparison["queries_new"] - comparison["queries_legacy"]
    comparison["chunks_indexed_diff"] = comparison["chunks_indexed_new"] - comparison["chunks_indexed_legacy"]
    return comparison


def _run_single_experiment(
    *,
    experiment: dict[str, Any],
    pages_by_doc: dict[str, list[Any]],
    queries_by_doc: dict[str, list[QueryRecord]],
    model,
    config: PipelineConfig,
    bundle_dir: Path,
    enable_subsection_boost: bool = True,
) -> dict[str, Any]:
    doc_id = str(experiment["doc_id"])
    experiment_name = str(experiment["name"])
    chunk_size = int(experiment["chunking"]["size_tokens"])
    chunk_overlap = int(experiment["chunking"]["overlap_tokens"])

    experiment_dir = bundle_dir / "pipeline_outputs" / experiment_name / doc_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    run_config = replace(
        config,
        chunking=replace(config.chunking, chunk_size_tokens=chunk_size, chunk_overlap_tokens=chunk_overlap),
        retrieval=replace(
            config.retrieval,
            dense_top_k=10,
            sparse_top_k=10,
            hybrid_top_k=10,
            rrf_k=20,
            dense_weight=0.5,
            sparse_weight=2.0,
        ),
        evaluation=replace(config.evaluation, ks=[1, 3, 5, 10]),
    )

    pages = pages_by_doc[doc_id]
    chunks = build_chunk_records(
        doc_id,
        pages,
        run_config.chunking,
        source_pdf_path=REPO_ROOT / str(experiment["pdf_path"]),
    )
    save_pages(pages, experiment_dir)
    save_chunks(chunks, experiment_dir)

    chunk_texts = [build_embedding_text(chunk) for chunk in chunks]
    chunk_vectors = _encode_texts(chunk_texts, model=model, config=run_config).astype("float32")
    index = build_faiss_index(chunk_vectors, run_config.faiss)
    save_embeddings(chunk_vectors, experiment_dir / "embeddings.npy")
    save_faiss_index(index, experiment_dir / "faiss.index")
    save_chunk_metadata(chunks, experiment_dir / "chunk_metadata.parquet")

    queries = queries_by_doc[doc_id]
    query_vectors = _encode_texts([query.query_text for query in queries], model=model, config=run_config).astype("float32")
    dense_chunk_hits = dense_retrieve_legacy_style(
        index,
        chunks,
        queries,
        query_vectors,
        top_k=10,
        max_k_search=100,
    )
    dense_page_hits = chunk_hits_to_page_hits(dense_chunk_hits, "dense_pages", chunk_limit=10)
    bm25 = build_bm25(chunks, run_config.bm25)
    sparse_chunk_hits = sparse_retrieve_legacy_style(bm25, chunks, queries, top_k=100)
    sparse_page_hits = chunk_hits_to_page_hits(sparse_chunk_hits, "bm25_pages", chunk_limit=10)
    raw_dense_scores, raw_dense_indices = search_faiss_stably(index, query_vectors, min(100, len(chunks)))
    _, _, hybrid_chunk_hits = hybrid_retrieve_legacy_style(
        chunks=chunks,
        queries=queries,
        dense_scores=raw_dense_scores,
        dense_indices=raw_dense_indices,
        bm25=bm25,
        max_k_search=100,
        dense_weight=run_config.retrieval.dense_weight,
        bm25_weight=run_config.retrieval.sparse_weight,
        rrf_k=run_config.retrieval.rrf_k,
        enable_subsection_boost=enable_subsection_boost,
    )
    hybrid_page_hits = chunk_hits_to_page_hits(hybrid_chunk_hits, "hybrid_pages", chunk_limit=10)

    save_hits(dense_page_hits, experiment_dir / "dense_page_hits.jsonl")
    save_hits(sparse_page_hits, experiment_dir / "bm25_page_hits.jsonl")
    save_hits(hybrid_page_hits, experiment_dir / "hybrid_page_hits.jsonl")

    run_info = {
        "system": "thesis_rag",
        "experiment": experiment_name,
        "doc_id": doc_id,
        "chunk_size_tokens": chunk_size,
        "chunk_overlap_tokens": chunk_overlap,
        "embedding_model": run_config.embedding.model_name,
        "apply_l2_normalization": bool(run_config.embedding.apply_l2_normalization),
        "rrf_k": int(run_config.retrieval.rrf_k),
        "dense_weight": float(run_config.retrieval.dense_weight),
        "bm25_weight": float(run_config.retrieval.sparse_weight),
        "bm25_k1": float(run_config.bm25.k1),
        "bm25_b": float(run_config.bm25.b),
        "faiss_index_type": run_config.faiss.index_type,
        "random_seed": int(run_config.runtime.random_seed),
        "device": run_config.runtime.device,
        "generated_utc": now_utc_iso(),
    }
    retrieval_metrics, per_query_results = _metrics_payload(
        queries=queries,
        hybrid_chunk_hits=hybrid_chunk_hits,
        ks=run_config.evaluation.ks,
        run_info=run_info,
    )
    diagnostics = build_query_diagnostics(queries, dense_page_hits, sparse_page_hits, hybrid_page_hits, evaluate_page_hits(queries, hybrid_page_hits))
    save_diagnostics_csv(diagnostics, experiment_dir / "diagnostics.csv")
    _write_json(experiment_dir / "retrieval_metrics.json", retrieval_metrics)
    _write_json(experiment_dir / "per_query_results.json", per_query_results)
    _write_json(experiment_dir / "metrics.json", _prep_metrics_payload(doc_id=doc_id, experiment_name=experiment_name, pages=pages, chunks=chunks, config=run_config))
    _write_json(experiment_dir / "run_manifest.json", {"run_info": run_info, "git_commit": git_commit_hash(REPO_ROOT)})

    return {
        "experiment": experiment_name,
        "document": doc_id,
        "chunk_size_tokens": chunk_size,
        "chunk_overlap_tokens": chunk_overlap,
        "queries": int(retrieval_metrics["metrics_by_k"]["1"]["num_queries"]),
        "page_hit1": float(retrieval_metrics["metrics_by_k"]["1"]["page_hit_rate_at_k"]),
        "mrr10": float(retrieval_metrics["metrics_by_k"]["10"]["mean_page_mrr_at_k"]),
        "chunks_indexed": int(len(chunks)),
        "embedding_model": run_config.embedding.model_name,
        "apply_l2_normalization": bool(run_config.embedding.apply_l2_normalization),
        "rrf_k": int(run_config.retrieval.rrf_k),
        "dense_weight": float(run_config.retrieval.dense_weight),
        "bm25_weight": float(run_config.retrieval.sparse_weight),
        "bm25_k1": float(run_config.bm25.k1),
        "bm25_b": float(run_config.bm25.b),
        "faiss_index_type": run_config.faiss.index_type,
    }


def main() -> None:
    args = parse_args()
    pipeline_config_path = (REPO_ROOT / args.pipeline_config).resolve()
    legacy_config_path = (REPO_ROOT / args.legacy_config).resolve()
    legacy_reference_csv = (REPO_ROOT / args.legacy_reference_csv).resolve()
    bundle_dir = Path(args.bundle_dir).resolve() if args.bundle_dir else _default_bundle_dir().resolve()

    if bundle_dir.exists():
        if not args.force:
            raise FileExistsError(f"Bundle directory already exists: {bundle_dir}")
        shutil.rmtree(bundle_dir)
    _prepare_bundle(bundle_dir, pipeline_config_path, legacy_config_path, legacy_reference_csv)
    configure_logging(bundle_dir / "logs" / "chunk_size_ablation.log", "INFO")

    pipeline_config = load_config(pipeline_config_path)
    pipeline_config.embedding.model_name = str((REPO_ROOT / "models" / "all-MiniLM-L6-v2").resolve())
    pipeline_config.paths.model_cache_dir = (REPO_ROOT / "models").resolve()
    pipeline_config.runtime.device = "cpu"
    pipeline_config.runtime.corpus_name = "grampian-thesis-chunk-size-ablation"
    pipeline_config.runtime.dataset_version = "legacy-250-query-ablation"
    set_global_determinism(pipeline_config.runtime.random_seed, pipeline_config.runtime.deterministic_torch)

    legacy_cfg = _load_yaml(legacy_config_path)
    experiments_by_doc = _group_legacy_experiments(legacy_cfg)
    _write_json(
        bundle_dir / "manifests" / "ablation_spec.json",
        {
            "created_utc": now_utc_iso(),
            "bundle_dir": str(bundle_dir),
            "pipeline_config_path": str(pipeline_config_path),
            "legacy_config_path": str(legacy_config_path),
            "legacy_reference_csv": str(legacy_reference_csv),
            "git_commit": git_commit_hash(REPO_ROOT),
            "global_settings": {
                "embedding_model": pipeline_config.embedding.model_name,
                "apply_l2_normalization": bool(pipeline_config.embedding.apply_l2_normalization),
                "faiss_index_type": pipeline_config.faiss.index_type,
                "device": pipeline_config.runtime.device,
                "random_seed": int(pipeline_config.runtime.random_seed),
                "evaluation_ks": [1, 3, 5, 10],
                "rrf_k": 20,
                "dense_weight": 0.5,
                "bm25_weight": 2.0,
                "bm25_k1": float(pipeline_config.bm25.k1),
                "bm25_b": float(pipeline_config.bm25.b),
            },
            "documents": experiments_by_doc,
        },
    )

    model, device = _load_model(pipeline_config)
    LOGGER.info("Loaded embedding model on %s", device)

    pages_by_doc: dict[str, list[Any]] = {}
    queries_by_doc: dict[str, list[QueryRecord]] = {}
    for doc_id, experiments in experiments_by_doc.items():
        first_experiment = experiments[0]
        document = DocumentRecord(doc_id=doc_id, pdf_path=str((REPO_ROOT / first_experiment["pdf_path"]).resolve()))
        LOGGER.info("Extracting pages for %s", doc_id)
        page_structures = extract_page_structures(document)
        pages_by_doc[doc_id] = build_page_records(doc_id, page_structures, pipeline_config.ocr)
        queries_by_doc[doc_id] = load_queries((REPO_ROOT / first_experiment["source_eval_set"]).resolve())

    per_doc_rows: list[dict[str, Any]] = []
    for doc_id, experiments in experiments_by_doc.items():
        for experiment in experiments:
            LOGGER.info(
                "Running %s (%s %s/%s)",
                experiment["name"],
                doc_id,
                experiment["chunking"]["size_tokens"],
                experiment["chunking"]["overlap_tokens"],
            )
            per_doc_rows.append(
                _run_single_experiment(
                    experiment=experiment,
                    pages_by_doc=pages_by_doc,
                    queries_by_doc=queries_by_doc,
                    model=model,
                    config=pipeline_config,
                    bundle_dir=bundle_dir,
                    enable_subsection_boost=not args.no_subsection_boost,
                )
            )

    per_doc_df = pd.DataFrame(per_doc_rows).sort_values(["document", "chunk_size_tokens", "chunk_overlap_tokens"])
    aggregate_df = _build_aggregate_table(per_doc_rows)
    settings_df = (
        per_doc_df[
            [
                "chunk_size_tokens",
                "chunk_overlap_tokens",
                "embedding_model",
                "apply_l2_normalization",
                "rrf_k",
                "dense_weight",
                "bm25_weight",
                "bm25_k1",
                "bm25_b",
                "faiss_index_type",
            ]
        ]
        .drop_duplicates()
        .sort_values(["chunk_size_tokens", "chunk_overlap_tokens"])
        .reset_index(drop=True)
    )
    settings_df["configuration"] = (
        settings_df["chunk_size_tokens"].astype(int).astype(str)
        + " / "
        + settings_df["chunk_overlap_tokens"].astype(int).astype(str)
    )
    aggregate_settings_df = aggregate_df.merge(
        settings_df[
            [
                "configuration",
                "embedding_model",
                "apply_l2_normalization",
                "rrf_k",
                "dense_weight",
                "bm25_weight",
                "bm25_k1",
                "bm25_b",
                "faiss_index_type",
            ]
        ],
        on="configuration",
        how="left",
        validate="one_to_one",
    )
    comparison_df = _comparison_frame(legacy_reference_csv, aggregate_df)

    per_doc_csv = bundle_dir / "tables" / "chunk_ablation_by_document.csv"
    aggregate_csv = bundle_dir / "tables" / "chunk_ablation_table.csv"
    aggregate_json = bundle_dir / "tables" / "chunk_ablation_table.json"
    aggregate_settings_csv = bundle_dir / "tables" / "chunk_ablation_table_with_settings.csv"
    comparison_csv = bundle_dir / "comparison" / "legacy_vs_thesis_rag_chunk_ablation.csv"
    comparison_json = bundle_dir / "comparison" / "legacy_vs_thesis_rag_chunk_ablation.json"
    comparison_md = bundle_dir / "comparison" / "legacy_vs_thesis_rag_chunk_ablation.md"

    per_doc_df.to_csv(per_doc_csv, index=False)
    aggregate_df.to_csv(aggregate_csv, index=False)
    aggregate_settings_df.to_csv(aggregate_settings_csv, index=False)
    _write_json(aggregate_json, json.loads(aggregate_df.to_json(orient="records")))
    comparison_df.to_csv(comparison_csv, index=False)
    _write_json(comparison_json, json.loads(comparison_df.to_json(orient="records")))
    _write_markdown_report(comparison_md, comparison_df)

    _write_json(
        bundle_dir / "manifests" / "execution_manifest.json",
        {
            "created_utc": now_utc_iso(),
            "bundle_dir": str(bundle_dir),
            "git_commit": git_commit_hash(REPO_ROOT),
            "python": sys.executable,
            "total_experiments": int(len(per_doc_rows)),
            "total_documents": int(len(experiments_by_doc)),
            "comparison_summary": {
                "num_rows": int(len(comparison_df)),
                "num_mismatched_rows": int(
                    (
                        (comparison_df["page_hit1_diff"].abs() > 1e-12)
                        | (comparison_df["mrr10_diff"].abs() > 1e-12)
                        | (comparison_df["queries_diff"] != 0)
                        | (comparison_df["chunks_indexed_diff"] != 0)
                    ).sum()
                ),
            },
        },
    )

    print(bundle_dir)


if __name__ == "__main__":
    main()
