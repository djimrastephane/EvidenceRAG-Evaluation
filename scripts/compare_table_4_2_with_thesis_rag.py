from __future__ import annotations

"""Compare legacy Table 4.2 against thesis_rag 224/56 method-level results.

This script builds an auditable bundle for the method comparison table used in
the thesis. It reuses the saved ``thesis_rag`` 5-document ``224/56`` benchmark
artifacts for:

- Dense (MiniLM)
- BM25-only
- Hybrid + subsection boost

and reconstructs:

- Hybrid (base)

by rerunning only the hybrid fusion stage with subsection boost disabled over
the saved chunk metadata, FAISS indexes, and evaluation query sets.
"""

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from thesis_rag.artifacts import load_chunks, load_queries
from thesis_rag.config import load_config
from thesis_rag.embedding import embed_queries
from thesis_rag.evaluator import evaluate_page_hits
from thesis_rag.ranking import chunk_hits_to_page_hits
from thesis_rag.retrieval_dense import search_faiss_stably
from thesis_rag.retrieval_hybrid import hybrid_retrieve_legacy_style
from thesis_rag.retrieval_sparse import build_bm25
from thesis_rag.schemas import RetrievalHit
from thesis_rag.utils import resolve_device, set_global_determinism


DEFAULT_LEGACY_CSV = REPO_ROOT / "results" / "current_method_comparison_2026-04-07" / "current_method_comparison_aggregate.csv"
DEFAULT_PIPELINE_CONFIG = REPO_ROOT / "configs" / "thesis_rag.yaml"
DEFAULT_THESIS_RAG_ROOT = REPO_ROOT / "results" / "thesis_ablations" / "chunk_size_ablation_2026-04-15" / "pipeline_outputs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "thesis_validations" / f"table_4_2_comparison_{date.today().isoformat()}"
DOC_IDS = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Table 4.2 against thesis_rag method-level results.")
    parser.add_argument("--legacy-csv", type=Path, default=DEFAULT_LEGACY_CSV)
    parser.add_argument("--pipeline-config", type=Path, default=DEFAULT_PIPELINE_CONFIG)
    parser.add_argument("--thesis-rag-root", type=Path, default=DEFAULT_THESIS_RAG_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _read_legacy_table(path: Path) -> dict[str, dict[str, float | int | str]]:
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    methods = {
        "Dense (MiniLM)": "dense",
        "BM25-only": "bm25",
        "Hybrid (base)": "hybrid_base",
        "Hybrid + subsection boost": "hybrid_boost",
    }
    selected: dict[str, dict[str, str]] = {}
    for row in rows:
        label = row["label"]
        if label not in methods:
            continue
        selected.setdefault(label, {})[row["k"]] = row
    missing = [label for label in methods if not {"1", "3", "10"}.issubset(set(selected.get(label, {})))]
    if missing:
        raise RuntimeError(f"Missing legacy rows for methods: {missing}")
    result: dict[str, dict[str, float | int | str]] = {}
    for label in methods:
        rows_by_k = selected[label]
        result[label] = {
            "source": str(path),
            "method": label,
            "queries_evaluated": int(rows_by_k["10"]["queries"]),
            "page_hit_at_1": float(rows_by_k["1"]["weighted_page_hit"]),
            "page_hit_at_3": float(rows_by_k["3"]["weighted_page_hit"]),
            "mrr_at_10": float(rows_by_k["10"]["weighted_page_mrr"]),
        }
    return result


def _load_hits(path: Path) -> list[RetrievalHit]:
    import pandas as pd

    frame = pd.read_csv(path)
    return [RetrievalHit(**row) for row in frame.to_dict(orient="records")]


def _aggregate_from_hits(doc_root: Path, *, method: str) -> list[dict[str, object]]:
    if method == "dense":
        filename = "dense_page_hits.csv"
    elif method == "bm25":
        filename = "bm25_page_hits.csv"
    elif method == "hybrid_boost":
        filename = "hybrid_page_hits.csv"
    else:
        raise ValueError(method)
    all_results: list[dict[str, object]] = []
    for doc_id in DOC_IDS:
        hits = _load_hits(doc_root / f"minilmcap_{doc_id}_chunk_224_56" / doc_id / filename)
        queries = load_queries(REPO_ROOT / "data_processed" / doc_id / "eval_set.json")
        results = evaluate_page_hits(queries, hits)
        all_results.extend(result.to_dict() for result in results)
    return all_results


def _aggregate_hybrid_base(doc_root: Path, config_path: Path) -> list[dict[str, object]]:
    import faiss

    config = load_config(config_path)
    config.embedding.model_name = str(REPO_ROOT / "models" / "all-MiniLM-L6-v2")
    config.retrieval.dense_top_k = 10
    config.retrieval.sparse_top_k = 10
    config.retrieval.hybrid_top_k = 10
    config.retrieval.rrf_k = 20
    config.retrieval.dense_weight = 0.5
    config.retrieval.sparse_weight = 2.0
    set_global_determinism(config.runtime.random_seed, config.runtime.deterministic_torch)
    device = resolve_device(config.runtime.device)
    all_results: list[dict[str, object]] = []
    for doc_id in DOC_IDS:
        artifact_dir = doc_root / f"minilmcap_{doc_id}_chunk_224_56" / doc_id
        chunks = load_chunks(artifact_dir / "chunk_metadata.parquet")
        queries = load_queries(REPO_ROOT / "data_processed" / doc_id / "eval_set.json")
        index = faiss.read_index(str(artifact_dir / "faiss.index"))
        query_vectors = embed_queries(
            [query.query_text for query in queries],
            config.embedding,
            device=device,
            cache_dir=str(config.paths.model_cache_dir),
        )
        bm25 = build_bm25(chunks, config.bm25)
        raw_dense_scores, raw_dense_indices = search_faiss_stably(
            index,
            query_vectors,
            min(max(100, config.retrieval.hybrid_top_k), len(chunks)),
        )
        _dense, _bm25, hybrid_chunk_hits = hybrid_retrieve_legacy_style(
            chunks=chunks,
            queries=queries,
            dense_scores=raw_dense_scores,
            dense_indices=raw_dense_indices,
            bm25=bm25,
            max_k_search=max(100, config.retrieval.hybrid_top_k),
            dense_weight=config.retrieval.dense_weight,
            bm25_weight=config.retrieval.sparse_weight,
            rrf_k=config.retrieval.rrf_k,
            enable_subsection_boost=False,
            subsection_boost=0.0,
        )
        page_hits = chunk_hits_to_page_hits(
            hybrid_chunk_hits,
            "hybrid_pages_base",
            chunk_limit=config.retrieval.hybrid_top_k,
        )
        results = evaluate_page_hits(queries, page_hits)
        all_results.extend(result.to_dict() for result in results)
    return all_results


def _metrics(rows: list[dict[str, object]]) -> dict[str, float | int]:
    total = len(rows)
    return {
        "queries_evaluated": total,
        "page_hit_at_1": sum(1 for row in rows if bool(row["hit_at_1"])) / total,
        "page_hit_at_3": sum(1 for row in rows if bool(row["hit_at_3"])) / total,
        "mrr_at_10": sum(float(row["reciprocal_rank"]) for row in rows) / total,
    }


def _build_thesis_rag_table(doc_root: Path, config_path: Path) -> dict[str, dict[str, float | int | str]]:
    dense_rows = _aggregate_from_hits(doc_root, method="dense")
    bm25_rows = _aggregate_from_hits(doc_root, method="bm25")
    boost_rows = _aggregate_from_hits(doc_root, method="hybrid_boost")
    base_rows = _aggregate_hybrid_base(doc_root, config_path)
    return {
        "Dense (MiniLM)": {
            "source": str(doc_root),
            "method": "Dense (MiniLM)",
            **_metrics(dense_rows),
        },
        "BM25-only": {
            "source": str(doc_root),
            "method": "BM25-only",
            **_metrics(bm25_rows),
        },
        "Hybrid (base)": {
            "source": str(doc_root),
            "method": "Hybrid (base)",
            **_metrics(base_rows),
        },
        "Hybrid + subsection boost": {
            "source": str(doc_root),
            "method": "Hybrid + subsection boost",
            **_metrics(boost_rows),
        },
    }


def _write_bundle(
    output_dir: Path,
    legacy: dict[str, dict[str, float | int | str]],
    thesis_rag: dict[str, dict[str, float | int | str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "legacy_table_4_2": legacy,
        "thesis_rag_table_4_2_equivalent": thesis_rag,
        "differences": {
            method: {
                "page_hit_at_1": float(thesis_rag[method]["page_hit_at_1"]) - float(legacy[method]["page_hit_at_1"]),
                "page_hit_at_3": float(thesis_rag[method]["page_hit_at_3"]) - float(legacy[method]["page_hit_at_3"]),
                "mrr_at_10": float(thesis_rag[method]["mrr_at_10"]) - float(legacy[method]["mrr_at_10"]),
                "queries_evaluated": int(thesis_rag[method]["queries_evaluated"]) - int(legacy[method]["queries_evaluated"]),
            }
            for method in legacy
        },
    }
    (output_dir / "table_4_2_comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fieldnames = ["system", "method", "source", "queries_evaluated", "page_hit_at_1", "page_hit_at_3", "mrr_at_10"]
    rows = []
    for method in legacy:
        rows.append({"system": "legacy_table_4_2", **legacy[method]})
        rows.append({"system": "thesis_rag_equivalent", **thesis_rag[method]})
        rows.append(
            {
                "system": "difference_new_minus_legacy",
                "method": method,
                "source": "",
                **payload["differences"][method],
            }
        )
    with (output_dir / "table_4_2_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Table 4.2 Comparison",
        "",
        "| Method | Legacy Hit@1 | New Hit@1 | Legacy Hit@3 | New Hit@3 | Legacy MRR@10 | New MRR@10 | Queries |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in legacy:
        lines.append(
            f"| {method} | {float(legacy[method]['page_hit_at_1']):.4f} | {float(thesis_rag[method]['page_hit_at_1']):.4f} | "
            f"{float(legacy[method]['page_hit_at_3']):.4f} | {float(thesis_rag[method]['page_hit_at_3']):.4f} | "
            f"{float(legacy[method]['mrr_at_10']):.4f} | {float(thesis_rag[method]['mrr_at_10']):.4f} | "
            f"{int(thesis_rag[method]['queries_evaluated'])} |"
        )
    lines.extend(
        [
            "",
            "## Differences (new - legacy)",
            "",
            "| Method | Hit@1 Δ | Hit@3 Δ | MRR@10 Δ |",
            "|---|---:|---:|---:|",
        ]
    )
    for method in legacy:
        diff = payload["differences"][method]
        lines.append(
            f"| {method} | {diff['page_hit_at_1']:+.4f} | {diff['page_hit_at_3']:+.4f} | {diff['mrr_at_10']:+.4f} |"
        )
    (output_dir / "table_4_2_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    legacy = _read_legacy_table(args.legacy_csv)
    thesis_rag = _build_thesis_rag_table(args.thesis_rag_root, args.pipeline_config)
    _write_bundle(args.output_dir, legacy, thesis_rag)
    print(args.output_dir)


if __name__ == "__main__":
    main()
