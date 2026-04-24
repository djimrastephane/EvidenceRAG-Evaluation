"""
Run repeated reproducibility checks for the current thesis_rag pipeline.

This mirrors the legacy appendix protocol:
1. keep the processed five-document corpus fixed,
2. run repeated retrieval passes under identical settings,
3. canonicalize the per-query ranked outputs, and
4. compare SHA-256 hashes across runs.

The current-pipeline variant operates directly on thesis_rag artifacts instead
of the legacy SearchService stack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from thesis_rag.artifacts import load_queries
from thesis_rag.config import load_config
from thesis_rag.embedding import embed_queries
from thesis_rag.ranking import chunk_hits_to_page_hits
from thesis_rag.retrieval_dense import search_faiss_stably
from thesis_rag.retrieval_hybrid import hybrid_retrieve_legacy_style
from thesis_rag.retrieval_sparse import build_bm25
from thesis_rag.schemas import ChunkRecord
from thesis_rag.utils import resolve_device, set_global_determinism


DEFAULT_DOC_IDS = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether thesis_rag retrieval outputs are reproducible across repeated runs."
    )
    parser.add_argument(
        "--data-root",
        default="data_variants/tiktoken_5docs",
        help="Root containing the fixed five-document thesis_rag artifacts.",
    )
    parser.add_argument(
        "--config",
        default="configs/thesis_rag.yaml",
        help="Base thesis_rag config used for query embedding and retrieval parameters.",
    )
    parser.add_argument(
        "--model-path",
        default="models/all-MiniLM-L6-v2",
        help="Local embedding model path to use for deterministic query embedding.",
    )
    parser.add_argument("--k", type=int, default=10, help="Top-k ranked pages to canonicalize.")
    parser.add_argument("--runs", type=int, default=30, help="Number of repeated runs to execute.")
    parser.add_argument(
        "--round-score-digits",
        type=int,
        default=10,
        help="Decimal digits to retain when canonicalizing floating-point scores.",
    )
    parser.add_argument(
        "--out-json",
        default="results/reproducibility/current_pipeline_grampian_5docs_repro.json",
        help="Where to write the final reproducibility report.",
    )
    parser.add_argument(
        "--worker-out",
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _stable_float(value: Any, digits: int) -> float | None:
    if value is None:
        return None
    return round(float(value), int(digits))


def _normalise_pages(value: Any, fallback_page: int) -> list[int]:
    if value is None:
        return [int(fallback_page)]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or not value:
        return [int(fallback_page)]
    pages: list[int] = []
    for item in value:
        if isinstance(item, dict) and "element" in item:
            pages.append(int(item["element"]))
        else:
            pages.append(int(item))
    return pages or [int(fallback_page)]


def _load_legacy_shaped_chunks(path: Path) -> list[ChunkRecord]:
    frame = pd.read_parquet(path)
    records: list[ChunkRecord] = []
    for row in frame.to_dict(orient="records"):
        chunk_id = str(row["chunk_id"])
        page_number = int(row.get("page_start") or row.get("page") or 0)
        suffix = chunk_id.rsplit("_", 1)[-1] if "_" in chunk_id else ""
        chunk_index = int(suffix) if suffix.isdigit() else 0
        records.append(
            ChunkRecord(
                chunk_id=chunk_id,
                doc_id=str(row["doc_id"]),
                page_number=page_number,
                chunk_index=chunk_index,
                text=str(row.get("chunk_text") or ""),
                token_count=int(row.get("chunk_tokens") or 0),
                word_count=int(row.get("word_count") or 0),
                chunk_id_global=str(row.get("chunk_id_global") or ""),
                page_start=int(row.get("page_start") or page_number),
                page_end=int(row.get("page_end") or page_number),
                pages=_normalise_pages(row.get("pages"), page_number),
                part=str(row.get("part") or ""),
                section_title=str(row.get("section_title") or ""),
                subsection_title=str(row.get("subsection_title") or ""),
                is_table=bool(row.get("is_table", False)),
                table_type=str(row.get("table_type")) if row.get("table_type") is not None else None,
            )
        )
    return records


def _canonical_payload(args: argparse.Namespace) -> dict[str, Any]:
    data_root = (REPO_ROOT / args.data_root).resolve()
    config = load_config(REPO_ROOT / args.config)
    config.embedding.model_name = str((REPO_ROOT / args.model_path).resolve())
    set_global_determinism(int(config.runtime.random_seed), bool(config.runtime.deterministic_torch))

    device = resolve_device(config.runtime.device)
    max_k_search = max(100, int(args.k))

    import faiss

    per_query: list[dict[str, Any]] = []
    artifact_signatures: dict[str, Any] = {}

    for doc_id in DEFAULT_DOC_IDS:
        doc_dir = data_root / doc_id
        chunks = _load_legacy_shaped_chunks(doc_dir / "chunks.parquet")
        queries = load_queries(doc_dir / "eval_set.json")
        artifact_signatures[doc_id] = {
            "chunks": int(len(chunks)),
            "queries": int(len(queries)),
            "embeddings_shape": list(np.load(doc_dir / "embeddings.npy", mmap_mode="r").shape),
        }

        index = faiss.read_index(str(doc_dir / "faiss.index"))
        bm25 = build_bm25(chunks, config.bm25)
        query_vectors = embed_queries(
            [query.query_text for query in queries],
            config.embedding,
            device=device,
            cache_dir=str(config.paths.model_cache_dir),
        )
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
        page_hits = chunk_hits_to_page_hits(
            hybrid_chunk_hits,
            "hybrid_pages",
            chunk_limit=int(args.k),
        )
        hits_by_query: dict[str, list[Any]] = {}
        for hit in page_hits:
            hits_by_query.setdefault(hit.query_id, []).append(hit)

        for query in queries:
            query_hits = sorted(
                hits_by_query.get(query.query_id, []),
                key=lambda hit: (int(hit.rank), -float(hit.score), str(hit.doc_id), int(hit.page_number), str(hit.chunk_id or "")),
            )
            per_query.append(
                {
                    "doc_id": str(query.doc_id),
                    "query_id": str(query.query_id),
                    "question": str(query.query_text),
                    "expected_pages": [int(page) for page in list(query.gold_pages)],
                    "results": [
                        {
                            "rank": int(hit.rank),
                            "doc_id": str(hit.doc_id),
                            "page_number": int(hit.page_number),
                            "score": _stable_float(hit.score, int(args.round_score_digits)),
                            "chunk_id": str(hit.chunk_id or ""),
                            "pages": [int(page) for page in list(hit.pages or [])],
                        }
                        for hit in query_hits
                    ],
                }
            )

    per_query.sort(key=lambda row: (str(row["doc_id"]), str(row["query_id"]), str(row["question"])))
    return {
        "config": {
            "data_root": str(data_root),
            "config_path": str((REPO_ROOT / args.config).resolve()),
            "model_path": str((REPO_ROOT / args.model_path).resolve()),
            "doc_ids": list(DEFAULT_DOC_IDS),
            "k": int(args.k),
            "round_score_digits": int(args.round_score_digits),
            "random_seed": int(config.runtime.random_seed),
            "deterministic_torch": bool(config.runtime.deterministic_torch),
            "retrieval": {
                "dense_weight": float(config.retrieval.dense_weight),
                "sparse_weight": float(config.retrieval.sparse_weight),
                "rrf_k": int(config.retrieval.rrf_k),
                "max_k_search": int(max_k_search),
            },
        },
        "artifact_signatures": artifact_signatures,
        "query_count": int(len(per_query)),
        "per_query": per_query,
    }


def _payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_worker(args: argparse.Namespace) -> dict[str, Any]:
    payload = _canonical_payload(args)
    payload["payload_hash"] = _payload_hash(payload)
    return payload


def _run_parent(args: argparse.Namespace) -> dict[str, Any]:
    out_path = (REPO_ROOT / args.out_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("FAISS_NO_AVX2", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    run_hashes: list[str] = []
    run_files: list[str] = []
    baseline_hash: str | None = None
    baseline_path: str | None = None
    mismatch_index: int | None = None
    query_count: int | None = None

    with tempfile.TemporaryDirectory(prefix="thesis_rag_repro_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        for run_idx in range(1, int(args.runs) + 1):
            worker_out = tmp_dir / f"run_{run_idx:03d}.json"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--data-root",
                str(args.data_root),
                "--config",
                str(args.config),
                "--model-path",
                str(args.model_path),
                "--k",
                str(args.k),
                "--round-score-digits",
                str(args.round_score_digits),
                "--worker-out",
                str(worker_out),
            ]
            subprocess.run(cmd, check=True, env=env, cwd=REPO_ROOT)
            payload = json.loads(worker_out.read_text(encoding="utf-8"))
            payload_hash = str(payload["payload_hash"])
            run_hashes.append(payload_hash)
            run_files.append(str(worker_out))
            query_count = int(payload.get("query_count", 0))
            if baseline_hash is None:
                baseline_hash = payload_hash
                baseline_path = str(worker_out)
            elif mismatch_index is None and payload_hash != baseline_hash:
                mismatch_index = run_idx

        unique_hash_count = len(set(run_hashes))
        report = {
            "status": "pass" if unique_hash_count == 1 else "fail",
            "claim_scope": "thesis_rag_hybrid_retrieval",
            "runs": int(args.runs),
            "k": int(args.k),
            "query_count": int(query_count or 0),
            "all_hashes_equal": unique_hash_count == 1,
            "baseline_hash": baseline_hash,
            "unique_hash_count": unique_hash_count,
            "first_mismatch_run": mismatch_index,
            "baseline_run_file": baseline_path,
            "run_hashes": run_hashes,
            "run_files": run_files,
            "notes": [
                "Hashes are computed over canonicalized per-query hybrid page rankings only.",
                "The canonical payload excludes timestamps, filesystem-specific temporary paths, and runtime latency metadata.",
                "Processed document artifacts, query sets, and retrieval parameters are held fixed across all runs.",
            ],
        }
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


def main() -> None:
    args = parse_args()
    if args.worker_out:
        payload = _run_worker(args)
        Path(args.worker_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return

    report = _run_parent(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
