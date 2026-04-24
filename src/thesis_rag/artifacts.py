from __future__ import annotations

"""Persistence helpers for thesis_rag intermediate and final artifacts.

The pipeline writes most stage outputs to both human-readable and
machine-efficient formats so that experiments are auditable after the fact.
This module centralises those read/write operations to keep file naming and
serialization consistent across preprocessing, indexing, retrieval, and
evaluation stages.
"""

from pathlib import Path
from typing import Iterable

import pandas as pd

from .schemas import ChunkRecord, PageRecord, QueryRecord, RetrievalHit
from .utils import read_json, write_json, write_jsonl


def save_pages(pages: list[PageRecord], out_dir: Path) -> Path:
    """Write page records to JSONL and Parquet inside a document output folder."""
    path = out_dir / "pages.jsonl"
    write_jsonl(path, [page.to_dict() for page in pages])
    pd.DataFrame([page.to_dict() for page in pages]).to_parquet(out_dir / "pages.parquet", index=False)
    return path


def save_chunks(chunks: list[ChunkRecord], out_dir: Path) -> Path:
    """Write chunk records to JSONL and Parquet inside a document output folder."""
    path = out_dir / "chunks.jsonl"
    records = [chunk.to_dict() for chunk in chunks]
    write_jsonl(path, records)
    pd.DataFrame(records).to_parquet(out_dir / "chunks.parquet", index=False)
    return path


def load_chunks(path: Path) -> list[ChunkRecord]:
    """Load chunk records from a Parquet manifest produced by the pipeline."""
    frame = pd.read_parquet(path)
    return [ChunkRecord(**row) for row in frame.to_dict(orient="records")]


def load_pages(path: Path) -> list[PageRecord]:
    """Load page records from a Parquet manifest produced by the pipeline."""
    frame = pd.read_parquet(path)
    return [PageRecord(**row) for row in frame.to_dict(orient="records")]


def save_queries(queries: list[QueryRecord], out_path: Path) -> None:
    """Persist a query set as JSON for reproducible retrieval experiments."""
    write_json(out_path, {"queries": [query.to_dict() for query in queries]})


def load_queries(path: Path) -> list[QueryRecord]:
    """Load and normalise query records from the stored evaluation JSON."""
    payload = read_json(path)
    rows = payload["queries"] if isinstance(payload, dict) and "queries" in payload else payload
    queries: list[QueryRecord] = []
    for row in rows:
        query_text = row.get("query_text") or row.get("question") or row.get("query")
        if not query_text:
            raise ValueError(f"Query record {row.get('query_id')!r} has no query_text/question/query field")
        queries.append(
            QueryRecord(
                query_id=row["query_id"],
                query_text=str(query_text).strip(),
                doc_id=row["doc_id"],
                gold_pages=list(row.get("gold_pages") or row.get("expected_pages") or []),
                expected_answer=row.get("expected_answer"),
                difficulty=row.get("difficulty"),
                evidence_layout=row.get("evidence_layout"),
                expected_section=row.get("expected_section"),
                expected_subsection=row.get("expected_subsection"),
            )
        )
    return queries


def save_hits(hits: Iterable[RetrievalHit], out_path: Path) -> None:
    """Persist retrieval hits to JSONL plus a CSV companion for easy inspection."""
    records = [hit.to_dict() for hit in hits]
    write_jsonl(out_path, records)
    pd.DataFrame(records).to_csv(out_path.with_suffix(".csv"), index=False)


def save_manifest(manifest: dict, out_path: Path) -> None:
    """Write run or stage metadata manifests as JSON."""
    write_json(out_path, manifest)
