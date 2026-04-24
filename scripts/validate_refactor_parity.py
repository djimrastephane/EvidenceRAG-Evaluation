from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
SCRIPTS_PATH = REPO_ROOT / "scripts"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))

from retrieval_eval_bm25 import BM25Index, tokenize
from thesis_rag.evaluator import evaluate_page_hits
from thesis_rag.fusion import reciprocal_rank_fusion
from thesis_rag.loader import extract_page_structures
from thesis_rag.preprocessing import build_chunk_records, build_page_records
from thesis_rag.ranking import chunk_hits_to_page_hits
from thesis_rag.retrieval_dense import dense_retrieve
from thesis_rag.schemas import ChunkRecord, DocumentRecord, OCRConfig, QueryRecord, RetrievalHit
from thesis_rag.utils import now_utc_iso


DOC_ID = "Grampian-2022-2023"
PDF_PATH = REPO_ROOT / "Data" / "Annual Accounts NHS Grampian" / "Preliminary_Test" / f"{DOC_ID}.pdf"
LEGACY_DIR = REPO_ROOT / "data_processed" / DOC_ID
REPORT_ROOT = REPO_ROOT / "runs" / "parity_validation"
REPORT_DIR = REPORT_ROOT / f"{now_utc_iso().replace(':', '-')}_{DOC_ID}"
QUERY_LIMIT = 10
MODEL_PATH = REPO_ROOT / "models" / "all-MiniLM-L6-v2"
MODEL = None


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=False)
    legacy_queries = load_benchmark_queries()
    (REPORT_DIR / "benchmark_queries.json").write_text(
        json.dumps({"doc_id": DOC_ID, "query_limit": QUERY_LIMIT, "queries": [asdict(q) for q in legacy_queries]}, indent=2),
        encoding="utf-8",
    )

    legacy = load_legacy_artifacts(legacy_queries)
    refactor = run_refactored_artifacts(legacy_queries)
    report = build_report(legacy, refactor, legacy_queries)

    (REPORT_DIR / "parity_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (REPORT_DIR / "parity_report.md").write_text(render_markdown(report), encoding="utf-8")
    print(REPORT_DIR)


def load_benchmark_queries() -> list[QueryRecord]:
    payload = json.loads((LEGACY_DIR / "eval_set.json").read_text(encoding="utf-8"))
    rows = payload["queries"][:QUERY_LIMIT]
    return [
        QueryRecord(
            query_id=row["query_id"],
            query_text=row["question"],
            doc_id=row["doc_id"],
            gold_pages=list(row["expected_pages"]),
            expected_answer=row.get("expected_answer"),
            difficulty=row.get("difficulty"),
            evidence_layout=row.get("evidence_layout"),
        )
        for row in rows
    ]


def load_legacy_artifacts(queries: list[QueryRecord]) -> dict[str, Any]:
    pages = pd.read_parquet(LEGACY_DIR / "pages.parquet")
    chunks = pd.read_parquet(LEGACY_DIR / "chunks.parquet")
    chunk_meta = pd.read_parquet(LEGACY_DIR / "chunk_meta.parquet")
    embeddings = np.load(LEGACY_DIR / "embeddings.npy")
    ocr_pages = pd.read_csv(LEGACY_DIR / "ocr_pages.csv") if (LEGACY_DIR / "ocr_pages.csv").exists() else pd.DataFrame()

    legacy_dense = json.loads((LEGACY_DIR / "retrieval_results.json").read_text(encoding="utf-8"))["results"]
    legacy_bm25 = json.loads((LEGACY_DIR / "retrieval_results_bm25.json").read_text(encoding="utf-8"))["results"]
    legacy_hybrid = json.loads((LEGACY_DIR / "retrieval_results_hybrid.json").read_text(encoding="utf-8"))["results"]

    query_ids = {query.query_id for query in queries}
    dense_map = {row["query_id"]: row for row in legacy_dense if row["query_id"] in query_ids}
    bm25_map = {row["query_id"]: row for row in legacy_bm25 if row["query_id"] in query_ids}
    hybrid_map = {row["query_id"]: row for row in legacy_hybrid if row["query_id"] in query_ids}

    return {
        "pages": pages,
        "chunks": chunks,
        "chunk_meta": chunk_meta,
        "embeddings": embeddings,
        "ocr_pages": ocr_pages,
        "dense_results": dense_map,
        "bm25_results": bm25_map,
        "hybrid_results": hybrid_map,
    }


def run_refactored_artifacts(queries: list[QueryRecord]) -> dict[str, Any]:
    page_structures = extract_page_structures(DocumentRecord(doc_id=DOC_ID, pdf_path=str(PDF_PATH)))
    pages = build_page_records(DOC_ID, page_structures, OCRConfig(enabled=True))
    chunks = build_chunk_records(
        DOC_ID,
        pages,
        config=type("ChunkCfg", (), {"chunk_size_tokens": 224, "chunk_overlap_tokens": 56, "min_chunk_words": 20})(),
        source_pdf_path=PDF_PATH,
    )

    vectors = encode_texts([chunk.text for chunk in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    query_vectors = encode_texts([query.query_text for query in queries])

    dense_chunk_hits = dense_retrieve(index, chunks, queries, query_vectors, top_k=10)
    dense_page_hits = chunk_hits_to_page_hits(dense_chunk_hits, "dense_pages", chunk_limit=10)

    bm25 = BM25Index([tokenize(chunk.text) for chunk in chunks], k1=1.5, b=0.75)
    sparse_chunk_hits = build_bm25_hits(bm25, chunks, queries, top_k=10)
    sparse_page_hits = chunk_hits_to_page_hits(sparse_chunk_hits, "bm25_pages", chunk_limit=10)

    fused_chunk_hits = reciprocal_rank_fusion(
        {"dense": dense_chunk_hits, "bm25": sparse_chunk_hits},
        rrf_k=20,
        weights={"dense": 0.5, "bm25": 2.0},
    )
    fused_page_hits = chunk_hits_to_page_hits(fused_chunk_hits, "hybrid_pages", chunk_limit=10)
    evaluation = evaluate_page_hits(queries, fused_page_hits)

    return {
        "pages": pages,
        "chunks": chunks,
        "vectors": vectors,
        "dense_page_hits": dense_page_hits,
        "bm25_page_hits": sparse_page_hits,
        "hybrid_page_hits": fused_page_hits,
        "evaluation": evaluation,
    }


def get_model() -> SentenceTransformer:
    global MODEL
    if MODEL is None:
        MODEL = SentenceTransformer(str(MODEL_PATH), device="cpu", cache_folder=str(REPO_ROOT / "models"))
    return MODEL


def encode_texts(texts: list[str]) -> np.ndarray:
    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if vectors.shape[1] != 384:
        raise ValueError(f"Unexpected embedding dimension: {vectors.shape}")
    return vectors.astype(np.float32)


def build_bm25_hits(bm25: BM25Index, chunks: list[ChunkRecord], queries: list[QueryRecord], top_k: int) -> list[RetrievalHit]:
    hits: list[RetrievalHit] = []
    for query in queries:
        scores = bm25.score_query(tokenize(query.query_text))
        ordered = sorted(enumerate(scores), key=lambda item: (-float(item[1]), chunks[item[0]].chunk_id))[:top_k]
        for rank, (idx, score) in enumerate(ordered, start=1):
            chunk = chunks[idx]
            hits.append(
                RetrievalHit(
                    query_id=query.query_id,
                    query_text=query.query_text,
                    rank=rank,
                    score=float(score),
                    retrieval_method="bm25",
                    doc_id=chunk.doc_id,
                    page_number=chunk.page_number,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                )
            )
    return hits


def build_report(legacy: dict[str, Any], refactor: dict[str, Any], queries: list[QueryRecord]) -> dict[str, Any]:
    legacy_page_count = int(len(legacy["pages"]))
    legacy_clean_pages = int((legacy["pages"]["clean_text"].fillna("").str.len() > 0).sum())
    ref_pages = refactor["pages"]
    ref_page_count = len(ref_pages)
    ref_clean_pages = sum(1 for page in ref_pages if page.clean_text.strip())

    legacy_chunk_count = int(len(legacy["chunks"]))
    ref_chunk_count = len(refactor["chunks"])

    legacy_mapping = Counter(
        tuple(_to_pages_tuple(row.get("pages"), row.get("page_start"), row.get("page_end")))
        for _, row in legacy["chunks"].iterrows()
    )
    ref_mapping = Counter((chunk.page_number,) for chunk in refactor["chunks"])

    dense_diffs = compare_rankings(queries, legacy["dense_results"], refactor["dense_page_hits"], method_key="dense")
    bm25_diffs = compare_rankings(queries, legacy["bm25_results"], refactor["bm25_page_hits"], method_key="bm25")
    hybrid_diffs = compare_rankings(queries, legacy["hybrid_results"], refactor["hybrid_page_hits"], method_key="hybrid")

    legacy_metrics = compute_legacy_metrics(queries, legacy["hybrid_results"])
    ref_metrics = compute_refactor_metrics(refactor["evaluation"])

    mismatches: list[dict[str, Any]] = []
    if legacy_clean_pages != ref_clean_pages or legacy_page_count != ref_page_count:
        mismatches.append(
            mismatch_entry(
                "cleaned_page_counts",
                {"legacy_page_records": legacy_page_count, "legacy_non_empty_clean_pages": legacy_clean_pages},
                {"refactor_page_records": ref_page_count, "refactor_non_empty_clean_pages": ref_clean_pages},
                "Preprocessing contracts differ; the refactor keeps one page record per source page but uses a simplified cleaning flow.",
                "regression" if legacy_clean_pages != ref_clean_pages else "acceptable",
            )
        )
    if int(len(legacy["ocr_pages"])) != sum(1 for page in ref_pages if page.ocr_used):
        mismatches.append(
            mismatch_entry(
                "ocr_page_counts",
                {"legacy_ocr_pages": int(len(legacy["ocr_pages"]))},
                {"refactor_ocr_pages": int(sum(1 for page in ref_pages if page.ocr_used))},
                "The refactor currently flags OCR need heuristically instead of preserving the legacy raw OCR decision path.",
                "regression",
            )
        )
    if legacy_chunk_count != ref_chunk_count:
        mismatches.append(
            mismatch_entry(
                "chunk_counts",
                {"legacy_chunks": legacy_chunk_count},
                {"refactor_chunks": ref_chunk_count},
                "The refactor does not yet reproduce segment-aware cross-page chunks, section-aware text assembly, or table chunk serialization.",
                "regression",
            )
        )
    if legacy_mapping != ref_mapping:
        mismatches.append(
            mismatch_entry(
                "chunk_to_page_mappings",
                {"legacy_top_spans": legacy_mapping.most_common(10)},
                {"refactor_top_spans": ref_mapping.most_common(10)},
                "Legacy chunks can span multiple pages and include table/cross-page chunks; refactor chunks are single-page only.",
                "regression",
            )
        )
    if legacy["embeddings"].shape != refactor["vectors"].shape:
        mismatches.append(
            mismatch_entry(
                "embedding_counts_and_dimensions",
                {"legacy_shape": list(legacy["embeddings"].shape)},
                {"refactor_shape": list(refactor["vectors"].shape)},
                "Embedding count drift follows chunk count drift. Dimension parity is expected because both use MiniLM-L6-v2.",
                "regression",
            )
        )
    mismatches.extend(dense_diffs)
    mismatches.extend(bm25_diffs)
    mismatches.extend(hybrid_diffs)
    if legacy_metrics != ref_metrics:
        mismatches.append(
            mismatch_entry(
                "page_level_metrics",
                {"legacy": legacy_metrics},
                {"refactor": ref_metrics},
                "Metric drift is downstream of preprocessing, chunking, and ranking differences.",
                "regression" if ref_metrics["mrr@10"] < legacy_metrics["mrr@10"] else "acceptable",
            )
        )

    return {
        "benchmark": {
            "doc_id": DOC_ID,
            "pdf_path": str(PDF_PATH),
            "legacy_dir": str(LEGACY_DIR),
            "query_limit": QUERY_LIMIT,
            "query_ids": [query.query_id for query in queries],
        },
        "summary": {
            "legacy_page_records": legacy_page_count,
            "refactor_page_records": ref_page_count,
            "legacy_ocr_pages": int(len(legacy["ocr_pages"])),
            "refactor_ocr_pages": int(sum(1 for page in ref_pages if page.ocr_used)),
            "legacy_chunk_count": legacy_chunk_count,
            "refactor_chunk_count": ref_chunk_count,
            "legacy_embedding_shape": list(legacy["embeddings"].shape),
            "refactor_embedding_shape": list(refactor["vectors"].shape),
            "legacy_metrics": legacy_metrics,
            "refactor_metrics": ref_metrics,
            "mismatch_count": len(mismatches),
        },
        "mismatches": mismatches,
    }


def compare_rankings(
    queries: list[QueryRecord],
    legacy_map: dict[str, Any],
    refactor_hits: list[RetrievalHit],
    *,
    method_key: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = {}
    for query in queries:
        grouped[query.query_id] = []
    for hit in refactor_hits:
        if hit.query_id in grouped and len(grouped[hit.query_id]) < 10:
            grouped[hit.query_id].append(hit.page_number)

    out: list[dict[str, Any]] = []
    for query in queries:
        legacy_pages = list(legacy_map[query.query_id]["per_k"]["10"]["retrieved_pages_ranked"])
        ref_pages = grouped.get(query.query_id, [])
        if legacy_pages != ref_pages:
            out.append(
                mismatch_entry(
                    f"{method_key}_top10::{query.query_id}",
                    {"legacy_top10_pages": legacy_pages},
                    {"refactor_top10_pages": ref_pages},
                    likely_cause_for_method(method_key),
                    "regression",
                )
            )
    return out


def likely_cause_for_method(method_key: str) -> str:
    if method_key == "dense":
        return "Dense ranking changed because the refactor builds a different chunk set and embeds raw chunk text without legacy section/subsection augmentation."
    if method_key == "bm25":
        return "BM25 ranking changed because the refactor chunk corpus differs and does not yet include legacy table serialization or cross-page chunks."
    return "Hybrid page ranking changed because both dense and BM25 candidates drifted, then fused with different page candidate sets."


def compute_legacy_metrics(queries: list[QueryRecord], hybrid_map: dict[str, Any]) -> dict[str, float]:
    hits_at_1 = 0
    hits_at_3 = 0
    rr_total = 0.0
    for query in queries:
        ranked_pages = list(hybrid_map[query.query_id]["per_k"]["10"]["retrieved_pages_ranked"])
        gold = set(query.gold_pages)
        hits_at_1 += int(any(page in gold for page in ranked_pages[:1]))
        hits_at_3 += int(any(page in gold for page in ranked_pages[:3]))
        rr_total += reciprocal_rank_from_pages(ranked_pages[:10], gold)
    denom = max(len(queries), 1)
    return {"hit@1": hits_at_1 / denom, "hit@3": hits_at_3 / denom, "mrr@10": rr_total / denom}


def compute_refactor_metrics(results) -> dict[str, float]:
    denom = max(len(results), 1)
    return {
        "hit@1": sum(int(item.hit_at_1) for item in results) / denom,
        "hit@3": sum(int(item.hit_at_3) for item in results) / denom,
        "mrr@10": sum(item.reciprocal_rank for item in results) / denom,
    }


def reciprocal_rank_from_pages(ranked_pages: list[int], gold_pages: set[int]) -> float:
    for index, page in enumerate(ranked_pages, start=1):
        if page in gold_pages:
            return 1.0 / index
    return 0.0


def mismatch_entry(name: str, legacy_output: Any, refactored_output: Any, likely_cause: str, judgement: str) -> dict[str, Any]:
    return {
        "name": name,
        "legacy_output": legacy_output,
        "refactored_output": refactored_output,
        "likely_cause": likely_cause,
        "judgement": judgement,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Refactor Parity Report",
        "",
        f"- Document: `{report['benchmark']['doc_id']}`",
        f"- Query subset ({report['benchmark']['query_limit']}): `{', '.join(report['benchmark']['query_ids'])}`",
        "",
        "## Summary",
        "",
        f"- Legacy page records: {report['summary']['legacy_page_records']}",
        f"- Refactor page records: {report['summary']['refactor_page_records']}",
        f"- Legacy OCR pages: {report['summary']['legacy_ocr_pages']}",
        f"- Refactor OCR pages: {report['summary']['refactor_ocr_pages']}",
        f"- Legacy chunk count: {report['summary']['legacy_chunk_count']}",
        f"- Refactor chunk count: {report['summary']['refactor_chunk_count']}",
        f"- Legacy embedding shape: {report['summary']['legacy_embedding_shape']}",
        f"- Refactor embedding shape: {report['summary']['refactor_embedding_shape']}",
        f"- Legacy metrics: {report['summary']['legacy_metrics']}",
        f"- Refactor metrics: {report['summary']['refactor_metrics']}",
        "",
        "## Mismatches",
        "",
    ]
    for item in report["mismatches"]:
        lines.extend(
            [
                f"### {item['name']}",
                "",
                f"- Legacy output: `{item['legacy_output']}`",
                f"- Refactored output: `{item['refactored_output']}`",
                f"- Likely cause: {item['likely_cause']}",
                f"- Assessment: `{item['judgement']}`",
                "",
            ]
        )
    return "\n".join(lines)


def _to_pages_tuple(pages_value: Any, page_start: Any, page_end: Any) -> tuple[int, ...]:
    if hasattr(pages_value, "tolist"):
        raw = pages_value.tolist()
        return tuple(int(x) for x in raw)
    if isinstance(pages_value, (list, tuple)):
        return tuple(int(x) for x in pages_value)
    out: list[int] = []
    if page_start is not None and not pd.isna(page_start):
        out.append(int(page_start))
    if page_end is not None and not pd.isna(page_end):
        end = int(page_end)
        if end not in out:
            out.append(end)
    return tuple(out)


if __name__ == "__main__":
    main()
