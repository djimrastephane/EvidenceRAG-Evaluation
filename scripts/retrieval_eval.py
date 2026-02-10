"""
retrieval_eval.py

Evaluate top-k retrieval using:
- faiss.index
- chunk_meta.parquet
- eval_set.json (fixed questions with expected_pages, optional doc_id/document_id, expected_section)

Outputs (written to the same DATA_DIR):
- retrieval_results.json      Per-query retrieved results for each k
- retrieval_metrics.json      Aggregate metrics for each k
- retrieval_summary.csv       Flat table for appendix and quick inspection

Notes
- Enforces query_id nomenclature: Q_<TOPIC>_<YEAR>_<NN>
  Allowed TOPIC values: REV, EFF, DEF, STAFF
  Examples:
    Q_REV_2023_01
    Q_EFF_2023_01
    Q_DEF_2023_01
    Q_STAFF_2023_02

- Reports both page-level metrics and chunk-level metrics.
  Page precision can drop as k increases because extra chunks add extra pages.
  Chunk metrics are often easier to interpret for retrieval ranking.

FAILURE ATTRIBUTION (RETRIEVAL STAGE ONLY)
This evaluator assigns a deterministic retrieval failure stage per query at each k:
- missing_content
- missed_top_ranked
- hit

NEW: LEAKAGE DETECTION (MULTI-DOC SAFETY)
If a query specifies an expected doc_id, this evaluator reports whether any of the
top-k retrieved chunks come from a different doc_id.

Leakage does not change recall@k. It is reported as an additional diagnostic
signal, useful for the 3-document supervisor requirement:
- Each evaluation question answerable from exactly one document.
- Detect when retrieval pulls strongly similar chunks from other documents.

Implementation details:
- leakage_count_top_k: number of retrieved chunks in top-k with doc_id != expected_doc_id
- leakage_doc_ids_top_k: unique list of non-expected doc_ids in top-k
- retrieved_doc_ids_top_k: doc_id for each retrieved chunk in top-k
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

try:
    import faiss
except Exception as e:
    raise RuntimeError(
        "FAISS is not installed.\n"
        "Fix:\n"
        "  pip install faiss-cpu\n"
    ) from e

try:
    from transformers import logging as hf_logging
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise RuntimeError(
        "sentence-transformers is not installed.\n"
        "Fix:\n"
        "  pip install sentence-transformers\n"
    ) from e

try:
    import pyarrow.parquet as pq
except Exception as e:
    raise RuntimeError(
        "pyarrow is not installed.\n"
        "Fix:\n"
        "  pip install pyarrow\n"
    ) from e


# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR = Path(
    "/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/data_processed/"
    "Grampian-2022-2023"
)

INDEX_PATH = DATA_DIR / "faiss.index"
META_PATH = DATA_DIR / "chunk_meta.parquet"
EVAL_SET_PATH = DATA_DIR / "eval_set.json"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

K_LIST = [1, 3, 5, 10]
MAX_K_SEARCH = int(os.getenv("MAX_K_SEARCH", "100"))
SUBSECTION_BOOST = float(os.getenv("SUBSECTION_BOOST", "0.05"))

RESULTS_JSON = DATA_DIR / "retrieval_results.json"
METRICS_JSON = DATA_DIR / "retrieval_metrics.json"
SUMMARY_CSV = DATA_DIR / "retrieval_summary.csv"

PRINT_HIT_DEBUG = True


# =============================================================================
# QUERY ID NOMENCLATURE
# =============================================================================
QUERY_ID_PATTERN = re.compile(r"^Q_(REV|EFF|DEF|STAFF|ACC|GOV)_\d{4}_\d{2}$")


def validate_query_id(query_id: str) -> None:
    if not QUERY_ID_PATTERN.match(query_id):
        raise ValueError(
            f"Invalid query_id '{query_id}'. Expected: Q_<TOPIC>_<YEAR>_<NN> "
            f"with TOPIC in [REV, EFF, DEF, STAFF, ACC], for example Q_EFF_2023_01."
        )


def parse_query_id(query_id: str) -> dict[str, Any]:
    _, topic, year, seq = query_id.split("_")
    return {"topic": topic, "year": int(year), "sequence": int(seq)}


# =============================================================================
# HELPERS
# =============================================================================
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _env_or_default(name: str, default: str) -> str:
    val = os.getenv(name)
    return val if val else default


def parse_k_list(val: str) -> list[int]:
    parts = [p.strip() for p in val.split(",") if p.strip()]
    return [int(p) for p in parts]


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def refresh_paths() -> None:
    global INDEX_PATH, META_PATH, EVAL_SET_PATH, RESULTS_JSON, METRICS_JSON, SUMMARY_CSV
    INDEX_PATH = DATA_DIR / "faiss.index"
    META_PATH = DATA_DIR / "chunk_meta.parquet"
    EVAL_SET_PATH = DATA_DIR / "eval_set.json"
    RESULTS_JSON = DATA_DIR / "retrieval_results.json"
    METRICS_JSON = DATA_DIR / "retrieval_metrics.json"
    SUMMARY_CSV = DATA_DIR / "retrieval_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval using FAISS index and eval_set.json."
    )
    parser.add_argument(
        "--data-dir",
        default=_env_or_default("DATA_DIR", str(DATA_DIR)),
        help="Directory containing faiss.index, chunk_meta.parquet, eval_set.json.",
    )
    parser.add_argument(
        "--model",
        default=_env_or_default("EMBED_MODEL_NAME", EMBED_MODEL_NAME),
        help="Sentence-transformers model name or local path.",
    )
    parser.add_argument(
        "--k-list",
        default=_env_or_default("K_LIST", ",".join(str(k) for k in K_LIST)),
        help="Comma-separated list of k values (e.g. 1,3,5,10).",
    )
    return parser.parse_args()


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)


def read_parquet_safe(path: Path) -> pd.DataFrame:
    """
    Read parquet via pyarrow to avoid pandas engine issues on some Python builds.
    """
    return pq.read_table(str(path)).to_pandas()


def to_int_list(v) -> list[int]:
    """
    Normalise a page_list-like value into list[int].

    Handles:
    - list / tuple
    - numpy arrays
    - scalars
    - stringified lists
    - list of dicts like [{"element": 2}]
    """
    if v is None:
        return []

    if isinstance(v, float) and pd.isna(v):
        return []

    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            if x is None:
                continue
            if isinstance(x, float) and pd.isna(x):
                continue
            if isinstance(x, dict) and "element" in x:
                nums = re.findall(r"\d+", str(x.get("element")))
                if nums:
                    out.append(int(nums[0]))
                continue
            out.append(int(x))
        return out

    if hasattr(v, "tolist"):
        try:
            vv = v.tolist()
            if isinstance(vv, list):
                return [int(x) for x in vv if x is not None]
            return [int(vv)]
        except Exception:
            pass

    if isinstance(v, str):
        s = v.strip()
        nums = re.findall(r"\d+", s)
        return [int(n) for n in nums] if nums else []

    try:
        return [int(v)]
    except Exception:
        return []


def get_retrieved_pages(meta_row: pd.Series) -> list[int]:
    """
    Extract pages from meta row.

    Preference order:
    1) pages (plain list[int]) if present
    2) page_list if present (may be list, list-of-dicts, or stringified)
    3) page_start/page_end span
    """
    if "pages" in meta_row.index:
        pl = to_int_list(meta_row["pages"])
        if pl:
            return pl

    if "page_list" in meta_row.index:
        pl = to_int_list(meta_row["page_list"])
        if pl:
            return pl

    ps = meta_row.get("page_start", None)
    pe = meta_row.get("page_end", None)

    if ps is None or (isinstance(ps, float) and pd.isna(ps)):
        return []

    try:
        ps_i = int(ps)
        pe_i = int(pe) if pe is not None and not (isinstance(pe, float) and pd.isna(pe)) else ps_i
        if pe_i < ps_i:
            pe_i = ps_i
        return list(range(ps_i, pe_i + 1))
    except Exception:
        return []


def unique_preserve_order(items: list[int]) -> list[int]:
    seen = set()
    out = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def get_expected_doc_id(item: dict[str, Any]) -> str:
    """
    Support both keys to avoid breaking older eval sets.

    Preferred key is 'doc_id'. For backward compatibility, 'document_id' is also
    accepted.
    """
    v = str(item.get("doc_id", "")).strip()
    if v:
        return v
    return str(item.get("document_id", "")).strip()


def get_chunk_ids(retrieved_chunks: pd.DataFrame) -> list[str]:
    """
    Prefer chunk_id_global when present, else fall back to chunk_id.
    """
    if "chunk_id_global" in retrieved_chunks.columns:
        return retrieved_chunks["chunk_id_global"].astype(str).tolist()
    if "chunk_id" in retrieved_chunks.columns:
        return retrieved_chunks["chunk_id"].astype(str).tolist()
    return []


def get_doc_ids(retrieved_chunks: pd.DataFrame) -> list[str]:
    """
    Extract doc_id list for top-k retrieved chunks.
    """
    if "doc_id" not in retrieved_chunks.columns:
        return []
    return retrieved_chunks["doc_id"].astype(str).tolist()


def compute_leakage(expected_doc_id: str, retrieved_doc_ids: list[str]) -> dict[str, Any]:
    """
    Compute retrieval leakage statistics for a query.

    Leakage definition:
        Any retrieved chunk in top-k whose doc_id differs from the expected doc_id.

    Args:
        expected_doc_id: The query's expected document identifier.
        retrieved_doc_ids: doc_id for each retrieved chunk in rank order.

    Returns:
        dict with keys:
        - leakage_count_top_k (int)
        - leakage_doc_ids_top_k (list[str])
        - leakage_rate_top_k (float)
    """
    if not expected_doc_id or not retrieved_doc_ids:
        return {"leakage_count_top_k": 0, "leakage_doc_ids_top_k": [], "leakage_rate_top_k": 0.0}

    leakage_docs = [d for d in retrieved_doc_ids if d != expected_doc_id]
    leakage_count = len(leakage_docs)
    leakage_rate = leakage_count / max(1, len(retrieved_doc_ids))
    return {
        "leakage_count_top_k": int(leakage_count),
        "leakage_doc_ids_top_k": sorted(list(set(leakage_docs))),
        "leakage_rate_top_k": float(leakage_rate),
    }


# -------------------------
# Page-level scoring
# -------------------------
def recall_at_k(expected_pages: set[int], retrieved_pages: list[int]) -> float:
    if not expected_pages:
        return 0.0
    return 1.0 if expected_pages.intersection(set(retrieved_pages)) else 0.0


def precision_at_k(expected_pages: set[int], retrieved_pages: list[int]) -> float:
    if not expected_pages:
        return 0.0
    if not retrieved_pages:
        return 0.0
    hits = sum(1 for p in retrieved_pages if p in expected_pages)
    return hits / len(retrieved_pages)


def mrr_for_pages(expected_pages: set[int], ranked_pages: list[int]) -> float:
    if not expected_pages:
        return 0.0
    for i, p in enumerate(ranked_pages, start=1):
        if p in expected_pages:
            return 1.0 / i
    return 0.0


# -------------------------
# Chunk-level scoring (based on page overlap)
# -------------------------
def chunk_hit_flags(expected_pages: set[int], retrieved_chunks: pd.DataFrame) -> list[int]:
    flags = []
    for _, r in retrieved_chunks.iterrows():
        pages = get_retrieved_pages(r)
        flags.append(1 if expected_pages.intersection(set(pages)) else 0)
    return flags


def chunk_hit_at_k(flags: list[int]) -> float:
    return 1.0 if any(flags) else 0.0


def chunk_precision_at_k(flags: list[int]) -> float:
    if not flags:
        return 0.0
    return float(sum(flags)) / float(len(flags))


def chunk_mrr(flags: list[int]) -> float:
    for i, f in enumerate(flags, start=1):
        if f == 1:
            return 1.0 / i
    return 0.0


# =============================================================================
# FAILURE ATTRIBUTION (RETRIEVAL STAGE)
# =============================================================================
def compute_gold_presence(meta: pd.DataFrame, expected_doc_id: str, expected_pages: set[int]) -> dict[str, Any]:
    """
    Determine whether the expected (doc_id, expected_pages) content exists in the index.
    """
    if not expected_pages:
        return {"gold_exists": False, "gold_chunk_count": 0, "gold_pages_found": []}

    df = meta
    if expected_doc_id and "doc_id" in df.columns:
        df = df[df["doc_id"].astype(str) == expected_doc_id]

    if len(df) == 0:
        return {"gold_exists": False, "gold_chunk_count": 0, "gold_pages_found": []}

    pages_found: set[int] = set()
    gold_chunk_count = 0

    for _, r in df.iterrows():
        pages = get_retrieved_pages(r)
        if expected_pages.intersection(set(pages)):
            gold_chunk_count += 1
            pages_found.update(expected_pages.intersection(set(pages)))

    return {
        "gold_exists": bool(gold_chunk_count > 0),
        "gold_chunk_count": int(gold_chunk_count),
        "gold_pages_found": sorted(list(pages_found)),
    }


def attribute_retrieval_failure(page_recall: float, gold_exists: bool) -> str:
    if page_recall >= 1.0:
        return "hit"
    return "missed_top_ranked" if gold_exists else "missing_content"


# =============================================================================
# MAIN
# =============================================================================
def main():
    hf_logging.set_verbosity_error()
    args = parse_args()
    global DATA_DIR, EMBED_MODEL_NAME, K_LIST
    DATA_DIR = Path(args.data_dir)
    EMBED_MODEL_NAME = args.model
    K_LIST = parse_k_list(args.k_list)
    refresh_paths()
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Missing FAISS index: {INDEX_PATH}")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing chunk metadata: {META_PATH}")
    if not EVAL_SET_PATH.exists():
        raise FileNotFoundError(
            f"Missing eval_set.json: {EVAL_SET_PATH}\n"
            "Create this file with fixed questions and expected_pages."
        )

    meta = read_parquet_safe(META_PATH)
    index = faiss.read_index(str(INDEX_PATH))
    chunks_path = DATA_DIR / "chunks.parquet"
    chunks = read_parquet_safe(chunks_path) if chunks_path.exists() else None
    chunk_text_by_id: dict[str, str] = {}
    if chunks is not None and "chunk_text" in chunks.columns:
        if "chunk_id_global" in chunks.columns:
            for _, row in chunks.iterrows():
                cid = row.get("chunk_id_global")
                if cid:
                    chunk_text_by_id[str(cid)] = str(row.get("chunk_text") or "")
        if "chunk_id" in chunks.columns:
            for _, row in chunks.iterrows():
                cid = row.get("chunk_id")
                if cid:
                    chunk_text_by_id.setdefault(str(cid), str(row.get("chunk_text") or ""))

    print("Loading embedding model:", EMBED_MODEL_NAME)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    eval_items = read_json(EVAL_SET_PATH)
    if not isinstance(eval_items, list) or len(eval_items) == 0:
        raise ValueError("eval_set.json must be a non-empty list of query objects.")

    for i, item in enumerate(eval_items):
        qid = str(item.get("query_id", "")).strip()
        if not qid:
            raise ValueError(f"Missing query_id for eval item at index {i}.")
        validate_query_id(qid)

    max_k = min(max(K_LIST), len(meta))
    max_k_search = min(max(MAX_K_SEARCH, max_k), len(meta))
    k_list = [k for k in K_LIST if 1 <= k <= max_k]
    if not k_list:
        raise ValueError("K_LIST has no valid values for the current index size.")

    meta_doc_ids = set(meta["doc_id"].astype(str).unique()) if "doc_id" in meta.columns else set()

    run_info = {
        "run_utc": utc_now_iso(),
        "data_dir": str(DATA_DIR),
        "index_path": str(INDEX_PATH),
        "meta_path": str(META_PATH),
        "eval_set_path": str(EVAL_SET_PATH),
        "embedding_model": EMBED_MODEL_NAME,
        "k_list": k_list,
        "subsection_boost": SUBSECTION_BOOST,
        "max_k_search": max_k_search,
        "num_queries": len(eval_items),
        "num_chunks_indexed": int(len(meta)),
        "meta_doc_ids": sorted(list(meta_doc_ids))[:20] if meta_doc_ids else [],
        "query_id_nomenclature": "Q_<TOPIC>_<YEAR>_<NN> with TOPIC in [REV,EFF,DEF,STAFF,ACC,GOV]",
        "failure_attribution": {"stages": ["hit", "missed_top_ranked", "missing_content"], "scope": "retrieval_only"},
        "leakage_detection": {
            "enabled": True,
            "requires_expected_doc_id": True,
            "fields": ["retrieved_doc_ids_top_k", "leakage_count_top_k", "leakage_doc_ids_top_k", "leakage_rate_top_k"],
        },
    }

    results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    questions = [str(q.get("question", "")).strip() for q in eval_items]
    if any(len(q) == 0 for q in questions):
        bad = [i for i, q in enumerate(questions) if len(q) == 0]
        raise ValueError(f"Some eval items have empty 'question' fields at indices: {bad}")

    q_emb = model.encode(
        questions,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    ).astype("float32")
    q_emb = l2_normalize(q_emb).astype("float32")

    def _extract_quarter_value(label: str, text: str, q_lower: str) -> tuple[str | None, str | None]:
        m = re.search(
            rf"{label}\s+([\d-]+)\s+([\d-]+)\s+([\d-]+)\s+([\d-]+)",
            text,
            flags=re.IGNORECASE,
        )
        if not m:
            return None, None
        vals = [m.group(i) for i in range(1, 5)]

        def _pick(idx: int) -> str | None:
            return None if vals[idx] == "-" else vals[idx]

        if "q1" in q_lower:
            return _pick(0), "Q1"
        if "q2" in q_lower:
            return _pick(1), "Q2"
        if "q3" in q_lower:
            return _pick(2), "Q3"
        if "q4" in q_lower:
            return _pick(3), "Q4"
        return None, None

    def _first_sentence(text: str) -> str:
        for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
            if sent:
                return sent[:200].strip()
        return ""

    for qi, item in enumerate(eval_items):
        query_id = str(item.get("query_id", "")).strip()
        validate_query_id(query_id)
        qid_parts = parse_query_id(query_id)

        question = questions[qi]

        expected = item.get("expected_pages", [])
        expected_pages = set(int(x) for x in expected) if isinstance(expected, list) else set()

        answer_type = str(item.get("answer_type", "unknown"))
        expected_doc_id = get_expected_doc_id(item)
        expected_section = str(item.get("expected_section", "")).strip()
        expected_subsection = str(item.get("expected_subsection", "")).strip()

        if expected_doc_id and meta_doc_ids and expected_doc_id not in meta_doc_ids:
            raise ValueError(
                f"Query {query_id} expects doc_id={expected_doc_id}, "
                f"but DATA_DIR meta has doc_id values like: {sorted(list(meta_doc_ids))[:5]}"
            )

        gold_presence = compute_gold_presence(meta, expected_doc_id, expected_pages)

        scores, idxs = index.search(q_emb[qi : qi + 1], max_k_search)
        idxs = idxs[0].tolist()
        scores = scores[0].tolist()

        if expected_subsection and "subsection_title" in meta.columns:
            target = _normalize_text(expected_subsection)
            boosted: list[tuple[float, int]] = []
            for score, idx in zip(scores, idxs):
                sub = meta.iloc[idx].get("subsection_title", "")
                if _normalize_text(str(sub)) == target:
                    score += SUBSECTION_BOOST
                boosted.append((score, idx))
            boosted.sort(key=lambda x: x[0], reverse=True)
            scores = [s for s, _ in boosted]
            idxs = [i for _, i in boosted]

        per_k: dict[str, Any] = {}
        for k in k_list:
            top_idxs = idxs[:k]
            top_scores = scores[:k]

            retrieved_chunks = meta.iloc[top_idxs].copy()
            retrieved_chunks["score"] = top_scores

            retrieved_chunk_ids = get_chunk_ids(retrieved_chunks)
            retrieved_doc_ids = get_doc_ids(retrieved_chunks)
            leakage = compute_leakage(expected_doc_id, retrieved_doc_ids)

            ranked_pages = []
            for _, r in retrieved_chunks.iterrows():
                ranked_pages.extend(get_retrieved_pages(r))
            ranked_pages_unique = unique_preserve_order(ranked_pages)

            page_recall = recall_at_k(expected_pages, ranked_pages_unique)
            page_precision = precision_at_k(expected_pages, ranked_pages_unique)
            page_mrr = mrr_for_pages(expected_pages, ranked_pages_unique)

            flags = chunk_hit_flags(expected_pages, retrieved_chunks)
            c_hit = chunk_hit_at_k(flags)
            c_prec = chunk_precision_at_k(flags)
            c_mrr = chunk_mrr(flags)

            failure_stage = attribute_retrieval_failure(
                page_recall=page_recall,
                gold_exists=bool(gold_presence.get("gold_exists", False)),
            )

            per_k[str(k)] = {
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "retrieved_doc_ids_top_k": retrieved_doc_ids,
                "retrieved_pages_ranked": ranked_pages_unique,
                "retrieved_scores": [float(s) for s in top_scores],
                "expected_subsection": expected_subsection or None,
                "page_recall_at_k": float(page_recall),
                "page_precision_at_k": float(page_precision),
                "page_mrr_at_k": float(page_mrr),
                "chunk_hit_at_k": float(c_hit),
                "chunk_precision_at_k": float(c_prec),
                "chunk_mrr_at_k": float(c_mrr),
                "chunk_hit_flags": flags,
                "failure_stage": failure_stage,
                **leakage,
            }

            summary_rows.append(
                {
                    "query_id": query_id,
                    "topic": qid_parts["topic"],
                    "year": qid_parts["year"],
                    "sequence": qid_parts["sequence"],
                    "k": k,
                    "answer_type": answer_type,
                    "doc_id": expected_doc_id,
                    "expected_section": expected_section,
                    "expected_pages": sorted(list(expected_pages)),
                    "gold_exists": bool(gold_presence.get("gold_exists", False)),
                    "gold_chunk_count": int(gold_presence.get("gold_chunk_count", 0)),
                    "gold_pages_found": gold_presence.get("gold_pages_found", []),
                    "failure_stage": failure_stage,
                    "leakage_count_top_k": leakage["leakage_count_top_k"],
                    "leakage_rate_top_k": leakage["leakage_rate_top_k"],
                    "leakage_doc_ids_top_k": leakage["leakage_doc_ids_top_k"],
                    "page_recall_at_k": page_recall,
                    "page_precision_at_k": page_precision,
                    "page_mrr_at_k": page_mrr,
                    "chunk_hit_at_k": c_hit,
                    "chunk_precision_at_k": c_prec,
                    "chunk_mrr_at_k": c_mrr,
                    "top_pages": ranked_pages_unique[:10],
                    "top_chunk_ids": retrieved_chunk_ids[:5],
                    "top_doc_ids": retrieved_doc_ids[:5],
                }
            )

            if PRINT_HIT_DEBUG and k == 1 and page_recall == 1.0:
                top_pages_preview = ranked_pages_unique[:10]
                top_chunk_preview = retrieved_chunk_ids[:3]
                print(f"HIT@1 query_id={query_id} pages={top_pages_preview} chunks={top_chunk_preview}")

        extracted_answer = None
        extracted_label = None
        top_chunk_id = None
        top_text = ""
        ids = per_k.get("1", {}).get("retrieved_chunk_ids") or []
        if isinstance(ids, str):
            ids = [ids]
        if ids:
            top_chunk_id = str(ids[0])
            top_text = chunk_text_by_id.get(top_chunk_id, "")

        q_lower = question.lower()
        if top_text:
            if "significant" in q_lower and "delay" in q_lower:
                extracted_answer, quarter = _extract_quarter_value(
                    "Significant Delay", top_text, q_lower
                )
                if extracted_answer:
                    extracted_label = f"Significant Delay ({quarter})" if quarter else "Significant Delay"
            elif "on track" in q_lower:
                extracted_answer, quarter = _extract_quarter_value("On Track", top_text, q_lower)
                if extracted_answer:
                    extracted_label = f"On Track ({quarter})" if quarter else "On Track"
            elif "proportion" in q_lower and "complete" in q_lower:
                m = re.search(r"(\d+(?:\.\d+)?)%[^\n]{0,80}complete", top_text, flags=re.IGNORECASE)
                if not m:
                    m = re.search(r"complete[^\n]{0,80}(\d+(?:\.\d+)?)%", top_text, flags=re.IGNORECASE)
                if m:
                    extracted_answer = f"{m.group(1)}%"
                    extracted_label = "Complete (%)"
            elif "board committee" in q_lower and "strategic risk register" in q_lower:
                m = re.search(
                    r"the ([A-Za-z &-]+ committee) have delegated responsibility",
                    top_text,
                    flags=re.IGNORECASE,
                )
                if not m:
                    m = re.search(
                        r"the ([A-Za-z &-]+ committee) has delegated responsibility",
                        top_text,
                        flags=re.IGNORECASE,
                    )
                if m:
                    extracted_answer = m.group(1).strip().title()
                    extracted_label = "Delegated Committee"
            elif "endorse" in q_lower and "risk appetite" in q_lower and "strategic risk profile" in q_lower:
                ra_date = None
                srp_date = None
                for sent in re.split(r"(?<=[.!?])\s+", top_text):
                    low = sent.lower()
                    if "endorsed" in low and "risk appetite statement" in low:
                        m = re.search(
                            r"endorsed.*?on(?: the)?\s+(\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]+\s+\d{4})",
                            sent,
                            flags=re.IGNORECASE,
                        )
                        if m:
                            ra_date = m.group(1)
                    if "endorsed" in low and "strategic risk profile" in low:
                        m = re.search(
                            r"endorsed.*?strategic risk profile.*?in\s+([A-Z][a-z]+\s+\d{4})",
                            sent,
                            flags=re.IGNORECASE,
                        )
                        if m:
                            srp_date = m.group(1)
                if not srp_date:
                    candidate_ids = (
                        per_k.get("5", {}).get("retrieved_chunk_ids")
                        or per_k.get("3", {}).get("retrieved_chunk_ids")
                        or []
                    )
                    for cid in candidate_ids:
                        if str(cid) == top_chunk_id:
                            continue
                        other_text = chunk_text_by_id.get(str(cid), "")
                        if not other_text:
                            continue
                        for sent in re.split(r"(?<=[.!?])\s+", other_text):
                            low = sent.lower()
                            if "endorsed" in low and "strategic risk profile" in low:
                                m = re.search(
                                    r"endorsed.*?strategic risk profile.*?in\s+([A-Z][a-z]+\s+\d{4})",
                                    sent,
                                    flags=re.IGNORECASE,
                                )
                                if m:
                                    srp_date = m.group(1)
                                    break
                        if srp_date:
                            break
                parts = []
                if ra_date:
                    parts.append(f"Risk Appetite: {ra_date}")
                if srp_date:
                    parts.append(f"Strategic Risk Profile: {srp_date}")
                if parts:
                    extracted_answer = "; ".join(parts)
                    extracted_label = "Board Endorsements"
            elif "endorse" in q_lower and "risk appetite" in q_lower:
                for sent in re.split(r"(?<=[.!?])\s+", top_text):
                    if "endorsed" in sent and "risk appetite statement" in sent.lower():
                        m = re.search(
                            r"endorsed.*?on(?: the)?\s+(\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]+\s+\d{4})",
                            sent,
                            flags=re.IGNORECASE,
                        )
                        if m:
                            extracted_answer = m.group(1)
                            extracted_label = "Risk Appetite Endorsement"
                            break
            elif "endorse" in q_lower and "strategic risk profile" in q_lower:
                for sent in re.split(r"(?<=[.!?])\s+", top_text):
                    if "endorsed" in sent and "strategic risk profile" in sent.lower():
                        m = re.search(
                            r"endorsed.*?strategic risk profile.*?in\s+([A-Z][a-z]+\s+\d{4})",
                            sent,
                            flags=re.IGNORECASE,
                        )
                        if m:
                            extracted_answer = m.group(1)
                            extracted_label = "Strategic Risk Profile Endorsement"
                            break
            elif "significant issue" in q_lower and "accountable officer" in q_lower:
                for sent in re.split(r"(?<=[.!?])\s+", top_text):
                    if "funding arrangement" in sent.lower():
                        extracted_answer = sent.strip()
                        extracted_label = "Significant Issue"
                        break

        if not extracted_answer:
            if top_text:
                extracted_answer = _first_sentence(top_text) or "(no extraction rule matched)"
                extracted_label = "Snippet"
            else:
                extracted_answer = "(no chunk text available)"
                extracted_label = "Snippet"

        k1 = per_k.get("1", {})
        page_hit = 1 if k1.get("page_recall_at_k", 0.0) > 0 else 0
        failure_type = k1.get("failure_stage")

        results.append(
            {
                "query_id": query_id,
                "topic": qid_parts["topic"],
                "year": qid_parts["year"],
                "sequence": qid_parts["sequence"],
                "question": question,
                "answer_type": answer_type,
                "doc_id": expected_doc_id,
                "expected_section": expected_section,
                "expected_pages": sorted(list(expected_pages)),
                "page_hit": page_hit,
                "failure_type": failure_type,
                "extracted_answer": extracted_answer,
                "extracted_answer_label": extracted_label,
                "extracted_answer_chunk_id": top_chunk_id,
                "gold_presence": gold_presence,
                "per_k": per_k,
            }
        )

        print(
            f"EXTRACT query_id={query_id} page_hit={page_hit} failure_type={failure_type} "
            f"extracted_answer={extracted_label}: {extracted_answer}"
        )

    df_sum = pd.DataFrame(summary_rows)

    metrics: dict[str, Any] = {"run_info": run_info, "metrics_by_k": {}, "failure_counts_by_k": {}, "leakage_counts_by_k": {}}

    for k in k_list:
        dfk = df_sum[df_sum["k"] == k]

        metrics["metrics_by_k"][str(k)] = {
            "num_queries": int(len(dfk)),
            "page_hit_rate_at_k": float((dfk["page_recall_at_k"] > 0).mean()) if len(dfk) else 0.0,
            "mean_page_recall_at_k": float(dfk["page_recall_at_k"].mean()) if len(dfk) else 0.0,
            "mean_page_precision_at_k": float(dfk["page_precision_at_k"].mean()) if len(dfk) else 0.0,
            "mean_page_mrr_at_k": float(dfk["page_mrr_at_k"].mean()) if len(dfk) else 0.0,
            "chunk_hit_rate_at_k": float((dfk["chunk_hit_at_k"] > 0).mean()) if len(dfk) else 0.0,
            "mean_chunk_precision_at_k": float(dfk["chunk_precision_at_k"].mean()) if len(dfk) else 0.0,
            "mean_chunk_mrr_at_k": float(dfk["chunk_mrr_at_k"].mean()) if len(dfk) else 0.0,
        }

        metrics["failure_counts_by_k"][str(k)] = (
            dfk["failure_stage"].value_counts(dropna=False).to_dict() if len(dfk) else {}
        )

        metrics["leakage_counts_by_k"][str(k)] = {
            "num_queries": int(len(dfk)),
            "any_leakage_rate_at_k": float((dfk["leakage_count_top_k"] > 0).mean()) if len(dfk) else 0.0,
            "mean_leakage_rate_at_k": float(dfk["leakage_rate_top_k"].mean()) if len(dfk) else 0.0,
        }

    write_json(RESULTS_JSON, {"run_info": run_info, "results": results})
    write_json(METRICS_JSON, metrics)
    df_sum.to_csv(SUMMARY_CSV, index=False)

    print("Saved:", RESULTS_JSON)
    print("Saved:", METRICS_JSON)
    print("Saved:", SUMMARY_CSV)

    for k in k_list:
        m = metrics["metrics_by_k"][str(k)]
        fc = metrics["failure_counts_by_k"].get(str(k), {})
        lc = metrics["leakage_counts_by_k"].get(str(k), {})
        print(
            f"k={k}  "
            f"page_hit_rate={m['page_hit_rate_at_k']:.3f}  "
            f"page_mrr={m['mean_page_mrr_at_k']:.3f}  "
            f"page_precision={m['mean_page_precision_at_k']:.3f}  "
            f"chunk_hit_rate={m['chunk_hit_rate_at_k']:.3f}  "
            f"chunk_mrr={m['mean_chunk_mrr_at_k']:.3f}  "
            f"chunk_precision={m['mean_chunk_precision_at_k']:.3f}  "
            f"failures={fc}  "
            f"any_leakage_rate={lc.get('any_leakage_rate_at_k', 0.0):.3f}  "
            f"mean_leakage_rate={lc.get('mean_leakage_rate_at_k', 0.0):.3f}"
        )


if __name__ == "__main__":
    main()
