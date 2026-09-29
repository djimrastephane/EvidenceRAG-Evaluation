"""
retrieval_eval.py

NOTE
- This script is a dense-first baseline evaluator kept for ablations/comparisons.
- It does not represent the production pipeline end-to-end retrieval mode.
- For pipeline-faithful evaluation, use `scripts/evaluate_pipeline.py` or
  `scripts/retrieval_eval_hybrid.py`.

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
  Allowed TOPIC values: REV, EFF, DEF, STAFF, ACC, GOV, TABLE
  Examples:
    Q_REV_2023_01
    Q_EFF_2023_01
    Q_DEF_2023_01
    Q_STAFF_2023_02

- Reports both page-level metrics and chunk-level metrics.
  Page precision can drop as k increases because extra chunks add extra pages.
  Chunk metrics are often easier to interpret for retrieval ranking.

FAILURE ATTRIBUTION (RETRIEVAL + GENERATION)
This evaluator assigns a deterministic failure type per query at k=1:
Retrieval-stage failures:
- FP1_MISSING_CONTENT
- FP2_MISSED_TOP_RANK
- FP3_NOT_IN_CONTEXT
Generation-stage failures:
- FP4_NOT_EXTRACTED
- FP5_WRONG_FORMAT
- FP6_INCORRECT_SPECIFICITY
- FP7_INCOMPLETE

Per-k retrieval-only failure stages are still reported:
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
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Optional

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
    import torch
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

repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from rag_pdf.question_router import QueryRoute, route_question
from rag_pdf.retrieval.rerank import (
    RerankConfig,
    numeric_density_boost,
    query_overlap_boost,
    segment_search_hit_boost,
    table_priority_boost,
)
from runtime_env import collect_runtime_provenance, critical_environment_checks
from rag_pdf.answer_scoring import (  # noqa: F401  (re-exported for other scripts)
    FAILURE_TYPES,
    categorize_failure_type,
    score_answer_correctness,
)


# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR = Path(
    "/Users/djimra/MSc Data Science Jan 2025/Thesis documents/EvidenceRAG-Evaluation/data_processed/"
    "Grampian-2022-2023"
)

INDEX_PATH = DATA_DIR / "faiss.index"
META_PATH = DATA_DIR / "chunk_meta.parquet"
EVAL_SET_PATH = DATA_DIR / "eval_set.json"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

K_LIST = [1, 3, 5, 10]
MAX_K_SEARCH = int(os.getenv("MAX_K_SEARCH", "100"))
SUBSECTION_BOOST = float(os.getenv("SUBSECTION_BOOST", "0.05"))
TABLE_CHUNK_BOOST = float(os.getenv("TABLE_CHUNK_BOOST", "0.08"))
MILESTONE_TEXT_BOOST = float(os.getenv("MILESTONE_TEXT_BOOST", "0.08"))
ENTITY_MATCH_BOOST = float(os.getenv("ENTITY_MATCH_BOOST", "0.04"))
NUMERIC_DENSITY_BOOST = float(os.getenv("NUMERIC_DENSITY_BOOST", "0.03"))
SEGMENT_SEARCH_HIT_BOOST = float(os.getenv("SEGMENT_SEARCH_HIT_BOOST", "0.03"))
MAX_ENTITY_MATCHES = int(os.getenv("MAX_ENTITY_MATCHES", "4"))
ENABLE_LEXICAL_RERANK = os.getenv("ENABLE_LEXICAL_RERANK", "1") != "0"
# Oracle: reads the gold expected_subsection label. Opt-in only (labelled ablations).
ENABLE_SUBSECTION_BOOST = os.getenv("ENABLE_SUBSECTION_BOOST", "0") == "1"

RESULTS_JSON = DATA_DIR / "retrieval_results.json"
METRICS_JSON = DATA_DIR / "retrieval_metrics.json"
SUMMARY_CSV = DATA_DIR / "retrieval_summary.csv"
TABLE_FACTS_PATH = DATA_DIR / "table_facts.parquet"

PRINT_HIT_DEBUG = True


# =============================================================================
# QUERY ID NOMENCLATURE
# =============================================================================
QUERY_ID_PATTERN_V1 = re.compile(r"^Q_(REV|EFF|DEF|STAFF|ACC|GOV|TABLE)_\d{4}_\d{2}$")
QUERY_ID_PATTERN_V2 = re.compile(r"^Q_(\d{4})_([A-Z]+)_(\d{2}|P\d+)$")


def validate_query_id(query_id: str) -> None:
    if QUERY_ID_PATTERN_V1.match(query_id) or QUERY_ID_PATTERN_V2.match(query_id):
        return
    if not (query_id.startswith("Q_") and len(query_id) >= 6):
        # Keep compatibility with historical datasets that used custom IDs.
        raise ValueError(
            f"Invalid query_id '{query_id}'. Expected: Q_<TOPIC>_<YEAR>_<NN> "
            f"with TOPIC in [REV, EFF, DEF, STAFF, ACC, GOV, TABLE], "
            "for example Q_EFF_2023_01."
        )


def parse_query_id(query_id: str) -> dict[str, Any]:
    parts = query_id.split("_")
    if len(parts) >= 4 and parts[1].isdigit():
        year = int(parts[1])
        topic = parts[2]
        seq_raw = parts[3]
        if seq_raw.isdigit():
            seq: Any = int(seq_raw)
        else:
            seq = seq_raw
        return {"topic": topic, "year": year, "sequence": seq}
    _, topic, year, seq = parts[:4]
    if str(seq).isdigit():
        seq_val: Any = int(seq)
    else:
        seq_val = seq
    return {"topic": topic, "year": int(year), "sequence": seq_val}


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


def resolve_torch_device(name: str) -> str:
    requested = str(name or "cpu").strip().lower()
    if requested == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if requested == "mps":
        return "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    if requested == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def parse_k_list(val: str) -> list[int]:
    parts = [p.strip() for p in val.split(",") if p.strip()]
    return [int(p) for p in parts]


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def _normalize_key(s: str) -> str:
    """Normalize free text into snake_case key format for robust matching."""
    t = _normalize_text(s).replace("&", " and ")
    t = re.sub(r"[^a-z0-9]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t


def _file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = read_json(path)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}


def _collect_pipeline_settings(data_dir: Path) -> dict[str, Any]:
    """
    Capture preprocessing/pipeline settings from metrics.json so retrieval runs
    can always be traced back to the exact chunking/config used.
    """
    metrics_path = data_dir / "metrics.json"
    metrics = _safe_json(metrics_path)
    params = metrics.get("params", {}) if isinstance(metrics, dict) else {}
    return {
        "source_metrics_path": str(metrics_path),
        "source_metrics_exists": metrics_path.exists(),
        "doc_id": metrics.get("doc_id"),
        "corpus_id": metrics.get("corpus_id"),
        "report_year": metrics.get("report_year"),
        "preprocess_run_utc": metrics.get("run_utc"),
        "preprocess_git_commit_short": metrics.get("git_commit_short"),
        "embedding_model_preprocess": metrics.get("embedding_model"),
        "chunk_size_tokens": params.get("chunk_size_tokens"),
        "chunk_overlap_tokens": params.get("chunk_overlap_tokens"),
        "segment_aware_chunking": params.get("segment_aware_chunking"),
        "whole_doc_markdown_mode": params.get("whole_doc_markdown_mode"),
        "markdown_header_carry_forward": params.get("markdown_header_carry_forward"),
        "markdown_table_injection": params.get("markdown_table_injection"),
        "primary_extractor": params.get("primary_extractor"),
        "min_chunk_words": params.get("min_chunk_words"),
    }


def _collect_eval_set_info(eval_set_path: Path, eval_obj: Any, query_count: int) -> dict[str, Any]:
    """
    Capture eval_set version/fingerprint for reproducibility and drift tracking.
    """
    stat = eval_set_path.stat()
    meta: dict[str, Any] = {}
    if isinstance(eval_obj, dict) and isinstance(eval_obj.get("_meta"), dict):
        meta = eval_obj.get("_meta", {})
    return {
        "path": str(eval_set_path),
        "sha1": _file_sha1(eval_set_path),
        "size_bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat(),
        "query_count": int(query_count),
        "meta_dataset_name": meta.get("dataset_name"),
        "meta_doc_id": meta.get("doc_id"),
        "meta_version": meta.get("version"),
    }


def is_table_metric_route(route: QueryRoute) -> bool:
    """Return True when routed intent belongs to table-metric extraction family."""
    return route.intent.startswith("table_metric_")


def _format_numeric_answer(value: float, is_percent: bool) -> str:
    """Format numeric answer string with optional percent suffix."""
    if float(value).is_integer():
        txt = str(int(value))
    else:
        txt = f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{txt}%" if is_percent else txt


def _milestone_text_relevance_boost(route: QueryRoute, question: str, chunk_text: str) -> float:
    """
    Compute lexical rerank boost for routed table-metric questions.

    This boost is only active for table-metric intents and is intended to lift
    milestone/deliverable tables in top-k ranking.
    """
    if not is_table_metric_route(route):
        return 0.0
    q = _normalize_text(question)
    t = _normalize_text(chunk_text)
    if not t:
        return 0.0
    boost = 0.0
    if route.intent in {
        "table_metric_significant_delay",
        "table_metric_on_track",
        "table_metric_complete_ratio",
    }:
        if any(k in t for k in ("milestone", "milestones", "deliverable", "deliverables")):
            boost += MILESTONE_TEXT_BOOST
        if "deliverable" in q and "complete" in q and any(k in t for k in ("milestone", "deliverable")) and "complete" in t:
            boost += 0.18
        if "on track" in q and "on track" in t:
            boost += 0.05
        if "significant" in q and "delay" in q and "significant delay" in t:
            boost += 0.05
        if "proportion" in q and "complete" in q and "complete" in t:
            boost += 0.04
        if "q1" in q and "q1" in t:
            boost += 0.03
        if "q2" in q and "q2" in t:
            boost += 0.03
        if "q3" in q and "q3" in t:
            boost += 0.03
        if "q4" in q and "q4" in t:
            boost += 0.03
    elif route.intent == "table_metric_staff_costs":
        if any(k in t for k in ("staff cost", "staff costs", "employee benefit", "remuneration", "pension")):
            boost += MILESTONE_TEXT_BOOST
        if "pension" in q and "pension" in t:
            boost += 0.06
        if "remuneration" in q and "remuneration" in t:
            boost += 0.06
        if "total" in q and "staff" in q and "cost" in q and "total" in t:
            boost += 0.04
    elif route.intent == "table_metric_emissions":
        if any(k in t for k in ("emission", "greenhouse gas", "co2", "carbon", "target emissions")):
            boost += MILESTONE_TEXT_BOOST
        if "target" in q and "target emissions" in t:
            boost += 0.06
        if "percentage change" in q and "percentage change" in t:
            boost += 0.06
    return boost


def extract_answer_from_table_facts(
    route: QueryRoute,
    table_facts_df: pd.DataFrame,
    candidate_pages: list[int],
    expected_doc_id: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """
    Attempt deterministic answer extraction from canonical table facts.

    Returns:
        (answer, label) if matched, else (None, None)
    """
    if table_facts_df is None or table_facts_df.empty:
        return None, None

    if not is_table_metric_route(route):
        return None, None
    row_terms = [str(x) for x in route.slots.get("row_terms", [])]
    quarter = route.slots.get("quarter")
    prefer_percent = bool(route.slots.get("prefer_percent", False))
    column_terms = [str(x) for x in route.slots.get("column_terms", [])]
    table_type_hint = str(route.slots.get("table_type_hint") or "").strip().lower()
    if not row_terms:
        return None, None

    facts = table_facts_df.copy()
    if expected_doc_id and "doc_id" in facts.columns:
        facts = facts[facts["doc_id"].astype(str) == str(expected_doc_id)]
    if candidate_pages:
        facts = facts[facts["page"].isin(candidate_pages)]
    if table_type_hint and "table_type" in facts.columns:
        typed = facts[facts["table_type"].astype(str).str.lower() == table_type_hint]
        if not typed.empty:
            facts = typed
    if facts.empty:
        return None, None

    facts["row_label_norm"] = facts["row_label_norm"].astype(str).map(_normalize_key)
    mask = pd.Series(False, index=facts.index)
    for term in row_terms:
        mask = mask | facts["row_label_norm"].str.contains(term, na=False)
    facts = facts[mask]
    if facts.empty:
        return None, None

    if column_terms and "column_header_norm" in facts.columns:
        facts["column_header_norm"] = facts["column_header_norm"].astype(str).map(_normalize_key)
        cmask = pd.Series(False, index=facts.index)
        for cterm in column_terms:
            cterm_norm = _normalize_key(cterm)
            cmask = cmask | facts["column_header_norm"].str.contains(cterm_norm, na=False)
        cfiltered = facts[cmask]
        if not cfiltered.empty:
            facts = cfiltered

    if quarter:
        qfacts = facts[facts["quarter"].astype(str).str.lower() == quarter]
        if not qfacts.empty:
            facts = qfacts

    if prefer_percent:
        pfacts = facts[facts["is_percent"] == True]  # noqa: E712
        if not pfacts.empty:
            facts = pfacts

    if facts.empty:
        return None, None

    if not quarter and "quarter" in facts.columns and facts["quarter"].notna().any():
        facts = facts.copy()
        rank = {"q1": 1, "q2": 2, "q3": 3, "q4": 4}
        facts["q_rank"] = facts["quarter"].astype(str).str.lower().map(rank).fillna(0)
        facts = facts.sort_values(["q_rank", "value_num"], ascending=[False, False])
    else:
        facts = facts.sort_values(["value_num"], ascending=[False])

    top = facts.iloc[0]
    answer = _format_numeric_answer(float(top["value_num"]), bool(top.get("is_percent", False)))
    row_label = str(top.get("row_label_raw", "")).strip() or str(top.get("row_label_norm", "")).strip()
    col_label = str(top.get("column_header_raw", "")).strip()
    label = f"TableFact ({row_label} | {col_label})"
    return answer, label


def refresh_paths() -> None:
    global INDEX_PATH, META_PATH, EVAL_SET_PATH, RESULTS_JSON, METRICS_JSON, SUMMARY_CSV, TABLE_FACTS_PATH
    INDEX_PATH = DATA_DIR / "faiss.index"
    META_PATH = DATA_DIR / "chunk_meta.parquet"
    EVAL_SET_PATH = DATA_DIR / "eval_set.json"
    RESULTS_JSON = DATA_DIR / "retrieval_results.json"
    METRICS_JSON = DATA_DIR / "retrieval_metrics.json"
    SUMMARY_CSV = DATA_DIR / "retrieval_summary.csv"
    TABLE_FACTS_PATH = DATA_DIR / "table_facts.parquet"


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
        "--device",
        default=_env_or_default("ST_MODEL_DEVICE", "cpu"),
        help="Embedding device: cpu, mps, cuda, or auto.",
    )
    parser.add_argument(
        "--k-list",
        default=_env_or_default("K_LIST", ",".join(str(k) for k in K_LIST)),
        help="Comma-separated list of k values (e.g. 1,3,5,10).",
    )
    parser.add_argument(
        "--no-lexical-rerank",
        action="store_true",
        help="Disable lexical rerank boosts for pure dense retrieval.",
    )
    parser.add_argument(
        "--oracle-subsection-boost",
        action="store_true",
        help=(
            "ORACLE ABLATION ONLY: boost chunks matching the gold expected_subsection label. "
            "Never use for reported system results. Also enabled by ENABLE_SUBSECTION_BOOST=1."
        ),
    )
    parser.add_argument(
        "--no-subsection-boost",
        action="store_true",
        help="Force the oracle subsection boost off (the default), overriding ENABLE_SUBSECTION_BOOST=1.",
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
    print(
        "[info] scripts/retrieval_eval.py is the dense-first baseline evaluator. "
        "Use scripts/evaluate_pipeline.py for pipeline-faithful hybrid evaluation."
    )
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
    table_facts_df = read_parquet_safe(TABLE_FACTS_PATH) if TABLE_FACTS_PATH.exists() else pd.DataFrame()
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
    model = SentenceTransformer(str(args.model), device=resolve_torch_device(args.device))

    eval_obj = read_json(EVAL_SET_PATH)
    if isinstance(eval_obj, list):
        eval_items = eval_obj
    elif isinstance(eval_obj, dict) and isinstance(eval_obj.get("queries"), list):
        eval_items = eval_obj.get("queries", [])
    else:
        eval_items = []
    if len(eval_items) == 0:
        raise ValueError("eval_set.json must be a non-empty list of query objects (or {'queries': [...]}).")

    for i, item in enumerate(eval_items):
        qid = str(item.get("query_id", "")).strip()
        if not qid:
            raise ValueError(f"Missing query_id for eval item at index {i}.")
        validate_query_id(qid)

    max_k = min(max(K_LIST), len(meta))
    max_k_search = min(max(MAX_K_SEARCH, max_k), len(meta))
    k_list = [k for k in K_LIST if 1 <= k <= max_k]
    enable_lexical_rerank = ENABLE_LEXICAL_RERANK and (not args.no_lexical_rerank)
    enable_subsection_boost = (ENABLE_SUBSECTION_BOOST or args.oracle_subsection_boost) and (not args.no_subsection_boost)
    if enable_subsection_boost:
        print("[warn] ORACLE subsection boost enabled: ranking uses gold expected_subsection labels.")
    if not k_list:
        raise ValueError("K_LIST has no valid values for the current index size.")

    meta_doc_ids = set(meta["doc_id"].astype(str).unique()) if "doc_id" in meta.columns else set()

    run_info = {
        "run_utc": utc_now_iso(),
        "data_dir": str(DATA_DIR),
        "runtime": collect_runtime_provenance(),
        "critical_environment_checks": critical_environment_checks(),
        "index_path": str(INDEX_PATH),
        "meta_path": str(META_PATH),
        "table_facts_path": str(TABLE_FACTS_PATH),
        "eval_set_path": str(EVAL_SET_PATH),
        "embedding_model": EMBED_MODEL_NAME,
        "k_list": k_list,
        "subsection_boost": SUBSECTION_BOOST,
        "table_chunk_boost": TABLE_CHUNK_BOOST,
        "milestone_text_boost": MILESTONE_TEXT_BOOST,
        "entity_match_boost": ENTITY_MATCH_BOOST,
        "numeric_density_boost": NUMERIC_DENSITY_BOOST,
        "segment_search_hit_boost": SEGMENT_SEARCH_HIT_BOOST,
        "max_entity_matches": MAX_ENTITY_MATCHES,
        "enable_lexical_rerank": enable_lexical_rerank,
        "enable_subsection_boost": enable_subsection_boost,
        "oracle_labels_in_ranking": bool(enable_subsection_boost),
        "max_k_search": max_k_search,
        "num_queries": len(eval_items),
        "num_chunks_indexed": int(len(meta)),
        "num_table_facts": int(len(table_facts_df)),
        "meta_doc_ids": sorted(list(meta_doc_ids))[:20] if meta_doc_ids else [],
        "query_id_nomenclature": "Q_<TOPIC>_<YEAR>_<NN> with TOPIC in [REV,EFF,DEF,STAFF,ACC,GOV,TABLE]",
        "failure_attribution": {
            "retrieval_stages": ["hit", "missed_top_ranked", "missing_content"],
            "failure_types": list(FAILURE_TYPES.keys()),
            "scope": "retrieval_and_generation",
        },
        "leakage_detection": {
            "enabled": True,
            "requires_expected_doc_id": True,
            "fields": ["retrieved_doc_ids_top_k", "leakage_count_top_k", "leakage_doc_ids_top_k", "leakage_rate_top_k"],
        },
        "pipeline_settings": _collect_pipeline_settings(DATA_DIR),
        "eval_set": _collect_eval_set_info(EVAL_SET_PATH, eval_obj, len(eval_items)),
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
    rerank_cfg = RerankConfig(
        table_chunk_boost=TABLE_CHUNK_BOOST,
        entity_match_boost=ENTITY_MATCH_BOOST,
        numeric_density_boost=NUMERIC_DENSITY_BOOST,
        segment_search_hit_boost=SEGMENT_SEARCH_HIT_BOOST,
        max_entity_matches=MAX_ENTITY_MATCHES,
    )

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
        route = route_question(question)

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

        if enable_lexical_rerank:
            boosted: list[tuple[float, int]] = []
            for score, idx in zip(scores, idxs):
                is_table_chunk = bool(meta.iloc[idx].get("is_table", False))
                cid = (
                    meta.iloc[idx].get("chunk_id_global")
                    if "chunk_id_global" in meta.columns
                    else meta.iloc[idx].get("chunk_id")
                )
                ctext = chunk_text_by_id.get(str(cid), "")
                score += table_priority_boost(
                    is_table_chunk=is_table_chunk,
                    route_intent=route.intent,
                    config=rerank_cfg,
                )
                score += query_overlap_boost(question=question, chunk_text=ctext, config=rerank_cfg)
                score += numeric_density_boost(question=question, chunk_text=ctext, config=rerank_cfg)
                score += segment_search_hit_boost(
                    question=question,
                    segment_has_search_hit=bool(meta.iloc[idx].get("segment_has_search_hit", False)),
                    config=rerank_cfg,
                )
                score += _milestone_text_relevance_boost(route, question, ctext)
                boosted.append((score, idx))
            boosted.sort(key=lambda x: x[0], reverse=True)
            scores = [s for s, _ in boosted]
            idxs = [i for _, i in boosted]

        if enable_subsection_boost and expected_subsection and "subsection_title" in meta.columns:
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
                    "route_intent": route.intent,
                    "route_confidence": route.confidence,
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

        k3_pages = per_k.get("3", {}).get("retrieved_pages_ranked", []) or []
        k1_pages = per_k.get("1", {}).get("retrieved_pages_ranked", []) or []
        candidate_page_source = k3_pages if k3_pages else k1_pages
        candidate_pages = [int(p) for p in candidate_page_source[:8] if str(p).isdigit()]
        fact_answer, fact_label = extract_answer_from_table_facts(
            route=route,
            table_facts_df=table_facts_df,
            candidate_pages=candidate_pages,
            expected_doc_id=expected_doc_id,
        )
        if fact_answer:
            extracted_answer = fact_answer
            extracted_label = fact_label

        if top_text and not extracted_answer:
            if is_table_metric_route(route):
                extracted_answer = _first_sentence(top_text) or "(no extraction rule matched)"
                extracted_label = "Snippet"

        if not extracted_answer:
            if top_text:
                extracted_answer = _first_sentence(top_text) or "(no extraction rule matched)"
                extracted_label = "Snippet"
            else:
                extracted_answer = "(no chunk text available)"
                extracted_label = "Snippet"

        k1 = per_k.get("1", {})
        page_hit = 1 if k1.get("page_recall_at_k", 0.0) > 0 else 0
        context_chunk_ids = (
            per_k.get("3", {}).get("retrieved_chunk_ids")
            or per_k.get("1", {}).get("retrieved_chunk_ids")
            or []
        )
        if isinstance(context_chunk_ids, str):
            context_chunk_ids = [context_chunk_ids]
        context_text = "\n".join(
            chunk_text_by_id.get(str(cid), "")
            for cid in context_chunk_ids
            if chunk_text_by_id.get(str(cid), "")
        )

        failure_type = categorize_failure_type(
            page_hit=page_hit,
            gold_exists=bool(gold_presence.get("gold_exists", False)),
            expected_answer=item.get("expected_answer"),
            answer_type=answer_type,
            context_text=context_text,
            extracted_answer=extracted_answer,
        )
        answer_correct, answer_status = score_answer_correctness(
            expected_answer=item.get("expected_answer"),
            answer_type=answer_type,
            extracted_answer=extracted_answer,
        )

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
                "route_intent": route.intent,
                "route_confidence": route.confidence,
                "page_hit": page_hit,
                "failure_type": failure_type,
                "failure_stage": FAILURE_TYPES.get(failure_type, "unknown"),
                "expected_answer": item.get("expected_answer"),
                "extracted_answer": extracted_answer,
                "extracted_answer_label": extracted_label,
                "answer_correct": answer_correct,
                "answer_status": answer_status,
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
    df_results = pd.DataFrame(results)

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

    scored = df_results[df_results["answer_correct"].notna()] if len(df_results) else pd.DataFrame()
    metrics["answer_scoring"] = {
        "num_queries_total": int(len(df_results)),
        "num_queries_scored": int(len(scored)) if len(df_results) else 0,
        "answer_accuracy": float(scored["answer_correct"].mean()) if len(scored) else None,
        "answer_status_counts": scored["answer_status"].value_counts(dropna=False).to_dict() if len(scored) else {},
    }

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

    a = metrics.get("answer_scoring", {})
    if a.get("num_queries_scored", 0):
        print(
            "answer_scoring "
            f"scored={a.get('num_queries_scored')} "
            f"accuracy={a.get('answer_accuracy'):.3f} "
            f"status_counts={a.get('answer_status_counts')}"
        )


if __name__ == "__main__":
    main()
