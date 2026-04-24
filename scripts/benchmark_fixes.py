"""
benchmark_fixes.py

Measures the actual runtime impact of two performance fixes:
  Fix #2 - regex caching in chunking (chunking.py)
  Fix #3 - iloc loop vs pre-selected rows in rerank (canonical_hybrid.py)

Run from the project root:
    python scripts/benchmark_fixes.py
"""

from __future__ import annotations

import re
import time
import functools
import statistics

import numpy as np
import pandas as pd


REPS = 500  # repetitions per timing loop


# =============================================================================
# Fix #2: regex compile caching
# =============================================================================

PATTERNS = (
    r"(\b\d+(?:\.\d+){1,4}\s+[A-Z][A-Za-z][^\n]{0,120})",
    r"^\d+(?:\.\d+){1,5}\b",
    r"(?i)^[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\s+(?:integration\s+joint\s+board\s*\(ijb\)|ijb)\b",
    r"(?i)\b(?:integration\s+joint\s+boards?|ijbs?)\b",
    r"^[A-Z][A-Z0-9 ,/&()\-]{8,}$",
)

@functools.lru_cache(maxsize=256)
def _compile_cached(pattern: str) -> re.Pattern:
    return re.compile(pattern)


def compile_before() -> list[re.Pattern]:
    """Old: compile on every call."""
    return [re.compile(p) for p in PATTERNS]


def compile_after() -> list[re.Pattern]:
    """New: lru_cache returns pre-compiled objects."""
    return [_compile_cached(p) for p in PATTERNS]


def benchmark_regex() -> dict:
    # Warm up cache
    compile_after()

    times_before, times_after = [], []
    for _ in range(REPS):
        t0 = time.perf_counter()
        compile_before()
        times_before.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        compile_after()
        times_after.append(time.perf_counter() - t0)

    avg_before = statistics.mean(times_before) * 1_000_000  # µs
    avg_after  = statistics.mean(times_after)  * 1_000_000
    return {
        "fix": "#2 regex caching (per page call)",
        "before_us": round(avg_before, 2),
        "after_us":  round(avg_after,  2),
        "speedup_x": round(avg_before / max(avg_after, 0.001), 1),
        "per_200_page_doc_before_ms": round(avg_before * 200 / 1000, 2),
        "per_200_page_doc_after_ms":  round(avg_after  * 200 / 1000, 2),
    }


# =============================================================================
# Fix #3: rerank iloc loop
# =============================================================================

def make_meta(n_chunks: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "chunk_id":        [f"doc:{i}" for i in range(n_chunks)],
        "chunk_id_global": [f"corpus:doc:{i}" for i in range(n_chunks)],
        "section_title":   [f"Section {i % 20}" for i in range(n_chunks)],
        "is_table":        rng.choice([True, False], n_chunks),
        "segment_has_search_hit": rng.choice([True, False], n_chunks),
        "subsection_title": [f"Sub {i % 50}" for i in range(n_chunks)],
    })


def rerank_before(ranked: list[int], meta: pd.DataFrame, scores: dict[int, float]) -> dict[int, float]:
    """Old: meta.iloc[idx] inside loop."""
    updated = dict(scores)
    for idx in ranked:
        row = meta.iloc[idx]
        _ = str(row.get("chunk_id_global") or row.get("chunk_id") or "")
        _ = bool(row.get("is_table", False))
        _ = bool(row.get("segment_has_search_hit", False))
        updated[idx] = updated.get(idx, 0.0) + 0.01
    return updated


def rerank_after(ranked: list[int], meta: pd.DataFrame, scores: dict[int, float]) -> dict[int, float]:
    """New: pre-select rows, then iterate."""
    updated = dict(scores)
    ranked_rows = meta.iloc[ranked]
    for idx, (_, row) in zip(ranked, ranked_rows.iterrows()):
        _ = str(row.get("chunk_id_global") or row.get("chunk_id") or "")
        _ = bool(row.get("is_table", False))
        _ = bool(row.get("segment_has_search_hit", False))
        updated[idx] = updated.get(idx, 0.0) + 0.01
    return updated


def benchmark_rerank() -> dict:
    meta = make_meta(1000)
    rng = np.random.default_rng(0)
    ranked = rng.choice(len(meta), size=20, replace=False).tolist()
    scores = {int(i): float(rng.random()) for i in ranked}

    times_before, times_after = [], []
    for _ in range(REPS):
        t0 = time.perf_counter()
        rerank_before(ranked, meta, scores)
        times_before.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        rerank_after(ranked, meta, scores)
        times_after.append(time.perf_counter() - t0)

    avg_before = statistics.mean(times_before) * 1_000  # ms
    avg_after  = statistics.mean(times_after)  * 1_000
    return {
        "fix": "#3 rerank iloc loop (per query, top-20, 1000-chunk index)",
        "before_ms": round(avg_before, 3),
        "after_ms":  round(avg_after,  3),
        "speedup_x": round(avg_before / max(avg_after, 0.0001), 2),
        "per_100_queries_before_ms": round(avg_before * 100, 1),
        "per_100_queries_after_ms":  round(avg_after  * 100, 1),
    }


# =============================================================================
# MAIN
# =============================================================================

def print_result(r: dict) -> None:
    print(f"\n  Fix {r['fix']}")
    print(f"  {'─' * 55}")
    if "before_us" in r:
        print(f"  Per-call latency:   {r['before_us']:>8.2f} µs  →  {r['after_us']:>6.2f} µs   ({r['speedup_x']}× faster)")
        print(f"  200-page document:  {r['per_200_page_doc_before_ms']:>8.2f} ms  →  {r['per_200_page_doc_after_ms']:>6.2f} ms")
    else:
        print(f"  Per-query latency:  {r['before_ms']:>8.3f} ms  →  {r['after_ms']:>6.3f} ms   ({r['speedup_x']}× faster)")
        print(f"  100-query eval run: {r['per_100_queries_before_ms']:>8.1f} ms  →  {r['per_100_queries_after_ms']:>6.1f} ms")


if __name__ == "__main__":
    print("=" * 60)
    print("  Pipeline fix benchmarks")
    print("=" * 60)

    r2 = benchmark_regex()
    r3 = benchmark_rerank()

    print_result(r2)
    print_result(r3)

    print(f"\n  Fixes #1 #4 #5 #6 — correctness only, zero latency impact.")
    print("=" * 60)