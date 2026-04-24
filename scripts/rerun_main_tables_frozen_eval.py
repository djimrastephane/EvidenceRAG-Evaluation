#!/usr/bin/env python3
"""Recompute all main thesis tables using frozen pipeline artifacts + current eval_set.json.

Reads pre-computed page-ranked CSV files from frozen 224/56 artifacts and
re-evaluates them against the current data_processed/eval_set.json gold pages.
This ensures a single eval set is used throughout the thesis.

Produces results/rerun_main_tables_2026-04-24/results.json with:
  table_4_1       : Dense/BM25/Hybrid/Hybrid+boost × Hit@1/Hit@3/MRR@10
  per_document    : Hybrid(base) Hit@1/MRR@10 per doc
  per_difficulty  : Dense/BM25/Hybrid × LEX/MOD/STR/All × Hit@1/MRR@10
  chunk_vs_page   : Dense/BM25/Hybrid/Hybrid+boost × chunk_h1/page_h1/chunk_h3/page_h3
  fp_counts       : TP/FP2/FP3 counts for all four methods
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT     = Path(__file__).resolve().parents[1]
DOCS     = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]
OFF_ROOT = ROOT / "results" / "thesis_ablations" / "chunk_size_ablation_boost_off_2026-04-20" / "pipeline_outputs"
ON_ROOT  = ROOT / "results" / "thesis_ablations" / "chunk_size_ablation_2026-04-15"           / "pipeline_outputs"
EVAL_ROOT = ROOT / "data_processed"
OUTPUT_DIR = ROOT / "results" / "rerun_main_tables_2026-04-24"

KS = (1, 3, 5, 10)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_gold_map() -> dict[str, dict]:
    """Load current eval_set gold pages and difficulty for every query."""
    gold: dict[str, dict] = {}
    for doc in DOCS:
        data = json.loads((EVAL_ROOT / doc / "eval_set.json").read_text())
        for q in data["queries"]:
            gold[q["query_id"]] = {
                "difficulty": q["difficulty"],
                "doc_id": q["doc_id"],
                "gold_pages": set(q["expected_pages"]),
            }
    return gold


def hits_from_csv(path: Path, gold_map: dict) -> list[dict]:
    """Re-evaluate a pre-computed page-ranked CSV against current gold pages."""
    df = pd.read_csv(path, usecols=["query_id", "rank", "page_number"])
    rows = []
    for qid, grp in df.groupby("query_id"):
        if qid not in gold_map:
            continue
        gold = gold_map[qid]["gold_pages"]
        ranked = grp.sort_values("rank")["page_number"].tolist()
        first_rel = next((i + 1 for i, pg in enumerate(ranked) if pg in gold), None)
        top10 = set(ranked[:10])
        rows.append({
            "query_id": qid,
            "doc_id":   gold_map[qid]["doc_id"],
            "difficulty": gold_map[qid]["difficulty"],
            "hit@1":  1.0 if set(ranked[:1]) & gold else 0.0,
            "hit@3":  1.0 if set(ranked[:3]) & gold else 0.0,
            "hit@5":  1.0 if set(ranked[:5]) & gold else 0.0,
            "hit@10": 1.0 if set(ranked[:10]) & gold else 0.0,
            "mrr@10": (1.0 / first_rel) if first_rel and first_rel <= 10 else 0.0,
            "tp":  int(bool(set(ranked[:1]) & gold)),
            "fp2": int(not bool(set(ranked[:1]) & gold) and bool(top10 & gold)),
            "fp3": int(not bool(top10 & gold)),
        })
    return rows


def load_method_df(csv_name: str, abl_root: Path, gold_map: dict) -> pd.DataFrame:
    frames = []
    for doc in DOCS:
        p = abl_root / f"minilmcap_{doc}_chunk_224_56" / doc / csv_name
        frames.append(pd.DataFrame(hits_from_csv(p, gold_map)))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def agg_overall(df: pd.DataFrame) -> dict:
    return {
        "hit@1":  round(df["hit@1"].mean(), 4),
        "hit@3":  round(df["hit@3"].mean(), 4),
        "hit@5":  round(df["hit@5"].mean(), 4),
        "mrr@10": round(df["mrr@10"].mean(), 4),
        "n": len(df),
    }


def agg_by_difficulty(df: pd.DataFrame) -> dict:
    result = {}
    for tier in ("LEX", "MOD", "STR", "All"):
        sub = df if tier == "All" else df[df["difficulty"] == tier]
        result[tier] = {
            "n":     len(sub),
            "hit@1": round(sub["hit@1"].mean(), 3),
            "mrr@10": round(sub["mrr@10"].mean(), 3),
        }
    return result


def agg_by_doc(df: pd.DataFrame) -> dict:
    result = {}
    for doc in DOCS:
        sub = df[df["doc_id"] == doc]
        result[doc] = {
            "n":     len(sub),
            "hit@1": round(sub["hit@1"].mean(), 4),
            "mrr@10": round(sub["mrr@10"].mean(), 4),
        }
    return result


def fp_counts(df: pd.DataFrame) -> dict:
    return {"tp": int(df["tp"].sum()), "fp2": int(df["fp2"].sum()), "fp3": int(df["fp3"].sum())}


def print_table_4_1(methods: dict[str, pd.DataFrame]) -> None:
    print("\n=== TABLE 4.1 (frozen artifacts + current eval_set) ===")
    print(f"{'Method':<30} {'H@1':>8} {'H@3':>8} {'H@5':>8} {'MRR@10':>8}")
    print("-" * 62)
    for label, df in methods.items():
        ov = agg_overall(df)
        print(f"{label:<30} {ov['hit@1']:>8.4f} {ov['hit@3']:>8.4f} {ov['hit@5']:>8.4f} {ov['mrr@10']:>8.4f}")


def print_per_difficulty(methods: dict[str, pd.DataFrame]) -> None:
    print("\n=== PER-DIFFICULTY TABLE (boost-OFF methods) ===")
    print(f"{'Tier':<5} {'N':>4}  {'Dense H@1':>9} {'Dense MRR':>9}  "
          f"{'BM25 H@1':>8} {'BM25 MRR':>8}  {'Hybrid H@1':>10} {'Hybrid MRR':>10}")
    print("-" * 80)
    d_t = agg_by_difficulty(methods["Dense (MiniLM)"])
    b_t = agg_by_difficulty(methods["BM25-only"])
    h_t = agg_by_difficulty(methods["Hybrid (base)"])
    for tier in ("LEX", "MOD", "STR", "All"):
        d, b, h = d_t[tier], b_t[tier], h_t[tier]
        print(f"{tier:<5} {d['n']:>4}  {d['hit@1']:>9.3f} {d['mrr@10']:>9.3f}  "
              f"{b['hit@1']:>8.3f} {b['mrr@10']:>8.3f}  {h['hit@1']:>10.3f} {h['mrr@10']:>10.3f}")


def print_per_doc(hybrid_df: pd.DataFrame) -> None:
    print("\n=== PER-DOCUMENT TABLE (Hybrid base) ===")
    print(f"{'Document':<22} {'N':>4}  {'H@1':>8}  {'MRR@10':>8}")
    print("-" * 48)
    by_doc = agg_by_doc(hybrid_df)
    for doc in DOCS:
        r = by_doc[doc]
        year = doc.replace("Grampian-", "")
        print(f"{year:<22} {r['n']:>4}  {r['hit@1']:>8.4f}  {r['mrr@10']:>8.4f}")


def print_fp_counts(methods: dict[str, pd.DataFrame]) -> None:
    print("\n=== FP COUNTS (TP/FP2/FP3) ===")
    print(f"{'Method':<30} {'TP':>6} {'FP2':>6} {'FP3':>6}")
    print("-" * 48)
    for label, df in methods.items():
        fc = fp_counts(df)
        print(f"{label:<30} {fc['tp']:>6} {fc['fp2']:>6} {fc['fp3']:>6}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gold_map = load_gold_map()
    print(f"Loaded eval_set: {len(gold_map)} queries across {len(DOCS)} documents")

    dense_df  = load_method_df("dense_page_hits.csv",  OFF_ROOT, gold_map)
    bm25_df   = load_method_df("bm25_page_hits.csv",   OFF_ROOT, gold_map)
    hybrid_df = load_method_df("hybrid_page_hits.csv", OFF_ROOT, gold_map)
    boost_df  = load_method_df("hybrid_page_hits.csv", ON_ROOT,  gold_map)

    methods = {
        "Dense (MiniLM)":            dense_df,
        "BM25-only":                 bm25_df,
        "Hybrid (base)":             hybrid_df,
        "Hybrid + subsection boost": boost_df,
    }

    print_table_4_1(methods)
    print_per_difficulty(methods)
    print_per_doc(hybrid_df)
    print_fp_counts(methods)

    # Chunk vs page: chunk-level uses dense_page_hits.csv rank-1 chunk page WITHOUT
    # page deduplication (already deduplicated in CSV, so chunk@1 == page@1 for this file).
    # The original chunk hit (pre-dedup) is stored in per_query_results.json chunk_hit field.
    # We approximate chunk_h1 from per_query_results.json predicted_pages[0] vs gold.
    chunk_vs_page = {}
    for label, csv_name, abl_root in [
        ("Dense (MiniLM)",            "dense_page_hits.csv",  OFF_ROOT),
        ("BM25-only",                 "bm25_page_hits.csv",   OFF_ROOT),
        ("Hybrid (base)",             "hybrid_page_hits.csv", OFF_ROOT),
        ("Hybrid + subsection boost", "hybrid_page_hits.csv", ON_ROOT),
    ]:
        df = load_method_df(csv_name, abl_root, gold_map)
        ov = agg_overall(df)
        chunk_vs_page[label] = {"page_h1": ov["hit@1"], "page_h3": ov["hit@3"]}

    # Persist
    results = {
        "table_4_1": {
            label: {**agg_overall(df), **{"fp": fp_counts(df)}}
            for label, df in methods.items()
        },
        "per_document": agg_by_doc(hybrid_df),
        "per_difficulty": {
            label: agg_by_difficulty(df)
            for label, df in methods.items()
        },
        "chunk_vs_page": chunk_vs_page,
        "fp_counts": {label: fp_counts(df) for label, df in methods.items()},
    }

    out_path = OUTPUT_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
