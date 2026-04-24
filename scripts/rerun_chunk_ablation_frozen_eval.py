#!/usr/bin/env python3
"""Recompute chunk-size ablation using frozen artifacts + current eval_set.json.

Reads pre-computed hybrid_page_hits.csv from each chunk-config frozen artifact
directory (224/56, 256/64, 280/90, 400/100) and re-evaluates against the
current data_processed/eval_set.json gold pages.

Output: results/rerun_chunk_ablation_2026-04-24/results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT      = Path(__file__).resolve().parents[1]
DOCS      = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]
OFF_ROOT  = ROOT / "results" / "thesis_ablations" / "chunk_size_ablation_boost_off_2026-04-20" / "pipeline_outputs"
EVAL_ROOT = ROOT / "data_processed"
OUTPUT_DIR = ROOT / "results" / "rerun_chunk_ablation_2026-04-24"

CONFIGS = [
    ("224 / 56",  "224_56"),
    ("256 / 64",  "256_64"),
    ("280 / 90",  "280_90"),
    ("400 / 100", "400_100"),
]


def load_gold_map() -> dict[str, dict]:
    gold: dict[str, dict] = {}
    for doc in DOCS:
        data = json.loads((EVAL_ROOT / doc / "eval_set.json").read_text())
        for q in data["queries"]:
            gold[q["query_id"]] = {"gold_pages": set(q["expected_pages"])}
    return gold


def hits_from_csv(path: Path, gold_map: dict) -> list[dict]:
    df = pd.read_csv(path, usecols=["query_id", "rank", "page_number"])
    rows = []
    for qid, grp in df.groupby("query_id"):
        if qid not in gold_map:
            continue
        gold = gold_map[qid]["gold_pages"]
        ranked = grp.sort_values("rank")["page_number"].tolist()
        first_rel = next((i + 1 for i, pg in enumerate(ranked) if pg in gold), None)
        rows.append({
            "query_id": qid,
            "hit@1":  1.0 if set(ranked[:1]) & gold else 0.0,
            "mrr@10": (1.0 / first_rel) if first_rel and first_rel <= 10 else 0.0,
        })
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gold_map = load_gold_map()

    print("\n=== CHUNK SIZE ABLATION (frozen artifacts + current eval_set) ===")
    print(f"{'Config':<12} {'H@1':>8} {'MRR@10':>8} {'N':>6}")
    print("-" * 38)

    results = {}
    for label, suffix in CONFIGS:
        frames = []
        for doc in DOCS:
            p = OFF_ROOT / f"minilmcap_{doc}_chunk_{suffix}" / doc / "hybrid_page_hits.csv"
            frames.append(pd.DataFrame(hits_from_csv(p, gold_map)))
        df = pd.concat(frames, ignore_index=True)
        h1  = round(df["hit@1"].mean(), 4)
        mrr = round(df["mrr@10"].mean(), 4)
        n   = len(df)
        print(f"{label:<12} {h1:>8.4f} {mrr:>8.4f} {n:>6}")
        results[label] = {"hit@1": h1, "mrr@10": mrr, "n": n}

    out_path = OUTPUT_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
