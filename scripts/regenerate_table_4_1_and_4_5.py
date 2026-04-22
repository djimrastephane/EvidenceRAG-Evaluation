"""
Recompute Table 4.1 and Table 4.5 from the canonical ablation outputs.

Canonical sources (all use the same post-fix 224/56 pipeline artifacts)
------------------------------------------------------------------------
Dense / BM25  (per-query hits from page-hit CSVs):
    OFF_ROOT/minilmcap_<doc>_chunk_224_56/<doc>/dense_page_hits.csv
    OFF_ROOT/minilmcap_<doc>_chunk_224_56/<doc>/bm25_page_hits.csv

Hybrid base   (from per_query_results.json — same numbers as retrieval_metrics.json):
    OFF_ROOT/minilmcap_<doc>_chunk_224_56/<doc>/per_query_results.json

Hybrid boost  (from per_query_results.json):
    ON_ROOT/minilmcap_<doc>_chunk_224_56/<doc>/per_query_results.json

Difficulty labels:
    data_processed/<doc>/eval_set.json

OFF_ROOT = results/thesis_ablations/chunk_size_ablation_boost_off_2026-04-20/pipeline_outputs
ON_ROOT  = results/thesis_ablations/chunk_size_ablation_2026-04-15/pipeline_outputs
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT    = Path(__file__).resolve().parents[1]
DOCS    = ["Grampian-2020-2021", "Grampian-2021-2022", "Grampian-2022-2023",
           "Grampian-2023-2024", "Grampian-2024-2025"]
OFF_ROOT = ROOT / "results" / "thesis_ablations" / "chunk_size_ablation_boost_off_2026-04-20" / "pipeline_outputs"
ON_ROOT  = ROOT / "results" / "thesis_ablations" / "chunk_size_ablation_2026-04-15"           / "pipeline_outputs"
KS = (1, 3, 5, 10)


# ── data loaders ─────────────────────────────────────────────────────────────

def load_gold_and_difficulty() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for doc in DOCS:
        data = json.loads((ROOT / "data_processed" / doc / "eval_set.json").read_text())
        for q in data["queries"]:
            out[q["query_id"]] = {
                "difficulty": q["difficulty"],
                "gold_pages": set(q["expected_pages"]),
            }
    return out


def hits_from_page_csv(hits_path: Path, gold_map: dict) -> list[dict]:
    """Compute hit@k + mrr@10 from a *_page_hits.csv ranked list."""
    df = pd.read_csv(hits_path, usecols=["query_id", "rank", "page_number"])
    rows = []
    for qid, grp in df.groupby("query_id"):
        gold = gold_map[qid]["gold_pages"]
        ranked = grp.sort_values("rank")["page_number"].tolist()
        first_rel = next((i + 1 for i, pg in enumerate(ranked) if pg in gold), None)
        row = {"query_id": qid,
               "hit@1":  1.0 if set(ranked[:1])  & gold else 0.0,
               "hit@3":  1.0 if set(ranked[:3])  & gold else 0.0,
               "hit@5":  1.0 if set(ranked[:5])  & gold else 0.0,
               "hit@10": 1.0 if set(ranked[:10]) & gold else 0.0,
               "mrr@10": (1.0 / first_rel) if first_rel and first_rel <= 10 else 0.0}
        rows.append(row)
    return rows


def hits_from_per_query_json(json_path: Path) -> list[dict]:
    """Compute hit@k + mrr@10 from per_query_results.json."""
    records = json.loads(json_path.read_text())
    rows = []
    for r in records:
        frr = r.get("first_relevant_rank")
        rows.append({
            "query_id": r["query_id"],
            "hit@1":    1.0 if r["hit_at_1"] else 0.0,
            "hit@3":    1.0 if r["hit_at_3"] else 0.0,
            "hit@5":    float(r.get("hit_at_5", r["hit_at_3"])),  # fallback
            "hit@10":   float(r.get("hit_at_10", r.get("reciprocal_rank", 0) > 0)),
            "mrr@10":   (1.0 / frr) if frr and frr <= 10 else 0.0,
        })
    return rows


def load_all_csv(filename: str, abl_root: Path, gold_map: dict) -> pd.DataFrame:
    frames = []
    for doc in DOCS:
        p = abl_root / f"minilmcap_{doc}_chunk_224_56" / doc / filename
        frames.append(pd.DataFrame(hits_from_page_csv(p, gold_map)))
    return pd.concat(frames, ignore_index=True)


def load_all_json(abl_root: Path, gold_map: dict) -> pd.DataFrame:
    frames = []
    for doc in DOCS:
        p = abl_root / f"minilmcap_{doc}_chunk_224_56" / doc / "per_query_results.json"
        rows = hits_from_per_query_json(p)
        df   = pd.DataFrame(rows)
        df["difficulty"] = df["query_id"].map(lambda qid: gold_map[qid]["difficulty"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ── aggregation ──────────────────────────────────────────────────────────────

def agg_overall(df: pd.DataFrame) -> dict:
    return {
        "H@1":    df["hit@1"].mean(),
        "H@3":    df["hit@3"].mean(),
        "H@5":    df.get("hit@5", df["hit@3"]).mean(),
        "MRR@10": df["mrr@10"].mean(),
        "N":      len(df),
    }


def agg_by_tier(df: pd.DataFrame) -> dict:
    tiers = {}
    for tier in ("LEX", "MOD", "STR"):
        sub = df[df["difficulty"] == tier]
        tiers[tier] = {"N": len(sub), "H@1": sub["hit@1"].mean(), "MRR@10": sub["mrr@10"].mean()}
    # All row
    tiers["All"] = {"N": len(df), "H@1": df["hit@1"].mean(), "MRR@10": df["mrr@10"].mean()}
    return tiers


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    gold_map = load_gold_and_difficulty()

    # Add difficulty to CSV-based frames
    def attach_diff(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["difficulty"] = df["query_id"].map(lambda qid: gold_map[qid]["difficulty"])
        return df

    dense_df        = attach_diff(load_all_csv("dense_page_hits.csv",  OFF_ROOT, gold_map))
    bm25_df         = attach_diff(load_all_csv("bm25_page_hits.csv",   OFF_ROOT, gold_map))
    hybrid_base_df  = load_all_json(OFF_ROOT, gold_map)   # already has difficulty
    hybrid_boost_df = load_all_json(ON_ROOT,  gold_map)

    methods = {
        "Dense (MiniLM)":            dense_df,
        "BM25-only":                 bm25_df,
        "Hybrid (base)":             hybrid_base_df,
        "Hybrid + subsection boost": hybrid_boost_df,
    }

    # ── Table 4.1 ──────────────────────────────────────────────────────────
    print("\n=== TABLE 4.1 ===")
    print(f"{'Method':<30} {'H@1':>8} {'H@3':>8} {'H@5':>8} {'MRR@10':>8}")
    print("-" * 60)
    for label, df in methods.items():
        ov = agg_overall(df)
        print(f"{label:<30} {ov['H@1']:>8.4f} {ov['H@3']:>8.4f} {ov['H@5']:>8.4f} {ov['MRR@10']:>8.4f}")

    # ── Table 4.5 ──────────────────────────────────────────────────────────
    print("\n=== TABLE 4.5 (boost-OFF: Dense / BM25 / Hybrid base) ===")
    print(f"{'Tier':<6} {'N':>4}  {'Dense H@1':>9} {'Dense MRR':>9}  "
          f"{'BM25 H@1':>8} {'BM25 MRR':>8}  {'Hybrid H@1':>10} {'Hybrid MRR':>10}")
    print("-" * 82)
    d_tiers = agg_by_tier(dense_df)
    b_tiers = agg_by_tier(bm25_df)
    h_tiers = agg_by_tier(hybrid_base_df)
    for tier in ("LEX", "MOD", "STR", "All"):
        d, b, h = d_tiers[tier], b_tiers[tier], h_tiers[tier]
        print(f"{tier:<6} {d['N']:>4}  {d['H@1']:>9.3f} {d['MRR@10']:>9.3f}  "
              f"{b['H@1']:>8.3f} {b['MRR@10']:>8.3f}  {h['H@1']:>10.3f} {h['MRR@10']:>10.3f}")

    # Save
    out = ROOT / "results" / "table_4_1_4_5_recomputed.json"
    payload = {label: {"overall": agg_overall(df), "tiers": agg_by_tier(df)}
               for label, df in methods.items()}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
