"""
Compare retrieval metrics before/after mixed-routing reprocessing.

Steps:
  1. Back up existing retrieval_results.json (pre-mixed-routing baseline).
  2. Re-run retrieval_eval.py for each eval doc.
  3. Load old vs new per-query results and compute McNemar + Wilcoxon tests.

Usage:
    python scripts/eval_mixed_routing_impact.py [--data-dir data_processed] [--device mps]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

EVAL_DOCS = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]

KS = [1, 3, 5, 10]


def backup_old_results(data_dir: Path, doc_id: str) -> dict | None:
    src = data_dir / doc_id / "retrieval_results.json"
    if not src.exists():
        return None
    bak = data_dir / doc_id / "retrieval_results_pre_mixed.json"
    if not bak.exists():
        shutil.copy2(src, bak)
        print(f"  Backed up {src.name} → retrieval_results_pre_mixed.json")
    else:
        print(f"  Backup already exists for {doc_id}, skipping copy.")
    with open(bak) as f:
        return json.load(f)


def run_eval(data_dir: Path, doc_id: str, device: str) -> dict:
    doc_dir = data_dir / doc_id
    cmd = [
        sys.executable,
        "scripts/retrieval_eval.py",
        "--data-dir", str(doc_dir),
        "--device", device,
    ]
    print(f"  Running eval for {doc_id}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR:\n{result.stderr[-2000:]}")
        sys.exit(1)
    results_path = doc_dir / "retrieval_results.json"
    with open(results_path) as f:
        return json.load(f)


def extract_per_query(results_json: dict, metric: str, k: int) -> np.ndarray:
    """Return array of per-query metric values at given k."""
    k_str = str(k)
    vals = []
    for r in results_json["results"]:
        pk = r.get("per_k", {}).get(k_str, {})
        vals.append(float(pk.get(metric, 0.0)))
    return np.array(vals)


def mcnemar_test(old_hits: np.ndarray, new_hits: np.ndarray) -> tuple[float, float]:
    """McNemar test on binary hit arrays. Returns (statistic, p_value)."""
    b = int(((old_hits == 0) & (new_hits == 1)).sum())  # old miss, new hit
    c = int(((old_hits == 1) & (new_hits == 0)).sum())  # old hit, new miss
    n = b + c
    if n == 0:
        return 0.0, 1.0
    # Exact binomial when n < 25, else normal approximation with continuity correction
    if n < 25:
        p = 2 * stats.binom.cdf(min(b, c), n, 0.5)
        return float(abs(b - c)), float(p)
    chi2 = (abs(b - c) - 1) ** 2 / n
    p = 1 - stats.chi2.cdf(chi2, df=1)
    return float(chi2), float(p)


def wilcoxon_test(old_vals: np.ndarray, new_vals: np.ndarray) -> tuple[float, float]:
    """Wilcoxon signed-rank test on paired continuous arrays."""
    diff = new_vals - old_vals
    if np.all(diff == 0):
        return 0.0, 1.0
    stat, p = stats.wilcoxon(diff, alternative="greater", zero_method="wilcox")
    return float(stat), float(p)


def run_comparison(old: dict, new: dict, doc_id: str) -> list[dict]:
    rows = []
    n = len(old["results"])
    for k in KS:
        for metric, test_fn, label in [
            ("page_recall_at_k", "mcnemar", "PageHit"),
            ("chunk_hit_at_k",   "mcnemar", "ChunkHit"),
            ("page_mrr_at_k",    "wilcoxon", "PageMRR"),
            ("chunk_mrr_at_k",   "wilcoxon", "ChunkMRR"),
        ]:
            old_v = extract_per_query(old, metric, k)
            new_v = extract_per_query(new, metric, k)
            old_mean = old_v.mean()
            new_mean = new_v.mean()
            delta = new_mean - old_mean

            if test_fn == "mcnemar":
                stat, p = mcnemar_test((old_v > 0).astype(float), (new_v > 0).astype(float))
            else:
                stat, p = wilcoxon_test(old_v, new_v)

            rows.append({
                "doc_id": doc_id,
                "k": k,
                "metric": label,
                "n_queries": n,
                "old_mean": round(old_mean, 4),
                "new_mean": round(new_mean, 4),
                "delta": round(delta, 4),
                "stat": round(stat, 4),
                "p_value": round(p, 4),
                "sig_05": p < 0.05,
                "sig_10": p < 0.10,
            })
    return rows


def aggregate_pooled(all_rows: list[dict]) -> list[dict]:
    """Pool all docs together for a single omnibus test per (k, metric)."""
    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_rows:
        groups[(r["k"], r["metric"])].append(r)

    pooled = []
    for (k, metric), group_rows in sorted(groups.items()):
        n_total = sum(r["n_queries"] for r in group_rows)
        old_mean = np.mean([r["old_mean"] for r in group_rows])
        new_mean = np.mean([r["new_mean"] for r in group_rows])
        delta = new_mean - old_mean
        # pool per-query arrays from each doc for omnibus test
        doc_ids = [r["doc_id"] for r in group_rows]

        pooled.append({
            "doc_id": "POOLED(" + ",".join(d.split("-")[0] for d in set(doc_ids)) + ")",
            "k": k,
            "metric": metric,
            "n_queries": n_total,
            "old_mean": round(old_mean, 4),
            "new_mean": round(new_mean, 4),
            "delta": round(delta, 4),
            "stat": None,
            "p_value": None,
            "sig_05": None,
            "sig_10": None,
        })
    return pooled


def print_table(df: pd.DataFrame, title: str) -> None:
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")
    print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_processed")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip re-running retrieval_eval (use existing new results)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    all_rows: list[dict] = []

    for doc_id in EVAL_DOCS:
        print(f"\n--- {doc_id} ---")
        old_data = backup_old_results(data_dir, doc_id)
        if old_data is None:
            print(f"  No baseline retrieval_results.json, skipping.")
            continue

        if not args.skip_eval:
            new_data = run_eval(data_dir, doc_id, args.device)
        else:
            new_path = data_dir / doc_id / "retrieval_results.json"
            with open(new_path) as f:
                new_data = json.load(f)

        rows = run_comparison(old_data, new_data, doc_id)
        all_rows.extend(rows)

    if not all_rows:
        print("No comparison data collected.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)

    # Per-doc summary at k=5 and k=10
    for k in [5, 10]:
        sub = df[df["k"] == k][["doc_id","metric","old_mean","new_mean","delta","p_value","sig_05"]]
        print_table(sub, f"Per-doc results at k={k}")

    # Pooled delta summary (avg across docs)
    print(f"\n{'='*72}")
    print("  Pooled deltas (mean across 5 docs)")
    print(f"{'='*72}")
    pooled = (
        df.groupby(["k", "metric"])
        .agg(
            old_mean=("old_mean", "mean"),
            new_mean=("new_mean", "mean"),
            delta=("delta", "mean"),
            n_sig_05=("sig_05", "sum"),
            n_docs=("doc_id", "count"),
        )
        .reset_index()
    )
    pooled["old_mean"] = pooled["old_mean"].round(4)
    pooled["new_mean"] = pooled["new_mean"].round(4)
    pooled["delta"] = pooled["delta"].round(4)
    print(pooled.to_string(index=False))

    # Save full results
    out_path = data_dir / "mixed_routing_eval_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"\nFull results saved to: {out_path}")

    # Summary verdict
    print(f"\n{'='*72}")
    print("  STATISTICAL VERDICT")
    print(f"{'='*72}")
    sig = df[df["sig_05"] == True]
    if sig.empty:
        print("No statistically significant improvements at α=0.05 across any doc/k/metric.")
    else:
        print(f"{len(sig)} significant results at α=0.05:")
        print(sig[["doc_id","k","metric","delta","p_value"]].to_string(index=False))

    marginal = df[(df["sig_10"] == True) & (df["sig_05"] == False)]
    if not marginal.empty:
        print(f"\n{len(marginal)} marginal results at α=0.10 (not 0.05):")
        print(marginal[["doc_id","k","metric","delta","p_value"]].to_string(index=False))

    pos_delta = df[df["delta"] > 0]
    neg_delta = df[df["delta"] < 0]
    print(f"\nDirection: {len(pos_delta)} metric/k/doc combos improved, "
          f"{len(neg_delta)} declined, "
          f"{len(df) - len(pos_delta) - len(neg_delta)} unchanged.")


if __name__ == "__main__":
    main()