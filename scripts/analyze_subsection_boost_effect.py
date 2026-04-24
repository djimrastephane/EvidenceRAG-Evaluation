from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("results/subsection_boost_on_off_2026-04-07")
DEFAULT_DOCS = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]
BOOTSTRAP_SEED = 42
BOOTSTRAP_REPS = 10000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze subsection boost ON vs OFF with bootstrap confidence intervals.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--docs", nargs="*", default=DEFAULT_DOCS)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    return parser.parse_args()


def load_results(path: Path) -> list[dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return list(obj["results"])


def paired_bootstrap_ci(values: np.ndarray, reps: int, seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(reps, n))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []

    for doc in args.docs:
        off = load_results(args.root / "off" / doc / "retrieval_results_hybrid.json")
        on = load_results(args.root / "on" / doc / "retrieval_results_hybrid.json")
        off_map = {r["query_id"]: r for r in off}
        on_map = {r["query_id"]: r for r in on}
        for qid in sorted(off_map):
            roff = off_map[qid]
            ron = on_map[qid]
            row = {
                "doc_id": doc,
                "query_id": qid,
                "failure_type_off": roff.get("failure_type"),
                "failure_type_on": ron.get("failure_type"),
                "fp2_off": 1 if str(roff.get("failure_type")) == "FP2_MISSED_TOP_RANK" else 0,
                "fp2_on": 1 if str(ron.get("failure_type")) == "FP2_MISSED_TOP_RANK" else 0,
            }
            for k in ("1", "3", "5", "10"):
                offk = roff["per_k"][k]
                onk = ron["per_k"][k]
                row[f"page_hit_off@{k}"] = float(offk["page_recall_at_k"] > 0)
                row[f"page_hit_on@{k}"] = float(onk["page_recall_at_k"] > 0)
                row[f"page_hit_delta@{k}"] = row[f"page_hit_on@{k}"] - row[f"page_hit_off@{k}"]
                row[f"page_mrr_off@{k}"] = float(offk["page_mrr_at_k"])
                row[f"page_mrr_on@{k}"] = float(onk["page_mrr_at_k"])
                row[f"page_mrr_delta@{k}"] = row[f"page_mrr_on@{k}"] - row[f"page_mrr_off@{k}"]
            rows.append(row)

    df = pd.DataFrame(rows)
    analysis_rows: list[dict[str, object]] = []
    report_lines = [
        "# Subsection Boost ON vs OFF: Paired Query Analysis",
        "",
        f"- Queries analyzed: `{len(df)}`",
        f"- Bootstrap replicates: `{args.bootstrap_reps}`",
        "",
    ]

    for k in ("1", "3", "5", "10"):
        hit_values = df[f"page_hit_delta@{k}"].to_numpy(dtype=float)
        mrr_values = df[f"page_mrr_delta@{k}"].to_numpy(dtype=float)
        hit_mean, hit_lo, hit_hi = paired_bootstrap_ci(hit_values, args.bootstrap_reps)
        mrr_mean, mrr_lo, mrr_hi = paired_bootstrap_ci(mrr_values, args.bootstrap_reps)
        wins = int((hit_values > 0).sum())
        losses = int((hit_values < 0).sum())
        unchanged = int((hit_values == 0).sum())
        mrr_wins = int((mrr_values > 0).sum())
        mrr_losses = int((mrr_values < 0).sum())
        analysis_rows.append(
            {
                "k": int(k),
                "page_hit_delta_mean": hit_mean,
                "page_hit_delta_ci_low": hit_lo,
                "page_hit_delta_ci_high": hit_hi,
                "page_hit_wins": wins,
                "page_hit_losses": losses,
                "page_hit_unchanged": unchanged,
                "page_mrr_delta_mean": mrr_mean,
                "page_mrr_delta_ci_low": mrr_lo,
                "page_mrr_delta_ci_high": mrr_hi,
                "page_mrr_wins": mrr_wins,
                "page_mrr_losses": mrr_losses,
            }
        )
        report_lines.extend(
            [
                f"## k={k}",
                f"- Page hit delta mean: `{hit_mean:+.4f}` with 95% bootstrap CI `[{hit_lo:+.4f}, {hit_hi:+.4f}]`",
                f"- Page hit wins/losses/unchanged: `{wins}` / `{losses}` / `{unchanged}`",
                f"- Page MRR delta mean: `{mrr_mean:+.4f}` with 95% bootstrap CI `[{mrr_lo:+.4f}, {mrr_hi:+.4f}]`",
                f"- Page MRR wins/losses: `{mrr_wins}` / `{mrr_losses}`",
                "",
            ]
        )

    fp2_off_rate = float(df["fp2_off"].mean())
    fp2_on_rate = float(df["fp2_on"].mean())
    fp2_delta = fp2_on_rate - fp2_off_rate
    fp2_off_count = int(df["fp2_off"].sum())
    fp2_on_count = int(df["fp2_on"].sum())
    fp2_values = (df["fp2_on"] - df["fp2_off"]).to_numpy(dtype=float)
    fp2_mean, fp2_lo, fp2_hi = paired_bootstrap_ci(fp2_values, args.bootstrap_reps)
    fp2_resolved = int(((df["fp2_off"] == 1) & (df["fp2_on"] == 0)).sum())
    fp2_created = int(((df["fp2_off"] == 0) & (df["fp2_on"] == 1)).sum())

    fp2_row = {
        "fp2_off_count": fp2_off_count,
        "fp2_on_count": fp2_on_count,
        "fp2_off_rate": fp2_off_rate,
        "fp2_on_rate": fp2_on_rate,
        "fp2_rate_delta": fp2_delta,
        "fp2_resolved_queries": fp2_resolved,
        "fp2_created_queries": fp2_created,
        "fp2_delta_ci_low": fp2_lo,
        "fp2_delta_ci_high": fp2_hi,
    }

    report_lines.extend(
        [
            "## FP2",
            f"- FP2 count changed from `{fp2_off_count}` to `{fp2_on_count}`",
            f"- FP2 rate changed from `{fp2_off_rate:.3%}` to `{fp2_on_rate:.3%}` (`{fp2_delta:+.3%}`)",
            f"- 95% bootstrap CI for FP2 rate delta: `[{fp2_lo:+.4f}, {fp2_hi:+.4f}]`",
            f"- Queries where FP2 was resolved: `{fp2_resolved}`",
            f"- Queries where FP2 was introduced: `{fp2_created}`",
            "",
        ]
    )

    out_dir = args.root
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(analysis_rows).to_csv(out_dir / "subsection_boost_query_bootstrap_summary.csv", index=False)
    pd.DataFrame([fp2_row]).to_csv(out_dir / "subsection_boost_fp2_summary.csv", index=False)
    df.to_csv(out_dir / "subsection_boost_query_level_deltas.csv", index=False)
    (out_dir / "subsection_boost_query_analysis.md").write_text("\n".join(report_lines), encoding="utf-8")
    print((out_dir / "subsection_boost_query_analysis.md").read_text())


if __name__ == "__main__":
    main()
