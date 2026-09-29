"""Regenerate the README retrieval table from the frozen gold-evaluation artifacts.

Runs retrieval_eval_hybrid.py for the five gold-labelled Grampian documents under the
production configuration and three baselines, verifying every run against the frozen
artifact manifest, then writes a summary with Wilson intervals and paired exact McNemar
tests against the production configuration.

The production run writes to data_processed/<doc>/ (the canonical, tracked location);
baselines write to <out-root>/<variant>/<doc>/.

Usage:
  python scripts/reproduce_readme_retrieval_table.py
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from scipy.stats import binomtest

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLD_DOCS = [f"Grampian-{y}-{y + 1}" for y in range(2020, 2025)]
KS = ["1", "3", "5", "10"]

# All variants use RRF k=20, dense 0.5 / BM25 2.0 unless overridden, and no oracle labels.
VARIANTS: dict[str, dict] = {
    "production": {"label": "Hybrid RRF + lexical rerank (production)", "args": []},
    "rrf_no_rerank": {"label": "Hybrid RRF, no rerank", "args": ["--no-lexical-rerank"]},
    "dense": {
        "label": "Dense only (MiniLM)",
        "args": ["--no-lexical-rerank", "--dense-weight", "1", "--bm25-weight", "0"],
    },
    "bm25": {
        "label": "BM25 only",
        "args": ["--no-lexical-rerank", "--dense-weight", "0", "--bm25-weight", "1"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data_processed"))
    parser.add_argument("--out-root", default=str(REPO_ROOT / "results" / "readme_retrieval_table"))
    parser.add_argument(
        "--manifest", default=str(REPO_ROOT / "results" / "frozen" / "gold_eval_artifacts_manifest.json")
    )
    parser.add_argument("--model", default="models/all-MiniLM-L6-v2")
    parser.add_argument("--skip-runs", action="store_true", help="Only rebuild the summary from existing outputs.")
    return parser.parse_args()


def output_dir(variant: str, doc: str, data_root: Path, out_root: Path) -> Path:
    return data_root / doc if variant == "production" else out_root / variant / doc


def run_variant(variant: str, doc: str, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "retrieval_eval_hybrid.py"),
        "--data-dir", str(Path(args.data_root) / doc),
        "--model", args.model,
        "--output-dir", str(output_dir(variant, doc, Path(args.data_root), Path(args.out_root))),
        "--verify-manifest", args.manifest,
        "--no-subsection-boost",
        *VARIANTS[variant]["args"],
    ]
    print(f"[run] {variant} {doc}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, stdout=subprocess.DEVNULL)


def load_per_query(variant: str, args: argparse.Namespace) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for doc in GOLD_DOCS:
        path = output_dir(variant, doc, Path(args.data_root), Path(args.out_root)) / "retrieval_results_hybrid.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["run_info"].get("oracle_labels_in_ranking"):
            raise ValueError(f"{path} was produced with oracle labels in ranking")
        for r in payload["results"]:
            pk = r["per_k"]
            rows[r["query_id"]] = {
                "doc": doc,
                **{f"hit@{k}": float(pk[k]["page_recall_at_k"] > 0) for k in KS},
                "mrr@10": float(pk["10"]["page_mrr_at_k"]),
            }
    return rows


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def summarise(args: argparse.Namespace) -> dict:
    per_query = {v: load_per_query(v, args) for v in VARIANTS}
    qids = sorted(per_query["production"])
    for v, rows in per_query.items():
        if sorted(rows) != qids:
            raise ValueError(f"{v}: query set differs from production")
    n = len(qids)

    per_doc = []
    for doc in GOLD_DOCS:
        dq = [q for q in qids if per_query["production"][q]["doc"] == doc]
        per_doc.append(
            {
                "doc": doc,
                "n": len(dq),
                **{m: mean(per_query["production"][q][m] for q in dq) for m in [*(f"hit@{k}" for k in KS), "mrr@10"]},
            }
        )

    methods = []
    for v, spec in VARIANTS.items():
        rows = per_query[v]
        entry = {"variant": v, "label": spec["label"], "n": n}
        for m in [*(f"hit@{k}" for k in KS), "mrr@10"]:
            entry[m] = mean(rows[q][m] for q in qids)
        for k in ("1", "5"):
            entry[f"hit@{k}_wilson95"] = wilson(entry[f"hit@{k}"], n)
        if v != "production":
            for k in ("1", "5"):
                m = f"hit@{k}"
                prod_only = sum(1 for q in qids if per_query["production"][q][m] > rows[q][m])
                other_only = sum(1 for q in qids if rows[q][m] > per_query["production"][q][m])
                p = binomtest(prod_only, prod_only + other_only, 0.5).pvalue if prod_only + other_only else 1.0
                entry[f"vs_production_{m}"] = {
                    "delta_production_minus_variant": mean(per_query["production"][q][m] for q in qids) - entry[m],
                    "production_only_hits": prod_only,
                    "variant_only_hits": other_only,
                    "mcnemar_exact_p": p,
                }
        methods.append(entry)

    return {"n_queries": n, "docs": GOLD_DOCS, "manifest": args.manifest, "per_doc_production": per_doc, "methods": methods}


def to_markdown(summary: dict) -> str:
    fmt = lambda x: f"{x:.3f}"
    lines = [
        "| Document | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 |",
        "|---|---|---|---|---|---|",
    ]
    for row in summary["per_doc_production"]:
        name = row["doc"].replace("Grampian-", "Grampian ").replace("-", "–")
        lines.append(f"| {name} | " + " | ".join(fmt(row[m]) for m in ["hit@1", "hit@3", "hit@5", "hit@10", "mrr@10"]) + " |")
    prod = next(m for m in summary["methods"] if m["variant"] == "production")
    lines.append("| **Mean** | " + " | ".join(f"**{fmt(prod[m])}**" for m in ["hit@1", "hit@3", "hit@5", "hit@10", "mrr@10"]) + " |")
    lines += [
        "",
        f"| Method (n = {summary['n_queries']}) | Hit@1 [95% Wilson] | Hit@5 [95% Wilson] | Hit@10 | MRR@10 | Hit@1 vs production (exact McNemar) |",
        "|---|---|---|---|---|---|",
    ]
    for m in summary["methods"]:
        lo1, hi1 = m["hit@1_wilson95"]
        lo5, hi5 = m["hit@5_wilson95"]
        cmp = m.get("vs_production_hit@1")
        cmp_txt = (
            "—"
            if cmp is None
            else f"{-cmp['delta_production_minus_variant']:+.3f} ({cmp['variant_only_hits']}/{cmp['production_only_hits']}, p = {cmp['mcnemar_exact_p']:.3f})"
        )
        lines.append(
            f"| {m['label']} | {fmt(m['hit@1'])} [{fmt(lo1)}, {fmt(hi1)}] | {fmt(m['hit@5'])} [{fmt(lo5)}, {fmt(hi5)}] "
            f"| {fmt(m['hit@10'])} | {fmt(m['mrr@10'])} | {cmp_txt} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "freeze_eval_artifacts.py"), "--verify", "--manifest", args.manifest],
        check=True,
        cwd=REPO_ROOT,
    )
    if not args.skip_runs:
        for variant in VARIANTS:
            for doc in GOLD_DOCS:
                run_variant(variant, doc, args)
    summary = summarise(args)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_root / "summary.md").write_text(to_markdown(summary), encoding="utf-8")
    print(to_markdown(summary))


if __name__ == "__main__":
    main()
