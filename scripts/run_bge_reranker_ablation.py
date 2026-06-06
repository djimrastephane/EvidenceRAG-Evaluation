"""
Reranker ablation using BAAI/bge-reranker-v2-m3.

Identical parameters to run_cross_encoder_ablation.py (same top-N depths,
fusion weights, documents, and eval script) so results are directly
comparable. The only change is the reranker model.

BGE-Reranker-v2-m3 is a multilingual XLM-RoBERTa cross-encoder trained on
a broader dataset than MS MARCO MiniLM. Scores are raw logits; the eval
pipeline min-max normalises them before fusion, so the same weights apply.

Configurations tested
---------------------
baseline      : hybrid, no cross-encoder, no subsection boost
ce_topn5_w02  : BGE top-5,  weight=0.2
ce_topn10_w02 : BGE top-10, weight=0.2
ce_topn20_w02 : BGE top-20, weight=0.2  (primary comparison)
ce_topn20_w01 : BGE top-20, weight=0.1
ce_topn20_w03 : BGE top-20, weight=0.3

Results are written to results/bge_reranker_ablation_<date>/.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_BASE = ROOT / "data_processed"
EVAL_SCRIPT = ROOT / "scripts" / "retrieval_eval_hybrid.py"
_CONDA_PYTHON = Path("/opt/anaconda3/envs/rag-pipeline/bin/python")
PYTHON = str(_CONDA_PYTHON) if _CONDA_PYTHON.exists() else sys.executable

DOC_IDS = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]

CE_MODEL = "models/bge-reranker-v2-m3"

CONFIGS = {
    "baseline": {
        "label": "Hybrid (base)",
        "ce": False,
        "ce_topn": 20,
        "ce_weight": 0.2,
        "subsection_boost": False,
    },
    "bge_topn5_w02": {
        "label": "BGE top-5 w=0.2",
        "ce": True,
        "ce_topn": 5,
        "ce_weight": 0.2,
        "subsection_boost": False,
    },
    "bge_topn10_w02": {
        "label": "BGE top-10 w=0.2",
        "ce": True,
        "ce_topn": 10,
        "ce_weight": 0.2,
        "subsection_boost": False,
    },
    "bge_topn20_w02": {
        "label": "BGE top-20 w=0.2",
        "ce": True,
        "ce_topn": 20,
        "ce_weight": 0.2,
        "subsection_boost": False,
    },
    "bge_topn20_w01": {
        "label": "BGE top-20 w=0.1",
        "ce": True,
        "ce_topn": 20,
        "ce_weight": 0.1,
        "subsection_boost": False,
    },
    "bge_topn20_w03": {
        "label": "BGE top-20 w=0.3",
        "ce": True,
        "ce_topn": 20,
        "ce_weight": 0.3,
        "subsection_boost": False,
    },
}

K_LIST = [1, 3, 5, 10]


def build_cmd(data_dir: Path, cfg: dict) -> list[str]:
    cmd = [
        PYTHON, str(EVAL_SCRIPT),
        "--data-dir", str(data_dir),
        "--model", str(ROOT / "models" / "all-MiniLM-L6-v2"),
        "--cross-encoder-model", CE_MODEL,
        "--cross-encoder-topn", str(cfg["ce_topn"]),
        "--cross-encoder-weight", str(cfg["ce_weight"]),
    ]
    if cfg["ce"]:
        cmd.append("--enable-cross-encoder-rerank")
    return cmd


def run_config(cfg_name: str, cfg: dict, out_dir: Path) -> dict:
    results_by_doc = {}
    for doc_id in DOC_IDS:
        data_dir = DATA_BASE / doc_id
        env = os.environ.copy()
        env["ENABLE_SUBSECTION_BOOST"] = "1" if cfg["subsection_boost"] else "0"
        env["SUBSECTION_BOOST"] = "0.05" if cfg["subsection_boost"] else "0.0"
        env["CE_DEVICE"] = "cpu"  # BGE-v2-m3 exceeds MPS memory budget; CPU is stable

        cmd = build_cmd(data_dir, cfg)
        print(f"  [{doc_id}] {cfg['label']} ...", flush=True)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(ROOT),
        )
        if result.returncode != 0:
            print(f"    ERROR: {result.stderr[-500:]}")
            results_by_doc[doc_id] = None
            continue

        metrics_path = data_dir / "retrieval_metrics_hybrid.json"
        if not metrics_path.exists():
            print(f"    WARN: metrics file not found at {metrics_path}")
            results_by_doc[doc_id] = None
            continue

        with metrics_path.open() as f:
            metrics = json.load(f)
        results_by_doc[doc_id] = metrics

        cfg_out = out_dir / cfg_name / doc_id
        cfg_out.mkdir(parents=True, exist_ok=True)
        with (cfg_out / "retrieval_metrics_hybrid.json").open("w") as f:
            json.dump(metrics, f, indent=2)

    return results_by_doc


def aggregate(results_by_doc: dict) -> dict:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for doc_id, metrics in results_by_doc.items():
        if metrics is None:
            continue
        by_k = metrics.get("metrics_by_k", {})
        for k in K_LIST:
            bucket = by_k.get(str(k), {})
            val = bucket.get("page_hit_rate_at_k")
            if val is not None:
                key = f"page_hit_at_{k}"
                totals[key] = totals.get(key, 0.0) + float(val)
                counts[key] = counts.get(key, 0) + 1
            mrr = bucket.get("mean_page_mrr_at_k")
            if mrr is not None and k == 10:
                totals["mrr_at_10"] = totals.get("mrr_at_10", 0.0) + float(mrr)
                counts["mrr_at_10"] = counts.get("mrr_at_10", 0) + 1
    return {k: totals[k] / counts[k] for k in totals if counts[k] > 0}


def build_summary_table(all_results: dict) -> pd.DataFrame:
    rows = []
    for cfg_name, cfg in CONFIGS.items():
        agg = all_results.get(cfg_name, {})
        rows.append({
            "config": cfg_name,
            "label": cfg["label"],
            "Page Hit@1": round(agg.get("page_hit_at_1", float("nan")), 4),
            "Page Hit@3": round(agg.get("page_hit_at_3", float("nan")), 4),
            "Page Hit@5": round(agg.get("page_hit_at_5", float("nan")), 4),
            "MRR@10":     round(agg.get("mrr_at_10",     float("nan")), 4),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = ROOT / "results" / f"bge_reranker_ablation_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {out_dir}")
    print(f"Reranker model: {CE_MODEL}")
    print(f"Documents: {DOC_IDS}")
    print(f"Configs: {list(CONFIGS.keys())}\n")

    all_results: dict[str, dict] = {}
    for cfg_name, cfg in CONFIGS.items():
        print(f"\n=== Config: {cfg['label']} ===")
        results_by_doc = run_config(cfg_name, cfg, out_dir)
        all_results[cfg_name] = aggregate(results_by_doc)

        (out_dir / cfg_name / "summary.json").write_text(
            json.dumps(
                {"config": cfg, "aggregated": all_results[cfg_name], "by_doc": {
                    doc: m for doc, m in results_by_doc.items() if m is not None
                }},
                indent=2,
            )
        )

    summary_df = build_summary_table(all_results)
    print("\n\n=== RESULTS SUMMARY ===")
    print(summary_df.to_string(index=False))

    summary_df.to_csv(out_dir / "bge_reranker_ablation_summary.csv", index=False)
    with (out_dir / "bge_reranker_ablation_results.json").open("w") as f:
        json.dump(all_results, f, indent=2)

    baseline = all_results.get("baseline", {})
    print("\n=== DELTA vs HYBRID BASE ===")
    for cfg_name, cfg in CONFIGS.items():
        if cfg_name == "baseline":
            continue
        agg = all_results.get(cfg_name, {})
        h1 = agg.get("page_hit_at_1", float("nan"))
        b1 = baseline.get("page_hit_at_1", float("nan"))
        mrr = agg.get("mrr_at_10", float("nan"))
        bm = baseline.get("mrr_at_10", float("nan"))
        print(f"  {cfg['label']:<35} Hit@1 {h1 - b1:+.4f}   MRR {mrr - bm:+.4f}")

    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
