from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCS = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]
DEFAULT_MODEL = "models/all-MiniLM-L6-v2"
DEFAULT_RUN_NAME = f"bm25_tokenizer_sensitivity_{datetime.now(timezone.utc).strftime('%Y%m%d')}"


@dataclass(frozen=True)
class RunSpec:
    setup: str
    tokenizer: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a small BM25 lexical-tokenization sensitivity check.")
    p.add_argument("--docs", default=",".join(DEFAULT_DOCS), help="Comma-separated document ids.")
    p.add_argument("--model-path", default=DEFAULT_MODEL, help="Dense model path for hybrid runs.")
    p.add_argument(
        "--tokenizers",
        default="default,no_hyphen",
        help="Comma-separated BM25 tokenizer variants to compare.",
    )
    p.add_argument(
        "--setups",
        default="bm25,hybrid",
        help="Comma-separated setups to run: bm25,hybrid",
    )
    p.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="Results subfolder name.")
    return p.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


@contextmanager
def preserve_outputs(doc_dirs: list[Path]) -> Any:
    backup_root = REPO_ROOT / "results" / "_tmp_bm25_tokenizer_backup"
    if backup_root.exists():
        shutil.rmtree(backup_root)
    backup_root.mkdir(parents=True, exist_ok=True)
    filenames = [
        "retrieval_results_bm25.json",
        "retrieval_metrics_bm25.json",
        "retrieval_summary_bm25.csv",
        "retrieval_results_hybrid.json",
        "retrieval_metrics_hybrid.json",
        "retrieval_summary_hybrid.csv",
    ]
    saved: list[tuple[Path, Path]] = []
    try:
        for doc_dir in doc_dirs:
            rel_dir = doc_dir.relative_to(REPO_ROOT)
            backup_dir = backup_root / rel_dir
            backup_dir.mkdir(parents=True, exist_ok=True)
            for name in filenames:
                src = doc_dir / name
                if src.exists():
                    dst = backup_dir / name
                    shutil.copy2(src, dst)
                    saved.append((src, dst))
        yield
    finally:
        for doc_dir in doc_dirs:
            for name in filenames:
                target = doc_dir / name
                if target.exists():
                    target.unlink()
        for target, backup in saved:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, target)
        if backup_root.exists():
            shutil.rmtree(backup_root)


def build_command(spec: RunSpec, data_dir: Path, model_path: str) -> list[str]:
    if spec.setup == "bm25":
        return [
            "python",
            "scripts/retrieval_eval_bm25.py",
            "--data-dir",
            str(data_dir),
            "--bm25-tokenizer",
            spec.tokenizer,
        ]
    if spec.setup == "hybrid":
        return [
            "python",
            "scripts/retrieval_eval_hybrid.py",
            "--data-dir",
            str(data_dir),
            "--model",
            str(model_path),
            "--bm25-tokenizer",
            spec.tokenizer,
        ]
    raise ValueError(f"Unsupported setup: {spec.setup}")


def load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_results(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_run(
    metrics_payloads: list[dict[str, Any]],
    results_payloads: list[dict[str, Any]],
    runtime_seconds: float,
    spec: RunSpec,
) -> dict[str, Any]:
    num_queries = 0
    hit1_sum = 0.0
    hit3_sum = 0.0
    mrr10_sum = 0.0
    fp2_count = 0
    for metrics in metrics_payloads:
        m1 = metrics["metrics_by_k"]["1"]
        m3 = metrics["metrics_by_k"]["3"]
        m10 = metrics["metrics_by_k"]["10"]
        nq = int(m1["num_queries"])
        num_queries += nq
        hit1_sum += float(m1["page_hit_rate_at_k"]) * nq
        hit3_sum += float(m3["page_hit_rate_at_k"]) * nq
        mrr10_sum += float(m10["mean_page_mrr_at_k"]) * nq
    for results in results_payloads:
        fp2_count += sum(
            1
            for row in results.get("results", [])
            if str(row.get("failure_type") or "") == "FP2_MISSED_TOP_RANK"
        )
    denom = max(1, num_queries)
    return {
        "setup": spec.setup,
        "bm25_tokenizer": spec.tokenizer,
        "Hit@1": round(hit1_sum / denom, 4),
        "Hit@3": round(hit3_sum / denom, 4),
        "MRR@10": round(mrr10_sum / denom, 4),
        "FP2 count": int(fp2_count),
        "runtime_seconds": round(float(runtime_seconds), 3),
        "num_queries": int(num_queries),
    }


def render_markdown(rows: list[dict[str, Any]]) -> str:
    headers = ["setup", "bm25_tokenizer", "Hit@1", "Hit@3", "MRR@10", "FP2 count", "runtime_seconds"]
    out = [
        "# BM25 Tokenizer Sensitivity",
        "",
        "| " + " | ".join(headers) + " |",
        "|---|---:|---:|---:|---:|---:|---:|".replace("---:|---:", "---|---").replace("setup", "setup"),
    ]
    out[3] = "|---|---|---:|---:|---:|---:|---:|"
    for row in rows:
        out.append(
            "| "
            + " | ".join(str(row[h]) for h in headers)
            + " |"
        )
    out.extend(
        [
            "",
            "Interpretation:",
            "- This sensitivity check changes only the BM25 lexical tokenizer while keeping the five-document evaluation set, 224/56 chunking, dense model, BM25 parameters, FAISS search, and scoring logic fixed.",
        ]
    )
    return "\n".join(out) + "\n"


def main() -> None:
    args = parse_args()
    docs = parse_csv_list(args.docs)
    tokenizers = parse_csv_list(args.tokenizers)
    setups = parse_csv_list(args.setups)
    specs = [RunSpec(setup=s, tokenizer=t) for s in setups for t in tokenizers]
    for spec in specs:
        if spec.setup not in {"bm25", "hybrid"}:
            raise ValueError(f"Unsupported setup '{spec.setup}'. Expected bm25 or hybrid.")
        if spec.tokenizer not in {"default", "no_hyphen"}:
            raise ValueError(f"Unsupported tokenizer '{spec.tokenizer}'. Expected default or no_hyphen.")

    doc_dirs = [REPO_ROOT / "data_processed" / doc for doc in docs]
    for doc_dir in doc_dirs:
        if not doc_dir.exists():
            raise FileNotFoundError(f"Missing data dir: {doc_dir}")

    out_dir = REPO_ROOT / "results" / "bm25_tokenizer_sensitivity" / args.run_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "run_utc": utc_now_iso(),
        "docs": docs,
        "model_path": str(args.model_path),
        "specs": [spec.__dict__ for spec in specs],
        "commands": [],
        "failures": [],
    }
    summary_rows: list[dict[str, Any]] = []

    run_env = os.environ.copy()
    run_env.setdefault("TOKENIZERS_PARALLELISM", "false")
    run_env.setdefault("OMP_NUM_THREADS", "1")
    run_env.setdefault("MKL_NUM_THREADS", "1")

    with preserve_outputs(doc_dirs):
        for spec in specs:
            metrics_payloads: list[dict[str, Any]] = []
            results_payloads: list[dict[str, Any]] = []
            setup_dir = out_dir / f"{spec.setup}_{spec.tokenizer}"
            setup_dir.mkdir(parents=True, exist_ok=True)
            total_runtime = 0.0
            failed_docs: list[str] = []
            for doc_dir in doc_dirs:
                cmd = build_command(spec, doc_dir, str(args.model_path))
                manifest["commands"].append({"doc_id": doc_dir.name, "setup": spec.setup, "tokenizer": spec.tokenizer, "cmd": cmd})
                started = time.perf_counter()
                try:
                    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, env=run_env)
                except subprocess.CalledProcessError as exc:
                    manifest["failures"].append(
                        {
                            "doc_id": doc_dir.name,
                            "setup": spec.setup,
                            "tokenizer": spec.tokenizer,
                            "returncode": int(exc.returncode),
                            "cmd": cmd,
                        }
                    )
                    failed_docs.append(doc_dir.name)
                    continue
                elapsed = time.perf_counter() - started
                total_runtime += elapsed

                if spec.setup == "bm25":
                    metrics_name = "retrieval_metrics_bm25.json"
                    results_name = "retrieval_results_bm25.json"
                    summary_name = "retrieval_summary_bm25.csv"
                else:
                    metrics_name = "retrieval_metrics_hybrid.json"
                    results_name = "retrieval_results_hybrid.json"
                    summary_name = "retrieval_summary_hybrid.csv"

                metrics_path = doc_dir / metrics_name
                results_path = doc_dir / results_name
                summary_path = doc_dir / summary_name
                metrics_payload = load_metrics(metrics_path)
                results_payload = load_results(results_path)
                metrics_payloads.append(metrics_payload)
                results_payloads.append(results_payload)

                doc_out = setup_dir / doc_dir.name
                doc_out.mkdir(parents=True, exist_ok=True)
                shutil.copy2(metrics_path, doc_out / metrics_name)
                shutil.copy2(results_path, doc_out / results_name)
                shutil.copy2(summary_path, doc_out / summary_name)

            if metrics_payloads and results_payloads:
                row = summarize_run(metrics_payloads, results_payloads, total_runtime, spec)
                row["failed_docs"] = ",".join(failed_docs)
                summary_rows.append(row)

    summary_rows = sorted(summary_rows, key=lambda r: (r["setup"], r["bm25_tokenizer"]))
    write_json(out_dir / "bm25_tokenizer_sensitivity_summary.json", summary_rows)
    pd.DataFrame(summary_rows).to_csv(out_dir / "bm25_tokenizer_sensitivity_summary.csv", index=False)
    (out_dir / "bm25_tokenizer_sensitivity_summary.md").write_text(render_markdown(summary_rows), encoding="utf-8")
    write_json(out_dir / "manifest.json", manifest)
    print(f"Saved BM25 tokenizer sensitivity results to {out_dir}")


if __name__ == "__main__":
    main()
