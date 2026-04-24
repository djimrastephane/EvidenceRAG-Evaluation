from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DOCS = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]
DEFAULT_MODELS = [
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
    "all-mpnet-base-v2",
]
DEFAULT_K_LIST = "1,3,5,10"
DEFAULT_DENSE_WEIGHT = 0.5
DEFAULT_BM25_WEIGHT = 2.0
DEFAULT_RRF_K = 20
DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75
DEFAULT_SUBSECTION_BOOST = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a controlled dense-encoder ablation over the fixed 224/56 thesis retrieval corpus."
    )
    parser.add_argument(
        "--base-data-root",
        default="data_processed",
        help="Root containing the fixed processed thesis document folders.",
    )
    parser.add_argument(
        "--out-root",
        default="results/dense_encoder_ablation",
        help="Root directory for staged artifacts and summaries.",
    )
    parser.add_argument(
        "--docs",
        default=",".join(DEFAULT_DOCS),
        help="Comma-separated document ids to include.",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated dense model names or paths.",
    )
    parser.add_argument(
        "--k-list",
        default=DEFAULT_K_LIST,
        help="Comma-separated k values passed unchanged to the evaluators.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter used to call the existing scripts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing timestamped run directory if it already exists.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Optional fixed run folder name under out-root. Defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--enable-subsection-boost",
        action="store_true",
        help="Enable subsection boost for hybrid runs so the ablation reflects the current promoted pipeline.",
    )
    parser.add_argument(
        "--subsection-boost",
        type=float,
        default=DEFAULT_SUBSECTION_BOOST,
        help="Subsection boost value passed to the hybrid evaluator when enabled.",
    )
    return parser.parse_args()


def utc_now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def run_cmd(cmd: list[str], env: dict[str, str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def resolve_script_path(repo_root: Path, relative_candidates: list[str]) -> Path:
    for candidate in relative_candidates:
        path = repo_root / candidate
        if path.exists():
            return path
    joined = ", ".join(relative_candidates)
    raise FileNotFoundError(f"Could not find any of the expected scripts: {joined}")


def with_model_loading_env(env: dict[str, str], resolved_model: str) -> dict[str, str]:
    out = env.copy()
    if Path(resolved_model).exists():
        out["HF_HUB_OFFLINE"] = "1"
        out["TRANSFORMERS_OFFLINE"] = "1"
    return out


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def slugify_model_name(model_name: str) -> str:
    out = []
    for ch in str(model_name).strip():
        out.append(ch if ch.isalnum() or ch in {"-", "_", "."} else "_")
    return "".join(out).strip("_") or "model"


def resolve_model_identifier(model_name: str, repo_root: Path) -> tuple[str, str]:
    raw = str(model_name).strip()
    if not raw:
        raise ValueError("Model name cannot be empty.")
    candidate_path = Path(raw)
    if candidate_path.exists():
        return str(candidate_path), slugify_model_name(candidate_path.name)
    local_model = repo_root / "models" / raw
    if local_model.exists():
        return str(local_model), slugify_model_name(raw)
    if "/" in raw:
        return raw, slugify_model_name(Path(raw).name)
    return f"sentence-transformers/{raw}", slugify_model_name(raw)


def ensure_fixed_chunking(doc_dir: Path) -> dict[str, Any]:
    metrics_path = doc_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json for controlled corpus: {metrics_path}")
    metrics = read_json(metrics_path)
    params = metrics.get("params", {}) if isinstance(metrics, dict) else {}
    chunk_size = params.get("chunk_size_tokens")
    overlap = params.get("chunk_overlap_tokens")
    if int(chunk_size) != 224 or int(overlap) != 56:
        raise ValueError(
            f"{doc_dir.name} is not the fixed 224/56 corpus "
            f"(found chunk_size_tokens={chunk_size}, chunk_overlap_tokens={overlap})."
        )
    return metrics


def stage_doc(source_doc_dir: Path, target_doc_dir: Path) -> None:
    target_doc_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("chunks.parquet", "eval_set.json", "metrics.json", "table_facts.parquet"):
        src = source_doc_dir / filename
        if src.exists():
            shutil.copy2(src, target_doc_dir / filename)


def pick_margin_scores(per_k: dict[str, Any]) -> list[float]:
    candidate_keys = sorted(
        [int(k) for k, payload in per_k.items() if isinstance(payload, dict) and len(payload.get("retrieved_scores", [])) >= 2]
    )
    if not candidate_keys:
        return []
    best_key = str(candidate_keys[-1])
    scores = per_k.get(best_key, {}).get("retrieved_scores", [])
    return [float(x) for x in scores[:2]]


def collect_query_rows(result_paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in result_paths:
        payload = read_json(path)
        for item in payload.get("results", []):
            per_k = item.get("per_k", {})
            k1 = per_k.get("1", {})
            k3 = per_k.get("3", {})
            k10 = per_k.get("10", {})
            scores = pick_margin_scores(per_k)
            margin = None
            if len(scores) >= 2:
                margin = float(scores[0] - scores[1])
            rows.append(
                {
                    "query_id": item.get("query_id"),
                    "doc_id": item.get("doc_id"),
                    "failure_type": item.get("failure_type"),
                    "hit_at_1": 1 if float(k1.get("page_recall_at_k", 0.0)) > 0 else 0,
                    "hit_at_3": 1 if float(k3.get("page_recall_at_k", 0.0)) > 0 else 0,
                    "mrr_at_10": float(k10.get("page_mrr_at_k", 0.0)),
                    "top1_top2_margin": margin,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No query-level retrieval rows were produced.")
    return df


def summarize_setup(
    *,
    model_label: str,
    setup: str,
    runtime_seconds: float,
    result_paths: list[Path],
) -> dict[str, Any]:
    df = collect_query_rows(result_paths)
    return {
        "model": model_label,
        "setup": setup,
        "Hit@1": round(float(df["hit_at_1"].mean()), 4),
        "Hit@3": round(float(df["hit_at_3"].mean()), 4),
        "MRR@10": round(float(df["mrr_at_10"].mean()), 4),
        "FP2 count": int((df["failure_type"] == "FP2_MISSED_TOP_RANK").sum()),
        "mean top1-top2 similarity margin": round(float(df["top1_top2_margin"].dropna().mean()), 6),
        "runtime_seconds": round(float(runtime_seconds), 3),
        "num_queries": int(len(df)),
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    base_data_root = (repo_root / args.base_data_root).resolve()
    docs = parse_csv_list(args.docs)
    model_inputs = parse_csv_list(args.models)
    if not docs:
        raise ValueError("At least one document id is required.")
    if not model_inputs:
        raise ValueError("At least one model is required.")

    run_name = args.run_name.strip() or f"dense_encoder_ablation_{utc_now_slug()}"
    run_root = (repo_root / args.out_root / run_name).resolve()
    if run_root.exists() and not args.force:
        raise FileExistsError(f"Run directory already exists: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    pythonpath_parts = [
        str(repo_root),
        str(repo_root / "src"),
        str(repo_root / "scripts"),
        str(repo_root / "submission_examiner" / "scripts"),
    ]
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    manifest: dict[str, Any] = {
        "run_name": run_name,
        "run_root": str(run_root),
        "base_data_root": str(base_data_root),
        "docs": docs,
        "models_requested": model_inputs,
        "k_list": args.k_list,
        "constraints": {
            "chunk_size_tokens": 224,
            "chunk_overlap_tokens": 56,
            "segment_aware_chunking": True,
            "whole_doc_markdown_mode": False,
            "bm25_k1": DEFAULT_BM25_K1,
            "bm25_b": DEFAULT_BM25_B,
            "rrf_k": DEFAULT_RRF_K,
            "dense_weight": DEFAULT_DENSE_WEIGHT,
            "bm25_weight": DEFAULT_BM25_WEIGHT,
            "faiss_index_type": "IndexFlatIP",
            "enable_subsection_boost": bool(args.enable_subsection_boost),
            "subsection_boost": float(args.subsection_boost),
        },
        "runs": [],
    }

    build_index_script = resolve_script_path(repo_root, ["scripts/build_index.py"])
    dense_eval_script = resolve_script_path(repo_root, ["scripts/retrieval_eval.py"])
    hybrid_eval_script = resolve_script_path(
        repo_root,
        [
            "scripts/retrieval_eval_hybrid.py",
            "submission_examiner/scripts/retrieval_eval_hybrid.py",
        ],
    )

    # Verify the fixed corpus before creating ablation artifacts.
    for doc_id in docs:
        source_doc_dir = base_data_root / doc_id
        if not source_doc_dir.exists():
            raise FileNotFoundError(f"Missing source document directory: {source_doc_dir}")
        ensure_fixed_chunking(source_doc_dir)
        if not (source_doc_dir / "chunks.parquet").exists():
            raise FileNotFoundError(f"Missing chunks.parquet: {source_doc_dir / 'chunks.parquet'}")
        if not (source_doc_dir / "eval_set.json").exists():
            raise FileNotFoundError(f"Missing eval_set.json: {source_doc_dir / 'eval_set.json'}")

    summary_rows: list[dict[str, Any]] = []

    for model_input in model_inputs:
        resolved_model, model_slug = resolve_model_identifier(model_input, repo_root=repo_root)
        model_env = with_model_loading_env(env, resolved_model)
        model_root = run_root / "artifacts" / model_slug
        source_root = model_root / "source_docs"
        source_root.mkdir(parents=True, exist_ok=True)

        for doc_id in docs:
            stage_doc(base_data_root / doc_id, source_root / doc_id)

        build_started = time.perf_counter()
        run_cmd(
            [
                str(args.python_bin),
                str(build_index_script),
                "--data-dir",
                str(source_root),
                "--model",
                resolved_model,
            ],
            env=model_env,
        )
        build_runtime = time.perf_counter() - build_started

        dense_result_paths: list[Path] = []
        hybrid_result_paths: list[Path] = []

        dense_started = time.perf_counter()
        for doc_id in docs:
            doc_dir = source_root / doc_id
            run_cmd(
                [
                    str(args.python_bin),
                    str(dense_eval_script),
                    "--data-dir",
                    str(doc_dir),
                    "--model",
                    resolved_model,
                    "--k-list",
                    args.k_list,
                ],
                env=model_env,
            )
            dense_result_paths.append(doc_dir / "retrieval_results.json")
        dense_runtime = time.perf_counter() - dense_started

        hybrid_started = time.perf_counter()
        hybrid_env = model_env.copy()
        if args.enable_subsection_boost:
            hybrid_env["ENABLE_SUBSECTION_BOOST"] = "1"
            hybrid_env["SUBSECTION_BOOST"] = str(float(args.subsection_boost))
        else:
            hybrid_env["ENABLE_SUBSECTION_BOOST"] = "0"
            hybrid_env["SUBSECTION_BOOST"] = "0.0"
        for doc_id in docs:
            doc_dir = source_root / doc_id
            run_cmd(
                [
                    str(args.python_bin),
                    str(hybrid_eval_script),
                    "--data-dir",
                    str(doc_dir),
                    "--model",
                    resolved_model,
                    "--k-list",
                    args.k_list,
                    "--rrf-k",
                    str(DEFAULT_RRF_K),
                    "--dense-weight",
                    str(DEFAULT_DENSE_WEIGHT),
                    "--bm25-weight",
                    str(DEFAULT_BM25_WEIGHT),
                    "--bm25-k1",
                    str(DEFAULT_BM25_K1),
                    "--bm25-b",
                    str(DEFAULT_BM25_B),
                ],
                env=hybrid_env,
            )
            hybrid_result_paths.append(doc_dir / "retrieval_results_hybrid.json")
        hybrid_runtime = time.perf_counter() - hybrid_started

        dense_summary = summarize_setup(
            model_label=model_slug,
            setup="dense-only",
            runtime_seconds=dense_runtime,
            result_paths=dense_result_paths,
        )
        hybrid_summary = summarize_setup(
            model_label=model_slug,
            setup="hybrid+boost" if args.enable_subsection_boost else "hybrid",
            runtime_seconds=hybrid_runtime,
            result_paths=hybrid_result_paths,
        )
        summary_rows.extend([dense_summary, hybrid_summary])
        manifest["runs"].append(
            {
                "model_input": model_input,
                "model_resolved": resolved_model,
                "model_slug": model_slug,
                "artifact_root": str(source_root),
                "build_runtime_seconds": round(float(build_runtime), 3),
                "dense_runtime_seconds": round(float(dense_runtime), 3),
                "hybrid_runtime_seconds": round(float(hybrid_runtime), 3),
                "enable_subsection_boost": bool(args.enable_subsection_boost),
                "subsection_boost": float(args.subsection_boost) if args.enable_subsection_boost else 0.0,
                "dense_result_paths": [str(p) for p in dense_result_paths],
                "hybrid_result_paths": [str(p) for p in hybrid_result_paths],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    ordered_cols = [
        "model",
        "setup",
        "Hit@1",
        "Hit@3",
        "MRR@10",
        "FP2 count",
        "mean top1-top2 similarity margin",
        "runtime_seconds",
        "num_queries",
    ]
    summary_df = summary_df[ordered_cols].sort_values(["setup", "model"]).reset_index(drop=True)

    summary_csv = run_root / "dense_encoder_ablation_summary.csv"
    summary_json = run_root / "dense_encoder_ablation_summary.json"
    summary_md = run_root / "dense_encoder_ablation_summary.md"
    manifest_json = run_root / "manifest.json"

    summary_df.to_csv(summary_csv, index=False)
    write_json(summary_json, json.loads(summary_df.to_json(orient="records")))
    summary_md.write_text(summary_df.to_markdown(index=False) + "\n", encoding="utf-8")
    write_json(manifest_json, manifest)

    print("Saved:", summary_csv)
    print("Saved:", summary_json)
    print("Saved:", summary_md)
    print("Saved:", manifest_json)


if __name__ == "__main__":
    main()
