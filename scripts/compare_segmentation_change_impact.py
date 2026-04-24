from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd


DEFAULT_DOCS = [
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]

PDF_BASE = Path("Data/Annual Accounts NHS Grampian/Preliminary_Test")
DATA_ROOT = Path("data_processed")
MODEL_PATH = "models/all-MiniLM-L6-v2"
RESULTS_ROOT = Path("results/segmentation_change_impact")
PYTHON_EXE = Path(
    os.getenv("RAG_PIPELINE_PYTHON", "/opt/anaconda3/envs/rag-pipeline/bin/python")
)


def run(cmd: list[str], *, extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "src")
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    if extra_env:
        env.update(extra_env)
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def read_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown_summary(path: Path, rows: list[dict[str, object]]) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    avg = {col: float(df[col].mean()) for col in df.columns if col != "doc_id"}

    lines = [
        "# Segmentation Change Impact Summary",
        "",
        "This table compares retrieval performance before and after the conservative segment-boundary change that removed the broad inline decimal split rule.",
        "",
        "| Report | Page hit@1 before | Page hit@1 after | Delta | Page hit@3 before | Page hit@3 after | Delta | Page MRR@10 before | Page MRR@10 after | Delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            f"| {row['doc_id']} | "
            f"{row['page_hit@1_before']:.3f} | {row['page_hit@1_after']:.3f} | {row['page_hit@1_delta']:+.3f} | "
            f"{row['page_hit@3_before']:.3f} | {row['page_hit@3_after']:.3f} | {row['page_hit@3_delta']:+.3f} | "
            f"{row['page_mrr@10_before']:.3f} | {row['page_mrr@10_after']:.3f} | {row['page_mrr@10_delta']:+.3f} |"
        )

    lines.extend(
        [
            f"| **Average** | "
            f"{avg['page_hit@1_before']:.3f} | {avg['page_hit@1_after']:.3f} | {avg['page_hit@1_delta']:+.3f} | "
            f"{avg['page_hit@3_before']:.3f} | {avg['page_hit@3_after']:.3f} | {avg['page_hit@3_delta']:+.3f} | "
            f"{avg['page_mrr@10_before']:.3f} | {avg['page_mrr@10_after']:.3f} | {avg['page_mrr@10_delta']:+.3f} |",
            "",
            "Interpretation:",
            f"- Average `page hit@1` changed from `{avg['page_hit@1_before']:.3f}` to `{avg['page_hit@1_after']:.3f}` (`{avg['page_hit@1_delta']:+.3f}`).",
            f"- Average `page hit@3` changed from `{avg['page_hit@3_before']:.3f}` to `{avg['page_hit@3_after']:.3f}` (`{avg['page_hit@3_delta']:+.3f}`).",
            f"- Average `page MRR@10` changed from `{avg['page_mrr@10_before']:.3f}` to `{avg['page_mrr@10_after']:.3f}` (`{avg['page_mrr@10_delta']:+.3f}`).",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def collect_metric_row(doc: str, phase: str, metrics: dict) -> dict[str, object]:
    out: dict[str, object] = {"doc_id": doc, "phase": phase}
    for k in ("1", "3", "5", "10"):
        vals = metrics.get("metrics_by_k", {}).get(k, {})
        out[f"page_hit@{k}"] = vals.get("page_hit_rate")
        out[f"page_mrr@{k}"] = vals.get("mean_page_mrr")
        out[f"chunk_hit@{k}"] = vals.get("chunk_hit_rate")
        out[f"chunk_mrr@{k}"] = vals.get("mean_chunk_mrr")
    return out


def copy_eval_outputs(doc_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("retrieval_metrics_hybrid.json", "retrieval_results_hybrid.json", "retrieval_summary_hybrid.csv", "metrics.json"):
        src = doc_dir / name
        if src.exists():
            shutil.copy2(src, target_dir / name)


def rebuild_index_for_doc(doc_dir: Path, model_path: str) -> None:
    cmd = f"""
from pathlib import Path
from sentence_transformers import SentenceTransformer
from scripts.build_index import build_index_for_doc

model = SentenceTransformer({model_path!r}, device='cpu')
build_index_for_doc(Path({str(doc_dir)!r}), model)
"""
    run([str(PYTHON_EXE), "-c", cmd])


def metrics_delta(before: dict, after: dict) -> dict[str, object]:
    row: dict[str, object] = {"doc_id": before["doc_id"]}
    for k in ("1", "3", "5", "10"):
        b = before.get("metrics_by_k", {}).get(k, {})
        a = after.get("metrics_by_k", {}).get(k, {})
        row[f"page_hit@{k}_before"] = b.get("page_hit_rate")
        row[f"page_hit@{k}_after"] = a.get("page_hit_rate")
        row[f"page_hit@{k}_delta"] = _safe_delta(a.get("page_hit_rate"), b.get("page_hit_rate"))
        row[f"page_mrr@{k}_before"] = b.get("mean_page_mrr")
        row[f"page_mrr@{k}_after"] = a.get("mean_page_mrr")
        row[f"page_mrr@{k}_delta"] = _safe_delta(a.get("mean_page_mrr"), b.get("mean_page_mrr"))
        row[f"chunk_hit@{k}_before"] = b.get("chunk_hit_rate")
        row[f"chunk_hit@{k}_after"] = a.get("chunk_hit_rate")
        row[f"chunk_hit@{k}_delta"] = _safe_delta(a.get("chunk_hit_rate"), b.get("chunk_hit_rate"))
        row[f"chunk_mrr@{k}_before"] = b.get("mean_chunk_mrr")
        row[f"chunk_mrr@{k}_after"] = a.get("mean_chunk_mrr")
        row[f"chunk_mrr@{k}_delta"] = _safe_delta(a.get("mean_chunk_mrr"), b.get("mean_chunk_mrr"))
    return row


def _safe_delta(a, b):
    if a is None or b is None:
        return None
    return float(a) - float(b)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare downstream retrieval impact of the conservative segmentation change.")
    parser.add_argument("--docs", nargs="*", default=DEFAULT_DOCS)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []

    for doc in args.docs:
        doc_dir = DATA_ROOT / doc
        pdf_path = PDF_BASE / f"{doc}.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing PDF: {pdf_path}")

        baseline_dir = args.results_root / doc / "before"
        after_dir = args.results_root / doc / "after"

        # Baseline eval on current stored artifacts
        run(
            [
                str(PYTHON_EXE),
                "scripts/retrieval_eval_hybrid.py",
                "--data-dir",
                str(doc_dir),
                "--model",
                str(args.model),
                "--device",
                "cpu",
            ]
        )
        before_metrics = read_metrics(doc_dir / "retrieval_metrics_hybrid.json")
        copy_eval_outputs(doc_dir, baseline_dir)

        # Reprocess + rebuild index + re-evaluate with new segmentation
        run(
            [
                str(PYTHON_EXE),
                "scripts/preprocess_hybrid.py",
                "--pdf-path",
                str(pdf_path),
                "--out-root",
                str(DATA_ROOT),
                "--cross-page-sentence-overlap",
            ]
        )
        rebuild_index_for_doc(doc_dir, str(args.model))
        run(
            [
                str(PYTHON_EXE),
                "scripts/retrieval_eval_hybrid.py",
                "--data-dir",
                str(doc_dir),
                "--model",
                str(args.model),
                "--device",
                "cpu",
            ]
        )
        after_metrics = read_metrics(doc_dir / "retrieval_metrics_hybrid.json")
        copy_eval_outputs(doc_dir, after_dir)

        rows.append(collect_metric_row(doc, "before", before_metrics))
        rows.append(collect_metric_row(doc, "after", after_metrics))
        summary.append(metrics_delta({"doc_id": doc, **before_metrics}, {"doc_id": doc, **after_metrics}))

    args.results_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.results_root / "segmentation_change_metrics_long.csv", index=False)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(args.results_root / "segmentation_change_metrics_delta.csv", index=False)
    write_markdown_summary(args.results_root / "segmentation_change_metrics_summary.md", summary)
    save_json(args.results_root / "run_manifest.json", {"docs": args.docs, "model": args.model})
    print(f"Wrote: {args.results_root / 'segmentation_change_metrics_long.csv'}")
    print(f"Wrote: {args.results_root / 'segmentation_change_metrics_delta.csv'}")
    print(f"Wrote: {args.results_root / 'segmentation_change_metrics_summary.md'}")


if __name__ == "__main__":
    main()
