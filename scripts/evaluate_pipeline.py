"""Convenience wrapper that runs hybrid retrieval evaluation for a single document corpus.

Delegates to retrieval_eval_hybrid.py with the canonical thesis_rag hyperparameters
(rrf_k=20, dense_weight=0.5, bm25_weight=2.0) and then to report_retrieval_metrics.py to
produce a human-readable summary. Accepts an optional cross-encoder reranking stage.
Intended as a single-command entry point when evaluating one processed document folder.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the production pipeline using the canonical hybrid Dense+BM25 RRF setup."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing faiss.index, chunk_meta.parquet, chunks.parquet, and eval_set.json.",
    )
    parser.add_argument(
        "--scope",
        choices=("doc", "global"),
        default="doc",
        help="Evaluate within the selected document only or against a global multi-document corpus.",
    )
    parser.add_argument(
        "--corpus-root",
        default="",
        help="Optional multi-document corpus root for --scope=global.",
    )
    parser.add_argument(
        "--eval-set-path",
        default="",
        help="Optional override path for eval_set.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory for evaluation artifacts.",
    )
    parser.add_argument(
        "--model",
        default="models/all-MiniLM-L6-v2",
        help="Sentence-transformers model name or local path.",
    )
    parser.add_argument(
        "--k-list",
        default="1,3,5,10",
        help="Comma-separated list of k values.",
    )
    parser.add_argument("--rrf-k", type=int, default=20, help="RRF constant k.")
    parser.add_argument("--dense-weight", type=float, default=0.5, help="Dense rank contribution weight.")
    parser.add_argument("--bm25-weight", type=float, default=2.0, help="BM25 rank contribution weight.")
    parser.add_argument("--bm25-k1", type=float, default=1.5, help="BM25 k1 parameter.")
    parser.add_argument("--bm25-b", type=float, default=0.75, help="BM25 b parameter.")
    parser.add_argument(
        "--enable-cross-encoder-rerank",
        action="store_true",
        help="Enable local cross-encoder reranking on top fused candidates.",
    )
    parser.add_argument(
        "--cross-encoder-model",
        default="models/bge-reranker-v2-m3",
        help="Cross-encoder model path when rerank is enabled.",
    )
    parser.add_argument("--cross-encoder-topn", type=int, default=50, help="Top-N fused candidates to rerank.")
    parser.add_argument("--cross-encoder-weight", type=float, default=0.2, help="Cross-encoder interpolation weight.")
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip report_retrieval_metrics.py after hybrid evaluation.",
    )
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    """Run a subprocess command from the repo root with standard environment guards; raise SystemExit on failure."""
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("FAISS_NO_AVX2", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    completed = subprocess.run(cmd, cwd=repo_root, env=env)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    """Run retrieval_eval_hybrid.py then report_retrieval_metrics.py for the given document directory."""
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data directory: {data_dir}")
    doc_id = data_dir.name

    hybrid_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "retrieval_eval_hybrid.py"),
        "--data-dir",
        str(data_dir),
        "--scope",
        str(args.scope),
        "--model",
        str(args.model),
        "--k-list",
        str(args.k_list),
        "--rrf-k",
        str(int(args.rrf_k)),
        "--dense-weight",
        str(float(args.dense_weight)),
        "--bm25-weight",
        str(float(args.bm25_weight)),
        "--bm25-k1",
        str(float(args.bm25_k1)),
        "--bm25-b",
        str(float(args.bm25_b)),
    ]
    if str(args.corpus_root).strip():
        hybrid_cmd.extend(["--corpus-root", str(args.corpus_root)])
    if str(args.eval_set_path).strip():
        hybrid_cmd.extend(["--eval-set-path", str(args.eval_set_path)])
    if str(args.output_dir).strip():
        hybrid_cmd.extend(["--output-dir", str(args.output_dir)])
    if args.enable_cross_encoder_rerank:
        hybrid_cmd.extend(
            [
                "--enable-cross-encoder-rerank",
                "--cross-encoder-model",
                str(args.cross_encoder_model),
                "--cross-encoder-topn",
                str(int(args.cross_encoder_topn)),
                "--cross-encoder-weight",
                str(float(args.cross_encoder_weight)),
            ]
        )

    run_cmd(hybrid_cmd)

    if args.skip_report:
        return

    report_root = Path(args.output_dir).resolve().parent if str(args.output_dir).strip() else data_dir.parent
    report_doc = Path(args.output_dir).resolve().name if str(args.output_dir).strip() else doc_id
    report_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "report_retrieval_metrics.py"),
        "--data-root",
        str(report_root),
        "--docs",
        str(report_doc),
    ]
    run_cmd(report_cmd)


if __name__ == "__main__":
    main()
