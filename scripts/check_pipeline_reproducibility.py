"""
Run repeated pipeline executions and check whether they produce byte-identical
canonical outputs under the same inputs and configuration.

This script acts as a reproducibility harness for the retrieval pipeline. It:
1. loads evaluation-ready document directories,
2. runs the SearchService over every eval-set query,
3. canonicalizes per-query outputs by removing unstable metadata and rounding
   floating-point scores,
4. hashes the canonical payload for each run, and
5. reports whether repeated runs produce identical hashes.

The final JSON report records the number of runs, the per-run hashes, the first
run that differs from baseline if any, and notes about reproducibility limits.
If `--include-generated-answer` is enabled, strict reproducibility is less
likely because local LLM generation may be stochastic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PARENT_PATH = REPO_ROOT.parent
SRC_PATH = REPO_ROOT / "src"
SCRIPTS_PATH = REPO_ROOT / "scripts"
if str(PARENT_PATH) not in sys.path:
    sys.path.insert(0, str(PARENT_PATH))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if SCRIPTS_PATH.exists() and str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))

from rag_pdf.services.search_helpers import doc_artifact_signature, read_eval_items
from rag_pdf.services.search_service import SearchService
from corpus_guard import list_eval_ready_doc_dirs, print_skipped_eval_ready_docs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether the pipeline returns byte-identical canonical outputs "
            "across repeated runs with the same inputs."
        )
    )
    parser.add_argument("--data-root", default="data_processed")
    parser.add_argument("--doc-pattern", default="Grampian-20*-20*")
    parser.add_argument(
        "--allow-incomplete-corpora",
        action="store_true",
        help="Include matching doc folders even if they are missing canonical evaluation artifacts.",
    )
    parser.add_argument("--model-path", default="models/all-MiniLM-L6-v2")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--runs", type=int, default=10, help="Number of repeated runs to execute.")
    parser.add_argument(
        "--include-generated-answer",
        action="store_true",
        help="Include local LLM generation in the checked output. This is expected to be less reproducible.",
    )
    parser.add_argument(
        "--round-score-digits",
        type=int,
        default=10,
        help="Decimal digits to retain when canonicalizing floating-point scores.",
    )
    parser.add_argument(
        "--out-json",
        default="results/reproducibility/pipeline_reproducibility_report.json",
        help="Where to write the reproducibility report.",
    )
    parser.add_argument(
        "--worker-out",
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _stable_float(value: Any, digits: int) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), int(digits))
    except Exception:
        return None


def _canonicalize_result(
    doc_id: str,
    query_id: str,
    question: str,
    out: dict[str, Any],
    round_score_digits: int,
) -> dict[str, Any]:
    results = []
    for row in list(out.get("results") or []):
        results.append(
            {
                "rank": int(row.get("rank") or 0),
                "chunk_id": str(row.get("chunk_id") or ""),
                "pages": [int(x) for x in list(row.get("pages") or []) if str(x).strip().isdigit()],
                "score": _stable_float(row.get("score"), round_score_digits),
                "fusion_score": _stable_float(row.get("fusion_score"), round_score_digits),
                "rrf_score": _stable_float(row.get("rrf_score"), round_score_digits),
                "dense_rank": int(row.get("dense_rank") or 0),
                "bm25_rank": int(row.get("bm25_rank") or 0),
                "dense_raw_score": _stable_float(row.get("dense_raw_score"), round_score_digits),
                "bm25_raw_score": _stable_float(row.get("bm25_raw_score"), round_score_digits),
                "section_title": str(row.get("section_title") or ""),
                "subsection_title": str(row.get("subsection_title") or ""),
                "is_table": bool(row.get("is_table", False)),
            }
        )
    return {
        "doc_id": str(doc_id),
        "query_id": str(query_id),
        "question": str(question),
        "k": int(out.get("k") or 0),
        "retrieval_mode": str(out.get("retrieval_mode") or ""),
        "retrieval_scope": str(out.get("retrieval_scope") or ""),
        "lexical_scope": str(out.get("lexical_scope") or ""),
        "filters_applied": out.get("filters_applied") or {},
        "expected_pages": [int(x) for x in list(out.get("expected_pages") or []) if str(x).strip().isdigit()],
        "hit_at_k": bool(out.get("hit_at_k", False)),
        "predicted_answer": out.get("predicted_answer"),
        "predicted_answer_raw": out.get("predicted_answer_raw"),
        "answer_source_chunk_id": str(out.get("answer_source_chunk_id") or ""),
        "generation_status": str(out.get("generation_status") or ""),
        "generation_confidence": _stable_float(out.get("generation_confidence"), round_score_digits),
        "generated_answer": out.get("generated_answer"),
        "generated_answer_raw": out.get("generated_answer_raw"),
        "results": results,
    }


def _canonical_payload(
    data_root: Path,
    doc_pattern: str,
    allow_incomplete_corpora: bool,
    model_path: Path,
    k: int,
    include_generated_answer: bool,
    round_score_digits: int,
) -> dict[str, Any]:
    if allow_incomplete_corpora:
        docs = sorted([p for p in data_root.glob(doc_pattern) if p.is_dir() and read_eval_items(p / "eval_set.json")])
    else:
        docs, skipped = list_eval_ready_doc_dirs(data_root, doc_pattern)
        print_skipped_eval_ready_docs(skipped)
        docs = [p for p in docs if read_eval_items(p / "eval_set.json")]
    if not docs:
        raise FileNotFoundError(f"No docs with eval sets found under {data_root} matching {doc_pattern}")

    service = SearchService(repo_root=REPO_ROOT, model_path=model_path)
    per_query: list[dict[str, Any]] = []
    artifact_signatures: dict[str, Any] = {}

    for doc_dir in docs:
        doc_id = doc_dir.name
        artifact_signatures[doc_id] = list(doc_artifact_signature(doc_dir))
        eval_items = read_eval_items(doc_dir / "eval_set.json")
        for item in eval_items:
            query_id = str(item.get("query_id") or "").strip()
            question = str(item.get("question") or "").strip()
            out = service.search(
                data_dir=doc_dir,
                question=question,
                k=int(k),
                query_id=query_id or None,
                include_generated_answer=bool(include_generated_answer),
            )
            per_query.append(
                _canonicalize_result(
                    doc_id=doc_id,
                    query_id=query_id,
                    question=question,
                    out=out,
                    round_score_digits=round_score_digits,
                )
            )

    per_query.sort(key=lambda row: (str(row["doc_id"]), str(row["query_id"]), str(row["question"])))
    return {
        "config": {
            "data_root": str(data_root.resolve()),
            "doc_pattern": str(doc_pattern),
            "model_path": str(model_path.resolve()),
            "k": int(k),
            "include_generated_answer": bool(include_generated_answer),
            "round_score_digits": int(round_score_digits),
        },
        "artifact_signatures": artifact_signatures,
        "query_count": int(len(per_query)),
        "per_query": per_query,
    }


def _payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_worker(args: argparse.Namespace) -> dict[str, Any]:
    data_root = (REPO_ROOT / args.data_root).resolve()
    model_path = (REPO_ROOT / args.model_path).resolve()
    payload = _canonical_payload(
        data_root=data_root,
        doc_pattern=str(args.doc_pattern),
        allow_incomplete_corpora=bool(args.allow_incomplete_corpora),
        model_path=model_path,
        k=int(args.k),
        include_generated_answer=bool(args.include_generated_answer),
        round_score_digits=int(args.round_score_digits),
    )
    payload["payload_hash"] = _payload_hash(payload)
    return payload


def _run_parent(args: argparse.Namespace) -> dict[str, Any]:
    out_path = (REPO_ROOT / args.out_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = dict(**subprocess.os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("FAISS_NO_AVX2", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    run_hashes: list[str] = []
    run_files: list[str] = []
    mismatch_index: int | None = None
    baseline_path: str | None = None
    baseline_hash: str | None = None

    with tempfile.TemporaryDirectory(prefix="pipeline_repro_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        for run_idx in range(1, int(args.runs) + 1):
            worker_out = tmp_dir / f"run_{run_idx:03d}.json"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--data-root",
                str(args.data_root),
                "--doc-pattern",
                str(args.doc_pattern),
                *(
                    ["--allow-incomplete-corpora"]
                    if args.allow_incomplete_corpora
                    else []
                ),
                "--model-path",
                str(args.model_path),
                "--k",
                str(int(args.k)),
                "--runs",
                "1",
                "--round-score-digits",
                str(int(args.round_score_digits)),
                "--worker-out",
                str(worker_out),
            ]
            if args.include_generated_answer:
                cmd.append("--include-generated-answer")
            subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)

            payload = json.loads(worker_out.read_text(encoding="utf-8"))
            run_hash = str(payload.get("payload_hash") or "")
            run_hashes.append(run_hash)
            run_files.append(str(worker_out))
            if baseline_hash is None:
                baseline_hash = run_hash
                baseline_path = str(worker_out)
            elif run_hash != baseline_hash and mismatch_index is None:
                mismatch_index = run_idx

        all_equal = bool(run_hashes) and len(set(run_hashes)) == 1
        report = {
            "status": "pass" if all_equal else "fail",
            "claim_scope": (
                "retrieval_plus_generation"
                if args.include_generated_answer
                else "retrieval_plus_deterministic_extraction"
            ),
            "runs": int(args.runs),
            "all_hashes_equal": bool(all_equal),
            "baseline_hash": baseline_hash,
            "unique_hash_count": int(len(set(run_hashes))),
            "first_mismatch_run": mismatch_index,
            "baseline_run_file": baseline_path,
            "run_hashes": run_hashes,
            "run_files": run_files,
            "notes": [
                "Hashes are computed over canonicalized per-query outputs only.",
                "The canonical payload excludes timestamps, absolute output paths, and runtime latency metadata.",
                "If include_generated_answer=true, strict reproducibility is less likely because the local LLM path may be stochastic.",
            ],
        }
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        return report


def main() -> None:
    args = parse_args()
    if args.worker_out:
        payload = _run_worker(args)
        worker_path = Path(args.worker_out)
        worker_path.parent.mkdir(parents=True, exist_ok=True)
        worker_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        return

    report = _run_parent(args)
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
