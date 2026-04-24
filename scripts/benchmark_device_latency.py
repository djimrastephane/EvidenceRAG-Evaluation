from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark CPU vs MPS latency for local retrieval model inference."
    )
    parser.add_argument(
        "--devices",
        default="cpu,mps",
        help="Comma-separated device list, e.g. cpu,mps.",
    )
    parser.add_argument(
        "--mode",
        choices=("search", "embedding"),
        default="search",
        help="Benchmark full local search path or embedding inference only.",
    )
    parser.add_argument(
        "--data-dir",
        default="data_processed/Grampian-2024-2025",
        help="Processed document directory for search mode.",
    )
    parser.add_argument(
        "--model",
        default="models/all-MiniLM-L6-v2",
        help="SentenceTransformer model path/name.",
    )
    parser.add_argument(
        "--eval-set",
        default="",
        help="Optional eval_set.json path. Defaults to <data-dir>/eval_set.json.",
    )
    parser.add_argument(
        "--question",
        default="What ceiling did the Scottish Government place on NHS Grampian's core operational spending for 2024/25?",
        help="Fallback question when eval_set is unavailable.",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=50,
        help="Measured query count per device.",
    )
    parser.add_argument(
        "--warmup-queries",
        type=int,
        default=5,
        help="Warmup query count per device.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k for local search mode.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if q <= 0:
        return ordered[0]
    if q >= 100:
        return ordered[-1]
    pos = (len(ordered) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _load_questions(args: argparse.Namespace) -> list[str]:
    eval_path = Path(args.eval_set) if args.eval_set else Path(args.data_dir) / "eval_set.json"
    if eval_path.exists():
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict) and isinstance(payload.get("queries"), list):
            items = payload["queries"]
        else:
            items = []
        questions = [str(item.get("question", "")).strip() for item in items if isinstance(item, dict)]
        questions = [q for q in questions if q]
        if questions:
            return questions
    return [str(args.question).strip()]


def _cycle_questions(questions: list[str], count: int) -> list[str]:
    if not questions:
        return []
    return [questions[i % len(questions)] for i in range(count)]


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_src_on_path(repo_root: Path) -> None:
    src_path = repo_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def _benchmark_embedding(device: str, args: argparse.Namespace, questions: list[str]) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(args.model), device=device)
    warmup = _cycle_questions(questions, args.warmup_queries)
    measured = _cycle_questions(questions, args.num_queries)

    for q in warmup:
        model.encode([q], convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)

    latencies_ms: list[float] = []
    for q in measured:
        start = time.perf_counter()
        model.encode([q], convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    return _summarize(device=device, mode="embedding", latencies_ms=latencies_ms, query_count=len(measured))


def _benchmark_search(device: str, args: argparse.Namespace, questions: list[str]) -> dict[str, Any]:
    repo_root = _resolve_repo_root()
    _ensure_src_on_path(repo_root)
    from rag_pdf.services.search_service import SearchService

    os.environ["ST_MODEL_DEVICE"] = device
    os.environ["CROSS_ENCODER_DEVICE"] = device
    service = SearchService(repo_root=repo_root, model_path=Path(args.model))
    data_dir = Path(args.data_dir)
    warmup = _cycle_questions(questions, args.warmup_queries)
    measured = _cycle_questions(questions, args.num_queries)

    for q in warmup:
        service.search(data_dir=data_dir, question=q, k=int(args.k), query_id=None, include_generated_answer=False)

    latencies_ms: list[float] = []
    for q in measured:
        start = time.perf_counter()
        service.search(data_dir=data_dir, question=q, k=int(args.k), query_id=None, include_generated_answer=False)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    return _summarize(device=device, mode="search", latencies_ms=latencies_ms, query_count=len(measured))


def _summarize(device: str, mode: str, latencies_ms: list[float], query_count: int) -> dict[str, Any]:
    total_ms = sum(latencies_ms)
    return {
        "device": device,
        "mode": mode,
        "query_count": int(query_count),
        "latency_ms": {
            "mean": round(statistics.mean(latencies_ms), 3),
            "median": round(statistics.median(latencies_ms), 3),
            "min": round(min(latencies_ms), 3),
            "max": round(max(latencies_ms), 3),
            "p95": round(_percentile(latencies_ms, 95), 3),
            "stdev": round(statistics.pstdev(latencies_ms), 3),
        },
        "throughput_qps": round((1000.0 * query_count / total_ms), 3) if total_ms > 0 else 0.0,
    }


def main() -> None:
    args = parse_args()
    devices = [d.strip().lower() for d in str(args.devices).split(",") if d.strip()]
    if not devices:
        raise ValueError("At least one device must be provided.")

    questions = _load_questions(args)
    results: list[dict[str, Any]] = []

    for device in devices:
        if args.mode == "embedding":
            results.append(_benchmark_embedding(device=device, args=args, questions=questions))
        else:
            results.append(_benchmark_search(device=device, args=args, questions=questions))

    payload = {
        "mode": args.mode,
        "model": str(args.model),
        "data_dir": str(args.data_dir),
        "devices": devices,
        "num_queries": int(args.num_queries),
        "warmup_queries": int(args.warmup_queries),
        "k": int(args.k),
        "results": results,
    }

    text = json.dumps(payload, indent=2)
    print(text)
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
