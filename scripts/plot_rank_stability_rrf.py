from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

src_path = repo_root / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

ui_path = repo_root / "app" / "ui"
if ui_path.exists() and str(ui_path) not in sys.path:
    sys.path.insert(0, str(ui_path))

import _matplotlib_env  # noqa: F401
import matplotlib.pyplot as plt
from rag_pdf.retrieval.hybrid_utils import BM25Index, l2_normalize, tokenize

try:
    import faiss
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(f"faiss is required to run this script: {exc}") from exc


@dataclass
class QueryArtifacts:
    query_id: str
    question: str
    rows: list[dict[str, object]]
    top_confidence_label: str
    top_confidence_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot stacked RRF rank-stability charts showing Dense vs BM25 contributions."
    )
    parser.add_argument(
        "--data-dir",
        default="data_variants/tiktoken_5docs/Grampian-2020-2021",
        help="Document artifact directory containing faiss.index, chunk_meta.parquet, chunks.parquet, eval_set.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/rank_stability/Grampian-2020-2021",
        help="Directory where charts and summary CSV will be written.",
    )
    parser.add_argument(
        "--model",
        default="models/all-MiniLM-L6-v2",
        help="SentenceTransformer model path or name.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Embedding device passed to SentenceTransformer.",
    )
    parser.add_argument(
        "--query-id",
        default="",
        help="Optional single query_id filter. If omitted, plot all queries in the eval set.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of fused chunks to show per chart.",
    )
    parser.add_argument(
        "--max-k-search",
        type=int,
        default=100,
        help="Depth used to reconstruct dense and BM25 rankings before RRF.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=20,
        help="RRF constant in 1 / (rrf_k + rank).",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=0.5,
        help="Dense branch weight inside RRF.",
    )
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=2.0,
        help="BM25 branch weight inside RRF.",
    )
    parser.add_argument(
        "--bm25-k1",
        type=float,
        default=1.5,
        help="BM25 k1 parameter.",
    )
    parser.add_argument(
        "--bm25-b",
        type=float,
        default=0.75,
        help="BM25 b parameter.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output PNG resolution.",
    )
    parser.add_argument(
        "--confidence-floor",
        type=float,
        default=0.10,
        help="Horizontal dashed threshold line for visually separating stronger vs marginal retrieval scores.",
    )
    return parser.parse_args()


def load_eval_queries(eval_set_path: Path) -> list[dict[str, object]]:
    eval_obj = json.loads(eval_set_path.read_text(encoding="utf-8"))
    if isinstance(eval_obj, list):
        return eval_obj
    if isinstance(eval_obj, dict) and isinstance(eval_obj.get("queries"), list):
        return list(eval_obj["queries"])
    raise ValueError(f"Unsupported eval set format in {eval_set_path}")


def load_stored_hybrid_results(results_path: Path) -> dict[str, dict[str, object]]:
    if not results_path.exists():
        return {}
    obj = json.loads(results_path.read_text(encoding="utf-8"))
    results = obj.get("results", []) if isinstance(obj, dict) else []
    out: dict[str, dict[str, object]] = {}
    for item in results:
        query_id = str(item.get("query_id", "")).strip()
        if query_id:
            out[query_id] = item
    return out


def safe_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return slug.strip("_") or "query"


def display_chunk_label(chunk_id: str) -> str:
    if ":" in chunk_id:
        return chunk_id.split(":", 1)[1]
    return chunk_id


def build_text_lookup(chunks: pd.DataFrame) -> dict[str, str]:
    text_by_id: dict[str, str] = {}
    for _, row in chunks.iterrows():
        chunk_id = str(row.get("chunk_id_global") or row.get("chunk_id") or "")
        if chunk_id and chunk_id not in text_by_id:
            text_by_id[chunk_id] = str(row.get("chunk_text") or "")
    return text_by_id


def get_chunk_id(row: pd.Series) -> str:
    return str(row.get("chunk_id_global") or row.get("chunk_id") or "")


def confidence_label(dense_contrib: float, bm25_contrib: float, total_rrf: float) -> tuple[str, float]:
    if total_rrf <= 0.0:
        return "low", 0.0
    balance = min(dense_contrib, bm25_contrib) / total_rrf
    if balance >= 0.30:
        return "high", balance
    if balance >= 0.18:
        return "medium", balance
    return "low", balance


def to_int_set(value: object) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, list):
        out: set[int] = set()
        for item in value:
            try:
                out.add(int(item))
            except Exception:
                continue
        return out
    return set()


def compute_rrf_breakdown(
    *,
    meta: pd.DataFrame,
    index: faiss.Index,
    bm25: BM25Index,
    question: str,
    question_embedding: np.ndarray,
    max_k_search: int,
    rrf_k: int,
    dense_weight: float,
    bm25_weight: float,
    top_k: int,
    expected_doc_id: str,
    expected_pages: set[int],
    stored_top_chunk_ids: list[str] | None = None,
    stored_top_flags: list[int] | None = None,
) -> list[dict[str, object]]:
    max_k_search = min(max_k_search, len(meta))
    dense_scores, dense_idxs = index.search(question_embedding[None, :], max_k_search)
    dense_ranked = [int(idx) for idx in dense_idxs[0].tolist()]
    bm25_scores = bm25.score_query(tokenize(question))
    bm25_ranked = [idx for idx, _ in sorted(enumerate(bm25_scores), key=lambda item: item[1], reverse=True)[:max_k_search]]

    dense_rank_map = {idx: rank for rank, idx in enumerate(dense_ranked, start=1)}
    bm25_rank_map = {idx: rank for rank, idx in enumerate(bm25_ranked, start=1)}

    candidates = sorted(set(dense_ranked).union(bm25_ranked))
    rows: list[dict[str, object]] = []
    for idx in candidates:
        dense_rank = dense_rank_map.get(idx)
        bm25_rank = bm25_rank_map.get(idx)
        dense_contrib = float(dense_weight / (rrf_k + dense_rank)) if dense_rank is not None else 0.0
        bm25_contrib = float(bm25_weight / (rrf_k + bm25_rank)) if bm25_rank is not None else 0.0
        total_rrf = dense_contrib + bm25_contrib
        row = meta.iloc[idx]
        chunk_id = get_chunk_id(row)
        row_doc_id = str(row.get("doc_id") or "")
        page_start = int(row.get("page_start")) if pd.notna(row.get("page_start")) else None
        page_end = int(row.get("page_end")) if pd.notna(row.get("page_end")) else page_start
        row_pages = set(range(page_start, page_end + 1)) if page_start is not None and page_end is not None else set()
        is_hit = bool(row_pages.intersection(expected_pages)) and (not expected_doc_id or row_doc_id == expected_doc_id)
        label, balance = confidence_label(dense_contrib, bm25_contrib, total_rrf)
        rows.append(
            {
                "chunk_index": int(idx),
                "chunk_id": chunk_id,
                "page": int(row.get("page_start")) if pd.notna(row.get("page_start")) else None,
                "dense_rank": dense_rank,
                "bm25_rank": bm25_rank,
                "dense_contrib": dense_contrib,
                "bm25_contrib": bm25_contrib,
                "rrf_score": total_rrf,
                "dense_raw_score": float(dense_scores[0][dense_rank - 1]) if dense_rank is not None else np.nan,
                "bm25_raw_score": float(bm25_scores[idx]) if idx < len(bm25_scores) else np.nan,
                "confidence_label": label,
                "confidence_balance": balance,
                "is_hit": is_hit,
            }
        )

    row_by_chunk_id = {str(row["chunk_id"]): row for row in rows}
    if stored_top_chunk_ids:
        ordered_rows: list[dict[str, object]] = []
        for rank_idx, chunk_id in enumerate(stored_top_chunk_ids[:top_k]):
            row = row_by_chunk_id.get(str(chunk_id))
            if row is None:
                continue
            row = dict(row)
            if stored_top_flags and rank_idx < len(stored_top_flags):
                row["is_hit"] = bool(int(stored_top_flags[rank_idx]))
            ordered_rows.append(row)
        rows = ordered_rows
    else:
        rows.sort(key=lambda item: (-float(item["rrf_score"]), str(item["chunk_id"])))
        rows = rows[:top_k]

    best_hit_seen = False
    for row in rows:
        if bool(row["is_hit"]) and not best_hit_seen:
            row["is_best_hit"] = True
            best_hit_seen = True
        else:
            row["is_best_hit"] = False
    return rows


def plot_query_chart(
    *,
    doc_id: str,
    query: QueryArtifacts,
    output_path: Path,
    dpi: int,
    confidence_floor: float,
) -> None:
    dense_color = "#2f5aa8"
    bm25_color = "#d97706"
    accent = "#1f2937"
    grid = "#d7dde8"
    bg = "#fbfcfe"

    labels = [
        f"{display_chunk_label(str(row['chunk_id']))} | pg.{row['page']}" if row.get("page") is not None else display_chunk_label(str(row["chunk_id"]))
        for row in query.rows
    ]
    dense_vals = np.asarray([float(row["dense_contrib"]) for row in query.rows], dtype=float)
    bm25_vals = np.asarray([float(row["bm25_contrib"]) for row in query.rows], dtype=float)
    totals = dense_vals + bm25_vals
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(10.5, 0.82 * len(labels)), 6.8))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    y_min = 0.0
    y_max = float(totals.max()) * 1.14
    ax.set_ylim(y_min, y_max)
    ax.axhline(confidence_floor, color="#c0392b", linestyle="--", linewidth=1.1, alpha=0.9, zorder=0)
    ax.text(
        0.995,
        confidence_floor,
        f" confidence floor = {confidence_floor:.2f}",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8,
        color="#c0392b",
    )

    hit_x = [xi for xi, row in zip(x, query.rows) if bool(row.get("is_best_hit"))]
    hit_y = [yy for yy, row in zip(totals, query.rows) if bool(row.get("is_best_hit"))]
    miss_x = [xi for xi, row in zip(x, query.rows) if not bool(row.get("is_best_hit"))]
    miss_y = [yy for yy, row in zip(totals, query.rows) if not bool(row.get("is_best_hit"))]

    if miss_x:
        ax.scatter(
            miss_x,
            miss_y,
            s=145,
            color="#64748b",
            marker="o",
            edgecolors="white",
            linewidths=0.9,
            zorder=3,
        )
    if hit_x:
        ax.scatter(
            hit_x,
            hit_y,
            s=210,
            color="#111827",
            marker="*",
            edgecolors="#f8fafc",
            linewidths=0.9,
            zorder=4,
        )

    ax.scatter(
        x - 0.12,
        dense_vals,
        s=42,
        color=dense_color,
        marker="o",
        edgecolors="white",
        linewidths=0.6,
        zorder=2,
        label="Dense contribution",
    )
    ax.scatter(
        x + 0.12,
        bm25_vals,
        s=42,
        color=bm25_color,
        marker="s",
        edgecolors="white",
        linewidths=0.6,
        zorder=2,
        label="BM25 contribution",
    )

    for rank_idx, (xi, total, row) in enumerate(zip(x, totals, query.rows), start=1):
        default_offset = max(totals.max() * 0.015, 0.0025)
        y_text = total + default_offset
        va = "bottom"
        if y_text > y_max * 0.985:
            y_text = total - default_offset * 1.3
            va = "top"
        if rank_idx <= 3:
            ax.text(
                xi,
                y_text,
                f"{total:.3f}",
                ha="center",
                va=va,
                fontsize=8,
                color=accent,
            )
        if row["confidence_label"] == "high":
            ax.scatter([xi], [total], s=34, color="#059669", zorder=5, edgecolors="white", linewidths=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="center", fontsize=8.0)
    ax.set_ylabel("RRF score")
    ax.set_xlabel("Chunks sorted by final rank")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, color=grid)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    question = query.question.strip()
    if len(question) > 115:
        question = question[:112].rstrip() + "..."
    fig.suptitle(
        f"{doc_id} | {query.query_id}",
        fontsize=13,
        color=accent,
        y=0.975,
    )
    fig.text(
        0.5,
        0.932,
        question,
        ha="center",
        va="top",
        fontsize=10.5,
        color=accent,
    )
    fig.text(
        0.5,
        0.900,
        "★ = best-ranked correct page | ● Dense score | ■ BM25 score | main marker = fused RRF",
        ha="center",
        va="top",
        fontsize=8.8,
        color=accent,
    )
    fig.text(
        0.5,
        0.875,
        (
            f"Confidence floor = {confidence_floor:.2f} | "
            f"Balance = {query.top_confidence_score:.2f} "
            "(0 = one signal dominates, 0.5 = even) | Below this line: weak candidates unlikely to be correct"
        ),
        ha="center",
        va="top",
        fontsize=8.8,
        color=accent,
    )
    fig.text(
        0.5,
        0.850,
        r"Dense contributions are numerically smaller than BM25 here, but $\bf{still\ influence\ the\ fused\ rank}$.",
        ha="center",
        va="top",
        fontsize=8.5,
        color=accent,
    )
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.subplots_adjust(top=0.86, bottom=0.32, right=0.96)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_parquet(data_dir / "chunk_meta.parquet")
    chunks = pd.read_parquet(data_dir / "chunks.parquet")
    index = faiss.read_index(str(data_dir / "faiss.index"))
    eval_queries = load_eval_queries(data_dir / "eval_set.json")
    stored_results = load_stored_hybrid_results(data_dir / "retrieval_results_hybrid.json")

    if args.query_id:
        eval_queries = [query for query in eval_queries if str(query.get("query_id", "")).strip() == args.query_id]
        if not eval_queries:
            raise ValueError(f"query_id={args.query_id!r} not found in {data_dir / 'eval_set.json'}")

    text_by_id = build_text_lookup(chunks)
    corpus_texts = [text_by_id.get(get_chunk_id(meta.iloc[idx]), "") for idx in range(len(meta))]
    bm25 = BM25Index([tokenize(text) for text in corpus_texts], k1=float(args.bm25_k1), b=float(args.bm25_b))

    model = SentenceTransformer(str(args.model), device=str(args.device))
    questions = [str(query.get("question", "")).strip() for query in eval_queries]
    question_embeddings = model.encode(
        questions,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    ).astype("float32")
    question_embeddings = l2_normalize(question_embeddings).astype("float32")

    doc_id = str(meta.iloc[0].get("doc_id") or data_dir.name) if len(meta) else data_dir.name
    summary_rows: list[dict[str, object]] = []

    for query, question_embedding in zip(eval_queries, question_embeddings):
        query_id = str(query.get("query_id", "")).strip()
        question = str(query.get("question", "")).strip()
        stored_result = stored_results.get(query_id, {})
        stored_per10 = {}
        if isinstance(stored_result.get("per_k"), dict):
            stored_per10 = stored_result["per_k"].get("10", {}) or {}
        rows = compute_rrf_breakdown(
            meta=meta,
            index=index,
            bm25=bm25,
            question=question,
            question_embedding=question_embedding,
            max_k_search=int(args.max_k_search),
            rrf_k=int(args.rrf_k),
            dense_weight=float(args.dense_weight),
            bm25_weight=float(args.bm25_weight),
            top_k=int(args.top_k),
            expected_doc_id=str(query.get("doc_id", "")).strip(),
            expected_pages=to_int_set(query.get("expected_pages")),
            stored_top_chunk_ids=list(stored_per10.get("retrieved_chunk_ids", []) or []),
            stored_top_flags=list(stored_per10.get("chunk_hit_flags", []) or []),
        )
        if not rows:
            continue

        top_row = rows[0]
        query_artifacts = QueryArtifacts(
            query_id=query_id,
            question=question,
            rows=rows,
            top_confidence_label=str(top_row["confidence_label"]),
            top_confidence_score=float(top_row["confidence_balance"]),
        )
        chart_path = output_dir / f"{safe_slug(query_id)}_rank_stability_rrf.png"
        plot_query_chart(
            doc_id=doc_id,
            query=query_artifacts,
            output_path=chart_path,
            dpi=int(args.dpi),
            confidence_floor=float(args.confidence_floor),
        )

        for rank, row in enumerate(rows, start=1):
            summary_rows.append(
                {
                    "doc_id": doc_id,
                    "query_id": query_id,
                    "question": question,
                    "fused_rank": rank,
                    **row,
                    "chart_path": str(chart_path),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "rank_stability_rrf_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_df['query_id'].nunique() if not summary_df.empty else 0} chart groups to {output_dir}")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
