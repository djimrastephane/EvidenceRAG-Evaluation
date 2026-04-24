from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "retrieval_pairwise_win_loss_tie_2026-03-26"


@dataclass(frozen=True)
class SourceSpec:
    label: str
    root: Path
    summary_name: str


SOURCES = {
    "dense": SourceSpec(
        label="Dense-only (all-MiniLM-L6-v2)",
        root=ROOT / "results" / "dense_encoder_ablation" / "smoke_l6_only" / "artifacts" / "all-MiniLM-L6-v2" / "source_docs",
        summary_name="retrieval_summary.csv",
    ),
    "hybrid": SourceSpec(
        label="Hybrid (MiniLM + BM25 default tokenizer)",
        root=ROOT / "results" / "bm25_tokenizer_sensitivity" / "thesis_bm25_tokenizer_sensitivity_20260325" / "hybrid_default",
        summary_name="retrieval_summary_hybrid.csv",
    ),
    "bm25": SourceSpec(
        label="BM25-only (default tokenizer)",
        root=ROOT / "results" / "bm25_tokenizer_sensitivity" / "thesis_bm25_tokenizer_sensitivity_20260325" / "bm25_default",
        summary_name="retrieval_summary_bm25.csv",
    ),
}


METRICS = {
    "Hit@1": (1, "page_recall_at_k"),
    "Hit@3": (3, "page_recall_at_k"),
    "MRR@10": (10, "page_mrr_at_k"),
}


def load_rows(spec: SourceSpec) -> dict[str, dict[int, dict[str, str]]]:
    rows: dict[str, dict[int, dict[str, str]]] = {}
    for doc_dir in sorted(spec.root.iterdir()):
        if not doc_dir.is_dir():
            continue
        path = doc_dir / spec.summary_name
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.setdefault(row["query_id"], {})[int(row["k"])] = row
    return rows


def compare(
    left: dict[str, dict[int, dict[str, str]]],
    right: dict[str, dict[int, dict[str, str]]],
) -> list[dict[str, object]]:
    query_ids = sorted(set(left).intersection(right))
    out: list[dict[str, object]] = []
    for metric_name, (k, field) in METRICS.items():
        wins = losses = ties = 0
        for qid in query_ids:
            left_val = float(left[qid][k][field])
            right_val = float(right[qid][k][field])
            if abs(left_val - right_val) < 1e-12:
                ties += 1
            elif left_val > right_val:
                wins += 1
            else:
                losses += 1
        out.append(
            {
                "metric": metric_name,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "queries_compared": len(query_ids),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["metric", "wins", "losses", "ties", "queries_compared"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_md(
    path: Path,
    title: str,
    left: SourceSpec,
    right: SourceSpec,
    rows: list[dict[str, object]],
) -> None:
    lines = [
        f"# {title}",
        "",
        f"- Left system: `{left.label}`",
        f"- Right system: `{right.label}`",
        f"- Left source: `{left.root}`",
        f"- Right source: `{right.root}`",
        "",
        "| Metric | Wins | Losses | Ties | Queries |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['metric']} | {row['wins']} | {row['losses']} | {row['ties']} | {row['queries_compared']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    loaded = {name: load_rows(spec) for name, spec in SOURCES.items()}

    comparisons = [
        ("hybrid_vs_dense", "Hybrid vs Dense", SOURCES["hybrid"], SOURCES["dense"]),
        ("hybrid_vs_bm25", "Hybrid vs BM25", SOURCES["hybrid"], SOURCES["bm25"]),
    ]

    summary_rows: list[dict[str, object]] = []
    for slug, title, left_spec, right_spec in comparisons:
        rows = compare(loaded[slug.split("_vs_")[0]], loaded[slug.split("_vs_")[1]])
        write_csv(OUT_DIR / f"{slug}_win_loss_tie.csv", rows)
        write_md(OUT_DIR / f"{slug}_win_loss_tie.md", title, left_spec, right_spec, rows)
        for row in rows:
            summary_rows.append(
                {
                    "comparison": title,
                    "metric": row["metric"],
                    "wins": row["wins"],
                    "losses": row["losses"],
                    "ties": row["ties"],
                    "queries_compared": row["queries_compared"],
                }
            )

    with (OUT_DIR / "all_win_loss_tie_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["comparison", "metric", "wins", "losses", "ties", "queries_compared"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
