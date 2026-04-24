from __future__ import annotations

"""Reproduce thesis Figure 4.2 from thesis_rag dense and boosted-hybrid runs.

Figure 4.2 compares rank-based retrieval dynamics for:

- Dense (MiniLM)
- Hybrid + subsection boost

The output bundle includes:

- a per-query rank survival table
- a Kaplan-Meier style curve table with bootstrap intervals
- PNG/PDF figure exports
"""

import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from random import Random

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from thesis_rag.artifacts import load_queries
from thesis_rag.evaluator import evaluate_page_hits
from thesis_rag.schemas import RetrievalHit


DOC_IDS = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]
ARTIFACT_ROOT = REPO_ROOT / "results" / "thesis_ablations" / "chunk_size_ablation_2026-04-15" / "pipeline_outputs"
ARTIFACT_ROOT_BOOST_OFF = REPO_ROOT / "results" / "thesis_ablations" / "chunk_size_ablation_boost_off_2026-04-20" / "pipeline_outputs"
OUT_DIR = REPO_ROOT / "results" / "thesis_figures" / f"figure_4_2_rank_survival_{date.today().isoformat()}"
OUT_PER_QUERY = OUT_DIR / "retrieval_rank_survival_compare.csv"
OUT_CURVE = OUT_DIR / "retrieval_rank_km_compare_curve.csv"
OUT_PNG = OUT_DIR / "retrieval_rank_km_compare_curve_current_thesis_rag.png"
OUT_PDF = OUT_DIR / "retrieval_rank_km_compare_curve_current_thesis_rag.pdf"
BOOTSTRAP_SAMPLES = 5000
MAX_RANK = 10


@dataclass(frozen=True)
class QueryOutcome:
    system: str
    doc_id: str
    query_id: str
    question: str
    expected_pages: list[int]
    ranked_pages_observed: list[int]
    top_k_limit: int
    first_correct_rank: float | None
    event: int
    time_rank: int
    censored: int
    paired_query_id: str


def _load_hits(path: Path) -> list[RetrievalHit]:
    import pandas as pd

    frame = pd.read_csv(path)
    return [RetrievalHit(**row) for row in frame.to_dict(orient="records")]


def _collect_results(system: str, filename: str, artifact_root: Path | None = None) -> list[QueryOutcome]:
    root = artifact_root if artifact_root is not None else ARTIFACT_ROOT
    outcomes: list[QueryOutcome] = []
    for doc_id in DOC_IDS:
        artifact_dir = root / f"minilmcap_{doc_id}_chunk_224_56" / doc_id
        hits = _load_hits(artifact_dir / filename)
        queries = load_queries(REPO_ROOT / "data_processed" / doc_id / "eval_set.json")
        grouped_pages: dict[str, list[int]] = {}
        for hit in hits:
            grouped_pages.setdefault(hit.query_id, []).append(int(hit.page_number))
        for result, query in zip(evaluate_page_hits(queries, hits), queries):
            ranked_pages = grouped_pages.get(query.query_id, [])
            first_rank = float(result.first_relevant_rank) if result.first_relevant_rank is not None else None
            time_rank = int(result.first_relevant_rank) if result.first_relevant_rank is not None else MAX_RANK
            censored = 0 if result.first_relevant_rank is not None else 1
            outcomes.append(
                QueryOutcome(
                    system=system,
                    doc_id=doc_id,
                    query_id=query.query_id,
                    question=query.query_text,
                    expected_pages=list(query.gold_pages),
                    ranked_pages_observed=ranked_pages[:MAX_RANK],
                    top_k_limit=MAX_RANK,
                    first_correct_rank=first_rank,
                    event=0 if censored else 1,
                    time_rank=time_rank,
                    censored=censored,
                    paired_query_id=f"{doc_id}::{query.query_id}",
                )
            )
    if len(outcomes) != 250:
        raise RuntimeError(f"Expected 250 outcomes for {system}, found {len(outcomes)}")
    return outcomes


def _km_curve(outcomes: list[QueryOutcome], rng: Random) -> list[dict[str, float | int | str]]:
    times = [outcome.time_rank for outcome in outcomes]
    events = [outcome.event for outcome in outcomes]
    rows: list[dict[str, float | int | str]] = []
    for rank in range(1, MAX_RANK + 1):
        at_risk = sum(1 for time in times if time >= rank)
        event_count = sum(1 for time, event in zip(times, events) if time == rank and event == 1)
        censored = sum(1 for time, event in zip(times, events) if time == rank and event == 0)
        survival = sum(1 for time, event in zip(times, events) if not (time == rank and event == 1))
        # Kaplan-Meier step value after processing events at this rank.
        surv_prob = sum(1 for time, event in zip(times, events) if event == 0 or time > rank) / len(outcomes)

        boot_values: list[float] = []
        for _ in range(BOOTSTRAP_SAMPLES):
            indices = [rng.randrange(len(outcomes)) for _ in range(len(outcomes))]
            sample = [outcomes[idx] for idx in indices]
            sample_surv = sum(1 for item in sample if item.censored == 1 or item.time_rank > rank) / len(sample)
            boot_values.append(sample_surv)
        boot_values.sort()
        lower = boot_values[int(0.025 * BOOTSTRAP_SAMPLES)]
        upper = boot_values[int(0.975 * BOOTSTRAP_SAMPLES)]
        mean_boot = sum(boot_values) / len(boot_values)

        rows.append(
            {
                "rank": rank,
                "at_risk": at_risk,
                "events": event_count,
                "censored": censored,
                "survival_probability": surv_prob,
                "mean_survival_probability": mean_boot,
                "ci_lower": lower,
                "ci_upper": upper,
            }
        )
    return rows


def _write_per_query(dense: list[QueryOutcome], hybrid: list[QueryOutcome]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_PER_QUERY.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "system",
                "doc_id",
                "query_id",
                "question",
                "expected_pages",
                "ranked_pages_observed",
                "top_k_limit",
                "first_correct_rank",
                "event",
                "time_rank",
                "censored",
                "paired_query_id",
            ],
        )
        writer.writeheader()
        for row in [*dense, *hybrid]:
            writer.writerow(
                {
                    "system": row.system,
                    "doc_id": row.doc_id,
                    "query_id": row.query_id,
                    "question": row.question,
                    "expected_pages": json.dumps(row.expected_pages),
                    "ranked_pages_observed": json.dumps(row.ranked_pages_observed),
                    "top_k_limit": row.top_k_limit,
                    "first_correct_rank": row.first_correct_rank if row.first_correct_rank is not None else "",
                    "event": row.event,
                    "time_rank": row.time_rank,
                    "censored": row.censored,
                    "paired_query_id": row.paired_query_id,
                }
            )


def _write_curve(dense_rows: list[dict[str, float | int | str]], hybrid_rows: list[dict[str, float | int | str]]) -> None:
    with OUT_CURVE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "system",
                "rank",
                "at_risk",
                "events",
                "censored",
                "survival_probability",
                "mean_survival_probability",
                "ci_lower",
                "ci_upper",
            ],
        )
        writer.writeheader()
        for system, rows in (("dense", dense_rows), ("hybrid", hybrid_rows)):
            for row in rows:
                writer.writerow({"system": system, **row})


def _plot(
    dense_rows: list[dict[str, float | int | str]],
    hybrid_base_rows: list[dict[str, float | int | str]],
    hybrid_boost_rows: list[dict[str, float | int | str]],
) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
        }
    )

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    styles = {
        "dense":        {"label": "Dense (MiniLM)",          "color": "#4C78A8", "ls": "--"},
        "hybrid_base":  {"label": "Hybrid (base)",           "color": "#72B7B2", "ls": ":"},
        "hybrid_boost": {"label": "Hybrid + subsection boost", "color": "#F58518", "ls": "-"},
    }
    for system, rows in (
        ("dense",        dense_rows),
        ("hybrid_base",  hybrid_base_rows),
        ("hybrid_boost", hybrid_boost_rows),
    ):
        xs = [int(row["rank"]) for row in rows]
        ys = [float(row["survival_probability"]) for row in rows]
        lo = [float(row["ci_lower"]) for row in rows]
        hi = [float(row["ci_upper"]) for row in rows]
        color = styles[system]["color"]
        ls    = styles[system]["ls"]
        ax.step(xs, ys, where="post", label=styles[system]["label"],
                color=color, linewidth=2.2, linestyle=ls)
        ax.fill_between(xs, lo, hi, step="post", color=color, alpha=0.12)

    ax.set_xlim(1, MAX_RANK)
    ax.set_ylim(0.0, 0.30)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Survival probability")
    ax.set_title("Rank-based comparison: dense, hybrid base, and boosted-hybrid retrieval")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=320, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Regenerate Figure 4.2 Kaplan-Meier-style retrieval curves comparing dense vs hybrid variants."""
    rng = Random(13)
    dense = _collect_results("dense", "dense_page_hits.csv")
    hybrid_base = _collect_results("hybrid_base", "hybrid_page_hits.csv", artifact_root=ARTIFACT_ROOT_BOOST_OFF)
    hybrid_boost = _collect_results("hybrid_boost", "hybrid_page_hits.csv")
    dense_rows = _km_curve(dense, rng)
    hybrid_base_rows = _km_curve(hybrid_base, rng)
    hybrid_boost_rows = _km_curve(hybrid_boost, rng)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_per_query(dense, hybrid_boost)
    _write_curve(dense_rows, hybrid_boost_rows)
    _plot(dense_rows, hybrid_base_rows, hybrid_boost_rows)
    summary = {
        "dense_hit_at_1": 1.0 - dense_rows[0]["survival_probability"],
        "hybrid_base_hit_at_1": 1.0 - hybrid_base_rows[0]["survival_probability"],
        "hybrid_boost_hit_at_1": 1.0 - hybrid_boost_rows[0]["survival_probability"],
        "output_dir": str(OUT_DIR),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
