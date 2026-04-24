from __future__ import annotations

import argparse
import csv
from pathlib import Path

import _matplotlib_env
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


SUMMARY_COLS = {
    "query_id": 0,
    "topic": 1,
    "year": 2,
    "query_num": 3,
    "k": 4,
    "answer_type": 5,
    "doc_id": 6,
    "section": 7,
    "gold_pages": 8,
    "failure_type": 14,
    "top_pages": 24,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge chunk-size retrieval_summary.csv files and build a comparison chart."
    )
    parser.add_argument("--summary-224", required=True, help="Path to retrieval_summary.csv for 224/56.")
    parser.add_argument("--summary-256", required=True, help="Path to retrieval_summary.csv for 256/64.")
    parser.add_argument("--summary-280", required=True, help="Path to retrieval_summary.csv for 280/90.")
    parser.add_argument("--out-csv", required=True, help="Output merged CSV path.")
    parser.add_argument("--out-png", required=True, help="Output chart PNG path.")
    parser.add_argument("--title", default="Chunk Size Comparison: Grampian-2022-2023")
    return parser.parse_args()


def load_summary(path: Path, label: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if str(row[SUMMARY_COLS["k"]]).strip().lower() == "k":
                continue
            failure_type = str(row[SUMMARY_COLS["failure_type"]]).strip()
            rows.append(
                {
                    "query_id": str(row[SUMMARY_COLS["query_id"]]).strip(),
                    "topic": str(row[SUMMARY_COLS["topic"]]).strip(),
                    "year": str(row[SUMMARY_COLS["year"]]).strip(),
                    "query_num": str(row[SUMMARY_COLS["query_num"]]).strip(),
                    "k": int(row[SUMMARY_COLS["k"]]),
                    "answer_type": str(row[SUMMARY_COLS["answer_type"]]).strip(),
                    "doc_id": str(row[SUMMARY_COLS["doc_id"]]).strip(),
                    "section": str(row[SUMMARY_COLS["section"]]).strip(),
                    "gold_pages": str(row[SUMMARY_COLS["gold_pages"]]).strip(),
                    f"status_{label}": failure_type,
                    f"hit_{label}": 1 if failure_type == "hit" else 0,
                    f"top_pages_{label}": str(row[SUMMARY_COLS["top_pages"]]).strip(),
                }
            )
    return pd.DataFrame(rows)


def build_merged_dataframe(paths: dict[str, Path]) -> pd.DataFrame:
    base_cols = ["query_id", "topic", "year", "query_num", "k", "answer_type", "doc_id", "section", "gold_pages"]
    merged: pd.DataFrame | None = None
    for label, path in paths.items():
        df = load_summary(path=path, label=label)
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=base_cols, how="outer")
    assert merged is not None
    merged = merged.sort_values(["k", "query_id"], kind="stable").reset_index(drop=True)
    return merged


def plot_hit_heatmaps(df: pd.DataFrame, out_png: Path, title: str) -> None:
    labels = ["224_56", "256_64", "280_90"]
    pretty = ["224 / 56", "256 / 64", "280 / 90"]
    k_values = [1, 3]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 15.5), sharey=True)
    fig.patch.set_facecolor("#f7f4ec")
    cmap = plt.matplotlib.colors.ListedColormap(["#d95d39", "#2d936c"])

    base_queries = df[df["k"] == 1][["query_id", "topic"]].drop_duplicates()
    disagreement_rows: list[tuple[str, int, str]] = []
    for _, row in base_queries.iterrows():
        query_id = str(row["query_id"])
        topic = str(row["topic"])
        vals: list[int] = []
        for k in k_values:
            sub = df[(df["query_id"] == query_id) & (df["k"] == k)]
            if sub.empty:
                continue
            vals.extend(
                [
                    int(sub["hit_224_56"].iloc[0]),
                    int(sub["hit_256_64"].iloc[0]),
                    int(sub["hit_280_90"].iloc[0]),
                ]
            )
        disagreement = sum(v != vals[0] for v in vals[1:]) if vals else 0
        disagreement_rows.append((query_id, disagreement, topic))
    query_order = [
        qid
        for qid, _, _ in sorted(disagreement_rows, key=lambda item: (-item[1], item[2], item[0]))
    ]

    for ax, k in zip(axes, k_values):
        sub = df[df["k"] == k].copy()
        sub["query_id"] = pd.Categorical(sub["query_id"], categories=query_order, ordered=True)
        sub = sub.sort_values(["query_id"], kind="stable")
        matrix = sub[[f"hit_{label}" for label in labels]].to_numpy(dtype=float)
        ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"Top-{k} Retrieval", fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks(np.arange(len(pretty)))
        ax.set_xticklabels(pretty, rotation=0, ha="center", fontsize=10)
        ax.set_yticks(np.arange(len(sub)))
        ax.set_yticklabels(sub["query_id"].tolist(), fontsize=8)
        ax.set_xlabel("Chunk size / overlap", fontsize=10)
        ax.set_xticks(np.arange(-0.5, len(pretty), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(sub), 1), minor=True)
        ax.grid(which="minor", color="#ffffff", linestyle="-", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(axis="y", length=0)
        ax.set_facecolor("#efe9dc")
        for spine in ax.spines.values():
            spine.set_visible(False)

        topics = sub["topic"].astype(str).tolist()
        for idx in range(1, len(topics)):
            if topics[idx] != topics[idx - 1]:
                ax.axhline(idx - 0.5, color="#d8cfbe", linewidth=2.2)

    axes[0].set_ylabel("Query ID", fontsize=10)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.955,
        "Queries are ordered by disagreement across chunk sizes; separators indicate topic changes.",
        ha="center",
        fontsize=10,
        color="#4d4d4d",
    )
    legend_handles = [
        Patch(facecolor="#2d936c", edgecolor="none", label="Hit"),
        Patch(facecolor="#d95d39", edgecolor="none", label="Missed top ranked"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
        fontsize=10,
    )
    plt.tight_layout(rect=[0.03, 0.05, 0.98, 0.94])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    paths = {
        "224_56": Path(args.summary_224),
        "256_64": Path(args.summary_256),
        "280_90": Path(args.summary_280),
    }
    merged = build_merged_dataframe(paths=paths)

    out_csv = Path(args.out_csv)
    out_png = Path(args.out_png)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    plot_hit_heatmaps(df=merged, out_png=out_png, title=args.title)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
