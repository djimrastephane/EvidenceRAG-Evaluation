"""Export the thesis chunk-ablation table (CSV, JSON, LaTeX) from frozen retrieval outputs.

Walks the per-experiment output directory tree, reads retrieval_metrics.json and metrics.json
from each (doc, chunk_size, overlap) combination, computes an aggregate Hit@1 and MRR@10
table with deltas relative to the 224/56 baseline, and writes the table in three formats
plus a per-document breakdown CSV and a scope manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from thesis_provenance import write_scope_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a thesis-facing chunk-ablation table from frozen per-document retrieval outputs."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Root containing per-experiment subdirectories with retrieval_metrics.json and metrics.json.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where CSV/JSON/TeX exports should be written.",
    )
    return parser.parse_args()


def _load_rows(data_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(data_root.rglob("retrieval_metrics.json")):
        experiment = path.parts[-3]
        document = path.parts[-2]
        size = int(experiment.split("_")[-2])
        overlap = int(experiment.split("_")[-1])

        retrieval_metrics = json.loads(path.read_text(encoding="utf-8"))
        prep_metrics = json.loads((path.parent / "metrics.json").read_text(encoding="utf-8"))

        k1 = retrieval_metrics["metrics_by_k"]["1"]
        k10 = retrieval_metrics["metrics_by_k"]["10"]

        rows.append(
            {
                "experiment": experiment,
                "document": document,
                "chunk_size_tokens": size,
                "chunk_overlap_tokens": overlap,
                "queries": int(k1["num_queries"]),
                "page_hit1": float(k1["page_hit_rate_at_k"]),
                "mrr10": float(k10["mean_page_mrr_at_k"]),
                "chunks_indexed": int(prep_metrics["counts"]["chunks_total"]),
            }
        )

    if not rows:
        raise RuntimeError(f"No retrieval_metrics.json files found under {data_root}")

    return pd.DataFrame(rows).sort_values(["document", "chunk_size_tokens", "chunk_overlap_tokens"])


def _build_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    aggregate = (
        df.groupby(["chunk_size_tokens", "chunk_overlap_tokens"], as_index=False)
        .agg(
            page_hit1=("page_hit1", "mean"),
            mrr10=("mrr10", "mean"),
            queries=("queries", "sum"),
            chunks_indexed=("chunks_indexed", "sum"),
        )
        .sort_values(["chunk_size_tokens", "chunk_overlap_tokens"])
        .reset_index(drop=True)
    )

    mask = (aggregate["chunk_size_tokens"] == 224) & (aggregate["chunk_overlap_tokens"] == 56)
    if not mask.any():
        raise RuntimeError("Could not find 224/56 baseline in aggregate data.")

    base_hit = float(aggregate.loc[mask, "page_hit1"].iloc[0])
    base_mrr = float(aggregate.loc[mask, "mrr10"].iloc[0])

    aggregate["delta_hit1"] = aggregate["page_hit1"] - base_hit
    aggregate["delta_mrr10"] = aggregate["mrr10"] - base_mrr
    aggregate["configuration"] = (
        aggregate["chunk_size_tokens"].astype(int).astype(str)
        + " / "
        + aggregate["chunk_overlap_tokens"].astype(int).astype(str)
    )
    return aggregate[
        ["configuration", "page_hit1", "delta_hit1", "mrr10", "delta_mrr10", "queries", "chunks_indexed"]
    ]


def _latex_table(df: pd.DataFrame) -> str:
    lines = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Configuration & Page Hit@1 & $\\Delta$Hit@1 & MRR@10 & $\\Delta$MRR@10 & Queries & Chunks Indexed \\\\",
        "\\midrule",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            (
                f"{row.configuration} & "
                f"{row.page_hit1:.3f} & "
                f"{row.delta_hit1:.3f} & "
                f"{row.mrr10:.3f} & "
                f"{row.delta_mrr10:.3f} & "
                f"{int(row.queries)} & "
                f"{int(row.chunks_indexed)} \\\\"
            )
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    per_doc = _load_rows(data_root=data_root)
    aggregate = _build_aggregate(df=per_doc)

    per_doc_csv = out_dir / "chunk_ablation_by_document.csv"
    aggregate_csv = out_dir / "chunk_ablation_table.csv"
    aggregate_json = out_dir / "chunk_ablation_table.json"
    aggregate_tex = out_dir / "chunk_ablation_table.tex"

    per_doc.to_csv(per_doc_csv, index=False)
    aggregate.to_csv(aggregate_csv, index=False)
    aggregate_json.write_text(
        json.dumps(json.loads(aggregate.to_json(orient="records")), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    aggregate_tex.write_text(_latex_table(aggregate), encoding="utf-8")
    write_scope_manifest(
        out_dir=out_dir,
        scope_name="tables",
        source_inputs={"data_root": str(data_root)},
        exported_files=[per_doc_csv, aggregate_csv, aggregate_json, aggregate_tex],
        notes=["Chunk ablation exports derived from frozen retrieval outputs."],
    )

    print(f"Wrote {per_doc_csv}")
    print(f"Wrote {aggregate_csv}")
    print(f"Wrote {aggregate_json}")
    print(f"Wrote {aggregate_tex}")


if __name__ == "__main__":
    main()
