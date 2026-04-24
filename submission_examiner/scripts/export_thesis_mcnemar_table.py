"""Export the thesis McNemar Hit@1 summary table (CSV, JSON, LaTeX) from a batch summary CSV.

Reads the batch summary CSV produced by run_mcnemar_hit1_batch.py, which contains per-cohort
discordant pair counts, exact binomial p-values, and significance flags for the hybrid vs
dense Hit@1 comparison. Writes the table in three formats and records a scope manifest
linking the exports to their source inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from thesis_provenance import write_scope_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a thesis-ready McNemar Hit@1 summary table.")
    p.add_argument("--batch-summary-csv", required=True, help="CSV from run_mcnemar_hit1_batch.py.")
    p.add_argument("--out-dir", required=True, help="Directory where CSV/JSON/TeX exports should be written.")
    return p.parse_args()


def _latex_table(df: pd.DataFrame) -> str:
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Cohort & Paired queries & Hybrid only & Dense only & $p$-value & Significant \\\\",
        "\\midrule",
    ]
    for row in df.itertuples(index=False):
        sig = "Yes" if bool(row.significant) else "No"
        lines.append(
            f"{row.cohort} & {int(row.n_paired_queries)} & {int(row.hybrid_correct_dense_wrong)} & {int(row.hybrid_wrong_dense_correct)} & {float(row.p_value):.4f} & {sig} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    csv_in = Path(args.batch_summary_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_in)
    csv_path = out_dir / "mcnemar_hit1_summary_table.csv"
    json_path = out_dir / "mcnemar_hit1_summary_table.json"
    tex_path = out_dir / "mcnemar_hit1_summary_table.tex"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(json.loads(df.to_json(orient="records")), indent=2), encoding="utf-8")
    tex_path.write_text(
        _latex_table(
            df[
                [
                    "cohort",
                    "n_paired_queries",
                    "hybrid_correct_dense_wrong",
                    "hybrid_wrong_dense_correct",
                    "p_value",
                    "significant",
                ]
            ]
        ),
        encoding="utf-8",
    )
    write_scope_manifest(
        out_dir=out_dir,
        scope_name="mcnemar",
        source_inputs={"batch_summary_csv": str(csv_in)},
        exported_files=[csv_path, json_path, tex_path],
        notes=["McNemar thesis exports derived from batch summary CSV."],
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
