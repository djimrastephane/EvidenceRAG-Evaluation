"""Export the thesis paired bootstrap summary table (CSV, JSON, LaTeX) from frozen run summaries.

Reads paired_bootstrap_summary.json files produced by paired_bootstrap_retrieval_compare.py,
assembles a cohort-level table of observed deltas and 95% confidence intervals for Hit@1,
Hit@3, and MRR@10 (hybrid vs dense), and writes the table in three formats plus a scope
manifest linking the exports back to their source inputs.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from thesis_provenance import write_scope_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a thesis-ready paired bootstrap summary table.")
    p.add_argument("--input-dir", required=True, help="Root containing paired_bootstrap_summary.json files.")
    p.add_argument("--out-dir", required=True, help="Directory where CSV/JSON/TeX exports should be written.")
    return p.parse_args()


def _row_from_summary(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    inputs = payload["inputs"]
    metrics = payload["metrics"]
    mrr_key = next(k for k in metrics.keys() if k.startswith("mrr_at_"))

    cohort = path.parent.name.replace("_hybrid_vs_dense", "")
    hit1 = metrics["hit_at_1"]
    hit3 = metrics["hit_at_3"]
    mrr = metrics[mrr_key]
    return {
        "cohort": cohort,
        "n_common_queries": int(inputs["n_common_queries"]),
        "n_bootstrap": int(inputs["n_bootstrap"]),
        "seed": int(inputs["seed"]),
        "hit1_delta": float(hit1["observed_delta"]),
        "hit1_ci_low": float(hit1["ci95_low"]),
        "hit1_ci_high": float(hit1["ci95_high"]),
        "hit3_delta": float(hit3["observed_delta"]),
        "hit3_ci_low": float(hit3["ci95_low"]),
        "hit3_ci_high": float(hit3["ci95_high"]),
        "mrr10_delta": float(mrr["observed_delta"]),
        "mrr10_ci_low": float(mrr["ci95_low"]),
        "mrr10_ci_high": float(mrr["ci95_high"]),
        "source_summary": str(path),
    }


def _latex_table(df: pd.DataFrame) -> str:
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Cohort & Queries & $\\Delta$Hit@1 & $\\Delta$Hit@3 & $\\Delta$MRR@10 \\\\",
        "\\midrule",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"{row.cohort} & {row.n_common_queries} & {row.hit1_delta:.3f} & {row.hit3_delta:.3f} & {row.mrr10_delta:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [_row_from_summary(path) for path in sorted(input_dir.rglob("paired_bootstrap_summary.json"))]
    if not rows:
        raise RuntimeError(f"No paired_bootstrap_summary.json files found under {input_dir}")

    df = pd.DataFrame(rows).sort_values(["cohort"])
    csv_path = out_dir / "paired_bootstrap_summary_table.csv"
    json_path = out_dir / "paired_bootstrap_summary_table.json"
    tex_path = out_dir / "paired_bootstrap_summary_table.tex"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(json.loads(df.to_json(orient="records")), indent=2), encoding="utf-8")
    tex_path.write_text(_latex_table(df[["cohort", "n_common_queries", "hit1_delta", "hit3_delta", "mrr10_delta"]]), encoding="utf-8")

    exported_files = [csv_path, json_path, tex_path]
    for png_name in [
        "paired_bootstrap_ci_panel_Grampian_2020_2025_hybrid_vs_dense.png",
        "paired_bootstrap_ci_panel_Grampian_2021_2025.png",
    ]:
        src = input_dir / png_name
        if src.exists():
            dst = out_dir / png_name
            shutil.copy2(src, dst)
            exported_files.append(dst)

    write_scope_manifest(
        out_dir=out_dir,
        scope_name="bootstrap",
        source_inputs={"input_dir": str(input_dir)},
        exported_files=exported_files,
        notes=["Paired bootstrap thesis exports derived from frozen bootstrap summaries."],
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
