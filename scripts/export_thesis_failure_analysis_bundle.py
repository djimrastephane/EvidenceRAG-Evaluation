"""Bundle FP1-FP7 failure-analysis artifacts into the frozen thesis export directory.

Copies counts CSVs, summary JSONs, per-query CSVs, and heatmap PNGs from one or two
completed run_current_pipeline_fp1_fp7.py output directories into a structured export
folder. Optionally includes a side-by-side comparison directory from compare_fp1_fp7_runs.py.
Writes an aggregate FP1-FP7 run summary table (CSV, JSON, LaTeX) and a scope manifest.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from thesis_provenance import write_scope_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export thesis-facing FP1-FP7 artifacts from one or two completed analysis directories."
    )
    p.add_argument("--baseline-dir", required=True, help="Directory with current_pipeline_fp1_fp7_* outputs.")
    p.add_argument("--candidate-dir", default="", help="Optional second directory, e.g. LLM-on run.")
    p.add_argument("--comparison-dir", default="", help="Optional comparison directory from compare_fp1_fp7_runs.py.")
    p.add_argument("--out-dir", required=True, help="Directory where frozen exports should be written.")
    return p.parse_args()


def _load_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "current_pipeline_fp1_fp7_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _label_from_dir(path: Path) -> str:
    name = path.name.lower()
    if "llm" in name and "norm" in name:
        return "llm_on_normalized"
    if "llm" in name:
        return "llm_on"
    if "norm" in name:
        return "retrieval_only_normalized"
    return "retrieval_only"


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _totals(summary: dict[str, Any]) -> dict[str, int]:
    totals = summary.get("totals_by_failure_type")
    if isinstance(totals, dict):
        return {str(k): int(v) for k, v in totals.items()}
    # fallback for older schema
    return {str(k): int(v) for k, v in (summary.get("totals") or {}).items()}


def _latex_table(df: pd.DataFrame) -> str:
    lines = [
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "Run & HIT & FP1 & FP2 & FP3 & FP4 & FP5 & FP6 & FP7 \\\\",
        "\\midrule",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"{row.run} & {row.HIT} & {row.FP1} & {row.FP2} & {row.FP3} & {row.FP4} & {row.FP5} & {row.FP6} & {row.FP7} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    baseline_dir = Path(args.baseline_dir).resolve()
    candidate_dir = Path(args.candidate_dir).resolve() if str(args.candidate_dir).strip() else None
    comparison_dir = Path(args.comparison_dir).resolve() if str(args.comparison_dir).strip() else None
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[tuple[str, Path, dict[str, Any]]] = []
    baseline_summary = _load_summary(baseline_dir)
    runs.append((_label_from_dir(baseline_dir), baseline_dir, baseline_summary))
    if candidate_dir is not None:
        candidate_summary = _load_summary(candidate_dir)
        runs.append((_label_from_dir(candidate_dir), candidate_dir, candidate_summary))

    summary_rows: list[dict[str, Any]] = []
    for run_label, run_dir, summary in runs:
        totals = _totals(summary)
        summary_rows.append(
            {
                "run": run_label,
                "HIT": int(totals.get("HIT", 0)),
                "FP1": int(totals.get("FP1_MISSING_CONTENT", 0) or totals.get("FP1", 0)),
                "FP2": int(totals.get("FP2_MISSED_TOP_RANK", 0) or totals.get("FP2", 0)),
                "FP3": int(totals.get("FP3_NOT_IN_CONTEXT", 0) or totals.get("FP3", 0)),
                "FP4": int(totals.get("FP4_NOT_EXTRACTED", 0) or totals.get("FP4", 0)),
                "FP5": int(totals.get("FP5_WRONG_FORMAT", 0) or totals.get("FP5", 0)),
                "FP6": int(totals.get("FP6_INCORRECT_SPECIFICITY", 0) or totals.get("FP6", 0)),
                "FP7": int(totals.get("FP7_INCOMPLETE", 0) or totals.get("FP7", 0)),
                "source_dir": str(run_dir),
            }
        )
        for name in [
            "current_pipeline_fp1_fp7_counts.csv",
            "current_pipeline_fp1_fp7_summary.json",
            "current_pipeline_fp1_fp7_per_query.csv",
            "current_pipeline_fp1_fp7_heatmap.png",
            "current_pipeline_fp1_fp7_heatmap_labeled.png",
        ]:
            _copy_if_exists(run_dir / name, out_dir / run_label / name)

    exported_files: list[Path] = []
    if comparison_dir is not None:
        for name in [
            "fp1_fp7_comparison_summary.json",
            "fp1_fp7_counts_delta.csv",
            "fp1_fp7_per_query_comparison.csv",
            "fp1_fp7_transition_matrix.csv",
            "fp1_fp7_heatmaps_side_by_side.png",
            "fp1_fp7_heatmaps_side_by_side_labeled.png",
            "fp1_fp7_heatmaps_side_by_side_norm.png",
            "fp1_fp7_heatmaps_side_by_side_norm_labeled.png",
        ]:
            _copy_if_exists(comparison_dir / name, out_dir / "comparison" / name)

    df = pd.DataFrame(summary_rows)
    csv_path = out_dir / "fp1_fp7_run_summary.csv"
    json_path = out_dir / "fp1_fp7_run_summary.json"
    tex_path = out_dir / "fp1_fp7_run_summary.tex"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(json.loads(df.to_json(orient="records")), indent=2), encoding="utf-8")
    tex_path.write_text(_latex_table(df[["run", "HIT", "FP1", "FP2", "FP3", "FP4", "FP5", "FP6", "FP7"]]), encoding="utf-8")

    for path in out_dir.rglob("*"):
        if path.is_file() and path.name != ".DS_Store":
            exported_files.append(path)
    write_scope_manifest(
        out_dir=out_dir,
        scope_name="failure_analysis",
        source_inputs={
            "baseline_dir": str(baseline_dir),
            "candidate_dir": str(candidate_dir) if candidate_dir else None,
            "comparison_dir": str(comparison_dir) if comparison_dir else None,
            "runs_exported": [row["run"] for row in summary_rows],
        },
        exported_files=sorted(exported_files),
        notes=["FP1-FP7 exports copied from completed analysis directories."],
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
