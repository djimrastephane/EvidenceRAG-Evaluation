"""Export the thesis RAGAS evaluation summary table (CSV, JSON, LaTeX) from completed runs.

Reads ragas_summary.json from one or more RAGAS run directories (each specified as
label::path), assembles a comparison table of context precision, context recall, answer
relevancy, and faithfulness, and writes the table in three formats plus a scope manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from thesis_provenance import write_scope_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a thesis-ready RAGAS summary comparison table.")
    p.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run label and summary directory in the form label::/path/to/ragas_run_dir",
    )
    p.add_argument("--out-dir", required=True, help="Directory where CSV/JSON/TeX exports should be written.")
    return p.parse_args()


def _parse_run_arg(value: str) -> tuple[str, Path]:
    parts = value.split("::", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid --run value '{value}'. Expected label::/path/to/run_dir")
    label, path = parts
    run_dir = Path(path).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing RAGAS run directory: {run_dir}")
    return label.strip(), run_dir


def _row(label: str, run_dir: Path) -> dict[str, object]:
    summary_path = run_dir / "ragas_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing ragas_summary.json: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    means = payload.get("metrics_mean", {})
    return {
        "run": label,
        "input_rows": int(payload.get("input_rows", 0)),
        "scored_rows": int(payload.get("scored_rows", 0)),
        "answer_relevancy": means.get("answer_relevancy"),
        "faithfulness": means.get("faithfulness"),
        "context_precision": means.get("context_precision"),
        "context_recall": means.get("context_recall"),
        "llm_model": payload.get("llm_model"),
        "embedding_model": payload.get("embedding_model"),
        "source_dir": str(run_dir),
    }


def _fmt(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "--"
    return f"{float(value):.3f}"


def _latex_table(df: pd.DataFrame) -> str:
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Run & Rows & Answer rel. & Faithfulness & Context prec. & Context recall \\\\",
        "\\midrule",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"{row.run} & {int(row.scored_rows)} & {_fmt(row.answer_relevancy)} & {_fmt(row.faithfulness)} & {_fmt(row.context_precision)} & {_fmt(row.context_recall)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.run:
        raise ValueError("Provide at least one --run label::dir argument.")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [_row(label, run_dir) for label, run_dir in (_parse_run_arg(v) for v in args.run)]
    df = pd.DataFrame(rows)

    csv_path = out_dir / "ragas_summary_table.csv"
    json_path = out_dir / "ragas_summary_table.json"
    tex_path = out_dir / "ragas_summary_table.tex"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(json.loads(df.to_json(orient="records")), indent=2), encoding="utf-8")
    tex_path.write_text(
        _latex_table(df[["run", "scored_rows", "answer_relevancy", "faithfulness", "context_precision", "context_recall"]]),
        encoding="utf-8",
    )
    write_scope_manifest(
        out_dir=out_dir,
        scope_name="ragas",
        source_inputs={
            "runs": [
                {"label": label, "run_dir": str(run_dir)}
                for label, run_dir in (_parse_run_arg(v) for v in args.run)
            ]
        },
        exported_files=[csv_path, json_path, tex_path],
        notes=["RAGAS thesis exports derived from completed ragas_summary.json runs."],
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
