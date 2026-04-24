"""Backfill scope manifests for an existing frozen thesis bundle that was built without them.

Scans each scope directory (tables, failure_analysis, bootstrap, mcnemar, ragas) inside the
given bundle and writes a manifest.json listing all tracked output files. Intended to be run
once on older bundles that predate the provenance workflow, or after adding files to an
existing scope. Has no effect on scopes that already have a manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from thesis_provenance import write_scope_manifest


DEFAULT_SCOPES = ["tables", "failure_analysis", "bootstrap", "mcnemar", "ragas"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backfill scope manifests for an existing frozen thesis bundle."
    )
    p.add_argument("--bundle-dir", required=True, help="Frozen thesis bundle directory.")
    p.add_argument(
        "--scope",
        action="append",
        default=[],
        help="Optional scope to backfill. Defaults to tables/failure_analysis/bootstrap/mcnemar/ragas.",
    )
    return p.parse_args()


def _runbook_processed_corpora_dir(bundle_dir: Path) -> str | None:
    runbook = bundle_dir / "RUNBOOK.md"
    if not runbook.exists():
        return None
    for line in runbook.read_text(encoding="utf-8").splitlines():
        if line.startswith("- processed corpora dir:"):
            parts = line.split("`")
            if len(parts) >= 2:
                return parts[1]
    return None


def _tracked_scope_files(scope_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(scope_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name in {".DS_Store", "manifest.json"}:
            continue
        if path.suffix.lower() not in {".csv", ".json", ".tex", ".png", ".md"}:
            continue
        files.append(path.resolve())
    return files


def _backfill_tables(bundle_dir: Path) -> None:
    out_dir = bundle_dir / "tables"
    write_scope_manifest(
        out_dir=out_dir,
        scope_name="tables",
        source_inputs={"data_root": _runbook_processed_corpora_dir(bundle_dir)},
        exported_files=_tracked_scope_files(out_dir),
        notes=["Backfilled from existing frozen bundle outputs."],
    )


def _backfill_failure_analysis(bundle_dir: Path) -> None:
    out_dir = bundle_dir / "failure_analysis"
    existing_manifest = out_dir / "manifest.json"
    source_inputs = {}
    if existing_manifest.exists():
        payload = json.loads(existing_manifest.read_text(encoding="utf-8"))
        source_inputs = {
            "baseline_dir": payload.get("baseline_dir"),
            "candidate_dir": payload.get("candidate_dir"),
            "comparison_dir": payload.get("comparison_dir"),
            "runs_exported": payload.get("runs_exported"),
        }
    write_scope_manifest(
        out_dir=out_dir,
        scope_name="failure_analysis",
        source_inputs=source_inputs,
        exported_files=_tracked_scope_files(out_dir),
        notes=["Backfilled from existing frozen bundle outputs."],
    )


def _backfill_bootstrap(bundle_dir: Path) -> None:
    out_dir = bundle_dir / "bootstrap"
    csv_path = out_dir / "paired_bootstrap_summary_table.csv"
    input_dir = None
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        source_paths = [Path(row["source_summary"]).resolve() for row in rows if row.get("source_summary")]
        if source_paths:
            input_dir = str(source_paths[0].parent.parent)
    write_scope_manifest(
        out_dir=out_dir,
        scope_name="bootstrap",
        source_inputs={"input_dir": input_dir},
        exported_files=_tracked_scope_files(out_dir),
        notes=["Backfilled from existing frozen bundle outputs."],
    )


def _backfill_mcnemar(bundle_dir: Path) -> None:
    out_dir = bundle_dir / "mcnemar"
    csv_path = out_dir / "mcnemar_hit1_summary_table.csv"
    batch_summary_csv = None
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        sample = next((row for row in rows if row.get("json_result_path")), None)
        if sample:
            sample_path = Path(sample["json_result_path"]).resolve()
            candidate = sample_path.parent / "mcnemar_hit1_batch_summary.csv"
            if candidate.exists():
                batch_summary_csv = str(candidate)
    write_scope_manifest(
        out_dir=out_dir,
        scope_name="mcnemar",
        source_inputs={"batch_summary_csv": batch_summary_csv},
        exported_files=_tracked_scope_files(out_dir),
        notes=["Backfilled from existing frozen bundle outputs."],
    )


def _backfill_ragas(bundle_dir: Path) -> None:
    out_dir = bundle_dir / "ragas"
    json_path = out_dir / "ragas_summary_table.json"
    runs: list[dict[str, str]] = []
    if json_path.exists():
        rows = json.loads(json_path.read_text(encoding="utf-8"))
        for row in rows:
            if not isinstance(row, dict):
                continue
            runs.append({"label": str(row.get("run") or ""), "run_dir": str(row.get("source_dir") or "")})
    write_scope_manifest(
        out_dir=out_dir,
        scope_name="ragas",
        source_inputs={"runs": runs},
        exported_files=_tracked_scope_files(out_dir),
        notes=["Backfilled from existing frozen bundle outputs."],
    )


def main() -> None:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir).resolve()
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle dir not found: {bundle_dir}")

    scopes = list(args.scope) if args.scope else list(DEFAULT_SCOPES)
    handlers = {
        "tables": _backfill_tables,
        "failure_analysis": _backfill_failure_analysis,
        "bootstrap": _backfill_bootstrap,
        "mcnemar": _backfill_mcnemar,
        "ragas": _backfill_ragas,
    }
    for scope in scopes:
        scope_dir = bundle_dir / scope
        if not scope_dir.exists():
            continue
        handler = handlers.get(scope)
        if handler is None:
            continue
        handler(bundle_dir)
        print(f"Backfilled manifest for {scope_dir}")


if __name__ == "__main__":
    main()
