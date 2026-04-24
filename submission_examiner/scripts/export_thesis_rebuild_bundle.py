"""Orchestrate all thesis freeze/export helpers to populate a frozen rebuild bundle.

Calls export_thesis_chunk_ablation_table.py, export_thesis_failure_analysis_bundle.py,
export_thesis_bootstrap_table.py, export_thesis_mcnemar_table.py, and
export_thesis_ragas_table.py in sequence, writing all outputs into the specified bundle
directory. Optionally compares against an earlier bundle for drift detection, then runs
backfill_thesis_export_manifests.py and check_thesis_export_provenance.py to verify
provenance. Use this as the single entry point when rebuilding the full thesis export.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run all thesis freeze/export helpers into one rebuild bundle."
    )
    p.add_argument(
        "--bundle-dir",
        required=True,
        help="Frozen rebuild bundle directory under results/thesis_rebuild_freeze/<run-name>.",
    )
    p.add_argument(
        "--chunk-data-root",
        default="",
        help="Per-experiment data root for chunk ablation exports.",
    )
    p.add_argument("--failure-baseline-dir", default="", help="Baseline FP1-FP7 directory.")
    p.add_argument("--failure-candidate-dir", default="", help="Candidate FP1-FP7 directory, e.g. LLM-on.")
    p.add_argument("--failure-comparison-dir", default="", help="Comparison directory from compare_fp1_fp7_runs.py.")
    p.add_argument("--bootstrap-input-dir", default="", help="Root containing paired bootstrap summaries.")
    p.add_argument("--mcnemar-batch-summary-csv", default="", help="Batch McNemar summary CSV.")
    p.add_argument(
        "--ragas-run",
        action="append",
        default=[],
        help="RAGAS run mapping in the form label::/path/to/run_dir. Can be passed multiple times.",
    )
    p.add_argument(
        "--compare-against-bundle",
        default="",
        help="Optional earlier frozen bundle. If provided, writes a drift report after exporting.",
    )
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def main() -> None:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir).resolve()
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle dir does not exist: {bundle_dir}")

    python_bin = sys.executable

    if str(args.chunk_data_root).strip():
        _run(
            [
                python_bin,
                "scripts/export_thesis_chunk_ablation_table.py",
                "--data-root",
                str(Path(args.chunk_data_root).resolve()),
                "--out-dir",
                str(bundle_dir / "tables"),
            ]
        )

    if str(args.failure_baseline_dir).strip():
        cmd = [
            python_bin,
            "scripts/export_thesis_failure_analysis_bundle.py",
            "--baseline-dir",
            str(Path(args.failure_baseline_dir).resolve()),
            "--out-dir",
            str(bundle_dir / "failure_analysis"),
        ]
        if str(args.failure_candidate_dir).strip():
            cmd.extend(["--candidate-dir", str(Path(args.failure_candidate_dir).resolve())])
        if str(args.failure_comparison_dir).strip():
            cmd.extend(["--comparison-dir", str(Path(args.failure_comparison_dir).resolve())])
        _run(cmd)

    if str(args.bootstrap_input_dir).strip():
        _run(
            [
                python_bin,
                "scripts/export_thesis_bootstrap_table.py",
                "--input-dir",
                str(Path(args.bootstrap_input_dir).resolve()),
                "--out-dir",
                str(bundle_dir / "bootstrap"),
            ]
        )

    if str(args.mcnemar_batch_summary_csv).strip():
        _run(
            [
                python_bin,
                "scripts/export_thesis_mcnemar_table.py",
                "--batch-summary-csv",
                str(Path(args.mcnemar_batch_summary_csv).resolve()),
                "--out-dir",
                str(bundle_dir / "mcnemar"),
            ]
        )

    if args.ragas_run:
        cmd = [
            python_bin,
            "scripts/export_thesis_ragas_table.py",
            "--out-dir",
            str(bundle_dir / "ragas"),
        ]
        for value in args.ragas_run:
            cmd.extend(["--run", value])
        _run(cmd)

    if str(args.compare_against_bundle).strip():
        _run(
            [
                python_bin,
                "scripts/check_thesis_bundle_drift.py",
                "--bundle-dir",
                str(bundle_dir),
                "--baseline-bundle",
                str(Path(args.compare_against_bundle).resolve()),
                "--out-dir",
                str(bundle_dir / "guardrails"),
            ]
        )

    _run(
        [
            python_bin,
            "scripts/backfill_thesis_export_manifests.py",
            "--bundle-dir",
            str(bundle_dir),
        ]
    )

    _run(
        [
            python_bin,
            "scripts/check_thesis_export_provenance.py",
            "--bundle-dir",
            str(bundle_dir),
            "--out-dir",
            str(bundle_dir / "guardrails"),
        ]
    )

    print("")
    print(f"Bundle exports updated under: {bundle_dir}")


if __name__ == "__main__":
    main()
