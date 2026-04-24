"""Verify that all examiner-facing files, directories, and environment checks are in place.

Run this as the first step in the examiner quickstart. It prints a summary of which
required files exist, whether critical dependencies are installed, and whether pinned
package versions match requirements.txt. Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from runtime_env import critical_environment_checks, pinned_requirements_status


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_DIR = (
    REPO_ROOT
    / "results"
    / "thesis_rebuild_freeze"
    / "thesis_rebuild_freeze_smoke_2026-03-18"
)
DEFAULT_EVAL_DOCS = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that the examiner-facing verification path is present and ready to run."
    )
    parser.add_argument(
        "--bundle-dir",
        default=str(DEFAULT_BUNDLE_DIR),
        help="Frozen bundle directory expected to back thesis-facing outputs.",
    )
    parser.add_argument(
        "--data-root",
        default="data_processed",
        help="Processed corpus root expected to contain the five eval-set documents.",
    )
    return parser.parse_args()


def _status(ok: bool) -> str:
    return "OK" if ok else "FAIL"


def _check(path: Path, *, kind: str = "file") -> tuple[bool, str]:
    exists = path.is_dir() if kind == "dir" else path.is_file()
    return exists, f"{kind}: {path}"


def main() -> int:
    """Check all required files, environment, and pinned packages; print a summary table.

    Returns 0 if every check passes, 1 if any required file is missing, any critical
    dependency is absent, or any pinned package version does not match.
    """
    args = parse_args()
    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    data_root = (REPO_ROOT / args.data_root).resolve()

    checks: list[tuple[bool, str]] = []
    checks.append(_check(REPO_ROOT / "environment.yml"))
    checks.append(_check(REPO_ROOT / "requirements.txt"))
    checks.append(_check(REPO_ROOT / "docs" / "EXAMINER_QUICKSTART.md"))
    checks.append(_check(REPO_ROOT / "docs" / "EXAMINER_SUBMISSION_MANIFEST.md"))
    checks.append(_check(REPO_ROOT / "scripts" / "check_examiner_path.py"))
    checks.append(_check(REPO_ROOT / "scripts" / "check_environment.py"))
    checks.append(_check(REPO_ROOT / "scripts" / "check_pipeline_reproducibility.py"))
    checks.append(_check(bundle_dir, kind="dir"))
    checks.append(_check(bundle_dir / "RUNBOOK.md"))
    checks.append(_check(bundle_dir / "manifests" / "environment_manifest.json"))
    checks.append(_check(bundle_dir / "tables" / "chunk_ablation_table.csv"))
    checks.append(_check(REPO_ROOT / "results" / "reproducibility" / "grampian_5docs_repro.json"))

    for doc_id in DEFAULT_EVAL_DOCS:
        checks.append(_check(data_root / doc_id / "eval_set.json"))

    env_checks = critical_environment_checks()
    pinned = pinned_requirements_status(REPO_ROOT / "requirements.txt")
    critical_ok = all(check["ok"] for check in env_checks)
    pinned_ok = all(row["matches"] for row in pinned)

    print("Examiner path check")
    print(f"  Repo root: {REPO_ROOT}")
    print(f"  Bundle dir: {bundle_dir}")
    print(f"  Data root: {data_root}")
    print("")
    print("Required files and directories")
    for ok, label in checks:
        print(f"  [{_status(ok)}] {label}")
    print("")
    print("Environment summary")
    print(f"  [{_status(critical_ok)}] critical dependencies")
    print(f"  [{_status(pinned_ok)}] pinned requirements alignment")
    print("")
    print("Recommended examiner commands")
    print("  1. conda activate rag-pipeline")
    print("  2. python scripts/check_examiner_path.py")
    print("  3. python scripts/check_environment.py --strict")
    print("  4. python scripts/check_pipeline_reproducibility.py --runs 2 --out-json results/reproducibility/examiner_repro_check.json")
    print("  5. Optional demo API: bash scripts/run_api_demo.sh")
    print("  6. Optional demo UI: bash scripts/run_streamlit_demo.sh")

    all_file_checks_ok = all(ok for ok, _ in checks)
    return 0 if (all_file_checks_ok and critical_ok and pinned_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
