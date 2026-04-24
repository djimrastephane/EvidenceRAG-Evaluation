"""Validate the Python environment against the pinned thesis pipeline dependencies.

Checks that all critical packages (faiss, sentence-transformers, camelot, tiktoken, etc.)
are installed, reports installed versions against requirements.txt, and optionally emits
the full environment report as JSON. Use --strict to fail non-zero on any mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from runtime_env import (
    collect_runtime_provenance,
    critical_environment_checks,
    pinned_requirements_status,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight environment validation for the final thesis pipeline."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full environment report as JSON.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any critical dependency check fails.",
    )
    return parser.parse_args()


def main() -> int:
    """Print the environment preflight report; return 1 under --strict if any check fails."""
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    report = collect_runtime_provenance()
    checks = critical_environment_checks()
    pinned = pinned_requirements_status(repo_root / "requirements.txt")
    report["critical_checks"] = checks
    report["requirements_alignment"] = pinned
    report["recommended_env_file"] = str(repo_root / "environment.yml")

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("Environment preflight")
        print(f"  Python: {report['python_executable']} ({report['python_version']})")
        print(f"  Conda env: {report.get('conda_default_env') or '-'}")
        print(f"  Virtual env: {report.get('virtual_env') or '-'}")
        print(f"  Recommended environment file: {report['recommended_env_file']}")
        print("")
        print("Critical checks")
        for check in checks:
            status = "OK" if check["ok"] else "FAIL"
            print(f"  [{status}] {check['name']}: {check['detail']}")
        print("")
        print("Pinned requirements alignment")
        for row in pinned:
            status = "OK" if row["matches"] else "DRIFT"
            installed_version = row["installed_version"] or "-"
            print(
                f"  [{status}] {row['distribution']}: expected {row['expected_version']}, "
                f"installed {installed_version}"
            )
        print("")
        print("Detected package versions")
        for name, info in sorted(report["dependency_report"]["modules"].items()):
            version = info["version"] or "-"
            status = "installed" if info["installed"] else "missing"
            print(f"  {name}: {status} ({version})")
        print("")
        print("Detected binaries")
        for name, info in sorted(report["dependency_report"]["commands"].items()):
            path = info["path"] or "-"
            status = "available" if info["available"] else "missing"
            print(f"  {name}: {status} ({path})")

    has_failures = any(not check["ok"] for check in checks)
    has_drift = any(not row["matches"] for row in pinned)
    if args.strict and (has_failures or has_drift):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
