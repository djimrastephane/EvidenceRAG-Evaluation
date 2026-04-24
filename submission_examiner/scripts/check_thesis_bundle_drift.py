"""Compare exported thesis bundle outputs against a baseline bundle and report any drift.

Hashes every tracked output file (CSV, JSON, TeX, PNG, MD) within the specified scope
directories and compares against a baseline bundle. Reports files that are changed, added,
or removed. Writes a drift report (CSV, JSON, Markdown) to <bundle>/guardrails/. Exits
cleanly even when drift is detected; use the report to decide whether changes are intended.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


DEFAULT_SCOPES = ["tables", "failure_analysis", "bootstrap", "mcnemar", "ragas"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare exported thesis bundle outputs against a baseline bundle and report drift."
    )
    p.add_argument("--bundle-dir", required=True, help="Current frozen bundle directory.")
    p.add_argument("--baseline-bundle", required=True, help="Earlier frozen bundle directory to compare against.")
    p.add_argument("--out-dir", default="", help="Optional output dir for guardrail reports. Defaults to <bundle>/guardrails.")
    p.add_argument(
        "--scope",
        action="append",
        default=[],
        help="Optional top-level subdirectory to compare. Can be repeated. Defaults to tables/failure_analysis/bootstrap/mcnemar/ragas.",
    )
    return p.parse_args()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _files_under(root: Path, scopes: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for scope in scopes:
        scoped = root / scope
        if not scoped.exists():
            continue
        for path in scoped.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".csv", ".json", ".tex", ".png", ".md"}:
                continue
            rel = str(path.relative_to(root))
            out[rel] = path
    return out


def main() -> None:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir).resolve()
    baseline_dir = Path(args.baseline_bundle).resolve()
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle dir not found: {bundle_dir}")
    if not baseline_dir.exists():
        raise FileNotFoundError(f"Baseline bundle not found: {baseline_dir}")

    scopes = list(args.scope) if args.scope else list(DEFAULT_SCOPES)
    out_dir = Path(args.out_dir).resolve() if str(args.out_dir).strip() else bundle_dir / "guardrails"
    out_dir.mkdir(parents=True, exist_ok=True)

    current_files = _files_under(bundle_dir, scopes=scopes)
    baseline_files = _files_under(baseline_dir, scopes=scopes)
    all_rel_paths = sorted(set(current_files) | set(baseline_files))

    rows: list[dict[str, object]] = []
    changed = 0
    added = 0
    removed = 0
    unchanged = 0

    for rel in all_rel_paths:
        current = current_files.get(rel)
        baseline = baseline_files.get(rel)
        if current is None:
            status = "removed"
            baseline_hash = _sha256(baseline)
            current_hash = None
            removed += 1
        elif baseline is None:
            status = "added"
            baseline_hash = None
            current_hash = _sha256(current)
            added += 1
        else:
            current_hash = _sha256(current)
            baseline_hash = _sha256(baseline)
            if current_hash == baseline_hash:
                status = "unchanged"
                unchanged += 1
            else:
                status = "changed"
                changed += 1

        rows.append(
            {
                "relative_path": rel,
                "status": status,
                "current_file": str(current) if current else "",
                "baseline_file": str(baseline) if baseline else "",
                "current_sha256": current_hash or "",
                "baseline_sha256": baseline_hash or "",
            }
        )

    csv_path = out_dir / "bundle_drift_report.csv"
    json_path = out_dir / "bundle_drift_report.json"
    md_path = out_dir / "bundle_drift_report.md"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "relative_path",
                "status",
                "current_file",
                "baseline_file",
                "current_sha256",
                "baseline_sha256",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    payload = {
        "bundle_dir": str(bundle_dir),
        "baseline_bundle": str(baseline_dir),
        "scopes": scopes,
        "summary": {
            "changed": changed,
            "added": added,
            "removed": removed,
            "unchanged": unchanged,
            "total_compared": len(rows),
        },
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        "# Thesis Bundle Drift Report",
        "",
        f"- bundle_dir: `{bundle_dir}`",
        f"- baseline_bundle: `{baseline_dir}`",
        f"- scopes: `{', '.join(scopes)}`",
        "",
        "## Summary",
        "",
        f"- changed: `{changed}`",
        f"- added: `{added}`",
        f"- removed: `{removed}`",
        f"- unchanged: `{unchanged}`",
        f"- total compared: `{len(rows)}`",
        "",
    ]
    if changed or added or removed:
        md_lines.append("## Non-identical outputs")
        md_lines.append("")
        for row in rows:
            if row["status"] == "unchanged":
                continue
            md_lines.append(f"- `{row['status']}`: `{row['relative_path']}`")
        md_lines.append("")
    else:
        md_lines.append("All scoped outputs matched exactly.")
        md_lines.append("")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    if changed or added or removed:
        print(f"Drift detected: changed={changed}, added={added}, removed={removed}")
    else:
        print("No drift detected.")


if __name__ == "__main__":
    main()
