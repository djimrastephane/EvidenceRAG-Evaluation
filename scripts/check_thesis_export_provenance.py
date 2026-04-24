"""Check that thesis-exported tables and figures in a frozen bundle are tied to scope manifests.

For each scope directory (tables, failure_analysis, bootstrap, mcnemar, ragas) inside the
frozen bundle, verifies that a manifest.json exists and that every tracked output file is
listed in it. Writes a provenance report (CSV, JSON, Markdown) to <bundle>/guardrails/.
Exits non-zero if any orphan files or missing manifests are found.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_SCOPES = ["tables", "failure_analysis", "bootstrap", "mcnemar", "ragas"]
TRACKED_SUFFIXES = {".csv", ".json", ".tex", ".png", ".md"}
MANIFEST_NAME = "manifest.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check that thesis-exported tables and figures in a frozen bundle are tied to scope manifests."
    )
    p.add_argument("--bundle-dir", required=True, help="Frozen thesis bundle directory.")
    p.add_argument(
        "--scope",
        action="append",
        default=[],
        help="Optional top-level scope to validate. Defaults to tables/failure_analysis/bootstrap/mcnemar/ragas.",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="Optional output dir for provenance reports. Defaults to <bundle>/guardrails.",
    )
    return p.parse_args()


def _load_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tracked_files(scope_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(scope_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == ".DS_Store":
            continue
        if path.suffix.lower() not in TRACKED_SUFFIXES:
            continue
        files.append(path.resolve())
    return files


def main() -> None:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir).resolve()
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle dir not found: {bundle_dir}")

    scopes = list(args.scope) if args.scope else list(DEFAULT_SCOPES)
    out_dir = Path(args.out_dir).resolve() if str(args.out_dir).strip() else bundle_dir / "guardrails"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    failures = 0

    for scope in scopes:
        scope_dir = bundle_dir / scope
        if not scope_dir.exists():
            rows.append(
                {
                    "scope": scope,
                    "status": "missing_scope",
                    "relative_path": scope,
                    "detail": "scope directory does not exist",
                }
            )
            failures += 1
            continue

        manifest_path = scope_dir / MANIFEST_NAME
        if not manifest_path.exists():
            rows.append(
                {
                    "scope": scope,
                    "status": "missing_manifest",
                    "relative_path": str(manifest_path.relative_to(bundle_dir)),
                    "detail": "scope manifest is missing",
                }
            )
            failures += 1
            continue

        manifest = _load_manifest(manifest_path)
        manifest_bundle = str(manifest.get("bundle_dir") or "")
        if manifest_bundle != str(bundle_dir):
            rows.append(
                {
                    "scope": scope,
                    "status": "bundle_mismatch",
                    "relative_path": str(manifest_path.relative_to(bundle_dir)),
                    "detail": f"manifest bundle_dir={manifest_bundle!r}",
                }
            )
            failures += 1

        exported_files = {str(Path(p).resolve()) for p in list(manifest.get("exported_files") or [])}
        tracked_files = _tracked_files(scope_dir)
        for path in tracked_files:
            rel = str(path.relative_to(bundle_dir))
            if path == manifest_path.resolve():
                rows.append(
                    {
                        "scope": scope,
                        "status": "ok_manifest",
                        "relative_path": rel,
                        "detail": "manifest present",
                    }
                )
                continue
            if str(path) not in exported_files:
                rows.append(
                    {
                        "scope": scope,
                        "status": "orphan_file",
                        "relative_path": rel,
                        "detail": "tracked export file is not listed in manifest",
                    }
                )
                failures += 1
                continue
            rows.append(
                {
                    "scope": scope,
                    "status": "ok",
                    "relative_path": rel,
                    "detail": "listed in manifest",
                }
            )

        for exported in sorted(exported_files):
            exported_path = Path(exported)
            if not exported_path.exists():
                rows.append(
                    {
                        "scope": scope,
                        "status": "missing_export",
                        "relative_path": exported,
                        "detail": "manifest lists a file that does not exist",
                    }
                )
                failures += 1

    csv_path = out_dir / "bundle_provenance_report.csv"
    json_path = out_dir / "bundle_provenance_report.json"
    md_path = out_dir / "bundle_provenance_report.md"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scope", "status", "relative_path", "detail"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    payload = {
        "bundle_dir": str(bundle_dir),
        "scopes": scopes,
        "status": "pass" if failures == 0 else "fail",
        "failure_count": failures,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        "# Thesis Bundle Provenance Report",
        "",
        f"- bundle_dir: `{bundle_dir}`",
        f"- scopes: `{', '.join(scopes)}`",
        f"- status: `{'pass' if failures == 0 else 'fail'}`",
        f"- failure_count: `{failures}`",
        "",
    ]
    bad_rows = [row for row in rows if not row["status"].startswith("ok")]
    if bad_rows:
        md_lines.append("## Issues")
        md_lines.append("")
        for row in bad_rows:
            md_lines.append(f"- `{row['status']}`: `{row['relative_path']}` ({row['detail']})")
        md_lines.append("")
    else:
        md_lines.append("All tracked thesis-exported tables and figures are tied to scope manifests in this bundle.")
        md_lines.append("")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
