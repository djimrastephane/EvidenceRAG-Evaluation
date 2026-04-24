from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def infer_bundle_dir(out_dir: Path) -> Path | None:
    out_dir = out_dir.resolve()
    for candidate in [out_dir, *out_dir.parents]:
        if (candidate / "RUNBOOK.md").exists() and (candidate / "manifests" / "environment_manifest.json").exists():
            return candidate
    return None


def write_scope_manifest(
    *,
    out_dir: Path,
    scope_name: str,
    source_inputs: dict[str, Any],
    exported_files: list[Path],
    notes: list[str] | None = None,
) -> Path:
    out_dir = out_dir.resolve()
    bundle_dir = infer_bundle_dir(out_dir)
    payload = {
        "scope_name": scope_name,
        "bundle_dir": str(bundle_dir) if bundle_dir else None,
        "scope_dir": str(out_dir),
        "source_inputs": source_inputs,
        "exported_files": [str(path.resolve()) for path in exported_files],
        "notes": list(notes or []),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path
