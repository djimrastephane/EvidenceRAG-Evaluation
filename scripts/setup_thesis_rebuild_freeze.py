"""Initialise a new frozen thesis rebuild directory with a locked config and environment manifest.

Creates the output bundle directory, copies and freezes the specified base retrieval config,
records a full runtime provenance snapshot (Python version, conda env, installed packages,
pinned requirements alignment, git commit), and writes a RUNBOOK.md explaining how to
reproduce the thesis-facing results from that bundle. This is the first step in the
controlled thesis export workflow.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root / "scripts") not in sys.path:
    sys.path.insert(0, str(repo_root / "scripts"))

from runtime_env import collect_runtime_provenance, critical_environment_checks, pinned_requirements_status


DEFAULT_ENV_LOCKS = {
    "REQUIRE_TIKTOKEN": "1",
    "ENABLE_LEXICAL_RERANK": "1",
    "ENABLE_SUBSECTION_BOOST": "1",
    "TABLE_CHUNK_BOOST": "0.08",
    "MILESTONE_TEXT_BOOST": "0.08",
    "ENTITY_MATCH_BOOST": "0.04",
    "NUMERIC_DENSITY_BOOST": "0.03",
    "SUBSECTION_BOOST": "0.05",
    "MAX_K_SEARCH": "100",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an isolated, reproducible thesis rebuild root with frozen config and manifest."
    )
    parser.add_argument(
        "--run-name",
        default=f"thesis_rebuild_freeze_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        help="Name for the frozen rebuild root.",
    )
    parser.add_argument(
        "--base-config",
        default="configs/retrieval_tuning_minilm_cap_5docs.yaml",
        help="Tracked YAML config to freeze into the run bundle.",
    )
    parser.add_argument(
        "--results-root",
        default="results/thesis_rebuild_freeze",
        help="Parent directory for frozen result bundles.",
    )
    parser.add_argument(
        "--data-root",
        default="data_variants/thesis_rebuild_freeze",
        help="Parent directory for frozen processed corpora.",
    )
    return parser.parse_args()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected mapping at YAML root: {path}")
    return obj


def _write_yaml(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=False)


def _make_runbook_text(
    run_name: str,
    frozen_config_path: Path,
    results_dir: Path,
    data_dir: Path,
) -> str:
    return "\n".join(
        [
            f"# Thesis Rebuild Freeze: {run_name}",
            "",
            "This bundle is the source of truth for the clean thesis rebuild.",
            "",
            "Rules:",
            "- Do not overwrite outputs from another run.",
            "- Generate thesis-facing tables/figures only from this bundle.",
            "- Keep the frozen config and manifest alongside the outputs.",
            "",
            "Important paths:",
            f"- frozen config: `{frozen_config_path}`",
            f"- results dir: `{results_dir}`",
            f"- processed corpora dir: `{data_dir}`",
            "",
            "Suggested execution order:",
            f"1. `python scripts/run_retrieval_ablation.py --config {frozen_config_path}`",
            (
                "2. `python scripts/export_thesis_chunk_ablation_table.py "
                f"--data-root {data_dir} --out-dir {results_dir / 'tables'}`"
            ),
            "3. Regenerate thesis tables/figures from the exported CSV/JSON/TeX only.",
            "",
            "If additional thesis experiments are added later, freeze each source config into this run bundle first.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()

    base_config_path = (repo_root / args.base_config).resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    run_name = str(args.run_name).strip()
    if not run_name:
        raise ValueError("run-name must be non-empty")

    results_root = (repo_root / args.results_root).resolve()
    data_root = (repo_root / args.data_root).resolve()
    run_results_dir = results_root / run_name
    run_data_dir = data_root / run_name

    for rel in ["configs", "manifests", "tables", "figures", "logs", "notes"]:
        (run_results_dir / rel).mkdir(parents=True, exist_ok=True)
    run_data_dir.mkdir(parents=True, exist_ok=True)

    frozen_cfg = _load_yaml(base_config_path)
    config_stem = base_config_path.stem
    frozen_cfg["python_bin"] = sys.executable
    frozen_cfg["ablation_root"] = str(run_data_dir / config_stem)
    frozen_cfg["output_dir"] = str(run_results_dir / config_stem)

    frozen_config_path = run_results_dir / "configs" / f"{config_stem}_frozen.yaml"
    _write_yaml(frozen_config_path, frozen_cfg)

    manifest = {
        "run_name": run_name,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "repo_root": str(repo_root),
        "git_commit": _git_commit(),
        "base_config_path": str(base_config_path),
        "frozen_config_path": str(frozen_config_path),
        "results_dir": str(run_results_dir),
        "data_dir": str(run_data_dir),
        "runtime_provenance": collect_runtime_provenance(),
        "critical_environment_checks": critical_environment_checks(),
        "requirements_txt_status": pinned_requirements_status(repo_root / "requirements.txt"),
        "env_locks": {key: os.getenv(key, default) for key, default in DEFAULT_ENV_LOCKS.items()},
    }
    manifest_path = run_results_dir / "manifests" / "environment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    runbook_path = run_results_dir / "RUNBOOK.md"
    runbook_path.write_text(
        _make_runbook_text(
            run_name=run_name,
            frozen_config_path=frozen_config_path,
            results_dir=run_results_dir,
            data_dir=run_data_dir / config_stem,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {frozen_config_path}")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {runbook_path}")
    print("")
    print("Next step:")
    print(f"python scripts/run_retrieval_ablation.py --config {frozen_config_path}")


if __name__ == "__main__":
    main()
