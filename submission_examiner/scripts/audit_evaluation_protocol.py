"""Audit the thesis evaluation protocol from tracked configs and frozen bundle metadata.

Reads the three tracked retrieval configs (tuning, final, sanity) and the frozen bundle
environment manifest, then produces a JSON and Markdown report summarising the evaluation
scope, document overlap between configs, and any protocol conflicts (e.g. shared tuning
and reporting corpus, environment drift). Intended to support the methodology section and
to be regenerated before submission to catch obvious issues.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the thesis evaluation protocol from tracked configs and frozen bundle metadata."
    )
    parser.add_argument(
        "--tuning-config",
        default="configs/retrieval_tuning_thesis_5docs_q50.yaml",
        help="Tracked tuning/ablation config used to explore settings.",
    )
    parser.add_argument(
        "--final-config",
        default="configs/retrieval_tuning_minilm_cap_5docs.yaml",
        help="Tracked config treated as the final thesis-facing selection.",
    )
    parser.add_argument(
        "--sanity-config",
        default="configs/retrieval_tuning_224_56_5docs.yaml",
        help="Tracked sanity config for the promoted 224/56 comparison run.",
    )
    parser.add_argument(
        "--bundle-dir",
        default="results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18",
        help="Frozen thesis bundle directory.",
    )
    parser.add_argument(
        "--out-json",
        default="results/reproducibility/evaluation_protocol_audit.json",
        help="JSON report output path.",
    )
    parser.add_argument(
        "--out-md",
        default="results/reproducibility/evaluation_protocol_audit.md",
        help="Markdown report output path.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected mapping YAML: {path}")
    return obj


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _config_summary(path: Path) -> dict[str, Any]:
    cfg = _load_yaml(path)
    experiments = list(cfg.get("experiments") or [])
    docs = sorted({str(exp.get("doc_id") or "").strip() for exp in experiments if str(exp.get("doc_id") or "").strip()})
    source_eval_sets = sorted(
        {str(exp.get("source_eval_set") or "").strip() for exp in experiments if str(exp.get("source_eval_set") or "").strip()}
    )
    modes = sorted({str(exp.get("mode") or "").strip() for exp in experiments if str(exp.get("mode") or "").strip()})
    chunking_signatures = sorted(
        {
            json.dumps(exp.get("chunking") or {}, sort_keys=True)
            for exp in experiments
            if isinstance(exp.get("chunking"), dict)
        }
    )
    rerank_enabled_values = sorted(
        {
            bool((exp.get("rerank") or {}).get("enabled"))
            for exp in experiments
            if isinstance(exp.get("rerank"), dict)
        }
    )
    return {
        "path": str(path.resolve()),
        "experiment_count": len(experiments),
        "doc_ids": docs,
        "source_eval_sets": source_eval_sets,
        "modes": modes,
        "chunking_variants": len(chunking_signatures),
        "rerank_enabled_values": rerank_enabled_values,
        "output_dir": str(cfg.get("output_dir") or ""),
        "ablation_root": str(cfg.get("ablation_root") or ""),
    }


def _bundle_summary(bundle_dir: Path) -> dict[str, Any]:
    env_manifest_path = bundle_dir / "manifests" / "environment_manifest.json"
    env_manifest = _load_json(env_manifest_path)
    frozen_config_path = Path(str(env_manifest.get("frozen_config_path") or "")).resolve()
    frozen_config_exists = frozen_config_path.exists()
    requirements_alignment = list(env_manifest.get("requirements_txt_status") or [])
    requirements_mismatches = [
        row for row in requirements_alignment if isinstance(row, dict) and not bool(row.get("matches"))
    ]
    return {
        "bundle_dir": str(bundle_dir.resolve()),
        "run_name": str(env_manifest.get("run_name") or ""),
        "git_commit": str(env_manifest.get("git_commit") or ""),
        "base_config_path": str(env_manifest.get("base_config_path") or ""),
        "frozen_config_path": str(frozen_config_path) if frozen_config_exists else str(env_manifest.get("frozen_config_path") or ""),
        "frozen_config_exists": bool(frozen_config_exists),
        "created_utc": str(env_manifest.get("created_utc") or ""),
        "conda_default_env": str((env_manifest.get("runtime_provenance") or {}).get("conda_default_env") or ""),
        "python_version": str((env_manifest.get("runtime_provenance") or {}).get("python_version") or ""),
        "requirements_mismatch_count": len(requirements_mismatches),
        "requirements_mismatches": requirements_mismatches,
        "env_locks": env_manifest.get("env_locks") or {},
    }


def _overlap_summary(*doc_lists: list[str]) -> list[str]:
    if not doc_lists:
        return []
    overlap = set(doc_lists[0])
    for docs in doc_lists[1:]:
        overlap &= set(docs)
    return sorted(overlap)


def _findings(
    *,
    tuning: dict[str, Any],
    final: dict[str, Any],
    sanity: dict[str, Any],
    bundle: dict[str, Any],
) -> list[str]:
    findings: list[str] = []
    overlap_tuning_final = _overlap_summary(tuning["doc_ids"], final["doc_ids"])
    if overlap_tuning_final:
        findings.append(
            "Tuning and final configs reuse the same document set: "
            + ", ".join(overlap_tuning_final)
            + ". This is acceptable only if the thesis states that selection and reporting share the same 5-doc corpus."
        )
    if bundle["requirements_mismatch_count"]:
        findings.append(
            f"Frozen bundle environment manifest records {bundle['requirements_mismatch_count']} pinned-package mismatches. "
            "This weakens the bundle as a final canonical environment record unless a cleaner rerun replaces it."
        )
    if str(bundle.get("conda_default_env") or "") == "base":
        findings.append(
            "Frozen bundle environment manifest was created from Conda 'base', not 'rag-pipeline'. "
            "That should be corrected before treating the bundle as the final thesis source of truth."
        )
    if final["chunking_variants"] > 1:
        findings.append(
            "Final config still contains multiple chunking variants, so it is an ablation/tuning-style config rather than a single locked final run config."
        )
    if not findings:
        findings.append("No obvious protocol conflicts detected from the tracked configs and frozen bundle metadata.")
    return findings


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Evaluation Protocol Audit",
        "",
        "## Scope",
        "",
        f"- tuning config: `{payload['tuning_config']['path']}`",
        f"- final config: `{payload['final_config']['path']}`",
        f"- sanity config: `{payload['sanity_config']['path']}`",
        f"- frozen bundle: `{payload['frozen_bundle']['bundle_dir']}`",
        "",
        "## Summary",
        "",
        f"- tuning docs: `{', '.join(payload['tuning_config']['doc_ids'])}`",
        f"- final docs: `{', '.join(payload['final_config']['doc_ids'])}`",
        f"- sanity docs: `{', '.join(payload['sanity_config']['doc_ids'])}`",
        f"- tuning/final doc overlap: `{', '.join(payload['overlap']['tuning_vs_final_doc_overlap'])}`",
        f"- frozen bundle git commit: `{payload['frozen_bundle']['git_commit']}`",
        f"- frozen bundle conda env: `{payload['frozen_bundle']['conda_default_env']}`",
        f"- frozen bundle requirements mismatch count: `{payload['frozen_bundle']['requirements_mismatch_count']}`",
        "",
        "## Findings",
        "",
    ]
    for finding in payload["findings"]:
        lines.append(f"- {finding}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The tracked configs show what was used for tuning-style exploration versus final promoted chunk settings.",
            "- The frozen bundle should be treated as canonical only if its environment manifest is also clean.",
            "- If the thesis reuses the same 5 documents for tuning and reporting, that should be stated explicitly as a limitation of the evaluation protocol.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    tuning_config = _config_summary((REPO_ROOT / args.tuning_config).resolve())
    final_config = _config_summary((REPO_ROOT / args.final_config).resolve())
    sanity_config = _config_summary((REPO_ROOT / args.sanity_config).resolve())
    bundle_summary = _bundle_summary((REPO_ROOT / args.bundle_dir).resolve())

    payload = {
        "status": "audit_complete",
        "tuning_config": tuning_config,
        "final_config": final_config,
        "sanity_config": sanity_config,
        "frozen_bundle": bundle_summary,
        "overlap": {
            "tuning_vs_final_doc_overlap": _overlap_summary(tuning_config["doc_ids"], final_config["doc_ids"]),
            "tuning_vs_sanity_doc_overlap": _overlap_summary(tuning_config["doc_ids"], sanity_config["doc_ids"]),
            "final_vs_sanity_doc_overlap": _overlap_summary(final_config["doc_ids"], sanity_config["doc_ids"]),
        },
        "findings": _findings(
            tuning=tuning_config,
            final=final_config,
            sanity=sanity_config,
            bundle=bundle_summary,
        ),
    }

    out_json = (REPO_ROOT / args.out_json).resolve()
    out_md = (REPO_ROOT / args.out_md).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out_md.write_text(_markdown_report(payload), encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
