# Thesis Rebuild Freeze Workflow

Use this workflow when regenerating thesis-facing results from scratch under tighter controls.

## Goal

Create one isolated result bundle that contains:

- the frozen config used for the rebuild
- the runtime and dependency manifest
- the processed corpora generated for that run
- the exported thesis tables derived from those outputs

Do not mix those outputs with older historical artifacts.

## Setup

Create a fresh bundle:

```bash
python scripts/setup_thesis_rebuild_freeze.py \
  --run-name thesis_rebuild_freeze_YYYY-MM-DD \
  --base-config configs/retrieval_tuning_minilm_cap_5docs.yaml
```

This writes:

- `results/thesis_rebuild_freeze/<run-name>/configs/..._frozen.yaml`
- `results/thesis_rebuild_freeze/<run-name>/manifests/environment_manifest.json`
- `results/thesis_rebuild_freeze/<run-name>/RUNBOOK.md`

## Rebuild

Run the ablation from the frozen config, not from the tracked source config:

```bash
python scripts/run_retrieval_ablation.py \
  --config results/thesis_rebuild_freeze/<run-name>/configs/<config>_frozen.yaml
```

## Export Thesis Table

Build the thesis-facing chunk-ablation table directly from the frozen per-document outputs:

```bash
python scripts/export_thesis_chunk_ablation_table.py \
  --data-root data_variants/thesis_rebuild_freeze/<run-name>/<config> \
  --out-dir results/thesis_rebuild_freeze/<run-name>/tables
```

Exports:

- `chunk_ablation_by_document.csv`
- `chunk_ablation_table.csv`
- `chunk_ablation_table.json`
- `chunk_ablation_table.tex`

## Additional Thesis Exports

The same freeze/export pattern can be used for other thesis-facing outputs.

### All-in-One Export

Once the required source outputs exist, export them into the frozen bundle in one command:

```bash
python scripts/export_thesis_rebuild_bundle.py \
  --bundle-dir results/thesis_rebuild_freeze/<run-name> \
  --chunk-data-root data_variants/thesis_rebuild_freeze/<run-name>/<config> \
  --failure-baseline-dir results/live_fp1_fp7_current_pipeline_norm_YYYY-MM-DD \
  --failure-candidate-dir results/live_fp1_fp7_current_pipeline_llm_norm_YYYY-MM-DD \
  --failure-comparison-dir results/live_fp1_fp7_compare_llm_vs_retrieval_norm \
  --bootstrap-input-dir results/tiktoken_refresh_YYYY-MM-DD/paired_bootstrap \
  --mcnemar-batch-summary-csv results/mcnemar_hit1_batch_grampian_2021_2025/mcnemar_hit1_batch_summary.csv \
  --ragas-run retrieval_only::results/ragas/run75_224_baseline \
  --ragas-run llm_on::results/ragas/run75_224_generated
```

This writes standard subdirectories under the bundle:

- `tables/`
- `failure_analysis/`
- `bootstrap/`
- `mcnemar/`
- `ragas/`

It also writes scope manifests and provenance guardrails under the bundle so each exported table or figure can be tied back to that frozen bundle.

### Patch LaTeX From The Frozen Bundle

Once the bundle exports are ready, patch the thesis chapters so they read directly from that bundle:

```bash
python scripts/patch_thesis_from_frozen_bundle.py \
  --bundle-dir results/thesis_rebuild_freeze/<run-name> \
  --write
```

Currently this rewires:

- the chunk-ablation table in `methodology.tex`
- the paired bootstrap figure in `results.tex`
- the retrieval-only FP1--FP7 heatmap in `results.tex`
- the retrieval-vs-LLM FP1--FP7 comparison figure in `results.tex`

### Drift Guardrail

To highlight when a new frozen bundle differs from an earlier one, add:

```bash
  --compare-against-bundle results/thesis_rebuild_freeze/<older-run>
```

This writes:

- `guardrails/bundle_drift_report.csv`
- `guardrails/bundle_drift_report.json`
- `guardrails/bundle_drift_report.md`

The guardrail compares exported CSV/JSON/TeX/PNG/MD outputs under:

- `tables/`
- `failure_analysis/`
- `bootstrap/`
- `mcnemar/`
- `ragas/`

and reports which files are unchanged, changed, added, or removed.

### Provenance Guardrail

Every thesis-exported scope now writes a `manifest.json` describing:

- the frozen bundle it belongs to
- the source inputs used for that export
- the concrete exported files owned by that scope

Validate those links with:

```bash
python scripts/check_thesis_export_provenance.py \
  --bundle-dir results/thesis_rebuild_freeze/<run-name>
```

This writes:

- `guardrails/bundle_provenance_report.csv`
- `guardrails/bundle_provenance_report.json`
- `guardrails/bundle_provenance_report.md`

If you are validating an older bundle created before scope manifests existed, backfill them first:

```bash
python scripts/backfill_thesis_export_manifests.py \
  --bundle-dir results/thesis_rebuild_freeze/<run-name>
```

### FP1--FP7 Failure Analysis

```bash
python scripts/export_thesis_failure_analysis_bundle.py \
  --baseline-dir results/live_fp1_fp7_current_pipeline_norm_YYYY-MM-DD \
  --candidate-dir results/live_fp1_fp7_current_pipeline_llm_norm_YYYY-MM-DD \
  --comparison-dir results/live_fp1_fp7_compare_llm_vs_retrieval_norm \
  --out-dir results/thesis_rebuild_freeze/<run-name>/failure_analysis
```

### Paired Bootstrap Hybrid vs Dense

```bash
python scripts/export_thesis_bootstrap_table.py \
  --input-dir results/tiktoken_refresh_YYYY-MM-DD/paired_bootstrap \
  --out-dir results/thesis_rebuild_freeze/<run-name>/bootstrap
```

### McNemar Hit@1 Batch Summary

```bash
python scripts/export_thesis_mcnemar_table.py \
  --batch-summary-csv results/mcnemar_hit1_batch_grampian_2021_2025/mcnemar_hit1_batch_summary.csv \
  --out-dir results/thesis_rebuild_freeze/<run-name>/mcnemar
```

### RAGAS Summary Comparison

```bash
python scripts/export_thesis_ragas_table.py \
  --run retrieval_only::results/ragas/run75_224_baseline \
  --run llm_on::results/ragas/run75_224_generated \
  --out-dir results/thesis_rebuild_freeze/<run-name>/ragas
```

## Controls

Keep these controls fixed for the whole rebuild:

- same input PDFs
- same evaluator script
- same dependency stack
- same env-var locks recorded in the manifest
- same frozen config copied into the result bundle

## Rule

If a thesis number appears in LaTeX, it should be traceable to one file under:

- `results/thesis_rebuild_freeze/<run-name>/tables`
- or a named subdirectory such as `failure_analysis/`, `bootstrap/`, `mcnemar/`, or `ragas/`

If it cannot be traced there, it is not part of the frozen rebuild.
