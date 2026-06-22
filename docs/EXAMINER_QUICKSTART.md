# Examiner Quickstart

Use this path if you want the shortest reliable way to verify the submitted pipeline.

## Recommended Path

```bash
conda activate rag-pipeline
python scripts/check_examiner_path.py
python scripts/check_environment.py --strict
python scripts/check_pipeline_reproducibility.py --runs 2 --out-json results/reproducibility/examiner_repro_check.json
python scripts/check_retrieval_parity_batch.py --out-json results/reproducibility/retrieval_parity_batch_smoke.json
python scripts/audit_evaluation_protocol.py
python scripts/check_thesis_export_provenance.py --bundle-dir results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18
```

What this does:

- confirms the expected examiner-facing files are present
- confirms the pinned environment is active and aligned
- reruns a short reproducibility check over the evaluation-ready corpus set
- checks a small multi-document retrieval parity batch and writes a standard parity report
- writes a protocol audit summarizing tuning scope, final promoted config scope, and frozen bundle metadata
- confirms thesis-exported tables and figures are linked to the frozen bundle by scope manifests

## What To Trust

For thesis-facing outputs, treat these as canonical:

- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/`
- `results/reproducibility/examiner_repro_check.json`
- `results/reproducibility/retrieval_parity_batch_smoke.json`
- `results/reproducibility/evaluation_protocol_audit.json`
- `docs/EXAMINER_SUBMISSION_MANIFEST.md`

Use the frozen bundle for traced tables and figures, not ad hoc local outputs.

## Reproducing Specific Thesis Tables and Figures

The following scripts use the refactored `src/thesis_rag/` module to regenerate individual thesis outputs from source PDFs and the frozen config. They require the source PDFs to be present.

```bash
# Table 2.1 — corpus era summary statistics
python scripts/reproduce_table_2_1_thesis_rag.py --config configs/thesis_rag.yaml

# Table 4.5 — per-document vs global retrieval comparison
python scripts/reproduce_table_4_5_doc_vs_global_thesis_rag.py --config configs/thesis_rag.yaml

# Appendix Table C.6 — BM25 tokenizer sensitivity
python scripts/reproduce_table_c6_bm25_tokenizer.py --config configs/thesis_rag.yaml

# Figure 4.1 — retrieval performance across chunk sizes
python scripts/reproduce_figure_4_1_thesis_rag.py --config configs/thesis_rag.yaml

# Figure 4.2 — chunk size ablation
python scripts/reproduce_figure_4_2_thesis_rag.py --config configs/thesis_rag.yaml
```

For tracing thesis numbers back to frozen artifacts without rerunning, use the frozen bundle directly:
- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/tables/`
- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/bootstrap/`
- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/ragas/`
- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/mcnemar/`

Note: the `RUNBOOK.md` inside that bundle contains absolute paths local to the original development machine. Use the scripts above as the portable alternative.

## Optional Demo Path

The demo UI and API are for inspection, not for verifying headline thesis claims.

API:

```bash
bash scripts/run_api_demo.sh
```

UI:

```bash
bash scripts/run_streamlit_demo.sh
```

Demo defaults stay local-only. Uploads are disabled unless explicitly enabled.

## Expected Scope

The examiner package is intended to support three things:

- verify the environment and pinned dependencies
- verify that reproducibility evidence is present
- trace thesis-facing outputs to the frozen bundle

It is not intended to rebuild every exploratory experiment from scratch.
