# Examiner Submission Manifest

This manifest lists the files and directories that should go into the final examiner package for this repository.

Scope:
- final pipeline code only
- final configs only
- evaluation/reproducibility code only
- frozen thesis-facing result artifacts only

Do not include exploratory, superseded, UI/demo, archive, or scratch material unless your thesis explicitly cites it.

## Include

### Top-level files

- `LICENSE`
- `README.md`
- `environment.yml`
- `requirements.txt`
- `preprocess_hybrid.py`

### Core library code

Include the entire `src/` tree. It contains three modules:

**`src/rag_pdf/`** — original evaluated pipeline (used by `retrieval_eval_hybrid.py` and the frozen bundle):
- `src/rag_pdf/__init__.py`
- `src/rag_pdf/boilerplate.py`
- `src/rag_pdf/chunking.py`
- `src/rag_pdf/config.py`
- `src/rag_pdf/extract_page.py`
- `src/rag_pdf/headings.py`
- `src/rag_pdf/metrics.py`
- `src/rag_pdf/ocr_quality.py`
- `src/rag_pdf/ocr_table_fallback.py`
- `src/rag_pdf/question_router.py`
- `src/rag_pdf/rotation_handler.py`
- `src/rag_pdf/schemas.py`
- `src/rag_pdf/sections.py`
- `src/rag_pdf/table_canonicalize.py`
- `src/rag_pdf/table_detect.py`
- `src/rag_pdf/table_extract.py`
- `src/rag_pdf/text_normalize.py`
- `src/rag_pdf/toc.py`
- `src/rag_pdf/retrieval/__init__.py`
- `src/rag_pdf/retrieval/canonical_hybrid.py`
- `src/rag_pdf/retrieval/hybrid_utils.py`
- `src/rag_pdf/retrieval/query_rewrite.py`
- `src/rag_pdf/retrieval/rerank.py`
- `src/rag_pdf/services/__init__.py`
- `src/rag_pdf/services/local_llm_service.py`
- `src/rag_pdf/services/numeric_extraction.py`
- `src/rag_pdf/services/numeric_normalization.py`
- `src/rag_pdf/services/process_service.py`
- `src/rag_pdf/services/search_helpers.py`
- `src/rag_pdf/services/search_service.py`
- `src/rag_pdf/services/storage_service.py`

**`src/thesis_rag/`** — refactored clean-room module (used by `reproduce_*.py` scripts):
- `src/thesis_rag/pipeline.py` — end-to-end orchestrator
- `src/thesis_rag/config.py`, `src/thesis_rag/schemas.py`
- `src/thesis_rag/chunking.py`, `src/thesis_rag/embedding.py`, `src/thesis_rag/indexing.py`
- `src/thesis_rag/retrieval_dense.py`, `src/thesis_rag/retrieval_sparse.py`, `src/thesis_rag/retrieval_hybrid.py`, `src/thesis_rag/fusion.py`
- `src/thesis_rag/evaluator.py`, `src/thesis_rag/ranking.py`, `src/thesis_rag/artifacts.py`
- `src/thesis_rag/loader.py`, `src/thesis_rag/preprocessing.py`, `src/thesis_rag/ocr.py`
- `src/thesis_rag/diagnostics.py`, `src/thesis_rag/utils.py`

**`src/generation/`** — constrained answer extraction utilities:
- `src/generation/__init__.py`
- `src/generation/constrained_extraction.py`

### Final pipeline and verification scripts

Include these scripts:

- `scripts/check_examiner_path.py`
- `scripts/check_environment.py`
- `scripts/check_retrieval_parity.py`
- `scripts/check_retrieval_parity_batch.py`
- `scripts/audit_evaluation_protocol.py`
- `scripts/preprocess_hybrid.py`
- `scripts/build_index.py`
- `scripts/build_global_indexes.py`
- `scripts/evaluate_pipeline.py`
- `scripts/retrieval_eval_hybrid.py`
- `scripts/report_retrieval_metrics.py`
- `scripts/run_full_pipeline.py`
- `scripts/check_pipeline_reproducibility.py`
- `scripts/check_thesis_export_provenance.py`
- `scripts/runtime_env.py`

### Thesis rebuild / frozen-bundle scripts

Include these scripts because they define the controlled thesis export workflow:

- `scripts/setup_thesis_rebuild_freeze.py`
- `scripts/run_retrieval_ablation.py`
- `scripts/export_thesis_chunk_ablation_table.py`
- `scripts/export_thesis_failure_analysis_bundle.py`
- `scripts/export_thesis_bootstrap_table.py`
- `scripts/export_thesis_mcnemar_table.py`
- `scripts/export_thesis_ragas_table.py`
- `scripts/export_thesis_rebuild_bundle.py`
- `scripts/backfill_thesis_export_manifests.py`
- `scripts/check_thesis_bundle_drift.py`
- `scripts/patch_thesis_from_frozen_bundle.py`

### Analysis scripts to regenerate thesis-cited comparison outputs

Include these only because they support the frozen bundle contents already present in `results/thesis_rebuild_freeze/...`:

- `scripts/run_current_pipeline_fp1_fp7.py`
- `scripts/paired_bootstrap_retrieval_compare.py`
- `scripts/run_mcnemar_hit1_batch.py`
- `scripts/run_ragas_eval.py`

### Thesis reproduction scripts

These scripts use `src/thesis_rag/` to regenerate specific thesis tables and figures from the raw PDFs and frozen config. They are the primary entry points for an examiner who wants to verify a specific claim:

- `scripts/reproduce_table_2_1_thesis_rag.py` — corpus era summary statistics (Table 2.1)
- `scripts/reproduce_table_4_5_doc_vs_global_thesis_rag.py` — per-document vs global retrieval comparison
- `scripts/reproduce_table_c6_bm25_tokenizer.py` — BM25 tokenizer sensitivity (Appendix C.6)
- `scripts/reproduce_figure_4_1_thesis_rag.py` — retrieval performance figure (Figure 4.1)
- `scripts/reproduce_figure_4_2_thesis_rag.py` — chunk size ablation figure (Figure 4.2)
- `scripts/reproduce_figure_4_2_postfix.py` — post-hoc figure 4.2 variant

### QA scripts

Include these lightweight checks:

- `qa/qa_check_preprocessing.py`
- `qa/validate_hybrid_pipeline.py`
- `qa/validate_text_extraction_hybrid.py`
- `qa/check_empty_pages.py`

### Final configs

Include these config files:

- `configs/README.md`
- `configs/thesis_rag.yaml` — primary pipeline config (224/56 tokens, MiniLM-L6-v2, RRF k=20)
- `configs/retrieval_tuning_minilm_cap_5docs.yaml` — promoted config used for frozen bundle rebuild
- `configs/retrieval_tuning_thesis_5docs_q50.yaml` — tuning exploration config (50 queries)
- `configs/retrieval_tuning_224_56_5docs.yaml` — focused 224/56 sanity comparison config

Only include these if you explicitly discuss them in the thesis:

- `configs/retrieval_tuning_thesis.yaml`
- `configs/retrieval_tuning_thesis_all_docs.yaml`
- `configs/retrieval_tuning.yaml`

Do not include the remaining config files unless cited.

### Documentation

Include:

- `docs/EXAMINER_QUICKSTART.md`
- `docs/EVALUATION_PROTOCOL.md`
- `docs/thesis_rebuild_freeze_workflow.md`
- `docs/EXAMINER_SUBMISSION_MANIFEST.md`

### Evaluation sets

Include these exact files:

- `data_processed/Grampian-2020-2021/eval_set.json`
- `data_processed/Grampian-2021-2022/eval_set.json`
- `data_processed/Grampian-2022-2023/eval_set.json`
- `data_processed/Grampian-2023-2024/eval_set.json`
- `data_processed/Grampian-2024-2025/eval_set.json`

### Reproducibility evidence

Include:

- `results/reproducibility/grampian_5docs_repro.json` — original 30-run hash check (PASS, all identical)
- `results/reproducibility/current_pipeline_grampian_5docs_repro_2026-04-17.json` — latest 30-run hash check (PASS, all identical)
- `results/reproducibility/retrieval_parity_batch_smoke.json`
- `results/reproducibility/evaluation_protocol_audit.json`
- `results/reproducibility/evaluation_protocol_audit.md`
- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/guardrails/bundle_provenance_report.json`

### Frozen thesis result bundle

Include the entire directory:

- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/`

That directory currently contains:

- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/RUNBOOK.md`
- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/configs/retrieval_tuning_minilm_cap_5docs_frozen.yaml`
- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/manifests/environment_manifest.json`
- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/tables/chunk_ablation_by_document.csv`
- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/tables/chunk_ablation_table.csv`
- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/tables/chunk_ablation_table.json`
- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/tables/chunk_ablation_table.tex`
- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/failure_analysis/fp1_fp7_run_summary.csv`
- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/failure_analysis/fp1_fp7_run_summary.json`
- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/failure_analysis/fp1_fp7_run_summary.tex`
- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/failure_analysis/manifest.json`
- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/bootstrap/paired_bootstrap_ci_panel_Grampian_2020_2025_hybrid_vs_dense.png`
- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/bootstrap/paired_bootstrap_summary_table.csv`
- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/bootstrap/paired_bootstrap_summary_table.json`
- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/bootstrap/paired_bootstrap_summary_table.tex`
- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/mcnemar/mcnemar_hit1_summary_table.csv`
- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/mcnemar/mcnemar_hit1_summary_table.json`
- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/mcnemar/mcnemar_hit1_summary_table.tex`
- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/ragas/ragas_summary_table.csv`
- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/ragas/ragas_summary_table.json`
- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/ragas/ragas_summary_table.tex`
- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/guardrails_selfcheck/bundle_drift_report.csv`
- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/guardrails_selfcheck/bundle_drift_report.json`
- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/guardrails_selfcheck/bundle_drift_report.md`
- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/guardrails/bundle_provenance_report.csv`
- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/guardrails/bundle_provenance_report.json`
- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/guardrails/bundle_provenance_report.md`

## Exclude

Do not include these directories in the examiner package:

- `app/`
- `archive/`
- `artifacts/`
- `figures/`
- `models/`
- `runs/`
- `tmp/`
- `__pycache__/`

Do not include these content classes unless explicitly cited in the thesis:

- UI/demo shell scripts
- one-off plotting scripts
- legacy retrieval evaluators not used in the final method
- exploratory ablation scripts not referenced in the final write-up
- presentation slides and architecture artwork under `docs/architecture/`
- alternate corpora under `data_variants/`
- processed parquet/index artifacts under `data_processed/` other than the five `eval_set.json` files

## Not In Repo But Should Be Provided Separately

If examiners need to rerun the full pipeline from scratch, provide these outside the code bundle:

- the five source PDFs used in the thesis
- any local embedding model or model download instructions
- any processed corpora too large for the submission system

## Packaging Rule

If a file is not needed to do one of these three things, leave it out:

- rebuild the final pipeline
- verify the reproducibility claim
- trace a thesis-reported number to a frozen output artifact
