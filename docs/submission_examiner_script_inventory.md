# Submission Examiner Script Inventory

Quick reference for scripts under `submission_examiner/scripts/`.

Conventions:
- `Main inputs` lists the primary CLI flags only.
- `Outputs` lists explicit files/directories where the script parser makes them clear; otherwise it says `stdout / in-place artifacts`.

| Script | What it does | Main inputs | Outputs |
|---|---|---|---|
| `build_global_indexes.py` | Build global dense index and lexical manifest from per-document artifacts. | `--data-root`, `--out-dir`, `--save-embeddings` | global index directory under `out-dir` |
| `build_index.py` | Build embeddings and FAISS index from `chunks.parquet`. | `--data-dir`, `--model` | in-place index/embedding artifacts |
| `check_environment.py` | Run preflight environment validation for the thesis-final pipeline. | `--json`, `--strict` | stdout / optional JSON-formatted report |
| `check_pipeline_reproducibility.py` | Re-run the pipeline and check whether canonical outputs are byte-identical. | `--data-root`, `--doc-pattern`, `--model-path`, `--runs`, `--out-json` | reproducibility report JSON |
| `check_thesis_bundle_drift.py` | Compare a current thesis bundle against a baseline bundle and report drift. | `--bundle-dir`, `--baseline-bundle`, `--out-dir`, `--scope` | drift report directory |
| `evaluate_pipeline.py` | Evaluate the production pipeline using canonical hybrid Dense+BM25 RRF retrieval. | `--data-dir`, `--model`, `--k-list`, `--rrf-k`, `--dense-weight`, `--bm25-weight` | stdout / evaluation artifacts |
| `export_thesis_bootstrap_table.py` | Export a thesis-ready paired-bootstrap summary table. | `--input-dir`, `--out-dir` | table export files in `out-dir` |
| `export_thesis_chunk_ablation_table.py` | Export a thesis-facing chunk-ablation table from frozen per-document retrieval outputs. | `--data-root`, `--out-dir` | table export files in `out-dir` |
| `export_thesis_failure_analysis_bundle.py` | Export thesis-facing FP1-FP7 artifacts from one or two completed analysis directories. | `--baseline-dir`, `--candidate-dir`, `--comparison-dir`, `--out-dir` | failure-analysis bundle in `out-dir` |
| `export_thesis_mcnemar_table.py` | Export a thesis-ready McNemar Hit@1 summary table. | `--batch-summary-csv`, `--out-dir` | table export files in `out-dir` |
| `export_thesis_ragas_table.py` | Export a thesis-ready RAGAS summary comparison table. | repeated `--run`, `--out-dir` | table export files in `out-dir` |
| `export_thesis_rebuild_bundle.py` | Run the thesis freeze/export helpers and collect outputs into one rebuild bundle. | `--bundle-dir`, optional source dirs such as `--chunk-data-root`, `--failure-baseline-dir`, repeated `--ragas-run` | rebuilt thesis bundle artifacts under `bundle-dir` |
| `paired_bootstrap_retrieval_compare.py` | Run paired bootstrap comparison between two retrieval result JSON files. | `--system-a`, `--system-b`, `--mrr-k`, `--n-bootstrap`, `--out-dir` | bootstrap comparison directory |
| `patch_thesis_from_frozen_bundle.py` | Copy selected frozen-bundle assets into the thesis source tree and patch LaTeX references. | `--bundle-dir`, `--thesis-root`, `--write` | dry-run report or in-place thesis file updates |
| `preprocess_hybrid.py` | Run hybrid PDF preprocessing with OCR fallback and chunking/table options. | `--pdf-path`, `--out-root`, chunking/markdown flags | processed document directory under `out-root` |
| `report_retrieval_metrics.py` | Build retrieval metrics reports in CSV, Markdown, and LaTeX. | `--data-root`, `--docs`, `--out-csv`, `--out-md`, `--out-tex`, query-level output flags | report files at requested output paths |
| `retrieval_eval_hybrid.py` | Evaluate retrieval using hybrid Dense+BM25 RRF fusion. | `--data-dir`, `--model`, `--k-list`, `--rrf-k`, `--dense-weight`, `--bm25-weight` | stdout / evaluation artifacts |
| `run_current_pipeline_fp1_fp7.py` | Run full FP1-FP7 failure analysis on the current hybrid `SearchService` pipeline. | `--data-root`, `--doc-pattern`, `--model-path`, `--k`, generation flags, `--out-dir` | FP1-FP7 analysis directory |
| `run_full_pipeline.py` | Run the full pipeline: preprocess, build index, evaluate, and report. | `--pdf-dir`, `--pdf-glob`, `--out-root`, `--model`, chunk-size flags | end-to-end pipeline outputs under `out-root` |
| `run_mcnemar_hit1_batch.py` | Run paired McNemar Hit@1 tests across multiple cohorts. | `--out-dir`, `--alpha`, repeated `--pair`, `--cohort-prefix` | McNemar batch output directory |
| `run_ragas_eval.py` | Run RAGAS against an exported JSONL dataset. | `--input-jsonl`, `--out-dir`, `--llm-model`, `--embedding-model`, `--sample-n` | RAGAS results directory |
| `run_retrieval_ablation.py` | Run retrieval A/B ablation experiments from a YAML config. | `--config`, `--only` | ablation run artifacts |
| `runtime_env.py` | Runtime environment helper module used by the examiner scripts. | none when imported directly | stdout / environment details when run directly |
| `setup_thesis_rebuild_freeze.py` | Create an isolated, reproducible thesis rebuild root with frozen config and manifest. | `--run-name`, `--base-config`, `--results-root`, `--data-root` | frozen rebuild directory plus manifest files |

## Minimal Working Set

If you only need the main examiner-facing path, these are the scripts to care about first:

| Script | Role |
|---|---|
| `preprocess_hybrid.py` | PDF to processed artifacts |
| `build_index.py` | processed chunks to embeddings/index |
| `retrieval_eval_hybrid.py` | hybrid retrieval evaluation |
| `evaluate_pipeline.py` | production-style evaluation wrapper |
| `run_full_pipeline.py` | end-to-end orchestrator |
| `run_current_pipeline_fp1_fp7.py` | failure-analysis runner |
| `report_retrieval_metrics.py` | final retrieval reporting |
| `setup_thesis_rebuild_freeze.py` | freeze reproducible rebuild inputs |
| `export_thesis_rebuild_bundle.py` | gather thesis-ready outputs |
| `check_pipeline_reproducibility.py` | reproducibility guardrail |
