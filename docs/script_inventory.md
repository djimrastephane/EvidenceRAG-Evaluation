# Script Inventory

Quick reference for script entrypoints under `scripts/` and `submission_examiner/scripts/`.

Conventions:
- `Inputs` lists the main CLI flags only, not every tuning flag.
- `Outputs` lists explicit files/directories when the parser names them; otherwise it says `stdout / in-place artifacts`.
- `submission_examiner/scripts/` mostly mirrors the thesis-finalized subset of the main scripts.

## Core Pipeline

| Script | Purpose | Main inputs | Outputs |
|---|---|---|---|
| `scripts/preprocess_hybrid.py` | Hybrid PDF preprocessing with OCR/table handling | `--pdf-path`, `--out-root`, chunking/table flags | processed doc folder under `out-root` |
| `scripts/build_index.py` | Build embeddings + FAISS index from chunks | `--data-dir`, `--model`, `--device` | in-place index/embedding artifacts |
| `scripts/build_global_indexes.py` | Build global dense + lexical indexes across docs | `--data-root`, `--out-dir` | global index directory |
| `scripts/retrieval_eval.py` | Dense retrieval eval | `--data-dir`, `--model`, `--k-list` | stdout / per-run metrics |
| `scripts/retrieval_eval_bm25.py` | BM25-only retrieval eval | `--data-dir`, `--k-list`, BM25 params | stdout / per-run metrics |
| `scripts/retrieval_eval_hybrid.py` | Hybrid dense+BM25 RRF eval | `--data-dir`, `--model`, fusion flags | stdout / per-run metrics |
| `scripts/retrieval_eval_splade_hybrid.py` | Dense+SPLADE hybrid eval | `--data-dir`, dense/SPLADE params | stdout / per-run metrics |
| `scripts/retrieval_eval_rewrites.py` | Retrieval eval with deterministic rewrites | `--data-dir`, `--model`, `--k-list` | stdout / per-run metrics |
| `scripts/evaluate_pipeline.py` | Production-style hybrid evaluation | `--data-dir`, `--model`, `--k-list`, fusion flags | stdout / result artifacts |
| `scripts/run_full_pipeline.py` | End-to-end preprocess -> index -> eval -> reports | `--pdf-dir`, `--out-root`, `--model` | processed/eval outputs under `out-root` |
| `scripts/run_batch.py` | Batch preprocess multiple PDFs from config | `--config`, `--force` | processed folders, logs, batch summary |
| `scripts/run_current_pipeline_fp1_fp7.py` | Live FP1-FP7 failure analysis on current pipeline | `--data-root`, `--doc-pattern`, `--out-dir` | FP1-FP7 analysis directory |
| `scripts/check_environment.py` | Environment/preflight checks | `--json`, `--strict` | stdout / JSON |
| `scripts/check_pipeline_reproducibility.py` | Re-run pipeline and test reproducibility | `--data-root`, `--doc-pattern`, `--runs`, `--out-json` | reproducibility JSON |
| `scripts/check_retrieval_parity.py` | Compare evaluator vs service on one query | `--data-dir`, `--query-id` | stdout |
| `scripts/check_retrieval_parity_batch.py` | Batch parity check | data root, doc/query sampling, `--out-json` | parity JSON |

## Retrieval Tuning And Ablation

| Script | Purpose | Main inputs | Outputs |
|---|---|---|---|
| `scripts/run_retrieval_ablation.py` | Run YAML-defined retrieval ablations | `--config`, `--only` | ablation run artifacts |
| `scripts/report_retrieval_ablation.py` | Summarize ablation outputs | `--summary-csv`, `--out-md`, `--out-csv` | markdown + CSV report |
| `scripts/run_dense_encoder_ablation.py` | Compare dense encoders on fixed corpus | corpus/model lists, `--out-root` | ablation outputs |
| `scripts/run_table_chunking_ablation.py` | Compare table chunking variants | sample CSV, source root, `--out-root` | ablation outputs |
| `scripts/run_intrinsic_ablation.py` | No-eval intrinsic preprocessing ablations | `--config`, filtering flags | stdout / variant outputs |
| `scripts/tune_hybrid_rrf_weights.py` | Grid search RRF weights | `--run-root`, weights, `--out-dir` | tuning results |
| `scripts/tune_hybrid_rrf_weights_cv.py` | Nested CV tuning for RRF weights | run root, grids, `--out-dir` | CV tuning results |
| `scripts/ablate_fusion_strategy_temporal.py` | Compare RRF vs score fusion by temporal fold | run root, weights, `--out-dir` | temporal ablation outputs |
| `scripts/bootstrap_fusion_delta_temporal.py` | Bootstrap deltas for fusion strategies | run root, bootstrap params, `--out-dir` | bootstrap outputs |
| `scripts/ablate_max_k_search.py` | Test `MAX_K_SEARCH` sensitivity | run root, k lists, `--out-dir` | ablation outputs |
| `scripts/compare_full_weight_settings.py` | Compare baseline vs candidate fusion settings | roots/weights, `--out-csv`, `--out-json` | comparison CSV/JSON |
| `scripts/compare_22456_cross_encoder.py` | Compare cross-encoder on/off for 224/56 setup | run root, weights, `--out-csv`, `--out-json` | comparison CSV/JSON |
| `scripts/run_bm25_tokenizer_sensitivity.py` | Test BM25 tokenization choices | docs, tokenizers, setups | stdout / run artifacts |
| `scripts/compare_tiktoken_vs_fallback.py` | Compare tokenizer fallback chunk stats | `--data-root`, `--out-csv`, docs | CSV |
| `scripts/plot_tiktoken_vs_fallback_comparison.py` | Plot tokenizer comparison | `--csv`, `--out` | figure |
| `scripts/plot_retrieval_compare_fallback_vs_tiktoken.py` | Plot retrieval deltas from tokenizer comparison | `--csv`, `--out` | figure |

## Generation And Answering Studies

| Script | Purpose | Main inputs | Outputs |
|---|---|---|---|
| `scripts/ablate_generation_context.py` | Test generation context size/chunk settings | data root, docs, `--out-dir` | ablation outputs |
| `scripts/ablate_generation_prompts.py` | Compare prompt variants | data root, docs, `--out-dir` | ablation outputs |
| `scripts/ablate_numeric_generation_models.py` | Compare local LLMs for numeric answers | input CSV, model list, `--out-csv` | CSV |
| `scripts/eval_strict_evidence_extraction.py` | Evaluate constrained extraction | sample CSV, data root, gen params, `--out-dir` | eval outputs |
| `scripts/judge_quoted_failures.py` | LLM-judge diagnosis of quoted-answer failures | ablation dir, data root, `--out-dir` | diagnosis outputs |
| `scripts/rescore_fp1_fp7_with_saved_generation.py` | Re-score FP1-FP7 using saved generations | saved-generation CSV, `--out-dir` | rescored analysis |

## Reporting, Exports, And Thesis Bundles

| Script | Purpose | Main inputs | Outputs |
|---|---|---|---|
| `scripts/report_retrieval_metrics.py` | Export retrieval metrics tables | docs, `--out-csv`, `--out-md`, `--out-tex` | CSV/MD/TeX reports |
| `scripts/export_consolidated_answers.py` | Merge per-doc retrieval results into canonical consolidated answers | `--data-root`, `--results-name`, `--out-dir` | consolidated CSV/JSONL |
| `scripts/export_ragas_dataset.py` | Build RAGAS-ready dataset | docs, model path, `--out-jsonl`, `--out-csv` | JSONL + CSV |
| `scripts/run_ragas_eval.py` | Run RAGAS on exported dataset | `--input-jsonl`, `--out-dir`, model flags | RAGAS results dir |
| `scripts/plot_ragas_results.py` | Plot RAGAS outputs | per-query CSV/JSONL, `--out-dir` | charts |
| `scripts/generate_ragas_comparison_chart.py` | Compare baseline vs generated RAGAS summaries | baseline/generated summaries, `--out-csv`, `--out-png` | CSV + PNG |
| `scripts/export_thesis_ragas_table.py` | Thesis-ready RAGAS table | repeated `--run`, `--out-dir` | thesis table outputs |
| `scripts/paired_bootstrap_retrieval_compare.py` | Bootstrap compare two retrieval systems | `--system-a`, `--system-b`, `--out-dir` | bootstrap outputs |
| `scripts/plot_paired_bootstrap_retrieval.py` | Plot bootstrap comparison | `--input-dir`, `--label`, `--output-dir` | plots |
| `scripts/plot_paired_bootstrap_panel.py` | Multi-cohort bootstrap panel | cohorts, input root, `--output-path` | figure |
| `scripts/export_thesis_bootstrap_table.py` | Thesis-ready bootstrap table | `--input-dir`, `--out-dir` | table exports |
| `scripts/mcnemar_hit1_compare.py` | McNemar test for Hit@1 | `--hybrid`, `--dense`, `--cohort`, `--out-dir` | stats outputs |
| `scripts/run_mcnemar_hit1_batch.py` | Batch McNemar across cohorts | repeated `--pair`, `--out-dir` | batch outputs |
| `scripts/export_thesis_mcnemar_table.py` | Thesis-ready McNemar table | batch summary CSV, `--out-dir` | table exports |
| `scripts/export_thesis_chunk_ablation_table.py` | Thesis chunk-ablation table | `--data-root`, `--out-dir` | table exports |
| `scripts/export_thesis_failure_analysis_bundle.py` | Thesis-facing FP1-FP7 bundle | baseline/candidate dirs, `--out-dir` | failure analysis bundle |
| `scripts/export_thesis_rebuild_bundle.py` | Collect all thesis rebuild/freeze exports | bundle dir plus source dirs | rebuild bundle artifacts |
| `scripts/setup_thesis_rebuild_freeze.py` | Create frozen thesis rebuild root | run/config/data roots | freeze directory + manifest |
| `scripts/check_thesis_bundle_drift.py` | Compare current vs baseline frozen bundle | bundle dirs, `--out-dir` | drift reports |
| `scripts/check_thesis_export_provenance.py` | Verify bundle outputs against manifests | bundle dir, `--out-dir` | provenance reports |
| `scripts/backfill_thesis_export_manifests.py` | Add missing manifests to existing bundle | bundle dir, scope flags | in-place manifest files |
| `scripts/patch_thesis_from_frozen_bundle.py` | Copy bundle outputs into thesis source and patch LaTeX | `--bundle-dir`, `--thesis-root`, `--write` | dry-run report or in-place patch |
| `scripts/audit_evaluation_protocol.py` | Audit evaluation protocol configs and freeze metadata | config paths, bundle dir, outputs | JSON + markdown audit |

## Data Audit And Corpus Utilities

| Script | Purpose | Main inputs | Outputs |
|---|---|---|---|
| `scripts/audit_eval_sets.py` | Audit eval labels for likely issues | data root, doc IDs, `--out-csv`, `--out-md` | CSV + markdown |
| `scripts/export_query_inventory.py` | Export flat query inventory | input root/glob, `--output-csv` | CSV |
| `scripts/export_query_inventory_for_qa.py` | QA-oriented query inventory export | input root/glob, `--output-csv` | CSV |
| `scripts/export_fp2_triage_csv.py` | Build FP2 triage CSV template | input CSV, output CSV | CSV |
| `scripts/export_fp6_to_fp4_audit.py` | Audit FP6 -> FP4 transitions after generation | baseline/candidate/comparison CSVs, `--out-csv` | CSV |
| `scripts/export_live_fp2_audit.py` | Export live FP2 missed-top-rank cases | data root, model path, output files | CSV + metadata JSON |
| `scripts/compare_fp2_dense_bias.py` | Compare FP2 cases before/after dense-biased rerun | baseline CSV, `--out-csv`, `--out-meta` | CSV + JSON |
| `scripts/summarize_fp2_dense_bias.py` | Compact summaries from FP2 dense-bias table | input CSV, recovered/remaining outputs | CSVs |
| `scripts/compare_fp1_fp7_runs.py` | Compare two FP1-FP7 runs | baseline/candidate dirs, `--out-dir` | comparison outputs |
| `scripts/compare_retrieval_roots.py` | Compare retrieval metrics from two processed roots | base/compare roots, `--out-csv` | CSV |
| `scripts/backfill_table_facts.py` | Rebuild `table_facts.parquet` from structured tables | `--data-dir` | in-place `table_facts.parquet` |
| `scripts/build_assisted_heading_labelset.py` | Seed heading-label review dataset | data root, sampling flags, `--out-csv` | CSV |
| `scripts/resample_heading_uncertain_batch.py` | Resample uncertain heading-label batch | `--in-csv`, `--out-csv` | CSV |

## Embeddings, Maps, And Visualisation

| Script | Purpose | Main inputs | Outputs |
|---|---|---|---|
| `scripts/analyze_vector_era_shift.py` | Compare embedding distributions across eras | data root, era doc lists, `--out-dir` | charts/tables |
| `scripts/plot_embedding_similarity_evidence.py` | Embedding similarity evidence charts | doc ID, data root, `--out-dir` | charts |
| `scripts/plot_embedding_relevance_similarity.py` | Relevant vs non-relevant similarity plots | doc ID, data root, `--out-dir` | charts |
| `scripts/plot_embedding_raw_tiles.py` | Raw embedding heatmap tiles | doc ID, data root, `--out-dir` | charts |
| `scripts/build_grampian_temporal_map.py` | Shared UMAP across Grampian docs | data root, `--out-dir` | CSV/plot artifacts |
| `scripts/export_wizmap_umap.py` | Export searchable UMAP/WizMap artifacts | doc/chunks/embeddings inputs, `--out-dir` | WizMap files |
| `scripts/build_searchable_wizmap_from_projection.py` | Build searchable WizMap files from projection CSV | input CSV, doc ID, out dir | searchable files |
| `scripts/build_grampian_temporal_html.py` | Standalone HTML viewer for temporal UMAP | `--input-csv`, `--out-html` | HTML |
| `scripts/build_grampian_temporal_html_wizstyle.py` | Plotly HTML viewer for temporal map | `--input-csv`, `--out-html` | HTML |
| `scripts/serve_wizmap_local.py` | Local static server for WizMap outputs | `--dir`, `--host`, `--port` | running local server |

## Benchmarking And Validation

| Script | Purpose | Main inputs | Outputs |
|---|---|---|---|
| `scripts/benchmark_search.py` | Benchmark retrieval latency/throughput/memory | mode, query/data/model flags, `--output-json` | JSON |
| `scripts/benchmark_device_latency.py` | CPU vs MPS latency benchmark | devices, mode, data/model flags, `--output-json` | JSON |
| `scripts/benchmark_table_extractors.py` | Compare current table extraction vs Docling | PDF inputs, page limits, `--out-dir` | benchmark outputs |
| `scripts/compare_table_benchmark.py` | Score benchmark results | per-page CSV, `--out-dir` | comparison outputs |
| `scripts/validate_camelot_upgrade.py` | Validate Camelot extraction upgrade | PDFs, metrics/eval inputs, `--output-dir` | validation outputs |

## Figures, Demos, And Small Utilities

| Script | Purpose | Main inputs | Outputs |
|---|---|---|---|
| `scripts/demo_progress.py` | Quick pipeline demo | `--data-dir`, `--page`, `--query-id` | stdout |
| `scripts/run_toc_sectioning_debug.py` | ToC/sectioning debug for one PDF | positional `input_pdf`, `output_dir` | debug directory |
| `scripts/export_table_markdown.py` | Dump table markdown files | `--doc-id`, output flags | markdown files |
| `scripts/generate_doc_pairwise_chart.py` | Pairwise win/loss/tie chart for one doc | `--doc-id`, `--out-dir` | chart |
| `scripts/make_retrieval_example_figure.py` | Thesis retrieval example figure | eval set + summaries + output | PNG |
| `scripts/make_balanced_retrieval_examples_figure.py` | Balanced two-example retrieval figure | eval set + summaries + queries + output | PNG |
| `scripts/make_ragas_chunk_comparison_thesis_chart.py` | Thesis chunk-comparison chart | summary CSV, outputs | PNG + CSV |
| `scripts/make_ragas_thesis_slide.py` | RAGAS comparison slide asset | summaries/charts, `--out-dir` | slide assets |
| `scripts/make_thesis_selection_report.py` | Selection charts/tables from ablation summary | summary CSV, `--out-dir` | report assets |
| `scripts/plot_fp_failure_heatmap.py` | FP1-FP7 failure heatmap | counts CSV, `--output` | PNG |
| `scripts/plot_mrr_delta_histogram_single.py` | Single bootstrap delta histogram | input CSV, cohort, output | PNG |
| `scripts/plot_minilm_cap_ablation_trend.py` | Chunk-size vs MiniLM cap trend plot | summary CSV, `--output` | PNG |
| `scripts/plot_win_loss_tie_chart.py` | Win/loss/tie chart helper | script-local config | figure |
| `scripts/export_win_loss_tie_tables.py` | Win/loss/tie table helper | script-local config | tables |
| `scripts/generate_accessible_pipeline_flowchart.py` | Pipeline flowchart asset generator | script-local config | figure |
| `scripts/generate_march20_numeric_ablation_slide.py` | Numeric ablation slide asset | script-local config | slide asset |
| `scripts/generate_march20_weekly_update_slide.py` | Weekly update slide asset | script-local config | slide asset |
| `scripts/plot_grampian_token_distribution.py` | Token distribution helper | script-local config | figure |
| `scripts/plot_effective_embedding_context_minilm.py` | plotting helper | script-local config | figure |
| `scripts/preprocess_pdf_rag.py` | legacy preprocessing helper | script-local config | legacy outputs |
| `scripts/test_integration.py` | manual integration test helper | script-local config | stdout |
| `scripts/test_rotation_fix.py` | manual rotation-fix test helper | script-local config | stdout |
| `scripts/runtime_env.py` | runtime path/env helper module | imported or direct stdout | env details |
| `scripts/corpus_guard.py` | corpus guard helper | script-local config | stdout |
| `scripts/thesis_provenance.py` | provenance helper | script-local config | stdout |
| `scripts/export_table.py` | legacy table export helper | script-local config | exported table files |

## Shell Launchers

| Script | Purpose | Inputs | Outputs |
|---|---|---|---|
| `scripts/run_api_demo.sh` | Launch API demo | none | running API |
| `scripts/run_streamlit_demo.sh` | Launch Streamlit demo | none | running UI |
| `scripts/build_examiner_submission.sh` | Build examiner submission bundle | script-local env | submission artifacts |
| `scripts/update_thesis_minilm_charts.py` | despite extension, acts like a shell helper in this inventory pass | script-local config | chart updates |

## `submission_examiner/scripts/`

This directory is the examiner-facing frozen subset. The main entrypoints are mirrored versions of:

- `preprocess_hybrid.py`
- `build_index.py`
- `build_global_indexes.py`
- `retrieval_eval_hybrid.py`
- `evaluate_pipeline.py`
- `run_full_pipeline.py`
- `run_current_pipeline_fp1_fp7.py`
- `run_retrieval_ablation.py`
- `run_ragas_eval.py`
- `report_retrieval_metrics.py`
- `paired_bootstrap_retrieval_compare.py`
- `run_mcnemar_hit1_batch.py`
- `setup_thesis_rebuild_freeze.py`
- `check_environment.py`
- `check_pipeline_reproducibility.py`
- `check_thesis_bundle_drift.py`
- `export_thesis_bootstrap_table.py`
- `export_thesis_chunk_ablation_table.py`
- `export_thesis_failure_analysis_bundle.py`
- `export_thesis_mcnemar_table.py`
- `export_thesis_ragas_table.py`
- `export_thesis_rebuild_bundle.py`
- `patch_thesis_from_frozen_bundle.py`
- `runtime_env.py`

For those mirrored scripts, behavior and parameter shapes are substantially the same as the versions under `scripts/`.
