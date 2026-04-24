#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-"$REPO_ROOT/submission_examiner"}"

case "$OUT_DIR" in
  /*) ;;
  *) OUT_DIR="$REPO_ROOT/$OUT_DIR" ;;
esac

if [[ "$OUT_DIR" == "/" || "$OUT_DIR" == "$REPO_ROOT" ]]; then
  echo "Refusing to write to unsafe output directory: $OUT_DIR" >&2
  exit 1
fi

FILES=(
  # Top-level
  "LICENSE"
  "README.md"
  "environment.yml"
  "requirements.txt"
  "preprocess_hybrid.py"

  # Documentation
  "docs/EXAMINER_QUICKSTART.md"
  "docs/EVALUATION_PROTOCOL.md"
  "docs/thesis_rebuild_freeze_workflow.md"
  "docs/EXAMINER_SUBMISSION_MANIFEST.md"

  # Environment and preflight checks
  "scripts/check_examiner_path.py"
  "scripts/check_environment.py"
  "scripts/check_retrieval_parity.py"
  "scripts/check_retrieval_parity_batch.py"
  "scripts/audit_evaluation_protocol.py"
  "scripts/runtime_env.py"

  # Core pipeline scripts
  "scripts/preprocess_hybrid.py"
  "scripts/build_index.py"
  "scripts/build_global_indexes.py"
  "scripts/evaluate_pipeline.py"
  "scripts/retrieval_eval_hybrid.py"
  "scripts/report_retrieval_metrics.py"
  "scripts/run_full_pipeline.py"

  # Reproducibility and provenance
  "scripts/check_pipeline_reproducibility.py"
  "scripts/check_thesis_export_provenance.py"
  "scripts/check_thesis_bundle_drift.py"

  # Thesis rebuild / frozen-bundle workflow
  "scripts/setup_thesis_rebuild_freeze.py"
  "scripts/run_retrieval_ablation.py"
  "scripts/export_thesis_chunk_ablation_table.py"
  "scripts/export_thesis_failure_analysis_bundle.py"
  "scripts/export_thesis_bootstrap_table.py"
  "scripts/export_thesis_mcnemar_table.py"
  "scripts/export_thesis_ragas_table.py"
  "scripts/export_thesis_rebuild_bundle.py"
  "scripts/backfill_thesis_export_manifests.py"
  "scripts/patch_thesis_from_frozen_bundle.py"

  # Statistical comparison scripts
  "scripts/run_current_pipeline_fp1_fp7.py"
  "scripts/paired_bootstrap_retrieval_compare.py"
  "scripts/run_mcnemar_hit1_batch.py"
  "scripts/run_ragas_eval.py"

  # Thesis reproduction scripts (regenerate thesis tables/figures)
  "scripts/reproduce_table_2_1_thesis_rag.py"
  "scripts/reproduce_table_4_5_doc_vs_global_thesis_rag.py"
  "scripts/reproduce_table_c6_bm25_tokenizer.py"
  "scripts/reproduce_figure_4_1_thesis_rag.py"
  "scripts/reproduce_figure_4_2_thesis_rag.py"
  "scripts/reproduce_figure_4_2_postfix.py"

  # QA checks
  "qa/qa_check_preprocessing.py"
  "qa/validate_hybrid_pipeline.py"
  "qa/validate_text_extraction_hybrid.py"
  "qa/check_empty_pages.py"

  # Configs: main pipeline config + tuning configs used in ablations
  "configs/README.md"
  "configs/thesis_rag.yaml"
  "configs/retrieval_tuning_minilm_cap_5docs.yaml"
  "configs/retrieval_tuning_thesis_5docs_q50.yaml"
  "configs/retrieval_tuning_224_56_5docs.yaml"

  # Evaluation sets (5-document Grampian corpus)
  "data_processed/Grampian-2020-2021/eval_set.json"
  "data_processed/Grampian-2021-2022/eval_set.json"
  "data_processed/Grampian-2022-2023/eval_set.json"
  "data_processed/Grampian-2023-2024/eval_set.json"
  "data_processed/Grampian-2024-2025/eval_set.json"

  # Reproducibility evidence
  "results/reproducibility/grampian_5docs_repro.json"
  "results/reproducibility/retrieval_parity_batch_smoke.json"
  "results/reproducibility/evaluation_protocol_audit.json"
  "results/reproducibility/evaluation_protocol_audit.md"
  "results/reproducibility/current_pipeline_grampian_5docs_repro_2026-04-17.json"

  # thesis_rag canonical results (rrf_k=20, dense_weight=0.5, bm25_weight=2.0)
  # Chunk ablation table
  "results/thesis_ablations/chunk_size_ablation_2026-04-15/tables/chunk_ablation_table.json"
  "results/thesis_ablations/chunk_size_ablation_2026-04-15/tables/chunk_ablation_table.csv"
  "results/thesis_ablations/chunk_size_ablation_2026-04-15/tables/chunk_ablation_by_document.csv"
  "results/thesis_ablations/chunk_size_ablation_2026-04-15/manifests/ablation_spec.json"
  # Bootstrap: hybrid vs dense, 5 cohorts
  "results/bootstrap_postfix_2026-04-20/paired_bootstrap_summary_all.json"
  # McNemar: hybrid vs dense Hit@1, 5 cohorts — summary + per-doc per-query source files
  "results/mcnemar_thesis_rag_postfix_2026-04-20/mcnemar_hit1_summary_table.json"
  "results/mcnemar_thesis_rag_postfix_2026-04-20/mcnemar_hit1_summary_table.csv"
  # Per-doc hybrid vs dense per-query hits (224/56) — lets examiner verify McNemar computation
  "results/thesis_ablations/post_fix_rerun_2026-04-19/pipeline_outputs/minilmcap_Grampian-2020-2021_chunk_224_56/Grampian-2020-2021/hybrid_page_hits.jsonl"
  "results/thesis_ablations/post_fix_rerun_2026-04-19/pipeline_outputs/minilmcap_Grampian-2020-2021_chunk_224_56/Grampian-2020-2021/dense_page_hits.jsonl"
  "results/thesis_ablations/post_fix_rerun_2026-04-19/pipeline_outputs/minilmcap_Grampian-2020-2021_chunk_224_56/Grampian-2020-2021/retrieval_metrics.json"
  "results/thesis_ablations/post_fix_rerun_2026-04-19/pipeline_outputs/minilmcap_Grampian-2021-2022_chunk_224_56/Grampian-2021-2022/hybrid_page_hits.jsonl"
  "results/thesis_ablations/post_fix_rerun_2026-04-19/pipeline_outputs/minilmcap_Grampian-2021-2022_chunk_224_56/Grampian-2021-2022/dense_page_hits.jsonl"
  "results/thesis_ablations/post_fix_rerun_2026-04-19/pipeline_outputs/minilmcap_Grampian-2021-2022_chunk_224_56/Grampian-2021-2022/retrieval_metrics.json"
  "results/thesis_ablations/post_fix_rerun_2026-04-19/pipeline_outputs/minilmcap_Grampian-2022-2023_chunk_224_56/Grampian-2022-2023/hybrid_page_hits.jsonl"
  "results/thesis_ablations/post_fix_rerun_2026-04-19/pipeline_outputs/minilmcap_Grampian-2022-2023_chunk_224_56/Grampian-2022-2023/dense_page_hits.jsonl"
  "results/thesis_ablations/post_fix_rerun_2026-04-19/pipeline_outputs/minilmcap_Grampian-2022-2023_chunk_224_56/Grampian-2022-2023/retrieval_metrics.json"
  "results/thesis_ablations/post_fix_rerun_2026-04-19/pipeline_outputs/minilmcap_Grampian-2023-2024_chunk_224_56/Grampian-2023-2024/hybrid_page_hits.jsonl"
  "results/thesis_ablations/post_fix_rerun_2026-04-19/pipeline_outputs/minilmcap_Grampian-2023-2024_chunk_224_56/Grampian-2023-2024/dense_page_hits.jsonl"
  "results/thesis_ablations/post_fix_rerun_2026-04-19/pipeline_outputs/minilmcap_Grampian-2023-2024_chunk_224_56/Grampian-2023-2024/retrieval_metrics.json"
  "results/thesis_ablations/post_fix_rerun_2026-04-19/pipeline_outputs/minilmcap_Grampian-2024-2025_chunk_224_56/Grampian-2024-2025/hybrid_page_hits.jsonl"
  "results/thesis_ablations/post_fix_rerun_2026-04-19/pipeline_outputs/minilmcap_Grampian-2024-2025_chunk_224_56/Grampian-2024-2025/dense_page_hits.jsonl"
  "results/thesis_ablations/post_fix_rerun_2026-04-19/pipeline_outputs/minilmcap_Grampian-2024-2025_chunk_224_56/Grampian-2024-2025/retrieval_metrics.json"
  # RAGAS: context precision and recall (top_k=5, boost off)
  "results/ragas_full_eval_boost_off_2026-04-20/scores.json"

  # thesis_rag FP1-FP7 failure analysis (canonical — uses rrf_k=20, boost off)
  "results/fp1_fp7_retrieval_boost_off_2026-04-20/current_pipeline_fp1_fp7_summary.json"
  "results/fp1_fp7_retrieval_boost_off_2026-04-20/current_pipeline_fp1_fp7_counts.csv"
  "results/fp1_fp7_retrieval_boost_off_2026-04-20/current_pipeline_fp1_fp7_per_query.csv"
  "results/fp1_fp7_llm_boost_off_2026-04-20/current_pipeline_fp1_fp7_summary.json"
  "results/fp1_fp7_llm_boost_off_2026-04-20/current_pipeline_fp1_fp7_counts.csv"
  "results/fp1_fp7_llm_boost_off_2026-04-20/current_pipeline_fp1_fp7_per_query.csv"
  # Figure D.1 — side-by-side normalised heatmap (retrieval vs LLM)
  "results/figure_d1_fp1_fp7_postfix_2026-04-20/fp1_fp7_heatmaps_side_by_side_norm_labeled.png"

  # Legacy frozen bundle provenance guardrails only (failure analysis superseded by thesis_rag runs above)
  "results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/guardrails/bundle_provenance_report.csv"
  "results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/guardrails/bundle_provenance_report.json"
  "results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/guardrails/bundle_provenance_report.md"
)

DIRS=(
  # Entire src tree: includes rag_pdf (original pipeline), thesis_rag (refactored module), generation
  "src"
)

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

copy_file() {
  local rel="$1"
  local src="$REPO_ROOT/$rel"
  local dst="$OUT_DIR/$rel"
  if [[ ! -f "$src" ]]; then
    echo "Missing required file: $rel" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
}

copy_dir() {
  local rel="$1"
  local src="$REPO_ROOT/$rel"
  local dst="$OUT_DIR/$rel"
  if [[ ! -d "$src" ]]; then
    echo "Missing required directory: $rel" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$dst")"
  cp -R "$src" "$dst"
}

for rel in "${FILES[@]}"; do
  copy_file "$rel"
done

for rel in "${DIRS[@]}"; do
  copy_dir "$rel"
done

find "$OUT_DIR" -name '.DS_Store' -delete
find "$OUT_DIR" -name '__pycache__' -type d -prune -exec rm -rf {} +

echo "Examiner submission package created at:"
echo "  $OUT_DIR"
