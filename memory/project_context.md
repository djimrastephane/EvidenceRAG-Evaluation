---
name: Project context
description: Key facts about the RAG pipeline thesis project — data dirs, pipeline versions, evaluation results, thesis values
type: project
---

## Canonical thesis results (2026-04-24 rerun: frozen artifacts + current eval_set)

All thesis numbers were updated on 2026-04-24 to use frozen 224/56 pipeline artifacts + current `data_processed/{doc}/eval_set.json`. This is the definitive baseline for all thesis comparisons.

**Why:** Multiple baselines existed (frozen eval_set, current eval_set, data_processed pipeline). Frozen artifacts + current eval_set is most defensible: one test set throughout, best preprocessing pipeline.

**Table 4.1 values** (hybrid base, boost=OFF, 250 queries):
- Dense H@1=0.768, H@3=0.892, MRR=0.8387
- BM25  H@1=0.704, H@3=0.840, MRR=0.7843
- Hybrid H@1=0.740, H@3=0.880, MRR=0.8122
- Hybrid+boost H@1=0.828, H@3=0.956, MRR=0.8894

**Per-doc hybrid base H@1**: 2020-21=0.760, 2021-22=0.780, 2022-23=0.620, 2023-24=0.820, 2024-25=0.720
FP2 counts: hybrid base fp2=48 (19.2%), fp3=17; boost fp2=37 (14.8%), fp3=6; reduction=4.4 pp, 22 queries resolved at rank 1.

**Chunk-vs-page** (from rerun_chunk_hits_2026-04-24):
- Dense: Chunk H@1=0.620, Page H@1=0.768 (gap=+0.148)
- BM25:  Chunk H@1=0.696, Page H@1=0.704 (gap=+0.008)
- Hybrid: Chunk H@1=0.732, Page H@1=0.740 (gap=+0.008)
- Hybrid+boost: Chunk H@1=0.820, Page H@1=0.828 (gap=+0.008)

**Chunk ablation** (from rerun_chunk_ablation_2026-04-24):
224/56=0.740/0.812, 256/64=0.748/0.817, 280/90=0.744/0.816, 400/100=0.720/0.799

**BM25 grid** (promoted k1=1.5, b=0.75 ranks 10th, MRR=0.7827; best MRR=0.7913):
Best: k1=1.2, b=0.75. Gap from best = 0.009 MRR.

**Doc-vs-global** (from rerun_doc_vs_global_2026-04-24):
boost-OFF constrained H@1=0.740, MRR=0.812; boost-ON H@1=0.828, MRR=0.889; global-ON H@1=0.544, MRR=0.680
Scope reduction: -34.3% H@1, -23.5% MRR.

**CE ablation** (from rerun_ce_ablation_2026-04-24, live retrieval, baseline=0.732):
Best: CE top-20 w=0.3 → H@1=0.748 (+0.016). Small gain because frozen pipeline already applies lexical reranking.

## Frozen artifact directories

- Boost-OFF: `results/thesis_ablations/chunk_size_ablation_boost_off_2026-04-20/pipeline_outputs/minilmcap_{doc}_chunk_224_56/{doc}/`
- Boost-ON:  `results/thesis_ablations/chunk_size_ablation_2026-04-15/pipeline_outputs/minilmcap_{doc}_chunk_224_56/{doc}/`
- Current eval_set: `data_processed/{doc}/eval_set.json`

## Re-run scripts (all in scripts/)

- `rerun_main_tables_frozen_eval.py` → `results/rerun_main_tables_2026-04-24/results.json`
- `rerun_chunk_ablation_frozen_eval.py` → `results/rerun_chunk_ablation_2026-04-24/results.json`
- `rerun_bm25_grid_frozen_eval.py` → `results/rerun_bm25_grid_2026-04-24/results.json`
- `rerun_doc_vs_global_frozen_eval.py` → `results/rerun_doc_vs_global_2026-04-24/results.json`
- `rerun_ce_ablation_frozen_eval.py` → `results/rerun_ce_ablation_2026-04-24/results.json`
- `compute_chunk_hits_frozen.py` → `results/rerun_chunk_hits_2026-04-24/results.json`

## Thesis files updated on 2026-04-24

- `chapters/results.tex` — Table 4.1, per-doc, per-difficulty, boost delta table, FP2/FP3 counts, inline text
- `chapters/introduction.tex` — Contributions section metrics
- `chapters/discussion.tex` — CE ablation table + narrative, cross-doc %, limitations section
- `appendix.tex` — chunk-vs-page, doc-vs-global, chunk ablation, BM25 grid, encoder table captions, FP2 count
- `figures/chunk_ablation_table.tex` — chunk ablation deltas
- `chapters/conclusion.tex` — Fixed factual error (BM25→Dense highest H@1) in previous session

## Data directory map

- `data_processed/` — Current eval_sets (authoritative test set post-2026-04-24)
- `results/thesis_ablations/` — Frozen pipeline artifacts (pre-computed embeddings, FAISS indices, hit CSVs)

## Key files

- Thesis: `/Users/djimra/MSc Data Science Jan 2025/Thesis documents/Thesis/University_of_Aberdeen_thesis_template/abdnthesis.tex`
- Chapters: `chapters/results.tex`, `chapters/discussion.tex`, `chapters/introduction.tex`, `appendix.tex`
- Pipeline source: `src/thesis_rag/` (thesis_rag module)

**How to apply:** All thesis numbers now consistent. If adding new results, compare against frozen-artifact+current-eval baseline (hybrid H@1=0.740, Dense H@1=0.768).
