---
name: project-status
description: Current state of the RAG pipeline project and recent work completed
metadata:
  type: project
---

Thesis submitted. Now productionising and documenting the RAG pipeline on GitHub.

**Why:** Post-submission phase — making the repo presentable and fully documented for public reference.

**How to apply:** Treat the pipeline as feature-complete; focus on documentation, issue tracking, and README quality rather than new features.

## Recent session work (2026-06-16)

- Fixed `top1_cosine_est=0` bug in Streamlit ranking margin panel (was returning `0.0` for BM25-only hits outside dense window; now returns `None`)
- Created and closed 22 retrospective research issues on GitHub covering all major pipeline design decisions:
  - #5 chunk size/overlap, #6 dense vs BM25 vs hybrid, #7 RRF ablation, #8 BM25 ablation, #9 token distribution
  - #10 table extraction/chunking, #11 embedding model selection, #12 re-ranking strategy
  - #13 query rewriting, #14 eval set construction, #15 RAGAS/generation quality
  - #16 OCR/rotation/mixed-routing validation, #17 Wizmap/UMAP embedding visualisation
  - #18 section/heading detection, #19 region classification/page routing, #20 text normalisation
  - #21 query routing/intent classification, #22 numeric extraction/normalisation
- README additions: retrieval performance summary table (Hit@1–10, MRR across 5 cohorts), PCA embedding space plot, Limitations section, Future Improvements section
- All issues closed; repo has no open issues

## Recent session work (2026-06-17)

- Fixed `qa/test_preprocessing.py` and `qa/test2_preprocessing.py`: both referenced a never-processed `nhs-england-annual-report-and-accounts-2024-to-2025` doc_id; repointed to the existing `data_processed/Grampian-2024-2025` artifacts (commit `0fe5e25`)
- Documented the `qa/` diagnostic scripts in README's Tests section, noting they are standalone scripts (not pytest cases) and require the doc to be preprocessed first (commit `4fc5953`)
- Verified: full `pytest tests/ qa/` suite passes (32/32), both qa scripts run cleanly standalone, CI and Docker GitHub Actions green on both commits, repo clean and in sync with `origin/main`