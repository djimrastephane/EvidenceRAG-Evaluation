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