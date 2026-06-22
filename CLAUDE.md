# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate rag-pipeline
```

Use `environment_py312_smoke.yml` for a lightweight install that skips optional packages.

## Common Commands

```bash
# Run tests
pytest tests/ qa/

# Run a single test file
pytest tests/test_thesis_rag_config_normalization.py -v

# Full preprocessing pipeline (one PDF)
python preprocess_hybrid.py --config configs/thesis_rag.yaml --pdf_path Data/<file>.pdf --doc_id <id>

# Build FAISS index from preprocessed chunks
python scripts/build_index.py --config configs/thesis_rag.yaml

# Evaluate retrieval
python scripts/retrieval_eval.py --config configs/thesis_rag.yaml

# Single query
python scripts/retrieve.py --config configs/thesis_rag.yaml --query "your question"

# Launch Streamlit demo
bash scripts/run_streamlit_demo.sh
```

## Architecture Overview

The pipeline has four sequential stages:

1. **Preprocessing** — Extract, clean, and chunk PDFs
   - PyMuPDF is the primary extractor; PDFPlumber is the per-page fallback for low-quality text
   - Rotation-aware boilerplate stripping (8% edge crop for portrait, 2% for rotated/landscape)
   - Sections and headings are detected to annotate chunks with structural context
   - Tables are extracted separately and injected back as markdown chunks
   - Chunks are token-based (default 224 tokens, 56 overlap) using tiktoken `cl100k_base`

2. **Indexing** (`scripts/build_index.py`) — Embed chunks and build FAISS index
   - Model: `sentence-transformers/all-MiniLM-L6-v2` → 384-dim vectors
   - Optional L2 normalization controlled by `embedding.apply_l2_normalization` in config
   - Outputs: `faiss.index`, `embeddings.npy`, `chunk_meta.parquet`

3. **Retrieval** — Hybrid dense + sparse
   - Dense: FAISS `IndexFlatIP` (inner product after L2 normalization = cosine)
   - Sparse: BM25 (`rank-bm25`)
   - Fusion: Reciprocal Rank Fusion (RRF) with configurable `rrf_k`, `dense_weight`, `sparse_weight`
   - Live code: `src/rag_pdf/retrieval/canonical_hybrid.py`

4. **Evaluation** (`scripts/retrieval_eval.py`) — Recall@k, MRR against `eval_set.json`

## Source Layout

```
src/rag_pdf/          # Core library — all pipeline logic lives here
  config.py           # PreprocessConfig, TableDetectConfig, RegionConfig dataclasses
  extract_page.py     # Per-page text extraction (PyMuPDF + PDFPlumber fallback)
  boilerplate.py      # Coordinate-based header/footer removal
  sections.py         # Section/subsection detection and canonicalization
  headings.py         # Font-size-based heading classification
  chunking.py         # Token-aware overlapping chunk splitting
  table_detect.py     # Table boundary detection
  table_extract.py    # Table content extraction with coordinate chunking
  table_camelot.py    # Camelot lattice/hybrid extraction mode
  table_chunking.py   # Row-based sharding for large tables
  metrics.py          # StepTimer, safe JSON/CSV writing
  text_normalize.py   # Hyphenation, encoding, whitespace normalization
  retrieval/          # Hybrid retrieval, query rewriting, reranking
  services/           # Search, storage, numeric extraction, LLM integration

scripts/              # Runnable entrypoints and experiment scripts
  preprocess_pdf_rag.py   # Main preprocessing entrypoint (~1100 lines)
  preprocess_hybrid.py    # Hybrid extraction wrapper
  build_index.py          # Indexing entrypoint
  retrieval_eval.py       # Evaluation entrypoint
  retrieve.py             # Single-query retrieval
  (80+ ablation/analysis/reproduction scripts for thesis experiments)

configs/              # YAML configs per experiment
  thesis_rag.yaml     # Primary config (all pipeline parameters)
  thesis_rag_smoke.yaml

tests/                # Unit tests (pytest)
qa/                   # Preprocessing validation tests
app/ui/               # Streamlit and FastAPI demo apps
```

## Key Config Parameters (`configs/thesis_rag.yaml`)

| Section | Key Parameters |
|---|---|
| `chunking` | `chunk_size_tokens: 224`, `chunk_overlap_tokens: 56`, `min_chunk_words: 20` |
| `embedding` | `model_name`, `apply_l2_normalization: true`, `expected_dimension: 384` |
| `retrieval` | `dense_top_k`, `sparse_top_k`, `hybrid_top_k`; production/eval default overrides to `rrf_k: 20`, `dense_weight: 0.5`, `sparse_weight: 2.0` (see `docs/experiments/HYBRID_RRF_DEFAULTS.md`) |
| `bm25` | `k1: 1.5`, `b: 0.75` |
| `evaluation` | `ks: [1, 3, 5, 10]` |

## Data Artifacts (git-ignored)

- `Data/` — Raw PDFs
- `data_processed/<doc_id>/` — `pages.parquet`, `sections.parquet`, `chunks.parquet`, `metrics.json`, `qa_report.json`
- `indexes/` — `faiss.index`, `embeddings.npy`, `chunk_meta.parquet`
- `runs/` — Timestamped experiment runs (browsable via Streamlit UI)
- `models/` — Cached HuggingFace model weights

## Eval Set Format

```json
[{"question": "...", "expected_pages": [5, 6], "doc_id": "optional"}]
```

Chunk IDs follow the pattern `<doc_id>:<chunk_index>` to support multi-document leakage detection during evaluation.