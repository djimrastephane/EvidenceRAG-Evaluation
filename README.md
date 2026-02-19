# RAG_Pipeline_Project

A lightweight, local Retrieval-Augmented Generation (RAG) pipeline for PDF reports. It covers:

- PDF preprocessing (cleaning, header/footer removal, sectioning, chunking)
- Embedding + FAISS index build
- Retrieval evaluation against a labeled question set

The scripts are designed for reproducible, page-accurate retrieval on large reports (e.g., NHS annual reports).

## Project Structure

- `Data/` — raw PDFs (ignored by git)
- `data_processed/` — per-document outputs from preprocessing and indexing (ignored by git)
- `preprocess_hybrid.py` — thin runner for the hybrid preprocessing pipeline
- `scripts/` — core preprocessing, indexing, and evaluation scripts
- `qa/` — validation and QA utilities for preprocessing output
- `figures/` — charts/figures produced by analysis scripts (ignored by git)

## Requirements

Python 3.10+ recommended. Core dependencies by script:

- `scripts/preprocess_pdf_rag.py`: `pymupdf`, `pandas`, `pyarrow` (optional: `tiktoken`)
- `scripts/build_index.py`: `faiss-cpu`, `sentence-transformers`, `pandas`, `pyarrow`, `numpy`
- `scripts/retrieval_eval.py`: `faiss-cpu`, `sentence-transformers`, `pandas`, `pyarrow`, `numpy`
- `scripts/benchmark_table_extractors.py`: `pymupdf`, `pdfplumber`, `pandas` (optional: `docling`, benchmark-only)
- OCR fallback: `pytesseract`, `pdf2image`, system `tesseract`, and `poppler` (for `pdftoppm`)

Example setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pymupdf pandas pyarrow tiktoken faiss-cpu sentence-transformers numpy
```

Core runtime policy:

- The active pipeline does not require Docling.
- Docling is used only for optional A/B benchmarking in `scripts/benchmark_table_extractors.py`.

## Configuration

Key constants to adjust before running each script:

- `scripts/preprocess_pdf_rag.py`
  - Paths: `PDF_PATH`, `DOC_ID`, `OUT_ROOT`
  - Chunking: `CHUNK_SIZE_TOKENS`, `CHUNK_OVERLAP_TOKENS`
  - Header/footer removal: `TOP_STRIP_FRAC`, `BOTTOM_STRIP_FRAC`, `HEADER_FOOTER_REPEAT_FRAC`, `TOP_LINE_K`, `BOT_LINE_K`
  - Heading detection: `HEADING_MAX_CHARS`, `HEADING_MIN_CHARS`, `HEADING_FONT_BOOST_FRAC`
  - Filters: `MIN_CHUNK_WORDS`

- `scripts/build_index.py`
  - Paths: `DATA_DIR`, `CHUNKS_PATH`, `METRICS_PATH`
  - Embeddings/index: `EMBED_MODEL_NAME`, `FAISS_INDEX_NAME`, `EMB_NPY_NAME`, `META_PARQUET_NAME`
  - Retrieval sanity check: `TOPK_DEFAULT`

- `scripts/retrieval_eval.py`
  - Paths: `DATA_DIR`, `INDEX_PATH`, `META_PATH`, `EVAL_SET_PATH`
  - Embeddings: `EMBED_MODEL_NAME`
  - Metrics/output: `K_LIST`, `RESULTS_JSON`, `METRICS_JSON`, `SUMMARY_CSV`

- `scripts/preprocess_hybrid.py`
  - OCR thresholds: `OCR_MIN_ALPHA_RATIO`, `OCR_MIN_DIGIT_RATIO`
- `scripts/report_retrieval_metrics.py`
  - Outputs: `retrieval_report.csv`, `retrieval_queries_report.csv`, `retrieval_failure_summary.csv`
- `scripts/run_full_pipeline.py`
  - Full pipeline: preprocess -> build index -> retrieval eval -> reports

## Quickstart

1) Point to your PDF

Edit `PDF_PATH` and `OUT_ROOT` in `scripts/preprocess_pdf_rag.py`.

2) Preprocess the PDF

```bash
python preprocess_hybrid.py
```

Outputs per document (under `data_processed/<DOC_ID>/`):

- `pages.parquet`
- `sections.parquet`
- `chunks.parquet`
- `metrics.json`
- `qa_report.json`
- `sample_chunks.md`
- `ocr_pages.csv` (pages processed with OCR, if enabled)
- Table chunks include a Markdown rendering of detected tables to preserve structure for retrieval.
- `table_facts.parquet` (canonical row/column/value facts derived from extracted tables)

3) Build embeddings + FAISS index

Edit `DATA_DIR` in `scripts/build_index.py` to point at the folder containing `chunks.parquet`, then run:

```bash
python scripts/build_index.py
```

Outputs:

- `faiss.index`
- `embeddings.npy`
- `chunk_meta.parquet`
- `metrics.json` (updated)

4) Evaluate retrieval

Create an `eval_set.json` in the same `DATA_DIR` used above, then run:

```bash
python scripts/retrieval_eval.py
```

Outputs:

- `retrieval_results.json`
- `retrieval_metrics.json`
- `retrieval_summary.csv`

When `eval_set.json` includes `expected_answer`, `retrieval_results.json` now also includes:

- `answer_correct` (true/false/null when not scored)
- `answer_status` (`correct`, `partial`, `incorrect`, `not_scored`)

And `retrieval_metrics.json` includes an `answer_scoring` block with aggregate answer accuracy.

5) Build reports (including failure summary)

```bash
.venv/bin/python scripts/report_retrieval_metrics.py
```

Outputs:

- `retrieval_report.csv` (metrics by doc/k)
- `retrieval_queries_report.csv` (per-query details)
- `retrieval_failure_summary.csv` (one-row failure counts)

## Failure Taxonomy (Evaluation)

The pipeline tags each query with a single failure type at k=1 to separate retrieval vs generation errors:

Retrieval-stage failures (FP1–FP3):
- `FP1_MISSING_CONTENT` — expected pages are not present in the index.
- `FP2_MISSED_TOP_RANK` — expected pages exist but are not retrieved at k=1.
- `FP3_NOT_IN_CONTEXT` — expected pages are retrieved, but the expected answer is not found in the retrieved context.

Generation-stage failures (FP4–FP7):
- `FP4_NOT_EXTRACTED` — answer appears in context but extraction returns nothing useful.
- `FP5_WRONG_FORMAT` — extracted answer does not match the required type (number/date/list).
- `FP6_INCORRECT_SPECIFICITY` — extracted answer is the wrong value despite the right context.
- `FP7_INCOMPLETE` — extracted answer partially matches the expected answer.

Success cases are labeled `HIT`. The per-query report includes both `failure_type` and `failure_stage`.

## Full Pipeline Runner

Runs preprocess -> build index -> retrieval eval -> reports in one command.

```bash
.venv/bin/python scripts/run_full_pipeline.py \
  --pdf-dir "/path/to/pdfs" \
  --out-root data_processed \
  --model models/all-MiniLM-L6-v2
```

## Notes

- Paths in scripts are currently absolute; update them to match your environment.
- If you see `fitz` import errors, uninstall the `fitz` package and install `pymupdf`.
- The FAISS index uses inner product on L2-normalized vectors to approximate cosine similarity.
- If `scripts/build_index.py` or `scripts/retrieval_eval.py` crashes with a SIGSEGV, run with:
  `OMP_NUM_THREADS=1 FAISS_NO_AVX2=1`.
- Git ignores `Data/`, `data_processed/`, `figures/`, and all `*.pdf` outputs by default.
- OCR requires `tesseract` on PATH; for Homebrew installs this is typically `/opt/homebrew/bin/tesseract`.
- `pdf2image` requires Poppler (`pdftoppm`) on PATH; for Homebrew installs this is typically `/opt/homebrew/bin/pdftoppm`.
- When loading `sentence-transformers/all-MiniLM-L6-v2`, you may see an `UNEXPECTED embeddings.position_ids` warning; it is harmless and can be ignored.

## Example eval_set.json

```json
[
  {
    "query_id": "Q001",
    "question": "What is the reporting period end date?",
    "expected_pages": [1],
    "answer_type": "date"
  },
  {
    "query_id": "Q002",
    "question": "What is the total staff costs figure?",
    "expected_pages": [120, 121],
    "answer_type": "number"
  }
]
```

## OCR Setup (Optional)

If you want OCR fallback for image-based pages, install the Python deps and system binaries:

```bash
pip install pytesseract pdf2image
brew install tesseract poppler
```

`ocr_pages.csv` columns:

- `page` — page number (1-based)
- `extractor_notes` — OCR usage reason
- `ocr_text_len` — raw OCR text length
- `clean_text_len` — final normalized text length

OCR behavior:

- Trigger: `clean_text` length < 50 characters.
- Accept: OCR result is used if normalized OCR text length >= 50.
- Tracking: pages using OCR are tagged with `extractor=ocr`.
- Debug: set `OCR_DEBUG=1` to print OCR errors during processing.

OCR metrics (metrics.json):

- `counts.ocr_raw_pages_detected` — OCR attempted in raw extraction stage
- `counts.ocr_raw_pages_accepted` — raw OCR accepted and used in extraction
- `counts.ocr_short_pages_triggered` — clean_text < 50 triggered OCR
- `counts.ocr_short_pages_accepted` — clean_text OCR accepted
- `derived.ocr_raw_acceptance_rate` — accepted / detected (raw OCR)
- `derived.ocr_short_acceptance_rate` — accepted / triggered (short-page OCR)
- `counts.sections_detected` — total sections inferred

Provenance fields (metrics.json):

- `run_utc` — ISO-8601 UTC timestamp for the run
- `git_commit_short` — short git commit hash, if available
- `embedding_model` — model name/path if provided in environment

Derived fields (metrics.json):

- `derived.chunks_per_page`
- `derived.tables_per_100_pages`

CLI + env options:

- `scripts/preprocess_hybrid.py`
  - CLI: `--pdf-path`, `--out-root`
  - Env: `PDF_PATH`, `OUT_ROOT`
- `scripts/build_index.py`
  - CLI: `--data-dir`, `--model`
  - Env: `DATA_DIR`, `EMBED_MODEL_NAME`
- `scripts/retrieval_eval.py`
  - CLI: `--data-dir`, `--model`, `--k-list`
  - Env: `DATA_DIR`, `EMBED_MODEL_NAME`, `K_LIST`

## Table Extractor A/B Benchmark

Benchmark the current extractor (Camelot/pdfplumber path) against Docling on detected table-like pages.

```bash
.venv/bin/python scripts/benchmark_table_extractors.py \
  --pdf-path "Data/Annual Accounts NHS Grampian/Preliminary_Test/Grampian-2022-2023.pdf" \
  --max-table-pages 12
```

Outputs are written to `data_processed/benchmarks/`:

- `table_extract_benchmark_per_page_<timestamp>.csv`
- `table_extract_benchmark_summary_<timestamp>.csv`
- `table_extract_benchmark_run_<timestamp>.json`

Notes:

- Docling is benchmark-only and not part of the core pipeline/runtime requirements.
- If Docling is not installed, the script still runs and records `docling_not_available`.
- To enable Docling comparison only for this benchmark, install it in your environment before running:

```bash
pip install docling
```

## Table Facts Backfill

For existing processed folders that already have `tables_structured.parquet`, you can generate canonical facts without rerunning full preprocessing:

```bash
.venv/bin/python scripts/backfill_table_facts.py --data-dir data_processed/Grampian-2024-2025
```

## Question Router (Modular QA)

`scripts/retrieval_eval.py` now uses an intent router to dispatch extraction:

- Router module: `src/rag_pdf/question_router.py`
- Current routed families:
  - `table_metric_*` (uses `table_facts.parquet` first)
    - includes milestone metrics, `staff_costs`, and `emissions` intents
  - governance intents (legacy regex path)
  - `unknown` fallback

To add new question classes later:

1. Add a new intent in `route_question(...)` in `src/rag_pdf/question_router.py`.
2. Add/extend extraction logic in `scripts/retrieval_eval.py` for that intent.
3. Re-run eval and inspect `route_intent` / `route_confidence` in `retrieval_results.json`.

## Retrieval Tuning / A-B Ablation

Use the ablation runner to compare:

- top-k settings
- chunking settings (optional rebuild mode)
- lexical/table rerank weights
- deterministic query rewrites

Config file:

- `configs/retrieval_tuning.yaml`

Run all configured experiments:

```bash
.venv/bin/python scripts/run_retrieval_ablation.py --config configs/retrieval_tuning.yaml
```

Run only selected experiments:

```bash
.venv/bin/python scripts/run_retrieval_ablation.py \
  --config configs/retrieval_tuning.yaml \
  --only baseline_current,baseline_rerank,rewrite_rerank
```

Outputs:

- `data_processed/ablation/retrieval_ablation_summary.csv`
- `data_processed/ablation/retrieval_ablation_best_by_k.csv`
- `data_processed/ablation/retrieval_ablation_summary.json`

Optional markdown report:

```bash
.venv/bin/python scripts/report_retrieval_ablation.py \
  --summary-csv data_processed/ablation/retrieval_ablation_summary.csv
```

## Batch Processing

Use the batch runner with a JSON config to process a folder of PDFs.

Example config: `config/batch.json`

```json
{
  "pdf_dir": "/path/to/pdfs",
  "pdf_glob": "*.pdf",
  "out_root": "/path/to/output_root",
  "embed_model_name": "/path/to/models/all-MiniLM-L6-v2"
}
```

Run:

```bash
.venv/bin/python scripts/run_batch.py --config config/batch.json
```

Batch runner flags + outputs:

- `--force` to reprocess even if outputs already exist
- Logs per PDF: `<out_root>/<DOC_ID>/preprocess.log`
- Summary CSV: `<out_root>/batch_summary.csv`

## Minimal Retrieval UI

A minimal upload/search interface is available in:

- API: `app/api/main.py`
- Streamlit UI: `app/ui/streamlit_app.py`
- Setup/run guide: `README_UI.md`

## License

MIT License (see `LICENSE`).
