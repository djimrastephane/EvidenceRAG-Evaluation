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
- OCR fallback: `pytesseract`, `pdf2image`, system `tesseract`, and `poppler` (for `pdftoppm`)

Example setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pymupdf pandas pyarrow tiktoken faiss-cpu sentence-transformers numpy
```

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

## Notes

- Paths in scripts are currently absolute; update them to match your environment.
- If you see `fitz` import errors, uninstall the `fitz` package and install `pymupdf`.
- The FAISS index uses inner product on L2-normalized vectors to approximate cosine similarity.
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

## License

MIT License (see `LICENSE`).
