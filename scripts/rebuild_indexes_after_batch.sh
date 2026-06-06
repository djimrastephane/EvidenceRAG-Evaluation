#!/usr/bin/env bash
set -euo pipefail

LOG="/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/data_processed/rebuild_indexes.log"
PROJ="/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project"

echo "[$(date)] Waiting for preprocess_hybrid.py to finish..." | tee "$LOG"

while pgrep -f "preprocess_hybrid.py" > /dev/null 2>&1; do
    sleep 10
done

echo "[$(date)] Preprocessing complete. Starting index rebuild..." | tee -a "$LOG"

cd "$PROJ"
conda run -n rag-pipeline python scripts/build_index.py \
    --data-dir data_processed \
    --device mps \
    2>&1 | tee -a "$LOG"

echo "[$(date)] All indexes rebuilt." | tee -a "$LOG"
