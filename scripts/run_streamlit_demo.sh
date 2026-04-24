#!/usr/bin/env bash
set -euo pipefail

UI_VARIANT="${1:-current}"

export DEMO_MODE="${DEMO_MODE:-1}"
export UI_DATA_ROOT="${UI_DATA_ROOT:-results/table_rechunk_ablation_2026-03-22/artifacts/row_blocks}"
export UI_TABLE_CHUNKING_MODE="${UI_TABLE_CHUNKING_MODE:-row_blocks}"

case "${UI_VARIANT}" in
  current)
    APP_PATH="app/ui/streamlit_app.py"
    ;;
  legacy)
    APP_PATH="app/ui/streamlit_app_legacy.py"
    ;;
  *)
    echo "Usage: $0 [current|legacy]" >&2
    exit 1
    ;;
esac

python -m streamlit run "${APP_PATH}" --server.port "${STREAMLIT_PORT:-8502}"
