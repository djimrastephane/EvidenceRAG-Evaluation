from __future__ import annotations

"""CLI entrypoint for corpus preprocessing.

This script runs the first thesis_rag stage over the configured PDF corpus and
materialises page-level and chunk-level artifacts inside a fresh run directory.
It is intentionally thin so that the actual logic remains in
``thesis_rag.pipeline.preprocess_corpus`` and can be reused from tests or other
automation without depending on Streamlit or notebooks.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from thesis_rag.config import load_config
from thesis_rag.pipeline import preprocess_corpus


def parse_args() -> argparse.Namespace:
    """Parse the YAML configuration path for a preprocessing run."""
    parser = argparse.ArgumentParser(description="Preprocess PDF reports into page and chunk artifacts.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    """Load configuration, run preprocessing, and print the created run path."""
    args = parse_args()
    run_dir = preprocess_corpus(load_config(args.config))
    print(run_dir)


if __name__ == "__main__":
    main()
