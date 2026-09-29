"""Write or verify the SHA-256 manifest of the frozen gold-evaluation retrieval artifacts.

The reported retrieval tables must be regenerable from committed artifacts. This script
records the hash of every retrieval input (faiss.index, chunk_meta.parquet, chunks.parquet,
eval_set.json) for the gold-labelled documents, and verifies the working tree against it.

Usage:
  python scripts/freeze_eval_artifacts.py --verify           # exit 1 on any drift
  python scripts/freeze_eval_artifacts.py --write            # re-freeze (intended changes only)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "results" / "frozen" / "gold_eval_artifacts_manifest.json"
GOLD_DOCS = [f"Grampian-{y}-{y + 1}" for y in range(2020, 2025)]
ARTIFACTS = ["faiss.index", "chunk_meta.parquet", "chunks.parquet", "eval_set.json"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="Record current artifact hashes in the manifest.")
    mode.add_argument("--verify", action="store_true", help="Check current artifacts against the manifest.")
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data_processed"))
    parser.add_argument("--docs", default=",".join(GOLD_DOCS), help="Comma-separated doc ids.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    return parser.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def describe_doc(doc_dir: Path) -> dict:
    missing = [name for name in ARTIFACTS if not (doc_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"{doc_dir.name}: missing {', '.join(missing)}")
    return {
        "sha256": {name: sha256(doc_dir / name) for name in ARTIFACTS},
        "num_chunks": int(len(pd.read_parquet(doc_dir / "chunk_meta.parquet"))),
    }


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    manifest_path = Path(args.manifest)
    docs = [d.strip() for d in args.docs.split(",") if d.strip()]
    current = {doc: describe_doc(data_root / doc) for doc in docs}

    if args.write:
        manifest = {
            "description": "Frozen retrieval inputs for the gold-labelled evaluation documents.",
            "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "embedding_model": "models/all-MiniLM-L6-v2",
            "artifacts": ARTIFACTS,
            "documents": current,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {manifest_path} ({len(docs)} documents)")
        return 0

    frozen = json.loads(manifest_path.read_text(encoding="utf-8")).get("documents", {})
    drift = []
    for doc, info in current.items():
        expected = (frozen.get(doc) or {}).get("sha256")
        if expected is None:
            drift.append(f"{doc}: not in manifest")
            continue
        drift.extend(f"{doc}/{name}" for name in ARTIFACTS if expected.get(name) != info["sha256"][name])
    if drift:
        print("Artifact drift against frozen manifest:\n  " + "\n  ".join(drift))
        return 1
    print(f"OK: {len(current)} documents match {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
