from __future__ import annotations

from pathlib import Path


EVAL_READY_REQUIRED_FILES = (
    "eval_set.json",
    "faiss.index",
    "chunk_meta.parquet",
    "chunks.parquet",
    "embeddings.npy",
)


def missing_eval_ready_files(doc_dir: Path) -> list[str]:
    return [name for name in EVAL_READY_REQUIRED_FILES if not (doc_dir / name).exists()]


def is_eval_ready_doc_dir(doc_dir: Path) -> bool:
    return not missing_eval_ready_files(doc_dir)


def list_eval_ready_doc_dirs(
    data_root: Path,
    doc_pattern: str,
) -> tuple[list[Path], list[tuple[Path, list[str]]]]:
    ready: list[Path] = []
    skipped: list[tuple[Path, list[str]]] = []
    for doc_dir in sorted(p for p in data_root.glob(doc_pattern) if p.is_dir()):
        missing = missing_eval_ready_files(doc_dir)
        if missing:
            skipped.append((doc_dir, missing))
            continue
        ready.append(doc_dir)
    return ready, skipped


def print_skipped_eval_ready_docs(skipped: list[tuple[Path, list[str]]]) -> None:
    if not skipped:
        return
    print("Skipping incomplete corpora that are not evaluation-ready:")
    for doc_dir, missing in skipped:
        print(f"- {doc_dir.name}: missing {', '.join(missing)}")
