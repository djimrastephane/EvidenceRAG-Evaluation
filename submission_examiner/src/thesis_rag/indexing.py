from __future__ import annotations

"""Dense index construction and persistence.

Exact FAISS search is used for deterministic thesis runs, so this module keeps
index creation intentionally simple: build a flat inner-product index, verify
cardinality, and persist both vectors and metadata needed to reconstruct the
retrieval stage later.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .schemas import ChunkRecord, FaissConfig

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    import faiss


def build_faiss_index(vectors: np.ndarray, config: FaissConfig) -> "faiss.Index":
    """Build the configured exact FAISS index and validate vector count."""
    import faiss

    if config.index_type != "IndexFlatIP":
        raise ValueError("Only exact IndexFlatIP is supported for deterministic thesis runs.")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    if index.ntotal != len(vectors):
        raise ValueError(f"FAISS index size {index.ntotal} does not match vector count {len(vectors)}")
    return index


def save_faiss_index(index: "faiss.Index", out_path: Path) -> None:
    """Persist a FAISS index to disk."""
    import faiss

    out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_path))


def save_embeddings(vectors: np.ndarray, out_path: Path) -> None:
    """Persist the dense embedding matrix as a NumPy array."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, vectors)


def save_chunk_metadata(chunks: list[ChunkRecord], out_path: Path) -> None:
    """Save chunk metadata used to map vector hits back to pages and text."""
    frame = pd.DataFrame([chunk.to_dict() for chunk in chunks])
    frame.to_parquet(out_path, index=False)
    if frame["chunk_id"].duplicated().any():
        duplicates = frame.loc[frame["chunk_id"].duplicated(), "chunk_id"].tolist()
        raise ValueError(f"Duplicate chunk ids in metadata: {duplicates[:5]}")
    LOGGER.info("Saved chunk metadata for %s chunks", len(frame))
