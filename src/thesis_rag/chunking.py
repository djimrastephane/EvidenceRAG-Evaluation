from __future__ import annotations

"""Token-count based chunking utilities.

The thesis pipeline uses explicit token budgeting rather than character counts
so chunk-size ablations map cleanly onto the embedding model context. These
helpers keep token counting and overlapping chunk creation deterministic across
platforms.
"""

from dataclasses import dataclass

try:
    import tiktoken
except ImportError:
    tiktoken = None


@dataclass(slots=True)
class TokenChunk:
    """A text span paired with its token count after chunk construction."""
    text: str
    token_count: int


def get_encoder():
    """Return the deterministic tokenizer used for chunk-size accounting."""
    if tiktoken is None:
        raise RuntimeError("tiktoken is required for deterministic token chunking.")
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, encoder) -> int:
    """Count tokens in text with the configured encoder."""
    return len(encoder.encode(text))


def chunk_text(text: str, chunk_size: int, overlap: int, encoder) -> list[TokenChunk]:
    """Split text into overlapping token chunks while preserving order."""
    stripped = text.strip()
    if not stripped:
        return []
    if overlap >= chunk_size:
        raise ValueError("chunk overlap must be smaller than chunk size")
    tokens = encoder.encode(stripped)
    chunks: list[TokenChunk] = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + chunk_size)
        chunk_text_value = encoder.decode(tokens[start:end]).strip()
        if chunk_text_value:
            chunks.append(TokenChunk(text=chunk_text_value, token_count=end - start))
        if end == len(tokens):
            break
        start = end - overlap
    return chunks
