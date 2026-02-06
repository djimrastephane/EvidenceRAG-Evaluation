from __future__ import annotations

try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None


def get_encoder():
    """Get tiktoken encoder for accurate token counting."""
    if tiktoken is None:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str, enc) -> int:
    """
    Count tokens in text.

    Uses tiktoken if available, otherwise estimates based on word count.
    """
    if enc is None:
        return max(1, int(len(text.split()) / 0.75))
    return len(enc.encode(text))


def chunk_text_by_tokens(
    text: str,
    chunk_tokens: int,
    overlap_tokens: int,
    enc,
) -> list[str]:
    """
    Split text into overlapping chunks by token count.

    Args:
        text: Text to chunk
        chunk_tokens: Target chunk size in tokens
        overlap_tokens: Overlap size in tokens
        enc: Tiktoken encoder (or None for word-based estimation)

    Returns:
        List of text chunks
    """
    text = text.strip()
    if not text:
        return []

    if enc is None:
        # Word-based fallback
        words = text.split()
        words_per_chunk = max(50, int(chunk_tokens * 0.75))
        words_overlap = max(10, int(overlap_tokens * 0.75))
        chunks = []
        start = 0
        while start < len(words):
            end = min(len(words), start + words_per_chunk)
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            if end == len(words):
                break
            start = max(0, end - words_overlap)
        return chunks

    # Token-based chunking
    toks = enc.encode(text)
    chunks = []
    start = 0
    while start < len(toks):
        end = min(len(toks), start + chunk_tokens)
        chunk = enc.decode(toks[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(toks):
            break
        start = max(0, end - overlap_tokens)
    return chunks
