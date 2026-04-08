"""
Stage 2 — Text Chunker
Sliding-window chunking with configurable size and overlap.
"""

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def sliding_window_chunk(
    text: str,
    max_tokens: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split *text* into overlapping chunks.

    Each chunk dict contains:
        - chunk_id (int)
        - text (str)
        - start_word (int)  — inclusive word index
        - end_word (int)    — exclusive word index
    """
    words = text.split()
    chunks: list[dict] = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "start_word": start,
            "end_word": end,
        })
        chunk_id += 1

        if end >= len(words):
            break
        start += max_tokens - overlap

    return chunks


def chunk_document(text: str, doc_id: str = "", **kwargs) -> list[dict]:
    """Chunk a document and attach metadata."""
    chunks = sliding_window_chunk(text, **kwargs)
    for c in chunks:
        c["doc_id"] = doc_id
    return chunks
