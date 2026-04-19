"""indexer.py — BM25 sparse index từ danh sách chunks."""

from typing import List, Dict

from rank_bm25 import BM25Okapi


def build_bm25_index(chunks: List[Dict]) -> BM25Okapi:
    """
    Xây BM25Okapi index từ danh sách chunks.
    Tokenize đơn giản bằng whitespace split.

    Args:
        chunks: List of {"text": str, ...}

    Returns:
        BM25Okapi instance đã index toàn bộ corpus.
    """
    corpus = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(corpus)
    print(f"BM25 index built: {len(corpus)} documents")
    return bm25
