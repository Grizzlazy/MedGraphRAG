"""retriever.py — Dense retrieval (FAISS) hoặc Hybrid (FAISS+BM25) tuỳ config."""

from typing import List, Dict, Optional

import numpy as np
import faiss

from .config import DENSE_TOPK, SPARSE_TOPK, RRF_K, USE_BM25
from .embedder import get_embed_client, get_embedding


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank)


def dense_retrieve(
    query: str,
    faiss_index: faiss.Index,
    chunks: List[Dict],
    dense_topk: int = DENSE_TOPK,
) -> List[Dict]:
    """Retrieve bằng FAISS cosine similarity."""
    client = get_embed_client()
    query_vec = get_embedding(query, client).reshape(1, -1)
    scores, indices = faiss_index.search(query_vec, dense_topk)

    return [
        {
            "chunk":       chunks[idx],
            "rrf_score":   float(score),
            "dense_score": float(score),
            "dense_rank":  rank,
            "sparse_rank": -1,
        }
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1)
    ]


def hybrid_retrieve(
    query: str,
    faiss_index: faiss.Index,
    chunks: List[Dict],
    bm25,
    dense_topk: int = DENSE_TOPK,
    sparse_topk: int = SPARSE_TOPK,
) -> List[Dict]:
    """Retrieve bằng FAISS + BM25, kết hợp bằng RRF."""
    client = get_embed_client()

    # Dense
    query_vec = get_embedding(query, client).reshape(1, -1)
    _, dense_indices = faiss_index.search(query_vec, dense_topk)
    dense_indices = dense_indices[0].tolist()

    # Sparse
    bm25_scores   = bm25.get_scores(query.lower().split())
    sparse_indices = np.argsort(bm25_scores)[::-1][:sparse_topk].tolist()

    # RRF fusion
    rrf: Dict[int, float] = {}
    dense_rank_map:  Dict[int, int] = {}
    sparse_rank_map: Dict[int, int] = {}

    for rank, idx in enumerate(dense_indices, start=1):
        rrf[idx] = rrf.get(idx, 0.0) + _rrf_score(rank)
        dense_rank_map[idx] = rank

    for rank, idx in enumerate(sparse_indices, start=1):
        rrf[idx] = rrf.get(idx, 0.0) + _rrf_score(rank)
        sparse_rank_map[idx] = rank

    merged = sorted(rrf.items(), key=lambda x: x[1], reverse=True)

    return [
        {
            "chunk":       chunks[idx],
            "rrf_score":   score,
            "dense_score": 0.0,
            "dense_rank":  dense_rank_map.get(idx, -1),
            "sparse_rank": sparse_rank_map.get(idx, -1),
        }
        for idx, score in merged
    ]


def retrieve(
    query: str,
    faiss_index: faiss.Index,
    chunks: List[Dict],
    bm25=None,
) -> List[Dict]:
    """
    Entry point duy nhất — tự chọn dense hoặc hybrid theo config.USE_BM25.
    """
    if USE_BM25 and bm25 is not None:
        return hybrid_retrieve(query, faiss_index, chunks, bm25)
    return dense_retrieve(query, faiss_index, chunks)
