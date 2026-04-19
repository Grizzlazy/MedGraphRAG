"""reranker.py — Cross-encoder reranking trên top-N candidates từ RRF."""

import threading
from typing import List, Dict

from .config import RERANK_TOPN, CROSS_ENCODER_MODEL

# Singleton với double-checked locking — thread-safe khi dùng ThreadPoolExecutor
_encoder = None
_lock    = threading.Lock()


def _get_encoder():
    global _encoder
    if _encoder is None:            # fast path — không cần lock sau lần đầu
        with _lock:                 # chỉ 1 thread vào được
            if _encoder is None:    # double-check sau khi có lock
                from sentence_transformers import CrossEncoder
                print(f"Loading CrossEncoder: {CROSS_ENCODER_MODEL} (one-time)...")
                _encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _encoder


def warmup():
    """Gọi trước khi spawn ThreadPool để đảm bảo model load trong main thread."""
    _get_encoder()


def rerank(
    query: str,
    candidates: List[Dict],
    top_n: int = RERANK_TOPN,
) -> List[Dict]:
    """
    Rerank top-N candidates bằng sentence-transformers CrossEncoder.

    CrossEncoder nhận cặp (query, passage) và tính relevance score trực tiếp
    — chính xác hơn bi-encoder (FAISS) nhưng chậm hơn, chỉ dùng trên top-N nhỏ.

    Model load 1 lần duy nhất (thread-safe singleton) để tránh overhead mỗi lần gọi.

    Args:
        query:      Câu hỏi gốc.
        candidates: List dicts từ hybrid_retrieve(), sắp xếp theo rrf_score.
        top_n:      Số candidates đưa vào cross-encoder (mặc định RERANK_TOPN).

    Returns:
        List candidates đã rerank, mỗi dict có thêm trường "ce_score".
        Fallback về thứ tự RRF nếu sentence-transformers không cài.
    """
    try:
        encoder = _get_encoder()
    except ImportError:
        print("[WARN] sentence-transformers chưa cài — bỏ qua reranking, "
              "dùng thứ tự RRF.\n"
              "       pip install sentence-transformers")
        return candidates[:top_n]

    pool = candidates[:top_n]
    pairs = [(query, c["chunk"]["text"]) for c in pool]
    ce_scores = encoder.predict(pairs)

    for c, score in zip(pool, ce_scores):
        c["ce_score"] = float(score)

    pool.sort(key=lambda x: x["ce_score"], reverse=True)
    return pool
