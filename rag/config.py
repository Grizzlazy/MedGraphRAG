"""config.py — Hằng số cấu hình cho toàn bộ RAG pipeline."""

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 512   # tokens mỗi chunk
CHUNK_OVERLAP = 50    # token overlap giữa 2 chunk liên tiếp

# ── Retrieval ─────────────────────────────────────────────────────────────────
USE_BM25    = False  # True: hybrid FAISS+BM25 (chậm, recall cao hơn)
                     # False: chỉ FAISS dense (nhanh, đủ tốt)
DENSE_TOPK  = 20   # số candidates từ FAISS
SPARSE_TOPK = 20   # số candidates từ BM25 (chỉ dùng khi USE_BM25=True)
RRF_K       = 60   # hằng số làm mượt RRF (chuẩn = 60)
RERANK_TOPN = 10   # số chunks đưa vào cross-encoder
FINAL_TOPK  = 5    # số chunks cuối đưa vào LLM prompt

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBED_DIM         = 768   # chiều vector nomic-embed-text / nomic-embed-text-v1.5
EMBED_CONCURRENCY = 32    # số request embed song song (dùng khi fallback HTTP)
LOCAL_EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"  # local model, không cần HTTP

# ── Reranking ─────────────────────────────────────────────────────────────────
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Evaluation ────────────────────────────────────────────────────────────────
OPTION_LETTERS    = ["A", "B", "C", "D", "E"]
EVAL_LLM_CONCURRENCY      = 16   # match OLLAMA_NUM_PARALLEL=3 (24GB VRAM: 3×6.6GB=19.8GB)
EVAL_RETRIEVE_CONCURRENCY = 12  # ThreadPool cho retrieve (24 core, CrossEncoder GPU)
