"""pipeline.py — RAGPipeline: orchestrate toàn bộ build + retrieve + generate."""

import os
import pickle
from typing import List, Dict, Optional

import faiss

from .config import FINAL_TOPK, RERANK_TOPN, USE_BM25
from .loader import load_txt_files, build_chunks
from .embedder import build_faiss_index
from .retriever import retrieve as _retrieve
from .reranker import rerank
from .generator import generate_answer


class RAGPipeline:
    """
    Interface thống nhất cho Traditional RAG pipeline.

    Tương đương với luồng (seq_ret → get_response) trong MedGraphRAG,
    cho phép so sánh trực tiếp kết quả giữa 2 phương pháp.

    Chế độ retrieval (config.py):
        USE_BM25 = False  → chỉ FAISS (nhanh, ~60ms/query)
        USE_BM25 = True   → FAISS + BM25 hybrid (chậm hơn, recall cao hơn)
    """

    def __init__(self):
        self.chunks:      List[Dict]   = []
        self.faiss_index: faiss.Index  = None
        self.bm25:        Optional[object] = None
        self._built = False

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, data_path: str, cache_dir: str = ".rag_cache") -> None:
        """
        Build FAISS index (+ BM25 nếu USE_BM25=True) từ data_path.
        Kết quả được cache để tránh tính lại embedding mỗi lần chạy.

        Args:
            data_path: Thư mục chứa .txt files (mimic_ex format).
            cache_dir: Thư mục lưu cache.
        """
        os.makedirs(cache_dir, exist_ok=True)
        cache_chunks = os.path.join(cache_dir, "chunks.pkl")
        cache_faiss  = os.path.join(cache_dir, "faiss.index")
        cache_bm25   = os.path.join(cache_dir, "bm25.pkl")

        chunks_ok = os.path.exists(cache_chunks)
        faiss_ok  = os.path.exists(cache_faiss)
        bm25_ok   = os.path.exists(cache_bm25)

        if chunks_ok and faiss_ok and (not USE_BM25 or bm25_ok):
            self._load_cache(cache_chunks, cache_faiss, cache_bm25 if USE_BM25 else None)
        else:
            self._build_from_scratch(data_path, cache_chunks, cache_faiss, cache_bm25)

        self._built = True

    def _load_cache(self, cache_chunks, cache_faiss, cache_bm25) -> None:
        print("Loading cached index...")
        with open(cache_chunks, "rb") as f:
            self.chunks = pickle.load(f)
        self.faiss_index = faiss.read_index(cache_faiss)
        if cache_bm25 and os.path.exists(cache_bm25):
            with open(cache_bm25, "rb") as f:
                self.bm25 = pickle.load(f)
            print(f"Cache loaded: {len(self.chunks)} chunks, "
                  f"{self.faiss_index.ntotal} FAISS vectors, BM25 enabled")
        else:
            print(f"Cache loaded: {len(self.chunks)} chunks, "
                  f"{self.faiss_index.ntotal} FAISS vectors (FAISS-only mode)")

    def _build_from_scratch(
        self, data_path, cache_chunks, cache_faiss, cache_bm25
    ) -> None:
        docs = load_txt_files(data_path)
        self.chunks = build_chunks(docs)

        print("Building FAISS index (embedding tất cả chunks)...")
        self.faiss_index, _ = build_faiss_index(self.chunks)

        with open(cache_chunks, "wb") as f:
            pickle.dump(self.chunks, f)
        faiss.write_index(self.faiss_index, cache_faiss)

        if USE_BM25:
            from .indexer import build_bm25_index
            self.bm25 = build_bm25_index(self.chunks)
            with open(cache_bm25, "wb") as f:
                pickle.dump(self.bm25, f)

        print("Index cached.")

    # ── Retrieve ───────────────────────────────────────────────────────────────

    def retrieve(self, query: str, final_k: int = FINAL_TOPK) -> List[Dict]:
        """
        Trả về top final_k chunks sau retrieval + cross-encoder reranking.

        Chế độ phụ thuộc vào config.USE_BM25:
            False → FAISS dense (~60ms/query)
            True  → FAISS + BM25 hybrid + RRF (~seconds/query)
        """
        if not self._built:
            raise RuntimeError("Gọi build() trước khi retrieve().")

        candidates = _retrieve(query, self.faiss_index, self.chunks, self.bm25)
        reranked   = rerank(query, candidates, top_n=RERANK_TOPN)
        return reranked[:final_k]

    # ── Query (retrieve + generate) ────────────────────────────────────────────

    def query(self, question: str, final_k: int = FINAL_TOPK) -> str:
        """Full pipeline: retrieve chunks → sinh câu trả lời bằng LLM."""
        top_chunks = self.retrieve(question, final_k)
        return generate_answer(question, top_chunks)
