"""
rag/ — Traditional RAG: FAISS + BM25 Hybrid Retrieval + Cross-encoder Reranking

Modules:
    config    — Hằng số cấu hình toàn pipeline
    loader    — Load .txt files và chunk theo token
    embedder  — Embedding (nomic-embed-text) và FAISS index
    indexer   — BM25 sparse index
    retriever — Hybrid RRF fusion (FAISS + BM25)
    reranker  — Cross-encoder reranking
    generator — Sinh câu trả lời qua LLM
    pipeline  — RAGPipeline: orchestrate toàn bộ
    evaluator — MCQ evaluation trên MedQA / MedMCQA / PubMedQA
    main      — CLI entry point

Quick start:
    from rag.pipeline import RAGPipeline

    rag = RAGPipeline()
    rag.build("./dataset/mimic_ex")
    answer = rag.query("What medications is the patient taking?")

CLI:
    python -m rag.main --data-path ./dataset/mimic_ex --query "..."
    python -m rag.main --data-path ./dataset/mimic_ex --eval medqa medmcqa pubmedqa
"""

from .pipeline  import RAGPipeline
from .evaluator import evaluate_dataset

__all__ = ["RAGPipeline", "evaluate_dataset"]
