"""
Stage 3 — Embedding Indexer
Generates and manages sentence embeddings for graph entities.
"""

from sentence_transformers import SentenceTransformer
from loguru import logger
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import EMBEDDING_MODEL, EMBEDDING_DIM


class EmbeddingIndexer:
    """Generates embeddings using all-MiniLM-L6-v2 (384 dims)."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
        self.dim = EMBEDDING_DIM
        logger.info(f"EmbeddingIndexer loaded: {model_name} ({self.dim}d)")

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings.tolist()

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two texts."""
        emb_a = self.model.encode(text_a, normalize_embeddings=True)
        emb_b = self.model.encode(text_b, normalize_embeddings=True)
        return float(emb_a @ emb_b)
