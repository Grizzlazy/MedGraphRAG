"""embedder.py — Embedding local (SentenceTransformer) + FAISS index.

Ưu tiên dùng local SentenceTransformer (GPU nếu có, CPU nếu không):
  - Không cần HTTP round-trip → ~0.02-0.05s/query (so với ~0.3s Ollama HTTP)
  - GPU inference ~0.005s/query với RTX 4090

Fallback sang Ollama HTTP nếu sentence-transformers chưa cài.
"""

import asyncio
import os
import threading
from typing import List, Dict, Tuple

import numpy as np
import faiss
from tqdm import tqdm

from .config import EMBED_DIM, EMBED_CONCURRENCY, LOCAL_EMBED_MODEL


# ── Local embedding model singleton (thread-safe) ────────────────────────────

_local_model = None
_local_lock  = threading.Lock()
_embed_lock  = threading.Lock()   # serialize concurrent model.encode() calls (not thread-safe)


def _get_local_model():
    global _local_model
    if _local_model is None:
        with _local_lock:
            if _local_model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    print(f"Loading local embed model: {LOCAL_EMBED_MODEL} on {device}")
                    _local_model = SentenceTransformer(
                        LOCAL_EMBED_MODEL,
                        trust_remote_code=True,
                        device=device,
                    )
                except ImportError:
                    _local_model = "unavailable"
    return _local_model if _local_model != "unavailable" else None


# ── Public embedding API ──────────────────────────────────────────────────────

def get_embed_client():
    """Trả về None — dùng local model, không cần HTTP client."""
    return None


def get_embedding(text: str, client=None) -> np.ndarray:
    """
    Embed 1 text. Ưu tiên local SentenceTransformer, fallback Ollama HTTP.
    Vector L2-normalize để cosine similarity = inner product.
    """
    text = text.replace("\x00", " ").strip()[:8000] or "N/A"

    model = _get_local_model()
    if model is not None:
        with _embed_lock:
            vec = model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return vec.astype(np.float32)

    # Fallback: Ollama HTTP
    return _get_embedding_http(text)


def _get_embedding_http(text: str) -> np.ndarray:
    """Fallback: gọi Ollama HTTP khi local model không có."""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "ollama"),
        base_url=os.getenv("OPENAI_API_BASE_URL", "http://localhost:11434/v1"),
    )
    model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    resp  = client.embeddings.create(input=text, model=model)
    vec   = np.array(resp.data[0].embedding, dtype=np.float32)
    norm  = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ── Batch embedding cho build index ──────────────────────────────────────────

def _embed_batch_local(texts: List[str]) -> List[np.ndarray]:
    """Embed toàn bộ texts bằng local model (batch, nhanh nhất)."""
    model = _get_local_model()
    texts = [t.replace("\x00", " ").strip()[:8000] or "N/A" for t in texts]
    vecs  = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    )
    return [v.astype(np.float32) for v in vecs]


async def _embed_all_async_http(texts: List[str], concurrency: int) -> List[np.ndarray]:
    """Fallback: embed bằng Ollama HTTP async."""
    from openai import AsyncOpenAI
    embed_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "ollama"),
        base_url=os.getenv("OPENAI_API_BASE_URL", "http://localhost:11434/v1"),
    )
    sem     = asyncio.Semaphore(concurrency)
    results = [None] * len(texts)

    async def _one(idx: int, text: str):
        text = text.replace("\x00", " ").strip()[:8000] or "N/A"
        async with sem:
            resp = await client.embeddings.create(input=text, model=embed_model)
        vec  = np.array(resp.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        results[idx] = vec

    tasks = [asyncio.create_task(_one(i, t)) for i, t in enumerate(texts)]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks),
                     desc=f"Embedding HTTP (x{concurrency})"):
        await coro

    await client.close()
    return results


def build_faiss_index(
    chunks: List[Dict],
    concurrency: int = EMBED_CONCURRENCY,
) -> Tuple[faiss.Index, np.ndarray]:
    """
    Tạo FAISS IndexFlatIP từ danh sách chunks.
    Dùng local SentenceTransformer (batch GPU) nếu có, fallback Ollama async HTTP.

    Returns:
        (faiss.Index, embedding_matrix shape (N, EMBED_DIM))
    """
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts):,} chunks...")

    local = _get_local_model()
    if local is not None:
        print(f"  → local model: {LOCAL_EMBED_MODEL}")
        embeddings = _embed_batch_local(texts)
    else:
        print(f"  → fallback Ollama HTTP (concurrency={concurrency})")
        embeddings = asyncio.run(_embed_all_async_http(texts, concurrency))

    matrix = np.vstack(embeddings).astype(np.float32)
    index  = faiss.IndexFlatIP(EMBED_DIM)
    index.add(matrix)

    print(f"FAISS index built: {index.ntotal:,} vectors")
    return index, matrix
