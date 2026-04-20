"""
_llm.py — LLM + Embedding interface cho nano_graphrag.

Dùng một AsyncOpenAI client duy nhất (shared) với httpx connection pool
để tránh connection storm khi gọi đồng thời nhiều request.
Timeout có thể cấu hình qua biến môi trường OPENAI_*_TIMEOUT.
"""
import os
from typing import Optional

import httpx
import numpy as np
from openai import AsyncOpenAI

from ._utils import compute_args_hash, wrap_embedding_func_with_attrs, EmbeddingFunc
from .base import BaseKVStorage

# ── Shared client (khởi tạo 1 lần, dùng cho cả LLM và embedding) ─────────────
_shared_http_client: Optional[httpx.AsyncClient] = None
_shared_openai_client: Optional[AsyncOpenAI] = None


def _http_timeout() -> httpx.Timeout:
    return httpx.Timeout(
        connect=float(os.getenv("OPENAI_CONNECT_TIMEOUT", "60")),
        read=float(os.getenv("OPENAI_READ_TIMEOUT", "600")),
        write=float(os.getenv("OPENAI_WRITE_TIMEOUT", "120")),
        pool=float(os.getenv("OPENAI_POOL_TIMEOUT", "60")),
    )


def _http_limits() -> httpx.Limits:
    return httpx.Limits(
        max_connections=int(os.getenv("OPENAI_MAX_CONNECTIONS", "200")),
        max_keepalive_connections=int(os.getenv("OPENAI_MAX_KEEPALIVE", "80")),
        keepalive_expiry=30.0,
    )


_shared_embed_http_client: Optional[httpx.AsyncClient] = None
_shared_embed_openai_client: Optional[AsyncOpenAI] = None


def get_shared_client() -> AsyncOpenAI:
    """Client cho LLM (OPENAI_API_BASE_URL). Singleton."""
    global _shared_http_client, _shared_openai_client
    if _shared_openai_client is None:
        _shared_http_client = httpx.AsyncClient(
            limits=_http_limits(),
            timeout=_http_timeout(),
        )
        _shared_openai_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "ollama"),
            base_url=os.getenv("OPENAI_API_BASE_URL", "http://localhost:11434/v1"),
            http_client=_shared_http_client,
        )
    return _shared_openai_client


def get_shared_embed_client() -> AsyncOpenAI:
    """
    Client riêng cho embedding.
    Dùng EMBEDDING_API_BASE_URL nếu có, không thì fallback về OPENAI_API_BASE_URL.
    Cho phép LLM và embedding chạy trên 2 server khác nhau.
    """
    global _shared_embed_http_client, _shared_embed_openai_client
    if _shared_embed_openai_client is None:
        embed_base_url = os.getenv(
            "EMBEDDING_API_BASE_URL",
            os.getenv("OPENAI_API_BASE_URL", "http://localhost:11434/v1"),
        )
        embed_api_key = os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", "ollama"))
        _shared_embed_http_client = httpx.AsyncClient(
            limits=_http_limits(),
            timeout=_http_timeout(),
        )
        _shared_embed_openai_client = AsyncOpenAI(
            api_key=embed_api_key,
            base_url=embed_base_url,
            http_client=_shared_embed_http_client,
        )
    return _shared_embed_openai_client


# ── Core LLM call ─────────────────────────────────────────────────────────────
async def openai_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    client = get_shared_client()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        cached = await hashing_kv.get_by_id(args_hash)
        if cached is not None:
            return cached["return"]

    response = await client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )
    result = response.choices[0].message.content

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": result, "model": model}}
        )
    return result


# ── Model wrappers ─────────────────────────────────────────────────────────────
async def qwen_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    model = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
    return await openai_complete_if_cache(
        model, prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def qwen_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    model = os.getenv("LLM_CHEAP_MODEL", os.getenv("LLM_MODEL", "qwen2.5:7b-instruct"))
    return await openai_complete_if_cache(
        model, prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


# ── Embedding ─────────────────────────────────────────────────────────────────
def build_local_embedding_func() -> EmbeddingFunc:
    """Embedding function dùng local model (Ollama / vLLM), shared client."""
    embedding_dim    = int(os.getenv("EMBEDDING_DIM", "768"))
    max_token_size   = int(os.getenv("EMBEDDING_MAX_TOKENS", "8192"))

    async def _embed(texts: list[str]) -> np.ndarray:
        client = get_shared_embed_client()
        model  = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        response = await client.embeddings.create(
            model=model, input=texts, encoding_format="float"
        )
        return np.array([dp.embedding for dp in response.data])

    return EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=max_token_size,
        func=_embed,
    )


# ── Backward-compatible aliases ────────────────────────────────────────────────
async def gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o", prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o-mini", prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
async def openai_embedding(texts: list[str]) -> np.ndarray:
    client = get_shared_client()
    response = await client.embeddings.create(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])
