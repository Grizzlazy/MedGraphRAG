"""generator.py — Sinh câu trả lời từ retrieved context qua LLM."""

import os
from typing import List, Dict

from openai import OpenAI, AsyncOpenAI


def _get_llm_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "ollama"),
        base_url=os.getenv("OPENAI_API_BASE_URL", "http://localhost:8101/v1"),
    )


def _get_async_llm_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "ollama"),
        base_url=os.getenv("OPENAI_API_BASE_URL", "http://localhost:8101/v1"),
    )


def call_llm(system: str, user: str) -> str:
    """Gọi LLM sync — dùng cho inference đơn lẻ."""
    client = _get_llm_client()
    resp = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.1,
        max_tokens=2048,
    )
    return resp.choices[0].message.content.strip()


async def call_llm_async(
    system: str,
    user: str,
    client: AsyncOpenAI,
) -> str:
    """Gọi LLM async — dùng cho batch evaluation."""
    resp = await client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.1,
        max_tokens=2048,
    )
    return resp.choices[0].message.content.strip()


def build_context(top_chunks: List[Dict]) -> str:
    """
    Ghép top-k chunks thành context block có đánh số nguồn.

    Ví dụ output:
        [1] (Source: patient_001.txt)
        The patient was admitted with...

        [2] (Source: patient_003.txt)
        Lab results showed...
    """
    parts = []
    for i, c in enumerate(top_chunks, start=1):
        src = c["chunk"]["source"]
        parts.append(f"[{i}] (Source: {src})\n{c['chunk']['text']}")
    return "\n\n".join(parts)


def generate_answer(query: str, top_chunks: List[Dict]) -> str:
    """
    Sinh câu trả lời từ top_chunks đã retrieve + rerank.

    Args:
        query:      Câu hỏi gốc của người dùng.
        top_chunks: List dicts từ pipeline.retrieve().

    Returns:
        Câu trả lời dạng text, có citation [1][2]...
    """
    context = build_context(top_chunks)
    system = (
        "You are a medical expert. Answer the question using ONLY the "
        "provided context. Cite sources using [number] notation. "
        "If the context is not relevant, say so."
    )
    user = f"Context:\n{context}\n\nQuestion: {query}"
    return call_llm(system, user)
