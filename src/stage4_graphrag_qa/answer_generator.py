"""
Stage 4 — Answer Generator
Generates grounded clinical answers using LLaMA-3.1 with chain-of-thought prompting.
"""

from pathlib import Path
from loguru import logger
import ollama as ollama_client

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS


ANSWER_PROMPT = """You are a medical AI assistant answering clinical questions based STRICTLY on the provided knowledge graph context.

## Rules:
1. Use chain-of-thought reasoning — think step by step.
2. Cite specific entities and relationships from the graph context.
3. If the answer cannot be determined from the context, say "Không đủ thông tin trong đồ thị tri thức để trả lời câu hỏi này."
4. Do NOT fabricate information not present in the context.
5. Provide your confidence level (1-5) at the end.

## Knowledge Graph Context:
{graph_context}

## Question:
{question}

## Answer (with reasoning steps):"""


class AnswerGenerator:
    """Generates clinical answers grounded in the knowledge graph."""

    def __init__(self, model: str = LLM_MODEL):
        self.model = model

    def generate(
        self,
        question: str,
        graph_context: str,
        temperature: float = LLM_TEMPERATURE,
    ) -> dict:
        """
        Generate an answer with chain-of-thought reasoning.

        Returns:
            {
                "answer": str,
                "raw_response": str,
                "model": str,
            }
        """
        prompt = ANSWER_PROMPT.format(
            graph_context=graph_context,
            question=question,
        )

        try:
            response = ollama_client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": temperature,
                    "num_predict": LLM_MAX_TOKENS,
                },
            )
            raw = response["message"]["content"].strip()

            return {
                "answer": raw,
                "raw_response": raw,
                "model": self.model,
            }

        except Exception as exc:
            logger.error(f"Answer generation failed: {exc}")
            return {
                "answer": f"Lỗi khi sinh câu trả lời: {exc}",
                "raw_response": "",
                "model": self.model,
            }

    def generate_with_sources(
        self,
        question: str,
        retrieval_result: dict,
    ) -> dict:
        """
        Full pipeline: build context from retrieval result → generate answer.

        Args:
            retrieval_result: Output of HybridRetriever.retrieve().

        Returns:
            {
                "answer": str,
                "query_type": str,
                "sources": list[dict],
                "model": str,
            }
        """
        context = retrieval_result.get("context", "")
        result = self.generate(question, context)

        # Collect source entities
        sources: list[dict] = []
        local = retrieval_result.get("local")
        if local:
            for node in local.get("seed_nodes", [])[:5]:
                sources.append({
                    "entity": node.get("name", ""),
                    "type": node.get("type", ""),
                    "score": node.get("score", 0.0),
                })

        result["query_type"] = retrieval_result.get("query_type", "unknown")
        result["sources"] = sources
        return result
