"""
Stage 4 — Global Retriever
Map-reduce summarisation over community reports for global queries.
"""

from pathlib import Path
from loguru import logger
import ollama as ollama_client

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.storage.neo4j_client import Neo4jClient
from config.settings import LLM_MODEL, LLM_TEMPERATURE


class GlobalRetriever:
    """Uses community summaries to answer broad, dataset-wide questions."""

    def __init__(self, neo4j_client: Neo4jClient | None = None):
        self.neo4j = neo4j_client or Neo4jClient()

    def retrieve(self, query: str) -> dict:
        """
        Fetch all community summaries and rank them by relevance.
        """
        communities = self.neo4j.get_communities()

        if not communities:
            logger.warning("Global retrieval: no communities found")
            return {"communities": [], "map_results": []}

        logger.info(f"Global retrieval: {len(communities)} communities available")
        return {"communities": communities}

    def map_reduce_answer(
        self,
        query: str,
        communities: list[dict],
        model: str = LLM_MODEL,
    ) -> str:
        """
        Map-reduce over community summaries:
        1. MAP: Ask each community for relevant information.
        2. REDUCE: Combine all responses into a final answer.
        """
        # MAP phase
        map_results: list[str] = []
        for comm in communities:
            summary = comm.get("summary", "")
            if not summary:
                continue

            map_prompt = (
                f"Given the following community summary, extract any information "
                f"relevant to the question. If no relevant information exists, reply 'N/A'.\n\n"
                f"Community Summary:\n{summary}\n\n"
                f"Question: {query}\n\n"
                f"Relevant information:"
            )

            try:
                response = ollama_client.chat(
                    model=model,
                    messages=[{"role": "user", "content": map_prompt}],
                    options={"temperature": LLM_TEMPERATURE, "num_predict": 512},
                )
                answer = response["message"]["content"].strip()
                if answer.upper() != "N/A" and len(answer) > 10:
                    map_results.append(answer)
            except Exception as exc:
                logger.warning(f"Map phase failed for community {comm.get('id')}: {exc}")

        if not map_results:
            return "Không tìm thấy thông tin liên quan trong đồ thị tri thức."

        # REDUCE phase
        combined = "\n\n---\n\n".join(
            f"Source {i+1}: {r}" for i, r in enumerate(map_results)
        )
        reduce_prompt = (
            f"Based on the following information from multiple sources, "
            f"synthesise a comprehensive answer to the question. "
            f"Include specific details and cite source numbers.\n\n"
            f"Question: {query}\n\n"
            f"Sources:\n{combined}\n\n"
            f"Comprehensive answer:"
        )

        try:
            response = ollama_client.chat(
                model=model,
                messages=[{"role": "user", "content": reduce_prompt}],
                options={"temperature": LLM_TEMPERATURE, "num_predict": 1024},
            )
            return response["message"]["content"].strip()
        except Exception as exc:
            logger.error(f"Reduce phase failed: {exc}")
            return "\n".join(map_results)

    def to_context_string(self, retrieval_result: dict) -> str:
        """Convert community summaries into LLM context."""
        lines = ["## Community Summaries:"]
        for comm in retrieval_result.get("communities", []):
            lines.append(
                f"\n### Community {comm.get('id', '?')} "
                f"({comm.get('member_count', 0)} members)\n"
                f"{comm.get('summary', 'No summary available')}"
            )
        return "\n".join(lines)
