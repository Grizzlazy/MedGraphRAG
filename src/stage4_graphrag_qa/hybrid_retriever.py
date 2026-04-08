"""
Stage 4 — Hybrid Retriever
Combines local and global retrieval strategies with learned weights.
"""

from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.stage4_graphrag_qa.local_retriever import LocalRetriever
from src.stage4_graphrag_qa.global_retriever import GlobalRetriever
from src.stage4_graphrag_qa.query_classifier import QueryClassifier
from src.storage.neo4j_client import Neo4jClient
from src.stage3_knowledge_graph.embedding_indexer import EmbeddingIndexer


class HybridRetriever:
    """Orchestrates local + global retrieval based on query classification."""

    def __init__(
        self,
        neo4j_client: Neo4jClient | None = None,
        embedder: EmbeddingIndexer | None = None,
        classifier_path: str | None = None,
    ):
        self.neo4j = neo4j_client or Neo4jClient()
        self.embedder = embedder or EmbeddingIndexer()
        self.local = LocalRetriever(self.neo4j, self.embedder)
        self.globe = GlobalRetriever(self.neo4j)
        self.classifier = QueryClassifier(classifier_path)

    def retrieve(self, query: str, query_type: str | None = None) -> dict:
        """
        Main retrieval entry point.

        Args:
            query:      User question.
            query_type: Override automatic classification (local/global/hybrid/aggregation).

        Returns:
            {
                "query_type": str,
                "context": str,       # Text context for the LLM
                "local": dict|None,
                "global": dict|None,
            }
        """
        if query_type is None:
            query_type = self.classifier.classify(query)

        logger.info(f"Query type: {query_type} — '{query[:80]}…'")

        local_result = None
        global_result = None
        context_parts: list[str] = []

        if query_type in ("local", "hybrid"):
            local_result = self.local.retrieve(query)
            context_parts.append(self.local.to_context_string(local_result))

        if query_type in ("global", "hybrid", "aggregation"):
            global_result = self.globe.retrieve(query)
            context_parts.append(self.globe.to_context_string(global_result))

        context = "\n\n".join(context_parts) if context_parts else "No context available."

        return {
            "query_type": query_type,
            "context": context,
            "local": local_result,
            "global": global_result,
        }
