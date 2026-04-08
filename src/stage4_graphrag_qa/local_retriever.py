"""
Stage 4 — Local Retriever
Vector search + Cypher subgraph expansion for entity-specific queries.
"""

from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.storage.neo4j_client import Neo4jClient
from src.stage3_knowledge_graph.embedding_indexer import EmbeddingIndexer
from config.settings import RETRIEVAL_TOP_K, SUBGRAPH_DEPTH


class LocalRetriever:
    """Retrieves a focused subgraph around the most relevant entities."""

    def __init__(
        self,
        neo4j_client: Neo4jClient | None = None,
        embedder: EmbeddingIndexer | None = None,
    ):
        self.neo4j = neo4j_client or Neo4jClient()
        self.embedder = embedder or EmbeddingIndexer()

    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        depth: int = SUBGRAPH_DEPTH,
    ) -> dict:
        """
        1. Embed the query.
        2. Vector-search for top-k seed entities.
        3. Expand subgraph out to *depth* hops.
        4. Collect all edges within the subgraph.
        """
        query_embedding = self.embedder.embed_text(query)

        # Step 1: seed nodes via vector similarity
        seed_nodes = self.neo4j.vector_search(query_embedding, top_k=top_k)
        seed_ids = [n["id"] for n in seed_nodes]

        if not seed_ids:
            logger.warning("Local retrieval: no seed nodes found")
            return {"seed_nodes": [], "subgraph_nodes": [], "subgraph_edges": []}

        # Step 2: expand subgraph
        subgraph_nodes = self.neo4j.run_query("""
            MATCH (seed:Entity)
            WHERE seed.id IN $ids
            CALL apoc.path.subgraphNodes(seed, {maxLevel: $depth})
            YIELD node
            WITH DISTINCT node
            RETURN node.id AS id, node.name AS name, node.type AS type,
                   node.communityId AS community, node.confidence AS confidence
        """, {"ids": seed_ids, "depth": depth})

        # Step 3: edges within subgraph
        all_node_ids = [n["id"] for n in subgraph_nodes]
        subgraph_edges = self.neo4j.run_query("""
            MATCH (a:Entity)-[r]->(b:Entity)
            WHERE a.id IN $ids AND b.id IN $ids
            RETURN a.name AS source, type(r) AS relation, b.name AS target,
                   r.confidence AS confidence
        """, {"ids": all_node_ids})

        logger.info(
            f"Local retrieval: {len(seed_nodes)} seeds → "
            f"{len(subgraph_nodes)} nodes, {len(subgraph_edges)} edges"
        )

        return {
            "seed_nodes": seed_nodes,
            "subgraph_nodes": subgraph_nodes,
            "subgraph_edges": subgraph_edges,
        }

    def to_context_string(self, retrieval_result: dict) -> str:
        """Convert retrieval result to a text context for the LLM."""
        lines = ["## Relevant Entities:"]
        for node in retrieval_result.get("subgraph_nodes", []):
            lines.append(f"- {node['name']} (type: {node['type']})")

        lines.append("\n## Relationships:")
        for edge in retrieval_result.get("subgraph_edges", []):
            lines.append(f"- {edge['source']} --[{edge['relation']}]--> {edge['target']}")

        return "\n".join(lines)
