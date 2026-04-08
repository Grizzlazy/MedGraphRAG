"""
MedGraphRAG — Neo4j Client Wrapper
Handles connections, queries, and vector search on the knowledge graph.
"""

from neo4j import GraphDatabase
from loguru import logger
from typing import Any

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


class Neo4jClient:
    """Thread-safe Neo4j driver wrapper."""

    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Connected to Neo4j at {uri}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self):
        self.driver.close()
        logger.info("Neo4j connection closed")

    def verify_connectivity(self) -> bool:
        try:
            self.driver.verify_connectivity()
            return True
        except Exception as exc:
            logger.error(f"Neo4j connectivity check failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------
    def run_query(self, cypher: str, params: dict | None = None) -> list[dict]:
        """Execute a Cypher query and return all records as dicts."""
        with self.driver.session() as session:
            result = session.run(cypher, params or {})
            return [dict(record) for record in result]

    def run_write(self, cypher: str, params: dict | None = None) -> Any:
        """Execute a write transaction."""
        with self.driver.session() as session:
            return session.execute_write(lambda tx: tx.run(cypher, params or {}).consume())

    # ------------------------------------------------------------------
    # Entity CRUD
    # ------------------------------------------------------------------
    def create_entity(self, entity: dict) -> None:
        """
        Upsert an Entity node with dynamic label.

        entity keys: id, name, type, source_doc, confidence, embedding (list[float])
        """
        cypher = """
            MERGE (e:Entity {id: $id})
            SET e.name        = $name,
                e.type        = $type,
                e.source_doc  = $source_doc,
                e.confidence  = $confidence,
                e.embedding   = $embedding,
                e.updated_at  = datetime()
            WITH e
            CALL apoc.create.addLabels(e, [$type]) YIELD node
            RETURN node
        """
        self.run_write(cypher, entity)

    def create_relation(
        self,
        head_id: str,
        tail_id: str,
        relation_type: str,
        confidence: float = 1.0,
    ) -> None:
        """Create a directed relationship between two Entity nodes."""
        cypher = f"""
            MATCH (h:Entity {{id: $head_id}})
            MATCH (t:Entity {{id: $tail_id}})
            MERGE (h)-[r:{relation_type}]->(t)
            SET r.confidence  = $confidence,
                r.created_at  = datetime()
        """
        self.run_write(cypher, {
            "head_id": head_id,
            "tail_id": tail_id,
            "confidence": confidence,
        })

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------
    def vector_search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        index_name: str = "entity_embeddings",
    ) -> list[dict]:
        """Approximate nearest-neighbour search via Neo4j vector index."""
        cypher = """
            CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
            YIELD node, score
            RETURN node.id   AS id,
                   node.name AS name,
                   node.type AS type,
                   score
            ORDER BY score DESC
        """
        return self.run_query(cypher, {
            "index_name": index_name,
            "top_k": top_k,
            "embedding": query_embedding,
        })

    # ------------------------------------------------------------------
    # Subgraph expansion
    # ------------------------------------------------------------------
    def get_subgraph(self, seed_ids: list[str], depth: int = 2) -> list[dict]:
        """Expand a subgraph around seed nodes up to *depth* hops."""
        cypher = f"""
            MATCH (seed:Entity)
            WHERE seed.id IN $seed_ids
            CALL apoc.path.subgraphAll(seed, {{maxLevel: $depth}})
            YIELD nodes, relationships
            UNWIND nodes AS n
            WITH DISTINCT n
            RETURN n.id AS id, n.name AS name, n.type AS type,
                   n.communityId AS community
        """
        return self.run_query(cypher, {"seed_ids": seed_ids, "depth": depth})

    # ------------------------------------------------------------------
    # Community helpers
    # ------------------------------------------------------------------
    def get_communities(self) -> list[dict]:
        """Return all Community summaries."""
        return self.run_query("""
            MATCH (c:Community)
            RETURN c.id AS id, c.summary AS summary, c.member_count AS member_count
            ORDER BY c.member_count DESC
        """)

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------
    def init_schema(self, schema_path: str) -> None:
        """Execute a Cypher schema file (constraints + indexes)."""
        from pathlib import Path
        statements = Path(schema_path).read_text(encoding="utf-8")
        for stmt in statements.split(";"):
            stmt = stmt.strip()
            if stmt and not stmt.startswith("//"):
                try:
                    self.run_write(stmt)
                except Exception as exc:
                    logger.warning(f"Schema statement skipped: {exc}")
        logger.info("Neo4j schema initialized")
