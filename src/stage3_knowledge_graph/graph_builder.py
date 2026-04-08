"""
Stage 3 — Knowledge Graph Builder
Populates Neo4j with entities and relations from the extraction pipeline.
"""

import uuid
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.storage.neo4j_client import Neo4jClient
from src.stage3_knowledge_graph.embedding_indexer import EmbeddingIndexer


class KnowledgeGraphBuilder:
    """Constructs the medical knowledge graph in Neo4j."""

    def __init__(self, neo4j_client: Neo4jClient | None = None):
        self.neo4j = neo4j_client or Neo4jClient()
        self.embedder = EmbeddingIndexer()
        logger.info("KnowledgeGraphBuilder initialised")

    # ------------------------------------------------------------------
    # Entity ingestion
    # ------------------------------------------------------------------
    def add_entities(self, entities: list[dict], source_doc: str = "") -> dict[str, str]:
        """
        Insert entities into the graph.

        Returns:
            name→id mapping for relation linking.
        """
        name_to_id: dict[str, str] = {}

        for entity in entities:
            eid = entity.get("id") or str(uuid.uuid4())
            embedding = self.embedder.embed_text(entity["text"])

            self.neo4j.create_entity({
                "id": eid,
                "name": entity["text"],
                "type": entity["type"],
                "source_doc": source_doc,
                "confidence": entity.get("confidence", 1.0),
                "embedding": embedding,
            })
            name_to_id[entity["text"]] = eid

        logger.info(f"Added {len(entities)} entities from {source_doc}")
        return name_to_id

    # ------------------------------------------------------------------
    # Relation ingestion
    # ------------------------------------------------------------------
    def add_relations(
        self,
        relations: list[dict],
        name_to_id: dict[str, str],
    ) -> int:
        """
        Insert relations into the graph, resolving head/tail names to node IDs.

        Returns the count of successfully added relations.
        """
        added = 0
        for rel in relations:
            head_id = name_to_id.get(rel["head"])
            tail_id = name_to_id.get(rel["tail"])

            if not head_id or not tail_id:
                logger.debug(
                    f"Skipping relation {rel['head']} -[{rel['relation']}]-> {rel['tail']} "
                    f"(unresolved IDs)"
                )
                continue

            self.neo4j.create_relation(
                head_id=head_id,
                tail_id=tail_id,
                relation_type=rel["relation"],
                confidence=rel.get("confidence", 0.8),
            )
            added += 1

        logger.info(f"Added {added}/{len(relations)} relations")
        return added

    # ------------------------------------------------------------------
    # Document-level ingestion
    # ------------------------------------------------------------------
    def build_from_extraction(
        self,
        entities: list[dict],
        relations: list[dict],
        source_doc: str = "",
    ) -> dict:
        """
        Full pipeline: insert entities then relations for one document.

        Returns summary stats.
        """
        name_to_id = self.add_entities(entities, source_doc)
        rel_count = self.add_relations(relations, name_to_id)

        # Register document node
        doc_id = str(uuid.uuid4())
        self.neo4j.run_write("""
            MERGE (d:Document {id: $id})
            SET d.name = $name, d.entity_count = $ec, d.relation_count = $rc
        """, {"id": doc_id, "name": source_doc, "ec": len(entities), "rc": rel_count})

        return {
            "document": source_doc,
            "entities_added": len(entities),
            "relations_added": rel_count,
        }

    # ------------------------------------------------------------------
    # Batch ingestion
    # ------------------------------------------------------------------
    def build_from_documents(
        self,
        documents: list[dict],
    ) -> list[dict]:
        """
        Process multiple documents.

        Args:
            documents: [{"doc_id": …, "entities": […], "relations": […]}, …]
        """
        results = []
        for doc in documents:
            stats = self.build_from_extraction(
                entities=doc["entities"],
                relations=doc["relations"],
                source_doc=doc.get("doc_id", ""),
            )
            results.append(stats)
        return results
