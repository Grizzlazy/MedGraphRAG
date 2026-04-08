"""
Stage 3 — UMLS Metathesaurus Enricher
Enriches the knowledge graph with standardised medical concepts and IS_A / PART_OF hierarchies.
"""

from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.storage.neo4j_client import Neo4jClient


class UMLSEnricher:
    """
    Enriches entities in Neo4j with UMLS concept identifiers (CUI)
    and adds hierarchical relations (IS_A, PART_OF).

    NOTE: Requires a local UMLS RRF installation or API key.
          This implementation provides the framework — actual UMLS lookup
          should be connected to your UMLS data source.
    """

    def __init__(self, neo4j_client: Neo4jClient | None = None, umls_data_dir: str | None = None):
        self.neo4j = neo4j_client or Neo4jClient()
        self.umls_dir = Path(umls_data_dir) if umls_data_dir else None
        self._concept_cache: dict[str, dict] = {}
        logger.info("UMLSEnricher initialised")

    # ------------------------------------------------------------------
    # UMLS Lookup (stub — connect to your UMLS data source)
    # ------------------------------------------------------------------
    def lookup_concept(self, term: str) -> dict | None:
        """
        Look up a medical term in UMLS and return its concept info.

        Returns:
            {"cui": "C0011849", "preferred_name": "Diabetes Mellitus", "semantic_type": "Disease or Syndrome"}
            or None if not found.
        """
        if term in self._concept_cache:
            return self._concept_cache[term]

        # TODO: Implement actual UMLS lookup via:
        #   - Local MRCONSO.RRF / MRREL.RRF files
        #   - UMLS REST API (https://uts-ws.nlm.nih.gov/rest/)
        #   - QuickUMLS library
        # Example with QuickUMLS:
        # from quickumls import QuickUMLS
        # matcher = QuickUMLS(self.umls_dir)
        # matches = matcher.match(term)
        # if matches:
        #     best = matches[0][0]
        #     result = {"cui": best["cui"], "preferred_name": best["term"], ...}
        #     self._concept_cache[term] = result
        #     return result

        logger.debug(f"UMLS lookup not implemented for: {term}")
        return None

    # ------------------------------------------------------------------
    # Enrichment
    # ------------------------------------------------------------------
    def enrich_entity(self, entity_id: str, entity_name: str) -> bool:
        """
        Look up an entity in UMLS and add CUI + preferred name to the node.

        Returns True if the entity was enriched.
        """
        concept = self.lookup_concept(entity_name)
        if not concept:
            return False

        self.neo4j.run_write("""
            MATCH (e:Entity {id: $id})
            SET e.umls_cui = $cui,
                e.umls_preferred = $preferred,
                e.umls_semantic_type = $sem_type
        """, {
            "id": entity_id,
            "cui": concept["cui"],
            "preferred": concept["preferred_name"],
            "sem_type": concept.get("semantic_type", ""),
        })
        return True

    def add_hierarchical_relations(self, entity_id: str, entity_name: str) -> int:
        """
        Add IS_A and PART_OF relations from UMLS hierarchy.

        Returns count of new relations added.
        """
        concept = self.lookup_concept(entity_name)
        if not concept:
            return 0

        # TODO: Query MRREL.RRF for parent concepts:
        # For each parent:
        #   - Create parent entity node if not exists
        #   - Create IS_A or PART_OF edge
        # Example:
        # parents = self._get_umls_parents(concept["cui"])
        # for parent in parents:
        #     self.neo4j.run_write(...)

        return 0

    def enrich_all_entities(self) -> dict:
        """Enrich all Entity nodes in the graph with UMLS data."""
        entities = self.neo4j.run_query("""
            MATCH (e:Entity)
            WHERE e.umls_cui IS NULL
            RETURN e.id AS id, e.name AS name
        """)

        enriched = 0
        for entity in entities:
            if self.enrich_entity(entity["id"], entity["name"]):
                enriched += 1

        logger.info(f"UMLS enrichment: {enriched}/{len(entities)} entities enriched")
        return {"total": len(entities), "enriched": enriched}
