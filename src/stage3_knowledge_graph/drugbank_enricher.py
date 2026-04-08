"""
Stage 3 — DrugBank Enricher
Adds drug interaction (INTERACTS_WITH) and contraindication data from DrugBank.
"""

from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.storage.neo4j_client import Neo4jClient


class DrugBankEnricher:
    """
    Enriches Medication entities with DrugBank interaction data.

    NOTE: Requires DrugBank data (XML or API). This provides the framework —
          connect to your DrugBank data source.
    """

    def __init__(self, neo4j_client: Neo4jClient | None = None, drugbank_path: str | None = None):
        self.neo4j = neo4j_client or Neo4jClient()
        self.drugbank_path = Path(drugbank_path) if drugbank_path else None
        self._interaction_cache: dict[str, list[dict]] = {}
        logger.info("DrugBankEnricher initialised")

    # ------------------------------------------------------------------
    # DrugBank Lookup (stub)
    # ------------------------------------------------------------------
    def lookup_interactions(self, drug_name: str) -> list[dict]:
        """
        Look up drug-drug interactions for a given medication.

        Returns:
            [{"drug": "Aspirin", "interaction_type": "pharmacodynamic",
              "description": "Increased bleeding risk"}, …]
        """
        if drug_name in self._interaction_cache:
            return self._interaction_cache[drug_name]

        # TODO: Implement actual DrugBank lookup via:
        #   - DrugBank XML parsing (full_database.xml)
        #   - DrugBank API
        #   - Pre-built SQLite database
        # Example:
        # interactions = parse_drugbank_interactions(self.drugbank_path, drug_name)
        # self._interaction_cache[drug_name] = interactions
        # return interactions

        logger.debug(f"DrugBank lookup not implemented for: {drug_name}")
        return []

    # ------------------------------------------------------------------
    # Enrichment
    # ------------------------------------------------------------------
    def enrich_medication(self, entity_id: str, drug_name: str) -> int:
        """
        Add INTERACTS_WITH and CONTRAINDICATES relations for a medication.

        Returns count of new relations added.
        """
        interactions = self.lookup_interactions(drug_name)
        added = 0

        for interaction in interactions:
            target_drug = interaction["drug"]

            # Find or create target drug node
            result = self.neo4j.run_query("""
                MATCH (e:Entity {type: 'Medication'})
                WHERE toLower(e.name) = toLower($name)
                RETURN e.id AS id LIMIT 1
            """, {"name": target_drug})

            if not result:
                continue

            target_id = result[0]["id"]

            # Create interaction edge
            rel_type = "CONTRAINDICATES" if "contraindic" in interaction.get("interaction_type", "").lower() else "INTERACTS_WITH"

            self.neo4j.run_write(f"""
                MATCH (a:Entity {{id: $src_id}})
                MATCH (b:Entity {{id: $tgt_id}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r.description = $desc,
                    r.source = 'DrugBank',
                    r.confidence = 0.95
            """, {
                "src_id": entity_id,
                "tgt_id": target_id,
                "desc": interaction.get("description", ""),
            })
            added += 1

        return added

    def enrich_all_medications(self) -> dict:
        """Enrich all Medication entities in the graph."""
        medications = self.neo4j.run_query("""
            MATCH (e:Entity {type: 'Medication'})
            RETURN e.id AS id, e.name AS name
        """)

        total_added = 0
        for med in medications:
            count = self.enrich_medication(med["id"], med["name"])
            total_added += count

        logger.info(f"DrugBank enrichment: {total_added} interactions added for {len(medications)} medications")
        return {"medications": len(medications), "interactions_added": total_added}
