"""
Stage 3 — Community Detector
Runs Leiden algorithm on the Neo4j graph and generates community summaries.
"""

from pathlib import Path
from loguru import logger
import ollama as ollama_client

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.storage.neo4j_client import Neo4jClient
from config.settings import LLM_MODEL, LLM_TEMPERATURE, COMMUNITY_SUMMARY_MAX_WORDS


class CommunityDetector:
    """Discovers communities via Leiden and generates LLM summaries."""

    def __init__(self, neo4j_client: Neo4jClient | None = None):
        self.neo4j = neo4j_client or Neo4jClient()
        logger.info("CommunityDetector initialised")

    # ------------------------------------------------------------------
    # Graph projection
    # ------------------------------------------------------------------
    def project_graph(self, projection_name: str = "medgraph") -> dict:
        """Create a GDS in-memory graph projection."""
        # Drop existing projection if any
        try:
            self.neo4j.run_write(f"CALL gds.graph.drop('{projection_name}', false)")
        except Exception:
            pass

        result = self.neo4j.run_query("""
            CALL gds.graph.project(
                $name,
                'Entity',
                '*',
                { relationshipProperties: ['confidence'] }
            )
            YIELD graphName, nodeCount, relationshipCount
            RETURN graphName, nodeCount, relationshipCount
        """, {"name": projection_name})

        if result:
            stats = result[0]
            logger.info(
                f"Projected graph '{stats['graphName']}': "
                f"{stats['nodeCount']} nodes, {stats['relationshipCount']} rels"
            )
            return stats
        return {}

    # ------------------------------------------------------------------
    # Leiden community detection
    # ------------------------------------------------------------------
    def detect_communities(self, projection_name: str = "medgraph") -> dict:
        """Run Leiden algorithm and write communityId to nodes."""
        result = self.neo4j.run_query("""
            CALL gds.leiden.write($name, {
                writeProperty: 'communityId',
                maxLevels: 10,
                gamma: 1.0,
                theta: 0.01
            })
            YIELD communityCount, modularity, ranLevels
            RETURN communityCount, modularity, ranLevels
        """, {"name": projection_name})

        if result:
            stats = result[0]
            logger.info(
                f"Leiden: {stats['communityCount']} communities, "
                f"modularity={stats['modularity']:.4f}, levels={stats['ranLevels']}"
            )
            return stats
        return {}

    # ------------------------------------------------------------------
    # Community summaries
    # ------------------------------------------------------------------
    def get_community_members(self) -> list[dict]:
        """Retrieve all communities and their members."""
        return self.neo4j.run_query("""
            MATCH (e:Entity)
            WHERE e.communityId IS NOT NULL
            WITH e.communityId AS cid, 
                 collect({name: e.name, type: e.type}) AS members
            RETURN cid, members, size(members) AS member_count
            ORDER BY member_count DESC
        """)

    def generate_summaries(
        self,
        max_words: int = COMMUNITY_SUMMARY_MAX_WORDS,
        model: str = LLM_MODEL,
    ) -> dict[int, str]:
        """Generate and store a natural-language summary for each community."""
        communities = self.get_community_members()
        summaries: dict[int, str] = {}

        for comm in communities:
            cid = comm["cid"]
            members = comm["members"]
            member_count = comm["member_count"]

            # Build entity list string (cap at 50 for prompt length)
            entity_list = ", ".join(
                f"{m['name']} ({m['type']})" for m in members[:50]
            )
            if member_count > 50:
                entity_list += f", … and {member_count - 50} more"

            prompt = (
                f"You are a medical knowledge expert. Summarise the following "
                f"community of related medical entities in {max_words} words or less. "
                f"Focus on clinical significance, common themes, and how these "
                f"entities relate to each other.\n\n"
                f"Community entities:\n{entity_list}\n\n"
                f"Summary:"
            )

            try:
                response = ollama_client.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": LLM_TEMPERATURE, "num_predict": 512},
                )
                summary = response["message"]["content"].strip()
            except Exception as exc:
                logger.warning(f"Summary generation failed for community {cid}: {exc}")
                summary = f"Community with {member_count} entities: {entity_list[:200]}"

            summaries[cid] = summary

            # Store in Neo4j
            self.neo4j.run_write("""
                MERGE (c:Community {id: $cid})
                SET c.summary = $summary,
                    c.member_count = $member_count
            """, {"cid": cid, "summary": summary, "member_count": member_count})

        logger.info(f"Generated summaries for {len(summaries)} communities")
        return summaries

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def run(self, projection_name: str = "medgraph") -> dict:
        """Full community detection pipeline: project → detect → summarise."""
        proj_stats = self.project_graph(projection_name)
        comm_stats = self.detect_communities(projection_name)
        summaries = self.generate_summaries()

        return {
            "projection": proj_stats,
            "communities": comm_stats,
            "summaries_generated": len(summaries),
        }
