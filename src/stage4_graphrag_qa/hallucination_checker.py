"""
Stage 4 — Hallucination Checker
Verifies factual claims in generated answers against the knowledge graph.
"""

import json
from pathlib import Path
from loguru import logger
import ollama as ollama_client

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.storage.neo4j_client import Neo4jClient
from config.settings import LLM_MODEL, LLM_TEMPERATURE


class HallucinationChecker:
    """Extracts claims from answers and verifies them against the graph."""

    def __init__(self, neo4j_client: Neo4jClient | None = None, model: str = LLM_MODEL):
        self.neo4j = neo4j_client or Neo4jClient()
        self.model = model

    # ------------------------------------------------------------------
    # Claim extraction
    # ------------------------------------------------------------------
    def extract_claims(self, answer: str) -> list[str]:
        """Use LLM to decompose an answer into atomic factual claims."""
        prompt = (
            "Extract all factual medical claims from the following answer. "
            "Each claim should be a simple, atomic statement about one entity or "
            "one relationship. Return ONLY a JSON array of strings.\n\n"
            f"Answer:\n{answer}\n\n"
            "Claims (JSON array):"
        )

        try:
            response = ollama_client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0.0, "num_predict": 1024},
            )
            claims = json.loads(response["message"]["content"])
            if isinstance(claims, dict) and "claims" in claims:
                claims = claims["claims"]
            return [str(c).strip() for c in claims if isinstance(c, str) and len(c) > 5]
        except Exception as exc:
            logger.warning(f"Claim extraction failed: {exc}")
            return []

    # ------------------------------------------------------------------
    # Claim verification
    # ------------------------------------------------------------------
    def verify_claim(self, claim: str) -> dict:
        """
        Verify a single claim by generating a Cypher query and checking the graph.

        Returns:
            {"claim": str, "verified": bool, "evidence_count": int, "cypher": str}
        """
        # Generate Cypher from claim
        prompt = (
            "Convert this medical claim into a Neo4j Cypher query that checks "
            "if the claim is supported by the knowledge graph.\n"
            "The graph has :Entity nodes with properties (id, name, type) and "
            "relationships like HAS_DIAGNOSIS, PRESCRIBED, EXHIBITS_SYMPTOM, "
            "UNDERWENT, MEASURED_AT, INTERACTS_WITH, etc.\n"
            "Return ONLY the Cypher query — nothing else. "
            "The query should RETURN count(*) AS evidence.\n\n"
            f"Claim: {claim}\n\n"
            "Cypher:"
        )

        try:
            response = ollama_client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_predict": 256},
            )
            cypher = response["message"]["content"].strip()
            cypher = cypher.strip("`").strip()

            # Execute the Cypher query
            result = self.neo4j.run_query(cypher)
            evidence_count = result[0].get("evidence", 0) if result else 0

            return {
                "claim": claim,
                "verified": evidence_count > 0,
                "evidence_count": evidence_count,
                "cypher": cypher,
            }

        except Exception as exc:
            logger.debug(f"Claim verification failed for '{claim[:50]}…': {exc}")
            return {
                "claim": claim,
                "verified": False,
                "evidence_count": 0,
                "cypher": "",
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Full check
    # ------------------------------------------------------------------
    def check_answer(self, answer: str) -> dict:
        """
        Full hallucination check pipeline.

        Returns:
            {
                "faithfulness_score": float (0-1),
                "total_claims": int,
                "verified_claims": int,
                "flagged_claims": list[str],
                "details": list[dict],
            }
        """
        claims = self.extract_claims(answer)
        if not claims:
            return {
                "faithfulness_score": 1.0,
                "total_claims": 0,
                "verified_claims": 0,
                "flagged_claims": [],
                "details": [],
            }

        details = [self.verify_claim(c) for c in claims]
        verified = sum(1 for d in details if d["verified"])
        flagged = [d["claim"] for d in details if not d["verified"]]
        score = verified / len(claims) if claims else 0.0

        logger.info(
            f"Hallucination check: {verified}/{len(claims)} claims verified "
            f"(faithfulness={score:.2f})"
        )

        return {
            "faithfulness_score": score,
            "total_claims": len(claims),
            "verified_claims": verified,
            "flagged_claims": flagged,
            "details": details,
        }
