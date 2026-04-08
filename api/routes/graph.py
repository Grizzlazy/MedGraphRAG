"""
MedGraphRAG — Graph Route
GET /graph/{entity_id} — Visualise entity subgraphs.
"""

from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from api.schemas import SubgraphResponse, GraphNode, GraphEdge
from src.storage.neo4j_client import Neo4jClient

router = APIRouter()


@router.get("/graph/{entity_id}", response_model=SubgraphResponse, tags=["Graph"])
async def get_entity_subgraph(
    entity_id: str,
    depth: int = Query(default=2, ge=1, le=5, description="Expansion depth"),
):
    """Return the subgraph around an entity for visualisation."""
    try:
        neo4j = Neo4jClient()

        # Check entity exists
        check = neo4j.run_query(
            "MATCH (e:Entity {id: $id}) RETURN e.name AS name", {"id": entity_id}
        )
        if not check:
            neo4j.close()
            raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")

        # Fetch subgraph nodes
        nodes_raw = neo4j.run_query("""
            MATCH (seed:Entity {id: $id})
            CALL apoc.path.subgraphNodes(seed, {maxLevel: $depth})
            YIELD node
            WITH DISTINCT node
            RETURN node.id AS id, node.name AS name, node.type AS type,
                   node.communityId AS community
        """, {"id": entity_id, "depth": depth})

        node_ids = [n["id"] for n in nodes_raw]

        # Fetch edges
        edges_raw = neo4j.run_query("""
            MATCH (a:Entity)-[r]->(b:Entity)
            WHERE a.id IN $ids AND b.id IN $ids
            RETURN a.name AS source, type(r) AS relation, b.name AS target,
                   r.confidence AS confidence
        """, {"ids": node_ids})

        neo4j.close()

        return SubgraphResponse(
            entity_id=entity_id,
            depth=depth,
            nodes=[GraphNode(**n) for n in nodes_raw],
            edges=[GraphEdge(**e) for e in edges_raw],
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Graph query failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/graph/search/{query}", tags=["Graph"])
async def search_entities(query: str, limit: int = Query(default=20, ge=1, le=100)):
    """Full-text search for entities by name."""
    try:
        neo4j = Neo4jClient()
        results = neo4j.run_query("""
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($query)
            RETURN e.id AS id, e.name AS name, e.type AS type
            LIMIT $limit
        """, {"query": query, "limit": limit})
        neo4j.close()
        return {"results": results}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
