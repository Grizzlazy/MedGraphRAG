"""
MedGraphRAG — Pydantic Schemas
Request / response models for the FastAPI endpoints.
"""

from pydantic import BaseModel, Field


# ======================================================================
# Ingest
# ======================================================================
class IngestResponse(BaseModel):
    status: str = "success"
    document_id: str
    entities_detected: int = 0
    relations_detected: int = 0
    message: str = ""


# ======================================================================
# Query
# ======================================================================
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Clinical question")
    query_type: str | None = Field(
        None, description="Override auto-detection: local / global / hybrid / aggregation"
    )
    check_hallucination: bool = Field(
        True, description="Run hallucination verification on the answer"
    )


class SourceInfo(BaseModel):
    entity: str
    type: str
    score: float = 0.0


class HallucinationReport(BaseModel):
    faithfulness_score: float = 1.0
    total_claims: int = 0
    verified_claims: int = 0
    flagged_claims: list[str] = []


class QueryResponse(BaseModel):
    answer: str
    query_type: str
    sources: list[SourceInfo] = []
    hallucination: HallucinationReport | None = None
    model: str = ""


# ======================================================================
# Graph
# ======================================================================
class GraphNode(BaseModel):
    id: str
    name: str
    type: str
    community: int | None = None


class GraphEdge(BaseModel):
    source: str
    relation: str
    target: str
    confidence: float = 1.0


class SubgraphResponse(BaseModel):
    entity_id: str
    depth: int
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []


# ======================================================================
# Health
# ======================================================================
class HealthResponse(BaseModel):
    status: str
    neo4j: bool
    postgres: bool
    minio: bool
