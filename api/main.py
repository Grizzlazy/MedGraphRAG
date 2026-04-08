"""
MedGraphRAG — FastAPI Application
Main entry point for the API server.
"""

from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.routes import ingest, query, graph
from api.schemas import HealthResponse
from src.storage.neo4j_client import Neo4jClient
from src.storage.postgres_client import PostgresClient
from src.storage.minio_client import MinioClient


# ======================================================================
# Lifespan
# ======================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown events."""
    logger.info("🏥 MedGraphRAG API starting …")

    # Initialise schema if needed
    try:
        neo4j = Neo4jClient()
        schema_path = str(Path(__file__).resolve().parent.parent / "config" / "neo4j_schema.cypher")
        neo4j.init_schema(schema_path)
        neo4j.close()
    except Exception as exc:
        logger.warning(f"Schema init skipped: {exc}")

    yield

    logger.info("MedGraphRAG API shutting down")


# ======================================================================
# App
# ======================================================================
app = FastAPI(
    title="MedGraphRAG API",
    description=(
        "GraphRAG-powered clinical question answering system "
        "for electronic health records."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(graph.router)


# ======================================================================
# Health check
# ======================================================================
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check connectivity to all backend services."""
    neo4j_ok = False
    pg_ok = False
    minio_ok = False

    try:
        neo4j = Neo4jClient()
        neo4j_ok = neo4j.verify_connectivity()
        neo4j.close()
    except Exception:
        pass

    try:
        pg = PostgresClient()
        pg_ok = True
        pg.close()
    except Exception:
        pass

    try:
        minio = MinioClient()
        minio_ok = True
    except Exception:
        pass

    status = "healthy" if all([neo4j_ok, pg_ok, minio_ok]) else "degraded"
    return HealthResponse(status=status, neo4j=neo4j_ok, postgres=pg_ok, minio=minio_ok)


@app.get("/", tags=["System"])
async def root():
    return {
        "name": "MedGraphRAG",
        "version": "1.0.0",
        "description": "GraphRAG for Electronic Health Records",
        "docs": "/docs",
    }


# ======================================================================
# CLI
# ======================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
