"""
MedGraphRAG — Query Route
POST /query — Ask clinical questions against the knowledge graph.
"""

from pathlib import Path
from fastapi import APIRouter, HTTPException
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from api.schemas import QueryRequest, QueryResponse, SourceInfo, HallucinationReport
from src.stage4_graphrag_qa.hybrid_retriever import HybridRetriever
from src.stage4_graphrag_qa.answer_generator import AnswerGenerator
from src.stage4_graphrag_qa.hallucination_checker import HallucinationChecker

router = APIRouter()

# Singletons (initialised lazily)
_retriever: HybridRetriever | None = None
_generator: AnswerGenerator | None = None
_checker: HallucinationChecker | None = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def _get_generator() -> AnswerGenerator:
    global _generator
    if _generator is None:
        _generator = AnswerGenerator()
    return _generator


def _get_checker() -> HallucinationChecker:
    global _checker
    if _checker is None:
        _checker = HallucinationChecker()
    return _checker


@router.post("/query", response_model=QueryResponse, tags=["QA"])
async def query_graphrag(request: QueryRequest):
    """
    Answer a clinical question using GraphRAG:
    1. Classify query type (local / global / hybrid)
    2. Retrieve relevant context from Neo4j
    3. Generate answer with chain-of-thought
    4. Optionally verify against graph (hallucination check)
    """
    try:
        retriever = _get_retriever()
        generator = _get_generator()

        # Retrieve
        retrieval = retriever.retrieve(request.question, request.query_type)

        # Generate
        result = generator.generate_with_sources(request.question, retrieval)

        # Hallucination check
        hallucination = None
        if request.check_hallucination:
            checker = _get_checker()
            hal_result = checker.check_answer(result["answer"])
            hallucination = HallucinationReport(
                faithfulness_score=hal_result["faithfulness_score"],
                total_claims=hal_result["total_claims"],
                verified_claims=hal_result["verified_claims"],
                flagged_claims=hal_result["flagged_claims"],
            )

        return QueryResponse(
            answer=result["answer"],
            query_type=result.get("query_type", "unknown"),
            sources=[SourceInfo(**s) for s in result.get("sources", [])],
            hallucination=hallucination,
            model=result.get("model", ""),
        )

    except Exception as exc:
        logger.error(f"Query failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
