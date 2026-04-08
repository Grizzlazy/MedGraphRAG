"""
MedGraphRAG — Ingest Route
POST /ingest — Upload a PDF and run the full pipeline.
"""

import uuid
import tempfile
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from api.schemas import IngestResponse
from src.storage.minio_client import MinioClient
from src.storage.postgres_client import PostgresClient
from src.stage1_deid.pdf_to_image import pdf_to_images
from src.stage1_deid.ocr_processor import ocr_document
from src.stage1_deid.layoutlmv3_ner import LayoutLMv3NER
from src.stage1_deid.postprocess import replace_pii_with_placeholders
from src.stage2_extraction.chunker import chunk_document
from src.stage2_extraction.entity_extractor import extract_entities_from_document
from src.stage2_extraction.relation_extractor import extract_relations_from_document
from src.stage2_extraction.fuzzy_dedup import deduplicate_entities, deduplicate_relations
from src.stage3_knowledge_graph.graph_builder import KnowledgeGraphBuilder

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse, tags=["Ingest"])
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF medical record and run the full MedGraphRAG pipeline:
    1. Store original PDF in MinIO
    2. Convert to images → OCR → NER de-identification
    3. Extract medical entities & relations
    4. Build knowledge graph in Neo4j
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    doc_id = str(uuid.uuid4())

    try:
        # --- 1. Save to MinIO ---
        content = await file.read()
        minio = MinioClient()
        object_name = minio.upload_bytes(content, f"pdfs/{doc_id}/{file.filename}", "application/pdf")

        # Register in PostgreSQL
        pg = PostgresClient()
        pg.register_document(doc_id, file.filename, object_name)

        # --- 2. Stage 1: OCR + DeID ---
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_pdf = Path(tmpdir) / file.filename
            tmp_pdf.write_bytes(content)

            # PDF → images
            image_paths = pdf_to_images(str(tmp_pdf), output_dir=str(Path(tmpdir) / "images"))

            # OCR each page
            ocr_results = []
            for img_path in image_paths:
                from src.stage1_deid.ocr_processor import ocr_page
                words = ocr_page(str(img_path))
                ocr_results.append(words)

            # NER de-identification
            ner_model = LayoutLMv3NER()
            pii_entities = ner_model.predict_document(
                [str(p) for p in image_paths], ocr_results,
            )

            # Replace PII with placeholders
            full_text = " ".join(
                " ".join(w["text"] for w in page) for page in ocr_results
            )
            anonymised_text, pii_mapping = replace_pii_with_placeholders(
                full_text, pii_entities,
            )

            # Store PII mapping (encrypted)
            pg.store_pii_mapping(doc_id, pii_mapping)

        # --- 3. Stage 2: Entity & Relation Extraction ---
        chunks = chunk_document(anonymised_text, doc_id=doc_id)
        raw_entities = extract_entities_from_document(chunks)
        entities = deduplicate_entities(raw_entities)

        chunk_entity_map = {}
        for e in raw_entities:
            cid = e.get("chunk_id", 0)
            chunk_entity_map.setdefault(cid, []).append(e)

        raw_relations = extract_relations_from_document(chunks, chunk_entity_map)
        relations = deduplicate_relations(raw_relations)

        # --- 4. Stage 3: Build Graph ---
        builder = KnowledgeGraphBuilder()
        stats = builder.build_from_extraction(entities, relations, source_doc=doc_id)

        pg.close()

        return IngestResponse(
            document_id=doc_id,
            entities_detected=stats["entities_added"],
            relations_detected=stats["relations_added"],
            message=f"Successfully ingested {file.filename}",
        )

    except Exception as exc:
        logger.error(f"Ingest failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
