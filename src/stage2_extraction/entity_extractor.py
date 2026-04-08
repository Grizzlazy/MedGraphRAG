"""
Stage 2 — Medical Entity Extractor
Uses LLaMA-3.1-8B via Ollama to extract medical entities from clinical text chunks.
"""

import json
from pathlib import Path
from loguru import logger
import ollama as ollama_client

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, ENTITY_TYPES


# ======================================================================
# Prompt template
# ======================================================================
ENTITY_PROMPT_TEMPLATE = """You are a medical NER system. Extract all medical entities from the following clinical text.

Entity types to extract:
- Diagnosis: Disease names, conditions, syndromes
- Medication: Drug names, prescriptions, dosages
- Symptom: Signs, symptoms reported by patient or observed
- Procedure: Surgical procedures, therapeutic interventions
- LabTest: Laboratory tests, imaging studies, diagnostic tests
- VitalSign: Blood pressure, heart rate, temperature, SpO2, etc.
- ClinicalDate: Admission dates, discharge dates, follow-up dates
- Department: Hospital departments, wards, units

Rules:
1. Extract ONLY entities that are explicitly mentioned in the text.
2. Do NOT infer or hallucinate entities.
3. Assign a confidence score between 0.0 and 1.0 for each entity.
4. Return ONLY a valid JSON array — no other text.

Clinical text:
\"\"\"
{chunk_text}
\"\"\"

Extracted entities (JSON array):"""


# ======================================================================
# Extractor
# ======================================================================
def extract_entities_from_chunk(
    chunk_text: str,
    model: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
) -> list[dict]:
    """
    Extract medical entities from a single text chunk via LLM.

    Returns:
        [{"text": "Type 2 Diabetes", "type": "Diagnosis", "confidence": 0.95}, …]
    """
    prompt = ENTITY_PROMPT_TEMPLATE.format(chunk_text=chunk_text)

    try:
        response = ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": temperature, "num_predict": LLM_MAX_TOKENS},
        )
        content = response["message"]["content"]
        entities = json.loads(content)

        # Normalise: handle both list and {"entities": [...]} formats
        if isinstance(entities, dict) and "entities" in entities:
            entities = entities["entities"]

        # Validate
        valid: list[dict] = []
        for e in entities:
            if isinstance(e, dict) and "text" in e and "type" in e:
                if e["type"] in ENTITY_TYPES:
                    valid.append({
                        "text": str(e["text"]).strip(),
                        "type": e["type"],
                        "confidence": float(e.get("confidence", 0.8)),
                    })
        return valid

    except Exception as exc:
        logger.warning(f"Entity extraction failed: {exc}")
        return []


def extract_entities_from_document(
    chunks: list[dict],
    model: str = LLM_MODEL,
) -> list[dict]:
    """
    Extract entities from all chunks of a document.

    Args:
        chunks: Output of `chunker.chunk_document()`.

    Returns:
        Flat list of entities, each tagged with chunk_id.
    """
    all_entities: list[dict] = []

    for chunk in chunks:
        entities = extract_entities_from_chunk(chunk["text"], model=model)
        for e in entities:
            e["chunk_id"] = chunk["chunk_id"]
            e["doc_id"] = chunk.get("doc_id", "")
        all_entities.extend(entities)

    logger.info(f"Extracted {len(all_entities)} entities from {len(chunks)} chunks")
    return all_entities
