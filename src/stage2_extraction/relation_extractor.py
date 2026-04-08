"""
Stage 2 — Relation Extractor
Uses LLaMA-3.1-8B with 5-shot in-context learning to find medical relationships.
"""

import json
from pathlib import Path
from loguru import logger
import ollama as ollama_client

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, RELATION_TYPES


# ======================================================================
# 5-Shot Prompt
# ======================================================================
RELATION_PROMPT_TEMPLATE = """You are a medical relation extraction system. Given clinical text and a list of previously extracted entities, identify relationships between them.

Allowed relationship types:
{relation_types}

### Example 1
Text: "Patient was diagnosed with Type 2 Diabetes and prescribed Metformin 500mg."
Entities: ["Type 2 Diabetes", "Metformin 500mg"]
Relations:
[{{"head": "Patient", "relation": "HAS_DIAGNOSIS", "tail": "Type 2 Diabetes", "confidence": 0.95}},
 {{"head": "Patient", "relation": "PRESCRIBED", "tail": "Metformin 500mg", "confidence": 0.93}}]

### Example 2
Text: "The patient exhibits persistent cough and high fever since admission on 2024-01-15."
Entities: ["persistent cough", "high fever", "2024-01-15"]
Relations:
[{{"head": "Patient", "relation": "EXHIBITS_SYMPTOM", "tail": "persistent cough", "confidence": 0.92}},
 {{"head": "Patient", "relation": "EXHIBITS_SYMPTOM", "tail": "high fever", "confidence": 0.90}},
 {{"head": "admission", "relation": "HAS_RESULT", "tail": "2024-01-15", "confidence": 0.88}}]

### Example 3
Text: "Blood pressure measured at 140/90 mmHg in the Cardiology department."
Entities: ["Blood pressure", "140/90 mmHg", "Cardiology"]
Relations:
[{{"head": "Blood pressure", "relation": "MEASURED_AT", "tail": "140/90 mmHg", "confidence": 0.94}},
 {{"head": "Patient", "relation": "ADMITTED_TO", "tail": "Cardiology", "confidence": 0.85}}]

### Example 4
Text: "Aspirin is contraindicated due to patient's allergy history."
Entities: ["Aspirin", "allergy history"]
Relations:
[{{"head": "Aspirin", "relation": "CONTRAINDICATES", "tail": "allergy history", "confidence": 0.91}}]

### Example 5
Text: "Warfarin interacts with Aspirin, increasing bleeding risk. Follow-up in 2 weeks."
Entities: ["Warfarin", "Aspirin", "2 weeks"]
Relations:
[{{"head": "Warfarin", "relation": "INTERACTS_WITH", "tail": "Aspirin", "confidence": 0.96}},
 {{"head": "Patient", "relation": "FOLLOWS_UP", "tail": "2 weeks", "confidence": 0.87}}]

### Now extract relations from:
Text: \"\"\"{chunk_text}\"\"\"
Entities: {entities}

Return ONLY a JSON array of relation objects. Each object must have: head, relation, tail, confidence.
Relations:"""


# ======================================================================
# Extractor
# ======================================================================
def extract_relations_from_chunk(
    chunk_text: str,
    entities: list[dict],
    model: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
) -> list[dict]:
    """
    Extract medical relations from a chunk given its entities.

    Returns:
        [{"head": …, "relation": …, "tail": …, "confidence": …}, …]
    """
    entity_names = [e["text"] for e in entities]
    prompt = RELATION_PROMPT_TEMPLATE.format(
        relation_types=", ".join(RELATION_TYPES),
        chunk_text=chunk_text,
        entities=json.dumps(entity_names),
    )

    try:
        response = ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": temperature, "num_predict": LLM_MAX_TOKENS},
        )
        content = response["message"]["content"]
        relations = json.loads(content)

        if isinstance(relations, dict) and "relations" in relations:
            relations = relations["relations"]

        # Validate
        valid: list[dict] = []
        for r in relations:
            if isinstance(r, dict) and all(k in r for k in ("head", "relation", "tail")):
                if r["relation"] in RELATION_TYPES:
                    valid.append({
                        "head": str(r["head"]).strip(),
                        "relation": r["relation"],
                        "tail": str(r["tail"]).strip(),
                        "confidence": float(r.get("confidence", 0.7)),
                    })
        return valid

    except Exception as exc:
        logger.warning(f"Relation extraction failed: {exc}")
        return []


def extract_relations_from_document(
    chunks: list[dict],
    chunk_entities: dict[int, list[dict]],
    model: str = LLM_MODEL,
) -> list[dict]:
    """
    Extract relations from all chunks.

    Args:
        chunks:         Output of chunker.
        chunk_entities: { chunk_id: [entity, …] }

    Returns:
        Flat list of relations with chunk_id.
    """
    all_relations: list[dict] = []

    for chunk in chunks:
        cid = chunk["chunk_id"]
        entities = chunk_entities.get(cid, [])
        if len(entities) < 2:
            continue  # Need at least 2 entities for a relation

        relations = extract_relations_from_chunk(chunk["text"], entities, model)
        for r in relations:
            r["chunk_id"] = cid
            r["doc_id"] = chunk.get("doc_id", "")
        all_relations.extend(relations)

    logger.info(f"Extracted {len(all_relations)} relations from {len(chunks)} chunks")
    return all_relations
