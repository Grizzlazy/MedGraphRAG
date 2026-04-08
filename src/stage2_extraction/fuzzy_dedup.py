"""
Stage 2 — Fuzzy Deduplication
Merges duplicate entities across chunks using fuzzy string matching.
"""

from rapidfuzz import fuzz
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import FUZZY_THRESHOLD


def deduplicate_entities(
    entities: list[dict],
    threshold: float = FUZZY_THRESHOLD,
) -> list[dict]:
    """
    Merge duplicate entities using fuzzy string matching.

    Two entities are considered duplicates if:
        1. They have the same ``type``.
        2. Their ``text`` similarity ≥ threshold (default 0.85).

    The version with the higher confidence is kept.

    Returns a deduplicated list.
    """
    unique: list[dict] = []

    for entity in entities:
        is_dup = False
        for existing in unique:
            # Must be the same entity type
            if entity["type"] != existing["type"]:
                continue

            sim = fuzz.ratio(entity["text"].lower(), existing["text"].lower()) / 100.0
            if sim >= threshold:
                # Keep the higher-confidence version
                if entity.get("confidence", 0) > existing.get("confidence", 0):
                    existing["text"] = entity["text"]
                    existing["confidence"] = entity["confidence"]
                # Merge chunk_ids if present
                if "chunk_ids" in existing:
                    existing["chunk_ids"].add(entity.get("chunk_id", -1))
                is_dup = True
                break

        if not is_dup:
            new_entry = entity.copy()
            new_entry["chunk_ids"] = {entity.get("chunk_id", -1)}
            unique.append(new_entry)

    logger.info(
        f"Dedup: {len(entities)} → {len(unique)} entities "
        f"(threshold={threshold})"
    )
    return unique


def deduplicate_relations(
    relations: list[dict],
    threshold: float = FUZZY_THRESHOLD,
) -> list[dict]:
    """
    Merge duplicate relations using fuzzy matching on head + tail.
    """
    unique: list[dict] = []

    for rel in relations:
        is_dup = False
        for existing in unique:
            if rel["relation"] != existing["relation"]:
                continue

            head_sim = fuzz.ratio(rel["head"].lower(), existing["head"].lower()) / 100.0
            tail_sim = fuzz.ratio(rel["tail"].lower(), existing["tail"].lower()) / 100.0

            if head_sim >= threshold and tail_sim >= threshold:
                if rel.get("confidence", 0) > existing.get("confidence", 0):
                    existing.update({
                        "head": rel["head"],
                        "tail": rel["tail"],
                        "confidence": rel["confidence"],
                    })
                is_dup = True
                break

        if not is_dup:
            unique.append(rel.copy())

    logger.info(f"Dedup relations: {len(relations)} → {len(unique)}")
    return unique
