"""
Stage 1 — Post-processing
Replaces detected PII entities with structured placeholders [TYPE_ID].
"""

from collections import defaultdict
from loguru import logger


def replace_pii_with_placeholders(
    text: str,
    entities: list[dict],
) -> tuple[str, dict[str, dict]]:
    """
    Replace every PII span in *text* with a placeholder ``[TYPE_NNN]``.

    Args:
        text:     Original document text.
        entities: Detected entities with keys ``text``, ``label``, ``start``, ``end``.

    Returns:
        (anonymised_text, reverse_mapping)

        reverse_mapping = {
            "[PATIENT_NAME_001]": {
                "original": "John Doe",
                "type": "PATIENT_NAME",
                "position": {"start": 10, "end": 18}
            }, …
        }
    """
    counters: dict[str, int] = defaultdict(int)
    reverse_mapping: dict[str, dict] = {}

    # Sort by start position descending → replace from end to avoid offset drift
    sorted_entities = sorted(entities, key=lambda e: e["start"], reverse=True)

    anonymised = text
    for entity in sorted_entities:
        entity_type = entity["label"]
        counters[entity_type] += 1
        placeholder = f"[{entity_type}_{counters[entity_type]:03d}]"

        start = entity["start"]
        end = entity["end"]

        anonymised = anonymised[:start] + placeholder + anonymised[end:]

        reverse_mapping[placeholder] = {
            "original": entity["text"],
            "type": entity_type,
            "position": {"start": start, "end": end},
        }

    logger.info(
        f"Replaced {len(sorted_entities)} PII entities with placeholders "
        f"({dict(counters)})"
    )
    return anonymised, reverse_mapping


def reconstruct_original(anonymised_text: str, reverse_mapping: dict[str, dict]) -> str:
    """Reverse the anonymisation — restore original PII values."""
    result = anonymised_text
    for placeholder, info in reverse_mapping.items():
        result = result.replace(placeholder, info["original"])
    return result


def create_word_level_placeholders(
    words: list[str],
    entities: list[dict],
) -> list[str]:
    """
    Replace words at entity positions with placeholders (word-index based).

    Useful when working with word-level OCR output rather than character offsets.

    Args:
        words:    List of OCR words.
        entities: Entities with ``start`` and ``end`` as word indices.

    Returns:
        New word list with PII words replaced.
    """
    result = list(words)
    counters: dict[str, int] = defaultdict(int)

    sorted_entities = sorted(entities, key=lambda e: e["start"], reverse=True)

    for entity in sorted_entities:
        entity_type = entity["label"]
        counters[entity_type] += 1
        placeholder = f"[{entity_type}_{counters[entity_type]:03d}]"

        start = entity["start"]
        end = entity["end"]
        result[start:end] = [placeholder]

    return result
