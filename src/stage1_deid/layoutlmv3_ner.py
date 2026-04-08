"""
Stage 1 — LayoutLMv3 NER Model
Wrapper around LayoutLMv3ForTokenClassification for de-identification.
"""

from pathlib import Path
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
)
from PIL import Image
import torch
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LABEL_LIST, LABEL2ID, ID2LABEL, NUM_LABELS, LAYOUTLMV3_MODEL


class LayoutLMv3NER:
    """LayoutLMv3-based NER model for PII detection in medical documents."""

    def __init__(self, model_path: str | None = None, device: str | None = None):
        """
        Args:
            model_path: Path to fine-tuned checkpoint, or None to load base model.
            device:     'cuda' / 'cpu' / None (auto-detect).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        load_from = model_path or LAYOUTLMV3_MODEL
        self.processor = LayoutLMv3Processor.from_pretrained(
            LAYOUTLMV3_MODEL, apply_ocr=False,
        )
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            load_from,
            num_labels=NUM_LABELS,
            label2id=LABEL2ID,
            id2label=ID2LABEL,
        ).to(self.device)
        self.model.eval()

        logger.info(f"LayoutLMv3NER loaded from {load_from} on {self.device}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        words: list[str],
        boxes: list[list[int]],
    ) -> list[dict]:
        """
        Run NER inference on a single page.

        Args:
            image: PIL Image of the page.
            words: OCR-extracted word list.
            boxes: Normalised bounding boxes [x0, y0, x1, y1] in [0, 1000].

        Returns:
            List of detected entities:
            [{"text": "John Doe", "label": "PATIENT_NAME", "start": 0, "end": 2,
              "confidence": 0.98, "bbox": [...]}, …]
        """
        encoding = self.processor(
            image, words, boxes=boxes,
            truncation=True, padding="max_length", max_length=512,
            return_tensors="pt",
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        outputs = self.model(**encoding)
        logits = outputs.logits  # (1, seq_len, num_labels)
        probs = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1).squeeze().cpu().tolist()
        confidences = probs.max(dim=-1).values.squeeze().cpu().tolist()

        # Align sub-token predictions back to word level
        word_ids = encoding.get("overflow_to_sample_mapping", None)
        # LayoutLMv3Processor aligns labels to word_ids internally;
        # here we decode directly from predictions using LABEL_LIST.

        entities = self._decode_bio(predictions, confidences, words, boxes)
        return entities

    def _decode_bio(
        self,
        tag_ids: list[int],
        confidences: list[float],
        words: list[str],
        boxes: list[list[int]],
    ) -> list[dict]:
        """Decode BIO tag sequence into entity spans."""
        entities: list[dict] = []
        current: dict | None = None

        # tag_ids may be longer than words (padding / special tokens)
        for i in range(min(len(tag_ids), len(words))):
            tag = LABEL_LIST[tag_ids[i]] if tag_ids[i] < len(LABEL_LIST) else "O"
            conf = confidences[i] if i < len(confidences) else 0.0

            if tag.startswith("B-"):
                # Save previous entity
                if current:
                    entities.append(current)
                label = tag[2:]
                current = {
                    "text": words[i],
                    "label": label,
                    "start": i,
                    "end": i + 1,
                    "confidence": conf,
                    "bbox": boxes[i],
                }
            elif tag.startswith("I-") and current:
                label = tag[2:]
                if label == current["label"]:
                    current["text"] += " " + words[i]
                    current["end"] = i + 1
                    current["confidence"] = min(current["confidence"], conf)
                    # Expand bbox
                    current["bbox"] = [
                        min(current["bbox"][0], boxes[i][0]),
                        min(current["bbox"][1], boxes[i][1]),
                        max(current["bbox"][2], boxes[i][2]),
                        max(current["bbox"][3], boxes[i][3]),
                    ]
                else:
                    entities.append(current)
                    current = None
            else:
                if current:
                    entities.append(current)
                    current = None

        if current:
            entities.append(current)

        return entities

    # ------------------------------------------------------------------
    # Full-document inference
    # ------------------------------------------------------------------
    def predict_document(
        self,
        image_paths: list[str],
        ocr_results: list[list[dict]],
    ) -> list[dict]:
        """
        Run NER on every page of a document.

        Args:
            image_paths: List of page image file paths.
            ocr_results: Per-page list of OCR word dicts (text, bbox).

        Returns:
            All detected entities with page numbers.
        """
        all_entities: list[dict] = []

        for page_idx, (img_path, page_words) in enumerate(
            zip(image_paths, ocr_results)
        ):
            image = Image.open(img_path).convert("RGB")
            words = [w["text"] for w in page_words]
            boxes = [w["bbox"] for w in page_words]

            page_entities = self.predict(image, words, boxes)
            for e in page_entities:
                e["page"] = page_idx + 1
            all_entities.extend(page_entities)

        logger.info(f"Detected {len(all_entities)} PII entities across {len(image_paths)} pages")
        return all_entities
