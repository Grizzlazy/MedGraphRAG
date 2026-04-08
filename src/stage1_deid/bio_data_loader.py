"""
Stage 1 — BIO Data Loader
Converts OCR words + ground-truth JSON into BIO-tagged samples for LayoutLMv3.
"""

import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from transformers import LayoutLMv3Processor
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LABEL2ID, LAYOUTLMV3_MODEL, LAYOUTLMV3_MAX_SEQ_LENGTH


class DeIDDataset(Dataset):
    """
    PyTorch Dataset that pairs document images with BIO NER labels.

    Each sample corresponds to one page and contains:
        - pixel_values (image tensor)
        - input_ids, attention_mask, bbox (from tokeniser)
        - labels (BIO tag ids, aligned to sub-tokens)
    """

    def __init__(
        self,
        ocr_dir: str,
        image_dir: str,
        gt_json_path: str,
        processor: LayoutLMv3Processor | None = None,
        max_length: int = LAYOUTLMV3_MAX_SEQ_LENGTH,
    ):
        """
        Args:
            ocr_dir:      Directory of per-document OCR JSON files.
            image_dir:    Directory of per-document page images.
            gt_json_path: Ground-truth JSON (from PDF-DeID dataset).
            max_length:   Maximum token sequence length.
        """
        self.max_length = max_length
        self.processor = processor or LayoutLMv3Processor.from_pretrained(
            LAYOUTLMV3_MODEL, apply_ocr=False,
        )

        self.samples = self._build_samples(ocr_dir, image_dir, gt_json_path)
        logger.info(f"DeIDDataset: loaded {len(self.samples)} page samples")

    # ------------------------------------------------------------------
    # Build samples
    # ------------------------------------------------------------------
    def _build_samples(self, ocr_dir: str, image_dir: str, gt_json_path: str) -> list[dict]:
        """Pair OCR words with BIO labels using ground-truth annotations."""
        with open(gt_json_path, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)

        samples = []
        ocr_dir = Path(ocr_dir)
        image_dir = Path(image_dir)

        for doc_name, gt_entities in ground_truth.items():
            doc_stem = Path(doc_name).stem
            ocr_file = ocr_dir / f"{doc_stem}.json"
            doc_img_dir = image_dir / doc_stem

            if not ocr_file.exists():
                logger.warning(f"Skipping {doc_name}: no OCR file")
                continue

            with open(ocr_file, "r", encoding="utf-8") as f:
                ocr_words = json.load(f)

            # Group words by page
            pages: dict[int, list[dict]] = {}
            for w in ocr_words:
                pg = w.get("page", 1)
                pages.setdefault(pg, []).append(w)

            for page_num, page_words in pages.items():
                img_path = doc_img_dir / f"page_{page_num:03d}.png"
                if not img_path.exists():
                    continue

                # Assign BIO tags
                bio_tags = self._assign_bio_tags(page_words, gt_entities, page_num)

                samples.append({
                    "image_path": str(img_path),
                    "words": [w["text"] for w in page_words],
                    "boxes": [w["bbox"] for w in page_words],
                    "bio_tags": bio_tags,
                })

        return samples

    def _assign_bio_tags(
        self,
        words: list[dict],
        gt_entities: list[dict],
        page_num: int,
    ) -> list[str]:
        """
        Assign BIO tags by checking overlap between OCR word bboxes and GT entity bboxes.
        """
        tags = ["O"] * len(words)

        # Filter GT entities for this page
        page_entities = [
            e for e in gt_entities
            if e.get("page", 1) == page_num
        ]

        for entity in page_entities:
            entity_type = entity.get("label", entity.get("type", ""))
            entity_bbox = entity.get("bbox")
            if not entity_bbox:
                continue

            first_match = True
            for i, word in enumerate(words):
                if self._bbox_overlap(word["bbox"], entity_bbox):
                    if first_match:
                        tags[i] = f"B-{entity_type}"
                        first_match = False
                    else:
                        tags[i] = f"I-{entity_type}"

        return tags

    @staticmethod
    def _bbox_overlap(box_a: list[int], box_b: list[int], threshold: float = 0.3) -> bool:
        """Check if two normalised bboxes have sufficient overlap (IoU-like)."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        if x2 <= x1 or y2 <= y1:
            return False

        inter = (x2 - x1) * (y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])

        if area_a == 0:
            return False

        return (inter / area_a) >= threshold

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")

        encoding = self.processor(
            image,
            sample["words"],
            boxes=sample["boxes"],
            word_labels=[LABEL2ID.get(t, 0) for t in sample["bio_tags"]],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Squeeze batch dim added by processor
        return {k: v.squeeze(0) for k, v in encoding.items()}
