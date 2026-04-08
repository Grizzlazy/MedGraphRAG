"""
Stage 1 — OCR Processor
Extracts text and bounding boxes from document images using Tesseract.
"""

from pathlib import Path
from PIL import Image
import pytesseract
from loguru import logger


def ocr_page(image_path: str, lang: str = "eng") -> list[dict]:
    """
    Run Tesseract OCR on a single page image.

    Returns a list of word-level dicts:
        [{"text": "John", "bbox": [x_min, y_min, x_max, y_max], "confidence": 95}, …]
    
    Bounding boxes are normalised to [0, 1000] scale as required by LayoutLMv3.
    """
    img = Image.open(image_path)
    width, height = img.size

    # Tesseract returns detailed word-level data
    data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)

    words: list[dict] = []
    n = len(data["text"])

    for i in range(n):
        text = data["text"][i].strip()
        if not text:
            continue

        conf = int(data["conf"][i])
        if conf < 0:
            continue

        # Raw pixel coordinates
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]

        # Normalise to [0, 1000]
        x_min = int(1000 * x / width)
        y_min = int(1000 * y / height)
        x_max = int(1000 * (x + w) / width)
        y_max = int(1000 * (y + h) / height)

        # Clamp
        x_min = max(0, min(x_min, 1000))
        y_min = max(0, min(y_min, 1000))
        x_max = max(0, min(x_max, 1000))
        y_max = max(0, min(y_max, 1000))

        words.append({
            "text": text,
            "bbox": [x_min, y_min, x_max, y_max],
            "confidence": conf,
            "block_num": data["block_num"][i],
            "line_num": data["line_num"][i],
            "word_num": data["word_num"][i],
        })

    logger.debug(f"OCR extracted {len(words)} words from {Path(image_path).name}")
    return words


def ocr_document(image_paths: list[str], lang: str = "eng") -> list[dict]:
    """
    Run OCR on all pages of a document.

    Returns a flat list of words with an additional 'page' key.
    """
    all_words: list[dict] = []
    for page_idx, img_path in enumerate(image_paths):
        page_words = ocr_page(img_path, lang)
        for w in page_words:
            w["page"] = page_idx + 1
        all_words.extend(page_words)
    logger.info(f"OCR complete: {len(all_words)} words across {len(image_paths)} pages")
    return all_words


def reconstruct_text(words: list[dict]) -> str:
    """Reconstruct full text from OCR word list (simple whitespace join)."""
    return " ".join(w["text"] for w in words)


# ======================================================================
# CLI
# ======================================================================
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Run OCR on document images")
    parser.add_argument("--image_dir", required=True, help="Dir of page images")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    img_dir = Path(args.image_dir)
    pages = sorted(img_dir.glob("*.png"))
    words = ocr_document([str(p) for p in pages])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(words)} words to {args.output}")
