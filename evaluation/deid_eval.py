"""
Evaluation — De-identification Metrics
End-to-end evaluation of the de-identification pipeline.
"""

import json
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stage1_deid.evaluate import compute_entity_metrics, print_report
from src.stage1_deid.pdf_to_image import pdf_to_images
from src.stage1_deid.ocr_processor import ocr_document, ocr_page
from src.stage1_deid.layoutlmv3_ner import LayoutLMv3NER


def evaluate_deid_pipeline(
    pdf_dir: str,
    gt_json_path: str,
    model_path: str | None = None,
    difficulty: str = "easy",
) -> dict:
    """
    Run end-to-end de-identification evaluation.

    Args:
        pdf_dir:      Directory containing test PDFs.
        gt_json_path: Ground truth JSON file.
        model_path:   Path to fine-tuned LayoutLMv3 checkpoint.
        difficulty:   Label for reporting (easy / medium / hard).

    Returns:
        Aggregated metrics dict.
    """
    with open(gt_json_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    ner_model = LayoutLMv3NER(model_path)

    all_preds: list[dict] = []
    all_gts: list[dict] = []

    pdf_dir = Path(pdf_dir)
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        doc_name = pdf_path.name
        gt_entities = ground_truth.get(doc_name, [])

        if not gt_entities:
            logger.warning(f"No ground truth for {doc_name}, skipping")
            continue

        # Pipeline
        image_paths = pdf_to_images(str(pdf_path))
        ocr_results = [ocr_page(str(p)) for p in image_paths]
        predictions = ner_model.predict_document(
            [str(p) for p in image_paths], ocr_results,
        )

        all_preds.extend(predictions)
        all_gts.extend(gt_entities)

        logger.info(f"{doc_name}: {len(predictions)} predicted, {len(gt_entities)} ground truth")

    # Compute metrics
    metrics = compute_entity_metrics(all_preds, all_gts, strict=True)
    metrics["difficulty"] = difficulty
    metrics["num_documents"] = len(list(pdf_dir.glob("*.pdf")))

    print(f"\n{'='*40} {difficulty.upper()} {'='*40}")
    print_report(metrics)

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate DeID pipeline")
    parser.add_argument("--pdf_dir", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--difficulty", default="easy")
    args = parser.parse_args()

    evaluate_deid_pipeline(args.pdf_dir, args.gt, args.model, args.difficulty)
