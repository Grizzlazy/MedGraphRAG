"""
Stage 1 — Trainer
Fine-tunes LayoutLMv3 for PII NER with early stopping.
"""

from pathlib import Path
import numpy as np
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import Dataset as HFDataset
from seqeval.metrics import (
    classification_report,
    f1_score as seqeval_f1,
    precision_score as seqeval_precision,
    recall_score as seqeval_recall,
)
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    LABEL_LIST, LABEL2ID, ID2LABEL, NUM_LABELS,
    LAYOUTLMV3_MODEL, CHECKPOINT_DIR,
    TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE,
    LEARNING_RATE, WEIGHT_DECAY, WARMUP_RATIO,
    NUM_EPOCHS, EARLY_STOPPING_PATIENCE,
)
from src.stage1_deid.bio_data_loader import DeIDDataset


# ======================================================================
# Metrics
# ======================================================================
def compute_metrics(eval_pred):
    """Compute seqeval metrics for NER."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    true_labels = []
    true_preds = []

    for pred_seq, label_seq in zip(predictions, labels):
        seq_labels = []
        seq_preds = []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            seq_labels.append(LABEL_LIST[l])
            seq_preds.append(LABEL_LIST[p] if p < len(LABEL_LIST) else "O")
        true_labels.append(seq_labels)
        true_preds.append(seq_preds)

    return {
        "precision": seqeval_precision(true_labels, true_preds),
        "recall": seqeval_recall(true_labels, true_preds),
        "f1": seqeval_f1(true_labels, true_preds),
    }


# ======================================================================
# Training function
# ======================================================================
def train(
    train_dataset: DeIDDataset,
    val_dataset: DeIDDataset,
    output_dir: str | None = None,
    resume_from_checkpoint: str | None = None,
):
    """
    Fine-tune LayoutLMv3 on the PII NER task.

    Args:
        train_dataset: Training DeIDDataset (Easy split).
        val_dataset:   Validation DeIDDataset (Medium split).
        output_dir:    Where to save checkpoints.
        resume_from_checkpoint: Path to resume training.
    """
    output_dir = output_dir or str(CHECKPOINT_DIR / "layoutlmv3-deid")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Initialising LayoutLMv3 for token classification …")
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        LAYOUTLMV3_MODEL,
        num_labels=NUM_LABELS,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="linear",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=True,
        logging_steps=10,
        logging_dir=str(Path(output_dir) / "logs"),
        remove_unused_columns=False,
        save_total_limit=3,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    logger.info("Starting training …")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save best model
    best_dir = str(Path(output_dir) / "best")
    trainer.save_model(best_dir)
    logger.info(f"Best model saved to {best_dir}")

    # Log final metrics
    metrics = trainer.evaluate()
    logger.info(f"Final eval metrics: {metrics}")

    return trainer, metrics


# ======================================================================
# CLI
# ======================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train LayoutLMv3 DeID NER")
    parser.add_argument("--ocr_dir", required=True, help="OCR JSON directory")
    parser.add_argument("--image_dir", required=True, help="Page images directory")
    parser.add_argument("--train_gt", required=True, help="Training GT JSON")
    parser.add_argument("--val_gt", required=True, help="Validation GT JSON")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    train_ds = DeIDDataset(args.ocr_dir, args.image_dir, args.train_gt)
    val_ds = DeIDDataset(args.ocr_dir, args.image_dir, args.val_gt)

    train(train_ds, val_ds, args.output_dir)
