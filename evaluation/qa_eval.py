"""
Evaluation — QA Metrics
Evaluates GraphRAG QA answers using ROUGE-L and BERTScore.
"""

import json
from pathlib import Path
from loguru import logger
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def compute_rouge_l(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE-L for a list of prediction-reference pairs."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = {"precision": [], "recall": [], "fmeasure": []}

    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        scores["precision"].append(result["rougeL"].precision)
        scores["recall"].append(result["rougeL"].recall)
        scores["fmeasure"].append(result["rougeL"].fmeasure)

    return {
        "rouge_l_precision": sum(scores["precision"]) / len(scores["precision"]),
        "rouge_l_recall": sum(scores["recall"]) / len(scores["recall"]),
        "rouge_l_f1": sum(scores["fmeasure"]) / len(scores["fmeasure"]),
    }


def compute_bert_score(predictions: list[str], references: list[str], lang: str = "en") -> dict:
    """Compute BERTScore for a list of prediction-reference pairs."""
    P, R, F1 = bert_score_fn(predictions, references, lang=lang, verbose=False)
    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item(),
    }


def evaluate_qa(
    questions_file: str,
    predictions: list[dict] | None = None,
    qa_system=None,
) -> dict:
    """
    Evaluate QA system on a test set.

    Args:
        questions_file: JSON with [{"question": ..., "reference_answer": ..., "type": ...}, ...]
        predictions:    Pre-computed predictions [{"answer": ...}, ...] (optional).
        qa_system:      Live system to generate answers if predictions not provided.

    Returns:
        Aggregated metrics.
    """
    with open(questions_file, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    if predictions is None and qa_system:
        predictions = []
        for item in test_set:
            result = qa_system.query(item["question"])
            predictions.append({"answer": result["answer"]})

    if predictions is None:
        raise ValueError("Must provide either predictions or qa_system")

    pred_texts = [p["answer"] for p in predictions]
    ref_texts = [t["reference_answer"] for t in test_set]

    # Compute metrics
    rouge = compute_rouge_l(pred_texts, ref_texts)
    bertscore = compute_bert_score(pred_texts, ref_texts)

    # Per-type breakdown
    type_metrics: dict[str, dict] = {}
    for item, pred in zip(test_set, predictions):
        qtype = item.get("type", "unknown")
        type_metrics.setdefault(qtype, {"preds": [], "refs": []})
        type_metrics[qtype]["preds"].append(pred["answer"])
        type_metrics[qtype]["refs"].append(item["reference_answer"])

    per_type = {}
    for qtype, data in type_metrics.items():
        per_type[qtype] = compute_rouge_l(data["preds"], data["refs"])
        per_type[qtype]["count"] = len(data["preds"])

    results = {
        "overall": {**rouge, **bertscore},
        "per_type": per_type,
        "num_questions": len(test_set),
    }

    # Print report
    print("\n" + "=" * 60)
    print("QA Evaluation Report")
    print("=" * 60)
    print(f"  Questions:       {results['num_questions']}")
    print(f"  ROUGE-L F1:      {rouge['rouge_l_f1']:.4f}")
    print(f"  BERTScore F1:    {bertscore['bertscore_f1']:.4f}")
    print("-" * 60)
    for qtype, m in per_type.items():
        print(f"  [{qtype}] n={m['count']}, ROUGE-L={m['rouge_l_f1']:.4f}")
    print("=" * 60 + "\n")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate QA system")
    parser.add_argument("--questions", required=True, help="Test questions JSON")
    parser.add_argument("--predictions", required=True, help="Predictions JSON")
    args = parser.parse_args()

    with open(args.predictions) as f:
        preds = json.load(f)

    evaluate_qa(args.questions, preds)
