"""
Evaluation — Faithfulness Score
Measures how well answers are grounded in the knowledge graph.
"""

import json
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stage4_graphrag_qa.hallucination_checker import HallucinationChecker


def evaluate_faithfulness(
    answers_file: str,
    neo4j_client=None,
) -> dict:
    """
    Evaluate faithfulness of a set of answers.

    Args:
        answers_file: JSON [{"question": ..., "answer": ...}, ...]

    Returns:
        Aggregated faithfulness metrics.
    """
    with open(answers_file, "r", encoding="utf-8") as f:
        answers = json.load(f)

    checker = HallucinationChecker(neo4j_client)

    results: list[dict] = []
    for item in answers:
        check = checker.check_answer(item["answer"])
        check["question"] = item.get("question", "")
        results.append(check)

    # Aggregate
    scores = [r["faithfulness_score"] for r in results]
    total_claims = sum(r["total_claims"] for r in results)
    verified_claims = sum(r["verified_claims"] for r in results)
    all_flagged = [c for r in results for c in r.get("flagged_claims", [])]

    avg_score = sum(scores) / len(scores) if scores else 0.0

    report = {
        "avg_faithfulness_score": avg_score,
        "total_answers": len(answers),
        "total_claims": total_claims,
        "verified_claims": verified_claims,
        "total_flagged": len(all_flagged),
        "per_answer": results,
    }

    # Print
    print("\n" + "=" * 60)
    print("Faithfulness Evaluation Report")
    print("=" * 60)
    print(f"  Answers evaluated:    {len(answers)}")
    print(f"  Avg faithfulness:     {avg_score:.4f}")
    print(f"  Total claims:         {total_claims}")
    print(f"  Verified claims:      {verified_claims}")
    print(f"  Flagged claims:       {len(all_flagged)}")
    print("=" * 60)

    # Show worst answers
    worst = sorted(results, key=lambda r: r["faithfulness_score"])[:5]
    if worst:
        print("\nLowest faithfulness scores:")
        for w in worst:
            print(f"  Score={w['faithfulness_score']:.2f} | Q: {w['question'][:60]}…")
    print()

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate faithfulness")
    parser.add_argument("--answers", required=True, help="Answers JSON file")
    args = parser.parse_args()

    evaluate_faithfulness(args.answers)
