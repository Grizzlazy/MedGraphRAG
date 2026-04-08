"""
Stage 1 — Evaluation
Computes Precision / Recall / F1 for de-identification NER.
"""

import json
from pathlib import Path
from collections import defaultdict
from loguru import logger


def compute_entity_metrics(
    predicted: list[dict],
    ground_truth: list[dict],
    strict: bool = True,
) -> dict:
    """
    Evaluate NER predictions against ground-truth entities.

    Args:
        predicted:    [{"text": …, "label": …, "start": …, "end": …}, …]
        ground_truth: Same format.
        strict:       If True, both span and type must match exactly.
                      If False, any overlap counts as a match (lenient).

    Returns:
        { "micro": {P, R, F1}, "per_type": { "PATIENT_NAME": {P, R, F1, …}, … } }
    """
    tp_per_type: dict[str, int] = defaultdict(int)
    fp_per_type: dict[str, int] = defaultdict(int)
    fn_per_type: dict[str, int] = defaultdict(int)

    # Build sets for matching
    gt_set = set()
    for e in ground_truth:
        key = (e["label"], e["start"], e["end"]) if strict else (e["label"],)
        gt_set.add((e["label"], e["start"], e["end"]))

    pred_set = set()
    for e in predicted:
        pred_set.add((e["label"], e["start"], e["end"]))

    matched_gt = set()
    matched_pred = set()

    for p in pred_set:
        if strict:
            if p in gt_set:
                tp_per_type[p[0]] += 1
                matched_gt.add(p)
                matched_pred.add(p)
            else:
                fp_per_type[p[0]] += 1
        else:
            # Lenient: any overlap
            found = False
            for g in gt_set:
                if g in matched_gt:
                    continue
                if p[0] == g[0] and _spans_overlap(p[1], p[2], g[1], g[2]):
                    tp_per_type[p[0]] += 1
                    matched_gt.add(g)
                    matched_pred.add(p)
                    found = True
                    break
            if not found:
                fp_per_type[p[0]] += 1

    for g in gt_set:
        if g not in matched_gt:
            fn_per_type[g[0]] += 1

    # Aggregate
    all_types = set(list(tp_per_type) + list(fp_per_type) + list(fn_per_type))
    per_type = {}
    total_tp = total_fp = total_fn = 0

    for t in sorted(all_types):
        tp = tp_per_type[t]
        fp = fp_per_type[t]
        fn = fn_per_type[t]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_type[t] = {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
        total_tp += tp
        total_fp += fp
        total_fn += fn

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    return {
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "per_type": per_type,
        "total": {"tp": total_tp, "fp": total_fp, "fn": total_fn},
    }


def _spans_overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
    return s1 < e2 and s2 < e1


def print_report(metrics: dict) -> None:
    """Pretty-print the evaluation report."""
    print("\n" + "=" * 70)
    print(f"{'Entity Type':<25} {'Prec':>8} {'Rec':>8} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 70)
    for etype, m in sorted(metrics["per_type"].items()):
        print(
            f"{etype:<25} {m['precision']:8.4f} {m['recall']:8.4f} "
            f"{m['f1']:8.4f} {m['tp']:5d} {m['fp']:5d} {m['fn']:5d}"
        )
    print("-" * 70)
    micro = metrics["micro"]
    total = metrics["total"]
    print(
        f"{'MICRO AVG':<25} {micro['precision']:8.4f} {micro['recall']:8.4f} "
        f"{micro['f1']:8.4f} {total['tp']:5d} {total['fp']:5d} {total['fn']:5d}"
    )
    print("=" * 70 + "\n")


# ======================================================================
# CLI
# ======================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate DeID NER")
    parser.add_argument("--pred", required=True, help="Predictions JSON")
    parser.add_argument("--gt", required=True, help="Ground truth JSON")
    parser.add_argument("--lenient", action="store_true", help="Use lenient matching")
    args = parser.parse_args()

    with open(args.pred) as f:
        preds = json.load(f)
    with open(args.gt) as f:
        gts = json.load(f)

    metrics = compute_entity_metrics(preds, gts, strict=not args.lenient)
    print_report(metrics)
