"""
main.py — CLI entry point cho RAG pipeline.

Usage:
  # Inference nhanh
  python -m rag.main --query "What medications is the patient currently taking?"

  # Evaluate 3 bộ gốc
  python -m rag.main --eval medqa medmcqa pubmedqa

  # Evaluate 6 MMLU medical subsets
  python -m rag.main --eval mmlu_anatomy mmlu_clinical_knowledge \\
      mmlu_college_biology mmlu_college_medicine \\
      mmlu_medical_genetics mmlu_professional_medicine

  # Evaluate full 9 bộ
  python -m rag.main --eval medqa medmcqa pubmedqa \\
      mmlu_anatomy mmlu_clinical_knowledge mmlu_college_biology \\
      mmlu_college_medicine mmlu_medical_genetics mmlu_professional_medicine

  # Giới hạn số mẫu (test nhanh)
  python -m rag.main --eval medqa --max-samples 50

  # Build lại index từ đầu
  python -m rag.main --rebuild --query "What is the diagnosis?"
"""

import argparse
import os
import shutil
from datetime import datetime

import pandas as pd

from .pipeline import RAGPipeline
from .generator import build_context, generate_answer
from .evaluator import evaluate_dataset


VALID_DATASETS = [
    # Core medical QA
    "medqa",
    "medmcqa",
    "pubmedqa",
    # MMLU medical subsets
    "mmlu_anatomy",
    "mmlu_clinical_knowledge",
    "mmlu_college_biology",
    "mmlu_college_medicine",
    "mmlu_medical_genetics",
    "mmlu_professional_medicine",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Traditional RAG — FAISS + Cross-encoder Reranking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    data = p.add_argument_group("Data")
    data.add_argument(
        "--data-path", default="./dataset/",
        help="Thư mục chứa .txt files (mimic_ex format)",
    )
    data.add_argument(
        "--cache-dir", default=".rag_cache",
        help="Thư mục cache FAISS index (default: .rag_cache)",
    )
    data.add_argument(
        "--rebuild", action="store_true",
        help="Xóa cache và build lại index từ đầu",
    )
    data.add_argument(
        "--eval-dir", default="./evaluation/test_qa",
        help="Thư mục chứa datasets đã download (default: ./evaluation/test_qa)",
    )

    mode = p.add_argument_group("Mode")
    mode.add_argument(
        "--query", type=str, default=None,
        help="Câu hỏi để thử inference (in ra chunks + câu trả lời)",
    )
    mode.add_argument(
        "--eval", nargs="+",
        choices=VALID_DATASETS,
        metavar="DS",
        help=(
            "Datasets để evaluate. Chọn từ:\n"
            "  Core: medqa, medmcqa, pubmedqa\n"
            "  MMLU: mmlu_anatomy, mmlu_clinical_knowledge, mmlu_college_biology,\n"
            "        mmlu_college_medicine, mmlu_medical_genetics, mmlu_professional_medicine"
        ),
    )

    ctrl = p.add_argument_group("Controls")
    ctrl.add_argument(
        "--max-samples", type=int, default=None,
        help="Giới hạn số mẫu mỗi dataset (None = toàn bộ)",
    )
    ctrl.add_argument(
        "--results-dir", default="./rag_eval_results",
        help="Thư mục lưu CSV kết quả (default: ./rag_eval_results)",
    )

    return p


def main():
    args = build_parser().parse_args()

    # ── Rebuild cache nếu yêu cầu ─────────────────────────────────────────────
    if args.rebuild and os.path.exists(args.cache_dir):
        shutil.rmtree(args.cache_dir)
        print(f"Cache {args.cache_dir!r} đã xóa — sẽ build lại.\n")

    # ── Build index ───────────────────────────────────────────────────────────
    rag = RAGPipeline()
    rag.build(args.data_path, cache_dir=args.cache_dir)

    # ── Inference mode ────────────────────────────────────────────────────────
    if args.query:
        print(f"\nQuery: {args.query}")
        print("─" * 60)

        top = rag.retrieve(args.query)

        print(f"Top {len(top)} chunks retrieved:\n")
        for i, c in enumerate(top, 1):
            src   = c["chunk"]["source"]
            score = f"score={c.get('dense_score', c.get('rrf_score', 0)):.4f}"
            if "ce_score" in c:
                score += f", ce={c['ce_score']:.3f}"
            print(f"  [{i}] {src} ({score})")
            print(f"       {c['chunk']['text'][:180].strip()}...\n")

        print("─" * 60)
        answer = generate_answer(args.query, top)
        print(f"\nAnswer:\n{answer}\n")
        return

    # ── Evaluation mode ───────────────────────────────────────────────────────
    if args.eval:
        run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.results_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        print(f"\nRun ID: {run_id}  →  {run_dir}")

        summary = []
        for ds in args.eval:
            print(f"\n{'='*60}")
            print(f"Evaluating: {ds}")
            print(f"{'='*60}")

            try:
                acc = evaluate_dataset(
                    rag,
                    dataset_name=ds,
                    max_samples=args.max_samples,
                    results_dir=run_dir,
                    eval_dir=args.eval_dir,
                )
                summary.append({"dataset": ds, "accuracy(%)": round(acc, 2), "status": "ok"})
            except Exception as e:
                print(f"\n  [ERROR] {ds}: {e}")
                summary.append({"dataset": ds, "accuracy(%)": 0.0, "status": str(e)[:80]})

        # ── Print summary table ───────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("SUMMARY — Traditional RAG (FAISS + CrossEncoder Rerank)")
        print("=" * 60)

        # Group: core vs mmlu
        core_rows = [r for r in summary if not r["dataset"].startswith("mmlu_")]
        mmlu_rows = [r for r in summary if r["dataset"].startswith("mmlu_")]

        if core_rows:
            print("\n  Core Medical QA:")
            for row in core_rows:
                status = "" if row["status"] == "ok" else f"  ← {row['status']}"
                print(f"    {row['dataset']:30s}  {row['accuracy(%)']:6.2f}%{status}")

        if mmlu_rows:
            print("\n  MMLU Medical:")
            for row in mmlu_rows:
                status = "" if row["status"] == "ok" else f"  ← {row['status']}"
                print(f"    {row['dataset']:30s}  {row['accuracy(%)']:6.2f}%{status}")

            mmlu_avg = sum(r["accuracy(%)"] for r in mmlu_rows) / len(mmlu_rows)
            print(f"    {'MMLU Average':30s}  {mmlu_avg:6.2f}%")

        if summary:
            total_avg = sum(r["accuracy(%)"] for r in summary) / len(summary)
            print(f"\n  {'Overall Average':30s}  {total_avg:6.2f}%")

        print("=" * 60)

        summary_path = os.path.join(run_dir, "rag_summary.csv")
        pd.DataFrame(summary).to_csv(summary_path, index=False)
        print(f"\nSummary saved → {summary_path}")
        return

    print("Không có --query hay --eval. Dùng --help để xem hướng dẫn.")


if __name__ == "__main__":
    main()
