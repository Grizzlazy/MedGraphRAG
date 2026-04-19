"""
main.py — CLI entry point cho RAG pipeline.

Usage:
  # Inference nhanh
  python -m rag.main --data-path ./dataset/mimic_ex \\
      --query "What medications is the patient currently taking?"

  # Evaluate trên MedQA (50 mẫu)
  python -m rag.main --data-path ./dataset/mimic_ex \\
      --eval medqa --max-samples 50

  # Full 3 benchmark
  python -m rag.main --data-path ./dataset/mimic_ex \\
      --eval medqa medmcqa pubmedqa

  # Build lại index từ đầu (xóa cache)
  python -m rag.main --data-path ./dataset/mimic_ex --rebuild \\
      --query "What is the diagnosis?"
"""

import argparse
import os
import shutil
from datetime import datetime

import pandas as pd

from .pipeline import RAGPipeline
from .generator import build_context, generate_answer
from .evaluator import evaluate_dataset


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Traditional RAG — FAISS + BM25 + Cross-encoder Reranking",
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
        help="Thư mục cache FAISS + BM25 index (default: .rag_cache)",
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
        choices=["medqa", "medmcqa", "pubmedqa"],
        metavar="DS",
        help="Datasets để evaluate. Chọn từ: medqa, medmcqa, pubmedqa",
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
            score = f"rrf={c['rrf_score']:.4f}"
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
        # Tạo thư mục con theo timestamp để không ghi đè lần eval trước
        run_id      = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir     = os.path.join(args.results_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        print(f"\nRun ID: {run_id}  →  {run_dir}")

        summary = []
        for ds in args.eval:
            print(f"\n{'='*55}")
            print(f"Evaluating: {ds}")
            print(f"{'='*55}")

            acc = evaluate_dataset(
                rag,
                dataset_name=ds,
                max_samples=args.max_samples,
                results_dir=run_dir,
                eval_dir=args.eval_dir,
            )
            summary.append({"dataset": ds, "accuracy(%)": round(acc, 2)})

        print("\n" + "=" * 55)
        print("SUMMARY — Traditional RAG (FAISS + Rerank)")
        print("=" * 55)
        for row in summary:
            print(f"  {row['dataset']:20s}  {row['accuracy(%)']:.2f}%")

        summary_path = os.path.join(run_dir, "rag_summary.csv")
        pd.DataFrame(summary).to_csv(summary_path, index=False)
        print(f"\nSummary saved → {summary_path}")
        return

    print("Không có --query hay --eval. Dùng --help để xem hướng dẫn.")


if __name__ == "__main__":
    main()
