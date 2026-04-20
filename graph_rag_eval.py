"""
graph_rag_eval.py — MCQ evaluation cho Medical-Graph-RAG (Neo4j + CAMEL pipeline).

Cùng interface với rag/main.py nhưng dùng Neo4j graph retrieval thay FAISS.

Usage:
  # Evaluate 1 dataset (test nhanh)
  python graph_rag_eval.py --eval medqa --max-samples 10 --workers 1

  # Evaluate 3 bộ gốc
  python graph_rag_eval.py --eval medqa medmcqa pubmedqa

  # Evaluate 6 MMLU medical subsets
  python graph_rag_eval.py --eval mmlu_anatomy mmlu_clinical_knowledge \
      mmlu_college_biology mmlu_college_medicine \
      mmlu_medical_genetics mmlu_professional_medicine

  # Evaluate full 9 bộ
  python graph_rag_eval.py --eval medqa medmcqa pubmedqa \
      mmlu_anatomy mmlu_clinical_knowledge mmlu_college_biology \
      mmlu_college_medicine mmlu_medical_genetics mmlu_professional_medicine

  # Giới hạn số mẫu
  python graph_rag_eval.py --eval medqa --max-samples 50

  # Inference nhanh (không eval)
  python graph_rag_eval.py --query "What is the main symptom of the patient?"

  # Legacy single-dataset mode (backward compat)
  python graph_rag_eval.py --dataset medqa --max-samples 10 --workers 1
"""

import argparse
import logging
import os
import re
import traceback

# Suppress Neo4j deprecation/notification warnings (id() deprecated, etc.)
logging.getLogger("neo4j").setLevel(logging.ERROR)
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from datasets import load_from_disk
from dotenv import load_dotenv
from tqdm import tqdm

from camel.storages import Neo4jGraph
from summerize import process_chunks
from retrieve import seq_ret
from utils import get_response

# ── Constants ─────────────────────────────────────────────────────────────────
OPTION_LETTERS = ["A", "B", "C", "D", "E"]

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


# ── Dataset formatting ────────────────────────────────────────────────────────
def format_sample(dataset_name: str, item) -> Dict:
    """Chuẩn hóa 1 sample về dạng thống nhất."""
    if dataset_name == "medqa":
        d = item["data"]
        return {
            "question": d["Question"],
            "options":  d["Options"],
            "answer":   str(d["Correct Option"]).strip().upper(),
        }

    if dataset_name == "medmcqa":
        opts = {
            "A": item["opa"], "B": item["opb"],
            "C": item["opc"], "D": item["opd"],
        }
        ans = {0: "A", 1: "B", 2: "C", 3: "D"}.get(item["cop"], "A")
        return {"question": item["question"], "options": opts, "answer": ans}

    if dataset_name == "pubmedqa":
        d   = item["data"]
        ctx = d.get("Context", [])
        if isinstance(ctx, list):
            ctx = " ".join(ctx)
        return {
            "question": d["Question"],
            "options":  d["Options"],
            "answer":   str(d["Correct Option"]).strip().upper(),
            "context":  ctx,
        }

    # mmlu_anatomy, mmlu_clinical_knowledge, mmlu_college_biology,
    # mmlu_college_medicine, mmlu_medical_genetics, mmlu_professional_medicine
    if dataset_name.startswith("mmlu_"):
        d = item["data"]
        return {
            "question": d["Question"],
            "options":  d["Options"],
            "answer":   str(d["Correct Option"]).strip().upper(),
        }

    raise ValueError(f"Unknown dataset: {dataset_name!r}")


# ── Helper functions ──────────────────────────────────────────────────────────
def _strip_think(text: str) -> str:
    """Xóa <think>...</think> block (Qwen3 thinking mode)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


def extract_letter(response: str, options: Dict) -> Optional[str]:
    cleaned = _strip_think(response)
    text = cleaned.strip().upper()

    if text and text[0] in options:
        return text[0]

    for pat in [r"answer[:\s]+([A-E])", r"\b([A-E])\b", r"\(([A-E])\)", r"\*\*([A-E])\*\*"]:
        m = re.search(pat, text)
        if m and m.group(1) in options:
            return m.group(1)

    low = cleaned.lower()
    for letter, opt_text in options.items():
        if str(opt_text).lower()[:40] in low:
            return letter
    return None


def build_mcq_prompt(question: str, options: Dict, context: str = "") -> str:
    ctx_block = f"Context:\n{context}\n\n" if context else ""
    opt_block = "\n".join(f"{k}. {v}" for k, v in options.items())
    return (
        f"{ctx_block}Question: {question}\n\n"
        f"Options:\n{opt_block}\n\n"
        "Reply with ONLY the single option letter (e.g. A). Your answer:"
    )


def load_local_dataset(name: str, eval_dir: str):
    local_path = os.path.join(eval_dir, name)
    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"Dataset '{name}' chưa có tại '{local_path}'.\n"
            f"Chạy: python evaluation/download.py"
        )
    print(f"  Loading {name} from {local_path} ...")
    return load_from_disk(local_path)


def make_neo4j() -> Neo4jGraph:
    url = os.getenv("NEO4J_URL")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    return Neo4jGraph(url=url, username=username, password=password)


# ── Evaluation core ──────────────────────────────────────────────────────────
def run_one(n4j: Neo4jGraph, dataset_name: str, idx: int, raw_item: dict) -> dict:
    sample = format_sample(dataset_name, raw_item)
    question = sample["question"]
    options = sample["options"]
    context = sample.get("context", "")
    prompt = build_mcq_prompt(question, options, context)

    try:
        summary = process_chunks(prompt)
        gid = seq_ret(n4j, summary)
        response = get_response(n4j, gid, prompt)
        predicted = extract_letter(response, options)
    except Exception as exc:
        print(f"\n  [WARN] sample {idx}: {exc}")
        traceback.print_exc()
        response = ""
        predicted = None

    return {
        "index": idx,
        "question": question[:300],
        "ground_truth": sample["answer"],
        "predicted": predicted,
        "correct": predicted == sample["answer"],
        "raw_response": _strip_think(str(response))[:400],
    }


def evaluate_dataset(
    n4j: Neo4jGraph,
    dataset_name: str,
    max_samples: Optional[int] = None,
    results_dir: str = "./camel_eval_results",
    eval_dir: str = "./evaluation/test_qa",
    num_workers: int = 2,
) -> float:
    """Chạy MCQ evaluation cho 1 dataset và lưu kết quả CSV."""
    ds = load_local_dataset(dataset_name, eval_dir)
    items = list(ds)
    if max_samples:
        items = items[:max_samples]

    print(f"\n  [{dataset_name}] {len(items)} samples | workers={num_workers}")

    results: List[dict] = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [
            pool.submit(run_one, n4j, dataset_name, i, item)
            for i, item in enumerate(items)
        ]
        for f in tqdm(futures, total=len(futures), desc=f"  {dataset_name}"):
            results.append(f.result())

    records = sorted(results, key=lambda r: r["index"])
    n_correct = sum(1 for r in records if r["correct"])
    accuracy = n_correct / len(records) * 100 if records else 0.0

    print(f"\n  [{dataset_name}] Accuracy: {accuracy:.2f}%  ({n_correct}/{len(records)})")

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"camel_{dataset_name}.csv")
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")

    return accuracy


# ── CLI ───────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Medical-Graph-RAG Evaluation (Neo4j + CAMEL pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    data = p.add_argument_group("Data")
    data.add_argument(
        "--eval-dir", default="./evaluation/test_qa",
        help="Thư mục chứa datasets local (default: ./evaluation/test_qa)",
    )

    mode = p.add_argument_group("Mode")
    mode.add_argument(
        "--query", type=str, default=None,
        help="Câu hỏi để thử inference (in ra câu trả lời)",
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
    # Legacy: --dataset (backward compat)
    mode.add_argument(
        "--dataset", nargs="+", default=None,
        help="(Legacy) Tương đương với --eval. Dùng --eval thay thế.",
    )

    ctrl = p.add_argument_group("Controls")
    ctrl.add_argument(
        "--max-samples", type=int, default=None,
        help="Giới hạn số mẫu mỗi dataset (None = toàn bộ)",
    )
    ctrl.add_argument(
        "--workers", type=int, default=16,
        help="Số worker threads song song khi eval (default: 16)",
    )
    ctrl.add_argument(
        "--results-dir", "--out-dir", default="./camel_eval_results",
        help="Thư mục lưu CSV kết quả (default: ./camel_eval_results)",
    )

    return p


def main():
    load_dotenv(Path(__file__).resolve().parent / ".env")
    args = build_parser().parse_args()

    # Merge --dataset (legacy) into --eval
    eval_datasets = args.eval or args.dataset
    if not eval_datasets and not args.query:
        print("Không có --eval, --dataset hay --query. Dùng --help để xem hướng dẫn.")
        return

    # ── Connect Neo4j ────────────────────────────────────────────────────────
    n4j = make_neo4j()

    # ── Inference mode ───────────────────────────────────────────────────────
    if args.query:
        # print(f"\nQuery: {args.query}")
        # print("─" * 60)
        summary = process_chunks(args.query)
        gid = seq_ret(n4j, summary)
        response = get_response(n4j, gid, args.query)
        # print(f"\nAnswer:\n{response}\n")
        if not eval_datasets:
            return

    # ── Evaluation mode ──────────────────────────────────────────────────────
    if eval_datasets:
        run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.results_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        print(f"\nRun ID: {run_id}  →  {run_dir}")

        summary = []
        for ds in eval_datasets:
            print(f"\n{'='*60}")
            print(f"Evaluating: {ds}")
            print(f"{'='*60}")

            try:
                acc = evaluate_dataset(
                    n4j,
                    dataset_name=ds,
                    max_samples=args.max_samples,
                    results_dir=run_dir,
                    eval_dir=args.eval_dir,
                    num_workers=args.workers,
                )
                summary.append({"dataset": ds, "accuracy(%)": round(acc, 2), "status": "ok"})
            except Exception as e:
                print(f"\n  [ERROR] {ds}: {e}")
                traceback.print_exc()
                summary.append({"dataset": ds, "accuracy(%)": 0.0, "status": str(e)[:80]})

        # ── Print summary table ──────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("SUMMARY — Medical-Graph-RAG (Neo4j + CAMEL)")
        print("=" * 60)

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

        summary_path = os.path.join(run_dir, "camel_summary.csv")
        pd.DataFrame(summary).to_csv(summary_path, index=False)
        print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    main()