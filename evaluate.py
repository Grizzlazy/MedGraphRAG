"""
evaluate.py — Reproduce MedGraphRAG paper evaluation metrics.

Paper metrics (arXiv:2408.04187):
  Table 1 — Multiple-choice accuracy (%) on:
    * MultiMedQA: MedQA, MedMCQA, PubMedQA,
                  MMLU (Anatomy, Clinical Knowledge, College Biology,
                         College Medicine, Medical Genetics, Professional Medicine)
    * Fact-checking: FakeHealth, PubHealth

  Table 2 — Long-form LLM-as-Judge on DiverseHealth:
    Pertinence (Pert.), Correctness (Cor.),
    Citation Precision (CP), Citation Recall (CR), Understandability (Und.)

Usage examples
--------------
# Quick smoke-test (3 samples from MedQA):
  python evaluate.py --datasets medqa --max-samples 3

# Full MCQ benchmark (9 datasets):
  python evaluate.py --datasets medqa medmcqa pubmedqa \\
      mmlu_anatomy mmlu_clinical_knowledge mmlu_college_biology \\
      mmlu_college_medicine mmlu_medical_genetics mmlu_professional_medicine

# Add fact-checking (local files required):
  python evaluate.py --datasets pubhealth fakehealth \\
      --pubhealth-path ./data/pubhealth_test.jsonl \\
      --fakehealth-path ./data/fakehealth_test.json

# Long-form evaluation:
  python evaluate.py --longform --diverse-health-path ./data/diverse_health.json
"""

import os
import json
import re
import time
import argparse
from statistics import mean
from typing import Optional

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from camel.storages import Neo4jGraph

from summerize import process_chunks
from retrieve import seq_ret
from utils import get_response, call_llm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPTION_LETTERS = ["A", "B", "C", "D", "E"]

# All datasets supported out-of-the-box (HuggingFace)
HF_DATASETS = {
    "medqa",
    "medmcqa",
    "pubmedqa",
    "mmlu_anatomy",
    "mmlu_clinical_knowledge",
    "mmlu_college_biology",
    "mmlu_college_medicine",
    "mmlu_medical_genetics",
    "mmlu_professional_medicine",
}

ALL_DATASETS = HF_DATASETS | {"pubhealth", "fakehealth"}

# LLM-as-Judge system prompt (Table 2 of paper)
JUDGE_SYS = (
    "You are a strict medical evaluator. "
    "Return ONLY a valid JSON object, no extra text."
)

JUDGE_USER_TMPL = """\
Evaluate the model response below on five dimensions (1=very poor, 5=excellent).

Question: {question}

Model Response:
{response}

Reference Answer:
{reference}

Dimensions:
  Pert  — Pertinence: is the response relevant to the question?
  Cor   — Correctness: is the medical information factually correct?
  CP    — Citation Precision: are cited sources accurate and relevant?
  CR    — Citation Recall: are all important sources cited?
  Und   — Understandability: is the response clear, even for laypersons?

Return exactly: {{"Pert": <1-5>, "Cor": <1-5>, "CP": <1-5>, "CR": <1-5>, "Und": <1-5>}}"""


# ---------------------------------------------------------------------------
# Dataset loaders
# Each loader returns a list of dicts:
#   {"question": str, "options": {"A": str, ...}, "answer": str}
# PubMedQA also includes "context": str.
# ---------------------------------------------------------------------------

def _letters(choices):
    return {OPTION_LETTERS[i]: c for i, c in enumerate(choices)}


def load_medqa(split: str = "test"):
    ds = load_dataset(
        "bigbio/med_qa", "med_qa_en_bigbio_qa",
        split=split, trust_remote_code=True,
    )
    out = []
    for item in ds:
        choices = item["choices"]
        raw_idx = item["answer_idx"]
        answer = (
            OPTION_LETTERS[int(raw_idx)]
            if isinstance(raw_idx, int)
            else str(raw_idx).strip().upper()
        )
        out.append({
            "question": item["question"],
            "options": _letters(choices),
            "answer": answer,
        })
    return out


def load_medmcqa(split: str = "validation"):
    ds = load_dataset("medmcqa", split=split)
    letter_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    out = []
    for item in ds:
        options = {
            "A": item["opa"], "B": item["opb"],
            "C": item["opc"], "D": item["opd"],
        }
        answer = letter_map.get(item["cop"], "A")
        out.append({"question": item["question"], "options": options, "answer": answer})
    return out


def load_pubmedqa(split: str = "test"):
    ds = load_dataset(
        "bigbio/pubmed_qa",
        "pubmed_qa_labeled_fold0_bigbio_qa",
        split=split, trust_remote_code=True,
    )
    answer_map = {"yes": "A", "no": "B", "maybe": "C"}
    options = {"A": "yes", "B": "no", "C": "maybe"}
    out = []
    for item in ds:
        raw = (item["answer"][0] if item["answer"] else "no").lower().strip()
        answer = answer_map.get(raw, "B")
        ctx = item.get("context", "")
        if isinstance(ctx, list):
            ctx = " ".join(ctx)
        out.append({
            "question": item["question"],
            "context": ctx,
            "options": options,
            "answer": answer,
        })
    return out


def load_mmlu(subset: str, split: str = "test"):
    ds = load_dataset("cais/mmlu", subset, split=split)
    out = []
    for item in ds:
        options = _letters(item["choices"])
        answer = OPTION_LETTERS[item["answer"]]
        out.append({"question": item["question"], "options": options, "answer": answer})
    return out


def load_pubhealth(path: str):
    """Load PubHealth from a local JSONL file (one JSON object per line).

    Expected fields: "claim" (str), "label" (str: true/false/mixture/unproven)
    Download: https://huggingface.co/datasets/nyu-mll/multi_nli  (or paper repo)
    """
    label_map = {"true": "A", "false": "B", "mixture": "C", "unproven": "D"}
    options = {"A": "true", "B": "false", "C": "mixture", "D": "unproven"}
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            label = str(item.get("label", "false")).lower()
            answer = label_map.get(label, "B")
            out.append({
                "question": item.get("claim", item.get("text", "")),
                "options": options,
                "answer": answer,
            })
    return out


def load_fakehealth(path: str):
    """Load FakeHealth from a local JSON file.

    Expected: list of {"text"/"claim": str, "label": "real"/"fake" or 1/0}
    Download: https://github.com/EnyanDai/FakeHealth
    """
    label_map = {"real": "A", "fake": "B", 1: "A", 0: "B"}
    options = {"A": "real", "B": "fake"}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    out = []
    for item in data:
        raw_label = item.get("label", "fake")
        answer = label_map.get(raw_label, "B")
        question = item.get("text", item.get("claim", ""))
        out.append({"question": question, "options": options, "answer": answer})
    return out


def load_diverse_health(path: str):
    """Load DiverseHealth long-form dataset.

    Expected: list of {"question": str, "reference_answer": str}
    """
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def get_samples(dataset_name: str, args) -> list:
    """Dispatch to the correct loader."""
    if dataset_name == "medqa":
        return load_medqa()
    if dataset_name == "medmcqa":
        return load_medmcqa()
    if dataset_name == "pubmedqa":
        return load_pubmedqa()
    if dataset_name.startswith("mmlu_"):
        subset = dataset_name[len("mmlu_"):]
        return load_mmlu(subset)
    if dataset_name == "pubhealth":
        if not args.pubhealth_path:
            raise FileNotFoundError("--pubhealth-path is required for pubhealth dataset")
        return load_pubhealth(args.pubhealth_path)
    if dataset_name == "fakehealth":
        if not args.fakehealth_path:
            raise FileNotFoundError("--fakehealth-path is required for fakehealth dataset")
        return load_fakehealth(args.fakehealth_path)
    raise ValueError(f"Unknown dataset: {dataset_name}")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def build_mcq_prompt(question: str, options: dict, context: str = "") -> str:
    ctx_block = f"Context:\n{context}\n\n" if context else ""
    opt_block = "\n".join(f"{k}. {v}" for k, v in options.items())
    return (
        f"{ctx_block}"
        f"Question: {question}\n\n"
        f"Options:\n{opt_block}\n\n"
        "Reply with ONLY the single option letter (e.g. A). Your answer:"
    )


def extract_letter(response: str, options: dict) -> Optional[str]:
    """Best-effort extraction of a single option letter from an LLM response."""
    text = response.strip().upper()

    # 1. Response starts with a valid letter
    if text and text[0] in options:
        return text[0]

    # 2. Common patterns: "Answer: B", "(B)", "**B**", "The answer is B"
    for pat in [
        r"answer[:\s]+([A-E])",
        r"\b([A-E])\b",
        r"\(([A-E])\)",
        r"\*\*([A-E])\*\*",
    ]:
        m = re.search(pat, text)
        if m and m.group(1) in options:
            return m.group(1)

    # 3. Partial option text present in response (case-insensitive)
    low = response.lower()
    for letter, opt_text in options.items():
        if opt_text.lower()[:40] in low:
            return letter

    return None


def run_medgraphrag(n4j, prompt: str) -> str:
    """Execute the MedGraphRAG inference pipeline (seq_ret → get_response)."""
    summary = process_chunks(prompt)
    gid = seq_ret(n4j, summary)
    return get_response(n4j, gid, prompt)


# ---------------------------------------------------------------------------
# MCQ Evaluation  (Table 1)
# ---------------------------------------------------------------------------

def evaluate_mcq(
    n4j,
    dataset_name: str,
    samples: list,
    max_samples: Optional[int],
    delay: float,
    results_dir: str,
) -> float:
    """Run MCQ evaluation and return accuracy (0–100)."""
    if max_samples:
        samples = samples[:max_samples]

    os.makedirs(results_dir, exist_ok=True)
    records = []
    n_correct = 0

    for idx, sample in enumerate(tqdm(samples, desc=f"  {dataset_name}")):
        question = sample["question"]
        options = sample["options"]
        ground_truth = sample["answer"]
        context = sample.get("context", "")

        prompt = build_mcq_prompt(question, options, context)

        try:
            response = run_medgraphrag(n4j, prompt)
            predicted = extract_letter(response, options)
        except Exception as exc:
            print(f"\n  [WARN] sample {idx} failed: {exc}")
            predicted = None
            response = ""

        correct = predicted == ground_truth
        if correct:
            n_correct += 1

        records.append({
            "index": idx,
            "question": question[:300],
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": correct,
            "raw_response": response[:600],
        })

        if delay > 0:
            time.sleep(delay)

    accuracy = n_correct / len(samples) * 100
    print(f"\n  [{dataset_name}] Accuracy: {accuracy:.2f}%  ({n_correct}/{len(samples)})")

    out_path = os.path.join(results_dir, f"{dataset_name}.csv")
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")

    return accuracy


# ---------------------------------------------------------------------------
# Long-form LLM-as-Judge  (Table 2)
# ---------------------------------------------------------------------------

def llm_judge(question: str, response: str, reference: str) -> dict:
    """Ask the local LLM to score a response on 5 dimensions."""
    user_msg = JUDGE_USER_TMPL.format(
        question=question[:500],
        response=response[:1500],
        reference=reference[:500],
    )
    raw = call_llm(JUDGE_SYS, user_msg)

    # Try to extract JSON
    m = re.search(r'\{[^}]+\}', raw)
    if m:
        try:
            scores = json.loads(m.group())
            for key in ("Pert", "Cor", "CP", "CR", "Und"):
                scores.setdefault(key, 0)
            return scores
        except json.JSONDecodeError:
            pass

    print(f"\n  [WARN] Could not parse judge output: {raw[:120]}")
    return {"Pert": 0, "Cor": 0, "CP": 0, "CR": 0, "Und": 0}


def evaluate_longform(
    n4j,
    dataset_name: str,
    samples: list,
    max_samples: Optional[int],
    delay: float,
    results_dir: str,
) -> dict:
    """Run long-form LLM-as-Judge evaluation. Returns average scores (scaled to %)."""
    if max_samples:
        samples = samples[:max_samples]

    os.makedirs(results_dir, exist_ok=True)
    dim_keys = ("Pert", "Cor", "CP", "CR", "Und")
    bucket = {k: [] for k in dim_keys}
    records = []

    for idx, sample in enumerate(tqdm(samples, desc=f"  {dataset_name} (longform)")):
        question = sample["question"]
        reference = sample.get("reference_answer", sample.get("answer", ""))

        try:
            response = run_medgraphrag(n4j, question)
            scores = llm_judge(question, response, reference)
        except Exception as exc:
            print(f"\n  [WARN] sample {idx} failed: {exc}")
            scores = {k: 0 for k in dim_keys}
            response = ""

        for k in dim_keys:
            bucket[k].append(scores.get(k, 0))

        records.append({
            "index": idx,
            "question": question[:300],
            "response": response[:600],
            "reference": reference[:300],
            **scores,
        })

        if delay > 0:
            time.sleep(delay)

    # Paper reports as percentage (score 1–5 → ×20 → %)
    avg = {k: mean(bucket[k]) * 20 for k in dim_keys}

    print(f"\n  [{dataset_name}] Long-form scores (%):")
    for k, v in avg.items():
        print(f"    {k}: {v:.1f}")

    out_path = os.path.join(results_dir, f"{dataset_name}_longform.csv")
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")

    return avg


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def print_summary(rows: list):
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="MedGraphRAG evaluation — reproduces Table 1 & 2 of arXiv:2408.04187",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Neo4j connection
    neo = p.add_argument_group("Neo4j")
    neo.add_argument("--neo4j-url",
                     default=os.getenv("NEO4J_URL", "bolt://localhost:7688"))
    neo.add_argument("--neo4j-username",
                     default=os.getenv("NEO4J_USERNAME", "neo4j"))
    neo.add_argument("--neo4j-password",
                     default=os.getenv("NEO4J_PASSWORD"))

    # What to evaluate
    ev = p.add_argument_group("Evaluation targets")
    ev.add_argument(
        "--datasets", nargs="+", metavar="DS",
        default=[
            "medqa", "medmcqa", "pubmedqa",
            "mmlu_anatomy", "mmlu_clinical_knowledge",
            "mmlu_college_biology", "mmlu_college_medicine",
            "mmlu_medical_genetics", "mmlu_professional_medicine",
        ],
        help=(
            "Datasets to evaluate. Choose from: "
            + ", ".join(sorted(ALL_DATASETS))
            + ". Pass 'all' to run every dataset."
        ),
    )
    ev.add_argument("--longform", action="store_true",
                    help="Also run long-form LLM-as-Judge on DiverseHealth")

    # Local file paths (only needed for those datasets)
    files = p.add_argument_group("Local file paths")
    files.add_argument("--pubhealth-path", metavar="PATH",
                       help="Path to PubHealth JSONL test file")
    files.add_argument("--fakehealth-path", metavar="PATH",
                       help="Path to FakeHealth JSON test file")
    files.add_argument("--diverse-health-path", metavar="PATH",
                       help="Path to DiverseHealth JSON file (for --longform)")

    # Run controls
    ctrl = p.add_argument_group("Run controls")
    ctrl.add_argument("--max-samples", type=int, default=None, metavar="N",
                      help="Limit each dataset to N samples (useful for testing)")
    ctrl.add_argument("--delay", type=float, default=0.5, metavar="SEC",
                      help="Sleep between LLM calls in seconds (default: 0.5)")
    ctrl.add_argument("--results-dir", default="./eval_results", metavar="DIR",
                      help="Directory to save per-sample CSV files (default: ./eval_results)")

    return p


def main():
    args = build_parser().parse_args()

    if not args.neo4j_password:
        raise SystemExit(
            "ERROR: Neo4j password is required.\n"
            "Set the NEO4J_PASSWORD environment variable or use --neo4j-password."
        )

    print("Connecting to Neo4j …")
    n4j = Neo4jGraph(
        url=args.neo4j_url,
        username=args.neo4j_username,
        password=args.neo4j_password,
    )
    print("Connected.\n")

    datasets_to_run = (
        sorted(ALL_DATASETS) if "all" in args.datasets else args.datasets
    )

    summary_rows = []

    # ------------------------------------------------------------------
    # MCQ / fact-checking evaluation
    # ------------------------------------------------------------------
    for ds_name in datasets_to_run:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        try:
            samples = get_samples(ds_name, args)
            acc = evaluate_mcq(
                n4j=n4j,
                dataset_name=ds_name,
                samples=samples,
                max_samples=args.max_samples,
                delay=args.delay,
                results_dir=args.results_dir,
            )
            summary_rows.append({"dataset": ds_name, "metric": "accuracy(%)", "value": round(acc, 2)})
        except FileNotFoundError as exc:
            print(f"  SKIP: {exc}")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            summary_rows.append({"dataset": ds_name, "metric": "accuracy(%)", "value": "ERROR"})

    # ------------------------------------------------------------------
    # Long-form evaluation
    # ------------------------------------------------------------------
    if args.longform:
        print(f"\n{'='*60}")
        print("Long-form evaluation: DiverseHealth")
        print(f"{'='*60}")

        if not args.diverse_health_path:
            print("  SKIP: --diverse-health-path not provided")
        else:
            try:
                lf_samples = load_diverse_health(args.diverse_health_path)
                avg_scores = evaluate_longform(
                    n4j=n4j,
                    dataset_name="diverse_health",
                    samples=lf_samples,
                    max_samples=args.max_samples,
                    delay=args.delay,
                    results_dir=args.results_dir,
                )
                for dim, val in avg_scores.items():
                    summary_rows.append({
                        "dataset": "diverse_health",
                        "metric": dim + "(%)",
                        "value": round(val, 1),
                    })
            except Exception as exc:
                print(f"  ERROR: {exc}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_summary(summary_rows)

    os.makedirs(args.results_dir, exist_ok=True)
    summary_path = os.path.join(args.results_dir, "summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
