"""evaluator.py — Async MCQ evaluation trên MedQA, MedMCQA, PubMedQA."""

import asyncio
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from .config import OPTION_LETTERS, EVAL_LLM_CONCURRENCY, EVAL_RETRIEVE_CONCURRENCY
from .generator import call_llm, call_llm_async, build_context, _get_async_llm_client
from .reranker import warmup as _warmup_reranker


# ── Dataset loaders ────────────────────────────────────────────────────────────

def load_local_dataset(name: str, eval_dir: str = "./evaluation/test_qa"):
    """
    Load dataset từ local disk (đã download bằng evaluation/download.py).

    Schema thực tế:
      medqa   → item["data"]["Question"], item["data"]["Options"], item["data"]["Correct Option"]
      medmcqa → item["question"], item["opa/b/c/d"], item["cop"] (int 0-3)
      pubmedqa→ item["data"]["Question"], item["data"]["Options"], item["data"]["Correct Option"]
    """
    from datasets import load_from_disk

    local_path = os.path.join(eval_dir, name)
    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"Dataset '{name}' chưa có tại '{local_path}'.\n"
            f"Chạy: python evaluation/download.py"
        )
    print(f"  Loading {name} from {local_path} ...")
    return load_from_disk(local_path)


# ── Sample formatting ──────────────────────────────────────────────────────────

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
            "options":  d["Options"],   # {"A": ..., "B": ..., "C": ..., "D": ...}
            "answer":   str(d["Correct Option"]).strip().upper(),
        }

    raise ValueError(f"Unknown dataset: {dataset_name!r}")


# ── Prompt building ────────────────────────────────────────────────────────────

def build_mcq_prompt(question: str, options: Dict, context: str = "") -> str:
    ctx_block = f"Context:\n{context}\n\n" if context else ""
    opt_block  = "\n".join(f"{k}. {v}" for k, v in options.items())
    return (
        f"{ctx_block}Question: {question}\n\n"
        f"Options:\n{opt_block}\n\n"
        "Reply with ONLY the single option letter (e.g. A). Your answer:"
    )


# ── Answer extraction ──────────────────────────────────────────────────────────

def _strip_think(text: str) -> str:
    """
    Xóa <think>...</think> block của Qwen3 thinking mode.
    Xử lý cả trường hợp tag chưa đóng (bị cắt do hết max_tokens).
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)  # unclosed tag
    return text.strip()


def extract_letter(response: str, options: Dict) -> Optional[str]:
    # Xóa thinking block trước khi parse
    cleaned  = _strip_think(response)
    text     = cleaned.strip().upper()

    if text and text[0] in options:
        return text[0]

    for pat in [
        r"answer[:\s]+([A-E])",
        r"\b([A-E])\b",
        r"\(([A-E])\)",
        r"\*\*([A-E])\*\*",
    ]:
        m = re.search(pat, text)
        if m and m.group(1) in options:
            return m.group(1)

    low = cleaned.lower()
    for letter, opt_text in options.items():
        if opt_text.lower()[:40] in low:
            return letter

    return None


# ── Async evaluation core ──────────────────────────────────────────────────────

async def _evaluate_all_async(
    samples: List[Dict],
    rag_pipeline,
    dataset_name: str,
    llm_concurrency: int,
    retrieve_concurrency: int,
) -> List[Dict]:
    """
    Chạy toàn bộ samples song song:
      - retrieve(): chạy trong ThreadPoolExecutor (CPU-bound, sync)
      - call_llm_async(): chạy với asyncio.Semaphore (I/O-bound, async)

    Flow:
      [Thread pool] retrieve() × N  (retrieve_concurrency threads)
          ↓ asyncio.gather (đúng thứ tự)
      [Async sem]   call_llm_async() × N  (llm_concurrency concurrent)
    """
    sys_prompt = (
        "You are a medical expert. Use the retrieved context if helpful, "
        "then answer the multiple-choice question. "
        "Reply with ONLY the single option letter (A, B, C, or D). "
        "Do NOT explain. Do NOT use <think> tags."
    )

    # Warmup CrossEncoder trong main thread trước khi spawn ThreadPool
    # → tránh race condition khi nhiều thread cùng load model lần đầu
    print("  Warming up CrossEncoder...")
    _warmup_reranker()

    loop     = asyncio.get_event_loop()
    llm_sem  = asyncio.Semaphore(llm_concurrency)
    client   = _get_async_llm_client()
    executor = ThreadPoolExecutor(max_workers=retrieve_concurrency)

    formatted = [format_sample(dataset_name, raw) for raw in samples]

    # ── Bước 1: retrieve song song, giữ đúng thứ tự bằng gather ──────────────
    print(f"  [1/2] Retrieving ({retrieve_concurrency} threads)...")

    async def _retrieve_one(sample: Dict) -> List[Dict]:
        prompt = build_mcq_prompt(
            sample["question"], sample["options"], sample.get("context", "")
        )
        return await loop.run_in_executor(executor, rag_pipeline.retrieve, prompt)

    pbar_r = tqdm(total=len(formatted), desc="    retrieve", leave=False)

    async def _retrieve_tracked(sample: Dict) -> List[Dict]:
        result = await _retrieve_one(sample)
        pbar_r.update(1)
        return result

    top_chunks_list: List[List[Dict]] = await asyncio.gather(
        *[_retrieve_tracked(s) for s in formatted]
    )
    pbar_r.close()

    # ── Bước 2: LLM call song song, giữ đúng thứ tự bằng gather ──────────────
    print(f"  [2/2] LLM inference (concurrency={llm_concurrency})...")

    async def _llm_one(idx: int, sample: Dict, top_chunks: List[Dict]) -> Dict:
        question = sample["question"]
        options  = sample["options"]
        context  = sample.get("context", "")

        rag_context = build_context(top_chunks)
        full_prompt = (
            f"Retrieved context from patient records:\n{rag_context}\n\n"
            + build_mcq_prompt(question, options, context)
        )

        try:
            async with llm_sem:
                response = await call_llm_async(sys_prompt, full_prompt, client)
            predicted = extract_letter(response, options)
        except Exception as exc:
            print(f"\n  [WARN] sample {idx}: {exc}")
            response  = ""
            predicted = None

        return {
            "index":        idx,
            "question":     question[:300],
            "ground_truth": sample["answer"],
            "predicted":    predicted,
            "correct":      predicted == sample["answer"],
            "raw_response": _strip_think(response)[:400],
        }

    pbar_l = tqdm(total=len(formatted), desc="    llm call", leave=False)

    async def _llm_tracked(idx: int, sample: Dict, top_chunks: List[Dict]) -> Dict:
        result = await _llm_one(idx, sample, top_chunks)
        pbar_l.update(1)
        return result

    records: List[Dict] = await asyncio.gather(
        *[_llm_tracked(i, s, tc)
          for i, (s, tc) in enumerate(zip(formatted, top_chunks_list))]
    )
    pbar_l.close()

    await client.close()
    executor.shutdown(wait=False)

    return list(records)  # đã đúng thứ tự nhờ gather


# ── Public API ─────────────────────────────────────────────────────────────────

def evaluate_dataset(
    rag_pipeline,
    dataset_name: str,
    max_samples: Optional[int] = None,
    results_dir: str = "./rag_eval_results",
    eval_dir: str = "./evaluation/test_qa",
    llm_concurrency: int = EVAL_LLM_CONCURRENCY,
    retrieve_concurrency: int = EVAL_RETRIEVE_CONCURRENCY,
) -> float:
    """
    Chạy MCQ evaluation async cho 1 dataset và lưu kết quả CSV.

    Args:
        rag_pipeline:         Instance của RAGPipeline đã build().
        dataset_name:         "medqa" | "medmcqa" | "pubmedqa"
        max_samples:          Giới hạn số mẫu (None = toàn bộ).
        results_dir:          Thư mục lưu CSV kết quả.
        eval_dir:             Thư mục chứa datasets local.
        llm_concurrency:      Số LLM call song song (default 8).
        retrieve_concurrency: Số retrieve() song song (default 4).

    Returns:
        Accuracy (%) trên toàn bộ samples đã chạy.
    """
    ds = load_local_dataset(dataset_name, eval_dir=eval_dir)
    samples = list(ds)
    if max_samples:
        samples = samples[:max_samples]

    print(f"\n  [{dataset_name}] {len(samples)} samples | "
          f"llm_concurrency={llm_concurrency} | "
          f"retrieve_concurrency={retrieve_concurrency}")

    records = asyncio.run(
        _evaluate_all_async(
            samples, rag_pipeline, dataset_name,
            llm_concurrency, retrieve_concurrency,
        )
    )

    n_correct = sum(r["correct"] for r in records)
    accuracy  = n_correct / len(records) * 100
    print(f"\n  [{dataset_name}] Accuracy: {accuracy:.2f}%  ({n_correct}/{len(records)})")

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"rag_{dataset_name}.csv")
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")

    return accuracy
