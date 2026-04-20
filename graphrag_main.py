"""
graphrag_main.py — CLI entry point cho GraphRAG pipeline (nano_graphrag).

Cùng cấu trúc với rag/main.py nhưng thay FAISS + CrossEncoder
bằng nano_graphrag.GraphRAG (GraphML + Milvus Lite + Community Report).

Chạy từ thư mục Medical-Graph-RAG:

  # ── Lần 1: Chỉ build index (tối ưu: batch insert, LLM cache) ──────────────
  python graphrag_main.py --build-only
  python graphrag_main.py --build-only --batch-size 50 --llm-async 4
  python graphrag_main.py --rebuild --build-only           # xóa cache cũ rồi build lại

  # ── Lần 2: Chỉ eval (không build lại) ────────────────────────────────────
  python graphrag_main.py --skip-build --eval medqa
  python graphrag_main.py --skip-build --eval medqa --max-samples 10

  # ── Build + eval trong 1 lần ──────────────────────────────────────────────
  python graphrag_main.py --eval medqa
  python graphrag_main.py --eval medqa medmcqa pubmedqa

  # ── Inference nhanh (không eval, không build lại) ─────────────────────────
  python graphrag_main.py --skip-build --query "What is the main symptom?"
"""

from __future__ import annotations

import argparse
import os
import re
import time
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from datasets import load_from_disk
from dotenv import load_dotenv
from tqdm import tqdm

# ── Load .env ─────────────────────────────────────────────────────────────────
_env = Path(__file__).resolve().parent / ".env"
if _env.is_file():
    load_dotenv(_env)

# nano_graphrag nằm cùng thư mục
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nano_graphrag import GraphRAG, QueryParam  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
OPTION_LETTERS            = ["A", "B", "C", "D", "E"]
EVAL_RETRIEVE_CONCURRENCY = 4   # ThreadPool cho GraphRAG.query (sync)

VALID_DATASETS = [
    "medqa",
    "medmcqa",
    "pubmedqa",
    "mmlu_anatomy",
    "mmlu_clinical_knowledge",
    "mmlu_college_biology",
    "mmlu_college_medicine",
    "mmlu_medical_genetics",
    "mmlu_professional_medicine",
]

# File đánh dấu index đã build xong (để --skip-build nhận ra)
_INDEX_DONE_MARKER = ".index_done"


# ── Marker helpers ────────────────────────────────────────────────────────────
def _marker_path(working_dir: str) -> Path:
    return Path(working_dir) / _INDEX_DONE_MARKER


def _is_index_built(working_dir: str) -> bool:
    graphml = Path(working_dir) / "graph_chunk_entity_relation.graphml"
    return graphml.exists() and _marker_path(working_dir).exists()


def _mark_index_done(working_dir: str) -> None:
    _marker_path(working_dir).write_text("ok", encoding="utf-8")


# ── Dataset formatter ─────────────────────────────────────────────────────────
def format_sample(dataset_name: str, item) -> Dict:
    """Chuẩn hóa 1 sample về dạng thống nhất."""
    if dataset_name in ("medqa", "pubmedqa") or dataset_name.startswith("mmlu_"):
        d = item["data"]
        sample = {
            "question": d["Question"],
            "options":  d["Options"],
            "answer":   str(d["Correct Option"]).strip().upper(),
        }
        if dataset_name == "pubmedqa":
            ctx = d.get("Context", [])
            sample["context"] = " ".join(ctx) if isinstance(ctx, list) else ctx
        return sample

    if dataset_name == "medmcqa":
        opts = {"A": item["opa"], "B": item["opb"],
                "C": item["opc"], "D": item["opd"]}
        ans  = {0: "A", 1: "B", 2: "C", 3: "D"}.get(item["cop"], "A")
        return {"question": item["question"], "options": opts, "answer": ans}

    raise ValueError(f"Unknown dataset: {dataset_name!r}")


def build_mcq_prompt(question: str, options: Dict, context: str = "") -> str:
    ctx_block = f"Context:\n{context}\n\n" if context else ""
    opt_block  = "\n".join(f"{k}. {v}" for k, v in options.items())
    return (
        f"{ctx_block}Question: {question}\n\n"
        f"Options:\n{opt_block}\n\n"
        "Reply with ONLY the single option letter (A, B, C or D). Your answer:"
    )


def _strip_think(text: str) -> str:
    """Xóa <think>...</think> block (Qwen3 thinking mode)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$",         "", text, flags=re.DOTALL)
    return text.strip()


def extract_letter(response: str, options: Dict) -> Optional[str]:
    cleaned = _strip_think(response)
    text    = cleaned.strip().upper()

    if text and text[0] in options:
        return text[0]

    for pat in [r"answer[:\s]+([A-E])", r"\b([A-E])\b",
                r"\(([A-E])\)", r"\*\*([A-E])\*\*"]:
        m = re.search(pat, text)
        if m and m.group(1) in options:
            return m.group(1)

    low = cleaned.lower()
    for letter, opt_text in options.items():
        if opt_text.lower()[:40] in low:
            return letter

    return None


# ── GraphRAG pipeline wrapper ─────────────────────────────────────────────────
class GraphRAGPipeline:
    """
    Wrapper quanh nano_graphrag.GraphRAG.

    Tối ưu so với insert từng file:
    - Batch insert → giảm số lần rebuild community (N file → N/batch_size lần)
    - enable_llm_cache → LLM call giống nhau không gọi lại
    - Tăng best_model_max_async + embedding_func_max_async → nhiều request song song
    - _is_index_built() → bỏ qua build nếu index đã tồn tại (--skip-build)
    """

    def __init__(
        self,
        working_dir: str,
        batch_size: int = 50,
        llm_async: int = 16,
        embed_async: int = 32,
        embed_batch: int = 64,
        enable_llm_cache: bool = True,
        chunk_size: int = 600,
        no_glean: bool = True,
        no_community: bool = False,
    ):
        self.working_dir      = working_dir
        self.batch_size       = batch_size
        self.llm_async        = llm_async
        self.embed_async      = embed_async
        self.embed_batch      = embed_batch
        self.enable_llm_cache = enable_llm_cache
        self.chunk_size       = chunk_size
        self.no_glean         = no_glean
        self.no_community     = no_community
        self._graph: Optional[GraphRAG] = None

    def _init_graph(self) -> GraphRAG:
        return GraphRAG(
            working_dir=self.working_dir,
            enable_llm_cache=self.enable_llm_cache,
            best_model_max_async=self.llm_async,
            cheap_model_max_async=self.llm_async,
            embedding_func_max_async=self.embed_async,
            embedding_batch_num=self.embed_batch,
            chunk_token_size=self.chunk_size,
            # gleaning=0: không gửi lại history cho LLM → giảm context ~50%
            entity_extract_max_gleaning=0 if self.no_glean else 1,
            enable_community_reports=not self.no_community,
        )

    # -- Build -----------------------------------------------------------------
    def build(self, data_path: str) -> None:
        """
        Batch insert toàn bộ .txt trong data_path.
        Dùng pipelining: đọc + chunk batch N+1 trên CPU trong khi
        LLM đang xử lý batch N → giảm idle time giữa 2 batch.
        """
        txt_files = sorted(Path(data_path).glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"Không tìm thấy .txt trong: {data_path}")

        Path(self.working_dir).mkdir(parents=True, exist_ok=True)

        total_batches = (len(txt_files) + self.batch_size - 1) // self.batch_size

        # Số CPU core dùng để đọc file song song (I/O bound)
        read_workers = min(16, os.cpu_count() or 4)

        print(f"\nBuilding GraphRAG index:")
        print(f"  Dataset     : {data_path}  ({len(txt_files)} files)")
        print(f"  Working     : {self.working_dir}")
        print(f"  Batch       : {self.batch_size} files/insert  "
              f"→ ~{total_batches} community rebuilds")
        print(f"  LLM async   : {self.llm_async}  |  "
              f"Embed async: {self.embed_async}  |  "
              f"LLM cache: {self.enable_llm_cache}")
        print(f"  Read workers: {read_workers} (file I/O parallel)\n")

        self._graph = self._init_graph()

        # Chia thành các batch
        batches: List[List[Path]] = [
            txt_files[i * self.batch_size:(i + 1) * self.batch_size]
            for i in range(total_batches)
        ]

        def _read_batch(fps: List[Path]) -> List[str]:
            """Đọc 1 batch file song song trên ThreadPool (CPU+I/O)."""
            def _read_one(fp: Path) -> Optional[str]:
                try:
                    t = fp.read_text(encoding="utf-8", errors="replace").strip()
                    return t if t else None
                except OSError as e:
                    print(f"  [WARN] đọc lỗi {fp.name}: {e}")
                    return None

            with ThreadPoolExecutor(max_workers=read_workers) as pool:
                results = list(pool.map(_read_one, fps))
            return [t for t in results if t is not None]

        # Pipeline: submit đọc batch tiếp theo ngay khi bắt đầu xử lý batch hiện tại
        with ThreadPoolExecutor(max_workers=1) as prefetch_pool:
            # Pre-fetch batch đầu tiên
            next_future = prefetch_pool.submit(_read_batch, batches[0])

            for batch_idx in range(total_batches):
                # Lấy kết quả batch hiện tại (đã được đọc trước)
                texts = next_future.result()

                # Ngay lập tức submit đọc batch tiếp theo (chạy song song với LLM)
                if batch_idx + 1 < total_batches:
                    next_future = prefetch_pool.submit(_read_batch, batches[batch_idx + 1])

                if not texts:
                    continue

                b = batches[batch_idx]
                print(f"  Batch {batch_idx + 1}/{total_batches}: "
                      f"{len(texts)} files  ({b[0].name} … {b[-1].name})")

                try:
                    t_batch = time.perf_counter()
                    self._graph.insert(texts)
                    print(
                        f"  [timing] batch {batch_idx + 1} wall: "
                        f"{time.perf_counter() - t_batch:.2f}s\n",
                        flush=True,
                    )
                except Exception as e:
                    print(f"  [ERROR] batch {batch_idx + 1} thất bại: {e}")
                    raise

        _mark_index_done(self.working_dir)
        print(f"\nIndex done. Artifacts in: {self.working_dir}")

    # -- Load (không build lại) -----------------------------------------------
    def load(self) -> None:
        """Load index đã có mà không insert thêm gì."""
        if not _is_index_built(self.working_dir):
            raise RuntimeError(
                f"Index chưa tồn tại tại '{self.working_dir}'.\n"
                f"Chạy trước: python graphrag_main.py --build-only"
            )
        print(f"Loading index từ: {self.working_dir}")
        self._graph = self._init_graph()

    # -- Query -----------------------------------------------------------------
    def query(self, question: str, mode: str = "local") -> str:
        if self._graph is None:
            raise RuntimeError("Gọi build() hoặc load() trước khi query().")
        return self._graph.query(question, param=QueryParam(mode=mode))


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_dataset(
    pipeline: GraphRAGPipeline,
    dataset_name: str,
    max_samples: Optional[int] = None,
    results_dir: str = "./graphrag_eval_results",
    eval_dir: str   = "./evaluation/test_qa",
    num_workers: int = EVAL_RETRIEVE_CONCURRENCY,
) -> float:
    local_path = Path(eval_dir) / dataset_name
    if not local_path.exists():
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' chưa có tại '{local_path}'.\n"
            f"Chạy: python evaluation/download.py"
        )

    ds      = load_from_disk(str(local_path))
    samples = list(ds)
    if max_samples:
        samples = samples[:max_samples]

    print(f"\n  [{dataset_name}] {len(samples)} samples | workers={num_workers}")

    def _run_one(idx_item):
        idx, raw = idx_item
        sample   = format_sample(dataset_name, raw)
        question = sample["question"]
        options  = sample["options"]
        context  = sample.get("context", "")
        prompt   = build_mcq_prompt(question, options, context)

        try:
            response  = pipeline.query(prompt)
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

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        results = list(
            tqdm(
                pool.map(_run_one, enumerate(samples)),
                total=len(samples),
                desc=f"  {dataset_name}",
            )
        )

    records   = sorted(results, key=lambda r: r["index"])
    n_correct = sum(r["correct"] for r in records)
    accuracy  = n_correct / len(records) * 100

    print(f"\n  [{dataset_name}] Accuracy: {accuracy:.2f}%  ({n_correct}/{len(records)})")

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"graphrag_{dataset_name}.csv")
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")

    return accuracy


# ── CLI ───────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GraphRAG pipeline (nano_graphrag) — cùng interface với rag/main.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    data = p.add_argument_group("Data")
    data.add_argument("--data-path", default="./dataset/",
                      help="Thư mục chứa .txt files (mặc định: ./dataset/)")
    data.add_argument("--working-dir", default="./graphrag_retrieval",
                      help="Thư mục lưu graph + KV + Milvus (mặc định: ./graphrag_retrieval)")
    data.add_argument("--rebuild", action="store_true",
                      help="Xóa working-dir và build lại index từ đầu")
    data.add_argument("--eval-dir", default="./evaluation/test_qa",
                      help="Thư mục chứa datasets local (mặc định: ./evaluation/test_qa)")

    run = p.add_argument_group("Run mode")
    run.add_argument("--build-only", action="store_true",
                     help="Chỉ build index, không eval / query (dùng lần đầu)")
    run.add_argument("--skip-build", action="store_true",
                     help="Bỏ qua build, load index sẵn có (dùng lần 2 trở đi)")

    infer = p.add_argument_group("Inference / Eval")
    infer.add_argument("--query", type=str, default=None,
                       help="Câu hỏi thử inference")
    infer.add_argument("--query-mode", default="local", choices=["local", "global"],
                       help="GraphRAG query mode (mặc định: local)")
    infer.add_argument("--eval", nargs="+", choices=VALID_DATASETS, metavar="DS",
                       help="Datasets để evaluate. Chọn từ: " + ", ".join(VALID_DATASETS))

    perf = p.add_argument_group("Performance")
    perf.add_argument("--batch-size", type=int, default=50, metavar="N",
                      help="Số file gộp mỗi lần insert (mặc định: 50)")
    perf.add_argument("--llm-async", type=int, default=16,
                      help="Số LLM call song song — nên = vLLM --max-num-seqs (mặc định: 16)")
    perf.add_argument("--embed-async", type=int, default=32,
                      help="Số embedding call song song (mặc định: 32)")
    perf.add_argument("--embed-batch", type=int, default=64,
                      help="Số text / batch embedding (mặc định: 64)")
    perf.add_argument("--chunk-size", type=int, default=600,
                      help="Token mỗi chunk (mặc định: 600). Giảm nếu gặp lỗi context overflow")
    perf.add_argument("--glean", action="store_true",
                      help="Bật gleaning (gửi lại history cho LLM). Mặc định: tắt để tránh overflow")
    perf.add_argument("--no-llm-cache", action="store_true",
                      help="Tắt LLM cache (mặc định: bật)")
    perf.add_argument("--no-community", action="store_true",
                      help="Bỏ qua clustering + community report (build nhanh cho eval QA)")

    ctrl = p.add_argument_group("Eval controls")
    ctrl.add_argument("--max-samples", type=int, default=None,
                      help="Giới hạn số mẫu mỗi dataset khi eval")
    ctrl.add_argument("--results-dir", default="./graphrag_eval_results",
                      help="Thư mục lưu CSV kết quả (mặc định: ./graphrag_eval_results)")
    ctrl.add_argument("--workers", type=int, default=EVAL_RETRIEVE_CONCURRENCY,
                      help=f"Số query song song khi eval (mặc định: {EVAL_RETRIEVE_CONCURRENCY})")

    return p


def graphrag_main():
    args = build_parser().parse_args()

    # -- Rebuild ---------------------------------------------------------------
    if args.rebuild and Path(args.working_dir).exists():
        shutil.rmtree(args.working_dir)
        print(f"Cache '{args.working_dir}' đã xóa — sẽ build lại.\n")

    # -- Khởi tạo pipeline -----------------------------------------------------
    pipeline = GraphRAGPipeline(
        working_dir=args.working_dir,
        batch_size=args.batch_size,
        llm_async=args.llm_async,
        embed_async=args.embed_async,
        embed_batch=args.embed_batch,
        enable_llm_cache=not args.no_llm_cache,
        chunk_size=args.chunk_size,
        no_glean=not args.glean,
        no_community=args.no_community,
    )

    # -- Build / Load index ----------------------------------------------------
    if args.skip_build:
        pipeline.load()
    else:
        pipeline.build(args.data_path)

    # -- Chỉ build, không eval/query -------------------------------------------
    if args.build_only:
        print("\nBuild hoàn tất. Chạy eval bằng:")
        print(f"  python graphrag_main.py --skip-build --eval medqa\n")
        return

    # -- Inference mode --------------------------------------------------------
    if args.query:
        print(f"\nQuery: {args.query}")
        print("─" * 60)
        answer = pipeline.query(args.query, mode=args.query_mode)
        print(f"\nAnswer:\n{answer}\n")
        if not args.eval:
            return

    # -- Evaluation mode -------------------------------------------------------
    if args.eval:
        run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.results_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        print(f"\nRun ID: {run_id}  →  {run_dir}")

        summary = []
        for ds in args.eval:
            print(f"\n{'='*60}\nEvaluating: {ds}\n{'='*60}")
            try:
                acc = evaluate_dataset(
                    pipeline,
                    dataset_name=ds,
                    max_samples=args.max_samples,
                    results_dir=run_dir,
                    eval_dir=args.eval_dir,
                    num_workers=args.workers,
                )
                summary.append({"dataset": ds, "accuracy(%)": round(acc, 2), "status": "ok"})
            except Exception as e:
                print(f"\n  [ERROR] {ds}: {e}")
                summary.append({"dataset": ds, "accuracy(%)": 0.0, "status": str(e)[:80]})

        print("\n" + "=" * 60)
        print("SUMMARY — GraphRAG (nano_graphrag local mode)")
        print("=" * 60)

        core_rows = [r for r in summary if not r["dataset"].startswith("mmlu_")]
        mmlu_rows = [r for r in summary if r["dataset"].startswith("mmlu_")]

        if core_rows:
            print("\n  Core Medical QA:")
            for row in core_rows:
                st = "" if row["status"] == "ok" else f"  ← {row['status']}"
                print(f"    {row['dataset']:30s}  {row['accuracy(%)']:6.2f}%{st}")

        if mmlu_rows:
            print("\n  MMLU Medical:")
            for row in mmlu_rows:
                st = "" if row["status"] == "ok" else f"  ← {row['status']}"
                print(f"    {row['dataset']:30s}  {row['accuracy(%)']:6.2f}%{st}")
            mmlu_avg = sum(r["accuracy(%)"] for r in mmlu_rows) / len(mmlu_rows)
            print(f"    {'MMLU Average':30s}  {mmlu_avg:6.2f}%")

        if summary:
            total_avg = sum(r["accuracy(%)"] for r in summary) / len(summary)
            print(f"\n  {'Overall Average':30s}  {total_avg:6.2f}%")

        print("=" * 60)

        summary_path = os.path.join(run_dir, "graphrag_summary.csv")
        pd.DataFrame(summary).to_csv(summary_path, index=False)
        print(f"\nSummary saved → {summary_path}")
        return

    print("Không có --query hay --eval. Dùng --help để xem hướng dẫn.")


if __name__ == "__main__":
    graphrag_main()
