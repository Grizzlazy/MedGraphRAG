"""loader.py — Load file .txt và chunk văn bản theo token."""

from pathlib import Path
from typing import List, Dict

import tiktoken
from tqdm import tqdm

from .config import CHUNK_SIZE, CHUNK_OVERLAP


def load_txt_files(data_path: str) -> List[Dict]:
    """
    Load tất cả file .txt từ data_path.
    Mỗi file tương ứng 1 document (1 bệnh nhân trong mimic_ex).

    Returns:
        List of {"source": filename, "text": content}
    """
    path = Path(data_path)
    files = sorted(path.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy .txt nào trong: {data_path}")

    docs = []
    for fp in files:
        text = fp.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            docs.append({"source": fp.name, "text": text})

    print(f"Loaded {len(docs)} documents from {data_path}")
    return docs


def chunk_document(doc: Dict, encoder) -> List[Dict]:
    """
    Chia 1 document thành các chunks kích thước CHUNK_SIZE token,
    với CHUNK_OVERLAP token trùng lặp giữa các chunk liên tiếp.

    Returns:
        List of {"source", "chunk_idx", "text"}
    """
    tokens = encoder.encode(doc["text"])
    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunk_text = encoder.decode(tokens[start:end])
        chunks.append({
            "source":    doc["source"],
            "chunk_idx": chunk_idx,
            "text":      chunk_text,
        })
        chunk_idx += 1
        if end == len(tokens):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def build_chunks(docs: List[Dict]) -> List[Dict]:
    """
    Chunk tất cả documents và trả về flat list.

    Returns:
        List of chunk dicts từ tất cả documents.
    """
    encoder = tiktoken.encoding_for_model("gpt-4o")
    all_chunks: List[Dict] = []
    for doc in tqdm(docs, desc="Chunking"):
        all_chunks.extend(chunk_document(doc, encoder))
    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks
