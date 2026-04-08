"""
Stage 1 — PDF to Image Converter
Converts each page of a PDF to a high-resolution image for OCR.
"""

from pathlib import Path
from pdf2image import convert_from_path
from loguru import logger
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import IMAGE_DPI, PROCESSED_DIR


def pdf_to_images(
    pdf_path: str,
    output_dir: str | None = None,
    dpi: int = IMAGE_DPI,
) -> list[Path]:
    """
    Convert a PDF file to a list of PIL images saved as PNGs.

    Args:
        pdf_path:   Path to the input PDF.
        output_dir: Directory to save images. Defaults to PROCESSED_DIR/images/<stem>.
        dpi:        Resolution (default 300).

    Returns:
        List of saved image paths.
    """
    pdf = Path(pdf_path)
    if output_dir is None:
        output_dir = PROCESSED_DIR / "images" / pdf.stem
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting {pdf.name} at {dpi} DPI …")
    images = convert_from_path(str(pdf), dpi=dpi)

    saved_paths: list[Path] = []
    for idx, img in enumerate(images):
        out_path = output_dir / f"page_{idx + 1:03d}.png"
        img.save(str(out_path), "PNG")
        saved_paths.append(out_path)

    logger.info(f"  → {len(saved_paths)} pages saved to {output_dir}")
    return saved_paths


def batch_convert(pdf_dir: str, output_root: str | None = None, dpi: int = IMAGE_DPI) -> dict[str, list[Path]]:
    """
    Convert all PDFs in a directory.

    Returns:
        { "filename.pdf": [page_path, …], … }
    """
    pdf_dir = Path(pdf_dir)
    results: dict[str, list[Path]] = {}

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDFs in {pdf_dir}")

    for pdf_path in tqdm(pdf_files, desc="Converting PDFs"):
        out_dir = Path(output_root) / pdf_path.stem if output_root else None
        results[pdf_path.name] = pdf_to_images(str(pdf_path), out_dir, dpi)

    return results


# ======================================================================
# CLI
# ======================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert PDFs to images")
    parser.add_argument("--pdf_dir", required=True, help="Directory containing PDFs")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--dpi", type=int, default=IMAGE_DPI)
    args = parser.parse_args()

    batch_convert(args.pdf_dir, args.output_dir, args.dpi)
