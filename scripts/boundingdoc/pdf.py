from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


@dataclass(slots=True)
class PdfToPngConfig:
    """Configuration for PDF → PNG conversion.

    Parameters
    ----------
    dpi: int
        Render DPI. Lower this to speed up at the cost of quality.
    suffix: str
        File suffix for saved images.
    thread_count: int | None
        Per-PDF parallel page rendering (passed to pdf2image/`pdftoppm`).
    max_workers: int | None
        Cross-PDF concurrency. If >1 and multiple PDFs are present, convert in parallel.
    use_paths_only: bool
        If True, have pdf2image write images directly to disk (skips PIL image objects),
        which is significantly faster and more memory-efficient.
    """

    dpi: int = 300
    suffix: str = ".png"
    thread_count: int | None = None
    max_workers: int | None = None
    use_paths_only: bool = True

    def __post_init__(self) -> None:
        if self.thread_count is not None and self.thread_count <= 0:
            fallback = os.cpu_count() or 1
            object.__setattr__(self, "thread_count", fallback)
        if self.max_workers is not None and self.max_workers <= 0:
            fallback = os.cpu_count() or 1
            object.__setattr__(self, "max_workers", fallback)


def _iter_pdf_files(source: Path) -> Iterable[Path]:
    if source.is_file() and source.suffix.lower() == ".pdf":
        yield source
        return
    if not source.is_dir():
        raise FileNotFoundError(f"PDF path does not exist: {source}")
    for pdf in sorted(source.glob("*.pdf")):
        if pdf.is_file():
            yield pdf


def _convert_single_pdf(pdf_path: Path, target_dir: Path, cfg: PdfToPngConfig) -> List[Path]:
    """Convert one PDF to PNGs into `target_dir` using the provided config."""
    kwargs: dict = {"dpi": cfg.dpi}
    if cfg.thread_count:
        kwargs["thread_count"] = cfg.thread_count

    if cfg.use_paths_only:
        # Let pdf2image/pdftoppm write files directly — much faster and lower RAM.
        kwargs.update({
            "output_folder": str(target_dir),
            "fmt": cfg.suffix.lstrip("."),
            "paths_only": True,
            "output_file": "page",
        })
        paths = convert_from_path(str(pdf_path), **kwargs)

        # Normalize names to 1.png, 2.png, ... to preserve previous contract.
        out_paths: List[Path] = []
        for idx, p in enumerate(sorted(map(Path, paths)), start=1):
            dst = target_dir / f"{idx}{cfg.suffix}"
            src = Path(p)
            if src != dst:
                # replace() is atomic on most OSes and faster than copy+remove
                src.replace(dst)
            out_paths.append(dst)
        return out_paths

    # Fallback: keep previous behavior creating PIL Images then saving
    pages = convert_from_path(str(pdf_path), **kwargs)
    out_paths = []
    for index, page in enumerate(tqdm(pages, desc=f"{pdf_path.name}", unit="page"), start=1):
        png_path = target_dir / f"{index}{cfg.suffix}"
        page.save(png_path, "PNG")
        out_paths.append(png_path)
    return out_paths


def convert_pdfs_to_pngs(
    input_path: Path,
    output_dir: Path,
    config: PdfToPngConfig | None = None,
) -> List[Path]:
    """Convert every PDF found under `input_path` into per-PDF folders of PNGs.

    Returns the list of generated PNG paths.
    """
    cfg = config or PdfToPngConfig()
    generated: List[Path] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = list(_iter_pdf_files(input_path))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found at: {input_path}")

    # If multiple PDFs and max_workers>1, process PDFs concurrently.
    if cfg.max_workers and cfg.max_workers > 1 and len(pdf_files) > 1:
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
            futures = []
            for pdf_path in pdf_files:
                target_dir = output_dir / pdf_path.stem
                target_dir.mkdir(parents=True, exist_ok=True)
                futures.append(executor.submit(_convert_single_pdf, pdf_path, target_dir, cfg))
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Converting PDFs", unit="pdf"):
                generated.extend(fut.result())
    else:
        pdf_iter = tqdm(pdf_files, desc="Converting PDFs", unit="pdf")
        for pdf_path in pdf_iter:
            target_dir = output_dir / pdf_path.stem
            target_dir.mkdir(parents=True, exist_ok=True)
            generated.extend(_convert_single_pdf(pdf_path, target_dir, cfg))

    return generated
