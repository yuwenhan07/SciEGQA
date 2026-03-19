#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from boundingdoc.mm_agent import MMAgent
from boundingdoc.pdf import PdfToPngConfig, convert_pdfs_to_pngs
from boundingdoc.pipeline import DocumentProcessingError
from boundingdoc.sam_crop import SamCropConfig, SamCropper, SamStats, run_parallel_sam
from boundingdoc.judge import JudgeConfig, QwenJudge, run_judge_root


@dataclass(slots=True)
class SamJudgePipelineConfig:
    work_root: Path
    sam: SamCropConfig
    judge: JudgeConfig
    pdf: PdfToPngConfig = field(default_factory=PdfToPngConfig)
    overwrite: bool = True
    judge_agent: MMAgent | None = None

    def __post_init__(self) -> None:
        self.work_root = Path(self.work_root)


@dataclass(slots=True)
class SamJudgeResources:
    sam_cropper: SamCropper | None = None
    judge: QwenJudge | None = None
    sam_lock: Lock | None = None
    judge_lock: Lock | None = None


@dataclass(slots=True)
class SamJudgeProcessingResult:
    doc_name: str
    work_root: Path
    merged_output: Path
    summary_output: Path
    pages: int
    total_crops: int
    kept_crops: int
    merged_records: int


def _page_sort_key(name: str) -> Tuple[int, str]:
    digits = "".join(ch for ch in name if ch.isdigit())
    if digits:
        try:
            return int(digits), name
        except ValueError:
            pass
    return 10**9, name


def merge_judge_results(report_dir: Path, output_path: Path) -> Tuple[Path, int]:
    records: List[Dict[str, object]] = []
    doc_name = report_dir.name
    for page_dir in sorted(report_dir.iterdir(), key=lambda p: _page_sort_key(p.name)):
        if not page_dir.is_dir():
            continue
        judge_file = page_dir / "judge" / "judge_results.json"
        if not judge_file.is_file():
            continue
        try:
            entries = json.loads(judge_file.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"WARNING: failed to read {judge_file}: {exc}")
            continue

        page_name = page_dir.name
        keep_map = {
            entry.get("image"): bool(entry.get("keep", False))
            for entry in entries
            if entry.get("image") is not None
        }

        clean_dir = page_dir / "clean_crops"
        summary_path = clean_dir / "clean_summary.json"
        summary_items: List[dict] = []
        if summary_path.is_file():
            try:
                summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
                if isinstance(summary_data, dict):
                    raw_items = summary_data.get("items", [])
                    if isinstance(raw_items, list):
                        summary_items = [item for item in raw_items if isinstance(item, dict)]
            except Exception as exc:
                print(f"WARNING: failed to parse {summary_path}: {exc}")

        if not summary_items:
            print(f"WARNING: clean_summary.json missing or empty for {page_dir}, skipping cleaned merge")
            continue

        for item in summary_items:
            image_name = item.get("image")
            if not image_name or not keep_map.get(image_name, False):
                continue

            bbox_value = item.get("bbox_xyxy") or item.get("bbox")
            if bbox_value is None and item.get("bbox_xywh"):
                x, y, w, h = item["bbox_xywh"]
                bbox_value = [x, y, x + w, y + h]

            cleaned_bbox: Optional[List[int]] = None
            if isinstance(bbox_value, (list, tuple)) and len(bbox_value) == 4:
                try:
                    cleaned_bbox = [int(float(b)) for b in bbox_value]
                except Exception:
                    cleaned_bbox = None

            type_value = item.get("type") or "unknown"
            records.append(
                {
                    "doc": doc_name,
                    "page": page_name,
                    "image": image_name,
                    "type": str(type_value),
                    "bbox": cleaned_bbox,
                }
            )
    records.sort(key=lambda r: (_page_sort_key(str(r.get("page", ""))), r.get("image") or ""))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path, len(records)


class SamJudgeProcessor:
    """Run PDF → SAM → Judge pipeline (no QA)."""

    def __init__(self, config: SamJudgePipelineConfig, resources: SamJudgeResources | None = None):
        self.config = config
        self._work_root = config.work_root.resolve()
        self._work_root.mkdir(parents=True, exist_ok=True)

        self._resources = resources or SamJudgeResources()

        self._parallel_sam = bool(
            (config.sam.devices and len(config.sam.devices) > 1)
            or (config.sam.num_workers and config.sam.num_workers > 1)
        )

        cropper = self._resources.sam_cropper
        if self._parallel_sam:
            self._cropper: SamCropper | None = None
        else:
            if cropper is None:
                cropper = SamCropper(config.sam)
                self._resources.sam_cropper = cropper
            self._cropper = cropper

        judge = self._resources.judge
        if judge is None:
            judge = QwenJudge(config.judge, agent_backend=config.judge_agent)
            self._resources.judge = judge
        self._judge = judge

        self._sam_lock = self._resources.sam_lock
        self._judge_lock = self._resources.judge_lock

    def process(self, pdf_path: Path, final_output_dir: Path | None = None) -> SamJudgeProcessingResult:
        pdf_path = Path(pdf_path).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise DocumentProcessingError(f"Expected a PDF file, got: {pdf_path}")

        doc_name = pdf_path.stem
        doc_root = self._work_root / doc_name
        if doc_root.exists():
            if self.config.overwrite:
                shutil.rmtree(doc_root)
            else:
                raise DocumentProcessingError(
                    f"Work directory already exists for '{doc_name}'. "
                    "Use overwrite=True to refresh."
                )
        doc_root.mkdir(parents=True, exist_ok=True)

        pages_root = doc_root / "pages"
        processed_root = doc_root / "processed"

        generated = convert_pdfs_to_pngs(pdf_path, pages_root, self.config.pdf)
        if not generated:
            raise DocumentProcessingError(f"No PNG pages generated from {pdf_path}")

        page_dir = pages_root / doc_name
        if not page_dir.exists():
            raise DocumentProcessingError(f"Missing rendered pages directory: {page_dir}")

        overall_start = perf_counter()

        def _log_stage(stage: str, stage_elapsed: float) -> None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            total_elapsed = perf_counter() - overall_start
            stage_name = stage.upper()
            print(
                f"TIMER {doc_name}: [{timestamp}] {stage_name} complete "
                f"(stage={stage_elapsed:.1f}s, total={total_elapsed:.1f}s)"
            )

        def _run_sam() -> SamStats:
            if self._parallel_sam:
                return run_parallel_sam(
                    self.config.sam,
                    page_dir,
                    processed_root,
                    devices=self.config.sam.devices,
                    num_workers=self.config.sam.num_workers,
                    queue_size=self.config.sam.queue_size,
                )
            assert self._cropper is not None
            return self._cropper.run(page_dir, processed_root)

        if self._sam_lock:
            with self._sam_lock:
                sam_start = perf_counter()
                sam_stats = _run_sam()
                sam_elapsed = perf_counter() - sam_start
        else:
            sam_start = perf_counter()
            sam_stats = _run_sam()
            sam_elapsed = perf_counter() - sam_start
        _log_stage("sam", sam_elapsed)

        if sam_stats.succeeded_images == 0:
            raise DocumentProcessingError(f"SAM cropping produced no crops for {pdf_path}")

        if self._judge_lock:
            with self._judge_lock:
                judge_start = perf_counter()
                judge_stats = run_judge_root(
                    processed_root,
                    self._judge,
                    batch_size=getattr(self._judge.config, "batch_size", 1),
                )
                judge_elapsed = perf_counter() - judge_start
        else:
            judge_start = perf_counter()
            judge_stats = run_judge_root(
                processed_root,
                self._judge,
                batch_size=getattr(self._judge.config, "batch_size", 1),
            )
            judge_elapsed = perf_counter() - judge_start
        _log_stage("judge", judge_elapsed)

        report_dir = processed_root / doc_name
        if not report_dir.exists():
            raise DocumentProcessingError(f"Processed directory missing for {doc_name}")

        merged_path = report_dir / f"{doc_name}_judge_results.jsonl"
        merged_path, merged_count = merge_judge_results(report_dir, merged_path)

        summary_path = report_dir / f"{doc_name}_judge_summary.json"
        summary_payload = {
            "doc": doc_name,
            "pages": judge_stats.get("pages", 0),
            "total_crops": judge_stats.get("images", 0),
            "kept_crops": judge_stats.get("keep_true", 0),
            "dropped_crops": judge_stats.get("keep_false", 0),
            "kept_after_clean": judge_stats.get("clean_kept", 0),
            "type_counts": judge_stats.get("type_counts", {}),
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        output_results = merged_path
        output_summary = summary_path
        if final_output_dir:
            final_output_dir = Path(final_output_dir).resolve()
            final_output_dir.mkdir(parents=True, exist_ok=True)
            output_results = final_output_dir / merged_path.name
            output_summary = final_output_dir / summary_path.name
            shutil.copy2(merged_path, output_results)
            shutil.copy2(summary_path, output_summary)

        total_elapsed = perf_counter() - overall_start
        _log_stage("complete", total_elapsed)

        total_crops = judge_stats.get("images", merged_count)
        kept_crops = judge_stats.get("clean_kept", 0)

        return SamJudgeProcessingResult(
            doc_name=doc_name,
            work_root=doc_root,
            merged_output=output_results,
            summary_output=output_summary,
            pages=judge_stats.get("pages", 0),
            total_crops=total_crops,
            kept_crops=kept_crops,
            merged_records=merged_count,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BoundingDoc SAM + Judge pipeline (no QA) and merge crop results."
    )
    parser.add_argument("pdf", type=Path, help="PDF file or directory containing PDF files to process")
    parser.add_argument(
        "--work_root",
        required=True,
        type=Path,
        help="Working directory used to store intermediate artefacts (pages, crops, etc.)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Optional directory where the merged judge results will be copied",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI used when rasterising PDF pages (default: 300)")
    parser.add_argument(
        "--pdf_thread_count",
        type=int,
        default=None,
        help="Per-PDF rendering threads passed to pdf2image/pdftoppm",
    )
    parser.add_argument(
        "--pdf_max_workers",
        type=int,
        default=None,
        help="Number of PDFs to convert concurrently",
    )
    parser.add_argument(
        "--sam_checkpoint",
        required=True,
        type=str,
        help="Path to the SAM checkpoint file",
    )
    parser.add_argument("--sam_device", default="cuda", help="Device for SAM inference (default: cuda)")
    parser.add_argument(
        "--sam_devices",
        nargs="+",
        default=None,
        help="Optional list of CUDA device identifiers dedicated to SAM (e.g. 4 5 6 7)",
    )
    parser.add_argument(
        "--sam_num_workers",
        type=int,
        default=None,
        help="Number of parallel SAM worker processes (defaults to number of devices)",
    )
    parser.add_argument(
        "--sam_queue_size",
        type=int,
        default=None,
        help="Maximum queued SAM images per worker (defaults to 32)",
    )
    parser.add_argument("--sam_pad_px", type=int, default=10, help="Padding applied around crops in pixels")
    parser.add_argument(
        "--sam_min_ratio",
        type=float,
        default=0.05,
        help="Minimum crop area ratio relative to the page (default: 0.05)",
    )
    parser.add_argument(
        "--sam_max_ratio",
        type=float,
        default=0.70,
        help="Maximum crop area ratio relative to the page (default: 0.70)",
    )
    parser.add_argument(
        "--judge_model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model name or path used for crop judging",
    )
    parser.add_argument(
        "--judge_max_new_tokens",
        type=int,
        default=128,
        help="Maximum tokens generated by the judge model (default: 128)",
    )
    parser.add_argument(
        "--judge_backend",
        choices=("hf", "vllm"),
        default="hf",
        help="Backend for judge model inference (default: hf)",
    )
    parser.add_argument(
        "--judge_gpu_devices",
        type=str,
        help="Comma-separated GPU ids used by the judge backend (relevant for vLLM or single-GPU HF agent)",
    )
    parser.add_argument(
        "--vlm_min_pixels",
        type=int,
        default=256 * 28 * 28,
        help="Minimum pixel count used by the VLM processor (default: 200704)",
    )
    parser.add_argument(
        "--vlm_max_pixels",
        type=int,
        default=1280 * 28 * 28,
        help="Maximum pixel count used by the VLM processor (default: 1003520)",
    )
    parser.add_argument(
        "--keep_workdir",
        action="store_true",
        help="Do not overwrite existing per-document working directories",
    )
    parser.add_argument(
        "--judge_batch_size",
        type=int,
        default=1,
        help="Number of crops to judge per vLLM request (default: 1)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Number of documents to process concurrently (default: 1)",
    )
    return parser.parse_args()


def _iter_pdfs(path: Path) -> Iterable[Path]:
    if path.is_file() and path.suffix.lower() == ".pdf":
        yield path.resolve()
        return
    if not path.is_dir():
        return
    for pdf in sorted(path.glob("*.pdf")):
        if pdf.is_file():
            yield pdf.resolve()


def _parse_devices(arg: Optional[str]) -> Optional[List[int]]:
    if not arg:
        return None
    devices: List[int] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        devices.append(int(part))
    return devices or None


def main() -> None:
    args = parse_args()
    pdfs: List[Path] = list(_iter_pdfs(args.pdf))
    if not pdfs:
        print(f"ERROR: no PDF files found at {args.pdf}")
        return

    judge_agent: Optional[MMAgent] = None
    if args.judge_backend == "vllm":
        judge_agent = MMAgent(
            model_name=args.judge_model,
            use_vllm=True,
            gpu_devices=_parse_devices(args.judge_gpu_devices),
            max_new_tokens=args.judge_max_new_tokens,
            min_pixels=args.vlm_min_pixels,
            max_pixels=args.vlm_max_pixels,
        )

    sam_config = SamCropConfig(
        checkpoint=args.sam_checkpoint,
        device=args.sam_device,
        pad_px=args.sam_pad_px,
        area_min_ratio=args.sam_min_ratio,
        area_max_ratio=args.sam_max_ratio,
        devices=tuple(
            f"cuda:{dev}" if str(dev).isdigit() else str(dev)
            for dev in args.sam_devices
        )
        if args.sam_devices
        else None,
        num_workers=args.sam_num_workers,
        queue_size=args.sam_queue_size or 32,
    )

    judge_config = JudgeConfig(
        model_name=args.judge_model,
        max_new_tokens=args.judge_max_new_tokens,
        batch_size=args.judge_batch_size,
    )

    pdf_config = PdfToPngConfig(
        dpi=args.dpi,
        thread_count=args.pdf_thread_count,
        max_workers=args.pdf_max_workers,
    )

    pipeline_config = SamJudgePipelineConfig(
        work_root=args.work_root,
        sam=sam_config,
        judge=judge_config,
        pdf=pdf_config,
        overwrite=not args.keep_workdir,
        judge_agent=judge_agent,
    )

    max_workers = max(1, args.max_workers)
    max_workers = min(max_workers, len(pdfs))

    resources = SamJudgeResources(
        sam_lock=Lock(),
        judge_lock=Lock() if pipeline_config.judge_agent is None else None,
    )
    processor = SamJudgeProcessor(pipeline_config, resources=resources)

    successes: List[str] = []
    failures: List[str] = []

    def _process(pdf_path: Path) -> None:
        try:
            result = processor.process(pdf_path, args.output_dir)
            print(
                f"SUCCESS: {pdf_path.name} → {result.merged_output} "
                f"(pages={result.pages}, total_crops={result.total_crops}, "
                f"kept_after_clean={result.kept_crops})"
            )
            successes.append(str(result.merged_output))
        except DocumentProcessingError as exc:
            print(f"ERROR: {pdf_path.name} → {exc}")
            failures.append(f"{pdf_path}: {exc}")
        except Exception as exc:  # pragma: no cover
            print(f"ERROR: {pdf_path.name} → {exc}")
            failures.append(f"{pdf_path}: {exc}")

    if max_workers == 1:
        for pdf_path in pdfs:
            _process(pdf_path)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process, pdf_path): pdf_path for pdf_path in pdfs}
            for future in as_completed(futures):
                future.result()

    print(f"\nCompleted {len(pdfs)} document(s).")
    if successes:
        print(f"  ✔ Successful: {len(successes)}")
    if failures:
        print(f"  ✖ Failed: {len(failures)}")
        for item in failures:
            print(f"    - {item}")


if __name__ == "__main__":
    main()
