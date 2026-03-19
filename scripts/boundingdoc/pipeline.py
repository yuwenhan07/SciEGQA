from __future__ import annotations

import shutil
from pathlib import Path
from threading import Lock
from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter

from .judge import JudgeConfig, QwenJudge, run_judge_root
from .mm_agent import MMAgent
from .pdf import PdfToPngConfig, convert_pdfs_to_pngs
from .qa import QAGeneratorConfig, QwenQAGenerator, run_qa_for_root
from .sam_crop import SamCropConfig, SamCropper, SamStats, run_parallel_sam


class DocumentProcessingError(RuntimeError):
    """Raised when a document cannot be processed end-to-end."""


@dataclass(slots=True)
class DocumentPipelineConfig:
    """Configuration bundle for the per-document processing pipeline."""

    work_root: Path
    sam: SamCropConfig
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    qa: QAGeneratorConfig = field(default_factory=QAGeneratorConfig)
    pdf: PdfToPngConfig = field(default_factory=PdfToPngConfig)
    overwrite: bool = True
    judge_agent: MMAgent | None = None
    qa_agent: MMAgent | None = None

    def __post_init__(self) -> None:
        self.work_root = Path(self.work_root)


@dataclass(slots=True)
class PipelineResources:
    """Shared heavy resources and optional locks for the document pipeline."""

    sam_cropper: SamCropper | None = None
    judge: QwenJudge | None = None
    qa: QwenQAGenerator | None = None
    sam_lock: Lock | None = None
    judge_lock: Lock | None = None
    qa_lock: Lock | None = None


@dataclass(slots=True)
class DocumentProcessingResult:
    """Summary of the artefacts generated for a single document."""

    doc_name: str
    work_root: Path
    qa_output: Path
    pages: int
    qa_images: int
    total_crops: int
    clean_crops: int
    qa_pairs: int


class DocumentProcessor:
    """Run the full PDF → crops → judge → QA pipeline for one document."""

    def __init__(self, config: DocumentPipelineConfig, resources: PipelineResources | None = None):
        self.config = config
        self._work_root = config.work_root.resolve()
        self._work_root.mkdir(parents=True, exist_ok=True)

        self._resources = resources or PipelineResources()

        self._parallel_sam = bool(
            (config.sam.devices and len(config.sam.devices) > 1)
            or (config.sam.num_workers and config.sam.num_workers > 1)
        )

        cropper = self._resources.sam_cropper
        if self._parallel_sam:
            self._cropper: SamCropper | None = cropper
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

        qa_generator = self._resources.qa
        if qa_generator is None:
            qa_generator = QwenQAGenerator(config.qa, agent_backend=config.qa_agent)
            self._resources.qa = qa_generator
        self._qa_generator = qa_generator

        self._sam_lock = self._resources.sam_lock
        self._judge_lock = self._resources.judge_lock
        self._qa_lock = self._resources.qa_lock

    def process(
        self,
        pdf_path: Path,
        final_output_dir: Path | None = None,
    ) -> DocumentProcessingResult:
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
                return run_parallel_sam(self.config.sam, page_dir, processed_root)
            assert self._cropper is not None
            return self._cropper.run(page_dir, processed_root)

        timings: dict[str, float] = {}
        timings_printed = False

        def _emit_timings() -> None:
            nonlocal timings_printed
            if timings_printed or not timings:
                return
            print(
                f"TIMER {doc_name}: "
                f"sam={timings.get('sam_sec', 0):.1f}s | "
                f"judge={timings.get('judge_sec', 0):.1f}s | "
                f"qa={timings.get('qa_sec', 0):.1f}s"
            )
            timings_printed = True

        if self._sam_lock:
            with self._sam_lock:
                sam_start = perf_counter()
                crop_stats = _run_sam()
                sam_elapsed = perf_counter() - sam_start
        else:
            sam_start = perf_counter()
            crop_stats = _run_sam()
            sam_elapsed = perf_counter() - sam_start
        timings["sam_sec"] = sam_elapsed
        _log_stage("sam", sam_elapsed)
        if crop_stats.succeeded_images == 0:
            raise DocumentProcessingError(f"SAM cropping produced no crops for {pdf_path}")

        judge_stats: dict[str, int] | None = None
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
        timings["judge_sec"] = judge_elapsed
        _log_stage("judge", judge_elapsed)

        qa_stats: dict[str, int] = {}
        if self._qa_lock:
            with self._qa_lock:
                qa_start = perf_counter()
                qa_stats = run_qa_for_root(
                    self._qa_generator,
                    processed_root,
                    batch_size=getattr(self._qa_generator.config, "batch_size", 1),
                )
                qa_elapsed = perf_counter() - qa_start
        else:
            qa_start = perf_counter()
            qa_stats = run_qa_for_root(
                self._qa_generator,
                processed_root,
                batch_size=getattr(self._qa_generator.config, "batch_size", 1),
            )
            qa_elapsed = perf_counter() - qa_start
        timings["qa_sec"] = qa_elapsed
        _log_stage("qa", qa_elapsed)

        merged_path = processed_root / doc_name / f"{doc_name}.jsonl"

        _emit_timings()

        if not merged_path.exists():
            raise DocumentProcessingError(f"QA output not found for {pdf_path}")

        output_path = merged_path
        if final_output_dir:
            final_output_dir = Path(final_output_dir).resolve()
            final_output_dir.mkdir(parents=True, exist_ok=True)
            output_path = final_output_dir / merged_path.name
            shutil.copy2(merged_path, output_path)

        _emit_timings()

        total_elapsed = perf_counter() - overall_start
        timings["total_sec"] = total_elapsed
        _log_stage("complete", total_elapsed)

        total_crops = judge_stats.get("images", 0) if judge_stats else 0
        clean_crops = judge_stats.get("clean_kept", 0) if judge_stats else 0
        qa_images = qa_stats.get("images", 0)
        qa_pairs = qa_stats.get("qa_pairs", 0)

        print(
            f"STATS {doc_name}: crops={total_crops}, clean={clean_crops}, "
            f"qa_images={qa_images}, qa_pairs={qa_pairs}"
        )

        return DocumentProcessingResult(
            doc_name=doc_name,
            work_root=doc_root,
            qa_output=output_path,
            pages=qa_stats.get("pages", 0),
            qa_images=qa_images,
            total_crops=total_crops,
            clean_crops=clean_crops,
            qa_pairs=qa_pairs,
        )
