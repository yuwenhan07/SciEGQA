#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, List, Sequence


PIPELINE_TEXT = (
    "SciEGQA automatic training-set construction pipeline.\n"
    "This CLI follows the Automatically Training Set Construction workflow in "
    "Scripts/instruction.md:\n"
    "1) PDF -> PNG pages\n"
    "2) SAM region segmentation and crop extraction\n"
    "3) VLM-based crop judgement and clean-up\n"
    "4) QA generation from clean evidence crops\n"
    "5) Final JSONL transformation for training"
)


def _parse_gpu_ids(arg: str | None) -> list[int] | None:
    if not arg:
        return None
    devices: list[int] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        devices.append(int(part))
    return devices or None


def _normalize_cuda_devices(devices: Sequence[str] | None, fallback: str) -> list[str]:
    raw_devices = list(devices) if devices is not None else [fallback]
    normalized: list[str] = []
    for dev in raw_devices:
        dev_str = str(dev).strip()
        if not dev_str:
            continue
        if dev_str.isdigit():
            dev_str = f"cuda:{dev_str}"
        normalized.append(dev_str)
    return normalized


def _iter_pdfs(path: Path) -> Iterable[Path]:
    if path.is_file() and path.suffix.lower() == ".pdf":
        yield path.resolve()
        return
    if not path.is_dir():
        return
    for pdf in sorted(path.glob("*.pdf")):
        if pdf.is_file():
            yield pdf.resolve()


def _build_judge_agent(args: argparse.Namespace) -> Any | None:
    if args.judge_backend != "vllm":
        return None
    from boundingdoc.mm_agent import MMAgent

    return MMAgent(
        model_name=args.judge_model,
        use_vllm=True,
        gpu_devices=_parse_gpu_ids(args.judge_gpu_devices),
        max_new_tokens=args.judge_max_new_tokens,
        min_pixels=args.vlm_min_pixels,
        max_pixels=args.vlm_max_pixels,
    )


def _build_shared_or_qa_agent(
    args: argparse.Namespace,
    judge_agent: Any | None,
) -> Any | None:
    qa_ids = _parse_gpu_ids(args.qa_gpu_devices)
    if args.qa_backend == "vllm":
        if args.share_vlm_agent and judge_agent and args.qa_model == args.judge_model:
            judge_agent.max_new_tokens = args.qa_max_new_tokens
            return judge_agent
        from boundingdoc.mm_agent import MMAgent

        return MMAgent(
            model_name=args.qa_model,
            use_vllm=True,
            gpu_devices=qa_ids,
            max_new_tokens=args.qa_max_new_tokens,
            min_pixels=args.vlm_min_pixels,
            max_pixels=args.vlm_max_pixels,
        )
    if args.qa_backend == "hf" and args.share_vlm_agent:
        if judge_agent and args.qa_model == args.judge_model:
            judge_agent.max_new_tokens = args.qa_max_new_tokens
            return judge_agent
        from boundingdoc.mm_agent import MMAgent

        return MMAgent(
            model_name=args.qa_model,
            use_vllm=False,
            gpu_devices=qa_ids,
            max_new_tokens=args.qa_max_new_tokens,
            min_pixels=args.vlm_min_pixels,
            max_pixels=args.vlm_max_pixels,
        )
    return None


def _build_judge(args: argparse.Namespace) -> QwenJudge:
    from boundingdoc.judge import JudgeConfig, QwenJudge

    agent = None
    if args.backend == "vllm":
        from boundingdoc.mm_agent import MMAgent

        agent = MMAgent(
            model_name=args.model,
            use_vllm=True,
            gpu_devices=_parse_gpu_ids(args.gpu_devices),
            max_new_tokens=args.max_new_tokens,
            min_pixels=args.vlm_min_pixels,
            max_pixels=args.vlm_max_pixels,
        )
    return QwenJudge(
        JudgeConfig(
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        ),
        agent_backend=agent,
    )


def _build_generator(args: argparse.Namespace) -> QwenQAGenerator:
    from boundingdoc.qa import QAGeneratorConfig, QwenQAGenerator

    agent = None
    if args.backend == "vllm":
        from boundingdoc.mm_agent import MMAgent

        agent = MMAgent(
            model_name=args.model,
            use_vllm=True,
            gpu_devices=_parse_gpu_ids(args.gpu_devices),
            max_new_tokens=args.max_new_tokens,
            min_pixels=args.vlm_min_pixels,
            max_pixels=args.vlm_max_pixels,
        )
    return QwenQAGenerator(
        QAGeneratorConfig(
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            batch_size=args.batch_size,
        ),
        agent_backend=agent,
    )


def _rename_if_needed(output_dir: Path, desired_path: Path) -> None:
    default_path = output_dir / "judge_results.json"
    if default_path == desired_path:
        return
    desired_path.parent.mkdir(parents=True, exist_ok=True)
    if default_path.exists():
        default_path.replace(desired_path)


def _add_pdf_render_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dpi", type=int, default=300, help="PDF render DPI")
    parser.add_argument(
        "--thread-count",
        dest="thread_count",
        type=int,
        default=None,
        help="Per-PDF rendering threads for pdf2image/pdftoppm",
    )
    parser.add_argument(
        "--max-workers",
        dest="max_workers",
        type=int,
        default=None,
        help="Number of PDFs to convert concurrently",
    )


def _add_sam_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        default="cuda",
        help="Computation device when --devices is not provided",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="Optional list of CUDA devices, for example: 0 1 or cuda:0 cuda:1",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel SAM worker processes",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=None,
        help="Maximum queued images waiting per worker",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Max page PNGs to process per document; <=0 means all pages",
    )
    parser.add_argument("--pad_px", type=int, default=10, help="Extra crop padding in pixels")
    parser.add_argument("--min_ratio", type=float, default=0.05, help="Minimum crop area ratio")
    parser.add_argument("--max_ratio", type=float, default=0.70, help="Maximum crop area ratio")


def _add_judge_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="Judge model name or path")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max generated tokens")
    parser.add_argument(
        "--backend",
        choices=("hf", "vllm"),
        default="hf",
        help="Inference backend",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Crops per judge request when the backend supports batching",
    )
    parser.add_argument(
        "--gpu_devices",
        type=str,
        help="Comma-separated GPU ids for the judge backend",
    )
    parser.add_argument(
        "--vlm_min_pixels",
        type=int,
        default=256 * 28 * 28,
        help="Minimum pixel count used by the VLM processor",
    )
    parser.add_argument(
        "--vlm_max_pixels",
        type=int,
        default=1280 * 28 * 28,
        help="Maximum pixel count used by the VLM processor",
    )


def _add_qa_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="QA model name or path")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max generated tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.05, help="Repetition penalty")
    parser.add_argument(
        "--backend",
        choices=("hf", "vllm"),
        default="hf",
        help="Inference backend",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Crops per QA request when the backend supports batching",
    )
    parser.add_argument(
        "--gpu_devices",
        type=str,
        help="Comma-separated GPU ids for the QA backend",
    )
    parser.add_argument(
        "--vlm_min_pixels",
        type=int,
        default=256 * 28 * 28,
        help="Minimum pixel count used by the VLM processor",
    )
    parser.add_argument(
        "--vlm_max_pixels",
        type=int,
        default=1280 * 28 * 28,
        help="Maximum pixel count used by the VLM processor",
    )


def _cmd_pdf2png(args: argparse.Namespace) -> int:
    from boundingdoc.pdf import PdfToPngConfig, convert_pdfs_to_pngs

    try:
        generated = convert_pdfs_to_pngs(
            args.pdfs,
            args.output,
            PdfToPngConfig(
                dpi=args.dpi,
                thread_count=args.thread_count,
                max_workers=args.max_workers,
            ),
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    if not generated:
        print("WARNING: no PNG images were generated.")
        return 0
    print(f"SUCCESS: generated {len(generated)} PNG image(s). Output directory: {args.output.resolve()}")
    return 0


def _cmd_sam(args: argparse.Namespace) -> int:
    from boundingdoc.sam_crop import SamCropConfig, SamCropper, run_parallel_sam

    devices = _normalize_cuda_devices(args.devices, args.device)
    primary_device = devices[0] if devices else args.device
    config = SamCropConfig(
        checkpoint=args.checkpoint,
        device=primary_device,
        pad_px=args.pad_px,
        area_min_ratio=args.min_ratio,
        area_max_ratio=args.max_ratio,
        devices=tuple(devices) if devices else None,
        num_workers=args.num_workers,
        queue_size=args.queue_size or 32,
        max_pages_per_doc=args.sample_size,
    )

    use_parallel = len(devices) > 1 or (args.num_workers and args.num_workers > 1)
    if use_parallel:
        stats = run_parallel_sam(
            config,
            args.input_dir,
            args.output_root,
            devices=devices,
            num_workers=args.num_workers,
            queue_size=args.queue_size,
        )
    else:
        stats = SamCropper(config).run(args.input_dir, args.output_root)

    print(
        f"Completed: processed {stats.processed_sets} set(s), {stats.processed_images} image(s), "
        f"successful crops for {stats.succeeded_images} image(s). Output root: {args.output_root.resolve()}"
    )
    return 0


def _cmd_judge(args: argparse.Namespace) -> int:
    from boundingdoc.judge import post_clean, run_judge_directory, run_judge_root

    if args.root_dir:
        root = args.root_dir
        if not root.is_dir():
            print(f"ERROR: root directory does not exist: {root}")
            return 1
        judge = _build_judge(args)
        totals = run_judge_root(root, judge, post_clean_only=args.post_clean, batch_size=args.batch_size)
        types = totals["type_counts"]
        print(
            f"Summary: processed {totals['pages']} page(s); evaluated {totals['images']} image(s); "
            f"kept {totals['clean_kept']} crop(s) after cleaning. Root directory: {root}"
        )
        print(
            f"keep=true: {totals['keep_true']} ; keep=false: {totals['keep_false']} ; "
            f"type=text:{types['text']} table:{types['table']} image:{types['image']} unknown:{types['unknown']}"
        )
        return 0

    if args.post_clean:
        if not (args.judge_json and args.crops_dir and args.clean_output):
            print("ERROR: post_clean mode requires --judge_json, --crops_dir, and --clean_output")
            return 1
        post_clean(
            judge_json=args.judge_json,
            crops_dir=args.crops_dir,
            output_dir=args.clean_output,
            filtered_json=args.filtered_json,
            filtered_dir=args.filtered_json_dir,
        )
        return 0

    if not args.image_dir:
        print("ERROR: provide either --root_dir or --image_dir")
        return 1
    if not args.image_dir.is_dir():
        print(f"ERROR: directory does not exist: {args.image_dir}")
        return 1

    judge = _build_judge(args)
    output_dir = args.output.parent if args.output.parent != Path() else Path(".")
    results = run_judge_directory(args.image_dir, judge, output_dir, batch_size=args.batch_size)
    _rename_if_needed(output_dir, args.output)
    print(f"Evaluation complete for {len(results)} image(s). Results saved to: {args.output}")
    return 0


def _cmd_qa(args: argparse.Namespace) -> int:
    from boundingdoc.qa import build_evidence_map, run_qa_for_directory, run_qa_for_root

    generator = _build_generator(args)
    if args.root_dir:
        stats = run_qa_for_root(generator, args.root_dir, args.clean_summary, batch_size=args.batch_size)
        print(
            f"Summary: processed {stats['pages']} page(s); "
            f"generated QA for {stats['images']} image(s). Root directory: {args.root_dir}"
        )
        return 0

    if not args.image_dir or not args.output:
        print("ERROR: single-directory mode requires both --image_dir and --output")
        return 1

    evidence = build_evidence_map(args.image_dir, args.clean_summary)
    stats = run_qa_for_directory(generator, args.image_dir, args.output, evidence, batch_size=args.batch_size)
    out_path = Path(args.output).with_suffix(".jsonl")
    print(f"SUCCESS: wrote {stats['images']} image record(s) to {out_path}")
    return 0


def _cmd_transform(args: argparse.Namespace) -> int:
    from boundingdoc.data_transform import transform_jsonl_tree

    transform_jsonl_tree(args.input_dir, args.output, args.type_map)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from boundingdoc.judge import JudgeConfig
    from boundingdoc.mm_agent import MMAgent
    from boundingdoc.pdf import PdfToPngConfig
    from boundingdoc.pipeline import (
        DocumentPipelineConfig,
        DocumentProcessingError,
        DocumentProcessor,
        PipelineResources,
    )
    from boundingdoc.qa import QAGeneratorConfig
    from boundingdoc.sam_crop import SamCropConfig

    pdfs: List[Path] = list(_iter_pdfs(args.pdf))
    if not pdfs:
        print(f"ERROR: no PDF files found at {args.pdf}")
        return 1

    judge_agent = None
    if args.judge_backend == "vllm":
        judge_agent = MMAgent(
            model_name=args.judge_model,
            use_vllm=True,
            gpu_devices=_parse_gpu_ids(args.judge_gpu_devices),
            max_new_tokens=args.judge_max_new_tokens,
            min_pixels=args.vlm_min_pixels,
            max_pixels=args.vlm_max_pixels,
        )
    elif args.judge_backend == "hf" and args.share_vlm_agent:
        judge_agent = MMAgent(
            model_name=args.judge_model,
            use_vllm=False,
            gpu_devices=_parse_gpu_ids(args.judge_gpu_devices),
            max_new_tokens=args.judge_max_new_tokens,
            min_pixels=args.vlm_min_pixels,
            max_pixels=args.vlm_max_pixels,
        )

    qa_agent = _build_shared_or_qa_agent(args, judge_agent)
    sam_devices = _normalize_cuda_devices(args.sam_devices, args.sam_device)

    pipeline_config = DocumentPipelineConfig(
        work_root=args.work_root,
        sam=SamCropConfig(
            checkpoint=args.sam_checkpoint,
            device=sam_devices[0] if sam_devices else args.sam_device,
            pad_px=args.sam_pad_px,
            area_min_ratio=args.sam_min_ratio,
            area_max_ratio=args.sam_max_ratio,
            devices=tuple(sam_devices) if sam_devices else None,
            num_workers=args.sam_num_workers,
            queue_size=args.sam_queue_size or 32,
        ),
        judge=JudgeConfig(
            model_name=args.judge_model,
            max_new_tokens=args.judge_max_new_tokens,
            batch_size=args.judge_batch_size,
        ),
        qa=QAGeneratorConfig(
            model_name=args.qa_model,
            max_new_tokens=args.qa_max_new_tokens,
            temperature=args.qa_temperature,
            top_p=args.qa_top_p,
            repetition_penalty=args.qa_repetition_penalty,
            batch_size=args.qa_batch_size,
            workers=max(1, args.qa_workers),
        ),
        pdf=PdfToPngConfig(
            dpi=args.dpi,
            thread_count=args.pdf_thread_count,
            max_workers=args.pdf_max_workers,
        ),
        overwrite=not args.keep_workdir,
        judge_agent=judge_agent,
        qa_agent=qa_agent,
    )

    max_workers = min(max(1, args.max_workers), len(pdfs))
    resources = PipelineResources(
        sam_lock=Lock(),
        judge_lock=Lock() if pipeline_config.judge_agent is None else None,
        qa_lock=Lock() if pipeline_config.qa_agent is None else None,
    )
    processor = DocumentProcessor(pipeline_config, resources=resources)

    successes: list[str] = []
    failures: list[str] = []

    def _process_one(pdf_path: Path) -> None:
        try:
            result = processor.process(pdf_path, args.output_dir)
            print(
                f"SUCCESS: {pdf_path.name} -> {result.qa_output} "
                f"(pages={result.pages}, crops={result.total_crops}, clean={result.clean_crops}, "
                f"qa_images={result.qa_images}, qa_pairs={result.qa_pairs})"
            )
            successes.append(str(result.qa_output))
        except DocumentProcessingError as exc:
            print(f"ERROR: {pdf_path.name} -> {exc}")
            failures.append(f"{pdf_path}: {exc}")
        except Exception as exc:
            print(f"ERROR: {pdf_path.name} -> {exc}")
            failures.append(f"{pdf_path}: {exc}")

    if max_workers == 1:
        for pdf_path in pdfs:
            _process_one(pdf_path)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_one, pdf_path): pdf_path for pdf_path in pdfs}
            for future in as_completed(futures):
                future.result()

    print(f"Completed {len(pdfs)} document(s).")
    if successes:
        print(f"Successful: {len(successes)}")
    if failures:
        print(f"Failed: {len(failures)}")
        for item in failures:
            print(f"  - {item}")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=PIPELINE_TEXT,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pdf_parser = subparsers.add_parser("pdf2png", help="Step 0: convert PDF files into page PNG images")
    pdf_parser.add_argument("pdfs", type=Path, help="PDF file or directory containing PDF files")
    pdf_parser.add_argument("output", type=Path, help="Output directory for PNG pages")
    _add_pdf_render_args(pdf_parser)
    pdf_parser.set_defaults(handler=_cmd_pdf2png)

    sam_parser = subparsers.add_parser("sam", help="Step 1: run SAM segmentation and save candidate crops")
    sam_parser.add_argument("--input_dir", required=True, type=Path, help="Root directory or single page folder of PNG images")
    sam_parser.add_argument(
        "--output_root",
        default=Path("./output"),
        type=Path,
        help="Output root; saves results as {output_root}/{doc}/{page}/...",
    )
    sam_parser.add_argument("--checkpoint", required=True, type=str, help="SAM checkpoint path")
    _add_sam_args(sam_parser)
    sam_parser.set_defaults(handler=_cmd_sam)

    judge_parser = subparsers.add_parser("judge", help="Step 2: judge candidate crops and run clean-up")
    judge_parser.add_argument("--root_dir", type=Path, help="Batch mode root directory containing SAM outputs")
    judge_parser.add_argument("--image_dir", type=Path, help="Single-directory mode crops directory")
    judge_parser.add_argument(
        "--output",
        type=Path,
        default=Path("judge_results.json"),
        help="Single-directory mode output path",
    )
    judge_parser.add_argument("--post_clean", action="store_true", help="Run clean-up only, skip inference")
    judge_parser.add_argument("--judge_json", type=Path, help="post_clean mode judge_results.json path")
    judge_parser.add_argument("--crops_dir", type=Path, help="post_clean mode crops directory")
    judge_parser.add_argument("--clean_output", type=Path, help="post_clean mode clean_crops output directory")
    judge_parser.add_argument("--filtered_json", type=Path, help="post_clean mode single *_filtered.json path")
    judge_parser.add_argument("--filtered_json_dir", type=Path, help="post_clean mode directory of *_filtered.json files")
    _add_judge_model_args(judge_parser)
    judge_parser.set_defaults(handler=_cmd_judge)

    qa_parser = subparsers.add_parser("qa", help="Step 3: generate QA pairs from clean evidence crops")
    qa_parser.add_argument("--root_dir", type=Path, help="Batch mode root directory containing report/page folders")
    qa_parser.add_argument("--image_dir", type=Path, help="Single-directory mode clean_crops directory")
    qa_parser.add_argument("--output", type=Path, help="Single-directory mode output JSONL path")
    qa_parser.add_argument("--clean_summary", type=Path, help="Optional explicit clean_summary.json path")
    _add_qa_model_args(qa_parser)
    qa_parser.set_defaults(handler=_cmd_qa)

    transform_parser = subparsers.add_parser(
        "transform",
        help="Step 4: merge generated QA JSONL files into the final training format",
    )
    transform_parser.add_argument(
        "--input_dir",
        required=True,
        type=Path,
        help="Root directory; recursively scans for generated *.jsonl files",
    )
    transform_parser.add_argument("--output", required=True, type=Path, help="Output JSONL path")
    transform_parser.add_argument("--type_map", type=Path, help="Optional JSON file mapping evidence.image to type")
    transform_parser.set_defaults(handler=_cmd_transform)

    run_parser = subparsers.add_parser("run", help="Recommended: run the full PDF -> crop -> judge -> QA pipeline")
    run_parser.add_argument("pdf", type=Path, help="PDF file or directory containing PDF files to process")
    run_parser.add_argument(
        "--work_root",
        required=True,
        type=Path,
        help="Working directory for intermediate pages, crops, clean_crops, and QA outputs",
    )
    run_parser.add_argument(
        "--output_dir",
        type=Path,
        help="Optional directory where final {doc}.jsonl files will be copied",
    )
    run_parser.add_argument("--dpi", type=int, default=300, help="DPI used when rasterising PDF pages")
    run_parser.add_argument(
        "--pdf_thread_count",
        type=int,
        default=None,
        help="Per-PDF rendering threads passed to pdf2image/pdftoppm",
    )
    run_parser.add_argument(
        "--pdf_max_workers",
        type=int,
        default=None,
        help="Number of PDFs to convert concurrently",
    )
    run_parser.add_argument("--sam_checkpoint", required=True, type=str, help="SAM checkpoint path")
    run_parser.add_argument("--sam_device", default="cuda", help="Device for SAM inference")
    run_parser.add_argument(
        "--sam_devices",
        nargs="+",
        default=None,
        help="Optional list of CUDA device identifiers dedicated to SAM",
    )
    run_parser.add_argument("--sam_num_workers", type=int, default=None, help="Number of SAM worker processes")
    run_parser.add_argument("--sam_queue_size", type=int, default=None, help="Maximum queued SAM images per worker")
    run_parser.add_argument("--sam_pad_px", type=int, default=10, help="Padding applied around crops in pixels")
    run_parser.add_argument("--sam_min_ratio", type=float, default=0.05, help="Minimum crop area ratio")
    run_parser.add_argument("--sam_max_ratio", type=float, default=0.70, help="Maximum crop area ratio")
    run_parser.add_argument(
        "--judge_model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model used for crop judgement",
    )
    run_parser.add_argument("--judge_max_new_tokens", type=int, default=128, help="Judge max generated tokens")
    run_parser.add_argument(
        "--qa_model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model used for QA generation",
    )
    run_parser.add_argument("--qa_max_new_tokens", type=int, default=512, help="QA max generated tokens")
    run_parser.add_argument("--qa_temperature", type=float, default=0, help="QA sampling temperature")
    run_parser.add_argument("--qa_top_p", type=float, default=1, help="QA top-p nucleus sampling")
    run_parser.add_argument(
        "--qa_repetition_penalty",
        type=float,
        default=1.05,
        help="QA repetition penalty",
    )
    run_parser.add_argument(
        "--judge_backend",
        choices=("hf", "vllm"),
        default="hf",
        help="Backend for judge inference",
    )
    run_parser.add_argument(
        "--judge_gpu_devices",
        type=str,
        help="Comma-separated GPU ids used by the judge backend",
    )
    run_parser.add_argument(
        "--qa_backend",
        choices=("hf", "vllm"),
        default="hf",
        help="Backend for QA generation inference",
    )
    run_parser.add_argument(
        "--qa_gpu_devices",
        type=str,
        help="Comma-separated GPU ids used by the QA backend",
    )
    run_parser.add_argument(
        "--vlm_min_pixels",
        type=int,
        default=256 * 28 * 28,
        help="Minimum pixel count used by the VLM processor",
    )
    run_parser.add_argument(
        "--vlm_max_pixels",
        type=int,
        default=1280 * 28 * 28,
        help="Maximum pixel count used by the VLM processor",
    )
    run_parser.add_argument(
        "--share_vlm_agent",
        action="store_true",
        help="Reuse a single VLM agent instance for both judge and QA when models match",
    )
    run_parser.add_argument(
        "--keep_workdir",
        action="store_true",
        help="Do not overwrite existing per-document working directories",
    )
    run_parser.add_argument(
        "--judge_batch_size",
        type=int,
        default=1,
        help="Crops per judge request when batching is available",
    )
    run_parser.add_argument(
        "--qa_batch_size",
        type=int,
        default=1,
        help="Crops per QA request when batching is available",
    )
    run_parser.add_argument(
        "--qa_workers",
        type=int,
        default=1,
        help="Number of parallel QA worker threads for supported backends",
    )
    run_parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Number of documents to process concurrently",
    )
    run_parser.set_defaults(handler=_cmd_run)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
