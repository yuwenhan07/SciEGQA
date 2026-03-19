"""Helper functions for the BoundingDoc pipeline steps.

This package centralises the shared logic that was spread across the
stepN_*.py scripts, so the individual CLIs can stay small and focused.
"""

from .pdf import convert_pdfs_to_pngs  # noqa: F401
from .sam_crop import SamCropper, SamCropConfig  # noqa: F401
from .judge import (
    QwenJudge,
    JudgeConfig,
    run_judge_directory,
    run_judge_root,
    post_clean,
)  # noqa: F401
from .qa import (
    QwenQAGenerator,
    QAGeneratorConfig,
    run_qa_for_directory,
    run_qa_for_root,
)  # noqa: F401
from .data_transform import transform_jsonl_tree  # noqa: F401
from .pipeline import (  # noqa: F401
    DocumentProcessor,
    DocumentPipelineConfig,
    DocumentProcessingResult,
    DocumentProcessingError,
    PipelineResources,
)
from .mm_agent import MMAgent  # noqa: F401

__all__ = [
    "convert_pdfs_to_pngs",
    "SamCropper",
    "SamCropConfig",
    "QwenJudge",
    "JudgeConfig",
    "run_judge_directory",
    "run_judge_root",
    "post_clean",
    "QwenQAGenerator",
    "QAGeneratorConfig",
    "run_qa_for_directory",
    "run_qa_for_root",
    "transform_jsonl_tree",
    "DocumentProcessor",
    "DocumentPipelineConfig",
    "DocumentProcessingResult",
    "DocumentProcessingError",
    "PipelineResources",
    "MMAgent",
]
