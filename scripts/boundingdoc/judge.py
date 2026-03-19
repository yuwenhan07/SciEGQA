from __future__ import annotations

import json
import re
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .mm_agent import MMAgent

PROMPT = (
    "Please determine whether this image represents a correct and sufficiently fine-grained content crop.\n"
    "**Positive case (keep=true) must satisfy ALL of the following:**\n"
    "A. Contains only **one** complete logical block (a full paragraph, full table, or full chart) with no obvious truncation at the borders;\n"
    "B. The meaningful content occupies a **significant portion** of the image (approximately ≥30%), not dominated by blank space;\n"
    "C. The image is clean — no headers, footers, page numbers, watermarks, or decorative frames unrelated to the main content.\n"
    "**Negative case (keep=false) if ANY of the following is true:**\n"
    "- The image is mostly blank or has excessive whitespace;\n"
    "- It only contains a header, footer, page number, year, or report title (e.g., \"4  2006 Annual Report\");\n"
    "- It contains only a caption or title **without** the corresponding figure or table body;\n"
    "- Multiple logical blocks appear simultaneously (e.g., double columns, cross-page layout, or figure + text mixed together);\n"
    "- The content is truncated (for example, only part of a figure, incomplete table borders, or paragraph cut at the top/bottom);\n"
    "- The cropped content is too small, blurry, or unreadable;\n"
    "- Large margins, edges, or decorative borders occupy most of the frame.\n"
    "\n"
    "In addition to keep, classify the primary content type of the crop as one of: \"text\", \"table\", or \"image\".\n"
    "- \"text\": a body paragraph or list; predominantly textual without tabular gridlines.\n"
    "- \"table\": a grid-like table or matrix with visible rows/columns.\n"
    "- \"image\": a chart/figure/graphic/photo/diagram (including plots and bar/line charts).\n"
    "\n"
    "Output a strict JSON object on a single line:\n"
    "{\"keep\": true|false, \"type\": \"text\"|\"table\"|\"image\"}\n"
    "**If uncertain, set keep=false and choose the most plausible type.**"
)


@dataclass(slots=True)
class JudgeConfig:
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    prompt: str = PROMPT
    max_new_tokens: int = 128
    batch_size: int = 1


@dataclass(slots=True)
class JudgeResult:
    image: str
    keep: bool
    content_type: str


class QwenJudge:
    """Wrapper around the Qwen VL model for crop quality judgement."""

    def __init__(self, config: JudgeConfig, agent_backend: MMAgent | None = None):
        self.config = config
        self._agent = agent_backend
        if self._agent is None:
            print(f"Loading model: {config.model_name}")
            self.processor = AutoProcessor.from_pretrained(
                config.model_name,
                trust_remote_code=True,
                use_fast=False,
            )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.model_name,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            self.model.eval()
        else:
            self.processor = None
            self.model = None

    @property
    def supports_batch(self) -> bool:
        return self._agent is not None

    def judge_image(self, image_path: Path) -> JudgeResult:
        if self._agent is not None:
            message = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.config.prompt},
                            {"type": "image", "image": str(image_path)},
                        ],
                    }
                ]
            ]
            decoded = self._agent.generate(
                message,
                max_new_tokens=self.config.max_new_tokens,
                temperature=0.0,
            )[0]
        else:
            with Image.open(image_path) as pil_image:
                img = pil_image.convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.config.prompt},
                        {"type": "image", "image": img},
                    ],
                }
            ]
            chat_text = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = self.processor(
                text=[chat_text],
                images=[img],
                return_tensors="pt",
            ).to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                )
            generated = output_ids[:, inputs["input_ids"].shape[1]:]
            decoded = self.processor.batch_decode(
                generated,
                skip_special_tokens=True,
            )[0]
        parsed = _parse_json_response(decoded)
        return JudgeResult(
            image=image_path.name,
            keep=bool(parsed.get("keep", False)),
            content_type=parsed.get("type", "unknown"),
        )

    def judge_batch(self, image_paths: Sequence[Path]) -> List[JudgeResult]:
        if not image_paths:
            return []
        if self._agent is None:
            return [self.judge_image(path) for path in image_paths]

        conversations: List[List[Dict]] = []
        for path in image_paths:
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.config.prompt},
                            {"type": "image", "image": str(path)},
                        ],
                    }
                ]
            )

        outputs = self._agent.generate(
            conversations,
            max_new_tokens=self.config.max_new_tokens,
            temperature=0.0,
        )
        results: List[JudgeResult] = []
        for path, text in zip(image_paths, outputs):
            parsed = _parse_json_response(text)
            results.append(
                JudgeResult(
                    image=path.name,
                    keep=bool(parsed.get("keep", False)),
                    content_type=parsed.get("type", "unknown"),
                )
            )
        return results


def _parse_json_response(text: str) -> Dict[str, object]:
    stripped = text.strip()
    for candidate in (stripped, _extract_first_braced_block(stripped)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            keep = bool(obj.get("keep", False))
            raw_type = str(obj.get("type", "unknown")).lower()
            if raw_type not in {"text", "table", "image"}:
                raw_type = "unknown"
            return {"keep": keep, "type": raw_type}
    return {"keep": False, "type": "unknown"}


def _extract_first_braced_block(text: str) -> Optional[str]:
    match = re.search(r"\{[\s\S]*?\}", text)
    return match.group(0) if match else None


def run_judge_directory(
    image_dir: Path,
    judge: QwenJudge,
    output_dir: Path,
    batch_size: int | None = None,
) -> List[JudgeResult]:
    image_paths = sorted(
        p for p in image_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    )
    results: List[JudgeResult] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    effective_batch = max(1, batch_size or getattr(judge.config, "batch_size", 1))
    if not judge.supports_batch:
        effective_batch = 1

    def _chunk(seq: Sequence[Path], size: int) -> Iterable[List[Path]]:
        for idx in range(0, len(seq), size):
            yield seq[idx : idx + size]

    def _emit_warning(context: str, exc: Exception) -> None:
        detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        print(f"WARNING: {context}")
        print(detail.rstrip())

    progress = tqdm(total=len(image_paths), desc=f"Evaluating: {image_dir.name}", leave=False)
    for chunk in _chunk(image_paths, effective_batch):
        try:
            batch_results = judge.judge_batch(chunk)
        except Exception as exc:
            name_hint = chunk[0].name if chunk else "empty-chunk"
            _emit_warning(f"judge_batch failed for {image_dir.name}/{name_hint}", exc)
            batch_results = []
            for path in chunk:
                try:
                    batch_results.append(judge.judge_image(path))
                except Exception as exc:
                    _emit_warning(f"skipped {path.name}", exc)
        results.extend(batch_results)
        progress.update(len(chunk))
    progress.close()

    results_json = [
        {"image": r.image, "keep": r.keep, "type": r.content_type} for r in results
    ]
    with (output_dir / "judge_results.json").open("w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)

    summary = _summarise_results(results)
    with (output_dir / "judge_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return results


def _summarise_results(results: Sequence[JudgeResult]) -> Dict[str, object]:
    counts = {"text": 0, "table": 0, "image": 0, "unknown": 0}
    keep_true = sum(1 for r in results if r.keep)
    for r in results:
        counts[r.content_type] = counts.get(r.content_type, 0) + 1
    return {
        "total": len(results),
        "keep_true": keep_true,
        "keep_false": len(results) - keep_true,
        "by_type": counts,
    }


def _iter_output_pages(root_dir: Path) -> Iterator[Tuple[str, str, Path, Path]]:
    for report_dir in sorted(root_dir.iterdir(), key=lambda p: p.name):
        if not report_dir.is_dir():
            continue
        for page_dir in sorted(report_dir.iterdir(), key=lambda p: (len(p.name), p.name)):
            if not page_dir.is_dir():
                continue
            crops_dir = page_dir / "crops"
            if crops_dir.is_dir():
                yield report_dir.name, page_dir.name, page_dir, crops_dir


def run_judge_root(
    root_dir: Path,
    judge: QwenJudge,
    post_clean_only: bool = False,
    batch_size: int | None = None,
) -> Dict[str, int]:
    totals = {
        "pages": 0,
        "images": 0,
        "keep_true": 0,
        "keep_false": 0,
        "clean_kept": 0,
        "type_counts": {"text": 0, "table": 0, "image": 0, "unknown": 0},
        "pages_processed": 0,
    }

    for report, page, final_dir, crops_dir in tqdm(
        list(_iter_output_pages(root_dir)), desc="Reports/Pages", unit="page"
    ):
        judge_dir = final_dir / "judge"
        clean_dir = final_dir / "clean_crops"
        judge_dir.mkdir(parents=True, exist_ok=True)

        if not post_clean_only:
            results = run_judge_directory(
                crops_dir,
                judge,
                judge_dir,
                batch_size=batch_size,
            )
        else:
            results = _load_existing_results(judge_dir / "judge_results.json")
            if results is None:
                print(f"WARNING: skipped {report}/{page}: missing judge_results.json")
                continue

        summary = _summarise_results(results)
        totals["pages"] += 1
        totals["images"] += summary["total"]
        totals["keep_true"] += summary["keep_true"]
        totals["keep_false"] += summary["keep_false"]
        for k, v in summary["by_type"].items():
            totals["type_counts"][k] = totals["type_counts"].get(k, 0) + v

        clean_summary = post_clean(
            judge_json=judge_dir / "judge_results.json",
            crops_dir=crops_dir,
            output_dir=clean_dir,
            filtered_dir=final_dir,
        )
        totals["clean_kept"] += clean_summary.get("kept_after_clean", 0)
        totals["pages_processed"] = totals.get("pages_processed", 0) + 1

    return totals


def _load_existing_results(path: Path) -> Optional[List[JudgeResult]]:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"WARNING: failed to read {path}: {exc}")
        return None
    results: List[JudgeResult] = []
    for entry in raw:
        try:
            results.append(
                JudgeResult(
                    image=str(entry.get("image")),
                    keep=bool(entry.get("keep", False)),
                    content_type=str(entry.get("type", "unknown")),
                )
            )
        except Exception:
            continue
    return results


def _load_filtered_map(filtered_json: Path | None, filtered_dir: Path | None) -> Dict[str, dict]:
    mapping: Dict[str, dict] = {}
    if filtered_dir:
        for path in sorted(filtered_dir.glob("*_filtered.json")):
            page = _infer_page_from_filename(path)
            try:
                mapping[page] = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"WARNING: skipped {path}: {exc}")
    elif filtered_json:
        page = _infer_page_from_filename(filtered_json)
        mapping[page] = json.loads(Path(filtered_json).read_text(encoding="utf-8"))
    return mapping


def _infer_page_from_filename(path: Path) -> str:
    match = re.match(r"(\d+)_", path.name)
    if match:
        return match.group(1)
    return path.stem.split("_")[0]


def _parse_crop_name(name: str) -> Tuple[str, int]:
    pattern = r"^(?P<page>\d+)_final_(?P<id>\d+)\.[^\.]+$"
    match = re.match(pattern, name)
    if not match:
        raise ValueError(f"Unable to parse crop filename: {name}")
    return match.group("page"), int(match.group("id"))


def _bbox_area_xyxy(box: Sequence[int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _intersect_area_xyxy(a: Sequence[int], b: Sequence[int]) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def _overlap_over_smaller(a: Sequence[int], b: Sequence[int]) -> float:
    denom = min(_bbox_area_xyxy(a), _bbox_area_xyxy(b))
    if denom <= 0:
        return 0.0
    return _intersect_area_xyxy(a, b) / denom


def post_clean(
    judge_json: Path,
    crops_dir: Path,
    output_dir: Path,
    filtered_json: Path | None = None,
    filtered_dir: Path | None = None,
    delete_supersets: bool = True,
) -> Dict[str, object]:
    """Clean judged crops by removing duplicates and copying valid crops.

    Deduplication rules (when two crops have high overlap and same content type):
    - type == "text": keep the smaller one (delete the larger one)
    - type in {"image", "table"}: keep the larger one (delete the smaller one)
    - Different or unknown types: do not delete (conservative handling)
    """

    judge_records = json.loads(judge_json.read_text(encoding="utf-8"))
    kept_names = [rec["image"] for rec in judge_records if rec.get("keep")]
    type_map = {rec["image"]: rec.get("type", "unknown") for rec in judge_records}

    filtered_map = _load_filtered_map(filtered_json, filtered_dir)
    if not filtered_map:
        raise FileNotFoundError("No valid *_filtered.json provided.")

    output_dir.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    for name in kept_names:
        src = crops_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)
            copied.append(name)
        else:
            print(f"WARNING: missing crop file: {src}")

    # Collect all candidates per page (with bbox/area/type)
    per_page: Dict[str, List[dict]] = {}
    for name in copied:
        try:
            page, seg_id = _parse_crop_name(name)
        except ValueError as exc:
            print(f"⚠️ {exc}")
            continue
        masks = filtered_map.get(page, {}).get("masks", [])
        bbox = None
        for mask in masks:
            if int(mask.get("id", -1)) == seg_id:
                bbox = mask.get("bbox_xyxy")
                if bbox is None and mask.get("bbox_xywh"):
                    x, y, w, h = mask["bbox_xywh"]
                    bbox = [x, y, x + w, y + h]
                break
        if not bbox:
            print(f"WARNING: could not find bbox for {name}")
            continue
        per_page.setdefault(page, []).append(
            {
                "image": name,
                "page": page,
                "id": seg_id,
                "bbox_xyxy": bbox,
                "area": _bbox_area_xyxy(bbox),
                "type": (type_map.get(name, "unknown") or "unknown").lower(),
            }
        )

    survivors: List[dict] = []
    for items in per_page.values():
        to_drop: set[str] = set()

        # === Updated overlap-based deduplication logic ===
        for i in range(len(items)):
            a = items[i]
            for j in range(i + 1, len(items)):
                b = items[j]

                # Skip if already marked for deletion
                if a["image"] in to_drop or b["image"] in to_drop:
                    continue

                overlap = _overlap_over_smaller(a["bbox_xyxy"], b["bbox_xyxy"])
                if overlap >= 0.90 and delete_supersets:
                    ta, tb = a["type"], b["type"]

                    # Only handle if types are the same and known
                    if ta == tb and ta in {"text", "image", "table"}:
                        if ta == "text":
                            # Text: keep smaller → delete larger
                            drop = a if a["area"] >= b["area"] else b
                            to_drop.add(drop["image"])
                        else:
                            # Image/Table: keep larger → delete smaller
                            drop = a if a["area"] <= b["area"] else b
                            to_drop.add(drop["image"])
                    else:
                        # Different or unknown types: skip deletion
                        continue

        for item in items:
            if item["image"] not in to_drop:
                survivors.append(item)
            else:
                try:
                    (output_dir / item["image"]).unlink(missing_ok=True)
                except Exception:
                    pass

    summary = {
        "kept_stage1": len(copied),
        "kept_after_clean": len(survivors),
        "output_dir": str(output_dir),
        "items": survivors,
    }
    with (output_dir / "clean_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(
        f"SUCCESS: cleaned {len(copied)} candidate crop(s) down to {len(survivors)}. Output directory: {output_dir}"
    )
    return summary
