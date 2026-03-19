#!/usr/bin/env python3
import argparse
import ast
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


@dataclass
class ParsedPrediction:
    pages: list[list[list[float]]]
    grouped_by_page: bool
    parse_mode: str


def strip_code_fence(text: str) -> str:
    return FENCE_RE.sub("", text).strip()


def parse_numeric_box(box: Any) -> list[float] | None:
    if isinstance(box, dict):
        for key in ("bbox_2d", "bbox", "box", "coordinates"):
            if key in box:
                return parse_numeric_box(box[key])
        return None

    if isinstance(box, (list, tuple)) and len(box) == 4:
        values: list[float] = []
        for item in box:
            if isinstance(item, (int, float)):
                values.append(float(item))
            elif isinstance(item, str):
                match = NUMBER_RE.fullmatch(item.strip())
                if not match:
                    return None
                values.append(float(match.group(0)))
            else:
                return None
        return values

    if isinstance(box, str):
        numbers = NUMBER_RE.findall(box)
        if len(numbers) == 4:
            return [float(num) for num in numbers]

    return None


def split_flat_number_sequence(value: Any) -> list[list[float]] | None:
    if not isinstance(value, (list, tuple)):
        return None

    numbers: list[float] = []
    for item in value:
        if isinstance(item, (int, float)):
            numbers.append(float(item))
        elif isinstance(item, str):
            match = NUMBER_RE.fullmatch(item.strip())
            if not match:
                return None
            numbers.append(float(match.group(0)))
        else:
            return None

    if not numbers or len(numbers) % 4 != 0:
        return None

    return [numbers[idx : idx + 4] for idx in range(0, len(numbers), 4)]


def recover_mixed_box_sequence(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list):
        return None

    saw_number = False
    saw_box = False
    numbers: list[float] = []

    for item in value:
        box = parse_numeric_box(item)
        if box is not None:
            saw_box = True
            numbers.extend(box)
            continue

        if isinstance(item, (int, float)):
            saw_number = True
            numbers.append(float(item))
            continue

        if isinstance(item, str):
            match = NUMBER_RE.fullmatch(item.strip())
            if match:
                saw_number = True
                numbers.append(float(match.group(0)))
                continue

        return None

    if not saw_number or not saw_box or len(numbers) % 4 != 0:
        return None

    return [numbers[idx : idx + 4] for idx in range(0, len(numbers), 4)]


def is_box_list(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    return all(parse_numeric_box(item) is not None for item in value)


def is_page_list(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    return all(is_box_list(item) for item in value)


def normalize_pages_from_nested_lists(data: Any) -> list[list[list[float]]] | None:
    if not isinstance(data, list):
        return None
    if is_box_list(data):
        return None

    pages: list[list[list[float]]] = []
    for item in data:
        if parse_numeric_box(item) is not None:
            pages.append([parse_numeric_box(item)])  # type: ignore[list-item]
            continue

        split_boxes = split_flat_number_sequence(item)
        if split_boxes is not None:
            pages.append(split_boxes)
            continue

        if is_box_list(item):
            page_boxes = [parse_numeric_box(box) for box in item]
            pages.append([box for box in page_boxes if box is not None])  # type: ignore[list-item]
            continue

        return None

    return pages


def parse_structured_text(text: str) -> Any:
    cleaned = strip_code_fence(text)
    if not cleaned:
        return []

    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(cleaned)
        except Exception:
            continue

    # Last resort: recover every `[x1, y1, x2, y2]`-like fragment from free text.
    box_strings = re.findall(r"\[[^\[\]]+\]", cleaned)
    boxes = []
    for box_str in box_strings:
        box = parse_numeric_box(box_str)
        if box is not None:
            boxes.append(box)
    if boxes:
        return boxes

    raise ValueError(f"Unable to parse structured boxes from: {text[:200]!r}")


def normalize_label(raw_label: str) -> list[list[list[float]]]:
    data = parse_structured_text(raw_label)
    split_boxes = split_flat_number_sequence(data)
    if split_boxes is not None:
        return [split_boxes]
    mixed_boxes = recover_mixed_box_sequence(data)
    if mixed_boxes is not None:
        return [mixed_boxes]
    nested_pages = normalize_pages_from_nested_lists(data)
    if nested_pages is not None:
        return nested_pages

    if parse_numeric_box(data) is not None:
        return [[parse_numeric_box(data)]]  # type: ignore[list-item]
    if is_box_list(data):
        return [[parse_numeric_box(item) for item in data if parse_numeric_box(item) is not None]]  # type: ignore[list-item]
    if is_page_list(data):
        pages: list[list[list[float]]] = []
        for page in data:
            page_boxes = [parse_numeric_box(item) for item in page]
            pages.append([box for box in page_boxes if box is not None])  # type: ignore[list-item]
        return pages

    raise ValueError(f"Unsupported label structure: {data!r}")


def normalize_prediction(raw_predict: Any, num_label_pages: int, single_box_per_page: bool) -> ParsedPrediction:
    if raw_predict is None:
        return ParsedPrediction(pages=[[] for _ in range(num_label_pages)], grouped_by_page=True, parse_mode="empty")

    data = parse_structured_text(str(raw_predict))
    split_boxes = split_flat_number_sequence(data)
    if split_boxes is not None:
        if num_label_pages > 1 and single_box_per_page and len(split_boxes) == num_label_pages:
            return ParsedPrediction(
                pages=[[box] for box in split_boxes],
                grouped_by_page=True,
                parse_mode="promoted_from_number_sequence",
            )
        return ParsedPrediction(pages=[split_boxes], grouped_by_page=False, parse_mode="flat_number_sequence")
    mixed_boxes = recover_mixed_box_sequence(data)
    if mixed_boxes is not None:
        return ParsedPrediction(pages=[mixed_boxes], grouped_by_page=False, parse_mode="mixed_box_sequence")
    nested_pages = normalize_pages_from_nested_lists(data)
    if nested_pages is not None:
        return ParsedPrediction(pages=nested_pages, grouped_by_page=True, parse_mode="page_grouped_recovered")

    if parse_numeric_box(data) is not None:
        box = parse_numeric_box(data)
        return ParsedPrediction(pages=[[box]] if box is not None else [[]], grouped_by_page=False, parse_mode="single_box")

    if is_page_list(data):
        pages: list[list[list[float]]] = []
        for page in data:
            page_boxes = [parse_numeric_box(item) for item in page]
            pages.append([box for box in page_boxes if box is not None])  # type: ignore[list-item]
        return ParsedPrediction(pages=pages, grouped_by_page=True, parse_mode="page_grouped")

    if is_box_list(data):
        boxes = [parse_numeric_box(item) for item in data]
        flat_boxes = [box for box in boxes if box is not None]  # type: ignore[list-item]

        if num_label_pages > 1 and single_box_per_page and len(flat_boxes) == num_label_pages:
            return ParsedPrediction(
                pages=[[box] for box in flat_boxes],
                grouped_by_page=True,
                parse_mode="promoted_by_order",
            )

        return ParsedPrediction(pages=[flat_boxes], grouped_by_page=False, parse_mode="flat_boxes")

    raise ValueError(f"Unsupported prediction structure: {data!r}")


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def best_match_sum(gt_boxes: list[list[float]], pred_boxes: list[list[float]]) -> float:
    if not gt_boxes or not pred_boxes:
        return 0.0

    ious = tuple(tuple(bbox_iou(gt, pred) for pred in pred_boxes) for gt in gt_boxes)

    @lru_cache(maxsize=None)
    def dp(gt_idx: int, used_mask: int) -> float:
        if gt_idx >= len(gt_boxes):
            return 0.0

        best = dp(gt_idx + 1, used_mask)
        for pred_idx in range(len(pred_boxes)):
            if used_mask & (1 << pred_idx):
                continue
            best = max(best, ious[gt_idx][pred_idx] + dp(gt_idx + 1, used_mask | (1 << pred_idx)))
        return best

    return dp(0, 0)


def score_grouped_pages(gt_pages: list[list[list[float]]], pred_pages: list[list[list[float]]]) -> tuple[float, float]:
    matched_sum = 0.0
    page_scores: list[float] = []
    total_gt = sum(len(page) for page in gt_pages)
    total_pred = sum(len(page) for page in pred_pages)

    max_pages = max(len(gt_pages), len(pred_pages))
    for page_idx in range(max_pages):
        gt_boxes = gt_pages[page_idx] if page_idx < len(gt_pages) else []
        pred_boxes = pred_pages[page_idx] if page_idx < len(pred_pages) else []
        page_match = best_match_sum(gt_boxes, pred_boxes)
        matched_sum += page_match
        page_den = max(len(gt_boxes), len(pred_boxes), 1)
        page_scores.append(page_match / page_den)

    sample_den = max(total_gt, total_pred, 1)
    return matched_sum / sample_den, sum(page_scores) / len(page_scores)


def score_flat_fallback(gt_pages: list[list[list[float]]], pred_pages: list[list[list[float]]]) -> tuple[float, float]:
    flat_gt = [box for page in gt_pages for box in page]
    flat_pred = [box for page in pred_pages for box in page]
    matched_sum = best_match_sum(flat_gt, flat_pred)
    sample_den = max(len(flat_gt), len(flat_pred), 1)
    score = matched_sum / sample_den
    return score, score


def build_output_paths(input_path: Path, output_dir: Path | None) -> tuple[Path, Path]:
    stem = input_path.stem
    base_dir = output_dir if output_dir is not None else input_path.parent
    return base_dir / f"{stem}_iou_scored.jsonl", base_dir / f"{stem}_iou_metrics.json"


def summarize_threshold_hits(scores: list[float], thresholds: list[float]) -> dict[str, dict[str, float | int]]:
    total = len(scores)
    summary: dict[str, dict[str, float | int]] = {}
    for threshold in thresholds:
        hits = sum(1 for score in scores if score >= threshold)
        summary[f"iou@{threshold:.1f}"] = {
            "count": hits,
            "total": total,
            "rate": round(hits / total, 6) if total else 0.0,
        }
    return summary


def evaluate_file(input_path: Path, scored_path: Path, metrics_path: Path) -> dict[str, Any]:
    total = 0
    valid = 0
    sample_iou_sum = 0.0
    page_iou_sum = 0.0
    parse_failures = 0
    grouped_count = 0
    fallback_count = 0
    parse_mode_counts: dict[str, int] = {}
    sample_scores: list[float] = []
    page_scores: list[float] = []
    thresholds = [0.3, 0.5, 0.7]

    with input_path.open("r", encoding="utf-8") as fin, scored_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            row = json.loads(line)
            total += 1

            scored_row = dict(row)
            scored_row["line_no"] = line_no

            try:
                gt_pages = normalize_label(row["label"])
                parsed_pred = normalize_prediction(
                    row.get("predict"),
                    num_label_pages=len(gt_pages),
                    single_box_per_page=all(len(page) == 1 for page in gt_pages),
                )

                if parsed_pred.grouped_by_page:
                    sample_iou, page_iou = score_grouped_pages(gt_pages, parsed_pred.pages)
                    grouped_count += 1
                else:
                    sample_iou, page_iou = score_flat_fallback(gt_pages, parsed_pred.pages)
                    fallback_count += 1

                valid += 1
                sample_iou_sum += sample_iou
                page_iou_sum += page_iou
                sample_scores.append(sample_iou)
                page_scores.append(page_iou)
                parse_mode_counts[parsed_pred.parse_mode] = parse_mode_counts.get(parsed_pred.parse_mode, 0) + 1

                scored_row["parsed_predict_pages"] = parsed_pred.pages
                scored_row["parsed_label_pages"] = gt_pages
                scored_row["predict_parse_mode"] = parsed_pred.parse_mode
                scored_row["grouped_by_page"] = parsed_pred.grouped_by_page
                scored_row["sample_iou"] = round(sample_iou, 6)
                scored_row["page_iou"] = round(page_iou, 6)
                scored_row["error"] = None
            except Exception as exc:
                parse_failures += 1
                scored_row["sample_iou"] = 0.0
                scored_row["page_iou"] = 0.0
                sample_scores.append(0.0)
                page_scores.append(0.0)
                scored_row["predict_parse_mode"] = "parse_error"
                scored_row["grouped_by_page"] = False
                scored_row["error"] = str(exc)

            fout.write(json.dumps(scored_row, ensure_ascii=False) + "\n")

    metrics = {
        "input_file": str(input_path),
        "samples": total,
        "valid_samples": valid,
        "parse_failures": parse_failures,
        "mean_sample_iou": round(sample_iou_sum / total, 6) if total else 0.0,
        "mean_sample_iou_valid_only": round(sample_iou_sum / valid, 6) if valid else 0.0,
        "mean_page_iou": round(page_iou_sum / total, 6) if total else 0.0,
        "mean_page_iou_valid_only": round(page_iou_sum / valid, 6) if valid else 0.0,
        "grouped_prediction_samples": grouped_count,
        "flat_fallback_samples": fallback_count,
        "predict_parse_modes": parse_mode_counts,
        "sample_iou_thresholds": summarize_threshold_hits(sample_scores, thresholds),
        "page_iou_thresholds": summarize_threshold_hits(page_scores, thresholds),
        "scored_file": str(scored_path),
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate bounding box IoU from generated_predictions.jsonl.")
    parser.add_argument("input_file", help="Path to generated_predictions.jsonl")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory used to write scored jsonl and metrics json. Defaults to the input file directory.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    scored_path, metrics_path = build_output_paths(input_path, output_dir)
    metrics = evaluate_file(input_path, scored_path, metrics_path)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
