#!/usr/bin/env python3

import argparse
import os
import json
import ast
import struct
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import re

from PIL import Image, UnidentifiedImageError

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_QA_PROMPT_PREFIX = "Answer the question using only the document image(s). Return only the final answer with no explanation."
MIN_CROP_EDGE = 28
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
DEFAULT_SOURCE_JSONL = "data/SciEGQA/pred_bbox/pred_bbox_demo.jsonl"
DEFAULT_PREDICTION_JSONL = "outputs/SciEGQA_pred_bbox/qwen3-vl-8b/generated_predictions.jsonl"
DEFAULT_DATASET_DIR = "data/SciEGQA/pred_crop"
DEFAULT_DATASET_NAME = "SciEGQA_pred_crop"


def read_chunks(data):
    offset = 8
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        crc = data[offset + 8 + length : offset + 12 + length]
        yield chunk_type, chunk_data, crc
        offset += 12 + length


def paeth_predictor(a, b, c):
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def load_png_rgb(path):
    data = path.read_bytes()
    if data[:8] != PNG_SIGNATURE:
        raise ValueError(f"Not a PNG file: {path}")

    width = height = None
    bit_depth = color_type = interlace = None
    idat_parts = []

    for chunk_type, chunk_data, _crc in read_chunks(data):
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _comp, _flt, interlace = struct.unpack(">IIBBBBB", chunk_data)
        elif chunk_type == b"IDAT":
            idat_parts.append(chunk_data)
        elif chunk_type == b"IEND":
            break

    if width is None:
        raise ValueError(f"Missing IHDR: {path}")
    if bit_depth != 8 or color_type != 2 or interlace != 0:
        raise ValueError(
            f"Unsupported PNG format for {path}: bit_depth={bit_depth}, "
            f"color_type={color_type}, interlace={interlace}"
        )

    bytes_per_pixel = 3
    stride = width * bytes_per_pixel
    raw = zlib.decompress(b"".join(idat_parts))
    expected = height * (1 + stride)
    if len(raw) != expected:
        raise ValueError(f"Unexpected decompressed size for {path}: {len(raw)} != {expected}")

    rows = []
    prev_row = bytearray(stride)
    offset = 0

    for _ in range(height):
        filter_type = raw[offset]
        offset += 1
        filtered = bytearray(raw[offset : offset + stride])
        offset += stride
        row = bytearray(stride)

        if filter_type == 0:
            row[:] = filtered
        elif filter_type == 1:
            for i in range(stride):
                left = row[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                row[i] = (filtered[i] + left) & 0xFF
        elif filter_type == 2:
            for i in range(stride):
                row[i] = (filtered[i] + prev_row[i]) & 0xFF
        elif filter_type == 3:
            for i in range(stride):
                left = row[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                up = prev_row[i]
                row[i] = (filtered[i] + ((left + up) // 2)) & 0xFF
        elif filter_type == 4:
            for i in range(stride):
                left = row[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                up = prev_row[i]
                up_left = prev_row[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                row[i] = (filtered[i] + paeth_predictor(left, up, up_left)) & 0xFF
        else:
            raise ValueError(f"Unsupported PNG filter {filter_type} in {path}")

        rows.append(bytes(row))
        prev_row = row

    return width, height, rows


def png_chunk(chunk_type, chunk_data):
    return (
        struct.pack(">I", len(chunk_data))
        + chunk_type
        + chunk_data
        + struct.pack(">I", zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF)
    )


def save_png_rgb(path, width, height, rows):
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + row for row in rows)
    compressed = zlib.compress(raw, level=9)
    png = PNG_SIGNATURE + png_chunk(b"IHDR", ihdr) + png_chunk(b"IDAT", compressed) + png_chunk(b"IEND", b"")
    path.write_bytes(png)


def is_valid_image(path, min_crop_edge):
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
        return width >= min_crop_edge and height >= min_crop_edge
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def crop_rows(rows, bbox):
    x1, y1, x2, y2 = bbox
    start = x1 * 3
    end = x2 * 3
    return [row[start:end] for row in rows[y1:y2]]


def clamp_bbox(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(round(x1)), width))
    x2 = max(0, min(int(round(x2)), width))
    y1 = max(0, min(int(round(y1)), height))
    y2 = max(0, min(int(round(y2)), height))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return [x1, y1, x2, y2]


def build_prompt(query, num_images, prompt_prefix):
    image_tokens = ["<image>" for _ in range(num_images)]
    parts = []
    if image_tokens:
        parts.extend(image_tokens)
    if prompt_prefix:
        parts.append(prompt_prefix.strip())
    parts.append(query.strip())
    return "\n".join(parts)


def strip_code_fence(text):
    if text is None:
        return ""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def extract_json_span(text):
    first_positions = [idx for idx in (text.find("["), text.find("{")) if idx != -1]
    start = min(first_positions) if first_positions else -1
    end = max(text.rfind("]"), text.rfind("}"))
    if start == -1 or end == -1 or end < start:
        return None
    return text[start : end + 1]


def parse_numeric_box(box):
    if isinstance(box, dict):
        for key in ("bbox_2d", "bbox", "box", "coordinates"):
            if key in box:
                return parse_numeric_box(box[key])
        return None

    if isinstance(box, (list, tuple)) and len(box) == 4:
        values = []
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


def is_box_list(value):
    return isinstance(value, list) and all(parse_numeric_box(item) is not None for item in value)


def is_page_list(value):
    return isinstance(value, list) and all(is_box_list(item) for item in value)


def split_flat_number_sequence(value):
    if not isinstance(value, (list, tuple)):
        return None

    numbers = []
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


def normalize_pages_from_nested_lists(data):
    if not isinstance(data, list) or is_box_list(data):
        return None

    pages = []
    for item in data:
        box = parse_numeric_box(item)
        if box is not None:
            pages.append([box])
            continue

        split_boxes = split_flat_number_sequence(item)
        if split_boxes is not None:
            pages.append(split_boxes)
            continue

        if is_box_list(item):
            pages.append([parse_numeric_box(box) for box in item if parse_numeric_box(box) is not None])
            continue

        return None

    return pages


def parse_structured_text(text):
    if text is None:
        return []
    cleaned = strip_code_fence(text)
    if not cleaned:
        return []

    json_span = extract_json_span(cleaned)
    candidates = [cleaned]
    if json_span is not None and json_span != cleaned:
        candidates.insert(0, json_span)

    for candidate in candidates:
        for loader in (json.loads, ast.literal_eval):
            try:
                return loader(candidate)
            except Exception:
                continue

    box_strings = re.findall(r"\[[^\[\]]+\]", cleaned)
    boxes = []
    for box_str in box_strings:
        box = parse_numeric_box(box_str)
        if box is not None:
            boxes.append(box)
    if boxes:
        return boxes

    return None


def parse_predicted_boxes(text, num_source_pages=1):
    payload = parse_structured_text(text)
    if payload is None:
        return None
    if parse_numeric_box(payload) is not None:
        return [[parse_numeric_box(payload)]]
    if is_page_list(payload):
        return [
            [parse_numeric_box(item) for item in page if parse_numeric_box(item) is not None]
            for page in payload
        ]
    if is_box_list(payload):
        flat_boxes = [parse_numeric_box(item) for item in payload if parse_numeric_box(item) is not None]
        if num_source_pages > 1 and len(flat_boxes) == num_source_pages:
            return [[box] for box in flat_boxes]
        return [flat_boxes]

    split_boxes = split_flat_number_sequence(payload)
    if split_boxes is not None:
        if num_source_pages > 1 and len(split_boxes) == num_source_pages:
            return [[box] for box in split_boxes]
        return [split_boxes]

    nested_pages = normalize_pages_from_nested_lists(payload)
    if nested_pages is not None:
        return nested_pages

    return None


def normalize_rel_box(box):
    normalized = []
    for value in box:
        value = float(value)
        if value <= 1.0:
            value *= 1000.0
        normalized.append(max(0.0, min(1000.0, value)))
    return normalized


def rel_to_abs_box(rel_box, width, height):
    x1, y1, x2, y2 = normalize_rel_box(rel_box)
    return clamp_bbox(
        [
            x1 / 1000.0 * width,
            y1 / 1000.0 * height,
            x2 / 1000.0 * width,
            y2 / 1000.0 * height,
        ],
        width,
        height,
    )


def build_type_labels(record, page_index, num_boxes):
    subimg_type = record.get("subimg_type", [])
    page_types = subimg_type[page_index] if page_index < len(subimg_type) else []
    labels = []
    for idx in range(num_boxes):
        if idx < len(page_types):
            value = page_types[idx]
            if isinstance(value, str):
                labels.append(value)
            else:
                labels.append("region")
        else:
            labels.append("region")
    return labels


def build_crops_from_prediction(record, predicted_pages, crop_dir, sample_index, min_crop_edge):
    source_images = record.get("images", [])
    all_rel_bbox = []
    all_abs_bbox = []
    crop_paths = []

    for page_index, image_path_str in enumerate(source_images):
        image_path = Path(image_path_str)
        if not image_path.is_file():
            raise FileNotFoundError(f"Source image not found: {image_path}")

        predicted_boxes = predicted_pages[page_index] if page_index < len(predicted_pages) else []
        if not predicted_boxes:
            continue

        width, height, rows = load_png_rgb(image_path)
        abs_page_boxes = []
        rel_page_boxes = []
        region_types = build_type_labels(record, page_index, len(predicted_boxes))

        for region_index, rel_box in enumerate(predicted_boxes, start=1):
            abs_box = rel_to_abs_box(rel_box, width, height)
            rel_box = normalize_rel_box(rel_box)
            crop_width = abs_box[2] - abs_box[0]
            crop_height = abs_box[3] - abs_box[1]
            if crop_width < min_crop_edge or crop_height < min_crop_edge:
                continue
            crop = crop_rows(rows, abs_box)
            page_id = record.get("evidence_page", [])
            page_name = page_id[page_index] if page_index < len(page_id) else page_index + 1
            region_type = region_types[region_index - 1]
            crop_name = (
                f"sample_{sample_index:05d}_{record.get('category', 'unknown')}_{record.get('doc_name', 'unknown')}"
                f"_p{page_name}_r{region_index}_{region_type}.png"
            )
            crop_path = crop_dir / crop_name
            save_png_rgb(crop_path, crop_width, crop_height, crop)
            if not is_valid_image(crop_path, min_crop_edge):
                try:
                    crop_path.unlink(missing_ok=True)
                except OSError:
                    pass
                continue

            crop_paths.append(str(crop_path))
            abs_page_boxes.append(abs_box)
            rel_page_boxes.append(rel_box)

        if abs_page_boxes:
            all_abs_bbox.append(abs_page_boxes)
            all_rel_bbox.append(rel_page_boxes)

    return crop_paths, all_abs_bbox, all_rel_bbox


def process_one_sample(task):
    sample_index, source_record, prediction_record, crop_dir_str, prompt_prefix, min_crop_edge = task
    crop_dir = Path(crop_dir_str)
    predicted_pages = parse_predicted_boxes(
        prediction_record.get("predict", ""),
        num_source_pages=len(source_record.get("images", [])),
    )
    if predicted_pages is None:
        return {"status": "skip", "reason": "invalid_predicted_boxes", "sample_index": sample_index}

    try:
        crop_paths, abs_bbox, rel_bbox = build_crops_from_prediction(
            source_record, predicted_pages, crop_dir, sample_index, min_crop_edge
        )
    except FileNotFoundError as exc:
        return {"status": "skip", "reason": str(exc), "sample_index": sample_index}

    if not crop_paths:
        return {"status": "skip", "reason": "no_usable_crops", "sample_index": sample_index}

    sample = {
        "messages": [
            {
                "role": "user",
                "content": build_prompt(source_record["query"], len(crop_paths), prompt_prefix),
            },
            {"role": "assistant", "content": source_record["answer"]},
        ],
        "images": crop_paths,
        "sciegqa_id": source_record.get("sciegqa_id", source_record.get("bbox_docvqa_id", sample_index)),
        "bbox_docvqa_id": source_record.get("bbox_docvqa_id", sample_index),
        "query": source_record["query"],
        "answer": source_record["answer"],
        "category": source_record.get("category"),
        "doc_name": source_record.get("doc_name"),
        "evidence_page": source_record.get("evidence_page"),
        "bbox": abs_bbox,
        "rel_bbox": rel_bbox,
        "subimg_type": source_record.get("subimg_type"),
        "image_mode": "pred_crop",
        "source_bbox_target": source_record.get("bbox_target"),
        "predicted_bbox_text": prediction_record.get("predict"),
    }
    return {"status": "ok", "sample_index": sample_index, "sample": sample}


def convert_dataset(args):
    source_jsonl = Path(args.source_jsonl).resolve()
    prediction_jsonl = Path(args.prediction_jsonl).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = dataset_dir / f"{args.dataset_name}.jsonl"
    crop_dir = dataset_dir / "images"
    crop_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0

    with source_jsonl.open("r", encoding="utf-8") as source_file:
        source_records = [json.loads(line) for line in source_file]
    with prediction_jsonl.open("r", encoding="utf-8") as prediction_file:
        prediction_records = [json.loads(line) for line in prediction_file]

    if len(source_records) != len(prediction_records):
        print(
            f"Warning: source records ({len(source_records)}) and prediction records ({len(prediction_records)}) differ. "
            f"Will only process the first {min(len(source_records), len(prediction_records))} line-aligned samples."
        )

    total = min(len(source_records), len(prediction_records))
    if args.max_samples is not None:
        total = min(total, args.max_samples)

    tasks = [
        (
            sample_index + 1,
            source_records[sample_index],
            prediction_records[sample_index],
            str(crop_dir),
            args.prompt_prefix,
            args.min_crop_edge,
        )
        for sample_index in range(total)
    ]

    if args.num_workers == 1:
        results = [process_one_sample(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            results = list(executor.map(process_one_sample, tasks, chunksize=args.chunksize))

    results.sort(key=lambda item: item["sample_index"])

    with output_jsonl.open("w", encoding="utf-8") as outfile:
        for result in results:
            if result["status"] != "ok":
                skipped += 1
                if args.verbose:
                    print(f"Skip sample {result['sample_index']}: {result['reason']}")
                continue
            outfile.write(json.dumps(result["sample"], ensure_ascii=False) + "\n")
            converted += 1

    dataset_info_path = dataset_dir / "dataset_info.json"
    if dataset_info_path.is_file():
        with dataset_info_path.open("r", encoding="utf-8") as file:
            dataset_info = json.load(file)
    else:
        dataset_info = {}

    dataset_info[args.dataset_name] = {
        "file_name": output_jsonl.name,
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }

    with dataset_info_path.open("w", encoding="utf-8") as file:
        json.dump(dataset_info, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"Converted {converted} SciEGQA sample(s) to {output_jsonl}")
    if skipped:
        print(f"Skipped {skipped} sample(s) without usable predicted boxes")
    print(f"Dataset config written to {dataset_info_path}")
    print(f"Cropped images saved to {crop_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a new crop-style SciEGQA dataset from predicted bounding boxes."
    )
    parser.add_argument(
        "--source-jsonl",
        default=DEFAULT_SOURCE_JSONL,
        help="Source SciEGQA bbox dataset jsonl. Must be line-aligned with prediction jsonl.",
    )
    parser.add_argument(
        "--prediction-jsonl",
        default=DEFAULT_PREDICTION_JSONL,
        help="Prediction jsonl generated from the SciEGQA bbox task.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help="Output directory for the rebuilt SciEGQA pred-crop dataset.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Dataset name registered in dataset_info.json.",
    )
    parser.add_argument(
        "--prompt-prefix",
        default=DEFAULT_QA_PROMPT_PREFIX,
        help="Instruction inserted before the question text.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional sample cap for debugging.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Number of CPU worker processes used for parsing and cropping.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=8,
        help="Chunksize passed to ProcessPoolExecutor.map.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print skipped sample reasons.",
    )
    parser.add_argument(
        "--min-crop-edge",
        type=int,
        default=MIN_CROP_EDGE,
        help="Skip crops whose width or height is smaller than this threshold.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    convert_dataset(parse_args())
