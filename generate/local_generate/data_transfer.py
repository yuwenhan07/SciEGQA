#!/usr/bin/env python3

import argparse
import json
import struct
import zlib
from pathlib import Path

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_QA_PROMPT_PREFIX = "Answer the question using only the document image(s). Return only the final answer with no explanation."
DEFAULT_BBOX_PROMPT_PREFIX = (
    "Locate the region or regions needed to answer the question in each document image. "
    "Return only JSON grouped by page in this format: [[[x1, y1, x2, y2], ...], ...]. "
    "Each outer item corresponds to one input image/page in order. "
    "Each inner item is one bounding box with normalized coordinates between 0 and 1000."
)
DEFAULT_INPUT_JSONL = "data/SciEGQA-Bench/SciEGQA-Bench.jsonl"
DEFAULT_IMAGE_DIR = "data/SciEGQA-Bench/Images"


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


def crop_rows(rows, bbox):
    x1, y1, x2, y2 = bbox
    start = x1 * 3
    end = x2 * 3
    return [row[start:end] for row in rows[y1:y2]]


def normalize_page_boxes(page_boxes):
    if not page_boxes:
        return []
    if isinstance(page_boxes[0], (int, float)):
        return [page_boxes]
    return page_boxes


def normalize_page_types(page_types, size):
    if not page_types:
        return ["unknown"] * size
    if isinstance(page_types, str):
        return [page_types]
    return page_types


def clamp_bbox(box, width, height):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2



def count_prompt_images(record, mode, images):
    if mode in {"page", "bbox"}:
        return len(record.get("evidence_page", []) or images)
    if mode == "document":
        return len(images)

    region_count = 0
    for page_boxes in record.get("bbox", []):
        for box in normalize_page_boxes(page_boxes):
            if len(box) == 4:
                region_count += 1

    return region_count or len(images)


def build_prompt(query, num_images, prompt_prefix):
    image_tokens = ["<image>" for _ in range(num_images)]
    parts = []
    if image_tokens:
        parts.extend(image_tokens)
    if prompt_prefix:
        parts.append(prompt_prefix.strip())
    parts.append(query.strip())
    return "\n".join(parts)


def normalize_rel_page_boxes(page_boxes):
    boxes = normalize_page_boxes(page_boxes)
    normalized = []
    for box in boxes:
        if len(box) != 4:
            continue
        normalized.append([float(v) for v in box])
    return normalized


def build_bbox_target(record):
    rel_bbox_pages = record.get("rel_bbox", [])
    grouped = []

    for page_boxes in rel_bbox_pages:
        normalized_boxes = normalize_rel_page_boxes(page_boxes)
        if not normalized_boxes:
            continue
        grouped.append(normalized_boxes)

    if not grouped:
        return None

    return json.dumps(grouped, ensure_ascii=False, separators=(",", ":"))


def get_prompt_prefix(args):
    if args.prompt_prefix is not None:
        return args.prompt_prefix
    if args.mode == "bbox":
        return DEFAULT_BBOX_PROMPT_PREFIX
    return DEFAULT_QA_PROMPT_PREFIX


def resolve_dataset_dir(args) -> Path:
    if args.dataset_dir is not None:
        return Path(args.dataset_dir).resolve()
    return Path("data") / "SciEGQA" / args.mode


def resolve_dataset_name(args) -> str:
    if args.dataset_name is not None:
        return args.dataset_name
    return f"SciEGQA_{args.mode}"


def build_page_images(record, benchmark_dir):
    category = record["category"]
    doc_name = record["doc_name"]
    image_paths = []
    for page in record.get("evidence_page", []):
        image_path = benchmark_dir / category / doc_name / f"{doc_name}_{page}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Page image not found: {image_path}")
        image_paths.append(str(image_path))
    return image_paths


def extract_page_number(image_path):
    stem = image_path.stem
    prefix = f"{image_path.parent.name}_"
    if stem.startswith(prefix):
        suffix = stem[len(prefix) :]
        if suffix.isdigit():
            return int(suffix)
    return stem


def build_document_images(record, benchmark_dir):
    category = record["category"]
    doc_name = record["doc_name"]
    doc_dir = benchmark_dir / category / doc_name
    if not doc_dir.is_dir():
        raise FileNotFoundError(f"Document image directory not found: {doc_dir}")

    image_paths = sorted(doc_dir.glob(f"{doc_name}_*.png"), key=extract_page_number)
    if not image_paths:
        raise FileNotFoundError(f"No page images found in: {doc_dir}")
    return [str(image_path) for image_path in image_paths]


def build_crop_images(record, benchmark_dir, crop_dir, sample_index):
    category = record["category"]
    doc_name = record["doc_name"]
    pages = record.get("evidence_page", [])
    bbox_pages = record.get("bbox", [])
    type_pages = record.get("subimg_type", [])
    crop_paths = []

    for page_idx, page in enumerate(pages):
        image_path = benchmark_dir / category / doc_name / f"{doc_name}_{page}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Page image not found: {image_path}")

        width, height, rows = load_png_rgb(image_path)
        page_boxes = normalize_page_boxes(bbox_pages[page_idx] if page_idx < len(bbox_pages) else [])
        page_types = normalize_page_types(type_pages[page_idx] if page_idx < len(type_pages) else [], len(page_boxes))

        for region_idx, box in enumerate(page_boxes, start=1):
            if len(box) != 4:
                continue

            bbox = clamp_bbox(box, width, height)
            region_type = page_types[region_idx - 1] if region_idx - 1 < len(page_types) else "unknown"
            crop_name = f"sample_{sample_index:05d}_{category}_{doc_name}_p{page}_r{region_idx}_{region_type}.png"
            crop_path = crop_dir / crop_name
            crop = crop_rows(rows, bbox)
            save_png_rgb(crop_path, bbox[2] - bbox[0], bbox[3] - bbox[1], crop)
            crop_paths.append(str(crop_path))

    return crop_paths


def convert_dataset(args):
    dataset_dir = resolve_dataset_dir(args)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir = Path(args.benchmark_dir).resolve()
    input_jsonl = Path(args.input_jsonl).resolve()
    dataset_name = resolve_dataset_name(args)
    output_jsonl = dataset_dir / f"{dataset_name}.jsonl"
    crop_dir = dataset_dir / "images"
    if args.mode == "crop":
        crop_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0

    with input_jsonl.open("r", encoding="utf-8") as infile, output_jsonl.open("w", encoding="utf-8") as outfile:
        for sample_index, line in enumerate(infile, start=1):
            if args.max_samples is not None and converted >= args.max_samples:
                break

            record = json.loads(line)
            if args.mode in {"page", "bbox"}:
                images = build_page_images(record, benchmark_dir)
            elif args.mode == "document":
                images = build_document_images(record, benchmark_dir)
            else:
                images = build_crop_images(record, benchmark_dir, crop_dir, sample_index)

            if not images:
                skipped += 1
                continue

            num_prompt_images = count_prompt_images(record, args.mode, images)
            prompt_prefix = get_prompt_prefix(args)
            user_content = build_prompt(record["query"], num_prompt_images, prompt_prefix)
            assistant_content = record["answer"]
            if args.mode == "bbox":
                assistant_content = build_bbox_target(record)
                if assistant_content is None:
                    skipped += 1
                    continue

            sample = {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
                "images": images,
                "sciegqa_id": sample_index,
                "bbox_docvqa_id": sample_index,
                "query": record["query"],
                "answer": record["answer"],
                "category": record.get("category"),
                "doc_name": record.get("doc_name"),
                "evidence_page": record.get("evidence_page"),
                "bbox": record.get("bbox"),
                "rel_bbox": record.get("rel_bbox"),
                "subimg_type": record.get("subimg_type"),
                "image_mode": args.mode,
            }
            if args.mode == "bbox":
                sample["bbox_target"] = assistant_content
            outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")
            converted += 1

    dataset_info_path = dataset_dir / "dataset_info.json"
    if dataset_info_path.is_file():
        with dataset_info_path.open("r", encoding="utf-8") as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}

    dataset_info[dataset_name] = {
        "file_name": output_jsonl.name,
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "images": "images",
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }

    with dataset_info_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Converted {converted} SciEGQA sample(s) to {output_jsonl}")
    if skipped:
        print(f"Skipped {skipped} sample(s) without usable images")
    print(f"Dataset config written to {dataset_info_path}")
    if args.mode == "crop":
        print(f"Cropped images saved to {crop_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert SciEGQA annotations into a LlamaFactory multimodal dataset."
    )
    parser.add_argument(
        "--input-jsonl",
        default=DEFAULT_INPUT_JSONL,
        help="Path to the source SciEGQA annotation jsonl file.",
    )
    parser.add_argument(
        "--benchmark-dir",
        default=DEFAULT_IMAGE_DIR,
        help="Root directory that contains category/doc/page PNG files for SciEGQA.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Output directory. Defaults to data/SciEGQA/<mode>.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset name to register in dataset_info.json. Defaults to SciEGQA_<mode>.",
    )
    parser.add_argument(
        "--mode",
        choices=["page", "crop", "bbox", "document"],
        default="page",
        help="Use evidence pages for QA, all document pages for QA, crop bbox regions, or generate normalized bbox targets from page images.",
    )
    parser.add_argument(
        "--prompt-prefix",
        default=None,
        help="Optional instruction inserted before the question text. Defaults depend on --mode.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of converted samples for debugging.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    convert_dataset(parse_args())
