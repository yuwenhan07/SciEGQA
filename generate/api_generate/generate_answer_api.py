#!/usr/bin/env python3

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path

from openai import OpenAI
from transformers.utils.versions import require_version


require_version("openai>=1.5.0", "To fix: pip install openai>=1.5.0")
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_MODEL = "gpt-4.1-mini"


def to_data_url(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"

    image_bytes = Path(image_path).read_bytes()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{image_base64}"


def build_multimodal_content(text: str, image_paths: list[str], max_images: int | None) -> list[dict]:
    image_token = "<image>"
    image_count = text.count(image_token)
    usable_images = image_paths[:image_count]
    if max_images is not None:
        usable_images = usable_images[:max_images]
        image_count = min(image_count, max_images)

    parts = text.split(image_token)
    content: list[dict] = []

    for index, part in enumerate(parts):
        if part:
            content.append({"type": "text", "text": part})

        if index < image_count:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": to_data_url(usable_images[index])},
                }
            )

    return content


def extract_prompt_and_label(sample: dict) -> tuple[str, str]:
    messages = sample.get("messages", [])
    user_message = next(msg for msg in messages if msg.get("role") == "user")
    assistant_message = next(msg for msg in messages if msg.get("role") == "assistant")
    return user_message["content"], assistant_message["content"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SciEGQA inference through the official OpenAI API."
    )
    parser.add_argument("--dataset-jsonl", required=True, help="Path to the dataset jsonl file.")
    parser.add_argument("--output", required=True, help="Path to the output jsonl file.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name.")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="OpenAI API base URL.",
    )
    parser.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help="Environment variable name that stores the API key.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="0-based start sample index.")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to send.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for images per prompt.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Optional top-p sampling value.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output jsonl: skip successful records and retry failed ones.",
    )
    return parser.parse_args()


def load_existing_records(output_path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    if not output_path.exists():
        return records

    with output_path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample_idx = record.get("sample_idx")
            if isinstance(sample_idx, int):
                records[sample_idx] = record

    return records


def rewrite_existing_records(output_path: Path, records: dict[int, dict]) -> None:
    with output_path.open("w", encoding="utf-8", buffering=1) as writer:
        for sample_idx in sorted(records):
            writer.write(json.dumps(records[sample_idx], ensure_ascii=False) + "\n")
            writer.flush()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"Environment variable {args.api_key_env} is not set. "
            f"Export your OpenAI API key before running this script."
        )

    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url,
    )

    dataset_path = Path(args.dataset_jsonl)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_records: dict[int, dict] = {}
    if args.resume:
        existing_records = load_existing_records(output_path)
        if existing_records:
            rewrite_existing_records(output_path, existing_records)

    with dataset_path.open("r", encoding="utf-8") as reader:
        rows = [json.loads(line) for line in reader]

    end_index = len(rows)
    if args.max_samples is not None:
        end_index = min(end_index, args.start_index + args.max_samples)

    write_mode = "a" if args.resume and output_path.exists() else "w"
    with output_path.open(write_mode, encoding="utf-8", buffering=1) as writer:
        for sample_idx in range(args.start_index, end_index):
            existing_record = existing_records.get(sample_idx)
            if existing_record is not None and existing_record.get("error") is None:
                print(
                    f"[{sample_idx}] bbox_docvqa_id={existing_record.get('bbox_docvqa_id')} "
                    f"resume_skip=True"
                )
                continue

            sample = rows[sample_idx]
            prompt_text, label = extract_prompt_and_label(sample)
            images = sample.get("images", [])
            request_images = images if args.max_images is None else images[: args.max_images]
            content = build_multimodal_content(prompt_text, request_images, args.max_images)

            record = {
                "sample_idx": sample_idx,
                "bbox_docvqa_id": sample.get("bbox_docvqa_id"),
                "num_images": len(images),
                "num_images_sent": len(request_images),
                "query": sample.get("query"),
                "label": label,
            }

            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": content}],
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                record["predict"] = response.choices[0].message.content
                record["error"] = None
            except Exception as exc:
                record["predict"] = None
                record["error"] = str(exc)

            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            writer.flush()
            print(
                f"[{sample_idx}] bbox_docvqa_id={record['bbox_docvqa_id']} "
                f"images={record['num_images_sent']}/{record['num_images']} "
                f"error={record['error'] is not None}"
            )


if __name__ == "__main__":
    main()
