#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_MODEL = "gpt5.2"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_record_id(record: dict[str, Any], fallback: int) -> Any:
    for key in ("doc_id", "bbox_docvqa_id", "sciegqa_id", "sample_idx"):
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return fallback


def to_bool_or_none(text: str | None) -> bool | str | None:
    if text is None:
        return None

    value = text.strip().lower()
    match = re.search(r"\b(correct|incorrect|true|false)\b", value)
    if match:
        token = match.group(1)
        if token in ("true", "correct"):
            return True
        if token in ("false", "incorrect"):
            return False

    if "incorrect" in value or "false" in value:
        return False
    if "correct" in value or "true" in value:
        return True

    return text


def extract_question(prompt: Any) -> str:
    if prompt is None:
        return ""

    text = str(prompt).strip()
    prefix = (
        "user\n\n"
        "Answer the question using only the document image(s). "
        "Return only the final answer with no explanation.\n"
    )
    suffix = "\nassistant"

    if text.startswith(prefix):
        text = text[len(prefix):]
    if text.endswith(suffix):
        text = text[: -len(suffix)]

    return text.strip()


def build_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    question = record.get("query") or extract_question(record.get("prompt"))
    ground_truth = record.get("label") or record.get("lable") or record.get("answer")
    response = record.get("predict") or record.get("pred_answer")

    sys_prompt = (
        "Your task is to evaluate whether the model's response correctly answers the question, "
        "based on the provided reference answer.\n"
        "This is part of an automated evaluation process, so your result must be STRICTLY either "
        "'correct' or 'incorrect'.\n"
        "Question: {question}\n"
        "Reference Answer: {ground_truth}\n"
        "Model Response: {response}\n\n"
        "Output only one word: correct or incorrect."
    )

    sys_prompt = sys_prompt.replace("{question}", str(question))
    sys_prompt = sys_prompt.replace("{ground_truth}", str(ground_truth))
    sys_prompt = sys_prompt.replace("{response}", str(response))
    return [{"role": "system", "content": sys_prompt}]


def build_output_path(input_path: Path, output_file: str | None) -> Path:
    if output_file is not None:
        return Path(output_file).resolve()
    return input_path.with_name(f"{input_path.stem}_judged.jsonl")


def evaluate_one(
    client: OpenAI,
    record: dict[str, Any],
    index: int,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> dict[str, Any]:
    record_id = normalize_record_id(record, index)
    uuid = record.get("uuid", index)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=build_messages(record),
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or " "
        judge_value = to_bool_or_none(content)
        if judge_value is True:
            judge = "True"
        elif judge_value is False:
            judge = "False"
        else:
            judge = judge_value
    except Exception as exc:
        print(f"[ERROR] Model call failed idx={index}, id={record_id}: {exc}")
        judge = None

    return {
        "uuid": uuid,
        "id": record_id,
        "judge": judge,
    }


def evaluate_file(
    input_path: Path,
    output_path: Path,
    base_url: str,
    api_key_env: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> dict[str, Any]:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"Environment variable {api_key_env} is not set. "
            "Export your OpenAI API key before running this script."
        )

    client = OpenAI(base_url=base_url, api_key=api_key)
    rows = read_jsonl(input_path)
    total = len(rows)
    if total == 0:
        raise ValueError(f"Input file is empty: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    judged_true = 0
    judged_false = 0
    judged_unknown = 0

    with output_path.open("w", encoding="utf-8") as writer:
        for index, row in enumerate(rows):
            result = evaluate_one(
                client=client,
                record=row,
                index=index,
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            writer.write(json.dumps(result, ensure_ascii=False) + "\n")

            if result["judge"] == "True":
                judged_true += 1
            elif result["judge"] == "False":
                judged_false += 1
            else:
                judged_unknown += 1

            finished = index + 1
            if finished % 10 == 0 or finished == total:
                print(f"[{finished}/{total}] completed")

    return {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "samples": total,
        "judged_true": judged_true,
        "judged_false": judged_false,
        "judged_unknown": judged_unknown,
        "accuracy": round(judged_true / total, 6) if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge QA correctness with the OpenAI API.")
    parser.add_argument("input_file", help="Path to the prediction jsonl file.")
    parser.add_argument(
        "--output-file",
        default=None,
        help="Path to the judged jsonl file. Defaults to <input>_judged.jsonl.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="OpenAI API base URL.",
    )
    parser.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help="Environment variable that stores the API key.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name used for judging.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling value.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum generated tokens for the judge call.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file).resolve()
    output_path = build_output_path(input_path, args.output_file)
    metrics = evaluate_file(
        input_path=input_path,
        output_path=output_path,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
