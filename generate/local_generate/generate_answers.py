#!/usr/bin/env python3

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_MODELS_JSON = SCRIPT_DIR / "SciEGQA_models.example.json"
DEFAULT_DATASET_DIR = "data/SciEGQA/page"
DEFAULT_DATASET_NAME = "SciEGQA_page"
DEFAULT_OUTPUT_ROOT = "outputs/SciEGQA/page"


def normalize_text(text):
    if text is None:
        return ""
    return " ".join(str(text).strip().lower().split())


def levenshtein_distance(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def anls_score(prediction, reference):
    pred = normalize_text(prediction)
    ref = normalize_text(reference)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0

    dist = levenshtein_distance(pred, ref)
    max_len = max(len(pred), len(ref))
    if max_len == 0:
        return 1.0

    nl = dist / max_len
    if nl >= 0.5:
        return 0.0
    return 1.0 - nl


def exact_match(prediction, reference):
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0


def contains_match(prediction, reference):
    pred = normalize_text(prediction)
    ref = normalize_text(reference)
    if not pred or not ref:
        return 0.0
    return 1.0 if ref in pred or pred in ref else 0.0


def load_model_configs(path):
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Model config JSON must be a list.")
    return data


def load_existing_metrics(metrics_path):
    with Path(metrics_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def has_complete_model_outputs(output_dir):
    required_files = [
        output_dir / "generated_predictions.jsonl",
        output_dir / "scored_predictions.jsonl",
        output_dir / "metrics.json",
    ]
    return all(path.is_file() for path in required_files)


def resolve_cli_command():
    cli = shutil.which("llamafactory-cli")
    if cli:
        return [cli]
    return [sys.executable, "-m", "llamafactory.cli"]


def build_subprocess_env():
    env = os.environ.copy()
    current = env.get("PYTHONPATH")
    src_candidates = [REPO_ROOT / "src"]
    existing_src = [str(path) for path in src_candidates if path.is_dir()]
    if existing_src:
        env["PYTHONPATH"] = os.pathsep.join(existing_src + ([current] if current else []))
    return env


def validate_dataset(dataset_dir: Path, dataset_name: str) -> None:
    dataset_info_path = dataset_dir / "dataset_info.json"
    if not dataset_info_path.is_file():
        raise FileNotFoundError(
            f"dataset_info.json not found in {dataset_dir}. Run data_transfer.py or pred_bbox_crop.py first."
        )

    with dataset_info_path.open("r", encoding="utf-8") as f:
        dataset_info = json.load(f)

    if dataset_name not in dataset_info:
        raise KeyError(f"Dataset {dataset_name!r} is not registered in {dataset_info_path}.")


def build_vllm_command(model_cfg, args, output_dir):
    script_path = Path(__file__).resolve().parent / "vllm_infer.py"
    command = [
        sys.executable,
        str(script_path),
        "--model_name_or_path",
        model_cfg["model_name_or_path"],
        "--dataset",
        args.dataset_name,
        "--dataset_dir",
        args.dataset_dir,
        "--template",
        model_cfg["template"],
        "--cutoff_len",
        str(model_cfg.get("cutoff_len", args.cutoff_len)),
        "--max_new_tokens",
        str(model_cfg.get("max_new_tokens", args.max_new_tokens)),
        "--image_max_pixels",
        str(model_cfg.get("image_max_pixels", args.image_max_pixels)),
        "--save_name",
        str(output_dir / "generated_predictions.jsonl"),
        "--matrix_save_name",
        str(output_dir / "generation_metrics.json"),
        "--batch_size",
        str(model_cfg.get("batch_size", args.vllm_batch_size)),
    ]
    if args.max_samples is not None:
        command.extend(["--max_samples", str(args.max_samples)])
    max_images_per_sample = model_cfg.get("max_images_per_sample", args.max_images_per_sample)
    if max_images_per_sample is not None:
        command.extend(["--max_images_per_sample", str(max_images_per_sample)])
    if model_cfg.get("adapter_name_or_path"):
        command.extend(["--adapter_name_or_path", model_cfg["adapter_name_or_path"]])
    if "temperature" in model_cfg:
        command.extend(["--temperature", str(model_cfg["temperature"])])
    if "top_p" in model_cfg:
        command.extend(["--top_p", str(model_cfg["top_p"])])
    if "top_k" in model_cfg:
        command.extend(["--top_k", str(model_cfg["top_k"])])
    if model_cfg.get("vllm_config"):
        value = model_cfg["vllm_config"]
        if not isinstance(value, str):
            value = json.dumps(value)
        command.extend(["--vllm_config", value])
    return command


def build_hf_command(model_cfg, args, output_dir):
    command = resolve_cli_command() + [
        "train",
        "--stage",
        "sft",
        "--do_predict",
        "true",
        "--predict_with_generate",
        "true",
        "--finetuning_type",
        model_cfg.get("finetuning_type", "full"),
        "--model_name_or_path",
        model_cfg["model_name_or_path"],
        "--eval_dataset",
        args.dataset_name,
        "--dataset_dir",
        args.dataset_dir,
        "--template",
        model_cfg["template"],
        "--cutoff_len",
        str(model_cfg.get("cutoff_len", args.cutoff_len)),
        "--max_new_tokens",
        str(model_cfg.get("max_new_tokens", args.max_new_tokens)),
        "--image_max_pixels",
        str(model_cfg.get("image_max_pixels", args.image_max_pixels)),
        "--per_device_eval_batch_size",
        str(model_cfg.get("per_device_eval_batch_size", args.per_device_eval_batch_size)),
        "--output_dir",
        str(output_dir),
        "--overwrite_output_dir",
        "true",
        "--report_to",
        "none",
        "--trust_remote_code",
        str(model_cfg.get("trust_remote_code", True)).lower(),
    ]
    if args.max_samples is not None:
        command.extend(["--max_samples", str(args.max_samples)])
    if model_cfg.get("adapter_name_or_path"):
        command.extend(["--adapter_name_or_path", model_cfg["adapter_name_or_path"]])
    precision = model_cfg.get("precision", args.hf_precision)
    if precision == "bf16":
        command.extend(["--bf16", "true"])
    elif precision == "fp16":
        command.extend(["--fp16", "true"])
    elif precision not in (None, "auto"):
        raise ValueError(f"Unsupported precision: {precision}")

    extra_args = model_cfg.get("extra_args", {})
    for key, value in extra_args.items():
        command.extend([f"--{key}", str(value)])
    return command


def compute_metrics(prediction_path, scored_path):
    total = 0
    exact = 0.0
    contains = 0.0
    anls = 0.0

    with Path(prediction_path).open("r", encoding="utf-8") as infile, Path(scored_path).open(
        "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            row = json.loads(line)
            pred = row.get("predict", "")
            label = row.get("label", "")
            row["normalized_predict"] = normalize_text(pred)
            row["normalized_label"] = normalize_text(label)
            row["exact_match"] = exact_match(pred, label)
            row["contains_match"] = contains_match(pred, label)
            row["anls"] = anls_score(pred, label)
            outfile.write(json.dumps(row, ensure_ascii=False) + "\n")

            total += 1
            exact += row["exact_match"]
            contains += row["contains_match"]
            anls += row["anls"]

    if total == 0:
        return {"samples": 0, "exact_match": 0.0, "contains_match": 0.0, "anls": 0.0}

    return {
        "samples": total,
        "exact_match": exact / total,
        "contains_match": contains / total,
        "anls": anls / total,
    }


def run_one_model(model_cfg, args, output_root):
    name = model_cfg["name"]
    backend = model_cfg.get("backend", args.backend)
    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume and has_complete_model_outputs(output_dir):
        print(f"[skip] {name} ({backend})")
        return load_existing_metrics(output_dir / "metrics.json")

    if backend == "vllm":
        command = build_vllm_command(model_cfg, args, output_dir)
    elif backend == "hf":
        command = build_hf_command(model_cfg, args, output_dir)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    print(f"[run] {name} ({backend})")
    print(" ".join(command))
    subprocess.run(command, check=True, env=build_subprocess_env())

    prediction_path = output_dir / "generated_predictions.jsonl"
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {prediction_path}")

    metrics = compute_metrics(prediction_path, output_dir / "scored_predictions.jsonl")
    metrics.update(
        {
            "name": name,
            "backend": backend,
            "model_name_or_path": model_cfg["model_name_or_path"],
            "template": model_cfg["template"],
        }
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return metrics


def write_summary(metrics_list, output_root):
    summary_json = output_root / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(metrics_list, f, ensure_ascii=False, indent=2)
        f.write("\n")

    summary_csv = output_root / "summary.csv"
    fieldnames = [
        "name",
        "backend",
        "model_name_or_path",
        "template",
        "samples",
        "exact_match",
        "contains_match",
        "anls",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_list:
            writer.writerow({key: row.get(key) for key in fieldnames})

    print(f"Summary written to {summary_json}")
    print(f"Summary written to {summary_csv}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run batch inference on SciEGQA datasets converted with the LlamaFactory protocol."
    )
    parser.add_argument(
        "--models-json",
        default=str(DEFAULT_MODELS_JSON),
        help="Path to a JSON file containing model definitions.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help="Directory that contains dataset_info.json for the converted SciEGQA dataset.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Dataset name inside dataset_info.json.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory used to store per-model results and summary files.",
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "hf"],
        default="vllm",
        help="Default backend if a model entry does not override it.",
    )
    parser.add_argument("--cutoff-len", type=int, default=20480, help="Default cutoff length.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Default generation length.")
    parser.add_argument("--image-max-pixels", type=int, default=26214400, help="Default image pixel budget.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample cap for debugging.")
    parser.add_argument("--vllm-batch-size", type=int, default=16, help="Default vLLM batch size.")
    parser.add_argument(
        "--max-images-per-sample",
        type=int,
        default=None,
        help="Truncate per-sample image count before vLLM inference. Defaults to vLLM's own image limit when unset.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=1,
        help="Default HF eval batch size.",
    )
    parser.add_argument(
        "--hf-precision",
        choices=["auto", "bf16", "fp16"],
        default="auto",
        help="Default precision for the HF backend.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip models whose outputs already exist in the output directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    validate_dataset(Path(args.dataset_dir).resolve(), args.dataset_name)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    model_configs = load_model_configs(args.models_json)

    metrics_list = []
    for model_cfg in model_configs:
        metrics = run_one_model(model_cfg, args, output_root)
        metrics_list.append(metrics)

    metrics_list.sort(key=lambda item: (-item["anls"], -item["contains_match"], -item["exact_match"], item["name"]))
    write_summary(metrics_list, output_root)


if __name__ == "__main__":
    main()
