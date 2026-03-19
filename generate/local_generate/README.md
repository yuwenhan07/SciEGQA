# Local Generate

This directory contains the local inference scripts for `SciEGQA`. The whole pipeline follows the **LlamaFactory multimodal data protocol**:

- each dataset directory contains a `dataset_info.json`
- each dataset is stored as a `jsonl`
- each sample uses the `ShareGPT` style format, with the core fields:
  - `messages`
  - `images`

In other words, the data conversion scripts in this directory all produce outputs that can be directly consumed by LlamaFactory and by `vllm_infer.py` in this folder.

## Workflow

There are usually two local benchmark / generation workflows.

### 1. Direct QA / bbox data conversion

Use `data_transfer.py` to convert the original annotations into a LlamaFactory-compatible dataset:

```bash
python generate/local_generate/data_transfer.py \
  --input-jsonl data/SciEGQA/raw/SciEGQA.jsonl \
  --benchmark-dir data/SciEGQA/benchmark \
  --dataset-dir data/SciEGQA/page \
  --dataset-name SciEGQA_page \
  --mode page
```

Common `mode` values:

- `page`: use the page images specified by `evidence_page` for QA
- `crop`: crop the annotated regions and use them for QA
- `bbox`: build a bbox grounding dataset
- `document`: use all pages of the document for QA

Outputs:

- `dataset_dir/<dataset_name>.jsonl`
- `dataset_dir/dataset_info.json`
- `dataset_dir/images/` for `crop` mode

### 2. Predict bbox first, then crop for QA

If you already have bbox predictions, use `pred_bbox_crop.py` to convert the predicted boxes into a new crop-style LlamaFactory dataset:

```bash
python generate/local_generate/pred_bbox_crop.py \
  --source-jsonl data/SciEGQA/bbox/SciEGQA_bbox.jsonl \
  --prediction-jsonl outputs/SciEGQA_bbox/qwen3-vl-8b/generated_predictions.jsonl \
  --dataset-dir data/SciEGQA/pred_crop \
  --dataset-name SciEGQA_pred_crop
```

This script will:

1. parse the bbox predictions from `prediction-jsonl`
2. crop regions from the original page images
3. rebuild them into a `ShareGPT + images` dataset for QA inference
4. automatically register the dataset in `dataset_info.json`

Outputs:

- `dataset_dir/<dataset_name>.jsonl`
- `dataset_dir/dataset_info.json`
- `dataset_dir/images/`

## Run local QA generation

After the dataset is ready, use `generate_answers.py` to run local inference:

```bash
python generate/local_generate/generate_answers.py \
  --models-json generate/local_generate/SciEGQA_models.example.json \
  --dataset-dir data/SciEGQA/page \
  --dataset-name SciEGQA_page \
  --output-root outputs/SciEGQA/page
```

This script will:

1. load the model definitions from `models-json`
2. call either `vllm_infer.py` in this directory or the LlamaFactory CLI
3. generate `generated_predictions.jsonl`
4. additionally compute:
   - `exact_match`
   - `contains_match`
   - `anls`
5. summarize the results into `summary.json` and `summary.csv`

A single model output directory usually contains:

- `generated_predictions.jsonl`
- `scored_predictions.jsonl`
- `metrics.json`

## Model configuration

See:

- `SciEGQA_models.example.json`

Minimal example:

```json
[
  {
    "name": "qwen3-vl-8b",
    "model_name_or_path": "/path/to/model",
    "template": "qwen3_vl_nothink",
    "backend": "vllm",
    "image_max_pixels": 262144,
    "max_new_tokens": 256,
    "temperature": 0,
    "top_p": 1
  }
]
```

## Scripts in this directory

- `data_transfer.py`: convert the original annotations into a LlamaFactory-compatible dataset
- `pred_bbox_crop.py`: convert bbox predictions into a crop-style LlamaFactory dataset
- `generate_answers.py`: run batch local inference and summarize QA metrics
- `vllm_infer.py`: local inference entry based on LlamaFactory / vLLM
- `SciEGQA_models.example.json`: example model configuration

## Notes

- this pipeline is designed around the **LlamaFactory data protocol**
- `dataset_info.json` is required, and both conversion scripts update it automatically
- the converted data keeps the compatibility field `bbox_docvqa_id` and also adds `sciegqa_id`
- if you changed any paths locally, prefer passing them explicitly through command-line arguments instead of relying on defaults
