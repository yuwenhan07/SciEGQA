# Metrics

This directory contains the evaluation scripts for `SciEGQA`. At the moment, it mainly includes two types of evaluation:

- `metrics/IoU_compute.py`: compute IoU for bbox predictions
- `metrics/Acc_judge.py`: use the OpenAI API to judge whether a QA answer is correct

## 1. IoU Evaluation

### Usage

Run IoU evaluation on a single prediction file:

```bash
python metrics/IoU_compute.py \
  outputs/SciEGQA_bbox/qwen3-vl-8b/generated_predictions.jsonl \
  --output-dir metrics/results
```

This produces:

- `metrics/results/generated_predictions_iou_scored.jsonl`
- `metrics/results/generated_predictions_iou_metrics.json`

If `--output-dir` is not provided, the outputs are written next to the input file.

### Input Format

The script expects a `generated_predictions.jsonl`-style file:

- `label` is the reference answer
- `predict` is the model output

See the scripts in the generate directory for the specific generation method to produce the pred_bbox results.

The reference bbox label is expected to be grouped by page:

```json
[
  [[x1, y1, x2, y2], ...],
  [[x1, y1, x2, y2], ...]
]
```

Each outer item corresponds to one page.

### Computation Logic

#### 1. Coordinate parsing

The script first tries JSON parsing, and also supports several noisy formats:

- JSON inside code fences
- Python list literals
- coordinates written as strings, such as `["218, 620, 883, 740"]`
- free text containing `[x1, y1, x2, y2]`

#### 2. Single-box IoU

```text
IoU = intersection_area / union_area
```

#### 3. Multi-box matching on one page

When a page has multiple predicted boxes and multiple reference boxes, the script does not compare them by order. Instead, it performs one-to-one maximum matching:

- each predicted box can match at most one reference box
- each reference box can match at most one predicted box
- the objective is to maximize the total IoU sum

The page score is defined as:

```text
page_iou = matched_iou_sum / max(num_pred_boxes, num_gt_boxes)
```

This penalizes both missing boxes and extra boxes.

#### 4. Multi-page logic

If the model output is already grouped by page:

```json
[
  [[...]],
  [[...]]
]
```

the script evaluates it page by page.

If the model only outputs a flat list of boxes:

```json
[
  [x1, y1, x2, y2],
  [x1, y1, x2, y2]
]
```

the script handles it in two ways:

1. If the reference is multi-page, each page has exactly one reference box, and the number of predicted boxes equals the number of pages, the script promotes the flat output back to page structure by order.
2. Otherwise it falls back to `flat fallback`, where all reference boxes are flattened and matched globally.

#### 5. Sample-level score

```text
sample_iou = matched_iou_sum / max(total_pred_boxes, total_gt_boxes)
```

In grouped mode, matching is restricted within each page.
In `flat fallback` mode, matching is global.

### Output Fields

Each row in `*_iou_scored.jsonl` includes:

- `sample_iou`
- `page_iou`
- `predict_parse_mode`
- `grouped_by_page`
- `parsed_predict_pages`
- `parsed_label_pages`
- `error`

The summary in `*_iou_metrics.json` includes:

- `mean_sample_iou`
- `mean_sample_iou_valid_only`
- `mean_page_iou`
- `mean_page_iou_valid_only`
- `sample_iou_thresholds`
- `page_iou_thresholds`
- `grouped_prediction_samples`
- `flat_fallback_samples`
- `predict_parse_modes`
- `parse_failures`

The default threshold summary includes:

- `IoU@0.3`
- `IoU@0.5`
- `IoU@0.7`

## 2. Acc Judge Evaluation

### Usage

First export your OpenAI API key:

```bash
export OPENAI_API_KEY=your_api_key_here
```

Then run:

```bash
python metrics/Acc_judge.py \
  outputs/SciEGQA_page/qwen3-vl-8b/generated_predictions.jsonl \
  --output-file metrics/results/generated_predictions_judged.jsonl
```

If `--output-file` is not provided, the script writes:

- `<input_stem>_judged.jsonl`

For example, if the input file is `generated_predictions.jsonl`, the default output will be:

- `generated_predictions_judged.jsonl`

### Input Format

`Acc_judge.py` supports two common input styles:

1. API / inference output format

- `query`
- `label`
- `predict`

2. `generated_predictions.jsonl`-style format

- `prompt`
- `label`
- `predict`

### Evaluation Logic

The script builds a judge prompt for each sample and sends it to the OpenAI API, asking the model to output:

- `correct`
- `incorrect`

The script then normalizes the response into:

- `True`
- `False`
- or the original output if it cannot be parsed reliably

### Common Arguments

- `--output-file`: output judged jsonl path
- `--base-url`: API base URL, default is `https://api.openai.com/v1`
- `--api-key-env`: API key environment variable, default is `OPENAI_API_KEY`
- `--model`: judge model name, default is `gpt5.2`
- `--temperature`: default `0.0`
- `--top-p`: default `1.0`
- `--max-tokens`: default `128`

### Output Format

The judged jsonl contains:

- `uuid`
- `id`
- `judge`

The script also prints a summary JSON to stdout, including:

- `samples`
- `judged_true`
- `judged_false`
- `judged_unknown`
- `accuracy`
