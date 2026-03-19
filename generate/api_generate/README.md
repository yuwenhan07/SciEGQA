# API Generate

This directory contains the API-based inference script for `SciEGQA`.



## Script

- `generate_answer_api.py`: run multimodal inference on a converted SciEGQA dataset through the OpenAI API

## Input Format

The script expects a dataset converted into the LlamaFactory-style multimodal format:

- each sample is a JSON object in a `jsonl`
- the main fields used by this script are:
  - `messages`
  - `images`
  - `query`
  - `bbox_docvqa_id` / `sciegqa_id`

The script reads:

- the user prompt from `messages`
- the reference answer from `messages`
- the image paths from `images`

It then converts the image files into `data:` URLs and sends them to the OpenAI API as multimodal input.

## Usage

```bash
export OPENAI_API_KEY=your_api_key_here

python generate/api_generate/generate_answer_api.py \
  --dataset-jsonl data/SciEGQA/page/SciEGQA_page.jsonl \
  --output outputs/SciEGQA_api/gpt-5.2/generated_predictions.jsonl \
  --model gpt-5.2
```

## Common Arguments

- `--dataset-jsonl`: input dataset jsonl path
- `--output`: output jsonl path
- `--model`: OpenAI model name
- `--base-url`: API base URL, default is the official OpenAI endpoint
- `--api-key-env`: environment variable for the API key, default is `OPENAI_API_KEY`
- `--start-index`: start sample index
- `--max-samples`: optional cap for debugging
- `--max-images`: optional cap on the number of images sent per sample
- `--temperature`: generation temperature
- `--top-p`: optional top-p value
- `--resume`: reuse an existing output file and skip successful records

## Output Format

The script writes one JSON object per line. Each record includes:

- `sample_idx`
- `bbox_docvqa_id`
- `query`
- `label`
- `predict`
- `error`
- `num_images`
- `num_images_sent`

This output can be passed to the evaluation scripts under `metrics/`.

## Notes

- make sure the image paths stored in the dataset actually exist on disk
- this script uses `chat.completions.create(...)`
- if you want to use another OpenAI-compatible provider, you can still override `--base-url` and `--api-key-env`
