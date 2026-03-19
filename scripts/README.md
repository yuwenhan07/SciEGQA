# SciEGQA Auto-Construction Pipeline

This folder now has one recommended entry point:

```bash
python Scripts/auto_train_pipeline.py ...
```

It implements the workflow described in the paper under `Automatically Training Set Construction` section:

1. `pdf2png`: render PDF pages to PNG images
2. `sam`: run SAM segmentation and save candidate crops
3. `judge`: filter bad crops and build `clean_crops`
4. `qa`: generate QA pairs from cleaned evidence regions
5. `transform`: merge per-document JSONL files into the final training format

## Recommended Usage

Run the full pipeline for one PDF or a directory of PDFs:

```bash
python Scripts/auto_train_pipeline.py run data/pdfs \
  --work_root data/auto_train_work \
  --output_dir data/auto_train_jsonl \
  --sam_checkpoint /path/to/sam_checkpoint.pth
```

If you want to run the stages manually:

```bash
python Scripts/auto_train_pipeline.py pdf2png <pdf_or_dir> <png_output_dir>
python Scripts/auto_train_pipeline.py sam --input_dir <png_output_dir> --output_root <sam_output_dir> --checkpoint <sam_ckpt>
python Scripts/auto_train_pipeline.py judge --root_dir <sam_output_dir>
python Scripts/auto_train_pipeline.py qa --root_dir <sam_output_dir>
python Scripts/auto_train_pipeline.py transform --input_dir <sam_output_dir> --output <final_jsonl>
```

## Directory Layout

The full `run` command writes one working directory per PDF:

```text
<work_root>/
  <doc_name>/
    pages/
      <doc_name>/
        1.png
        2.png
        ...
    processed/
      <doc_name>/
        <page_id>/
          crops/
          judge/
          clean_crops/
            clean_summary.json
          qa_pairs.jsonl
      <doc_name>.jsonl
```

This layout matches the automated pipeline:

- `pages/`: rasterized page images
- `crops/`: raw SAM candidate regions
- `judge/`: VLM judgement output
- `clean_crops/`: filtered evidence regions kept for QA generation
- `qa_pairs.jsonl`: QA generated for one page
- `<doc_name>.jsonl`: merged QA output for one document

## Script Roles

The following files are compatibility wrappers around the unified CLI:

- `step0_pdf2png.py`
- `step1_sam_crop.py`
- `step2_judge_and_clean.py`
- `step3_generate_qa.py`
- `step4_data_tran.py`
- `process_doc.py`

They are kept so existing commands do not break, but new usage should prefer `auto_train_pipeline.py`.

The remaining files are lower-level modules or legacy/specialized utilities:

- `boundingdoc/`: shared implementation used by the CLI
- `process_sam_judge.py`: PDF -> SAM -> judge only
- `process_qa_from_judge.py`: QA generation from pre-merged judge outputs
- `run_all_serial.sh`: serial batch launcher for a specific local environment

## Practical Notes

- `run` is the cleanest way to reproduce the paper pipeline end-to-end.
- Use the `step*` wrappers only when debugging a specific stage.
- Keep model/backend settings explicit when mixing SAM GPUs and VLM GPUs.
- `transform` expects the merged per-document JSONL files produced by the previous stages.
