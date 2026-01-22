#!/bin/bash
# 遍历 input_judgement 下所有模型版本的子文件夹，并运行 judge_code/judge.py
# 输出路径保持相同层级结构到 judge/ 目录下

INPUT_ROOT="../judgement_input/input_qwen3_32B/"
OUTPUT_ROOT="../judgement_output/qwen3_32B"

# 遍历所有 .jsonl 文件
find "$INPUT_ROOT" -type f -name "*.jsonl" | while read -r file; do
  rel_path=$(python3 -c "import os.path; print(os.path.relpath('$file', '$INPUT_ROOT'))")

  out_path="$OUTPUT_ROOT/$rel_path"

  mkdir -p "$(dirname "$out_path")"

  echo "Running: nohup python judge.py '$file' '$out_path' &"

  nohup python judge.py "$file" "$out_path" &
done