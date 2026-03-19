from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .mm_agent import MMAgent


GEN_PROMPT = (
    "You are an expert-level question-and-answer generator. Based strictly on the visible content of the given image, "
    "generate 3 challenging and diverse questions with their answers. You must rely only on the information shown in the image—no external knowledge or assumptions.\n"
    "Requirements:\n"
    "1) Questions should require deeper reasoning, multi-step understanding, or comparisons within the image (e.g., comparing numbers, interpreting relationships in a chart, or summarizing key insights);\n"
    "2) All questions must still be fully answerable using only the visible content;\n"
    "3) Focus on non-trivial factual or interpretive details such as numerical trends, contrasts, hierarchy, or explicit textual statements;\n"
    "4) Avoid yes/no or overly simple lookup questions unless absolutely necessary;\n"
    "5) Each QA pair must include a type field where type ∈ {span, number, date, yesno, categorical};\n"
    "6) Answers should be concise, precise, and supported directly by the image content.\n"
    "Output Format:\n"
    "Produce exactly three <qa> blocks (unless no valid questions exist). Each block must follow this template:\n"
    "<qa>\n"
    "<question>Question text</question>\n"
    "<answer>Answer text</answer>\n"
    "<type>span|number|date|yesno|categorical</type>\n"
    "<difficulty>easy|medium|hard</difficulty>\n"
    "</qa>\n"
    "If the image does not provide enough information for any valid question, output the single token NO_VALID_QA.\n"
    "Do not include any additional commentary."
)

_QA_BLOCK_PATTERN = re.compile(r"<qa\b[^>]*>(.*?)</qa>", re.IGNORECASE | re.DOTALL)
_TAG_PATTERNS = {
    "question": re.compile(r"<question>(.*?)</question>", re.IGNORECASE | re.DOTALL),
    "answer": re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL),
    "type": re.compile(r"<type>(.*?)</type>", re.IGNORECASE | re.DOTALL),
    "difficulty": re.compile(r"<difficulty>(.*?)</difficulty>", re.IGNORECASE | re.DOTALL),
}
_VALID_QA_TYPES = {"span", "number", "date", "yesno", "categorical"}
_JUDGE_OUTPUT_PATTERN = re.compile(r'"\s*keep\s*"\s*:\s*(true|false)', re.IGNORECASE)


@dataclass(slots=True)
class QAGeneratorConfig:
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    prompt: str = GEN_PROMPT
    max_new_tokens: int = 512
    temperature: float = 0
    top_p: float = 1
    repetition_penalty: float = 1.05
    batch_size: int = 1
    workers: int = 1


@dataclass(slots=True)
class QAItem:
    q: str
    a: str
    qa_type: str
    difficulty: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "q": self.q,
            "a": self.a,
            "type": self.qa_type,
            "difficulty": self.difficulty,
        }


@dataclass(slots=True)
class QAGenerationResult:
    qas: List[QAItem]
    raw_output: str


class QwenQAGenerator:
    """Generate question-answer pairs directly from an image."""

    def __init__(self, config: QAGeneratorConfig, agent_backend: MMAgent | None = None):
        self.config = config
        self._agent = agent_backend
        if self._agent is None:
            print(f"Loading model: {config.model_name}")
            self.processor = AutoProcessor.from_pretrained(
                config.model_name,
                trust_remote_code=True,
                use_fast=False,
            )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.model_name,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            self.model.eval()
        else:
            self.processor = None
            self.model = None

    @property
    def supports_batch(self) -> bool:
        return self._agent is not None

    def generate_for_image(self, image_path: Path) -> QAGenerationResult:
        decoded = ""
        try:
            if self._agent is not None:
                prompt = [
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.config.prompt},
                                {"type": "image", "image": str(image_path)},
                            ],
                        }
                    ]
                ]
                decoded = self._agent.generate(
                    prompt,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    repetition_penalty=self.config.repetition_penalty,
                )[0]
            else:
                with Image.open(image_path) as pil_image:
                    img = pil_image.convert("RGB")
                prompt = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.config.prompt},
                            {"type": "image", "image": img},
                        ],
                    }
                ]
                chat_text = self.processor.apply_chat_template(
                    prompt,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                try:
                    inputs = self.processor(
                        text=[chat_text],
                        images=[img],
                        return_tensors="pt",
                    ).to(self.model.device)
                except Exception:
                    return QAGenerationResult([], "")

                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        do_sample=True,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        repetition_penalty=self.config.repetition_penalty,
                    )
                generated = output[:, inputs["input_ids"].shape[1]:]
                decoded = self.processor.batch_decode(
                    generated,
                    skip_special_tokens=True,
                )[0]
        except Exception:
            return QAGenerationResult([], decoded)
        return QAGenerationResult(_parse_qas(decoded), decoded)

    def generate_batch(self, image_paths: Sequence[Path]) -> List[QAGenerationResult]:
        if not image_paths:
            return []
        if self._agent is None:
            return [self.generate_for_image(path) for path in image_paths]

        conversations: List[List[dict]] = []
        for path in image_paths:
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.config.prompt},
                            {"type": "image", "image": str(path)},
                        ],
                    }
                ]
            )

        outputs = self._agent.generate(
            conversations,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
        )
        return [QAGenerationResult(_parse_qas(text), text) for text in outputs]


def _parse_qas(text: str) -> List[QAItem]:
    tagged = _parse_tagged_qas(text)
    if tagged:
        return tagged
    json_like = _parse_json_like_qas(text)
    if json_like:
        return json_like
    return _parse_generic_qas(text)


def _parse_tagged_qas(text: str) -> List[QAItem]:
    candidates: List[dict] = []
    for block in _QA_BLOCK_PATTERN.findall(text):
        question = _extract_tag(block, "question")
        answer = _extract_tag(block, "answer")
        qa_type = _extract_tag(block, "type")
        difficulty = _extract_tag(block, "difficulty") or "medium"
        if not question or not answer:
            continue
        candidates.append({"q": question, "a": answer, "type": qa_type, "difficulty": difficulty})
    return _build_items(candidates)


def _parse_json_like_qas(text: str) -> List[QAItem]:
    parsed = _safe_json_parse(text)
    candidates = parsed.get("qas", [])
    if not isinstance(candidates, list):
        return []
    return _build_items(candidates)


def _parse_generic_qas(text: str) -> List[QAItem]:
    candidates: List[dict] = []
    current: dict = {}
    last_field: Optional[str] = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if "q" in current and "a" in current:
                candidates.append(
                    {
                        "q": current.get("q", ""),
                        "a": current.get("a", ""),
                        "type": current.get("type", ""),
                        "difficulty": current.get("difficulty", ""),
                    }
                )
            current = {}
            last_field = None
            continue
        if _is_question_line(line):
            if "q" in current and "a" in current:
                candidates.append(
                    {
                        "q": current.get("q", ""),
                        "a": current.get("a", ""),
                        "type": current.get("type", ""),
                        "difficulty": current.get("difficulty", ""),
                    }
                )
                current = {}
            current["q"] = _split_field(line)
            last_field = "q"
        elif _is_answer_line(line):
            current["a"] = _split_field(line)
            last_field = "a"
        elif line.lower().startswith("type"):
            current["type"] = _split_field(line)
            last_field = "type"
        elif line.lower().startswith("difficulty"):
            current["difficulty"] = _split_field(line)
            last_field = "difficulty"
        else:
            if last_field and last_field in current:
                current[last_field] = f"{current[last_field]} {line}".strip()
    if "q" in current and "a" in current:
        candidates.append(
            {
                "q": current.get("q", ""),
                "a": current.get("a", ""),
                "type": current.get("type", ""),
                "difficulty": current.get("difficulty", ""),
            }
        )
    return _build_items(candidates)


def _is_question_line(text: str) -> bool:
    low = text.strip().lower()
    return bool(re.match(r"q(?:uestion)?(?:\b|\d)", low))


def _is_answer_line(text: str) -> bool:
    low = text.strip().lower()
    return bool(re.match(r"a(?:nswer)?(?:\b|\d)", low))


def _split_field(line: str) -> str:
    stripped = line.strip()
    for sep in (":", "-", "："):
        if sep in stripped:
            return stripped.split(sep, 1)[1].strip()
    return line.strip()


def _build_items(candidates: Iterable[dict]) -> List[QAItem]:
    seen: set[str] = set()
    cleaned: List[QAItem] = []
    for qa in candidates:
        if not isinstance(qa, dict):
            continue
        question = _clean_tag_text(_to_text(qa.get("q")))
        answer = _clean_tag_text(_to_text(qa.get("a")))
        qa_type = _to_text(qa.get("type")).lower()
        difficulty = _clean_tag_text(_to_text(qa.get("difficulty"))) or "medium"
        if not question or not answer:
            continue
        if qa_type not in _VALID_QA_TYPES:
            qa_type = "span"
        key = question.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(QAItem(question, answer, qa_type, difficulty))
        if len(cleaned) >= 6:
            break
    return cleaned


def _should_retry_generation(result: QAGenerationResult) -> bool:
    if result.qas:
        return False
    raw = (result.raw_output or "").strip()
    if not raw:
        return False
    return _looks_like_judge_output(raw)


def _looks_like_judge_output(raw: str) -> bool:
    if not raw:
        return False
    if "qas" in raw.lower():
        return False
    return bool(_JUDGE_OUTPUT_PATTERN.search(raw))


def _extract_tag(block: str, tag: str) -> str:
    pattern = _TAG_PATTERNS.get(tag)
    if pattern is None:
        return ""
    match = pattern.search(block)
    if not match:
        return ""
    return _clean_tag_text(match.group(1))


def _clean_tag_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _safe_json_parse(text: str) -> Dict[str, object]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "qas" in obj:
            return obj
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict) and "qas" in obj:
                return obj
        except Exception:
            pass
    return {"qas": []}


def _to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v).strip() for v in value if v is not None)
    if isinstance(value, dict):
        if "text" in value:
            return str(value["text"]).strip()
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value).strip()


def run_qa_for_directory(
    generator: QwenQAGenerator,
    image_dir: Path,
    output_jsonl: Path,
    evidence_map: Optional[Dict[str, dict]] = None,
    batch_size: int | None = None,
) -> Dict[str, int]:
    evidence_map = evidence_map or build_evidence_map(image_dir)
    images = sorted(
        p for p in image_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    )
    records: List[dict] = []
    output_jsonl = output_jsonl.with_suffix(".jsonl")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    raw_output_dir = output_jsonl.parent / "qa_raw_outputs"
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    effective_batch = max(1, batch_size or getattr(generator.config, "batch_size", 1))
    if not generator.supports_batch:
        effective_batch = 1

    def _chunk(seq: Sequence[Path], size: int) -> Iterable[List[Path]]:
        for idx in range(0, len(seq), size):
            yield seq[idx : idx + size]

    def _generate_chunk(chunk: Sequence[Path]) -> List[tuple[Path, QAGenerationResult]]:
        try:
            batch_qas = list(generator.generate_batch(chunk))
            if len(batch_qas) != len(chunk):
                adjusted = batch_qas[: len(chunk)]
                while len(adjusted) < len(chunk):
                    adjusted.append(QAGenerationResult([], ""))
                batch_qas = adjusted
        except Exception:
            batch_qas = []
            for path in chunk:
                try:
                    batch_qas.append(generator.generate_for_image(path))
                except Exception:
                    batch_qas.append(QAGenerationResult([], ""))
        results: List[tuple[Path, QAGenerationResult]] = []
        for path, qa_result in zip(chunk, batch_qas):
            if _should_retry_generation(qa_result):
                try:
                    retry = generator.generate_for_image(path)
                except Exception:
                    retry = None
                if retry and (retry.qas or retry.raw_output != qa_result.raw_output):
                    qa_result = retry
            results.append((path, qa_result))
        return results

    progress = tqdm(total=len(images), desc=f"Generating QA: {image_dir.name}", leave=False)
    chunks = list(_chunk(images, effective_batch))
    backend = getattr(generator, "_agent", None)
    is_vllm = bool(getattr(backend, "use_vllm", False))
    use_parallel = (
        generator.supports_batch
        and not is_vllm  # vLLM backend is not thread-safe for concurrent generate() calls
        and getattr(generator.config, "workers", 1) > 1
        and len(chunks) > 1
    )

    if use_parallel:
        max_workers = min(generator.config.workers, len(chunks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_generate_chunk, chunk) for chunk in chunks]
            for future in futures:
                results = future.result()
                for path, qa_result in results:
                    evidence = evidence_map.get(
                        path.name,
                        {"page": [], "bbox": None, "image": f"./{path.name}", "type": "unknown"},
                    )
                    raw_path = raw_output_dir / f"{path.name}.txt"
                    try:
                        raw_path.write_text(qa_result.raw_output, encoding="utf-8")
                    except Exception:
                        pass
                    records.append(
                        {
                            "image": path.name,
                            "evidence": evidence,
                            "qas": [qa.to_dict() for qa in qa_result.qas],
                        }
                    )
                progress.update(len(results))
    else:
        for chunk in chunks:
            results = _generate_chunk(chunk)
            for path, qa_result in results:
                evidence = evidence_map.get(
                    path.name,
                    {"page": [], "bbox": None, "image": f"./{path.name}", "type": "unknown"},
                )
                raw_path = raw_output_dir / f"{path.name}.txt"
                try:
                    raw_path.write_text(qa_result.raw_output, encoding="utf-8")
                except Exception:
                    pass
                records.append(
                    {
                        "image": path.name,
                        "evidence": evidence,
                        "qas": [qa.to_dict() for qa in qa_result.qas],
                    }
                )
            progress.update(len(results))
    progress.close()

    with output_jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total_pairs = sum(len(rec["qas"]) for rec in records)
    return {"images": len(records), "qa_pairs": total_pairs}


def run_qa_for_root(
    generator: QwenQAGenerator,
    root_dir: Path,
    clean_summary: Path | None = None,
    batch_size: int | None = None,
) -> Dict[str, int]:
    total_pages = 0
    total_images = 0
    total_pairs = 0
    for report, page, clean_dir in tqdm(list(_iter_clean_dirs(root_dir)), desc="Iterating reports/pages", unit="page"):
        out_json = clean_dir / "qa_pairs.jsonl"
        ev_map = build_evidence_map(clean_dir, clean_summary)
        processed = run_qa_for_directory(generator, clean_dir, out_json, ev_map, batch_size=batch_size)
        total_pages += 1
        total_images += processed.get("images", 0)
        total_pairs += processed.get("qa_pairs", 0)

    _merge_report_level(root_dir)
    return {"pages": total_pages, "images": total_images, "qa_pairs": total_pairs}


def _iter_clean_dirs(root_dir: Path) -> Iterator[tuple[str, str, Path]]:
    for report_dir in sorted(root_dir.iterdir(), key=lambda p: p.name):
        if not report_dir.is_dir():
            continue
        for page_dir in sorted(report_dir.iterdir(), key=lambda p: (len(p.name), p.name)):
            if not page_dir.is_dir():
                continue
            clean_dir = page_dir / "clean_crops"
            if clean_dir.is_dir():
                yield report_dir.name, page_dir.name, clean_dir


def build_evidence_map(
    image_dir: Path,
    clean_summary_path: Path | None = None,
) -> Dict[str, dict]:
    candidates: List[Path] = []
    if clean_summary_path and Path(clean_summary_path).is_file():
        candidates.append(Path(clean_summary_path))
    for cand in [
        image_dir / "clean_summary.json",
        image_dir.parent / "clean_summary.json",
    ]:
        if cand.is_file():
            candidates.append(cand)
    for cand in sorted(image_dir.glob("clean_summary*.json")):
        if cand.is_file():
            candidates.append(cand)

    evidence_map: Dict[str, dict] = {}
    for cand in candidates:
        try:
            data = json.loads(cand.read_text(encoding="utf-8"))
        except Exception:
            continue

        items = []
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            items = data["items"]
        elif isinstance(data, list):
            items = data
        else:
            continue

        for item in items:
            try:
                image_name = item.get("image") or item.get("img") or item.get("filename")
                if not image_name:
                    continue
                page_raw = item.get("page")
                if isinstance(page_raw, list):
                    page_list = [int(p) for p in page_raw if isinstance(p, int)]
                elif page_raw is None:
                    page_list = []
                else:
                    try:
                        page_list = [int(page_raw)]
                    except Exception:
                        page_list = []
                bbox = item.get("bbox_xyxy") or item.get("bbox") or item.get("bbox_xywh")
                ev_type = item.get("type", "unknown")
                evidence_map[image_name] = {
                    "page": page_list,
                    "bbox": bbox,
                    "image": f"{image_dir.name}/{image_name}",
                    "type": ev_type,
                }
            except Exception:
                continue
        if evidence_map:
            break
    return evidence_map


def _merge_report_level(root_dir: Path) -> None:
    print("Merging qa_pairs.jsonl files per report folder ...")
    for report_dir in sorted(root_dir.iterdir(), key=lambda p: p.name):
        if not report_dir.is_dir():
            continue
        merged: List[str] = []
        for qa_file in report_dir.rglob("qa_pairs.jsonl"):
            try:
                with qa_file.open("r", encoding="utf-8") as f:
                    merged.extend(line.strip() for line in f if line.strip())
            except Exception as exc:
                print(f"WARNING: failed to read {qa_file}: {exc}")
        if merged:
            output = report_dir / f"{report_dir.name}.jsonl"
            with output.open("w", encoding="utf-8") as f:
                for line in merged:
                    f.write(line + "\n")
            print(f"SUCCESS: merged {len(merged)} item(s) → {output}")
        else:
            print(f"WARNING: no qa_pairs.jsonl found in {report_dir}")
