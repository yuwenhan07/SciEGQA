from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from tqdm import tqdm


@dataclass(slots=True)
class TransformConfig:
    input_dir: Path
    output_file: Path
    type_map: Optional[Path] = None


def _load_type_map(path: Path | None) -> Optional[Dict[str, str]]:
    if not path:
        return None
    if not path.exists():
        print(f"[warn] type_map file not found: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[warn] failed to load type_map '{path}': {exc}")
        return None


def _iter_jsonl_files(root: Path) -> Iterable[Path]:
    # Support both legacy layout (processed/<doc>/<doc>.jsonl) and flattened layout
    # (processed/<doc>.jsonl) by scanning for jsonl files under any "processed" folder.
    patterns = ["*/processed/*.jsonl", "*/processed/*/*.jsonl"]
    seen: set[Path] = set()
    for pattern in patterns:
        for file_path in sorted(root.glob(pattern)):
            if file_path.is_file() and file_path not in seen:
                seen.add(file_path)
                yield file_path

def _doc_name_for(path: Path, base_dir: Path) -> str:
    try:
        rel = path.relative_to(base_dir)
        return rel.parts[0] if len(rel.parts) >= 2 else path.parent.name
    except Exception:
        return path.parent.name


def _get_subimg_type(evidence_path: str, type_map: Optional[Dict[str, str]]) -> str:
    if type_map and evidence_path in type_map:
        return str(type_map[evidence_path])
    return "unknown"


def _expand_jsonl(
    file_path: Path,
    base_dir: Path,
    type_map: Optional[Dict[str, str]],
) -> List[dict]:
    results: List[dict] = []
    doc_name = _doc_name_for(file_path, base_dir)
    with file_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            evidence = record.get("evidence", {})
            pages = evidence.get("page", [])
            bbox = evidence.get("bbox")
            if not pages or not isinstance(bbox, list) or len(bbox) != 4:
                continue
            page = pages[0]
            ev_image = evidence.get("image")
            ev_type = evidence.get("type")
            subimg_type = ev_type or _get_subimg_type(ev_image, type_map)
            for qa in record.get("qas", []):
                results.append(
                    {
                        "query": qa.get("q", ""),
                        "answer": qa.get("a", ""),
                        "doc_name": doc_name,
                        "evidence_page": [page],
                        "bbox": [[bbox]],
                        "subimg_type": [[subimg_type]],
                    }
                )
    return results


def transform_jsonl_tree(
    input_dir: Path,
    output_file: Path,
    type_map_path: Path | None = None,
) -> int:
    files = list(_iter_jsonl_files(input_dir))
    if not files:
        print(f"No jsonl files found under {input_dir}")
        return 0

    type_map = _load_type_map(type_map_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with output_file.open("w", encoding="utf-8") as fout:
        for file_path in tqdm(files, desc="Processing"):
            expanded = _expand_jsonl(file_path, input_dir, type_map)
            for item in expanded:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            total += len(expanded)

    print(f"Merged {total} QA record(s). Output written to {output_file}")
    return total
