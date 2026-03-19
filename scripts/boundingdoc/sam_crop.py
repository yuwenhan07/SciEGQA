from __future__ import annotations

import json
import random
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from tqdm import tqdm


@dataclass(slots=True)
class SamCropConfig:
    """Configuration for running SAM based cropping."""

    checkpoint: str
    device: str = "cuda"
    pad_px: int = 10
    area_min_ratio: float = 0.05
    area_max_ratio: float = 0.70
    model_id: str = "default"
    devices: tuple[str, ...] | None = None
    num_workers: int | None = None
    queue_size: int = 32
    max_pages_per_doc: int | None = 5

    def __post_init__(self) -> None:
        if self.devices is not None:
            if isinstance(self.devices, str):
                devices_tuple = (self.devices,)
            else:
                devices_tuple = tuple(str(d) for d in self.devices)
            object.__setattr__(self, "devices", devices_tuple)
            if self.device == "cuda" and devices_tuple:
                object.__setattr__(self, "device", devices_tuple[0])
        if self.queue_size < 1:
            object.__setattr__(self, "queue_size", 1)
        if self.max_pages_per_doc is not None and self.max_pages_per_doc <= 0:
            object.__setattr__(self, "max_pages_per_doc", None)


@dataclass(slots=True)
class CropCandidate:
    mask_id: int
    bbox_xywh: Tuple[int, int, int, int]
    bbox_xyxy: Tuple[int, int, int, int]
    area: int
    predicted_iou: float
    stability_score: float
    ratio: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.mask_id,
            "bbox_xywh": list(self.bbox_xywh),
            "bbox_xyxy": list(self.bbox_xyxy),
            "area": int(self.area),
            "predicted_iou": float(self.predicted_iou),
            "stability_score": float(self.stability_score),
            "ratio": float(self.ratio),
        }


@dataclass(slots=True)
class SamStats:
    processed_sets: int = 0
    processed_images: int = 0
    succeeded_images: int = 0


def _select_subset(items: Sequence[Path], limit: int | None) -> List[Path]:
    """Return up to `limit` items while keeping their original ordering."""

    if limit is None or limit <= 0 or len(items) <= limit:
        return list(items)
    indices = sorted(random.sample(range(len(items)), limit))
    return [items[i] for i in indices]


class SamCropper:
    """Wraps SAM mask generation and keeps a clean interface for the CLI."""

    def __init__(self, config: SamCropConfig):
        self.config = config
        model_ctor = sam_model_registry[config.model_id]
        self._model = model_ctor(checkpoint=config.checkpoint)
        self._model.to(config.device)
        self._mask_generator = SamAutomaticMaskGenerator(self._model)

    @staticmethod
    def _numeric_key(path: Path) -> Tuple[int, str]:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        try:
            return (int(digits), path.stem)
        except ValueError:
            return (10**9, path.stem)

    @staticmethod
    def _iter_sets(root_dir: Path) -> Iterable[Tuple[str, Path]]:
        root_dir = root_dir.resolve()
        pngs_at_root = list(root_dir.glob("*.png"))
        if pngs_at_root:
            yield root_dir.name, root_dir
        for entry in sorted(root_dir.iterdir(), key=lambda p: p.name):
            if entry.is_dir():
                yield entry.name, entry

    def _load_image(self, image_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return bgr, rgb

    def _keep_candidate(self, bbox_xywh: Sequence[int], page_size: Tuple[int, int]) -> Tuple[bool, float]:
        px, py, pw, ph = bbox_xywh
        pw = max(0, int(pw))
        ph = max(0, int(ph))
        width, height = page_size
        total_area = max(1, width * height)
        ratio = (pw * ph) / total_area
        keep = self.config.area_min_ratio <= ratio <= self.config.area_max_ratio
        return keep, ratio

    def _to_xyxy(self, bbox_xywh: Sequence[int], limits: Tuple[int, int]) -> Tuple[int, int, int, int]:
        x, y, w, h = map(int, bbox_xywh)
        width, height = limits
        x1 = max(0, min(width, x))
        y1 = max(0, min(height, y))
        x2 = max(0, min(width, x + w))
        y2 = max(0, min(height, y + h))
        return x1, y1, x2, y2

    def _save_crop(
        self,
        image: np.ndarray,
        bbox_xywh: Sequence[int],
        page_size: Tuple[int, int],
        dest_dir: Path,
        name: str,
    ) -> bool:
        width, height = page_size
        x, y, w, h = map(int, bbox_xywh)
        x1 = max(0, min(width, x - self.config.pad_px))
        y1 = max(0, min(height, y - self.config.pad_px))
        x2 = max(0, min(width, x + w + self.config.pad_px))
        y2 = max(0, min(height, y + h + self.config.pad_px))
        if x2 <= x1 or y2 <= y1:
            return False
        crop = image[y1:y2, x1:x2]
        dest_path = dest_dir / name
        if not cv2.imwrite(str(dest_path), crop):
            return False
        try:
            with Image.open(dest_path) as pil_img:
                pil_img.verify()
        except Exception:
            dest_path.unlink(missing_ok=True)
            return False
        return True

    def process_image(self, image_path: Path, output_root: Path, set_name: str) -> Tuple[bool, str]:
        try:
            bgr, rgb = self._load_image(image_path)
        except FileNotFoundError as exc:
            return False, str(exc)

        masks = self._mask_generator.generate(rgb)
        page_h, page_w = bgr.shape[:2]
        base_name = image_path.stem
        image_out_root = output_root / set_name / base_name
        crop_dir = image_out_root / "crops"
        crop_dir.mkdir(parents=True, exist_ok=True)

        candidates: List[CropCandidate] = []
        for idx, mask in enumerate(masks, start=1):
            bbox_xywh = tuple(map(int, mask.get("bbox", (0, 0, page_w, page_h))))
            bbox_xyxy = self._to_xyxy(bbox_xywh, (page_w, page_h))
            keep, ratio = self._keep_candidate(bbox_xywh, (page_w, page_h))
            candidates.append(
                CropCandidate(
                    mask_id=idx,
                    bbox_xywh=bbox_xywh,
                    bbox_xyxy=bbox_xyxy,
                    area=int(mask.get("area", int(mask["segmentation"].astype(bool).sum()))),
                    predicted_iou=float(mask.get("predicted_iou", -1.0)),
                    stability_score=float(mask.get("stability_score", -1.0)),
                    ratio=ratio,
                )
            )

        kept = [c for c in candidates if self.config.area_min_ratio <= c.ratio <= self.config.area_max_ratio]
        saved = 0
        for cand in kept:
            filename = f"{base_name}_final_{cand.mask_id:03d}.png"
            if self._save_crop(bgr, cand.bbox_xywh, (page_w, page_h), crop_dir, filename):
                saved += 1

        filtered_json = {
            "image_path": str(image_path),
            "image_size": {"height": page_h, "width": page_w},
            "criteria": {
                "min_ratio": self.config.area_min_ratio,
                "max_ratio": self.config.area_max_ratio,
                "pad_px": self.config.pad_px,
            },
            "kept_ids": [c.mask_id for c in kept],
            "ratios": {str(c.mask_id): c.ratio for c in kept},
            "masks": [c.to_dict() for c in kept],
        }
        filtered_path = image_out_root / f"{base_name}_filtered.json"
        with filtered_path.open("w", encoding="utf-8") as f:
            json.dump(filtered_json, f, ensure_ascii=False, indent=2)

        return True, f"{base_name}: kept {len(kept)} segment(s), saved {saved} crop(s)"

    def run(self, input_dir: Path, output_root: Path) -> SamStats:
        stats = SamStats()
        output_root = output_root.resolve()

        for set_name, set_dir in tqdm(list(self._iter_sets(input_dir)), desc="Sets", unit="set"):
            pngs = sorted(set_dir.glob("*.png"), key=self._numeric_key)
            selected_pngs = _select_subset(pngs, self.config.max_pages_per_doc)
            if not selected_pngs:
                continue
            stats.processed_sets += 1
            stats.processed_images += len(selected_pngs)

            for img_path in tqdm(selected_pngs, desc=f"Processing {set_name}", leave=False, unit="img"):
                ok, msg = self.process_image(img_path, output_root, set_name)
                if ok:
                    stats.succeeded_images += 1
                else:
                    print(msg)
            print(f"[{set_name}] Output directory: {output_root / set_name}")


def _config_to_kwargs(config: SamCropConfig) -> Dict[str, object]:
    return {
        "checkpoint": config.checkpoint,
        "device": config.device,
        "pad_px": config.pad_px,
        "area_min_ratio": config.area_min_ratio,
        "area_max_ratio": config.area_max_ratio,
        "model_id": config.model_id,
        "max_pages_per_doc": config.max_pages_per_doc,
    }


def _sam_worker_main(
    config_kwargs: Dict[str, object],
    device: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    """Worker process loop that consumes images and runs SAM cropping."""

    config_dict = dict(config_kwargs)
    config_dict["device"] = device
    cropper = SamCropper(SamCropConfig(**config_dict))

    while True:
        task = task_queue.get()
        if task is None:
            break
        idx, image_path, output_root, set_name = task
        try:
            ok, msg = cropper.process_image(Path(image_path), Path(output_root), set_name)
        except Exception as exc:  # pragma: no cover - defensive safety
            ok = False
            msg = f"{image_path}: worker failure → {exc}"
        result_queue.put((idx, ok, msg, set_name))


def run_parallel_sam(
    config: SamCropConfig,
    input_dir: Path,
    output_root: Path,
    devices: Sequence[str] | None = None,
    num_workers: int | None = None,
    queue_size: int | None = None,
) -> SamStats:
    """Run SAM cropping in parallel across multiple worker processes."""

    base_devices: Sequence[str] | None = devices or config.devices or (config.device,)
    devices = [str(d) for d in base_devices if str(d).strip()]
    if not devices:
        raise ValueError("At least one device must be provided for parallel SAM processing")

    if queue_size is None:
        queue_size = config.queue_size
    queue_size = max(1, queue_size)

    ctx = mp.get_context("spawn")
    task_queue: mp.Queue = ctx.Queue(maxsize=queue_size)
    result_queue: mp.Queue = ctx.Queue()

    input_dir = Path(input_dir).resolve()
    output_root = Path(output_root).resolve()

    stats = SamStats()
    tasks: List[Tuple[str, Path]] = []
    for set_name, set_dir in SamCropper._iter_sets(input_dir):
        pngs = sorted(set_dir.glob("*.png"), key=SamCropper._numeric_key)
        selected_pngs = _select_subset(pngs, config.max_pages_per_doc)
        if not selected_pngs:
            continue
        stats.processed_sets += 1
        for img_path in selected_pngs:
            stats.processed_images += 1
            tasks.append((set_name, img_path))

    if not tasks:
        return stats

    if num_workers is None:
        num_workers = config.num_workers or len(devices)
    num_workers = max(1, num_workers)
    worker_devices = [devices[i % len(devices)] for i in range(num_workers)]

    config_kwargs = _config_to_kwargs(config)

    workers: List[mp.Process] = []
    for device in worker_devices:
        process = ctx.Process(
            target=_sam_worker_main,
            args=(config_kwargs, device, task_queue, result_queue),
        )
        process.start()
        workers.append(process)

    for idx, (set_name, img_path) in enumerate(tasks):
        task_queue.put((idx, str(img_path), str(output_root), set_name))

    for _ in workers:
        task_queue.put(None)

    completed = 0
    per_set_remaining: Dict[str, int] = {}
    for set_name, _ in tasks:
        per_set_remaining[set_name] = per_set_remaining.get(set_name, 0) + 1

    progress = tqdm(total=len(tasks), desc="SAM images", unit="img")

    try:
        while completed < len(tasks):
            idx, ok, msg, set_name = result_queue.get()
            if ok:
                stats.succeeded_images += 1
            else:
                print(msg)
            remaining = per_set_remaining.get(set_name, 0) - 1
            if remaining <= 0:
                print(f"[{set_name}] Output directory: {output_root / set_name}")
                per_set_remaining.pop(set_name, None)
            else:
                per_set_remaining[set_name] = remaining
            completed += 1
            progress.update(1)
    finally:
        progress.close()
        for process in workers:
            process.join(timeout=5)
        for process in workers:
            if process.is_alive():
                process.terminate()
        for process in workers:
            process.join()
        task_queue.close()
        task_queue.join_thread()
        result_queue.close()
        result_queue.join_thread()

    return stats
