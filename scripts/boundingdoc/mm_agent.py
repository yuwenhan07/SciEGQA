from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from modelscope import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from vllm import LLM, SamplingParams


@contextmanager
def _cuda_visible_scope(devices: Optional[List[int]]):
    if not devices:
        yield
        return
    joined = ",".join(str(d) for d in devices)
    prev = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = joined
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev


class MMAgent:
    """Thin wrapper for Qwen2.5-VL that supports both HF and vLLM backends."""

    def __init__(
        self,
        model_name: str,
        use_vllm: bool = False,
        gpu_devices: Optional[List[int]] = None,
        trust_remote_code: bool = True,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
        max_new_tokens: int = 1024,
    ):
        self.use_vllm = use_vllm
        self.model_name = model_name
        self.gpu_devices = gpu_devices
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            trust_remote_code=trust_remote_code,
        )

        if use_vllm:
            tp = len(gpu_devices) if gpu_devices else None
            with _cuda_visible_scope(gpu_devices):
                visible_count = torch.cuda.device_count() if tp is None else tp
                if visible_count < 1:
                    raise RuntimeError("vLLM backend requires at least one CUDA device.")
                self.llm = LLM(
                    model=model_name,
                    tensor_parallel_size=visible_count,
                    trust_remote_code=trust_remote_code,
                )
            self.model = None
        else:
            device_map = self._determine_device_map()
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype="auto",
                trust_remote_code=trust_remote_code,
            )
            self.model.eval()
            self.llm = None

    def _determine_device_map(self) -> Union[str, Dict[str, str]]:
        if not self.gpu_devices or len(self.gpu_devices) != 1:
            return "auto"
        return {"": f"cuda:{self.gpu_devices[0]}"}

    def _build_text_and_images(
        self,
        messages: List[List[Dict]],
    ) -> Tuple[List[str], List[List[Image.Image]]]:
        texts: List[str] = []
        images: List[List[Image.Image]] = []
        for convo in messages:
            text = self.processor.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=True
            )
            img_list: List[Image.Image] = []
            for turn in convo:
                for chunk in turn.get("content", []):
                    if chunk.get("type") != "image":
                        continue
                    source = chunk.get("image")
                    if source is None:
                        continue
                    if isinstance(source, Image.Image):
                        img = source.convert("RGB")
                        width, height = img.size
                    else:
                        try:
                            with Image.open(source) as pil_img:
                                pil_img.load()
                                width, height = pil_img.size
                                if width == 0 or height == 0:
                                    raise ValueError(f"Invalid empty image: {source}")
                                img = pil_img.convert("RGB")
                        except ZeroDivisionError as exc:
                            raise ValueError(f"Invalid PNG (division by zero) when loading {source}") from exc
                        except Exception as exc:
                            raise ValueError(f"Failed to load image {source}: {exc}") from exc
                    if width == 0 or height == 0:
                        raise ValueError(f"Invalid empty image: {source}")
                    img_list.append(img)
            images.append(img_list)
            texts.append(text)
        return texts, images

    def _prepare_hf_inputs(self, texts: List[str], images: List[List[Image.Image]]):
        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            target = (
                f"cuda:{self.gpu_devices[0]}"
                if self.gpu_devices and len(self.gpu_devices) >= 1
                else "cuda"
            )
            inputs = {k: v.to(target) for k, v in inputs.items()}
        return inputs

    def _prepare_vllm_requests(
        self, texts: List[str], images: List[List[Image.Image]]
    ) -> List[Dict]:
        reqs: List[Dict] = []
        for text, img_list in zip(texts, images):
            reqs.append(
                {
                    "prompt": text,
                    "multi_modal_data": {"image": img_list},
                }
            )
        return reqs

    def generate(
        self,
        messages: List[List[Dict]],
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.0,
        **kwargs,
    ) -> List[str]:
        if not isinstance(messages, list) or (messages and not isinstance(messages[0], list)):
            raise ValueError("`messages` must be List[List[Dict]]")

        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        texts, images = self._build_text_and_images(messages)

        if self.use_vllm:
            sampling_params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )
            outputs = self.llm.generate(
                self._prepare_vllm_requests(texts, images),
                sampling_params,
                use_tqdm=False,
            )
            return [o.outputs[0].text.strip() if o.outputs else "" for o in outputs]

        inputs = self._prepare_hf_inputs(texts, images)
        with torch.inference_mode():
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
        input_ids = inputs["input_ids"]
        results: List[str] = []
        for idx in range(gen_ids.size(0)):
            prompt_len = input_ids[idx].size(0)
            out_ids = gen_ids[idx][prompt_len:]
            text = self.processor.batch_decode(
                out_ids.unsqueeze(0).to("cpu"),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            results.append(text.strip())
        return results
