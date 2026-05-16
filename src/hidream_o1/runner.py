from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from hidream_o1.model_cache import resolve_model_path
from hidream_o1.schemas import GenerationRequest


DEFAULT_MODEL_ID = "HiDream-ai/HiDream-O1-Image"
DEFAULT_HF_CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"
SUPPORTED_ATTENTION_BACKENDS = {"auto", "sdpa", "flash"}


class HiDreamRunner:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        model_path: str | Path | None = None,
        hf_cache_root: str | Path = DEFAULT_HF_CACHE_ROOT,
        attention_backend: str = "sdpa",
    ):
        self.model_id = model_id
        self.model_path = Path(model_path) if model_path else None
        self.hf_cache_root = Path(hf_cache_root)
        attention_backend = attention_backend.lower()
        if attention_backend not in SUPPORTED_ATTENTION_BACKENDS:
            raise ValueError("ATTENTION_BACKEND must be one of: auto, flash, sdpa")
        if attention_backend == "auto":
            attention_backend = "sdpa"
        self.attention_backend = attention_backend
        self.use_flash_attn = attention_backend == "flash"
        self._lock = threading.Lock()
        self._processor = None
        self._model = None
        self._resolved_model_path: Path | None = None

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "HiDreamRunner":
        env = env or os.environ
        return cls(
            model_id=env.get("HIDREAM_MODEL_ID", DEFAULT_MODEL_ID),
            model_path=env.get("HIDREAM_MODEL_PATH") or None,
            hf_cache_root=env.get("HIDREAM_HF_CACHE_ROOT", DEFAULT_HF_CACHE_ROOT),
            attention_backend=env.get("ATTENTION_BACKEND", "sdpa").lower(),
        )

    @property
    def resolved_model_path(self) -> Path:
        if self._resolved_model_path is None:
            self._resolved_model_path = resolve_model_path(
                self.model_id,
                self.hf_cache_root,
                self.model_path,
            )
        return self._resolved_model_path

    def generate(self, request: GenerationRequest, ref_image_paths: list[str]):
        processor, model = self._load()
        kwargs = request.generation_kwargs()
        if kwargs.get("timesteps_list") == "DEFAULT_TIMESTEPS":
            from hidream_o1.vendor.models.pipeline import DEFAULT_TIMESTEPS

            kwargs["timesteps_list"] = DEFAULT_TIMESTEPS

        from hidream_o1.vendor.models.pipeline import generate_image

        layout_bboxes = request.layout_bboxes
        if isinstance(layout_bboxes, (dict, list)):
            layout_bboxes = json.dumps(layout_bboxes)

        return generate_image(
            model=model,
            processor=processor,
            prompt=request.prompt,
            ref_image_paths=ref_image_paths,
            height=request.height,
            width=request.width,
            seed=request.seed,
            keep_original_aspect=request.keep_original_aspect,
            layout_bboxes=layout_bboxes,
            use_flash_attn=self.use_flash_attn,
            **kwargs,
        )

    def _load(self):
        if self._processor is not None and self._model is not None:
            return self._processor, self._model

        with self._lock:
            if self._processor is not None and self._model is not None:
                return self._processor, self._model

            import torch
            from transformers import AutoProcessor

            from hidream_o1.vendor.models.qwen3_vl_transformers import (
                Qwen3VLForConditionalGeneration,
            )

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for HiDream O1 inference")

            model_path = str(self.resolved_model_path)
            processor = AutoProcessor.from_pretrained(model_path)
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
            ).eval()
            _add_special_tokens(_get_tokenizer(processor))

            self._processor = processor
            self._model = model
            return processor, model


def _get_tokenizer(processor: Any):
    tokenizer = getattr(processor, "tokenizer", None)
    return tokenizer if tokenizer is not None else processor


def _add_special_tokens(tokenizer: Any) -> None:
    tokenizer.boi_token = "<|boi_token|>"
    tokenizer.bor_token = "<|bor_token|>"
    tokenizer.eor_token = "<|eor_token|>"
    tokenizer.bot_token = "<|bot_token|>"
    tokenizer.tms_token = "<|tms_token|>"
