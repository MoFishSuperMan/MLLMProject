"""Model stack factory for mock and real RAG backends."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .index import FaissVectorIndex, VectorIndex
from .models import MockEmbedder, MockGenerator, MockReranker, MockVisualSummarizer
from .real_models import (
    BGE_M3_MODEL_ID,
    BGE_RERANKER_MODEL_ID,
    QWEN3_VL_MODEL_ID,
    BgeM3Embedder,
    BgeReranker,
    Qwen3VLGenerationConfig,
    Qwen3VLModel,
)


@dataclass(slots=True)
class ModelConfig:
    """Configuration for choosing mock or real model components."""

    use_real_models: bool = False
    vlm_model_id: str = QWEN3_VL_MODEL_ID
    embedding_model_id: str = BGE_M3_MODEL_ID
    reranker_model_id: str = BGE_RERANKER_MODEL_ID
    dtype: str = "bf16"
    device_map: str | None = "auto"
    enable_vlm_summary: bool = True
    vlm_max_new_tokens: int = 512
    vlm_max_images: int = 3
    embedding_device: str | None = None
    reranker_device: str | None = None
    attn_implementation: str | None = None

    @classmethod
    def from_env(cls) -> "ModelConfig":
        return cls(
            use_real_models=parse_bool(os.getenv("MLLMPROJECT_USE_REAL_MODELS"), default=False),
            vlm_model_id=os.getenv("MLLMPROJECT_QWEN3_MODEL_PATH")
            or os.getenv("MLLMPROJECT_VLM_MODEL_ID", QWEN3_VL_MODEL_ID),
            embedding_model_id=os.getenv("MLLMPROJECT_EMBEDDING_MODEL_ID", BGE_M3_MODEL_ID),
            reranker_model_id=os.getenv("MLLMPROJECT_RERANKER_MODEL_ID", BGE_RERANKER_MODEL_ID),
            dtype=os.getenv("MLLMPROJECT_TORCH_DTYPE", "bf16"),
            device_map=none_if_empty(os.getenv("MLLMPROJECT_DEVICE_MAP", "auto")),
            enable_vlm_summary=parse_bool(os.getenv("MLLMPROJECT_ENABLE_VLM_SUMMARY"), default=True),
            vlm_max_new_tokens=int(os.getenv("MLLMPROJECT_VLM_MAX_NEW_TOKENS", "512")),
            vlm_max_images=int(os.getenv("MLLMPROJECT_VLM_MAX_IMAGES", "3")),
            embedding_device=none_if_empty(os.getenv("MLLMPROJECT_EMBEDDING_DEVICE")),
            reranker_device=none_if_empty(os.getenv("MLLMPROJECT_RERANKER_DEVICE")),
            attn_implementation=none_if_empty(os.getenv("MLLMPROJECT_ATTENTION_IMPL")),
        )


class ModelStack:
    """Factory that creates compatible retrieval and generation components."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self._qwen3_vl: Qwen3VLModel | None = None

    @classmethod
    def from_env(cls) -> "ModelStack":
        return cls(ModelConfig.from_env())

    def create_embedder(self):
        if self.config.use_real_models:
            return BgeM3Embedder(
                model_id=self.config.embedding_model_id,
                device=self.config.embedding_device,
            )
        return MockEmbedder()

    def create_index(self, embedder: Any | None = None):
        embedder = embedder or self.create_embedder()
        if self.config.use_real_models:
            return FaissVectorIndex(embedder=embedder)
        return VectorIndex(embedder=embedder)

    def create_reranker(self):
        if self.config.use_real_models:
            return BgeReranker(
                model_id=self.config.reranker_model_id,
                device=self.config.reranker_device,
            )
        return MockReranker()

    def create_generator(self):
        if self.config.use_real_models:
            return self._get_qwen3_vl()
        return MockGenerator()

    def create_visual_summarizer(self):
        if self.config.use_real_models and self.config.enable_vlm_summary:
            return self._get_qwen3_vl()
        return MockVisualSummarizer()

    def _get_qwen3_vl(self) -> Qwen3VLModel:
        if self._qwen3_vl is None:
            self._qwen3_vl = Qwen3VLModel(
                Qwen3VLGenerationConfig(
                    model_id=self.config.vlm_model_id,
                    dtype=self.config.dtype,
                    device_map=self.config.device_map,
                    attn_implementation=self.config.attn_implementation,
                    max_new_tokens=self.config.vlm_max_new_tokens,
                    max_images=self.config.vlm_max_images,
                )
            )
        return self._qwen3_vl


def parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def none_if_empty(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
