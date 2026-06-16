"""Model stack factory for mock and real RAG backends."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .api_models import DashScopeQwenConfig, DashScopeQwenModel
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

    backend: str = "mock"
    use_real_models: bool = False
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_chat_model: str = "qwen-plus"
    dashscope_vision_model: str = "qwen-vl-plus"
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
        backend = os.getenv("MLLMPROJECT_MODEL_BACKEND") or os.getenv("MLLMPROJECT_BACKEND") or "mock"
        return cls(
            backend=backend.strip().lower(),
            use_real_models=parse_bool(os.getenv("MLLMPROJECT_USE_REAL_MODELS"), default=False),
            dashscope_api_key=os.getenv("MLLMPROJECT_DASHSCOPE_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY", ""),
            dashscope_base_url=os.getenv(
                "MLLMPROJECT_DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            dashscope_chat_model=os.getenv("MLLMPROJECT_DASHSCOPE_CHAT_MODEL", "qwen-plus"),
            dashscope_vision_model=os.getenv("MLLMPROJECT_DASHSCOPE_VISION_MODEL", "qwen-vl-plus"),
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
        self._dashscope_qwen: DashScopeQwenModel | None = None

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
        if self.config.backend == "dashscope":
            return self._get_dashscope_qwen()
        if self.config.use_real_models:
            return self._get_qwen3_vl()
        return MockGenerator()

    def create_visual_summarizer(self):
        if self.config.backend == "dashscope":
            return self._get_dashscope_qwen()
        if self.config.use_real_models and self.config.enable_vlm_summary:
            return self._get_qwen3_vl()
        return MockVisualSummarizer()

    def _get_dashscope_qwen(self) -> DashScopeQwenModel:
        if self._dashscope_qwen is None:
            self._dashscope_qwen = DashScopeQwenModel(
                DashScopeQwenConfig(
                    api_key=self.config.dashscope_api_key,
                    base_url=self.config.dashscope_base_url,
                    chat_model=self.config.dashscope_chat_model,
                    vision_model=self.config.dashscope_vision_model,
                    max_tokens=self.config.vlm_max_new_tokens,
                    max_images=self.config.vlm_max_images,
                )
            )
        return self._dashscope_qwen

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
