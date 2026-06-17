"""Unified text-image embeddings for multimodal retrieval."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

from .models import MockEmbedder, tokenize
from .schemas import Chunk


class ImageFeatureEmbedder:
    """Native image feature encoder used when no heavy ViT weights are loaded."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed_image(self, image_path: str, dim: int | None = None) -> list[float]:
        target_dim = dim or self.dim
        vector = [0.0] * target_dim
        path = Path(image_path)
        visual_tokens = ["image", "visual", "figure", path.stem.lower()]
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB").resize((32, 32))
                stat = ImageStat.Stat(rgb)
                means = [value / 255.0 for value in stat.mean]
                extrema = rgb.getextrema()
                width, height = image.size
                visual_tokens.extend(
                    [
                        f"aspect_{round(width / max(height, 1), 1)}",
                        "wide" if width >= height else "tall",
                        dominant_color_name(means),
                    ]
                )
                for index, value in enumerate(means):
                    vector[index % target_dim] += value
                for index, channel in enumerate(extrema):
                    span = (channel[1] - channel[0]) / 255.0
                    vector[(index + 8) % target_dim] += span
        except Exception:
            visual_tokens.append("missing_image")

        for token in visual_tokens:
            add_hash_token(vector, token)
        return normalize(vector)


class NativeVitImageEmbedder:
    """Optional ViT/CLIP-style image encoder loaded lazily from transformers."""

    def __init__(self, model_id: str, device: str | None = None) -> None:
        self.model_id = model_id
        self.device = device
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def _load(self) -> tuple[Any, Any, Any]:
        if self._processor is None or self._model is None or self._torch is None:
            try:
                import torch
                from transformers import AutoImageProcessor, AutoModel
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise ImportError("NativeVitImageEmbedder requires torch and transformers.") from exc
            self._processor = AutoImageProcessor.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(self.model_id)
            if self.device and hasattr(self._model, "to"):
                self._model = self._model.to(self.device)
            self._model.eval()
            self._torch = torch
        return self._torch, self._processor, self._model

    def embed_image(self, image_path: str, dim: int | None = None) -> list[float]:
        torch, processor, model = self._load()
        with Image.open(image_path) as image:
            inputs = processor(images=image.convert("RGB"), return_tensors="pt")
        if self.device:
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        vector = pooled[0].detach().float().cpu().tolist()
        if dim and len(vector) != dim:
            vector = resize_vector(vector, dim)
        return normalize(vector)


class UnifiedMultimodalEmbedder:
    """Embeds text queries and text/image chunks into one vector space."""

    def __init__(
        self,
        text_embedder: Any | None = None,
        image_embedder: Any | None = None,
        image_weight: float = 0.38,
    ) -> None:
        self.text_embedder = text_embedder or MockEmbedder()
        self.image_embedder = image_embedder or ImageFeatureEmbedder()
        self.image_weight = image_weight

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        return self.text_embedder.embed_text(texts)

    def embed_query(self, query: str) -> list[float]:
        return self.embed_text([query])[0]

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        text_vectors = self.embed_text([chunk.content for chunk in chunks])
        vectors: list[list[float]] = []
        for chunk, text_vector in zip(chunks, text_vectors):
            vector = list(text_vector)
            modalities = ["text"]
            if chunk.image_path:
                image_vector = self.image_embedder.embed_image(chunk.image_path, dim=len(vector))
                vector = fuse_vectors(vector, image_vector, self.image_weight)
                modalities.append("image")
            chunk.metadata = {
                **(chunk.metadata or {}),
                "embedding_modalities": modalities,
                "embedding_space": "unified_text_image",
            }
            vectors.append(normalize(vector))
        return vectors


def fuse_vectors(text_vector: list[float], image_vector: list[float], image_weight: float) -> list[float]:
    text_weight = 1.0 - image_weight
    return [
        text_weight * float(left) + image_weight * float(right)
        for left, right in zip(text_vector, image_vector)
    ]


def add_hash_token(vector: list[float], token: str) -> None:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "little") % len(vector)
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    vector[index] += sign
    for subtoken in tokenize(token):
        sub_digest = hashlib.sha256(subtoken.encode("utf-8")).digest()
        sub_index = int.from_bytes(sub_digest[:4], "little") % len(vector)
        vector[sub_index] += 0.5


def dominant_color_name(means: list[float]) -> str:
    labels = ("red", "green", "blue")
    return labels[max(range(len(means)), key=lambda index: means[index])] if means else "neutral"


def resize_vector(vector: list[float], dim: int) -> list[float]:
    if dim <= 0:
        return []
    resized = [0.0] * dim
    for index, value in enumerate(vector):
        resized[index % dim] += float(value)
    return resized


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in vector)) or 1.0
    return [float(value) / norm for value in vector]

