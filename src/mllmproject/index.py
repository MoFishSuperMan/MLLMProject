"""Lightweight vector index with a FAISS-free fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .embeddings import HashEmbedding, dot
from .io_utils import read_json, write_json
from .models import MockEmbedder
from .schemas import Chunk, Evidence


class VectorIndex:
    """Small in-memory index used before FAISS is wired in."""

    def __init__(self, embedder: Any | None = None) -> None:
        self.embedder = embedder or MockEmbedder()
        self.chunks: list[Chunk] = []
        self.vectors: Any = []

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        self.vectors = self.embedder.embed_text([chunk.content for chunk in self.chunks])

    def search(self, query: str, top_k: int = 5, source_types: set[str] | None = None) -> list[Evidence]:
        if not self.chunks:
            return []
        query_vector = self.embedder.embed_text([query])[0]
        scored: list[tuple[float, Chunk]] = []
        for chunk, vector in zip(self.chunks, self.vectors):
            if source_types and chunk.source_type not in source_types:
                continue
            scored.append((similarity(query_vector, vector), chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [Evidence.from_chunk(chunk, score=score) for score, chunk in scored[:top_k]]

    def save(self, path: str | Path) -> None:
        payload = {
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "vectors": vectors_to_jsonable(self.vectors),
        }
        write_json(path, payload)

    @classmethod
    def load(cls, path: str | Path, embedder: Any | None = None) -> "VectorIndex":
        payload = read_json(path)
        index = cls(embedder=embedder)
        index.chunks = [Chunk.from_dict(item) for item in payload.get("chunks", [])]
        index.vectors = payload.get("vectors", [])
        return index


class LocalVectorIndex(VectorIndex):
    """Compatibility wrapper used by the text baseline CLI/tests."""

    def __init__(self, embedder: Any | None = None) -> None:
        super().__init__(embedder=embedder or HashEmbedding())

    @classmethod
    def from_chunks(cls, chunks: list[Chunk], embedder: Any | None = None) -> "LocalVectorIndex":
        index = cls(embedder=embedder)
        index.build(chunks)
        return index


SimpleVectorIndex = VectorIndex


class FaissVectorIndex(VectorIndex):
    """FAISS-backed vector index for the real RAG backend."""

    def __init__(self, embedder: Any | None = None) -> None:
        super().__init__(embedder=embedder)
        self.faiss_index: Any | None = None
        self.dim: int | None = None

    def build(self, chunks: list[Chunk]) -> None:
        faiss = require_faiss()
        import numpy as np

        self.chunks = list(chunks)
        if not self.chunks:
            self.vectors = np.empty((0, 0), dtype="float32")
            self.dim = None
            self.faiss_index = None
            return
        vectors = self.embedder.embed_text([chunk.content for chunk in self.chunks])
        array = np.asarray(vectors, dtype="float32")
        if array.ndim != 2:
            raise ValueError("Embedding model must return a 2D array-like value.")
        self.vectors = array
        self.dim = int(array.shape[1])
        self.faiss_index = faiss.IndexFlatIP(self.dim)
        self.faiss_index.add(array)

    def search(self, query: str, top_k: int = 5, source_types: set[str] | None = None) -> list[Evidence]:
        if not self.chunks or self.faiss_index is None:
            return []
        if top_k <= 0:
            return []

        import numpy as np

        query_vector = np.asarray(self.embedder.embed_text([query]), dtype="float32")
        if query_vector.ndim != 2 or query_vector.shape[0] != 1:
            raise ValueError("Embedding model must return exactly one query vector.")

        if source_types:
            scored: list[tuple[float, Chunk]] = []
            for chunk, vector in zip(self.chunks, self.vectors):
                if chunk.source_type not in source_types:
                    continue
                score = float(np.dot(query_vector[0], vector))
                scored.append((score, chunk))
            scored.sort(key=lambda item: item[0], reverse=True)
            return [Evidence.from_chunk(chunk, score=score) for score, chunk in scored[:top_k]]

        scores, indices = self.faiss_index.search(query_vector, min(top_k, len(self.chunks)))
        evidences: list[Evidence] = []
        for score, index in zip(scores[0], indices[0]):
            if int(index) < 0:
                continue
            evidences.append(Evidence.from_chunk(self.chunks[int(index)], score=float(score)))
        return evidences

    def save(self, path: str | Path) -> None:
        payload = {
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "vectors": vectors_to_jsonable(self.vectors),
            "index_type": "faiss_flat_ip",
        }
        write_json(path, payload)

    @classmethod
    def load(cls, path: str | Path, embedder: Any | None = None) -> "FaissVectorIndex":
        faiss = require_faiss()
        import numpy as np

        payload = read_json(path)
        index = cls(embedder=embedder)
        index.chunks = [Chunk.from_dict(item) for item in payload.get("chunks", [])]
        index.vectors = np.asarray(payload.get("vectors", []), dtype="float32")
        if index.vectors.ndim == 2 and index.vectors.shape[0] > 0:
            index.dim = int(index.vectors.shape[1])
            index.faiss_index = faiss.IndexFlatIP(index.dim)
            index.faiss_index.add(index.vectors)
        return index


def require_faiss():
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "FaissVectorIndex requires `faiss`/`faiss-cpu`. "
            "Install the real backend extras before using FAISS retrieval."
        ) from exc
    return faiss


def similarity(left: Any, right: Any) -> float:
    if isinstance(left, dict) and isinstance(right, dict):
        return float(dot(left, right))

    if hasattr(left, "tolist"):
        left = left.tolist()
    if hasattr(right, "tolist"):
        right = right.tolist()

    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    numerator = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = sum(a * a for a in left_values) ** 0.5 or 1.0
    right_norm = sum(b * b for b in right_values) ** 0.5 or 1.0
    return float(numerator / (left_norm * right_norm))


def vectors_to_jsonable(vectors: Any) -> Any:
    if hasattr(vectors, "tolist"):
        return vectors.tolist()
    return vectors
