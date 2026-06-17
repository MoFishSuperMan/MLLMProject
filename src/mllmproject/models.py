"""Mock model implementations that can later be replaced by BGE/Qwen."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

from .model_interfaces import AnswerGenerator, EmbeddingModel, RerankerModel, VisionSummaryModel
from .schemas import Citation, Evidence


class MockEmbedder(EmbeddingModel):
    """Deterministic hashing embedder for local smoke tests.

    It is not semantically strong, but it lets the retrieval/evaluation pipeline
    run before large model downloads are ready.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class MockReranker(RerankerModel):
    """Lexical reranker used until BGE-reranker is wired in."""

    def rerank(self, query: str, evidences: list[Evidence]) -> list[Evidence]:
        query_tokens = set(tokenize(query))
        scored: list[Evidence] = []
        for evidence in evidences:
            content_tokens = set(tokenize(evidence.content))
            overlap = len(query_tokens & content_tokens)
            lexical_score = overlap / max(len(query_tokens), 1)
            evidence.score = evidence.score + lexical_score
            scored.append(evidence)
        return sorted(scored, key=lambda item: item.score, reverse=True)


class MockGenerator(AnswerGenerator):
    """Simple answer generator with mandatory citations."""

    def generate_answer(self, query: str, evidences: list[Evidence], route: str, route_reason: str) -> tuple[str, list[Citation]]:
        if not evidences:
            return "答案：未检索到足够证据，无法回答。\n来源：[]", []

        top = evidences[: min(4, len(evidences))]
        snippets = [
            f"{index}. {compact_text(evidence.content, max_len=420)}"
            for index, evidence in enumerate(top, start=1)
        ]
        answer = "答案：根据检索到的证据，可以整理如下：\n" + "\n".join(snippets)
        answer += "\n来源：" + "; ".join(
            f"[page={evidence.page}, chunk={evidence.chunk_id or evidence.evidence_id}]"
            for evidence in top
        )
        citations = [
            Citation(
                page=evidence.page,
                source_type=evidence.source_type,
                chunk_id=evidence.chunk_id,
                bbox=evidence.bbox,
                evidence_id=evidence.evidence_id,
            )
            for evidence in top
        ]
        return answer, citations


class MockVisualSummarizer(VisionSummaryModel):
    """Page-level visual summary stub for multimodal RAG development."""

    def generate_visual_summary(self, image_path: str) -> str:
        name = Path(image_path).stem.replace("_", " ")
        return f"{name} 的页面截图，可能包含文档版面、图表、表格或图片区域，可作为视觉检索证据。"


def tokenize(text: str) -> list[str]:
    text = text.lower()
    latin = re.findall(r"[a-z0-9_]+", text)
    chinese = re.findall(r"[\u4e00-\u9fff]", text)
    return latin + chinese


def compact_text(text: str, max_len: int = 160) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3] + "..."
