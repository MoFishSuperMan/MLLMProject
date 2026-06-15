"""Optional real model adapters for the document RAG backend.

The classes in this module import heavy ML dependencies lazily so the default
test suite can run without downloading Qwen/BGE weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from .model_interfaces import AnswerGenerator, EmbeddingModel, RerankerModel, VisionSummaryModel
from .schemas import Citation, Evidence


QWEN3_VL_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
BGE_M3_MODEL_ID = "BAAI/bge-m3"
BGE_RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"


class BgeM3Embedder(EmbeddingModel):
    """BGE-M3 text embedder backed by sentence-transformers."""

    def __init__(
        self,
        model_id: str = BGE_M3_MODEL_ID,
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise ImportError(
                    "BgeM3Embedder requires `sentence-transformers`. "
                    "Install the real backend extras before using BGE-M3."
                ) from exc
            kwargs = {"device": self.device} if self.device else {}
            self._model = SentenceTransformer(self.model_id, **kwargs)
        return self._model

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._load().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype("float32").tolist()


class BgeReranker(RerankerModel):
    """BGE reranker backed by sentence-transformers CrossEncoder."""

    def __init__(
        self,
        model_id: str = BGE_RERANKER_MODEL_ID,
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise ImportError(
                    "BgeReranker requires `sentence-transformers`. "
                    "Install the real backend extras before using BGE reranking."
                ) from exc
            kwargs = {"device": self.device} if self.device else {}
            self._model = CrossEncoder(self.model_id, **kwargs)
        return self._model

    def rerank(self, query: str, evidences: list[Evidence]) -> list[Evidence]:
        if not evidences:
            return []
        pairs = [(query, evidence.content) for evidence in evidences]
        scores = self._load().predict(pairs, batch_size=self.batch_size)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        reranked: list[Evidence] = []
        for evidence, score in zip(evidences, scores):
            evidence.score = float(score)
            reranked.append(evidence)
        return sorted(reranked, key=lambda item: item.score, reverse=True)


@dataclass(slots=True)
class Qwen3VLGenerationConfig:
    model_id: str = QWEN3_VL_MODEL_ID
    dtype: str = "bf16"
    device_map: str | None = "auto"
    attn_implementation: str | None = None
    max_new_tokens: int = 512
    max_images: int = 3


class Qwen3VLModel(AnswerGenerator, VisionSummaryModel):
    """Qwen3-VL adapter for visual summaries and grounded answers."""

    def __init__(self, config: Qwen3VLGenerationConfig | None = None) -> None:
        self.config = config or Qwen3VLGenerationConfig()
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def _load(self) -> tuple[Any, Any, Any]:
        if self._processor is None or self._model is None or self._torch is None:
            torch, processor_cls, model_cls = import_qwen3_vl_dependencies()
            model_kwargs: dict[str, Any] = {"dtype": resolve_torch_dtype(torch, self.config.dtype)}
            if self.config.device_map:
                model_kwargs["device_map"] = self.config.device_map
            if self.config.attn_implementation:
                model_kwargs["attn_implementation"] = self.config.attn_implementation

            self._processor = processor_cls.from_pretrained(self.config.model_id)
            self._model = model_cls.from_pretrained(self.config.model_id, **model_kwargs)
            self._torch = torch
        return self._torch, self._processor, self._model

    def generate_visual_summary(self, image_path: str) -> str:
        prompt = (
            "请用中文简洁概括这页文档截图中的主要内容，重点说明是否包含图表、"
            "表格、公式、页面标题或关键结论。"
        )
        return self._generate([image_message(image_path), text_message(prompt)])

    def generate_answer(
        self,
        query: str,
        evidences: list[Evidence],
        route: str,
        route_reason: str,
    ) -> tuple[str, list[Citation]]:
        if not evidences:
            return "答案：没有检索到足够证据，无法可靠回答。\n来源：[]", []

        content: list[dict[str, str]] = []
        seen_images: set[str] = set()
        for evidence in evidences:
            if evidence.image_path and evidence.image_path not in seen_images and len(seen_images) < self.config.max_images:
                content.append(image_message(evidence.image_path))
                seen_images.add(evidence.image_path)

        content.append(text_message(build_grounded_answer_prompt(query, evidences, route, route_reason)))
        raw_answer = self._generate(content)
        labels = extract_cited_labels(raw_answer)
        citations = citations_from_labels(labels, evidences)
        if not citations:
            citations = stable_citations(evidences, limit=3)
        answer = append_backend_sources(strip_model_sources(raw_answer), citations)
        return answer, citations

    def _generate(self, content: list[dict[str, str]]) -> str:
        torch, processor, model = self._load()
        messages = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_device = resolve_model_device(model)
        if model_device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(model_device)
        input_len = len(inputs.input_ids[0]) if hasattr(inputs, "input_ids") else 0

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=self.config.max_new_tokens, do_sample=False)

        try:
            generated_ids = generated_ids[:, input_len:]
        except Exception:  # pragma: no cover - supports mocked tensor-like outputs
            pass
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
        return str(decoded[0]).strip() if decoded else ""


def import_qwen3_vl_dependencies():
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "Qwen3VLModel requires `torch` and `transformers`. "
            "Install the real backend extras before using Qwen3-VL."
        ) from exc
    return torch, AutoProcessor, AutoModelForImageTextToText


def resolve_torch_dtype(torch: Any, dtype: str):
    normalized = dtype.strip().lower()
    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    if normalized in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype}")


def resolve_model_device(model: Any):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except Exception:
        return None


def image_message(image_path: str) -> dict[str, str]:
    return {"type": "image", "image": str(Path(image_path))}


def text_message(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def build_grounded_answer_prompt(
    query: str,
    evidences: list[Evidence],
    route: str,
    route_reason: str,
) -> str:
    evidence_lines = []
    for index, evidence in enumerate(evidences, start=1):
        metadata_notes = evidence_prompt_metadata(evidence)
        evidence_lines.append(
            "\n".join(
                [
                    f"[E{index}] page={evidence.page}, type={evidence.source_type}, "
                    f"chunk={evidence.chunk_id or evidence.evidence_id}, score={evidence.score:.4f}",
                    metadata_notes,
                    compact(evidence.content, 900),
                ]
            )
        )
    chart_instruction = ""
    if any(evidence.source_type == "chart_region" for evidence in evidences):
        chart_instruction = (
            "\n图表题要求：先识别图表中的类别、数值、坐标轴或图例；"
            "如果 chart_question_type=count，只数题目要求的类别、柱子或可见 item，"
            "不要把年份、坐标轴刻度、数值标签、标题或图例文字算进去；"
            "如果是 count/difference/min/max/value 题，Final answer 只能写数字；"
            "如果是 Yes/No 比较题，Final answer 只能写 Yes 或 No。"
        )
    target_instruction = ""
    if any((evidence.metadata or {}).get("target_value") for evidence in evidences):
        target_instruction = (
            "\n局部读数要求：如果证据标出 target_value，请围绕该目标年份或目标数值，"
            "结合左侧/下方坐标轴和对应曲线、柱子位置读数；不要只抄最近的坐标轴刻度。"
            "Final answer 只写读出的短数值或实体。"
        )
    return (
        "你是一个多模态文档问答后端。请只依据给定证据回答，不能编造。"
        "如果证据不足，请明确说明。\n"
        "引用证据时只能使用 [E1]、[E2] 这样的证据编号。"
        "不要自己编写 page 或 chunk，也不要输出“来源：”。\n"
        f"问题：{query}\n"
        f"路由：{route}\n"
        f"路由原因：{route_reason}\n"
        "证据：\n"
        + "\n\n".join(evidence_lines)
        + chart_instruction
        + target_instruction
        + "\n\n请输出中文答案，并在相关句子后使用证据编号，例如 [E1]。"
        "最后一行必须严格使用格式：Final answer: <short answer>。"
    )


def evidence_prompt_metadata(evidence: Evidence) -> str:
    metadata = evidence.metadata or {}
    parts = []
    if metadata.get("page_bbox"):
        parts.append(f"page_bbox={metadata['page_bbox']}")
    if metadata.get("chart_question_type"):
        parts.append(f"chart_question_type={metadata['chart_question_type']}")
    if metadata.get("chart_numbers"):
        parts.append("chart_numbers=" + ", ".join(str(value) for value in metadata["chart_numbers"][:20]))
    if metadata.get("chart_labels"):
        parts.append("chart_labels=" + ", ".join(str(value) for value in metadata["chart_labels"][:20]))
    if metadata.get("target_value"):
        parts.append(f"target_value={metadata['target_value']}")
    return "; ".join(parts)


def stable_citations(evidences: list[Evidence], limit: int = 3) -> list[Citation]:
    citations: list[Citation] = []
    for evidence in evidences[:limit]:
        citations.append(
            Citation(
                page=evidence.page,
                source_type=evidence.source_type,
                chunk_id=evidence.chunk_id,
                bbox=evidence.bbox,
                region_id=evidence.region_id,
                evidence_id=evidence.evidence_id,
            )
        )
    return citations


def extract_cited_labels(answer: str) -> list[int]:
    labels: list[int] = []
    for match in re.finditer(r"\[\s*E\s*(\d+)\s*\]", answer):
        label = int(match.group(1))
        if label not in labels:
            labels.append(label)
    return labels


def citations_from_labels(labels: list[int], evidences: list[Evidence]) -> list[Citation]:
    citations: list[Citation] = []
    for label in labels:
        index = label - 1
        if index < 0 or index >= len(evidences):
            continue
        evidence = evidences[index]
        citations.append(
            Citation(
                page=evidence.page,
                source_type=evidence.source_type,
                chunk_id=evidence.chunk_id,
                bbox=evidence.bbox,
                region_id=evidence.region_id,
                evidence_id=evidence.evidence_id,
            )
        )
    return citations


def strip_model_sources(answer: str) -> str:
    stripped = re.split(r"\n?\s*来源[:：]", answer, maxsplit=1)[0].strip()
    return stripped


def append_backend_sources(answer: str, citations: list[Citation]) -> str:
    if not citations:
        return f"{answer}\n来源：[]"
    parts = []
    for citation in citations:
        chunk = citation.chunk_id or citation.evidence_id or "-"
        parts.append(f"[page={citation.page}, chunk={chunk}]")
    return f"{answer}\n来源：" + "; ".join(parts)


def compact(text: str, max_len: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3] + "..."
