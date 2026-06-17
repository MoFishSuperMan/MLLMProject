"""API-backed model adapters for cloud-hosted Qwen models."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from .model_interfaces import AnswerGenerator, VisionSummaryModel
from .real_models import (
    append_backend_sources,
    citations_from_labels,
    compact,
    evidence_prompt_metadata,
    extract_cited_labels,
    stable_citations,
    strip_model_sources,
)
from .schemas import Citation, Evidence


DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_CHAT_MODEL = "qwen3.5-omni-flash"
DASHSCOPE_VISION_MODEL = "qwen3.5-omni-flash"


@dataclass(slots=True)
class DashScopeQwenConfig:
    api_key: str
    base_url: str = DASHSCOPE_BASE_URL
    chat_model: str = DASHSCOPE_CHAT_MODEL
    vision_model: str = DASHSCOPE_VISION_MODEL
    max_tokens: int = 1024
    temperature: float = 0.1
    max_images: int = 3


class DashScopeQwenModel(AnswerGenerator, VisionSummaryModel):
    """Qwen API adapter using DashScope's OpenAI-compatible endpoint."""

    def __init__(self, config: DashScopeQwenConfig) -> None:
        if not config.api_key:
            raise ValueError("DASHSCOPE_API_KEY or MLLMPROJECT_DASHSCOPE_API_KEY is required.")
        self.config = config
        self._client: Any | None = None

    def generate_visual_summary(self, image_path: str) -> str:
        prompt = (
            "请用中文简洁概括这页文档截图中的主要内容，重点说明是否包含图表、"
            "表格、公式、页面标题或关键结论。"
        )
        return self._chat(
            model=self.config.vision_model,
            messages=[{"role": "user", "content": [image_content(image_path), text_content(prompt)]}],
        )

    def generate_answer(
        self,
        query: str,
        evidences: list[Evidence],
        route: str,
        route_reason: str,
    ) -> tuple[str, list[Citation]]:
        if not evidences:
            return "答案：没有检索到足够证据，无法可靠回答。\n来源：[]", []

        content: list[dict[str, Any]] = []
        seen_images: set[str] = set()
        for evidence in evidences:
            if evidence.image_path and evidence.image_path not in seen_images and len(seen_images) < self.config.max_images:
                content.append(image_content(evidence.image_path))
                seen_images.add(evidence.image_path)

        content.append(text_content(build_grounded_answer_prompt(query, evidences, route, route_reason)))
        raw_answer = self._chat(model=self.config.vision_model if seen_images else self.config.chat_model, messages=[
            {"role": "user", "content": content}
        ])
        labels = extract_cited_labels(raw_answer)
        citations = citations_from_labels(labels, evidences)
        if not citations:
            citations = stable_citations(evidences, limit=3)
        answer = append_backend_sources(strip_model_sources(raw_answer), citations)
        return answer, citations

    def _chat(self, model: str, messages: list[dict[str, Any]]) -> str:
        client = self._load_client()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        content = response.choices[0].message.content
        return str(content).strip() if content else ""

    def _load_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise ImportError("DashScopeQwenModel requires `openai`. Run `pip install openai`.") from exc
            self._client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
        return self._client


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
    return (
        "你是一个多模态文档问答后端。请只依据给定证据回答，不能编造。\n"
        "引用证据时只能使用 [E1]、[E2] 这样的证据编号。不要自己编写 page 或 chunk，"
        "也不要输出“来源：”。\n"
        f"问题：{query}\n"
        f"路由：{route}\n"
        f"路由原因：{route_reason}\n"
        "证据：\n"
        + "\n\n".join(evidence_lines)
        + "\n\n请输出中文答案，并在相关句子后使用证据编号。"
        "最后一行必须严格使用格式：Final answer: <short answer>。"
    )


def text_content(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def image_content(image_path: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}}


def image_to_data_url(image_path: str) -> str:
    path = Path(image_path)
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(suffix, "image/png")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"
