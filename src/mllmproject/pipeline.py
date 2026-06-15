"""End-to-end mock RAG pipeline used by CLI, evaluation, and future UI."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ingest import add_page_visual_evidence, load_document
from .model_stack import ModelStack
from .router import route_question
from .schemas import AnswerResult, Document, Evidence


@dataclass
class RagPipeline:
    document: Document
    index: Any
    reranker: Any
    generator: Any

    @classmethod
    def from_file(
        cls,
        source_path: str | Path,
        output_dir: str | Path = "data/processed",
        include_visual: bool = True,
        render_pages: bool = True,
        doc_id: str | None = None,
        chunk_chars: int = 700,
        overlap: int = 80,
        model_stack: ModelStack | None = None,
        embedder: Any | None = None,
        index: Any | None = None,
        reranker: Any | None = None,
        generator: Any | None = None,
        visual_summarizer: Any | None = None,
    ) -> "RagPipeline":
        model_stack = model_stack or ModelStack.from_env()
        document = load_document(
            source_path,
            output_dir=output_dir,
            render_pages=render_pages,
            chunk_chars=chunk_chars,
            overlap=overlap,
            doc_id=doc_id,
        )
        if include_visual:
            add_page_visual_evidence(
                document,
                summarizer=visual_summarizer or model_stack.create_visual_summarizer(),
            )
        index = index or model_stack.create_index(embedder=embedder)
        index.build(document.chunks)
        return cls(
            document=document,
            index=index,
            reranker=reranker or model_stack.create_reranker(),
            generator=generator or model_stack.create_generator(),
        )

    @classmethod
    def from_document(
        cls,
        document: Document,
        include_visual: bool = True,
        model_stack: ModelStack | None = None,
        embedder: Any | None = None,
        index: Any | None = None,
        reranker: Any | None = None,
        generator: Any | None = None,
        visual_summarizer: Any | None = None,
    ) -> "RagPipeline":
        model_stack = model_stack or ModelStack.from_env()
        if include_visual:
            add_page_visual_evidence(
                document,
                summarizer=visual_summarizer or model_stack.create_visual_summarizer(),
            )
        index = index or model_stack.create_index(embedder=embedder)
        index.build(document.chunks)
        return cls(
            document=document,
            index=index,
            reranker=reranker or model_stack.create_reranker(),
            generator=generator or model_stack.create_generator(),
        )

    def answer(self, question: str, mode: str = "auto", top_k: int = 5) -> tuple[AnswerResult, float]:
        start = time.perf_counter()
        route, reason, source_types = self._resolve_mode(question, mode)
        search_k = max(top_k * 2, top_k + 8)
        evidences = self.index.search(question, top_k=search_k, source_types=source_types)
        evidences = self.reranker.rerank(question, evidences)
        evidences = prioritize_region_evidence(evidences)[:top_k]
        answer, citations = self.generator.generate_answer(question, evidences, route=route, route_reason=reason)
        result = AnswerResult(
            answer=answer,
            citations=citations,
            route=route,
            route_reason=reason,
            evidences=evidences,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        return result, latency_ms

    def _resolve_mode(self, question: str, mode: str) -> tuple[str, str, set[str] | None]:
        decision = route_question(question, mode=mode)
        source_types = set(decision.retrieval_modes) if decision.retrieval_modes else None
        return decision.route, decision.reason, source_types


def evidence_to_markdown(evidences: list[Evidence]) -> str:
    if not evidences:
        return "未检索到证据。"
    lines: list[str] = []
    for index, evidence in enumerate(evidences, start=1):
        chunk = evidence.chunk_id or evidence.evidence_id
        lines.append(
            f"{index}. page={evidence.page}, type={evidence.source_type}, score={evidence.score:.3f}, id={chunk}\n"
            f"   {evidence.content[:180]}"
        )
    return "\n".join(lines)


def prioritize_region_evidence(evidences: list[Evidence]) -> list[Evidence]:
    def priority(evidence: Evidence) -> tuple[int, int, float, float]:
        region_score = float((evidence.metadata or {}).get("score", 0.0))
        if evidence.source_type == "chart_region":
            return (2, 1 if evidence.image_path else 0, region_score, evidence.score)
        if evidence.source_type == "region":
            return (1, 1 if evidence.image_path else 0, region_score, evidence.score)
        return (0, 1 if evidence.image_path else 0, region_score, evidence.score)

    return sorted(evidences, key=priority, reverse=True)
