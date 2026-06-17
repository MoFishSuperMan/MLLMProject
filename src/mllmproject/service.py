"""UI-independent service layer for document ingestion and QA."""

from __future__ import annotations

from pathlib import Path

from .model_stack import ModelConfig, ModelStack
from .multimodal import draw_evidence_preview
from .pipeline import RagPipeline
from .schemas import AnswerResult, Document, Evidence


class RagService:
    """Stateful service shared by the Gradio app, scripts, and evaluation."""

    def __init__(
        self,
        processed_root: str | Path = "data/processed",
        model_stack: ModelStack | None = None,
        model_config: ModelConfig | None = None,
    ) -> None:
        self.processed_root = Path(processed_root)
        if model_stack is not None:
            self.model_stack = model_stack
        elif model_config is not None:
            self.model_stack = ModelStack(model_config)
        else:
            self.model_stack = None
        self.pipeline: RagPipeline | None = None
        self.document: Document | None = None
        self.index_chunks = []

    def ingest_document(self, file_path: str | Path) -> Document:
        self.pipeline = RagPipeline.from_file(
            file_path,
            output_dir=self.processed_root,
            include_visual=True,
            model_stack=self.model_stack,
        )
        self.document = self.pipeline.document
        self.index_chunks = self.pipeline.index.chunks
        return self.document

    def ask(self, question: str, mode: str = "Auto Router", top_k: int = 5) -> AnswerResult:
        if self.pipeline is None:
            raise RuntimeError("请先上传并解析文档。")
        if not question.strip():
            raise ValueError("问题不能为空。")
        result, _latency_ms = self.pipeline.answer(question, mode=normalize_mode(mode), top_k=top_k)
        return result

    def make_citation_previews(self, evidences: list[Evidence]) -> list[tuple[str, str]]:
        if self.document is None:
            return []

        preview_dir = self.processed_root / self.document.doc_id / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        previews: list[tuple[str, str]] = []

        for evidence in evidences:
            if not evidence.image_path:
                continue
            target = preview_dir / f"{evidence.evidence_id}.png"
            preview = draw_evidence_preview(evidence, target)
            if preview:
                previews.append(preview.as_gallery_item())
        return previews


def normalize_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized == "text-rag":
        return "text-rag"
    if normalized == "mm-rag":
        return "mm-rag"
    return "auto"
