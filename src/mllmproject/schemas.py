"""Shared data contracts for the document RAG demo.

This file intentionally keeps backward compatibility with the text baseline
while adding the fields required by the multimodal frontend demo.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


JsonDict = dict[str, Any]
BBox = list[float] | list[int] | None
RouteName = Literal["text_route", "table_route", "vision_route", "hybrid_route"]


@dataclass(slots=True)
class PageText:
    page: int
    text: str

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "PageText":
        return cls(page=int(data["page"]), text=str(data.get("text", "")))


@dataclass(slots=True)
class Page:
    doc_id: str
    page: int
    text: str = ""
    image_path: str | None = None
    width: int | None = None
    height: int | None = None

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    page: int
    source_type: str
    content: str
    bbox: BBox = None
    image_path: str | None = None
    region_id: str | None = None
    metadata: JsonDict = field(default_factory=dict)

    def to_evidence(self, score: float) -> "Evidence":
        return Evidence.from_chunk(self, score=score)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "Chunk":
        return cls(
            chunk_id=str(data["chunk_id"]),
            doc_id=str(data["doc_id"]),
            page=int(data["page"]),
            source_type=str(data.get("source_type", "text")),
            content=str(data.get("content", "")),
            bbox=data.get("bbox"),
            image_path=data.get("image_path"),
            region_id=data.get("region_id"),
            metadata=data.get("metadata") or {},
        )


@dataclass(slots=True)
class Document:
    doc_id: str
    source_path: str = ""
    pages: list[Page] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)
    file_name: str = ""
    file_path: str = ""

    def __post_init__(self) -> None:
        if not self.file_path and self.source_path:
            self.file_path = self.source_path
        if not self.source_path and self.file_path:
            self.source_path = self.file_path
        if not self.file_name:
            self.file_name = self.file_path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] if self.file_path else self.doc_id

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class Evidence:
    evidence_id: str
    doc_id: str
    page: int
    source_type: str
    content: str
    score: float
    chunk_id: str | None = None
    bbox: BBox = None
    image_path: str | None = None
    region_id: str | None = None
    metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def from_chunk(cls, chunk: Chunk, score: float) -> "Evidence":
        return cls(
            evidence_id=chunk.region_id or chunk.chunk_id,
            doc_id=chunk.doc_id,
            page=chunk.page,
            source_type=chunk.source_type,
            content=chunk.content,
            score=score,
            chunk_id=chunk.chunk_id,
            bbox=chunk.bbox,
            image_path=chunk.image_path,
            region_id=chunk.region_id,
            metadata=dict(chunk.metadata),
        )

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class Citation:
    page: int
    source_type: str = "text"
    chunk_id: str | None = None
    bbox: BBox = None
    region_id: str | None = None
    evidence_id: str | None = None

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class RouteDecision:
    route: RouteName
    reason: str
    retrieval_modes: list[str]

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class AnswerResult:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    evidences: list[Evidence] = field(default_factory=list)
    route: str = ""
    route_reason: str = ""

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class EvalPrediction:
    sample_id: str
    question: str
    gold_answer: str
    predicted_answer: str
    gold_page: int | None
    cited_pages: list[int]
    retrieved_pages: list[int]
    retrieved_evidence_ids: list[str]
    route: str
    latency_ms: float
    gold_pages: list[int] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        return asdict(self)
