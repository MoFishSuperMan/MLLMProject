"""FastAPI bridge for the React document QA frontend."""

from __future__ import annotations

import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .index import vectors_to_jsonable
from .io_utils import ensure_dir, write_json
from .model_stack import ModelConfig, ModelStack, parse_bool
from .multimodal import draw_evidence_preview
from .pipeline import RagPipeline, prioritize_region_evidence
from .schemas import AnswerResult, Chunk, Citation, Document, Evidence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_UPLOAD_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".txt", ".md"}


class ApiError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class FileRecord:
    file_id: str
    file_name: str
    original_name: str
    mime_type: str
    size_bytes: int
    source_path: Path
    status: str = "uploaded"
    page_count: int | None = None
    chunk_count: int | None = None
    visual_region_count: int | None = None
    created_at: str = field(default_factory=iso_now)
    updated_at: str = field(default_factory=iso_now)
    error_message: str | None = None
    document: Document | None = None
    pipeline: RagPipeline | None = None
    disabled_chunk_ids: set[str] = field(default_factory=set)


@dataclass(slots=True)
class JobRecord:
    job_id: str
    file_id: str
    status: str = "queued"
    progress: float = 0.0
    stage: str = "queued"
    message: str = "Waiting to parse."
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    created_at: str = field(default_factory=iso_now)
    updated_at: str = field(default_factory=iso_now)


class QueryRequest(BaseModel):
    question: str
    file_ids: list[str] = Field(default_factory=list)
    selected_chunk_ids: list[str] = Field(default_factory=list)
    model: str = "qwen3_vl_local"
    mode: str = "auto"
    top_k: int = 5
    include_disabled: bool = False


class ParseRequest(BaseModel):
    include_visual: bool = True
    chunking: dict[str, int] | None = None


class ChunkPatchRequest(BaseModel):
    enabled: bool


class ApiStore:
    def __init__(
        self,
        project_root: str | Path = PROJECT_ROOT,
        use_real_models: bool | None = None,
        model_stack: ModelStack | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.upload_root = ensure_dir(self.project_root / "data" / "uploads")
        self.processed_root = ensure_dir(self.project_root / "data" / "processed")
        self.preview_root = ensure_dir(self.project_root / "data" / "api_previews")
        self.lock = RLock()
        self.files: dict[str, FileRecord] = {}
        self.jobs: dict[str, JobRecord] = {}
        self.model_config = build_model_config(self.project_root, use_real_models=use_real_models)
        self.model_stack = model_stack or ModelStack(self.model_config)

    def add_upload(self, upload: UploadFile) -> FileRecord:
        original_name = Path(upload.filename or "upload").name
        suffix = Path(original_name).suffix.lower()
        if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
            raise ApiError(
                "unsupported_file_type",
                f"Unsupported file type: {suffix or 'unknown'}.",
                status_code=415,
                details={"file_name": original_name},
            )

        file_id = f"file_{uuid.uuid4().hex[:12]}"
        upload_dir = ensure_dir(self.upload_root / file_id)
        source_path = upload_dir / f"original{suffix}"
        with source_path.open("wb") as output:
            shutil.copyfileobj(upload.file, output)

        record = FileRecord(
            file_id=file_id,
            file_name=original_name,
            original_name=original_name,
            mime_type=upload.content_type or guess_mime_type(source_path),
            size_bytes=source_path.stat().st_size,
            source_path=source_path,
        )
        with self.lock:
            self.files[file_id] = record
        return record

    def add_local_file(self, source: str | Path) -> FileRecord:
        source_path = Path(source)
        if not source_path.exists():
            raise ApiError(
                "sample_not_found",
                "Sample file not found.",
                status_code=404,
                details={"path": str(source_path)},
            )
        suffix = source_path.suffix.lower()
        if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
            raise ApiError("unsupported_file_type", f"Unsupported file type: {suffix}.", status_code=415)

        file_id = f"file_{uuid.uuid4().hex[:12]}"
        upload_dir = ensure_dir(self.upload_root / file_id)
        target = upload_dir / f"original{suffix}"
        shutil.copy2(source_path, target)
        record = FileRecord(
            file_id=file_id,
            file_name=source_path.name,
            original_name=source_path.name,
            mime_type=guess_mime_type(source_path),
            size_bytes=target.stat().st_size,
            source_path=target,
        )
        with self.lock:
            self.files[file_id] = record
        return record

    def create_parse_job(self, file_id: str) -> JobRecord:
        record = self.require_file(file_id)
        job = JobRecord(job_id=f"job_{uuid.uuid4().hex[:12]}", file_id=file_id)
        with self.lock:
            record.status = "queued"
            record.updated_at = iso_now()
            self.jobs[job.job_id] = job
        return job

    def parse_file(self, job_id: str, include_visual: bool = True, chunking: dict[str, int] | None = None) -> None:
        job = self.require_job(job_id)
        record = self.require_file(job.file_id)
        self._update_job(job, status="parsing", progress=0.08, stage="parsing", message="Parsing document.")
        with self.lock:
            record.status = "parsing"
            record.error_message = None
            record.updated_at = iso_now()

        try:
            chunk_chars = int((chunking or {}).get("max_chars", 700))
            overlap = int((chunking or {}).get("overlap", 80))
            pipeline = RagPipeline.from_file(
                record.source_path,
                output_dir=self.processed_root,
                include_visual=include_visual,
                render_pages=True,
                doc_id=record.file_id,
                chunk_chars=chunk_chars,
                overlap=overlap,
                model_stack=self.model_stack,
            )
            if chunking:
                write_json(self.processed_root / record.file_id / "parse_options.json", {"chunking": chunking})

            self._update_job(job, progress=0.86, stage="indexing", message="Saving index metadata.")
            save_pipeline_index(pipeline, self.processed_root / record.file_id / "index.json")
            visual_count = count_visual_regions(pipeline.document.chunks)
            result = {
                "page_count": len(pipeline.document.pages),
                "chunk_count": len(pipeline.document.chunks),
                "visual_region_count": visual_count,
            }

            with self.lock:
                record.pipeline = pipeline
                record.document = pipeline.document
                record.status = "ready"
                record.page_count = result["page_count"]
                record.chunk_count = result["chunk_count"]
                record.visual_region_count = result["visual_region_count"]
                record.updated_at = iso_now()
            self._write_file_manifest(record)
            self._update_job(
                job,
                status="ready",
                progress=1.0,
                stage="ready",
                message=f"Parsed {result['page_count']} pages and {result['chunk_count']} chunks.",
                result=result,
            )
        except Exception as exc:  # pragma: no cover - exercised through API error paths in integration
            error = {"code": "parse_failed", "message": str(exc), "details": {"file_id": record.file_id}}
            with self.lock:
                record.status = "failed"
                record.error_message = str(exc)
                record.updated_at = iso_now()
            self._update_job(
                job,
                status="failed",
                progress=1.0,
                stage="failed",
                message=str(exc),
                error=error,
            )

    def list_files(self) -> list[FileRecord]:
        with self.lock:
            return list(self.files.values())

    def require_file(self, file_id: str) -> FileRecord:
        with self.lock:
            record = self.files.get(file_id)
        if record is None:
            raise ApiError("file_not_found", "File not found.", status_code=404, details={"file_id": file_id})
        return record

    def require_job(self, job_id: str) -> JobRecord:
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            raise ApiError("job_not_found", "Job not found.", status_code=404, details={"job_id": job_id})
        return job

    def ready_records(self, file_ids: list[str]) -> list[FileRecord]:
        if not file_ids:
            file_ids = [record.file_id for record in self.list_files()]
        records = [self.require_file(file_id) for file_id in file_ids]
        for record in records:
            if record.status != "ready" or record.pipeline is None or record.document is None:
                raise ApiError(
                    "file_not_ready",
                    "File is still parsing.",
                    status_code=409,
                    details={"file_id": record.file_id, "status": record.status},
                )
        return records

    def chunks_for_file(
        self,
        file_id: str,
        page: int = 1,
        page_size: int = 10,
        source_type: str | None = None,
        enabled: bool | None = None,
        q: str | None = None,
    ) -> dict[str, Any]:
        record = self.require_file(file_id)
        if record.status != "ready" or record.document is None:
            raise ApiError(
                "file_not_ready",
                "File is still parsing.",
                status_code=409,
                details={"file_id": file_id, "status": record.status},
            )

        chunks = list(record.document.chunks)
        if source_type:
            chunks = [chunk for chunk in chunks if normalize_source_type(chunk.source_type) == source_type]
        if enabled is not None:
            chunks = [chunk for chunk in chunks if (chunk.chunk_id not in record.disabled_chunk_ids) == enabled]
        if q:
            needle = q.casefold()
            chunks = [
                chunk
                for chunk in chunks
                if needle in chunk.content.casefold() or needle in chunk.chunk_id.casefold()
            ]

        page_size = min(max(page_size, 1), 100)
        page = max(page, 1)
        start = (page - 1) * page_size
        visible = chunks[start : start + page_size]
        return {
            "file_id": record.file_id,
            "file_name": record.file_name,
            "total": len(chunks),
            "page": page,
            "page_size": page_size,
            "page_count": record.page_count or 0,
            "visual_region_count": record.visual_region_count or 0,
            "chunks": [chunk_to_api(chunk, record) for chunk in visible],
        }

    def set_chunk_enabled(self, chunk_id: str, enabled: bool) -> dict[str, Any]:
        record, chunk = self.find_chunk(chunk_id)
        with self.lock:
            if enabled:
                record.disabled_chunk_ids.discard(chunk.chunk_id)
            else:
                record.disabled_chunk_ids.add(chunk.chunk_id)
            record.updated_at = iso_now()
        return {"chunk_id": chunk.chunk_id, "enabled": enabled}

    def find_chunk(self, evidence_id: str) -> tuple[FileRecord, Chunk]:
        with self.lock:
            records = list(self.files.values())
        for record in records:
            if not record.document:
                continue
            for chunk in record.document.chunks:
                if evidence_id in {chunk.chunk_id, chunk.region_id}:
                    return record, chunk
        raise ApiError("chunk_not_found", "Chunk not found.", status_code=404, details={"chunk_id": evidence_id})

    def page_image_path(self, file_id: str, page: int) -> Path:
        record = self.require_file(file_id)
        if record.status != "ready" or record.document is None:
            raise ApiError(
                "file_not_ready",
                "File is still parsing.",
                status_code=409,
                details={"file_id": file_id, "status": record.status},
            )
        for item in record.document.pages:
            if item.page == page and item.image_path:
                path = Path(item.image_path)
                if path.exists():
                    return path
        raise ApiError(
            "page_image_not_found",
            "Page image not found.",
            status_code=404,
            details={"file_id": file_id, "page": page},
        )

    def evidence_preview_path(self, evidence_id: str) -> Path:
        record, chunk = self.find_chunk(evidence_id)
        evidence = Evidence.from_chunk(chunk, score=float((chunk.metadata or {}).get("score", 1.0)))
        if not evidence.image_path:
            return self.page_image_path(record.file_id, evidence.page)
        target = self.preview_root / record.file_id / f"{safe_path_id(evidence.evidence_id)}.png"
        preview = draw_evidence_preview(evidence, target)
        path = Path(preview.image_path if preview else evidence.image_path)
        if not path.exists():
            raise ApiError(
                "preview_not_found",
                "Evidence preview not found.",
                status_code=404,
                details={"evidence_id": evidence_id},
            )
        return path

    def answer(self, request: QueryRequest) -> dict[str, Any]:
        question = request.question.strip()
        if not question:
            raise ApiError("empty_question", "Question cannot be empty.", status_code=400)
        records = self.ready_records(request.file_ids)
        pipeline = self.pipeline_for_records(records)

        start = time.perf_counter()
        route, reason, source_types = pipeline._resolve_mode(question, request.mode)
        search_k = max(request.top_k * 2, request.top_k + 8)
        searched = pipeline.index.search(question, top_k=search_k, source_types=source_types)
        if not request.include_disabled:
            disabled = set().union(*(record.disabled_chunk_ids for record in records))
            searched = [evidence for evidence in searched if (evidence.chunk_id or evidence.evidence_id) not in disabled]
        searched = pipeline.reranker.rerank(question, searched)
        selected = self.selected_evidences(request.selected_chunk_ids, records)
        evidences = merge_evidences(selected, prioritize_region_evidence(searched))
        evidence_limit = max(request.top_k, len(selected), 1)
        evidences = evidences[:evidence_limit]
        answer, citations = pipeline.generator.generate_answer(question, evidences, route=route, route_reason=reason)
        latency_ms = (time.perf_counter() - start) * 1000
        result = AnswerResult(answer=answer, citations=citations, route=route, route_reason=reason, evidences=evidences)
        return answer_to_api(
            result=result,
            question=question,
            model=request.model,
            model_label=model_label_for(request.model),
            selected_chunk_ids=request.selected_chunk_ids,
            latency_ms=latency_ms,
            records=records,
        )

    def selected_evidences(self, selected_chunk_ids: list[str], records: list[FileRecord]) -> list[Evidence]:
        if not selected_chunk_ids:
            return []
        allowed_ids = {record.file_id for record in records}
        selected: list[Evidence] = []
        for chunk_id in selected_chunk_ids:
            try:
                record, chunk = self.find_chunk(chunk_id)
            except ApiError:
                continue
            if record.file_id not in allowed_ids:
                continue
            evidence = Evidence.from_chunk(chunk, score=max(float((chunk.metadata or {}).get("score", 1.0)), 1.0))
            evidence.metadata = {**evidence.metadata, "selected_by_user": True}
            selected.append(evidence)
        return selected

    def pipeline_for_records(self, records: list[FileRecord]) -> RagPipeline:
        if len(records) == 1:
            pipeline = records[0].pipeline
            if pipeline is None:
                raise ApiError("file_not_ready", "File is still parsing.", status_code=409)
            return pipeline

        documents = [record.document for record in records if record.document is not None]
        combined = Document(
            doc_id="combined_session",
            file_name="Combined session",
            pages=[page for document in documents for page in document.pages],
            chunks=[chunk for document in documents for chunk in document.chunks],
            metadata={"file_ids": [record.file_id for record in records]},
        )
        return RagPipeline.from_document(combined, include_visual=False, model_stack=self.model_stack)

    def _write_file_manifest(self, record: FileRecord) -> None:
        write_json(self.processed_root / record.file_id / "api_file.json", file_to_api(record))

    def _update_job(
        self,
        job: JobRecord,
        status: str | None = None,
        progress: float | None = None,
        stage: str | None = None,
        message: str | None = None,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        with self.lock:
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = progress
            if stage is not None:
                job.stage = stage
            if message is not None:
                job.message = message
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error
            job.updated_at = iso_now()


def create_app(store: ApiStore | None = None) -> FastAPI:
    api_store = store or ApiStore()
    app = FastAPI(title="MLLMProject API", version="0.1.0")
    app.state.store = api_store
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(ApiError)
    async def api_error_handler(_request, exc: ApiError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
        )

    @app.get("/api/v1/models")
    def list_models() -> dict[str, Any]:
        return {"models": model_options(api_store.model_config.use_real_models)}

    @app.post("/api/v1/files")
    async def upload_files(files: list[UploadFile] = File(...)) -> dict[str, Any]:
        if not files:
            raise ApiError("invalid_request", "At least one file is required.", status_code=400)
        records = [api_store.add_upload(upload) for upload in files]
        return {"files": [file_to_api(record) for record in records]}

    @app.post("/api/v1/files/{file_id}/parse")
    def start_parse(file_id: str, request: ParseRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
        job = api_store.create_parse_job(file_id)
        background_tasks.add_task(api_store.parse_file, job.job_id, request.include_visual, request.chunking)
        return {"job_id": job.job_id, "file_id": job.file_id, "status": job.status}

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        return job_to_api(api_store.require_job(job_id))

    @app.get("/api/v1/files")
    def list_files() -> dict[str, Any]:
        return {"files": [file_to_api(record) for record in api_store.list_files()]}

    @app.get("/api/v1/files/{file_id}")
    def get_file(file_id: str) -> dict[str, Any]:
        return {"file": file_to_api(api_store.require_file(file_id))}

    @app.get("/api/v1/files/{file_id}/chunks")
    def list_chunks(
        file_id: str,
        page: int = 1,
        page_size: int = 10,
        source_type: str | None = None,
        enabled: bool | None = None,
        q: str | None = None,
    ) -> dict[str, Any]:
        return api_store.chunks_for_file(
            file_id=file_id,
            page=page,
            page_size=page_size,
            source_type=source_type,
            enabled=enabled,
            q=q,
        )

    @app.patch("/api/v1/chunks/{chunk_id}")
    def patch_chunk(chunk_id: str, request: ChunkPatchRequest) -> dict[str, Any]:
        return api_store.set_chunk_enabled(chunk_id, request.enabled)

    @app.get("/api/v1/files/{file_id}/pages/{page}/image")
    def page_image(file_id: str, page: int):
        path = api_store.page_image_path(file_id, page)
        return FileResponse(path, media_type=guess_mime_type(path))

    @app.get("/api/v1/evidence/{evidence_id}/preview")
    def evidence_preview(evidence_id: str):
        path = api_store.evidence_preview_path(evidence_id)
        return FileResponse(path, media_type=guess_mime_type(path))

    @app.post("/api/v1/query")
    def query(request: QueryRequest) -> dict[str, Any]:
        return api_store.answer(request)

    @app.post("/api/v1/demo/sample-session")
    def sample_session(background_tasks: BackgroundTasks) -> dict[str, Any]:
        record = api_store.add_local_file(api_store.project_root / "多模态大模型大作业说明.pdf")
        job = api_store.create_parse_job(record.file_id)
        background_tasks.add_task(api_store.parse_file, job.job_id, True, None)
        return {
            "files": [file_to_api(record)],
            "active_file_id": record.file_id,
            "jobs": [{"job_id": job.job_id, "file_id": record.file_id, "status": job.status}],
        }

    return app


def build_model_config(project_root: Path, use_real_models: bool | None) -> ModelConfig:
    default_real = parse_bool(os.getenv("MLLMPROJECT_USE_REAL_MODELS"), default=True)
    config = ModelConfig.from_env()
    config.use_real_models = default_real if use_real_models is None else use_real_models
    local_model_dir = project_root / "model"
    if config.use_real_models and not os.getenv("MLLMPROJECT_QWEN3_MODEL_PATH") and local_model_dir.exists():
        config.vlm_model_id = str(local_model_dir)
    if not os.getenv("MLLMPROJECT_VLM_MAX_IMAGES"):
        config.vlm_max_images = 1
    if not os.getenv("MLLMPROJECT_VLM_MAX_NEW_TOKENS"):
        config.vlm_max_new_tokens = 96
    return config


def model_options(use_real_models: bool) -> list[dict[str, Any]]:
    return [
        {
            "id": "qwen3_vl_local",
            "label": "Qwen3-VL-8B",
            "provider": "local",
            "description": "Qwen3-VL-8B with BGE retrieval and reranking.",
            "enabled": use_real_models,
            "is_default": use_real_models,
            "supports_vision": True,
            "supports_text": True,
        },
        {
            "id": "local_mock",
            "label": "Local Mock",
            "provider": "mock",
            "description": "Lightweight local pipeline for tests and fallback development.",
            "enabled": not use_real_models,
            "is_default": not use_real_models,
            "supports_vision": False,
            "supports_text": True,
        },
    ]


def file_to_api(record: FileRecord) -> dict[str, Any]:
    return {
        "file_id": record.file_id,
        "file_name": record.file_name,
        "original_name": record.original_name,
        "mime_type": record.mime_type,
        "size_bytes": record.size_bytes,
        "status": record.status,
        "page_count": record.page_count,
        "chunk_count": record.chunk_count,
        "visual_region_count": record.visual_region_count,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "error_message": record.error_message,
    }


def job_to_api(job: JobRecord) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "file_id": job.file_id,
        "status": job.status,
        "progress": job.progress,
        "stage": job.stage,
        "message": job.message,
        "result": job.result,
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


def chunk_to_api(chunk: Chunk, record: FileRecord, score: float | None = None) -> dict[str, Any]:
    evidence_id = chunk.region_id or chunk.chunk_id
    source_type = normalize_source_type(chunk.source_type)
    return {
        "chunk_id": chunk.chunk_id,
        "evidence_id": evidence_id,
        "file_id": record.file_id,
        "file_name": record.file_name,
        "page": chunk.page,
        "source_type": source_type,
        "title": chunk_title(chunk, source_type),
        "content": chunk.content,
        "score": float(score if score is not None else (chunk.metadata or {}).get("score", 0.8)),
        "bbox": chunk.bbox,
        "region_id": chunk.region_id,
        "image_url": f"/api/v1/files/{record.file_id}/pages/{chunk.page}/image" if chunk.page else None,
        "preview_url": f"/api/v1/evidence/{evidence_id}/preview" if chunk.image_path else None,
        "enabled": chunk.chunk_id not in record.disabled_chunk_ids,
        "metadata": chunk.metadata or {},
    }


def evidence_to_api(evidence: Evidence, records_by_id: dict[str, FileRecord]) -> dict[str, Any]:
    record = records_by_id.get(evidence.doc_id)
    if record is None:
        record = next(iter(records_by_id.values()))
    chunk = Chunk(
        chunk_id=evidence.chunk_id or evidence.evidence_id,
        doc_id=evidence.doc_id,
        page=evidence.page,
        source_type=evidence.source_type,
        content=evidence.content,
        bbox=evidence.bbox,
        image_path=evidence.image_path,
        region_id=evidence.region_id or evidence.evidence_id,
        metadata=evidence.metadata,
    )
    return chunk_to_api(chunk, record, score=evidence.score)


def citation_to_api(citation: Citation, records_by_id: dict[str, FileRecord]) -> dict[str, Any]:
    file_id = ""
    file_name = ""
    if citation.evidence_id:
        for record in records_by_id.values():
            if record.document and any(
                citation.evidence_id in {chunk.chunk_id, chunk.region_id} for chunk in record.document.chunks
            ):
                file_id = record.file_id
                file_name = record.file_name
                break
    if not file_id and records_by_id:
        record = next(iter(records_by_id.values()))
        file_id = record.file_id
        file_name = record.file_name

    evidence_id = citation.evidence_id or citation.region_id or citation.chunk_id or ""
    return {
        "citation_id": f"cite_{uuid.uuid4().hex[:8]}",
        "evidence_id": evidence_id,
        "chunk_id": citation.chunk_id,
        "file_id": file_id,
        "file_name": file_name,
        "page": citation.page,
        "source_type": normalize_source_type(citation.source_type),
        "bbox": citation.bbox,
        "quote": None,
        "preview_url": f"/api/v1/evidence/{evidence_id}/preview" if evidence_id else None,
    }


def answer_to_api(
    result: AnswerResult,
    question: str,
    model: str,
    model_label: str,
    selected_chunk_ids: list[str],
    latency_ms: float,
    records: list[FileRecord],
) -> dict[str, Any]:
    records_by_id = {record.file_id: record for record in records}
    return {
        "answer_id": f"ans_{uuid.uuid4().hex[:12]}",
        "question": question,
        "answer": result.answer,
        "model": model,
        "model_label": model_label,
        "route": result.route,
        "route_reason": result.route_reason,
        "selected_chunk_ids": selected_chunk_ids,
        "citations": [citation_to_api(citation, records_by_id) for citation in result.citations],
        "evidences": [evidence_to_api(evidence, records_by_id) for evidence in result.evidences],
        "latency_ms": round(latency_ms, 2),
        "created_at": iso_now(),
    }


def normalize_source_type(source_type: str) -> str:
    normalized = source_type.strip().lower()
    if normalized in {"text", "figure", "table", "visual"}:
        return normalized
    if normalized in {"chart", "chart_region", "plot", "image", "region"}:
        return "figure"
    return "visual"


def chunk_title(chunk: Chunk, source_type: str) -> str:
    title = (chunk.metadata or {}).get("title") or (chunk.metadata or {}).get("section")
    if title:
        return str(title)
    if source_type == "text":
        return f"Page {chunk.page} text chunk"
    if source_type == "figure":
        return f"Page {chunk.page} visual region"
    if source_type == "table":
        return f"Page {chunk.page} table chunk"
    return f"Page {chunk.page} visual summary"


def merge_evidences(selected: list[Evidence], searched: list[Evidence]) -> list[Evidence]:
    merged: list[Evidence] = []
    seen: set[str] = set()
    for evidence in [*selected, *searched]:
        key = evidence.chunk_id or evidence.evidence_id
        if key in seen:
            continue
        seen.add(key)
        merged.append(evidence)
    return merged


def count_visual_regions(chunks: list[Chunk]) -> int:
    return sum(1 for chunk in chunks if normalize_source_type(chunk.source_type) in {"figure", "table", "visual"})


def save_pipeline_index(pipeline: RagPipeline, path: Path) -> None:
    try:
        pipeline.index.save(path)
    except Exception:
        payload = {
            "chunks": [chunk.to_dict() for chunk in pipeline.index.chunks],
            "vectors": vectors_to_jsonable(getattr(pipeline.index, "vectors", [])),
        }
        write_json(path, payload)


def model_label_for(model_id: str) -> str:
    return {"qwen3_vl_local": "Qwen3-VL-8B", "local_mock": "Local Mock"}.get(model_id, model_id)


def guess_mime_type(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    return {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".txt": "text/plain",
        ".md": "text/markdown",
    }.get(suffix, "application/octet-stream")


def safe_path_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)


app = create_app()
