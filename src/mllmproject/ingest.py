"""Document ingestion for text chunks and page images."""

from __future__ import annotations

import re
import subprocess
import uuid
from pathlib import Path

from .io_utils import ensure_dir, write_json
from .chunking import chunk_pages as chunk_page_texts
from .models import MockVisualSummarizer
from .multimodal import draw_bbox_preview, make_page_visual_chunks
from .pdf import extract_document_text, make_doc_id
from .schemas import Chunk, Document, Page, PageText


def load_document(
    source_path: str | Path,
    output_dir: str | Path = "data/processed",
    render_pages: bool = True,
    chunk_chars: int = 700,
    overlap: int = 80,
    doc_id: str | None = None,
) -> Document:
    return DocumentIngestor(
        processed_root=output_dir,
        chunk_chars=chunk_chars,
        overlap=overlap,
        render_pages=render_pages,
    ).ingest(source_path, doc_id=doc_id)


def extract_pages(source: Path, doc_id: str, doc_dir: Path, render_pages: bool = True) -> list[Page]:
    suffix = source.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_pages(source, doc_id, doc_dir, render_pages=render_pages)
    if suffix in {".txt", ".md"}:
        page_texts = extract_document_text(source)
        return [Page(doc_id=doc_id, page=item.page, text=item.text) for item in page_texts]
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        return [Page(doc_id=doc_id, page=1, text="", image_path=str(source))]
    raise ValueError(f"Unsupported document type: {source.suffix}")


def extract_pdf_pages(source: Path, doc_id: str, doc_dir: Path, render_pages: bool = True) -> list[Page]:
    try:
        return extract_pdf_pages_with_pymupdf(source, doc_id, doc_dir, render_pages=render_pages)
    except ImportError:
        return extract_pdf_pages_with_cli(source, doc_id, doc_dir, render_pages=render_pages)


def extract_pdf_pages_with_pymupdf(source: Path, doc_id: str, doc_dir: Path, render_pages: bool = True) -> list[Page]:
    import fitz  # type: ignore[import-not-found]

    page_dir = ensure_dir(doc_dir / "pages")
    pages: list[Page] = []
    with fitz.open(source) as pdf:
        for index, pdf_page in enumerate(pdf, start=1):
            text = pdf_page.get_text("text")
            image_path = None
            width = int(pdf_page.rect.width)
            height = int(pdf_page.rect.height)
            if render_pages:
                pixmap = pdf_page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
                image_path = str(page_dir / f"page_{index:03d}.png")
                pixmap.save(image_path)
                width = pixmap.width
                height = pixmap.height
            pages.append(Page(doc_id=doc_id, page=index, text=text, image_path=image_path, width=width, height=height))
    return pages


def extract_pdf_pages_with_cli(source: Path, doc_id: str, doc_dir: Path, render_pages: bool = True) -> list[Page]:
    text = ""
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(source), "-"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        text = result.stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        text = ""

    page_texts = split_pdftotext_pages(text)
    page_dir = ensure_dir(doc_dir / "pages")
    image_paths: list[str | None] = [None] * max(len(page_texts), 1)
    if render_pages:
        try:
            render_dir = ensure_dir(page_dir / f"_render_{uuid.uuid4().hex}")
            prefix = render_dir / "page"
            subprocess.run(["pdftoppm", "-png", "-r", "130", str(source), str(prefix)], check=True, capture_output=True)
            image_paths = [str(image) for image in sorted(render_dir.glob("page-*.png"))]
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

    total_pages = max(len(page_texts), len(image_paths), 1)
    pages: list[Page] = []
    for index in range(total_pages):
        pages.append(
            Page(
                doc_id=doc_id,
                page=index + 1,
                text=page_texts[index] if index < len(page_texts) else "",
                image_path=image_paths[index] if index < len(image_paths) else None,
            )
        )
    return pages


def split_pdftotext_pages(text: str) -> list[str]:
    if not text.strip():
        return [""]
    pages = [page.strip() for page in text.split("\f")]
    return [page for page in pages if page]


def chunk_pages(pages: list[Page], chunk_chars: int = 700, overlap: int = 80) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        blocks = split_text_blocks(page.text)
        page_chunks: list[str] = []
        for block in blocks:
            page_chunks.extend(sliding_chunks(block, chunk_chars=chunk_chars, overlap=overlap))
        if not page_chunks and page.text.strip():
            page_chunks = [page.text.strip()]

        for local_index, content in enumerate(page_chunks, start=1):
            chunk_id = f"{page.doc_id}_p{page.page}_c{local_index}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=page.doc_id,
                    page=page.page,
                    source_type="text",
                    content=content,
                    image_path=page.image_path,
                )
            )
    return chunks


def split_text_blocks(text: str) -> list[str]:
    text = text.replace("\r\n", "\n")
    blocks = [block.strip() for block in re.split(r"\n\s*\n+", text)]
    blocks = [re.sub(r"[ \t]+", " ", block) for block in blocks if block.strip()]
    if blocks:
        return blocks
    return [text.strip()] if text.strip() else []


def sliding_chunks(text: str, chunk_chars: int = 700, overlap: int = 80) -> list[str]:
    if len(text) <= chunk_chars:
        return [text]
    chunks: list[str] = []
    step = max(chunk_chars - overlap, 1)
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_chars].strip()
        if piece:
            chunks.append(piece)
        if start + chunk_chars >= len(text):
            break
    return chunks


def add_page_visual_evidence(document: Document, summarizer: MockVisualSummarizer | None = None) -> list[Chunk]:
    summarizer = summarizer or MockVisualSummarizer()
    visual_chunks = make_page_visual_chunks(document, summary_fn=summarizer.generate_visual_summary)
    document.chunks.extend(visual_chunks)
    processed_dir = document.metadata.get("processed_dir")
    if processed_dir:
        write_json(Path(processed_dir) / "chunks_with_visual.json", [chunk.to_dict() for chunk in document.chunks])
    return visual_chunks


class DocumentIngestor:
    """Ingest documents for the Gradio demo engine."""

    def __init__(
        self,
        processed_root: str | Path = "data/processed",
        chunk_chars: int = 700,
        overlap: int = 80,
        render_pages: bool = True,
    ) -> None:
        self.processed_root = Path(processed_root)
        self.chunk_chars = chunk_chars
        self.overlap = overlap
        self.render_pages = render_pages

    def ingest(self, file_path: str | Path, doc_id: str | None = None) -> Document:
        source = Path(file_path)
        if not source.exists():
            raise FileNotFoundError(source)

        doc_id = doc_id or make_doc_id(source)
        doc_dir = ensure_dir(self.processed_root / doc_id)
        pages = extract_pages(source, doc_id, doc_dir, render_pages=self.render_pages)
        page_texts = [PageText(page=page.page, text=page.text) for page in pages]
        chunks = chunk_page_texts(
            doc_id,
            page_texts,
            max_chars=self.chunk_chars,
            overlap=self.overlap,
        )

        document = Document(
            doc_id=doc_id,
            file_name=source.name,
            file_path=str(source),
            source_path=str(source),
            pages=pages,
            chunks=chunks,
            metadata={"processed_dir": str(doc_dir)},
        )
        write_json(doc_dir / "document.json", document.to_dict())
        write_json(doc_dir / "metadata.json", self.metadata(document))
        write_json(doc_dir / "chunks.json", [chunk.to_dict() for chunk in chunks])
        return document

    def metadata(self, document: Document) -> dict:
        return {
            "doc_id": document.doc_id,
            "source_path": document.source_path,
            "file_name": document.file_name,
            "page_count": len(document.pages),
            "chunk_count": len(document.chunks),
            "processed_dir": document.metadata.get("processed_dir", ""),
            "render_pages": self.render_pages,
        }
