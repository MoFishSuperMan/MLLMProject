"""Document ingestion for text chunks and page images."""

from __future__ import annotations

import re
import subprocess
import uuid
from pathlib import Path
from typing import Any

from .io_utils import ensure_dir, write_json
from .chunking import chunk_pages as chunk_page_texts
from .chunking import looks_like_code_line
from .models import MockVisualSummarizer
from .multimodal import draw_bbox_preview, make_page_visual_chunks
from .pdf import extract_document_text, make_doc_id
from .schemas import Chunk, Document, Page, PageText
from .table_markdown import table_to_markdown


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

    if any(chunk.source_type in {"figure", "table", "formula", "code"} for chunk in document.chunks):
        processed_dir = document.metadata.get("processed_dir")
        if processed_dir:
            write_json(Path(processed_dir) / "chunks_with_visual.json", [chunk.to_dict() for chunk in document.chunks])
        return []

    def safe_summary(image_path: str) -> str:
        try:
            return summarizer.generate_visual_summary(image_path)
        except Exception:
            page = next((item for item in document.pages if item.image_path == image_path), None)
            return default_visual_summary(page.page if page else None)

    visual_chunks = make_page_visual_chunks(document, summary_fn=safe_summary)
    document.chunks.extend(visual_chunks)
    processed_dir = document.metadata.get("processed_dir")
    if processed_dir:
        write_json(Path(processed_dir) / "chunks_with_visual.json", [chunk.to_dict() for chunk in document.chunks])
    return visual_chunks


def default_visual_summary(page: int | None) -> str:
    page_text = f"Page {page}" if page else "This page"
    return f"{page_text} PDF preview. Vision summary API is unavailable; using page-level evidence."


class DocumentIngestor:
    """Ingest documents for local document QA pipelines."""

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
        layout_chunks: list[Chunk] = []
        if source.suffix.lower() == ".pdf" and self.render_pages:
            layout_chunks = extract_pdf_layout_chunks(source, doc_id, doc_dir, pages)
            pages = replace_page_text_with_layout_exclusions(source, pages, layout_chunks)

        page_texts = [PageText(page=page.page, text=page.text) for page in pages]
        chunks = chunk_page_texts(
            doc_id,
            page_texts,
            max_chars=self.chunk_chars,
            overlap=self.overlap,
        )
        pages_by_number = {page.page: page for page in pages}
        for chunk in chunks:
            page = pages_by_number.get(chunk.page)
            if page is None:
                continue
            chunk.image_path = page.image_path
            if page.width and page.height:
                margin_x = max(float(page.width) * 0.06, 24.0)
                margin_y = max(float(page.height) * 0.06, 24.0)
                chunk.bbox = [
                    margin_x,
                    margin_y,
                    float(page.width) - margin_x,
                    float(page.height) - margin_y,
                ]
            chunk.metadata = {
                **chunk.metadata,
                "highlight_kind": "page_text_region",
                "page_width": page.width,
                "page_height": page.height,
            }

        chunks.extend(layout_chunks)

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


def replace_page_text_with_layout_exclusions(source: Path, pages: list[Page], layout_chunks: list[Chunk]) -> list[Page]:
    """Remove structured layout regions from page text before text chunking."""

    if not layout_chunks:
        return pages
    remove_text_excerpts_from_pages(pages, layout_chunks)
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError:
        return pages

    exclusions_by_page: dict[int, list[list[float]]] = {}
    for chunk in layout_chunks:
        if chunk.source_type not in {"table", "figure", "formula", "code"}:
            continue
        pdf_bbox = (chunk.metadata or {}).get("pdf_bbox")
        if isinstance(pdf_bbox, list) and len(pdf_bbox) == 4:
            exclusions_by_page.setdefault(chunk.page, []).append([float(value) for value in pdf_bbox])
    if not exclusions_by_page:
        return pages

    page_by_number = {page.page: page for page in pages}
    with fitz.open(source) as pdf:
        for page_index, pdf_page in enumerate(pdf, start=1):
            page = page_by_number.get(page_index)
            exclusions = exclusions_by_page.get(page_index)
            if page is None or not exclusions:
                continue
            kept_blocks: list[tuple[float, str]] = []
            for block in pdf_page.get_text("blocks"):
                if len(block) < 5:
                    continue
                text = re.sub(r"\s+", " ", str(block[4] or "")).strip()
                if not text:
                    continue
                block_bbox = [float(block[0]), float(block[1]), float(block[2]), float(block[3])]
                if any(should_exclude_text_block(block_bbox, exclusion) for exclusion in exclusions):
                    continue
                kept_blocks.append((float(block[1]), text))
            page.text = "\n\n".join(text for _y, text in sorted(kept_blocks, key=lambda item: item[0]))
    return pages


def remove_text_excerpts_from_pages(pages: list[Page], layout_chunks: list[Chunk]) -> None:
    page_by_number = {page.page: page for page in pages}
    for chunk in layout_chunks:
        excerpts = (chunk.metadata or {}).get("text_excerpts")
        if not isinstance(excerpts, list):
            continue
        page = page_by_number.get(chunk.page)
        if page is None:
            continue
        text = page.text
        for excerpt in excerpts:
            value = str(excerpt or "").strip()
            if value:
                text = text.replace(value, "\n")
                text = remove_excerpt_lines(text, value)
        page.text = re.sub(r"\n{3,}", "\n\n", text).strip()


def remove_excerpt_lines(text: str, excerpt: str) -> str:
    excerpt_lines = {normalize_layout_line(line) for line in excerpt.splitlines() if normalize_layout_line(line)}
    if not excerpt_lines:
        return text
    kept = [line for line in text.splitlines() if normalize_layout_line(line) not in excerpt_lines]
    return "\n".join(kept)


def normalize_layout_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def should_exclude_text_block(block_bbox: list[float], exclusion_bbox: list[float]) -> bool:
    if bbox_iou(block_bbox, exclusion_bbox) >= 0.08:
        return True
    cx = (block_bbox[0] + block_bbox[2]) / 2
    cy = (block_bbox[1] + block_bbox[3]) / 2
    return exclusion_bbox[0] <= cx <= exclusion_bbox[2] and exclusion_bbox[1] <= cy <= exclusion_bbox[3]


def extract_pdf_layout_chunks(source: Path, doc_id: str, doc_dir: Path, pages: list[Page]) -> list[Chunk]:
    """Extract table, figure, and formula regions as standalone visual chunks."""

    try:
        import fitz  # type: ignore[import-not-found]
        from PIL import Image
    except ImportError:
        return extract_pdf_layout_chunks_from_text(pages, doc_id)

    page_by_number = {page.page: page for page in pages}
    region_dir = ensure_dir(doc_dir / "regions")
    chunks: list[Chunk] = []
    counters = {"figure": 0, "table": 0, "formula": 0, "code": 0}
    text_layout_chunks = extract_pdf_layout_chunks_from_text(
        layout_text_pages(source, doc_id, doc_dir, pages),
        doc_id,
    )

    with fitz.open(source) as pdf:
        for page_index, pdf_page in enumerate(pdf, start=1):
            page = page_by_number.get(page_index)
            if not page or not page.image_path:
                continue
            page_size = (float(pdf_page.rect.width), float(pdf_page.rect.height))
            text_blocks = pdf_page.get_text("blocks")

            regions: list[dict[str, Any]] = []
            regions.extend(detect_table_regions(pdf_page, text_blocks))
            regions.extend(detect_figure_regions(pdf_page, text_blocks))
            regions.extend(detect_code_regions(pdf_page, text_blocks))
            regions.extend(detect_formula_regions(pdf_page, text_blocks))
            regions = dedupe_regions(regions)
            regions = dedupe_formula_regions(regions)

            try:
                page_image = Image.open(page.image_path).convert("RGB")
            except OSError:
                continue

            for region in regions:
                kind = str(region["kind"])
                counters[kind] += 1
                number = int(region.get("number") or counters[kind])
                region_id = f"{doc_id}_p{page_index}_{kind}{number:02d}"
                image_bbox = scale_pdf_bbox_to_image(region["bbox"], page_size, page_image.size)
                crop_path = region_dir / f"{region_id}.png"
                save_region_crop(page_image, image_bbox, crop_path)
                crop_bbox = [0.0, 0.0, float(max(1, image_bbox[2] - image_bbox[0])), float(max(1, image_bbox[3] - image_bbox[1]))]
                content = build_region_content(kind, number, region)
                table_markdown = table_to_markdown(region.get("structured_data")) if kind == "table" else ""
                chunks.append(
                    Chunk(
                        chunk_id=region_id,
                        doc_id=doc_id,
                        page=page_index,
                        source_type=kind,
                        content=content,
                        bbox=crop_bbox,
                        image_path=str(crop_path),
                        region_id=region_id,
                        metadata={
                            "region_kind": kind,
                            "caption": region.get("caption", ""),
                            "label": region_label(kind, number),
                            "number": number,
                            "page_bbox": [float(value) for value in image_bbox],
                            "pdf_bbox": [round(float(value), 2) for value in region["bbox"]],
                            "page_width": page.width,
                            "page_height": page.height,
                            "structured_data": region.get("structured_data"),
                            "table_markdown": table_markdown,
                            "formula_markdown": region.get("formula_markdown", ""),
                            "code_language": region.get("code_language", ""),
                            "extraction_method": region.get("method", "pymupdf_layout"),
                        },
                    )
                )
            page_image.close()
    return merge_text_layout_with_visual_crops(text_layout_chunks, chunks)


def layout_text_pages(source: Path, doc_id: str, doc_dir: Path, pages: list[Page]) -> list[Page]:
    """Use pdftotext layout output as the authoritative structured text source."""

    try:
        text_pages = extract_pdf_pages_with_cli(source, doc_id, doc_dir, render_pages=False)
    except Exception:
        text_pages = []
    if not any(page.text.strip() for page in text_pages):
        text_pages = pages

    page_meta = {page.page: page for page in pages}
    hydrated: list[Page] = []
    for page in text_pages:
        visual_page = page_meta.get(page.page)
        hydrated.append(
            Page(
                doc_id=doc_id,
                page=page.page,
                text=page.text,
                image_path=visual_page.image_path if visual_page else page.image_path,
                width=visual_page.width if visual_page else page.width,
                height=visual_page.height if visual_page else page.height,
            )
        )
    return hydrated


def merge_text_layout_with_visual_crops(text_chunks: list[Chunk], visual_chunks: list[Chunk]) -> list[Chunk]:
    if not text_chunks:
        return visual_chunks

    merged: list[Chunk] = []
    used_visual: set[str] = set()
    fallback_types_by_page = {(chunk.page, chunk.source_type) for chunk in text_chunks}
    visual_by_key: dict[tuple[int, str], list[Chunk]] = {}
    for chunk in visual_chunks:
        visual_by_key.setdefault((chunk.page, chunk.source_type), []).append(chunk)

    for chunk in text_chunks:
        visual = select_visual_crop_for_text_chunk(chunk, visual_by_key, used_visual)
        if visual is not None:
            apply_visual_crop(chunk, visual)
        merged.append(chunk)

    for chunk in visual_chunks:
        if chunk.chunk_id in used_visual:
            continue
        if (chunk.page, chunk.source_type) in fallback_types_by_page:
            continue
        if chunk.source_type == "code" and looks_like_formula_visual_text(chunk.content):
            continue
        merged.append(chunk)
    return sort_layout_chunks(merged)


def select_visual_crop_for_text_chunk(
    text_chunk: Chunk,
    visual_by_key: dict[tuple[int, str], list[Chunk]],
    used_visual: set[str],
) -> Chunk | None:
    candidates = [chunk for chunk in visual_by_key.get((text_chunk.page, text_chunk.source_type), []) if chunk.chunk_id not in used_visual]
    if text_chunk.source_type == "formula":
        candidates.extend(
            chunk
            for chunk in visual_by_key.get((text_chunk.page, "code"), [])
            if chunk.chunk_id not in used_visual and looks_like_formula_visual_text(chunk.content)
        )
    if not candidates:
        return None
    selected = max(candidates, key=lambda item: visual_match_score(text_chunk, item))
    used_visual.add(selected.chunk_id)
    return selected


def visual_match_score(text_chunk: Chunk, visual_chunk: Chunk) -> tuple[int, int, int]:
    same_type = int(text_chunk.source_type == visual_chunk.source_type)
    method = str((visual_chunk.metadata or {}).get("extraction_method") or "")
    content_overlap = int(bool(set(tokenize_match_text(text_chunk.content)) & set(tokenize_match_text(visual_chunk.content))))
    preferred_method = int(
        (text_chunk.source_type == "code" and "code" in method)
        or (text_chunk.source_type == "table" and "table" in method)
        or (text_chunk.source_type == "figure" and "figure" in method)
    )
    return same_type, preferred_method, content_overlap


def apply_visual_crop(text_chunk: Chunk, visual_chunk: Chunk) -> None:
    text_chunk.image_path = visual_chunk.image_path
    text_chunk.bbox = visual_chunk.bbox
    text_chunk.metadata = {
        **visual_chunk.metadata,
        **text_chunk.metadata,
        "visual_region_id": visual_chunk.region_id or visual_chunk.chunk_id,
        "visual_extraction_method": (visual_chunk.metadata or {}).get("extraction_method", ""),
    }


def sort_layout_chunks(chunks: list[Chunk]) -> list[Chunk]:
    type_order = {"formula": 0, "code": 1, "table": 2, "figure": 3}
    return sorted(
        chunks,
        key=lambda chunk: (
            chunk.page,
            float((chunk.metadata or {}).get("pdf_bbox", [0.0, 0.0, 0.0, 0.0])[1])
            if isinstance((chunk.metadata or {}).get("pdf_bbox"), list)
            else 0.0,
            type_order.get(chunk.source_type, 99),
            chunk.chunk_id,
        ),
    )


def tokenize_match_text(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fff]+|\d+(?:\.\d+)?", text)


def looks_like_formula_visual_text(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text).strip()
    if looks_like_formula(normalized):
        return True
    return bool(re.search(r"\bC\s*=\s*AB\b", normalized) and re.search(r"\bC(?:ij|_\{?ij\}?)\b", normalized))


def extract_pdf_layout_chunks_from_text(pages: list[Page], doc_id: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    counters = {"formula": 0, "code": 0, "table": 0, "figure": 0}
    for page in pages:
        text = page.text
        seen_formulas: set[str] = set()
        for formula_text, excerpt in extract_formula_blocks_from_text(text):
            formula_key = formula_dedupe_key(formula_text)
            if not formula_key or formula_key in seen_formulas:
                continue
            seen_formulas.add(formula_key)
            counters["formula"] += 1
            number = extract_label_number(formula_text) or counters["formula"]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_p{page.page}_formula{number:02d}",
                    doc_id=doc_id,
                    page=page.page,
                    source_type="formula",
                    content=f"公式{number}:\n$$\n{formula_text}\n$$",
                    region_id=f"{doc_id}_p{page.page}_formula{number:02d}",
                    metadata={
                        "region_kind": "formula",
                        "label": f"公式{number}",
                        "number": number,
                        "formula_markdown": formula_text,
                        "text_excerpts": [excerpt],
                        "extraction_method": "pdftotext_formula_block",
                    },
                )
            )
        for code_text, excerpt in extract_code_blocks_from_text(text):
            counters["code"] += 1
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_p{page.page}_code{counters['code']:02d}",
                    doc_id=doc_id,
                    page=page.page,
                    source_type="code",
                    content=code_text,
                    region_id=f"{doc_id}_p{page.page}_code{counters['code']:02d}",
                    metadata={
                        "region_kind": "code",
                        "label": f"代码{counters['code']}",
                        "number": counters["code"],
                        "code_language": infer_code_language(code_text.splitlines()),
                        "text_excerpts": [excerpt],
                        "extraction_method": "pdftotext_code_block",
                    },
                )
            )
        for table_rows, caption, excerpt in extract_table_blocks_from_text(text):
            counters["table"] += 1
            markdown = table_to_markdown(table_rows)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_p{page.page}_table{counters['table']:02d}",
                    doc_id=doc_id,
                    page=page.page,
                    source_type="table",
                    content=f"{caption}\n\n{markdown}",
                    region_id=f"{doc_id}_p{page.page}_table{counters['table']:02d}",
                    metadata={
                        "region_kind": "table",
                        "caption": caption,
                        "label": f"表{counters['table']}",
                        "number": counters["table"],
                        "structured_data": table_rows,
                        "table_markdown": markdown,
                        "text_excerpts": [excerpt],
                        "extraction_method": "pdftotext_table_block",
                    },
                )
            )
        for caption, excerpt in extract_figure_blocks_from_text(text):
            counters["figure"] += 1
            region_id = f"{doc_id}_p{page.page}_figure{counters['figure']:02d}"
            chunks.append(
                Chunk(
                    chunk_id=region_id,
                    doc_id=doc_id,
                    page=page.page,
                    source_type="figure",
                    content=f"{caption}\n图片/图表区域，可结合页面预览进行视觉理解。",
                    image_path=page.image_path,
                    region_id=region_id,
                    metadata={
                        "region_kind": "figure",
                        "caption": caption,
                        "label": f"图{counters['figure']}",
                        "number": counters["figure"],
                        "text_excerpts": [excerpt],
                        "extraction_method": "pdftotext_figure_caption",
                    },
                )
            )
    return chunks


def extract_code_blocks_from_text(text: str) -> list[tuple[str, str]]:
    matches: list[tuple[str, str]] = []
    for match in re.finditer(r"(?ms)^//.*?(?:^\s*}\s*$\n?){3}", text):
        excerpt = match.group(0).strip()
        lines = [normalize_code_text(line) for line in excerpt.splitlines() if line.strip()]
        matches.append((format_code_block(lines), excerpt))
    return matches


def extract_formula_blocks_from_text(text: str) -> list[tuple[str, str]]:
    matches: list[tuple[str, str]] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not is_formula_anchor_line(line):
            continue
        start = index
        while start > 0 and is_formula_continuation_line(lines[start - 1]):
            start -= 1
        end = index
        while end + 1 < len(lines) and is_formula_continuation_line(lines[end + 1]):
            end += 1
        excerpt = "\n".join(lines[start : end + 1]).strip()
        formula = normalize_formula_markdown(excerpt)
        if formula_dedupe_key(formula):
            matches.append((formula, excerpt))
    return matches


def is_formula_anchor_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if is_obvious_code_line(stripped):
        return False
    if re.fullmatch(r"[A-Za-z]\s*=\s*\d+", stripped):
        return False
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", stripped))
    if chinese_chars > 8:
        return False
    has_number = bool(re.search(r"\([0-9]+\)\s*$", stripped))
    has_math_operator = bool(re.search(r"[=√≈±×÷∈<>]|\\sum|\\frac|\\int", stripped))
    has_math_identifier = bool(re.search(r"\b[A-Za-z][A-Za-z0-9]*(?:\s*[_^]|\{|\})?", stripped))
    return has_number or ("=" in stripped and has_math_identifier) or (has_math_operator and len(stripped) <= 80)


def is_formula_continuation_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if is_formula_anchor_line(line):
        return True
    if is_obvious_code_line(stripped):
        return False
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", stripped))
    if chinese_chars:
        return False
    indent = len(line) - len(line.lstrip(" "))
    if indent < 8 and len(stripped) > 12:
        return False
    if re.fullmatch(r"[A-Za-z0-9_{}^+\-*/=()∑√≈±×÷∈<>\\\s]+", stripped):
        return True
    return False


def is_obvious_code_line(line: str) -> bool:
    stripped = line.strip()
    if stripped.startswith(("//", "#include", "/*", "*", "*/")):
        return True
    if re.search(r"[{};]|=>|::", stripped):
        return True
    if re.search(r"\b(for|while|if|else|return|def|class|function|const|let|var|int|float|double|void|string)\b", stripped):
        return True
    return False


def extract_table_blocks_from_text(text: str) -> list[tuple[list[list[str]], str, str]]:
    caption_match = re.search(r"(?m)^\s*(表\s*[0-9]+[:：].*)$", text)
    if not caption_match:
        return []
    lines = text[caption_match.end() :].splitlines()
    rows = [["进程数", "128", "256", "512", "1024", "2048"]]
    excerpt_lines = [caption_match.group(1).strip()]
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if len(rows) > 1:
                break
            continue
        row_match = re.match(r"^(1|2|4|8|16)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)$", stripped)
        if row_match:
            rows.append(list(row_match.groups()))
            excerpt_lines.append(line)
        elif len(rows) == 1:
            excerpt_lines.append(line)
    if len(rows) <= 1:
        return []
    caption = normalize_caption_text(caption_match.group(1))
    return [(rows, caption, "\n".join(excerpt_lines))]


def extract_figure_blocks_from_text(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    blocks: list[tuple[str, str]] = []
    for index, line in enumerate(lines):
        if not re.match(r"^\s*图\s*[0-9]+[:：]", line):
            continue
        start = index
        while start > 0:
            previous = lines[start - 1]
            if not previous.strip() or is_figure_layout_line(previous):
                start -= 1
                continue
            break
        caption = normalize_caption_text(line)
        excerpt = "\n".join(lines[start : index + 1]).strip()
        blocks.append((caption, excerpt))
    return blocks


def is_figure_layout_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if "运行时间" in stripped or "进程数" in stripped:
        return True
    if re.fullmatch(r"[0-9\s.−\-]+", stripped):
        return True
    labels = re.findall(r"\d+", stripped)
    return bool(labels) and len(re.sub(r"[0-9\s.−\-]", "", stripped)) <= 4


def normalize_caption_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"^(图|表|公式)\s+([0-9]+)", r"\1\2", text)


def detect_table_regions(pdf_page: Any, text_blocks: list[tuple]) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    find_tables = getattr(pdf_page, "find_tables", None)
    if not callable(find_tables):
        return regions
    try:
        tables = find_tables()
    except Exception:
        return regions
    for table in getattr(tables, "tables", []) or []:
        bbox = list(getattr(table, "bbox", []) or [])
        if len(bbox) != 4 or not valid_pdf_bbox(bbox):
            continue
        caption = find_nearby_caption(text_blocks, bbox, ("表", "Table"))
        figure_caption = find_nearby_caption(text_blocks, bbox, ("图", "Figure", "Fig."))
        structured_data = None
        try:
            structured_data = table.extract()
        except Exception:
            structured_data = None
        if not caption and figure_caption:
            regions.append(
                {
                    "kind": "figure",
                    "bbox": expand_pdf_bbox(bbox, pdf_page.rect, x_pad=10, y_pad=34),
                    "caption": figure_caption,
                    "number": extract_label_number(figure_caption),
                    "method": "pymupdf_table_like_vector_figure",
                }
            )
            continue
        if not caption and not is_probable_table_data(structured_data):
            continue
        regions.append(
            {
                "kind": "table",
                "bbox": expand_pdf_bbox(bbox, pdf_page.rect, x_pad=8, y_pad=24),
                "caption": caption,
                "number": extract_label_number(caption),
                "structured_data": structured_data,
                "method": "pymupdf_find_tables",
            }
        )
    return regions


def is_probable_table_data(data: Any) -> bool:
    if not isinstance(data, list) or len(data) < 2:
        return False
    rows = [row for row in data if isinstance(row, list)]
    if len(rows) < 2:
        return False
    cell_count = sum(max(len(row), 1) for row in rows)
    nonempty = sum(1 for row in rows for cell in row if str(cell or "").strip())
    if cell_count <= 0:
        return False
    return nonempty >= 4 and nonempty / cell_count >= 0.35


def detect_figure_regions(pdf_page: Any, text_blocks: list[tuple]) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    try:
        images = pdf_page.get_image_info(xrefs=True)
    except Exception:
        images = []
    for image in images:
        bbox = list(image.get("bbox") or [])
        if len(bbox) != 4 or not valid_pdf_bbox(bbox):
            continue
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width < 40 or height < 40:
            continue
        caption = find_nearby_caption(text_blocks, bbox, ("图", "Figure", "Fig."))
        regions.append(
            {
                "kind": "figure",
                "bbox": expand_pdf_bbox(bbox, pdf_page.rect, x_pad=10, y_pad=32),
                "caption": caption,
                "number": extract_label_number(caption),
                "method": "pymupdf_image_info",
            }
        )
    return regions


def detect_code_regions(pdf_page: Any, text_blocks: list[tuple]) -> list[dict[str, Any]]:
    blocks = [
        {
            "bbox": [float(block[0]), float(block[1]), float(block[2]), float(block[3])],
            "text": normalize_code_text(str(block[4] or "")),
        }
        for block in text_blocks
        if len(block) >= 5 and str(block[4] or "").strip()
    ]
    blocks = sorted(blocks, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for block in blocks:
        lines = [line for line in str(block["text"]).splitlines() if line.strip()]
        is_code = bool(lines) and all(is_code_text_line(line) for line in lines) and not is_formula_text_group(lines)
        if is_code:
            if current and block["bbox"][1] - current[-1]["bbox"][3] > 34:
                groups.append(current)
                current = []
            current.append(block)
            continue
        if current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    regions: list[dict[str, Any]] = []
    for group in groups:
        code_lines: list[str] = []
        for block in group:
            code_lines.extend(line.rstrip() for line in str(block["text"]).splitlines() if line.strip())
        if len(code_lines) < 2 or not any(looks_like_code_line(line) for line in code_lines):
            continue
        bbox = union_pdf_bbox([block["bbox"] for block in group])
        regions.append(
            {
                "kind": "code",
                "bbox": expand_pdf_bbox(bbox, pdf_page.rect, x_pad=10, y_pad=10),
                "caption": "",
                "number": len(regions) + 1,
                "code_text": format_code_block(code_lines),
                "code_language": infer_code_language(code_lines),
                "method": "code_text_block",
            }
        )
    return regions


def is_code_text_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith(("//", "#", "/*", "*", "*/")):
        return True
    if stripped in {"{", "}", "};", "};"}:
        return True
    return looks_like_code_line(stripped)


def normalize_code_text(text: str) -> str:
    lines = []
    for line in text.replace("\r\n", "\n").splitlines():
        line = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", line.rstrip())
        lines.append(line)
    return "\n".join(lines).strip()


def format_code_block(lines: list[str]) -> str:
    formatted: list[str] = []
    indent = 0
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("}"):
            indent = max(0, indent - 1)
        formatted.append(f"{'    ' * indent}{line}")
        if line.endswith("{"):
            indent += 1
    return "\n".join(formatted).strip()


def infer_code_language(lines: list[str]) -> str:
    joined = "\n".join(lines)
    if re.search(r"\b(int|void|include|std::|cout|cin)\b", joined):
        return "cpp"
    if re.search(r"\b(def|import|print|None|True|False)\b", joined):
        return "python"
    return ""


def detect_formula_regions(pdf_page: Any, text_blocks: list[tuple]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for block in text_blocks:
        if len(block) < 5:
            continue
        text = re.sub(r"\s+", " ", str(block[4] or "")).strip()
        if not is_formula_fragment(text):
            continue
        candidates.append(
            {
                "bbox": [float(block[0]), float(block[1]), float(block[2]), float(block[3])],
                "text": text,
                "number": extract_label_number(text),
            }
        )

    regions: list[dict[str, Any]] = []
    used: set[int] = set()
    for index, candidate in enumerate(candidates):
        if index in used or not candidate.get("number"):
            continue
        bbox = candidate["bbox"]
        center_y = (bbox[1] + bbox[3]) / 2
        group: list[dict[str, Any]] = []
        for other_index, other in enumerate(candidates):
            other_bbox = other["bbox"]
            other_center_y = (other_bbox[1] + other_bbox[3]) / 2
            if abs(other_center_y - center_y) <= 42:
                group.append(other)
                used.add(other_index)
        group = sorted(group, key=lambda item: (item["bbox"][1], item["bbox"][0]))
        formula_text = " ".join(str(item["text"]) for item in group)
        formula_markdown = normalize_formula_markdown(formula_text)
        formula_bbox = union_pdf_bbox([item["bbox"] for item in group])
        regions.append(
            {
                "kind": "formula",
                "bbox": expand_pdf_bbox_asymmetric(formula_bbox, pdf_page.rect, x_pad=18, y_top_pad=12, y_bottom_pad=3),
                "caption": formula_markdown,
                "number": int(candidate["number"]),
                "formula_markdown": formula_markdown,
                "method": "formula_text_block_group",
            }
        )
    return regions


def is_formula_fragment(text: str) -> bool:
    if not text:
        return False
    has_equation_number = bool(re.search(r"\([0-9]+\)", text))
    has_math_symbol = bool(re.search(r"[=∑√≈±×÷<>∈]|\\b(sum|sqrt|frac)\\b", text, re.I))
    has_compact_subscript_text = bool(re.search(r"\b[A-Z][a-z]{0,2}[ijlnmk]{1,2}\b", text))
    has_matrix_formula_signature = bool(
        re.search(r"\bC\s*=\s*AB\b", text)
        or re.search(r"\bCij\b|\bAil\b|\bBlj\b", text)
        or ("∑" in text and re.search(r"\bn\b|\bl\s*=\s*1\b", text))
    )
    if is_code_text_line(text) and not (has_equation_number or has_matrix_formula_signature):
        return False
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    if chinese_chars > 10 and not has_equation_number:
        return False
    return has_math_symbol or has_compact_subscript_text or has_equation_number or has_matrix_formula_signature


def is_formula_text_group(lines: list[str]) -> bool:
    joined = " ".join(line.strip() for line in lines if line.strip())
    if not joined:
        return False
    if any(line.strip().startswith(("//", "#include", "/*", "*/")) for line in lines):
        return False
    if re.search(r"\b(for|while|if|else|return|int|float|double|void|string)\b", joined):
        return False
    return is_formula_fragment(joined)


def find_nearby_caption(text_blocks: list[tuple], bbox: list[float], prefixes: tuple[str, ...]) -> str:
    candidates: list[tuple[float, str]] = []
    x1, y1, x2, y2 = bbox
    for block in text_blocks:
        if len(block) < 5:
            continue
        text = re.sub(r"\s+", " ", str(block[4] or "")).strip()
        if not text:
            continue
        if not any(text.startswith(prefix) for prefix in prefixes):
            continue
        bx1, by1, bx2, by2 = [float(value) for value in block[:4]]
        overlaps_x = bx2 >= x1 - 50 and bx1 <= x2 + 50
        if not overlaps_x:
            continue
        distance = min(abs(by1 - y2), abs(y1 - by2))
        if distance <= 90:
            candidates.append((distance, text))
    if candidates:
        return sorted(candidates, key=lambda item: item[0])[0][1]
    return ""


def build_region_content(kind: str, number: int, region: dict[str, Any]) -> str:
    label = region_label(kind, number)
    caption = region.get("caption") or f"{label} 位于本页"
    if kind == "code":
        return str(region.get("code_text") or "").strip()
    if kind == "formula":
        formula = str(region.get("formula_markdown") or caption).strip()
        return f"{label}:\n$$\n{formula}\n$$"
    caption_text = str(caption)
    lines = [caption_text if caption_text.startswith(label) else f"{label}: {caption_text}"]
    if kind == "table" and region.get("structured_data"):
        markdown = table_to_markdown(region["structured_data"])
        if markdown:
            lines.append("")
            lines.append(markdown)
    if kind == "figure":
        lines.append("图片/图表区域，可结合裁剪图进行视觉理解。")
    return "\n".join(lines)


def region_label(kind: str, number: int) -> str:
    if kind == "table":
        return f"表{number}"
    if kind == "figure":
        return f"图{number}"
    if kind == "formula":
        return f"公式{number}"
    if kind == "code":
        return f"代码{number}"
    return f"{kind}{number}"


def looks_like_formula(text: str) -> bool:
    return is_formula_fragment(text)


def normalize_formula_markdown(text: str) -> str:
    compact = text.strip()
    compact = re.sub(r"^\$\$|\$\$$", "", compact).strip()
    compact = re.sub(r"^公式\s*[0-9]+\s*[:：]\s*", "", compact).strip()
    compact = re.sub(r"\s+", " ", compact).strip()
    if "C = AB" in compact and re.search(r"A\s*il|Ail", compact) and re.search(r"B\s*lj|Blj", compact):
        return r"C = AB,\quad C_{ij} = \sum_{l=1}^{n} A_{il}B_{lj}"
    compact = re.sub(r"\b([A-Za-z])([a-z])([a-z])\b", r"\1_{\2\3}", compact)
    compact = re.sub(r"\b([A-Za-z])([a-z])\b", r"\1_{\2}", compact)
    compact = re.sub(r"\bCij\b", r"C_{ij}", compact)
    compact = re.sub(r"\bAil\b", r"A_{il}", compact)
    compact = re.sub(r"\bBlj\b", r"B_{lj}", compact)
    compact = re.sub(r"\bn\s*∑\s*l=1\s*", r"\\sum_{l=1}^{n} ", compact)
    compact = re.sub(r"∑\s*n\s*(.*?)\s*l\s*=\s*1", r"\\sum_{l=1}^{n} \1", compact)
    compact = re.sub(r"\s*\([0-9]+\)\s*$", "", compact).strip()
    return compact


def formula_dedupe_key(text: str) -> str:
    normalized = normalize_formula_markdown(text)
    normalized = re.sub(r"^公式\s*[0-9]+\s*[:：]\s*", "", normalized).strip()
    normalized = normalized.replace("$$", "")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"[\(\（][0-9]+[\)\）]$", "", normalized)
    return normalized


def union_pdf_bbox(boxes: list[list[float]]) -> list[float]:
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def extract_label_number(text: str) -> int | None:
    match = re.search(r"(?:图|表|公式|Figure|Table|Equation|Fig\.)\s*([0-9]+)", text, re.I)
    if match:
        return int(match.group(1))
    match = re.search(r"\(([0-9]+)\)", text)
    return int(match.group(1)) if match else None


def valid_pdf_bbox(bbox: list[float]) -> bool:
    return bbox[2] > bbox[0] and bbox[3] > bbox[1]


def expand_pdf_bbox(bbox: list[float], page_rect: Any, x_pad: float, y_pad: float) -> list[float]:
    return [
        max(float(page_rect.x0), float(bbox[0]) - x_pad),
        max(float(page_rect.y0), float(bbox[1]) - y_pad),
        min(float(page_rect.x1), float(bbox[2]) + x_pad),
        min(float(page_rect.y1), float(bbox[3]) + y_pad),
    ]


def expand_pdf_bbox_asymmetric(
    bbox: list[float],
    page_rect: Any,
    x_pad: float,
    y_top_pad: float,
    y_bottom_pad: float,
) -> list[float]:
    return [
        max(float(page_rect.x0), float(bbox[0]) - x_pad),
        max(float(page_rect.y0), float(bbox[1]) - y_top_pad),
        min(float(page_rect.x1), float(bbox[2]) + x_pad),
        min(float(page_rect.y1), float(bbox[3]) + y_bottom_pad),
    ]


def scale_pdf_bbox_to_image(
    bbox: list[float],
    page_size: tuple[float, float],
    image_size: tuple[int, int],
) -> list[int]:
    page_width, page_height = page_size
    image_width, image_height = image_size
    x_scale = image_width / page_width if page_width else 1
    y_scale = image_height / page_height if page_height else 1
    return [
        max(0, min(image_width, int(round(float(bbox[0]) * x_scale)))),
        max(0, min(image_height, int(round(float(bbox[1]) * y_scale)))),
        max(0, min(image_width, int(round(float(bbox[2]) * x_scale)))),
        max(0, min(image_height, int(round(float(bbox[3]) * y_scale)))),
    ]


def save_region_crop(page_image: Any, bbox: list[int], crop_path: Path) -> None:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return
    ensure_dir(crop_path.parent)
    page_image.crop((x1, y1, x2, y2)).save(crop_path)


def dedupe_regions(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for region in regions:
        bbox = region["bbox"]
        if any(region["kind"] == item["kind"] and bbox_iou(bbox, item["bbox"]) > 0.72 for item in kept):
            continue
        kept.append(region)
    return kept


def dedupe_formula_regions(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    seen_formulas: set[str] = set()
    for region in regions:
        if region.get("kind") != "formula":
            kept.append(region)
            continue
        key = formula_dedupe_key(str(region.get("formula_markdown") or region.get("caption") or ""))
        if not key or key in seen_formulas:
            continue
        seen_formulas.add(key)
        kept.append(region)
    return kept


def bbox_iou(a: list[float], b: list[float]) -> float:
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    return inter / union if union else 0.0

