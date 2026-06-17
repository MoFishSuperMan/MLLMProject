from __future__ import annotations

import re

from .schemas import Chunk, PageText


def _paragraphs(text: str) -> list[str]:
    text = text.replace("\r\n", "\n")
    parts = re.split(r"\n\s*\n+", text)
    paragraphs = [normalize_paragraph(part) for part in parts if part.strip()]
    if paragraphs:
        return paragraphs
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines


def normalize_paragraph(text: str) -> str:
    stripped = text.strip("\n")
    if is_code_block(stripped) or is_formula_block(stripped):
        return stripped.strip()
    return re.sub(r"[ \t]+", " ", stripped).strip()


def _split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def chunk_pages(
    doc_id: str,
    pages: list[PageText],
    max_chars: int = 900,
    overlap: int = 120,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    counter = 0

    for page in pages:
        current = ""
        current_type = "text"

        def append_current() -> None:
            nonlocal counter, current, current_type
            if not current.strip():
                return
            counter += 1
            metadata = {"structured_kind": current_type} if current_type in {"code", "formula"} else {}
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_p{page.page}_c{counter:04d}",
                    doc_id=doc_id,
                    page=page.page,
                    source_type=current_type,
                    content=strip_code_fence(current.strip()) if current_type == "code" else current.strip(),
                    metadata=metadata,
                )
            )
            current = ""

        for paragraph in _paragraphs(page.text):
            paragraph_type = classify_paragraph(paragraph)
            if len(paragraph) > max_chars:
                append_current()
                for piece in _split_long_text(paragraph, max_chars, overlap):
                    counter += 1
                    source_type = paragraph_type
                    metadata = {"structured_kind": source_type} if source_type in {"code", "formula"} else {}
                    chunks.append(
                        Chunk(
                            chunk_id=f"{doc_id}_p{page.page}_c{counter:04d}",
                            doc_id=doc_id,
                            page=page.page,
                            source_type=source_type,
                            content=strip_code_fence(piece) if source_type == "code" else piece,
                            metadata=metadata,
                        )
                    )
                continue

            if current and paragraph_type != current_type:
                append_current()
            if not current:
                current_type = paragraph_type

            next_text = f"{current}\n\n{paragraph}".strip() if current else paragraph
            if len(next_text) > max_chars and current:
                append_current()
                current_type = paragraph_type
                current = paragraph
            else:
                current = next_text

        append_current()

    return chunks


def classify_paragraph(text: str) -> str:
    if is_formula_block(text):
        return "formula"
    return "code" if is_code_block(text) else "text"


def is_formula_block(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("$$") or stripped.startswith("\\["):
        return True
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", stripped))
    if chinese_chars > 4 and not re.search(r"^\s*公式\s*[0-9]+\s*[:：]|\bC\s*=\s*AB\b", stripped):
        return False
    if re.search(r"[∑]|\\sum", stripped) and not re.search(
        r"=|\([0-9]+\)|\b[A-Z]_\{?[a-z0-9]+\}?|\b[A-Z][a-z]{1,2}\b",
        stripped,
    ):
        return False
    if re.search(r"[∑√≈±×÷∈]|\\sum|\\frac|\\int", stripped):
        return len(stripped) <= 180 or bool(re.search(r"\([0-9]+\)", stripped))
    if re.search(r"\b[A-Z]_\{?[a-z0-9]+\}?\b", stripped) and "=" in stripped:
        return True
    if re.search(r"\bC\s*=\s*AB\b", stripped) and re.search(r"\bC(?:ij|_\{?ij\}?)\b", stripped):
        return True
    return False


def is_code_block(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if is_formula_block(stripped):
        return False
    if stripped.startswith("```") or stripped.startswith("~~~"):
        return True
    lines = [line for line in stripped.splitlines() if line.strip()]
    if (
        len(lines) >= 2
        and sum(1 for line in lines if line.startswith(("    ", "\t"))) / len(lines) >= 0.6
        and any(looks_like_code_line(line) for line in lines)
    ):
        return True
    code_hits = sum(1 for line in lines if looks_like_code_line(line))
    return len(lines) >= 2 and code_hits / len(lines) >= 0.45


def looks_like_code_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith(("//", "#include", "/*", "*", "*/")):
        return True
    if stripped in {"{", "}", "};"}:
        return True
    if re.search(r"\b(def|class|import|from|return|if|else|elif|for|while|try|except|function|const|let|var|public|private|void|int|float|string)\b", stripped):
        return True
    if re.search(r"</?[A-Za-z][^>]*>", stripped):
        return True
    if re.search(r"[{};]|=>|::", stripped):
        return True
    if re.search(r"[A-Za-z_][A-Za-z0-9_]*(?:\[[^\]]+\])?\s*(?:[+\-*/%]?=|<|>)", stripped):
        return True
    return bool(re.match(r"^\$?\s*(python|pip|npm|uv|git|docker|curl)\b", stripped))


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith(("```", "~~~")):
        closing = lines[-1].strip()
        if closing.startswith(("```", "~~~")):
            return "\n".join(lines[1:-1]).strip()
    return stripped
