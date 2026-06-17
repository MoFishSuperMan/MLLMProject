"""Table normalization helpers for Markdown-ready evidence chunks."""

from __future__ import annotations

import re
from typing import Any

from .vision_regions import OcrBox


def table_to_markdown(rows: list[list[Any]] | tuple[tuple[Any, ...], ...] | None) -> str:
    if not rows:
        return ""
    normalized = normalize_rows(rows)
    if not normalized:
        return ""
    width = max(len(row) for row in normalized)
    padded = [row + [""] * (width - len(row)) for row in normalized]
    header = padded[0]
    body = padded[1:] if len(padded) > 1 else []
    divider = ["---"] * width
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(divider) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def normalize_rows(rows: Any) -> list[list[str]]:
    normalized: list[list[str]] = []
    for raw_row in rows:
        if raw_row is None:
            continue
        if not isinstance(raw_row, (list, tuple)):
            raw_row = [raw_row]
        row = [normalize_cell(cell) for cell in raw_row]
        if any(cell for cell in row):
            normalized.append(row)
    return normalized


def normalize_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("|", "\\|")


def ocr_boxes_to_markdown(boxes: list[OcrBox], y_tolerance: int = 14) -> str:
    """Reconstruct a simple Markdown table from OCR boxes grouped by rows."""

    if not boxes:
        return ""
    sorted_boxes = sorted(boxes, key=lambda box: ((box.bbox[1] + box.bbox[3]) / 2, box.bbox[0]))
    rows: list[list[OcrBox]] = []
    centers: list[float] = []
    for box in sorted_boxes:
        center_y = (box.bbox[1] + box.bbox[3]) / 2
        target = None
        for index, row_center in enumerate(centers):
            if abs(center_y - row_center) <= y_tolerance:
                target = index
                break
        if target is None:
            rows.append([box])
            centers.append(center_y)
        else:
            rows[target].append(box)
            centers[target] = (centers[target] + center_y) / 2
    text_rows = [[box.text for box in sorted(row, key=lambda item: item.bbox[0])] for row in rows]
    return table_to_markdown(text_rows)

