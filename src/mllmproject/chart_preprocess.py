"""ChartQA-specific preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from PIL import Image, ImageChops

from .vision_regions import BBoxInt, OcrResult, RegionCandidate, clamp_bbox


@dataclass(slots=True)
class ChartInfo:
    question_type: str
    numbers: list[str]
    labels: list[str]
    page_bbox: BBoxInt

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_type": self.question_type,
            "numbers": self.numbers,
            "labels": self.labels,
            "page_bbox": self.page_bbox,
        }


def classify_chart_question(question: str) -> str:
    normalized = question.lower()
    if any(pattern in normalized for pattern in ("how many more", "how much more", "difference", "subtract", "minus", "gap")):
        return "difference"
    if any(pattern in normalized for pattern in ("how many", "number of", "count")):
        return "count"
    if any(pattern in normalized for pattern in ("more than", "less than", "higher", "lower", "greater")):
        return "compare"
    if any(pattern in normalized for pattern in ("lowest", "highest", "minimum", "maximum", "smallest", "largest")):
        return "minmax"
    if any(pattern in normalized for pattern in ("value", "what's", "what is")):
        return "value"
    return "other"


def build_chart_candidate(
    question: str,
    ocr_result: OcrResult,
    page_image: Image.Image,
    doc_id: str,
) -> tuple[RegionCandidate, ChartInfo]:
    bbox = trim_whitespace_bbox(page_image)
    numbers = extract_chart_numbers(ocr_result)
    labels = extract_chart_labels(ocr_result)
    question_type = classify_chart_question(question)
    candidate = RegionCandidate(
        region_id=f"{doc_id}_p1_chart_region",
        page=1,
        source_type="chart_region",
        bbox=bbox,
        image_path="",
        reason=f"ChartQA {question_type} route",
        score=1.0,
    )
    return candidate, ChartInfo(question_type=question_type, numbers=numbers, labels=labels, page_bbox=bbox)


def trim_whitespace_bbox(image: Image.Image, threshold: int = 245, padding: int = 12) -> BBoxInt:
    rgb = image.convert("RGB")
    white = Image.new("RGB", rgb.size, (255, 255, 255))
    diff = ImageChops.difference(rgb, white).convert("L")
    mask = diff.point(lambda pixel: 255 if pixel > 255 - threshold else 0)
    bbox = mask.getbbox()
    if not bbox:
        return [0, 0, rgb.width, rgb.height]
    x1, y1, x2, y2 = bbox
    clamped = clamp_bbox([x1 - padding, y1 - padding, x2 + padding, y2 + padding], rgb.width, rgb.height)
    return clamped or [0, 0, rgb.width, rgb.height]


def extract_chart_numbers(ocr_result: OcrResult) -> list[str]:
    seen: set[str] = set()
    numbers: list[str] = []
    for match in re.finditer(r"(?<!\d)[-+]?\d+(?:\.\d+)?%?", ocr_result.text):
        value = match.group(0)
        if value not in seen:
            seen.add(value)
            numbers.append(value)
    return numbers


def extract_chart_labels(ocr_result: OcrResult, limit: int = 40) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for box in ocr_result.boxes:
        text = " ".join(box.text.split())
        if not text or re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", text):
            continue
        if text.lower() in seen:
            continue
        seen.add(text.lower())
        labels.append(text)
        if len(labels) >= limit:
            break
    return labels


def chart_region_content(question: str, chart_info: ChartInfo, ocr_text: str) -> str:
    lines = [
        f"ChartQA region for question: {question}",
        f"Question type: {chart_info.question_type}",
        f"Chart page bbox: {chart_info.page_bbox}",
        "Candidate numbers: " + (", ".join(chart_info.numbers) if chart_info.numbers else "(none)"),
        "Candidate labels: " + (", ".join(chart_info.labels[:20]) if chart_info.labels else "(none)"),
    ]
    if ocr_text.strip():
        lines.append("OCR in chart region:\n" + ocr_text.strip())
    return "\n".join(lines)
