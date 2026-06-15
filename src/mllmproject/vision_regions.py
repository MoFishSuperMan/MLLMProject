"""OCR-box aware region selection and crop helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Iterable

from PIL import Image

from .io_utils import ensure_dir
from .schemas import Chunk, Document


BBoxInt = list[int]


@dataclass(slots=True)
class OcrBox:
    text: str
    bbox: BBoxInt
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OcrResult:
    text: str
    boxes: list[OcrBox]

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "boxes": [box.to_dict() for box in self.boxes]}


@dataclass(slots=True)
class RegionCandidate:
    region_id: str
    page: int
    source_type: str
    bbox: BBoxInt
    image_path: str
    reason: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


RECIPIENT_HINTS = ("to whom", "sent", "addressed", "recipient", "收件", "发送给")
TIME_HINTS = ("what time", "time is", "time", "when", "几点", "时间")
CHART_HINTS = (
    "value",
    "year",
    "actual",
    "chart",
    "bar",
    "graph",
    "plot",
    "lowest",
    "highest",
    "difference",
    "sum",
    "柱",
    "图表",
)
AD_HINTS = (
    "advertise",
    "advertised",
    "advertisement",
    " ad ",
    "brand",
    "logo",
    "product",
    "fashion",
    "clothing",
    "clothes",
    "wear",
    "lifestyle",
)


STOPWORDS = {
    "what",
    "which",
    "where",
    "when",
    "whom",
    "whose",
    "does",
    "this",
    "that",
    "with",
    "from",
    "document",
    "shown",
    "value",
}


def ocr_result_from_easyocr(raw_lines: Iterable[Any]) -> OcrResult:
    boxes: list[OcrBox] = []
    for item in raw_lines:
        parsed = parse_easyocr_item(item)
        if parsed:
            boxes.append(parsed)
    text = "\n".join(box.text for box in boxes if box.text.strip())
    return OcrResult(text=text, boxes=boxes)


def parse_easyocr_item(item: Any) -> OcrBox | None:
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return None
    raw_bbox = item[0]
    text = str(item[1]).strip()
    if not text:
        return None
    confidence = float(item[2]) if len(item) > 2 and is_number(item[2]) else 0.0
    bbox = points_to_bbox(raw_bbox)
    if not bbox:
        return None
    return OcrBox(text=text, bbox=bbox, confidence=confidence)


def points_to_bbox(points: Any) -> BBoxInt | None:
    try:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
    except Exception:
        return None
    if not xs or not ys:
        return None
    return [int(round(min(xs))), int(round(min(ys))), int(round(max(xs))), int(round(max(ys)))]


def select_region_candidates(
    question: str,
    ocr_result: OcrResult,
    page_size: tuple[int, int],
    doc_id: str,
    max_regions: int = 2,
) -> list[RegionCandidate]:
    width, height = page_size
    if width <= 0 or height <= 0 or max_regions <= 0:
        return []

    candidates: list[RegionCandidate] = []
    normalized = question.lower()
    target_candidates = target_numeric_candidates(
        question=question,
        ocr_result=ocr_result,
        page_size=page_size,
        doc_id=doc_id,
    )
    candidates.extend(target_candidates)

    keywords = question_keywords(question)
    matched_boxes = [
        box
        for box in ocr_result.boxes
        if any(keyword in box.text.lower() for keyword in keywords)
    ]
    if matched_boxes:
        bbox = expand_bbox(union_bbox([box.bbox for box in matched_boxes]), width, height, x_pad=0.12, y_pad=0.20)
        candidates.append(
            RegionCandidate(
                region_id=f"{doc_id}_p1_region_keyword",
                page=1,
                source_type="region",
                bbox=bbox,
                image_path="",
                reason="OCR keyword neighborhood",
                score=0.95,
            )
        )
        if any(hint in normalized for hint in TIME_HINTS):
            row_bbox = same_row_context_bbox(matched_boxes, ocr_result.boxes, width, height)
            candidates.append(
                RegionCandidate(
                    region_id=f"{doc_id}_p1_region_time_row",
                    page=1,
                    source_type="region",
                    bbox=row_bbox,
                    image_path="",
                    reason="time/table row heuristic with left time column",
                    score=1.08,
                )
            )

    if is_ad_question(normalized):
        candidates.extend(
            ad_region_candidates(
                question=question,
                ocr_result=ocr_result,
                page_size=page_size,
                doc_id=doc_id,
            )
        )

    if any(hint in normalized for hint in RECIPIENT_HINTS):
        candidates.append(
            RegionCandidate(
                region_id=f"{doc_id}_p1_region_recipient",
                page=1,
                source_type="region",
                bbox=[0, 0, width, max(1, int(height * 0.58))],
                image_path="",
                reason="recipient/form header heuristic",
                score=0.90,
            )
        )

    if any(hint in normalized for hint in CHART_HINTS):
        candidates.append(
            RegionCandidate(
                region_id=f"{doc_id}_p1_region_chart",
                page=1,
                source_type="region",
                bbox=[int(width * 0.04), int(height * 0.04), int(width * 0.96), int(height * 0.96)],
                image_path="",
                reason="chart-like question heuristic",
                score=0.85,
            )
        )

    deduped = dedupe_candidates(candidates)
    return sorted(deduped, key=lambda item: item.score, reverse=True)[:max_regions]


def materialize_region_chunks(
    document: Document,
    candidates: list[RegionCandidate],
    ocr_result: OcrResult,
    source_image: Image.Image,
    page_size: tuple[int, int],
    output_dir: str | Path,
    question: str,
    max_region_side: int = 2048,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    target_dir = ensure_dir(output_dir)
    for index, candidate in enumerate(candidates, start=1):
        region_id = candidate.region_id or f"{document.doc_id}_p1_region{index}"
        crop_path = target_dir / f"{safe_stem(region_id)}.png"
        crop_size, original_bbox = save_scaled_crop(
            source_image=source_image,
            page_bbox=candidate.bbox,
            page_size=page_size,
            output_path=crop_path,
            max_region_side=max_region_side,
        )
        if not crop_size:
            continue
        candidate.image_path = str(crop_path)
        region_ocr = ocr_text_in_bbox(ocr_result.boxes, candidate.bbox)
        content = build_region_content(candidate, question, region_ocr)
        chunks.append(
            Chunk(
                chunk_id=region_id,
                doc_id=document.doc_id,
                page=candidate.page,
                source_type=candidate.source_type,
                content=content,
                bbox=[0, 0, crop_size[0], crop_size[1]],
                image_path=str(crop_path),
                region_id=region_id,
                metadata={
                    "region_kind": candidate.source_type,
                    "reason": candidate.reason,
                    "score": candidate.score,
                    "page_bbox": candidate.bbox,
                    "original_bbox": original_bbox,
                    "question": question,
                    "ocr_text": region_ocr,
                    "target_value": candidate.reason.replace("target numeric focus: ", "")
                    if candidate.reason.startswith("target numeric focus:")
                    else "",
                },
            )
        )
    return chunks


def target_numeric_candidates(
    question: str,
    ocr_result: OcrResult,
    page_size: tuple[int, int],
    doc_id: str,
) -> list[RegionCandidate]:
    width, height = page_size
    normalized = question.lower()
    if not any(hint in normalized for hint in CHART_HINTS):
        return []

    targets = [
        value
        for value in re.findall(r"\b\d{4}\b|\b\d+(?:\.\d+)?\b", question)
        if not re.fullmatch(r"1000", value)
    ]
    if not targets:
        return []

    candidates: list[RegionCandidate] = []
    for target in targets:
        matched = [box for box in ocr_result.boxes if box.text.strip() == target]
        for box in matched[:1]:
            center_x = int(round((box.bbox[0] + box.bbox[2]) / 2))
            if re.fullmatch(r"\d{4}", target):
                bbox = [
                    max(0, int(width * 0.05)),
                    max(0, int(height * 0.15)),
                    min(width, center_x + int(width * 0.12)),
                    min(height, int(height * 0.88)),
                ]
            else:
                bbox = expand_bbox(box.bbox, width, height, x_pad=2.5, y_pad=8.0)
            candidates.append(
                RegionCandidate(
                    region_id=f"{doc_id}_p1_region_target_{safe_stem(target)}",
                    page=1,
                    source_type="region",
                    bbox=bbox,
                    image_path="",
                    reason=f"target numeric focus: {target}",
                    score=1.10,
                )
            )
    return candidates


def ad_region_candidates(
    question: str,
    ocr_result: OcrResult,
    page_size: tuple[int, int],
    doc_id: str,
) -> list[RegionCandidate]:
    width, height = page_size
    keywords = question_keywords(question)
    generic = {"advertise", "advertised", "advertisement", "product", "brand", "logo", "what", "which"}
    informative_keywords = [word for word in keywords if word not in generic]
    matched_boxes = [
        box
        for box in ocr_result.boxes
        if any(keyword in box.text.lower() for keyword in informative_keywords)
    ]
    logo_boxes = [
        box
        for box in ocr_result.boxes
        if is_logo_like_text(box.text) or any(hint.strip() and hint.strip() in box.text.lower() for hint in AD_HINTS)
    ]
    candidates: list[RegionCandidate] = []
    if any(hint in question.lower() for hint in ("fashion", "clothing", "clothes", "wear", "lifestyle")):
        bottom_logo_boxes = [box for box in logo_boxes if ((box.bbox[1] + box.bbox[3]) / 2) >= height * 0.45]
        if bottom_logo_boxes:
            bbox = expand_bbox(
                union_bbox([box.bbox for box in bottom_logo_boxes]),
                width,
                height,
                x_pad=1.1,
                y_pad=1.5,
            )
            candidates.append(
                RegionCandidate(
                    region_id=f"{doc_id}_p1_region_ad_logo",
                    page=1,
                    source_type="region",
                    bbox=bbox,
                    image_path="",
                    reason="fashion/clothing advertisement brand-logo focus crop; read the brand name in the logo area",
                    score=1.18,
                )
            )
    anchor_boxes = matched_boxes or logo_boxes
    if anchor_boxes:
        bbox = expand_bbox(union_bbox([box.bbox for box in anchor_boxes[:8]]), width, height, x_pad=1.6, y_pad=1.8)
        candidates.append(
            RegionCandidate(
                region_id=f"{doc_id}_p1_region_ad_product",
                page=1,
                source_type="region",
                bbox=bbox,
                image_path="",
                reason="advertisement/product/brand focus crop; inspect nearby logo, product name, and headline",
                score=1.14,
            )
        )
    candidates.append(
        RegionCandidate(
            region_id=f"{doc_id}_p1_region_ad_layout",
            page=1,
            source_type="region",
            bbox=[0, 0, width, height],
            image_path="",
            reason="advertisement/product/brand layout fallback crop; inspect logo, headline, and product packaging",
            score=1.03,
        )
    )
    return candidates


def is_ad_question(normalized_question: str) -> bool:
    padded = f" {normalized_question} "
    return any(hint in padded for hint in AD_HINTS)


def is_logo_like_text(text: str) -> bool:
    stripped = re.sub(r"[^A-Za-z0-9&]+", "", text)
    if len(stripped) < 4:
        return False
    letters = [char for char in stripped if char.isalpha()]
    if len(letters) < 4:
        return False
    uppercase = sum(1 for char in letters if char.isupper())
    return uppercase / len(letters) >= 0.65


def same_row_context_bbox(
    anchors: list[OcrBox],
    boxes: list[OcrBox],
    width: int,
    height: int,
) -> BBoxInt:
    anchor_bbox = union_bbox([box.bbox for box in anchors])
    anchor_center_y = (anchor_bbox[1] + anchor_bbox[3]) / 2
    row_height = max(anchor_bbox[3] - anchor_bbox[1], 24)
    y_pad = max(int(row_height * 2.4), 56)
    row_boxes = [
        box
        for box in boxes
        if abs(((box.bbox[1] + box.bbox[3]) / 2) - anchor_center_y) <= y_pad
    ]
    bbox = union_bbox([box.bbox for box in row_boxes]) if row_boxes else anchor_bbox
    expanded = [
        0,
        max(0, bbox[1] - y_pad),
        min(width, max(bbox[2] + int(width * 0.05), int(width * 0.62))),
        min(height, bbox[3] + y_pad),
    ]
    return clamp_bbox(expanded, width, height) or bbox


def save_scaled_crop(
    source_image: Image.Image,
    page_bbox: BBoxInt,
    page_size: tuple[int, int],
    output_path: str | Path,
    max_region_side: int = 2048,
) -> tuple[tuple[int, int] | None, BBoxInt | None]:
    source = source_image.convert("RGB")
    original_bbox = scale_bbox(page_bbox, from_size=page_size, to_size=source.size)
    original_bbox = clamp_bbox(original_bbox, source.width, source.height)
    if not original_bbox:
        return None, None
    crop = source.crop(tuple(original_bbox))
    if max_region_side and max(crop.size) > max_region_side:
        crop.thumbnail((max_region_side, max_region_side))
    ensure_dir(Path(output_path).parent)
    crop.save(output_path)
    return crop.size, original_bbox


def scale_bbox(bbox: BBoxInt, from_size: tuple[int, int], to_size: tuple[int, int]) -> BBoxInt:
    from_width, from_height = from_size
    to_width, to_height = to_size
    if from_width <= 0 or from_height <= 0:
        return bbox
    x_scale = to_width / from_width
    y_scale = to_height / from_height
    return [
        int(round(bbox[0] * x_scale)),
        int(round(bbox[1] * y_scale)),
        int(round(bbox[2] * x_scale)),
        int(round(bbox[3] * y_scale)),
    ]


def clamp_bbox(bbox: BBoxInt, width: int, height: int) -> BBoxInt | None:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width, int(x1)))
    y1 = max(0, min(height, int(y1)))
    x2 = max(0, min(width, int(x2)))
    y2 = max(0, min(height, int(y2)))
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def ocr_text_in_bbox(boxes: list[OcrBox], bbox: BBoxInt) -> str:
    x1, y1, x2, y2 = bbox
    selected = []
    for box in boxes:
        cx = (box.bbox[0] + box.bbox[2]) / 2
        cy = (box.bbox[1] + box.bbox[3]) / 2
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            selected.append(box.text)
    return "\n".join(selected)


def build_region_content(candidate: RegionCandidate, question: str, region_ocr: str) -> str:
    parts = [
        f"Region evidence for question: {question}",
        f"Region type: {candidate.source_type}",
        f"Reason: {candidate.reason}",
        f"Page bbox: {candidate.bbox}",
    ]
    if region_ocr.strip():
        parts.append("OCR in region:\n" + region_ocr.strip())
    return "\n".join(parts)


def question_keywords(question: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]{2,}", question.lower())
    return [word for word in words if word not in STOPWORDS]


def union_bbox(bboxes: list[BBoxInt]) -> BBoxInt:
    return [
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    ]


def expand_bbox(bbox: BBoxInt, width: int, height: int, x_pad: float = 0.08, y_pad: float = 0.12) -> BBoxInt:
    box_width = bbox[2] - bbox[0]
    box_height = bbox[3] - bbox[1]
    expanded = [
        int(bbox[0] - box_width * x_pad),
        int(bbox[1] - box_height * y_pad),
        int(bbox[2] + box_width * x_pad),
        int(bbox[3] + box_height * y_pad),
    ]
    return clamp_bbox(expanded, width, height) or bbox


def dedupe_candidates(candidates: list[RegionCandidate]) -> list[RegionCandidate]:
    seen: set[tuple[int, int, int, int, str]] = set()
    deduped: list[RegionCandidate] = []
    for candidate in candidates:
        key = (*candidate.bbox, candidate.source_type)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "region"


def is_number(value: Any) -> bool:
    try:
        float(value)
    except Exception:
        return False
    return True
