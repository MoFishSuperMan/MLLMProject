from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mllmproject.answer_extraction import extract_short_answer, normalize_entity_answer
from mllmproject.chart_preprocess import build_chart_candidate, chart_region_content
from mllmproject.chunking import chunk_pages
from mllmproject.io_utils import ensure_dir, write_csv, write_json
from mllmproject.metrics import anls, exact_match
from mllmproject.model_stack import ModelConfig, ModelStack
from mllmproject.pipeline import RagPipeline
from mllmproject.schemas import Chunk, Document, Page, PageText
from mllmproject.vision_regions import (
    OcrBox,
    OcrResult,
    materialize_region_chunks,
    ocr_result_from_easyocr,
    ocr_text_in_bbox,
    select_region_candidates,
)


DATASET_SPECS = {
    "docvqa": {
        "repo": "lmms-lab/DocVQA",
        "config": "DocVQA",
        "split": "validation",
    },
    "chartqa": {
        "repo": "lmms-lab/ChartQA",
        "config": "default",
        "split": "test",
    },
}

MODES = ("text-rag", "mm-rag")
DEFAULT_LOCAL_VLM_PATH = ROOT / "model"
DEFAULT_VLM_MODEL = str(DEFAULT_LOCAL_VLM_PATH) if (DEFAULT_LOCAL_VLM_PATH / "config.json").exists() else "Qwen/Qwen3-VL-8B-Instruct"


@dataclass(slots=True)
class BenchmarkSample:
    dataset: str
    sample_id: str
    question: str
    answers: list[str]
    image: Any
    metadata: dict[str, Any]

    @property
    def gold_answer(self) -> str:
        return self.answers[0] if self.answers else ""


class EasyOcrCache:
    def __init__(self, languages: list[str] | None = None, gpu: bool = True) -> None:
        self.languages = languages or ["en"]
        self.gpu = gpu
        self._reader: Any | None = None

    def read(self, image_path: Path) -> OcrResult:
        if self._reader is None:
            try:
                import easyocr
            except ImportError as exc:
                raise ImportError(
                    "EasyOCR is required for benchmark OCR. "
                    'Install with `python -m pip install -e ".[benchmarks]"`.'
                ) from exc
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
        lines = self._reader.readtext(str(image_path), detail=1, paragraph=False)
        return ocr_result_from_easyocr(lines)

    def read_text(self, image_path: Path) -> str:
        return self.read(image_path).text


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DocVQA/ChartQA subset benchmarks for Text-RAG vs MM-RAG.")
    parser.add_argument("--datasets", nargs="+", default=["docvqa", "chartqa"], choices=sorted(DATASET_SPECS))
    parser.add_argument("--limit-per-dataset", type=int, default=20)
    parser.add_argument("--exclude-sample-ids", nargs="*", default=[], help="Skip these sanitized benchmark sample ids.")
    parser.add_argument(
        "--include-sample-ids",
        nargs="*",
        default=[],
        help="Only run these sanitized benchmark sample ids when provided.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-dir", default="data/eval/benchmarks/real_subset")
    parser.add_argument("--cache-dir", default="data/eval/benchmarks/cache")
    parser.add_argument("--mock", action="store_true", help="Use mock models for a fast integration smoke run.")
    parser.add_argument("--no-ocr-gpu", action="store_true", help="Run EasyOCR on CPU.")
    parser.add_argument("--vlm-model-id", default=DEFAULT_VLM_MODEL)
    parser.add_argument("--embedding-model-id", default="BAAI/bge-m3")
    parser.add_argument("--reranker-model-id", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-images", type=int, default=1)
    parser.add_argument("--max-image-side", type=int, default=1280)
    parser.add_argument("--region-cache-dir", default="data/eval/benchmarks/cache_regions")
    parser.add_argument("--max-region-side", type=int, default=2048)
    parser.add_argument("--max-regions-per-sample", type=int, default=2)
    parser.add_argument("--disable-region-crops", action="store_true")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--reranker-device", default="cpu")
    return parser.parse_args()


def load_dataset_samples(
    dataset_name: str,
    limit: int,
    exclude_sample_ids: set[str] | None = None,
    include_sample_ids: set[str] | None = None,
) -> list[BenchmarkSample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required for benchmark loading. "
            'Install with `python -m pip install -e ".[benchmarks]"`.'
        ) from exc

    spec = DATASET_SPECS[dataset_name]
    dataset = load_dataset(spec["repo"], spec["config"], split=spec["split"], streaming=False)
    exclude_sample_ids = exclude_sample_ids or set()
    include_sample_ids = include_sample_ids or set()
    samples: list[BenchmarkSample] = []
    for index, row in enumerate(dataset):
        if len(samples) >= limit:
            break
        sample = sample_from_row(dataset_name, index, row)
        if sample.sample_id in exclude_sample_ids:
            continue
        if include_sample_ids and sample.sample_id not in include_sample_ids:
            continue
        samples.append(sample)
    return samples


def sample_from_row(dataset_name: str, index: int, row: dict[str, Any]) -> BenchmarkSample:
    if dataset_name == "docvqa":
        answers = [str(answer) for answer in row.get("answers", []) if str(answer).strip()]
        raw_id = row.get("questionId") or f"{index:06d}"
        sample_id = f"docvqa_{raw_id}"
        metadata = {
            "question_types": row.get("question_types", []),
            "docId": row.get("docId"),
            "ucsf_document_id": row.get("ucsf_document_id"),
            "ucsf_document_page_no": row.get("ucsf_document_page_no"),
        }
    elif dataset_name == "chartqa":
        answers = [str(row.get("answer", "")).strip()]
        sample_id = f"chartqa_{index:06}"
        metadata = {"type": row.get("type", "")}
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return BenchmarkSample(
        dataset=dataset_name,
        sample_id=sanitize_id(sample_id),
        question=str(row["question"]),
        answers=[answer for answer in answers if answer],
        image=row["image"],
        metadata=metadata,
    )


def sanitize_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sample"


def save_sample_image(sample: BenchmarkSample, cache_dir: Path, max_image_side: int | None = 1280) -> Path:
    dataset_dir = ensure_dir(cache_dir / sample.dataset)
    image = sample.image
    suffix = ".png"
    if hasattr(image, "format") and image.format:
        suffix = "." + str(image.format).lower().replace("jpeg", "jpg")
    image_path = dataset_dir / f"{sample.sample_id}{suffix}"
    if not hasattr(image, "save"):
        raise TypeError(f"Unsupported image payload for {sample.sample_id}: {type(image)!r}")

    should_write = not image_path.exists()
    if image_path.exists() and max_image_side:
        from PIL import Image

        with Image.open(image_path) as existing:
            should_write = max(existing.size) > max_image_side
    if should_write:
        image_to_save = image
        if max_image_side and max(getattr(image, "size", (0, 0))) > max_image_side:
            image_to_save = image.copy()
            image_to_save.thumbnail((max_image_side, max_image_side))
        image_to_save.save(image_path)
    return image_path


def make_document(
    sample: BenchmarkSample,
    image_path: Path,
    ocr_text: str,
    ocr_boxes: list[OcrBox] | None = None,
) -> Document:
    from PIL import Image

    with Image.open(image_path) as image:
        width, height = image.size
    doc_id = sample.sample_id
    page = Page(
        doc_id=doc_id,
        page=1,
        text=ocr_text,
        image_path=str(image_path),
        width=int(width),
        height=int(height),
    )
    chunks = chunk_pages(doc_id, [PageText(page=1, text=ocr_text)], max_chars=900, overlap=120)
    return Document(
        doc_id=doc_id,
        source_path=str(image_path),
        file_name=image_path.name,
        file_path=str(image_path),
        pages=[page],
        chunks=chunks,
        metadata={
            "benchmark_dataset": sample.dataset,
            "sample_id": sample.sample_id,
            "ocr_text_chars": len(ocr_text),
            "ocr_box_count": len(ocr_boxes or []),
        },
    )


def make_region_chunks(
    sample: BenchmarkSample,
    document: Document,
    image_path: Path,
    ocr_result: OcrResult,
    region_cache_dir: Path,
    max_region_side: int,
    max_regions_per_sample: int,
) -> list[Chunk]:
    if max_regions_per_sample <= 0:
        return []
    from PIL import Image

    page = document.pages[0]
    if not page.width or not page.height:
        return []
    page_size = (int(page.width), int(page.height))
    with Image.open(image_path) as page_image:
        page_snapshot = page_image.convert("RGB")

    source_image = sample.image if hasattr(sample.image, "crop") else page_snapshot
    output_dir = region_cache_dir / sample.dataset / sample.sample_id

    if sample.dataset == "chartqa":
        candidate, chart_info = build_chart_candidate(
            question=sample.question,
            ocr_result=ocr_result,
            page_image=page_snapshot,
            doc_id=document.doc_id,
        )
        chunks = materialize_region_chunks(
            document=document,
            candidates=[candidate],
            ocr_result=ocr_result,
            source_image=source_image,
            page_size=page_size,
            output_dir=output_dir,
            question=sample.question,
            max_region_side=max_region_side,
        )
        if chunks:
            region_ocr = ocr_text_in_bbox(ocr_result.boxes, candidate.bbox)
            chunks[0].content = chart_region_content(sample.question, chart_info, region_ocr)
            chunks[0].metadata.update(
                {
                    "chart_question_type": chart_info.question_type,
                    "chart_numbers": chart_info.numbers,
                    "chart_labels": chart_info.labels,
                    "chart_page_bbox": chart_info.page_bbox,
                }
            )
        return chunks[:max_regions_per_sample]

    candidates = select_region_candidates(
        question=sample.question,
        ocr_result=ocr_result,
        page_size=page_size,
        doc_id=document.doc_id,
        max_regions=max_regions_per_sample,
    )
    return materialize_region_chunks(
        document=document,
        candidates=candidates,
        ocr_result=ocr_result,
        source_image=source_image,
        page_size=page_size,
        output_dir=output_dir,
        question=sample.question,
        max_region_side=max_region_side,
    )


def run_mode(
    sample: BenchmarkSample,
    document: Document,
    mode: str,
    model_stack: ModelStack,
    shared: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    include_visual = mode == "mm-rag"
    index = model_stack.create_index(embedder=shared["embedder"])
    pipeline = RagPipeline.from_document(
        document=document,
        include_visual=include_visual,
        model_stack=model_stack,
        index=index,
        reranker=shared["reranker"],
        generator=shared["generator"],
        visual_summarizer=shared["visual_summarizer"] if include_visual else None,
    )
    result, latency_ms = pipeline.answer(sample.question, mode=mode, top_k=top_k)
    raw_prediction = result.answer
    extracted_answer = extract_short_answer(raw_prediction, sample.question)
    normalized_answer = normalize_for_gold(extracted_answer, sample.answers, sample.question)
    score = score_answer(normalized_answer, sample.answers, sample.question)
    failure_label = label_failure(result.evidences, score)
    chart_metadata = first_chart_metadata(result.evidences)
    return {
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
        "question": sample.question,
        "gold_answer": sample.gold_answer,
        "gold_answers": "|".join(sample.answers),
        "gold_page": 1,
        "prediction": raw_prediction,
        "raw_prediction": raw_prediction,
        "extracted_answer": extracted_answer,
        "normalized_answer": normalized_answer,
        "mode": mode.replace("-", "_"),
        "em": score["em"],
        "anls": score["anls"],
        "latency_ms": latency_ms,
        "answer_match": score["answer_match"],
        "failure_label": failure_label,
        "route": result.route,
        "ocr_text_chars": document.metadata.get("ocr_text_chars", 0),
        "ocr_box_count": document.metadata.get("ocr_box_count", 0),
        "region_chunk_count": document.metadata.get("region_chunk_count", 0),
        "evidence_count": len(result.evidences),
        "top_evidence_types": "|".join(evidence.source_type for evidence in result.evidences),
        "top_evidence_ids": "|".join(evidence.evidence_id for evidence in result.evidences),
        "top_evidence_image_paths": "|".join(str(evidence.image_path or "") for evidence in result.evidences),
        "top_region_page_bboxes": "|".join(
            str(evidence.metadata.get("page_bbox", "")) for evidence in result.evidences
        ),
        "chart_question_type": chart_metadata.get("chart_question_type", ""),
        "chart_numbers": "|".join(str(value) for value in chart_metadata.get("chart_numbers", [])),
        "chart_labels": "|".join(str(value) for value in chart_metadata.get("chart_labels", [])[:20]),
    }


def score_answer(prediction: str, answers: list[str], question: str = "") -> dict[str, float]:
    if not answers:
        return {"em": 0.0, "anls": 0.0, "answer_match": 0.0}
    normalized_answers = [normalize_entity_answer(answer, question) for answer in answers]
    em_score = max(exact_match(prediction, answer) for answer in normalized_answers)
    anls_score = max(anls(prediction, answer) for answer in normalized_answers)
    return {
        "em": float(em_score),
        "anls": float(anls_score),
        "answer_match": 1.0 if em_score >= 1.0 or anls_score > 0.0 else 0.0,
    }


def normalize_for_gold(prediction: str, answers: list[str], question: str = "") -> str:
    normalized = normalize_entity_answer(prediction, question)
    for answer in answers:
        gold = normalize_entity_answer(answer, question)
        if entity_substring_equivalent(normalized, gold, question):
            return gold
        if numeric_equivalent(normalized, gold, question):
            return gold
    return normalized


def entity_substring_equivalent(prediction: str, gold: str, question: str = "") -> bool:
    pred_key = compact_alnum(prediction)
    gold_key = compact_alnum(gold)
    if not pred_key or not gold_key or pred_key == gold_key:
        return False
    if len(gold_key) < 4:
        return False
    if gold_key not in pred_key:
        return False
    lowered = question.lower()
    entity_hints = (
        "advertise",
        "advertised",
        "advertisement",
        "brand",
        "company",
        "name",
        "product",
        "title",
        "whose",
        "which",
        "what is the name",
    )
    return any(hint in lowered for hint in entity_hints)


def compact_alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def numeric_equivalent(prediction: str, gold: str, question: str = "") -> bool:
    pred_number = first_number(prediction)
    gold_number = first_number(gold)
    if pred_number is None or gold_number is None:
        return False
    if abs(pred_number - gold_number) <= 1e-9:
        return True
    lowered = question.lower()
    if any(hint in lowered for hint in ("percent", "percentage", "how many more", "how much more", "difference")):
        if pred_number > 1 and 0 < gold_number <= 1 and abs((pred_number / 100.0) - gold_number) <= 1e-9:
            return True
        if gold_number > 1 and 0 < pred_number <= 1 and abs(pred_number - (gold_number / 100.0)) <= 1e-9:
            return True
    if relaxed_numeric_tolerance(question) and abs(pred_number - gold_number) <= 0.1 + 1e-9:
        return True
    return False


def relaxed_numeric_tolerance(question: str = "") -> bool:
    lowered = question.lower()
    hints = (
        "average",
        "mean",
        "round",
        "approximately",
        "about",
        "estimate",
        "value",
        "difference",
        "how many more",
        "how much more",
    )
    return any(hint in lowered for hint in hints)


def first_number(text: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(text))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def label_failure(evidences: list[Any], score: dict[str, float]) -> str:
    if not evidences:
        return "no_evidence"
    if score["answer_match"] < 1.0:
        return "answer_mismatch"
    return "ok"


def first_chart_metadata(evidences: list[Any]) -> dict[str, Any]:
    for evidence in evidences:
        metadata = getattr(evidence, "metadata", {}) or {}
        if metadata.get("chart_question_type"):
            return metadata
    return {}


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["dataset"]), str(row["mode"])), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (dataset, mode), group in sorted(grouped.items()):
        count = float(len(group))
        summary_rows.append(
            {
                "dataset": dataset,
                "mode": mode,
                "count": count,
                "em": mean_field(group, "em"),
                "anls": mean_field(group, "anls"),
                "answer_match_rate": mean_field(group, "answer_match"),
                "latency_ms": mean_field(group, "latency_ms"),
            }
        )
    return summary_rows


def mean_field(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get(key, 0.0)) for row in rows) / len(rows)


def write_markdown_summary(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# DocVQA/ChartQA RAG Comparison",
        "",
        "| Dataset | Mode | Count | EM | ANLS | Answer Match | Latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {dataset} | {mode} | {count:.0f} | {em:.4f} | {anls:.4f} | {answer_match_rate:.4f} | {latency_ms:.2f} |".format(
                **row
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_model_stack(args: argparse.Namespace) -> tuple[ModelStack, dict[str, Any]]:
    config = ModelConfig(
        use_real_models=not args.mock,
        vlm_model_id=args.vlm_model_id,
        embedding_model_id=args.embedding_model_id,
        reranker_model_id=args.reranker_model_id,
        vlm_max_new_tokens=args.max_new_tokens,
        vlm_max_images=args.max_images,
        embedding_device=args.embedding_device,
        reranker_device=args.reranker_device,
    )
    stack = ModelStack(config)
    shared = {
        "embedder": stack.create_embedder(),
        "reranker": stack.create_reranker(),
        "generator": stack.create_generator(),
        "visual_summarizer": stack.create_visual_summarizer(),
    }
    return stack, shared


def main() -> None:
    configure_console()
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    cache_dir = ensure_dir(args.cache_dir)
    region_cache_dir = ensure_dir(args.region_cache_dir)
    ocr = EasyOcrCache(gpu=not args.no_ocr_gpu)
    model_stack, shared = build_model_stack(args)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda value, **_: value

    all_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for dataset_name in args.datasets:
        samples = load_dataset_samples(
            dataset_name,
            args.limit_per_dataset,
            set(args.exclude_sample_ids),
            set(args.include_sample_ids),
        )
        dataset_rows: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
        for sample in tqdm(samples, desc=f"{dataset_name} samples"):
            image_path = save_sample_image(sample, cache_dir, max_image_side=args.max_image_side)
            started = time.perf_counter()
            ocr_result = ocr.read(image_path)
            ocr_text = ocr_result.text
            document = make_document(sample, image_path, ocr_text, ocr_result.boxes)
            manifest.append(
                {
                    "dataset": sample.dataset,
                    "sample_id": sample.sample_id,
                    "question": sample.question,
                    "gold_answers": sample.answers,
                    "gold_page": 1,
                    "image_path": str(image_path),
                    "ocr_text_chars": len(ocr_text),
                    "ocr_box_count": len(ocr_result.boxes),
                    "ocr_boxes": [box.to_dict() for box in ocr_result.boxes],
                    "metadata": sample.metadata,
                    "prepare_latency_ms": (time.perf_counter() - started) * 1000,
                }
            )
            for mode in MODES:
                # Each mode gets its own document object because MM-RAG appends visual evidence.
                mode_document = make_document(sample, image_path, ocr_text, ocr_result.boxes)
                if mode == "mm-rag" and not args.disable_region_crops:
                    region_chunks = make_region_chunks(
                        sample=sample,
                        document=mode_document,
                        image_path=image_path,
                        ocr_result=ocr_result,
                        region_cache_dir=region_cache_dir,
                        max_region_side=args.max_region_side,
                        max_regions_per_sample=args.max_regions_per_sample,
                    )
                    mode_document.chunks.extend(region_chunks)
                    mode_document.metadata["region_chunk_count"] = len(region_chunks)
                row = run_mode(sample, mode_document, mode, model_stack, shared, args.top_k)
                dataset_rows[mode].append(row)
                all_rows.append(row)
                label = mode.replace("-", "_")
                write_json(output_dir / f"{dataset_name}_{label}_details.json", dataset_rows[mode])
                write_csv(output_dir / f"{dataset_name}_{label}_scores.csv", dataset_rows[mode])

        for mode, rows in dataset_rows.items():
            label = mode.replace("-", "_")
            write_json(output_dir / f"{dataset_name}_{label}_details.json", rows)
            write_csv(output_dir / f"{dataset_name}_{label}_scores.csv", rows)

    summary_rows = summarize(all_rows)
    write_json(output_dir / "samples_manifest.json", manifest)
    write_csv(output_dir / "comparison_summary.csv", summary_rows)
    write_json(output_dir / "comparison_summary.json", summary_rows)
    write_markdown_summary(output_dir / "comparison_summary.md", summary_rows)
    print(f"Wrote benchmark outputs to {output_dir}")


if __name__ == "__main__":
    main()
