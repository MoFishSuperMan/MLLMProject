"""Batch evaluation helpers for RAG baselines and demo comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import read_json, write_csv, write_json
from .metrics import aggregate_scores, score_prediction
from .service import RagService
from .text_baseline import TextBaselinePipeline
from .schemas import EvalPrediction


DEFAULT_COMPARISON_MODES = ("text-rag", "mm-rag", "auto")


def run_evaluation(
    doc_path: str | Path,
    samples_path: str | Path,
    output_dir: str | Path = "data/eval/results",
    mode: str = "auto",
    top_k: int = 5,
) -> dict[str, Any]:
    normalized_mode = mode.strip().lower()
    runner = build_runner(doc_path, normalized_mode)
    samples = load_eval_samples(samples_path)

    predictions: list[EvalPrediction] = []
    score_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for index, sample in enumerate(samples, start=1):
        sample_id = sample.get("sample_id") or f"sample_{index:03d}"
        result, latency_ms = timed_ask(runner, sample["question"], mode=mode, top_k=top_k)
        retrieved_pages = [evidence.page for evidence in result.evidences]
        retrieved_evidence_ids = [evidence.evidence_id for evidence in result.evidences]
        cited_pages = [citation.page for citation in result.citations]
        prediction = EvalPrediction(
            sample_id=sample_id,
            question=sample["question"],
            gold_answer=sample.get("answer", ""),
            predicted_answer=result.answer,
            gold_page=sample.get("gold_page"),
            cited_pages=cited_pages,
            retrieved_pages=retrieved_pages,
            retrieved_evidence_ids=retrieved_evidence_ids,
            route=result.route,
            latency_ms=latency_ms,
            gold_pages=normalize_gold_pages_from_sample(sample),
        )
        predictions.append(prediction)
        score_row = enrich_score_row(score_prediction(prediction), prediction, sample)
        score_rows.append(score_row)
        detail_rows.append(
            {
                **prediction.to_dict(),
                "answer": result.answer,
                "mode": normalize_mode_label(mode),
                "gold_type": sample.get("gold_type", ""),
                "question_type": sample.get("question_type", ""),
                "route_reason": result.route_reason,
                "failure_label": score_row["failure_label"],
                "case_status": score_row["case_status"],
                "top_evidence": [evidence.to_dict() for evidence in result.evidences],
            }
        )

    summary = aggregate_scores(score_rows)
    summary.update(aggregate_case_fields(score_rows))
    breakdowns = {
        "by_route": aggregate_by(score_rows, "route"),
        "by_question_type": aggregate_by(score_rows, "question_type"),
        "failure_counts": count_by(score_rows, "failure_label"),
        "case_status_counts": count_by(score_rows, "case_status"),
    }
    target = Path(output_dir)
    label = normalize_mode_label(mode)
    write_json(target / f"{label}_details.json", detail_rows)
    write_json(target / f"{label}_summary.json", {"mode": label, "metrics": summary, "breakdowns": breakdowns})
    write_csv(target / f"{label}_scores.csv", score_rows)
    return {"summary": summary, "breakdowns": breakdowns, "details": detail_rows, "scores": score_rows}


def run_comparison(
    doc_path: str | Path,
    samples_path: str | Path,
    output_dir: str | Path = "data/eval/results",
    modes: tuple[str, ...] | list[str] = DEFAULT_COMPARISON_MODES,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run several modes and write a compact comparison table."""

    results: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for mode in modes:
        result = run_evaluation(
            doc_path=doc_path,
            samples_path=samples_path,
            output_dir=output_dir,
            mode=mode,
            top_k=top_k,
        )
        label = normalize_mode_label(mode)
        row = {"mode": label, **result["summary"]}
        summary_rows.append(row)
        results[label] = result

    target = Path(output_dir)
    write_csv(target / "comparison_summary.csv", summary_rows)
    write_json(target / "comparison_summary.json", summary_rows)
    return {"summary": summary_rows, "modes": results}


def load_eval_samples(samples_path: str | Path) -> list[dict[str, Any]]:
    samples = read_json(samples_path)
    if not isinstance(samples, list):
        raise ValueError("Evaluation samples must be a JSON list.")
    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            raise ValueError(f"Evaluation sample #{index} must be an object.")
        if not sample.get("question"):
            raise ValueError(f"Evaluation sample #{index} is missing 'question'.")
    return samples


def enrich_score_row(
    row: dict[str, Any],
    prediction: EvalPrediction,
    sample: dict[str, Any],
) -> dict[str, Any]:
    enriched = {
        **row,
        "question_type": sample.get("question_type", ""),
        "gold_type": sample.get("gold_type", ""),
        "gold_page": prediction.gold_page,
        "gold_pages": "|".join(str(page) for page in prediction.gold_pages),
        "top1_page": prediction.retrieved_pages[0] if prediction.retrieved_pages else None,
        "retrieved_pages": "|".join(str(page) for page in prediction.retrieved_pages),
        "cited_pages": "|".join(str(page) for page in prediction.cited_pages),
    }
    enriched["retrieval_success"] = 1.0 if float(enriched["recall_at_5"]) >= 1.0 else 0.0
    enriched["citation_success"] = 1.0 if float(enriched["citation_accuracy"]) >= 1.0 else 0.0
    enriched["answer_match"] = 1.0 if float(enriched["em"]) >= 1.0 or float(enriched["anls"]) > 0.0 else 0.0
    enriched["case_success"] = 1.0 if enriched["retrieval_success"] and enriched["citation_success"] else 0.0
    enriched["case_status"] = "success" if enriched["case_success"] else "failure"
    enriched["failure_label"] = label_failure(enriched)
    return enriched


def label_failure(row: dict[str, Any]) -> str:
    if float(row.get("recall_at_5", 0.0)) < 1.0:
        return "retrieval_miss"
    if float(row.get("citation_accuracy", 0.0)) < 1.0:
        return "citation_miss"
    if float(row.get("recall_at_1", 0.0)) < 1.0:
        return "rerank_miss"
    if float(row.get("answer_match", 0.0)) < 1.0:
        return "answer_mismatch"
    return "ok"


def aggregate_case_fields(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "case_success_rate": 0.0,
            "retrieval_success_rate": 0.0,
            "citation_success_rate": 0.0,
            "answer_match_rate": 0.0,
        }
    count = float(len(rows))
    return {
        "case_success_rate": sum(float(row.get("case_success", 0.0)) for row in rows) / count,
        "retrieval_success_rate": sum(float(row.get("retrieval_success", 0.0)) for row in rows) / count,
        "citation_success_rate": sum(float(row.get("citation_success", 0.0)) for row in rows) / count,
        "answer_match_rate": sum(float(row.get("answer_match", 0.0)) for row in rows) / count,
    }


def aggregate_by(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = str(row.get(key) or "unknown")
        grouped.setdefault(label, []).append(row)
    return {
        label: aggregate_scores(group) | aggregate_case_fields(group)
        for label, group in sorted(grouped.items())
    }


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get(key) or "unknown")
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def normalize_mode_label(mode: str) -> str:
    return mode.strip().lower().replace(" ", "_").replace("-", "_")


def normalize_gold_pages_from_sample(sample: dict[str, Any]) -> list[int]:
    raw_pages = sample.get("gold_pages")
    if raw_pages is None:
        raw_pages = []
    if isinstance(raw_pages, int):
        pages = [raw_pages]
    else:
        pages = [int(page) for page in raw_pages if page is not None]
    gold_page = sample.get("gold_page")
    if gold_page is not None and int(gold_page) not in pages:
        pages.append(int(gold_page))
    return pages


def build_runner(doc_path: str | Path, normalized_mode: str):
    if normalized_mode in {"text", "text-rag", "baseline"}:
        return TextBaselinePipeline.from_document(doc_path)
    service = RagService()
    service.ingest_document(doc_path)
    return service


def timed_ask(runner, question: str, mode: str, top_k: int):
    import time

    start = time.perf_counter()
    if isinstance(runner, TextBaselinePipeline):
        result = runner.ask(question, top_k=top_k)
    else:
        result = runner.ask(question, mode=normalize_engine_mode(mode), top_k=top_k)
    latency_ms = (time.perf_counter() - start) * 1000
    return result, latency_ms


def normalize_engine_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized in {"text", "text-rag", "baseline"}:
        return "Text-RAG"
    if normalized in {"mm", "mm-rag", "multimodal"}:
        return "MM-RAG"
    return "Auto Router"
