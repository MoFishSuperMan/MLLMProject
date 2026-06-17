"""Evaluation metrics for document RAG experiments."""

from __future__ import annotations

import re
import string
from statistics import mean
from typing import Any

from .schemas import EvalPrediction


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[，。！？；：“”‘’（）【】《》、]", "", text)
    return text


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def levenshtein(left: str, right: str) -> int:
    left = normalize_answer(left)
    right = normalize_answer(right)
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def normalized_levenshtein_similarity(prediction: str, gold: str) -> float:
    prediction = normalize_answer(prediction)
    gold = normalize_answer(gold)
    max_len = max(len(prediction), len(gold))
    if max_len == 0:
        return 1.0
    return 1.0 - levenshtein(prediction, gold) / max_len


def anls(prediction: str, gold: str, threshold: float = 0.5) -> float:
    score = normalized_levenshtein_similarity(prediction, gold)
    return score if score >= threshold else 0.0


def normalize_gold_pages(gold_page: int | None = None, gold_pages: list[int] | None = None) -> list[int]:
    pages: list[int] = []
    for page in gold_pages or []:
        if page is not None and int(page) not in pages:
            pages.append(int(page))
    if gold_page is not None and int(gold_page) not in pages:
        pages.append(int(gold_page))
    return pages


def recall_at_k(retrieved_pages: list[int], gold_page: int | None, k: int, gold_pages: list[int] | None = None) -> float:
    targets = normalize_gold_pages(gold_page, gold_pages)
    if not targets:
        return 0.0
    return 1.0 if any(page in retrieved_pages[:k] for page in targets) else 0.0


def reciprocal_rank(retrieved_pages: list[int], gold_page: int | None, gold_pages: list[int] | None = None) -> float:
    targets = set(normalize_gold_pages(gold_page, gold_pages))
    if not targets:
        return 0.0
    for index, page in enumerate(retrieved_pages, start=1):
        if page in targets:
            return 1.0 / index
    return 0.0


def citation_accuracy(cited_pages: list[int], gold_page: int | None, gold_pages: list[int] | None = None) -> float:
    targets = set(normalize_gold_pages(gold_page, gold_pages))
    if not targets:
        return 0.0
    return 1.0 if any(page in targets for page in cited_pages) else 0.0


def score_prediction(prediction: EvalPrediction) -> dict[str, Any]:
    return {
        "sample_id": prediction.sample_id,
        "route": prediction.route,
        "em": exact_match(prediction.predicted_answer, prediction.gold_answer),
        "anls": anls(prediction.predicted_answer, prediction.gold_answer),
        "recall_at_1": recall_at_k(prediction.retrieved_pages, prediction.gold_page, 1, prediction.gold_pages),
        "recall_at_5": recall_at_k(prediction.retrieved_pages, prediction.gold_page, 5, prediction.gold_pages),
        "mrr": reciprocal_rank(prediction.retrieved_pages, prediction.gold_page, prediction.gold_pages),
        "citation_accuracy": citation_accuracy(prediction.cited_pages, prediction.gold_page, prediction.gold_pages),
        "latency_ms": prediction.latency_ms,
    }


def aggregate_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    numeric_keys = [
        "em",
        "anls",
        "recall_at_1",
        "recall_at_5",
        "mrr",
        "citation_accuracy",
        "latency_ms",
    ]
    if not rows:
        return {key: 0.0 for key in numeric_keys} | {"count": 0.0}
    summary = {key: mean(float(row.get(key, 0.0)) for row in rows) for key in numeric_keys}
    summary["count"] = float(len(rows))
    return summary
