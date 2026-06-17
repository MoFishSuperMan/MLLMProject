"""GRPO-style lightweight router used by the Auto model option.

This module intentionally does not train a real Qwen-0.5B model during normal
demo runs. It provides the same engineering surface: a lightweight route model,
candidate VLM scoring, a GRPO-like grouped policy update hook, and an inference
decision that the backend can actually use.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re
from typing import Any

from .schemas import RouteDecision


VLM_CANDIDATES = ("qwen2_5_vl", "llama_3_2_vision", "deepseek_vl")
ALL_RETRIEVAL_MODES = ["text", "code", "table", "figure", "formula", "image", "page", "visual", "region", "chart_region"]


@dataclass(slots=True)
class RouterFeatures:
    text: float = 0.0
    table: float = 0.0
    vision: float = 0.0
    chart: float = 0.0
    reasoning: float = 0.0
    citation: float = 1.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class CandidateScore:
    model_id: str
    route: str
    retrieval_modes: list[str]
    reward: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GrpoPolicyState:
    """Small policy table that can be updated from grouped rewards."""

    model_bias: dict[str, float] = field(default_factory=dict)
    route_bias: dict[str, float] = field(default_factory=dict)
    learning_rate: float = 0.08

    @classmethod
    def default(cls) -> "GrpoPolicyState":
        return cls(
            model_bias={
                "qwen2_5_vl": 0.08,
                "llama_3_2_vision": 0.02,
                "deepseek_vl": 0.04,
            },
            route_bias={
                "text_route": 0.00,
                "table_route": 0.05,
                "vision_route": 0.06,
                "hybrid_route": 0.04,
            },
        )

    def update_from_group_rewards(self, scores: list[CandidateScore]) -> dict[str, float]:
        """Apply a GRPO-like centered reward update.

        The hook is here so the project has the "evaluate -> update route policy"
        code path. Demo inference does not call it unless tests or future scripts
        provide scored candidates.
        """

        if not scores:
            return {}
        mean_reward = sum(item.reward for item in scores) / len(scores)
        variance = sum((item.reward - mean_reward) ** 2 for item in scores) / len(scores)
        scale = math.sqrt(variance) or 1.0
        updates: dict[str, float] = {}
        for item in scores:
            advantage = (item.reward - mean_reward) / scale
            model_delta = self.learning_rate * advantage
            route_delta = self.learning_rate * advantage * 0.5
            self.model_bias[item.model_id] = self.model_bias.get(item.model_id, 0.0) + model_delta
            self.route_bias[item.route] = self.route_bias.get(item.route, 0.0) + route_delta
            updates[f"model:{item.model_id}"] = model_delta
            updates[f"route:{item.route}"] = route_delta
        return updates


class LightweightQwen05Router:
    """A Qwen-0.5B-shaped router interface backed by deterministic features."""

    model_id = "Qwen-0.5B-router"

    def featurize(self, question: str) -> RouterFeatures:
        normalized = question.lower()
        features = RouterFeatures()
        features.vision = score_keywords(
            normalized,
            ("image", "figure", "chart", "plot", "graph", "color", "visual", "picture", "trend", "bar", "line"),
        )
        features.table = score_keywords(
            normalized,
            ("table", "row", "column", "cell", "value", "average", "sum", "minimum", "maximum", "largest", "smallest"),
        )
        features.chart = score_keywords(
            normalized,
            ("chart", "plot", "graph", "bar", "line", "axis", "legend", "trend", "difference", "higher", "lower"),
        )
        features.reasoning = score_keywords(
            normalized,
            ("why", "compare", "difference", "more than", "less than", "calculate", "infer", "which", "how many"),
        )
        features.text = max(0.15, 1.0 - max(features.vision, features.table, features.chart) * 0.45)
        if re.search(r"[\u4e00-\u9fff]", question):
            features.vision = max(features.vision, score_keywords(question, ("图", "图片", "图表", "趋势", "颜色", "视觉")))
            features.table = max(features.table, score_keywords(question, ("表", "表格", "数值", "平均", "最大", "最小", "占比")))
            features.text = max(features.text, score_keywords(question, ("定义", "背景", "摘要", "方法", "结论", "章节")))
        return features


class GrpoAutoRouter:
    """Inference-time router that mirrors the planned GRPO route decision layer."""

    def __init__(
        self,
        policy: GrpoPolicyState | None = None,
        lightweight_model: LightweightQwen05Router | None = None,
    ) -> None:
        self.policy = policy or GrpoPolicyState.default()
        self.lightweight_model = lightweight_model or LightweightQwen05Router()

    def decide(self, question: str, mode: str = "auto") -> RouteDecision:
        normalized_mode = mode.strip().lower()
        if normalized_mode in {"text-rag", "text", "baseline"}:
            return RouteDecision(
                route="text_route",
                reason="manual Text-RAG route; Auto GRPO router bypassed",
                retrieval_modes=["text"],
                selected_model="text_baseline",
                router_name=self.lightweight_model.model_id,
                policy_trace={"mode": normalized_mode},
            )
        if normalized_mode in {"mm-rag", "mm", "multimodal"}:
            decision = self._best_decision(question, forced_route="hybrid_route")
            decision.reason = "manual MM-RAG route with GRPO-style VLM scoring; " + decision.reason
            return decision
        return self._best_decision(question)

    def _best_decision(self, question: str, forced_route: str | None = None) -> RouteDecision:
        features = self.lightweight_model.featurize(question)
        candidates = self._score_candidates(features, forced_route=forced_route)
        best = max(candidates, key=lambda item: item.reward)
        trace = {
            "router_model": self.lightweight_model.model_id,
            "features": features.to_dict(),
            "candidate_scores": [item.to_dict() for item in candidates],
            "policy": {
                "model_bias": dict(self.policy.model_bias),
                "route_bias": dict(self.policy.route_bias),
                "update_hook": "GrpoPolicyState.update_from_group_rewards",
            },
        }
        return RouteDecision(
            route=best.route,
            reason=best.reason,
            retrieval_modes=best.retrieval_modes,
            selected_model=best.model_id,
            router_name="grpo_auto_router",
            policy_trace=trace,
        )

    def _score_candidates(self, features: RouterFeatures, forced_route: str | None = None) -> list[CandidateScore]:
        route = forced_route or choose_route(features)
        retrieval_modes = retrieval_modes_for(route)
        scores: list[CandidateScore] = []
        for model_id in VLM_CANDIDATES:
            reward = self.policy.model_bias.get(model_id, 0.0) + self.policy.route_bias.get(route, 0.0)
            if model_id == "qwen2_5_vl":
                reward += features.text * 0.18 + features.table * 0.18 + features.citation * 0.08
            elif model_id == "llama_3_2_vision":
                reward += features.vision * 0.22 + features.reasoning * 0.12
            elif model_id == "deepseek_vl":
                reward += features.chart * 0.24 + features.table * 0.12 + features.reasoning * 0.08
            scores.append(
                CandidateScore(
                    model_id=model_id,
                    route=route,
                    retrieval_modes=retrieval_modes,
                    reward=round(reward, 6),
                    reason=build_reason(model_id, route, features),
                )
            )
        return scores


def route_question_with_grpo(question: str, mode: str = "auto") -> RouteDecision:
    return GrpoAutoRouter().decide(question, mode=mode)


def choose_route(features: RouterFeatures) -> str:
    if features.chart >= 0.30 or features.vision >= 0.45:
        return "vision_route"
    if features.table >= 0.30:
        return "table_route"
    if features.reasoning >= 0.30 and max(features.vision, features.table, features.chart) >= 0.25:
        return "hybrid_route"
    return "text_route"


def retrieval_modes_for(route: str) -> list[str]:
    if route == "text_route":
        return ["text", "code", "formula", "table", "figure"]
    if route == "table_route":
        return ["chart_region", "region", "table", "figure", "formula", "page", "text"]
    if route == "vision_route":
        return ["chart_region", "region", "figure", "table", "formula", "image", "page", "text"]
    return ALL_RETRIEVAL_MODES


def build_reason(model_id: str, route: str, features: RouterFeatures) -> str:
    strongest = max(
        ("text", features.text),
        ("table", features.table),
        ("vision", features.vision),
        ("chart", features.chart),
        ("reasoning", features.reasoning),
        key=lambda item: item[1],
    )
    return (
        f"GRPO-style auto route selected {route} and candidate VLM {model_id}; "
        f"strongest feature={strongest[0]}:{strongest[1]:.2f}. "
        "Policy update hook is implemented but not trained during demo inference."
    )


def score_keywords(text: str, keywords: tuple[str, ...]) -> float:
    hits = sum(1 for keyword in keywords if keyword and keyword in text)
    return min(1.0, hits / 3.0)
