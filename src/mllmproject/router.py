"""Rule-based routing for the demo-stage decision module."""

from __future__ import annotations

from .schemas import RouteDecision

VISION_KEYWORDS = ("图", "图片", "趋势", "颜色", "坐标", "柱状", "折线", "曲线", "figure", "chart", "plot")
TABLE_KEYWORDS = ("表", "表格", "数值", "最大", "最小", "占比", "平均", "增长率", "table", "value")
TEXT_KEYWORDS = ("第几节", "主要结论", "定义", "背景", "方法", "贡献", "摘要", "section", "summary")
ALL_MODES = ["text", "table", "figure", "formula", "image", "page", "visual", "region", "chart_region"]


def route_question(question: str, mode: str = "Auto Router") -> RouteDecision:
    normalized_mode = mode.strip().lower()
    if normalized_mode in {"text-rag", "text", "baseline"}:
        return RouteDecision(
            route="text_route",
            reason="手动选择 Text-RAG，仅检索文本 chunk",
            retrieval_modes=["text"],
        )
    if normalized_mode in {"mm-rag", "mm", "multimodal"}:
        return RouteDecision(
            route="hybrid_route",
            reason="手动选择 MM-RAG，检索文本与视觉 evidence",
            retrieval_modes=ALL_MODES,
        )

    normalized = question.lower()
    vision_hits = [keyword for keyword in VISION_KEYWORDS if keyword in normalized]
    table_hits = [keyword for keyword in TABLE_KEYWORDS if keyword in normalized]
    text_hits = [keyword for keyword in TEXT_KEYWORDS if keyword in normalized]

    if vision_hits:
        return RouteDecision(
            route="vision_route",
            reason=f"命中视觉关键词：{', '.join(vision_hits[:3])}",
            retrieval_modes=["chart_region", "region", "figure", "image", "page", "text"],
        )
    if table_hits:
        return RouteDecision(
            route="table_route",
            reason=f"命中表格/数值关键词：{', '.join(table_hits[:3])}",
            retrieval_modes=["chart_region", "region", "table", "page", "text"],
        )
    if text_hits:
        return RouteDecision(
            route="text_route",
            reason=f"命中文本关键词：{', '.join(text_hits[:3])}",
            retrieval_modes=["text"],
        )
    return RouteDecision(
        route="text_route",
        reason="未命中特定关键词，默认使用文本检索以减少 mock 视觉摘要噪声",
        retrieval_modes=["text"],
    )
