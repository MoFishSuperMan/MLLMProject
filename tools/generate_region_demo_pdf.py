from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image as ReportImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "pdf"
ASSET_DIR = OUT_DIR / "assets"
PDF_PATH = OUT_DIR / "mllm_region_demo.pdf"


def build_chart(path: Path) -> None:
    width, height = 900, 420
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    margin_left, margin_top, margin_right, margin_bottom = 90, 48, 42, 72
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    draw.rectangle([0, 0, width - 1, height - 1], outline=(225, 230, 236), width=2)
    draw.line([margin_left, margin_top, margin_left, margin_top + plot_h], fill=(80, 86, 100), width=3)
    draw.line([margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h], fill=(80, 86, 100), width=3)

    for idx, value in enumerate([0.40, 0.50, 0.60, 0.70, 0.80, 0.90]):
        y = margin_top + plot_h - int((value - 0.4) / 0.5 * plot_h)
        draw.line([margin_left, y, margin_left + plot_w, y], fill=(238, 241, 245), width=1)
        draw.text((22, y - 8), f"{value:.2f}", fill=(88, 96, 112))

    labels = ["Text", "Layout", "Vision", "Rerank"]
    baseline = [0.48, 0.57, 0.66, 0.72]
    multimodal = [0.55, 0.69, 0.82, 0.88]

    def point(index: int, value: float) -> tuple[int, int]:
        x = margin_left + int(index / (len(labels) - 1) * plot_w)
        y = margin_top + plot_h - int((value - 0.4) / 0.5 * plot_h)
        return x, y

    for series, color in [(baseline, (116, 126, 148)), (multimodal, (36, 107, 232))]:
        points = [point(index, value) for index, value in enumerate(series)]
        draw.line(points, fill=color, width=5, joint="curve")
        for x, y in points:
            draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill=color)

    for index, label in enumerate(labels):
        x, _ = point(index, 0.4)
        draw.text((x - 26, margin_top + plot_h + 22), label, fill=(42, 50, 66))

    draw.text((margin_left, 16), "Figure 1. Accuracy improves after layout and vision chunks", fill=(24, 32, 48))
    draw.rectangle([610, 46, 850, 108], fill=(248, 250, 252), outline=(225, 230, 236))
    draw.line([630, 68, 680, 68], fill=(116, 126, 148), width=5)
    draw.text((690, 59), "Text-only RAG", fill=(42, 50, 66))
    draw.line([630, 92, 680, 92], fill=(36, 107, 232), width=5)
    draw.text((690, 83), "Multimodal RAG", fill=(42, 50, 66))
    image.save(path)


def make_styles():
    font_name = "DemoCJK"
    pdfmetrics.registerFont(TTFont(font_name, r"C:\Windows\Fonts\simhei.ttf"))
    styles = getSampleStyleSheet()
    base = ParagraphStyle(
        "DemoBase",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=10.5,
        leading=16,
        textColor=colors.HexColor("#263244"),
        spaceAfter=7,
    )
    title = ParagraphStyle(
        "DemoTitle",
        parent=base,
        fontSize=20,
        leading=28,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=12,
    )
    heading = ParagraphStyle(
        "DemoHeading",
        parent=base,
        fontSize=14,
        leading=20,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=8,
        spaceAfter=8,
    )
    caption = ParagraphStyle(
        "DemoCaption",
        parent=base,
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#475569"),
        alignment=1,
        spaceAfter=10,
    )
    return base, title, heading, caption


def build_pdf() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    chart_path = ASSET_DIR / "figure1_accuracy_trend.png"
    build_chart(chart_path)

    base, title, heading, caption = make_styles()
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="Multimodal RAG Region Demo",
    )

    story = [
        Paragraph("多模态 RAG 区域定位 Demo", title),
        Paragraph(
            "本文档用于测试系统是否能把 PDF 中的表格、图片和公式区域拆成独立 chunk，并回答结构化问题。"
            "推荐测试问题包括：图1表示什么趋势？表1中哪个区域类型的识别置信度最高？公式1如何计算最终得分？"
            "第2节主要结论是什么？",
            base,
        ),
        Paragraph("1. 图像区域：性能趋势", heading),
        Paragraph(
            "图1比较了 Text-only RAG 与 Multimodal RAG 在不同处理阶段的准确率。"
            "随着 Layout chunk、Vision chunk 和 Rerank 模块加入，蓝色曲线持续上升，说明独立的图像与表格区域能提升检索和回答质量。",
            base,
        ),
        ReportImage(str(chart_path), width=160 * mm, height=74 * mm),
        Paragraph("图1 多模态 RAG 在引入布局与视觉 chunk 后呈现稳定上升趋势。", caption),
        Paragraph("2. 表格区域：区域识别结果", heading),
        Paragraph(
            "表1展示了不同区域类型的定位结果。表格 chunk 应保留行列结构，便于回答最大值、对比、统计等结构化问题。",
            base,
        ),
    ]

    table_data = [
        ["区域类型", "编号", "页码", "识别置信度", "主要用途"],
        ["图片", "图1", "1", "0.92", "趋势解释"],
        ["表格", "表1", "1", "0.95", "结构化问答"],
        ["公式", "公式1", "2", "0.88", "得分计算"],
    ]
    table = Table(table_data, colWidths=[28 * mm, 22 * mm, 18 * mm, 28 * mm, 54 * mm])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "DemoCJK"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f3ff")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("GRID", (0, 0), (-1, -1), 0.7, colors.HexColor("#cbd5e1")),
                ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#94a3b8")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 1), (3, -1), "CENTER"),
                ("LEADING", (0, 0), (-1, -1), 14),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend(
        [
            table,
            Paragraph("表1 多模态文档区域识别结果，表格保留了区域类型、编号、页码、置信度和用途。", caption),
            PageBreak(),
            Paragraph("3. 公式区域：融合得分", heading),
            Paragraph(
                "公式1定义了最终检索得分。系统在回答公式相关问题时，应能定位公式区域，并解释各变量含义。",
                base,
            ),
            Spacer(1, 8),
            Paragraph(
                "公式1  S_final = 0.50 * S_text + 0.30 * S_layout + 0.20 * S_vision    (1)",
                ParagraphStyle(
                    "Formula",
                    parent=base,
                    fontName="DemoCJK",
                    fontSize=13,
                    leading=20,
                    alignment=1,
                    backColor=colors.HexColor("#fff7ed"),
                    borderColor=colors.HexColor("#fed7aa"),
                    borderWidth=0.8,
                    borderPadding=8,
                    spaceAfter=12,
                ),
            ),
            Paragraph(
                "其中 S_text 表示文本相似度，S_layout 表示布局区域匹配得分，S_vision 表示图像或表格视觉证据得分。"
                "当问题提到图、表、趋势、公式或区域时，系统应优先检索对应的视觉 chunk。",
                base,
            ),
            Paragraph("4. 第2节主要结论", heading),
            Paragraph(
                "第2节的主要结论是：表格必须作为独立 chunk 保留结构化行列数据；图片必须作为独立 chunk 保留裁剪图和图号；"
                "公式必须作为独立 chunk 保留公式编号和变量解释。这三类区域能够显著改善多模态文档问答的可解释性。",
                base,
            ),
        ]
    )

    doc.build(story)


if __name__ == "__main__":
    build_pdf()
    print(PDF_PATH)
