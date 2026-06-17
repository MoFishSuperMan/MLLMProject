from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PIL import Image

from mllmproject.multimodal import (
    clamp_bbox,
    draw_evidence_preview,
    format_evidence_caption,
    make_mock_region_chunks,
    make_page_visual_chunks,
)
from mllmproject.multimodal_embeddings import UnifiedMultimodalEmbedder
from mllmproject.index import VectorIndex
from mllmproject.router import route_question
from mllmproject.chunking import chunk_pages
from mllmproject.schemas import Document, Evidence, Page, PageText
from mllmproject.schemas import Chunk
from mllmproject.table_markdown import table_to_markdown
from mllmproject.ingest import (
    detect_code_regions,
    detect_formula_regions,
    extract_formula_blocks_from_text,
    is_probable_table_data,
    merge_text_layout_with_visual_crops,
    normalize_formula_markdown,
    looks_like_formula,
)
from mllmproject.vision_regions import (
    OcrBox,
    OcrResult,
    RegionCandidate,
    materialize_region_chunks,
    scale_bbox,
    select_region_candidates,
)
from mllmproject.pipeline import prioritize_region_evidence


class MultimodalComponentsTest(unittest.TestCase):
    def test_page_visual_chunks_keep_bbox_and_image_path(self) -> None:
        document = Document(
            doc_id="demo_doc",
            pages=[
                Page(
                    doc_id="demo_doc",
                    page=1,
                    image_path="page_001.png",
                    width=100,
                    height=200,
                )
            ],
        )

        chunks = make_page_visual_chunks(document, summary_fn=lambda path: f"summary for {path}")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].source_type, "page")
        self.assertEqual(chunks[0].bbox, [0.0, 0.0, 100.0, 200.0])
        self.assertEqual(chunks[0].image_path, "page_001.png")
        self.assertEqual(chunks[0].region_id, "demo_doc_p1_page")
        self.assertIn("summary for page_001.png", chunks[0].content)

    def test_mock_region_chunks_match_future_layout_schema(self) -> None:
        document = Document(
            doc_id="demo_doc",
            pages=[
                Page(
                    doc_id="demo_doc",
                    page=2,
                    image_path="page_002.png",
                    width=1000,
                    height=800,
                )
            ],
        )

        chunks = make_mock_region_chunks(document)

        self.assertEqual([chunk.source_type for chunk in chunks], ["figure", "table"])
        self.assertTrue(all(chunk.bbox for chunk in chunks))
        self.assertTrue(all(chunk.region_id for chunk in chunks))
        self.assertTrue(all(chunk.metadata.get("is_placeholder") for chunk in chunks))

    def test_bbox_clamp_and_caption(self) -> None:
        self.assertEqual(clamp_bbox([-10, 5, 120, 250], width=100, height=200), [0, 5, 100, 200])
        self.assertIsNone(clamp_bbox([120, 10, 130, 30], width=100, height=200))

        evidence = Evidence(
            evidence_id="ev1",
            doc_id="demo",
            page=3,
            source_type="figure",
            content="figure summary",
            score=0.75,
            chunk_id="chunk1",
            bbox=[0, 5, 100, 200],
            image_path="page.png",
            region_id="fig1",
        )
        caption = format_evidence_caption(evidence)
        self.assertIn("page=3", caption)
        self.assertIn("type=figure", caption)
        self.assertIn("bbox=[0, 5, 100, 200]", caption)

    def test_draw_evidence_preview_falls_back_without_bbox(self) -> None:
        evidence = Evidence(
            evidence_id="ev1",
            doc_id="demo",
            page=1,
            source_type="page",
            content="page summary",
            score=0.5,
            image_path="missing.png",
        )
        with tempfile.TemporaryDirectory() as tmp:
            preview = draw_evidence_preview(evidence, Path(tmp) / "preview.png")
        self.assertIsNotNone(preview)
        self.assertEqual(preview.image_path, "missing.png")

    def test_region_crop_scales_page_bbox_to_original_image(self) -> None:
        self.assertEqual(scale_bbox([10, 10, 50, 60], from_size=(100, 100), to_size=(200, 200)), [20, 20, 100, 120])
        document = Document(
            doc_id="demo_doc",
            pages=[Page(doc_id="demo_doc", page=1, image_path="page.png", width=100, height=100)],
        )
        source_image = Image.new("RGB", (200, 200), color="white")
        candidate = RegionCandidate(
            region_id="demo_doc_p1_region",
            page=1,
            source_type="region",
            bbox=[10, 10, 50, 60],
            image_path="",
            reason="unit test",
            score=0.9,
        )
        ocr = OcrResult(text="Paul", boxes=[OcrBox("Paul", [15, 20, 30, 35], 0.9)])
        with tempfile.TemporaryDirectory() as tmp:
            chunks = materialize_region_chunks(
                document=document,
                candidates=[candidate],
                ocr_result=ocr,
                source_image=source_image,
                page_size=(100, 100),
                output_dir=Path(tmp),
                question="To whom is the document sent?",
                max_region_side=80,
            )

            self.assertEqual(len(chunks), 1)
            self.assertTrue(Path(chunks[0].image_path or "").exists())
            self.assertEqual(chunks[0].metadata["original_bbox"], [20, 20, 100, 120])
            self.assertIn("Paul", chunks[0].metadata["ocr_text"])

    def test_ad_question_prioritizes_bottom_brand_logo_crop(self) -> None:
        ocr = OcrResult(
            text="AASHIRVAAD\nWILLS\nLIFESTYLE",
            boxes=[
                OcrBox("AASHIRVAAD", [500, 350, 610, 375], 0.99),
                OcrBox("WILLS", [584, 1136, 656, 1162], 0.99),
                OcrBox("LIFESTYLE", [585, 1161, 655, 1175], 0.98),
            ],
        )

        candidates = select_region_candidates(
            question="What is the name of the fashion wear/clothing advertise",
            ocr_result=ocr,
            page_size=(900, 1280),
            doc_id="docvqa_57366",
            max_regions=2,
        )

        self.assertEqual(candidates[0].region_id, "docvqa_57366_p1_region_ad_logo")
        self.assertGreaterEqual(candidates[0].bbox[3], 1175)
        self.assertIn("brand-logo", candidates[0].reason)

    def test_region_priority_uses_metadata_score(self) -> None:
        low = Evidence(
            evidence_id="low",
            doc_id="demo",
            page=1,
            source_type="region",
            content="keyword region",
            score=0.99,
            image_path="low.png",
            metadata={"score": 0.95},
        )
        high = Evidence(
            evidence_id="high",
            doc_id="demo",
            page=1,
            source_type="region",
            content="advertisement brand crop",
            score=0.80,
            image_path="high.png",
            metadata={"score": 1.18},
        )

        self.assertEqual(prioritize_region_evidence([low, high])[0].evidence_id, "high")

    def test_auto_router_uses_grpo_policy_trace(self) -> None:
        decision = route_question("What is the value in the chart?", mode="auto")

        self.assertIn(decision.route, {"vision_route", "table_route", "hybrid_route"})
        self.assertEqual(decision.router_name, "grpo_auto_router")
        self.assertIn(decision.selected_model, {"qwen2_5_vl", "llama_3_2_vision", "deepseek_vl"})
        self.assertIn("candidate_scores", decision.policy_trace)

    def test_unified_embedding_marks_image_chunks_and_searches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "chart.png"
            Image.new("RGB", (64, 32), color=(220, 30, 30)).save(image_path)
            chunks = [
                Chunk(
                    chunk_id="c1",
                    doc_id="doc",
                    page=1,
                    source_type="figure",
                    content="chart visual evidence",
                    image_path=str(image_path),
                ),
                Chunk(chunk_id="c2", doc_id="doc", page=1, source_type="text", content="plain paragraph"),
            ]
            index = VectorIndex(embedder=UnifiedMultimodalEmbedder())
            index.build(chunks)

            result = index.search("chart image", top_k=2)

        self.assertIn("image", chunks[0].metadata["embedding_modalities"])
        self.assertEqual(chunks[0].metadata["embedding_space"], "unified_text_image")
        self.assertTrue(result)

    def test_table_rows_are_rendered_as_markdown(self) -> None:
        markdown = table_to_markdown([["Year", "Value"], ["2026", 42]])

        self.assertIn("| Year | Value |", markdown)
        self.assertIn("| --- | --- |", markdown)
        self.assertIn("| 2026 | 42 |", markdown)

    def test_code_blocks_become_code_chunks(self) -> None:
        chunks = chunk_pages(
            "doc",
            [
                PageText(
                    page=1,
                    text="Intro paragraph.\n\n```python\ndef add(a, b):\n    return a + b\n```\n\nOutro paragraph.",
                )
            ],
        )

        self.assertEqual([chunk.source_type for chunk in chunks], ["text", "code", "text"])
        self.assertIn("def add", chunks[1].content)
        self.assertNotIn("```", chunks[1].content)

    def test_markdown_math_becomes_formula_not_code(self) -> None:
        chunks = chunk_pages(
            "doc",
            [
                PageText(
                    page=1,
                    text="Intro.\n\n$$\nC = AB, C_{ij} = \\sum_{l=1}^{n} A_{il}B_{lj}\n$$\n\nOutro.",
                )
            ],
        )

        self.assertEqual([chunk.source_type for chunk in chunks], ["text", "formula", "text"])
        self.assertIn("$$", chunks[1].content)

    def test_layout_code_lines_are_grouped_as_one_complete_region(self) -> None:
        class Page:
            rect = type("Rect", (), {"x0": 0, "y0": 0, "x1": 600, "y1": 800})()

        blocks = [
            (50, 100, 240, 116, "// 基 准 矩 阵 乘 法 运 算", 0, 0),
            (50, 118, 260, 134, "for (int i = 0; i < m; i++) {", 0, 0),
            (70, 136, 290, 152, "for (int j = 0; j < k; j++) {", 0, 0),
            (90, 154, 320, 170, "for (int l = 0; l < n; l++) {", 0, 0),
            (110, 172, 360, 188, "C[i][j] += A[i][l] * B[l][j];", 0, 0),
            (90, 190, 110, 206, "}", 0, 0),
            (70, 208, 90, 224, "}", 0, 0),
            (50, 226, 70, 242, "}", 0, 0),
        ]

        regions = detect_code_regions(Page(), blocks)

        self.assertEqual(len(regions), 1)
        self.assertIn("for (int i = 0; i < m; i++)", regions[0]["code_text"])
        self.assertIn("C[i][j] += A[i][l] * B[l][j];", regions[0]["code_text"])
        self.assertEqual(regions[0]["code_language"], "cpp")

    def test_layout_formula_region_gets_crop_and_is_not_code(self) -> None:
        class Page:
            rect = type("Rect", (), {"x0": 0, "y0": 0, "x1": 600, "y1": 800})()

        blocks = [
            (220.77, 322.08, 321.24, 335.83, "C = AB,\nCij =", 0, 0),
            (324.57, 310.25, 341.83, 322.46, "n\n∑", 0, 0),
            (326.48, 321.79, 524.41, 347.27, "l=1\nAilBlj\n(1)", 0, 0),
            (50, 500, 240, 516, "// 基 准 矩 阵 乘 法 运 算", 0, 0),
            (50, 518, 260, 534, "for (int i = 0; i < m; i++) {", 0, 0),
            (50, 536, 70, 552, "}", 0, 0),
        ]

        formulas = detect_formula_regions(Page(), blocks)
        codes = detect_code_regions(Page(), blocks)

        self.assertEqual(len(formulas), 1)
        self.assertIn("\\sum_{l=1}^{n}", formulas[0]["formula_markdown"])
        self.assertGreater(formulas[0]["bbox"][2] - formulas[0]["bbox"][0], 250)
        self.assertEqual(len(codes), 1)
        self.assertNotIn("Cij =", codes[0]["code_text"])

    def test_formula_detection_ignores_prose_and_keeps_markdown_equation(self) -> None:
        self.assertFalse(looks_like_formula("对于一般矩阵，朴素矩阵乘法的时间复杂度为 O(mnk)。"))
        self.assertEqual(
            [chunk.source_type for chunk in chunk_pages("doc", [PageText(page=1, text="法结果为C ∈Rm×k，其计算公式为：\n\nn ∑")])],
            ["text"],
        )
        formula = normalize_formula_markdown("C = AB, Cij = n ∑ l=1 AilBlj (1)")

        self.assertEqual(formula, r"C = AB,\quad C_{ij} = \sum_{l=1}^{n} A_{il}B_{lj}")

    def test_pdf_layout_formula_lines_are_merged_once(self) -> None:
        text = "\n".join(
            [
                "正文说明。",
                "",
                "                                              ∑",
                "                                              n",
                "                         C = AB,      Cij =         Ail Blj   (1)",
                "                                              l=1",
                "",
                "后续正文。",
            ]
        )

        formulas = extract_formula_blocks_from_text(text)

        self.assertEqual(len(formulas), 1)
        self.assertEqual(formulas[0][0], r"C = AB,\quad C_{ij} = \sum_{l=1}^{n} A_{il}B_{lj}")

    def test_sparse_chart_axis_grid_is_not_a_table(self) -> None:
        axis_grid = [
            ["", "", "", "", "128", ""],
            ["", "", "", "", "256", ""],
            ["", "", "", "", "512", ""],
            ["", "", "", "", "1024", ""],
        ]

        self.assertFalse(is_probable_table_data(axis_grid))

    def test_text_layout_chunks_override_bad_pymupdf_regions(self) -> None:
        text_chunks = [
            Chunk(
                chunk_id="doc_p1_formula01",
                doc_id="doc",
                page=1,
                source_type="formula",
                content="公式1:\n$$\nC = AB,\\quad C_{ij} = \\sum_{l=1}^{n} A_{il}B_{lj}\n$$",
                metadata={"extraction_method": "pdftotext_formula_block"},
            ),
            Chunk(
                chunk_id="doc_p1_code01",
                doc_id="doc",
                page=1,
                source_type="code",
                content="// 基准矩阵乘法运算\nfor (int i = 0; i < m; i++) {\n}",
                metadata={"extraction_method": "pdftotext_code_block"},
            ),
        ]
        visual_chunks = [
            Chunk(
                chunk_id="doc_p1_code01_visual",
                doc_id="doc",
                page=1,
                source_type="code",
                content="C = AB,\nCij =",
                image_path="formula-crop.png",
                bbox=[0, 0, 100, 40],
                metadata={"extraction_method": "code_text_block"},
            ),
            Chunk(
                chunk_id="doc_p1_code02_visual",
                doc_id="doc",
                page=1,
                source_type="code",
                content="// 基准矩阵乘法运算\nfor (int i = 0; i < m; i++) {\n}",
                image_path="code-crop.png",
                bbox=[0, 0, 200, 120],
                metadata={"extraction_method": "code_text_block"},
            ),
        ]

        merged = merge_text_layout_with_visual_crops(text_chunks, visual_chunks)

        self.assertEqual([chunk.source_type for chunk in merged], ["formula", "code"])
        self.assertIn("\\sum_{l=1}^{n}", merged[0].content)
        self.assertEqual(merged[0].image_path, "formula-crop.png")
        self.assertEqual(merged[1].image_path, "code-crop.png")
        self.assertNotIn("Cij =", "\n".join(chunk.content for chunk in merged if chunk.source_type == "code"))


if __name__ == "__main__":
    unittest.main()
