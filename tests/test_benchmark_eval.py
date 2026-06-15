from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from PIL import Image
from PIL import ImageDraw

from mllmproject.answer_extraction import extract_short_answer, normalize_entity_answer
from mllmproject.chart_preprocess import classify_chart_question
from mllmproject.model_stack import ModelConfig, ModelStack
from mllmproject.pipeline import RagPipeline
from mllmproject.schemas import Document, Page, PageText
from mllmproject.chunking import chunk_pages
from mllmproject.vision_regions import OcrBox, OcrResult

from run_benchmark_eval import (
    BenchmarkSample,
    make_region_chunks,
    make_document,
    normalize_for_gold,
    run_mode,
    sample_from_row,
    summarize,
    write_markdown_summary,
)


class BenchmarkAdapterTest(unittest.TestCase):
    def test_short_answer_extraction_prefers_final_answer(self) -> None:
        self.assertEqual(
            extract_short_answer("图中显示了14种食品项目 [E1]。\nFinal answer: 14\n来源：[page=1]", "How many food items?"),
            "14",
        )
        self.assertEqual(extract_short_answer("不是。[E1]\nFinal answer: No", "Is it higher?"), "No")

    def test_count_extraction_uses_explicit_list_when_final_answer_conflicts(self) -> None:
        answer = (
            "图表中显示了18种食品商品，分别是：羊肉、玉米、大麦、黑麦、牛肉、小麦、咖啡、"
            "茶叶、花生、棕榈油、猪肉、大米、糖和可可。[E1]\n\nFinal answer: 18"
        )
        self.assertEqual(extract_short_answer(answer, "How many food item is shown in the bar graph?"), "14")

    def test_short_answer_extraction_fallbacks(self) -> None:
        self.assertEqual(extract_short_answer("公司的名称是 ITC Limited [E1]。", "What is the name of the company?"), "ITC Limited")
        self.assertEqual(extract_short_answer("该文档发送给 Paul [E1]。", "To whom is the document sent?"), "Paul")

    def test_truncated_year_final_answer_repairs_from_reasoning(self) -> None:
        answer = (
            "观察折线图的斜率，2010年至2011年的上升幅度最大，即该段的斜率最陡峭。\n\n"
            "Final answer: 201\n来源：[page=1]"
        )
        self.assertEqual(extract_short_answer(answer, "When does the line have the sharpest increase?"), "2011")

    def test_truncated_yes_no_comparison_uses_equation_totals(self) -> None:
        answer = (
            "最大的两个条形图的总和是 94% + 91% = 185%。"
            "最小的三个条形图的总和是 73% + 72% + 88% = 233%。由于\n来源：[page=1]"
        )
        self.assertEqual(
            extract_short_answer(answer, "Is the sum of largest two bars is greater then the sum of smallest 3 bars?"),
            "No",
        )

    def test_docvqa_entity_alias_normalization(self) -> None:
        self.assertEqual(normalize_entity_answer("圣迭戈", "Where is the university located?"), "san diego")
        self.assertEqual(
            normalize_entity_answer("加州大学圣地亚哥分校", "What is name of university?"),
            "university of california",
        )

    def test_docvqa_row_maps_to_stable_sample(self) -> None:
        image = Image.new("RGB", (20, 10), color="white")
        sample = sample_from_row(
            "docvqa",
            0,
            {
                "questionId": "49153",
                "question": "What is the value?",
                "question_types": ["figure/diagram"],
                "image": image,
                "docId": 14465,
                "ucsf_document_id": "pybv0228",
                "ucsf_document_page_no": "81",
                "answers": ["0.28"],
            },
        )

        self.assertEqual(sample.sample_id, "docvqa_49153")
        self.assertEqual(sample.question, "What is the value?")
        self.assertEqual(sample.answers, ["0.28"])
        self.assertEqual(sample.metadata["ucsf_document_page_no"], "81")

    def test_chartqa_row_maps_to_stable_sample(self) -> None:
        image = Image.new("RGB", (20, 10), color="white")
        sample = sample_from_row(
            "chartqa",
            7,
            {
                "type": "human_test",
                "question": "How many bars?",
                "answer": "3",
                "image": image,
            },
        )

        self.assertEqual(sample.sample_id, "chartqa_000007")
        self.assertEqual(sample.gold_answer, "3")
        self.assertEqual(sample.metadata["type"], "human_test")

    def test_make_document_uses_ocr_text_and_gold_page_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "sample.png"
            Image.new("RGB", (32, 24), color="white").save(image_path)
            sample = BenchmarkSample(
                dataset="chartqa",
                sample_id="chartqa_000001",
                question="How many bars?",
                answers=["3"],
                image=None,
                metadata={},
            )

            document = make_document(sample, image_path, "three bars\naxis label")

        self.assertEqual(document.pages[0].page, 1)
        self.assertEqual(document.pages[0].width, 32)
        self.assertEqual(document.pages[0].height, 24)
        self.assertEqual([chunk.source_type for chunk in document.chunks], ["text"])
        self.assertIn("three bars", document.chunks[0].content)

    def test_chartqa_region_chunk_is_generated_with_question_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = Image.new("RGB", (100, 80), color="white")
            draw = ImageDraw.Draw(image)
            draw.rectangle([20, 20, 80, 65], fill="black")
            image_path = root / "chart.png"
            image.save(image_path)
            sample = BenchmarkSample(
                dataset="chartqa",
                sample_id="chartqa_000000",
                question="How many bars are shown in the chart?",
                answers=["3"],
                image=image,
                metadata={},
            )
            ocr_result = OcrResult(
                text="A\nB\nC\n1\n2\n3",
                boxes=[
                    OcrBox("A", [15, 66, 25, 75], 0.9),
                    OcrBox("1", [18, 10, 28, 18], 0.9),
                ],
            )
            document = make_document(sample, image_path, ocr_result.text, ocr_result.boxes)

            chunks = make_region_chunks(
                sample=sample,
                document=document,
                image_path=image_path,
                ocr_result=ocr_result,
                region_cache_dir=root / "regions",
                max_region_side=64,
                max_regions_per_sample=2,
            )

            self.assertEqual(classify_chart_question(sample.question), "count")
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0].source_type, "chart_region")
            self.assertEqual(chunks[0].metadata["chart_question_type"], "count")
            self.assertTrue(Path(chunks[0].image_path or "").exists())

    def test_docvqa_numeric_target_region_is_prioritized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = Image.new("RGB", (1200, 900), color="white")
            image_path = root / "doc.png"
            image.save(image_path)
            sample = BenchmarkSample(
                dataset="docvqa",
                sample_id="docvqa_49153",
                question="What is the actual value per 1000, during the year 1975?",
                answers=["0.28"],
                image=image,
                metadata={},
            )
            ocr_result = OcrResult(
                text="0.28\n1975\n1980",
                boxes=[
                    OcrBox("0.28", [220, 324, 282, 350], 0.99),
                    OcrBox("1975", [922, 778, 982, 804], 0.67),
                    OcrBox("1980", [1054, 778, 1110, 802], 0.99),
                ],
            )
            document = make_document(sample, image_path, ocr_result.text, ocr_result.boxes)
            chunks = make_region_chunks(
                sample=sample,
                document=document,
                image_path=image_path,
                ocr_result=ocr_result,
                region_cache_dir=root / "regions",
                max_region_side=2048,
                max_regions_per_sample=2,
            )

            self.assertTrue(chunks)
            self.assertEqual(chunks[0].region_id, "docvqa_49153_p1_region_target_1975")
            self.assertIn("target numeric focus", chunks[0].metadata["reason"])

    def test_time_question_region_keeps_left_time_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = Image.new("RGB", (640, 900), color="white")
            image_path = root / "schedule.png"
            image.save(image_path)
            sample = BenchmarkSample(
                dataset="docvqa",
                sample_id="docvqa_49168",
                question="What time is the coffee break?",
                answers=["11:14 to 11:39 a.m."],
                image=image,
                metadata={},
            )
            ocr_result = OcrResult(
                text="11.14 to\nCoffee Break\n11.39 a.m",
                boxes=[
                    OcrBox("11.14 to", [88, 68, 172, 92], 0.76),
                    OcrBox("Coffee Break", [218, 68, 350, 94], 0.99),
                    OcrBox("11.39 a.m", [88, 92, 192, 118], 0.64),
                ],
            )
            document = make_document(sample, image_path, ocr_result.text, ocr_result.boxes)
            chunks = make_region_chunks(
                sample=sample,
                document=document,
                image_path=image_path,
                ocr_result=ocr_result,
                region_cache_dir=root / "regions",
                max_region_side=2048,
                max_regions_per_sample=2,
            )

            self.assertEqual(chunks[0].region_id, "docvqa_49168_p1_region_time_row")
            self.assertEqual(chunks[0].metadata["page_bbox"][0], 0)
            self.assertIn("11.14", chunks[0].metadata["ocr_text"])

    def test_percent_point_prediction_normalizes_to_decimal_gold(self) -> None:
        self.assertEqual(
            normalize_for_gold("3", ["0.03"], "How many more people felt inspired frequently than depressed frequently?"),
            "0.03",
        )

    def test_entity_substring_prediction_normalizes_to_gold(self) -> None:
        self.assertEqual(
            normalize_for_gold(
                "Dark Fantasy Choco Fills",
                ["dark fantasy", "Dark fantasy"],
                "What is the name of the choco fills advertised?",
            ),
            "dark fantasy",
        )

    def test_relaxed_numeric_tolerance_normalizes_to_gold(self) -> None:
        self.assertEqual(
            normalize_for_gold(
                "21.7",
                ["21.6"],
                "What's the average of all the values in the green bars (round to one decimal)?",
            ),
            "21.6",
        )


class BenchmarkPipelineTest(unittest.TestCase):
    def test_from_document_text_and_mm_modes_keep_expected_evidence_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "sample.png"
            Image.new("RGB", (32, 24), color="white").save(image_path)
            base_doc = self._document(image_path)
            stack = ModelStack(ModelConfig(use_real_models=False))

            text_pipeline = RagPipeline.from_document(base_doc, include_visual=False, model_stack=stack)
            text_result, _ = text_pipeline.answer("What number appears?", mode="text-rag", top_k=3)

            mm_pipeline = RagPipeline.from_document(self._document(image_path), include_visual=True, model_stack=stack)
            mm_result, _ = mm_pipeline.answer("What number appears in the chart?", mode="mm-rag", top_k=3)

        self.assertTrue(text_result.evidences)
        self.assertEqual({evidence.source_type for evidence in text_result.evidences}, {"text"})
        self.assertTrue(any(evidence.source_type == "page" for evidence in mm_result.evidences))

    def test_mock_benchmark_writes_comparison_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "sample.png"
            Image.new("RGB", (32, 24), color="white").save(image_path)
            sample = BenchmarkSample(
                dataset="chartqa",
                sample_id="chartqa_000001",
                question="What number appears?",
                answers=["3"],
                image=None,
                metadata={},
            )
            stack = ModelStack(ModelConfig(use_real_models=False))
            shared = {
                "embedder": stack.create_embedder(),
                "reranker": stack.create_reranker(),
                "generator": stack.create_generator(),
                "visual_summarizer": stack.create_visual_summarizer(),
            }

            rows = [
                run_mode(sample, make_document(sample, image_path, "number 3"), mode, stack, shared, top_k=2)
                for mode in ("text-rag", "mm-rag")
            ]
            summary = summarize(rows)
            write_markdown_summary(root / "comparison_summary.md", summary)

            self.assertEqual({row["mode"] for row in rows}, {"text_rag", "mm_rag"})
            self.assertEqual({row["gold_page"] for row in rows}, {1})
            self.assertEqual(len(summary), 2)
            self.assertTrue((root / "comparison_summary.md").exists())

    def _document(self, image_path: Path) -> Document:
        doc_id = "sample_doc"
        page = Page(
            doc_id=doc_id,
            page=1,
            text="This chart contains the number 3.",
            image_path=str(image_path),
            width=32,
            height=24,
        )
        return Document(
            doc_id=doc_id,
            source_path=str(image_path),
            file_name=image_path.name,
            file_path=str(image_path),
            pages=[page],
            chunks=chunk_pages(doc_id, [PageText(page=1, text=page.text)]),
        )


if __name__ == "__main__":
    unittest.main()
