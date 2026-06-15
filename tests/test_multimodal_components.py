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
from mllmproject.schemas import Document, Evidence, Page
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


if __name__ == "__main__":
    unittest.main()
