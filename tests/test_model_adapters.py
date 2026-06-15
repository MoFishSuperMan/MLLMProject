from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mllmproject.index import FaissVectorIndex
from mllmproject.model_stack import ModelConfig, ModelStack
from mllmproject.pipeline import RagPipeline
from mllmproject.real_models import (
    QWEN3_VL_MODEL_ID,
    append_backend_sources,
    citations_from_labels,
    extract_cited_labels,
    strip_model_sources,
    Qwen3VLGenerationConfig,
    Qwen3VLModel,
)
from mllmproject.schemas import Chunk
from mllmproject.schemas import Evidence


class DummyTorch:
    bfloat16 = "BF16"
    float16 = "FP16"
    float32 = "FP32"

    class no_grad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, traceback):
            return False


class DummyInputs(dict):
    input_ids = [[1, 2, 3, 4]]

    def to(self, device):
        self["moved_to"] = device
        return self


class DummyGenerated:
    def __getitem__(self, key):
        return self


class DummyProcessor:
    loaded_model_id = ""
    last_messages = None

    @classmethod
    def from_pretrained(cls, model_id):
        cls.loaded_model_id = model_id
        return cls()

    def apply_chat_template(self, messages, **kwargs):
        self.__class__.last_messages = messages
        return DummyInputs(input_ids=[[1, 2, 3, 4]])

    def batch_decode(self, generated_ids, skip_special_tokens=True):
        return ["decoded qwen answer"]


class DummyModel:
    loaded_model_id = ""
    loaded_kwargs = {}
    device = "cuda:0"

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.loaded_model_id = model_id
        cls.loaded_kwargs = kwargs
        return cls()

    def generate(self, **kwargs):
        self.last_generate_kwargs = kwargs
        return DummyGenerated()


class Qwen3VLAdapterTest(unittest.TestCase):
    def test_qwen3_vl_adapter_uses_bf16_and_chat_template(self) -> None:
        model = Qwen3VLModel(Qwen3VLGenerationConfig(model_id=QWEN3_VL_MODEL_ID, dtype="bf16"))

        with patch(
            "mllmproject.real_models.import_qwen3_vl_dependencies",
            return_value=(DummyTorch, DummyProcessor, DummyModel),
        ):
            summary = model.generate_visual_summary("page.png")

        self.assertEqual(summary, "decoded qwen answer")
        self.assertEqual(DummyProcessor.loaded_model_id, QWEN3_VL_MODEL_ID)
        self.assertEqual(DummyModel.loaded_model_id, QWEN3_VL_MODEL_ID)
        self.assertEqual(DummyModel.loaded_kwargs["dtype"], DummyTorch.bfloat16)
        self.assertEqual(DummyModel.loaded_kwargs["device_map"], "auto")
        messages = DummyProcessor.last_messages
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"][0]["type"], "image")
        self.assertEqual(messages[0]["content"][1]["type"], "text")

    def test_citation_labels_are_mapped_to_backend_sources(self) -> None:
        evidences = [
            Evidence(evidence_id="ev1", doc_id="doc", page=4, source_type="text", content="a", score=0.9, chunk_id="c1"),
            Evidence(evidence_id="ev2", doc_id="doc", page=10, source_type="text", content="b", score=0.8, chunk_id="c2"),
        ]
        labels = extract_cited_labels("答案依据 [E2]，并补充 [ E1]。来源：page=999")
        citations = citations_from_labels(labels, evidences)
        answer = append_backend_sources(strip_model_sources("答案依据 [E2]。\n来源：page=999"), citations)

        self.assertEqual(labels, [2, 1])
        self.assertEqual([citation.page for citation in citations], [10, 4])
        self.assertNotIn("page=999", answer)
        self.assertIn("[page=10, chunk=c2]", answer)


class PipelineModelStackTest(unittest.TestCase):
    def test_pipeline_accepts_fake_visual_summarizer_and_returns_citations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "page.png"
            self._write_png(image_path)

            pipeline = RagPipeline.from_file(
                image_path,
                output_dir=root / "processed",
                include_visual=True,
                model_stack=ModelStack(ModelConfig(use_real_models=False)),
                visual_summarizer=FakeVisualSummarizer(),
            )

            visual_chunks = [chunk for chunk in pipeline.document.chunks if chunk.source_type == "page"]
            self.assertEqual(len(visual_chunks), 1)
            self.assertIn("FAKE_VISUAL_SUMMARY", visual_chunks[0].content)

            result, latency_ms = pipeline.answer("What is on this page?", mode="mm-rag", top_k=1)
            self.assertGreaterEqual(latency_ms, 0.0)
            self.assertTrue(result.answer)
            self.assertTrue(result.citations)
            self.assertTrue(result.evidences)
            self.assertEqual(result.route, "hybrid_route")
            self.assertTrue(result.route_reason)

    def _write_png(self, path: Path) -> None:
        from PIL import Image

        image = Image.new("RGB", (32, 24), color=(255, 255, 255))
        image.save(path)


class FakeVisualSummarizer:
    def generate_visual_summary(self, image_path: str) -> str:
        return f"FAKE_VISUAL_SUMMARY for {Path(image_path).name}"


class FixedEmbedder:
    vectors = {
        "apple chunk": [1.0, 0.0],
        "banana chunk": [0.0, 1.0],
        "apple query": [1.0, 0.0],
    }

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]


class FaissVectorIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError:
            self.skipTest("faiss/faiss-cpu is not installed in this environment")

    def test_faiss_search_and_reload_are_stable(self) -> None:
        chunks = [
            Chunk(chunk_id="c1", doc_id="doc", page=1, source_type="text", content="apple chunk"),
            Chunk(chunk_id="c2", doc_id="doc", page=2, source_type="text", content="banana chunk"),
        ]
        index = FaissVectorIndex(embedder=FixedEmbedder())
        index.build(chunks)

        result = index.search("apple query", top_k=2)
        self.assertEqual([evidence.chunk_id for evidence in result], ["c1", "c2"])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.json"
            index.save(path)
            loaded = FaissVectorIndex.load(path, embedder=FixedEmbedder())
            reloaded = loaded.search("apple query", top_k=2)

        self.assertEqual([evidence.chunk_id for evidence in reloaded], ["c1", "c2"])


if __name__ == "__main__":
    unittest.main()
