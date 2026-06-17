from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi.testclient import TestClient
from PIL import Image

from mllmproject.api import ApiStore, create_app, model_label_for, model_options
from mllmproject.schemas import Chunk, Document, Page


class FastApiBridgeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        store = ApiStore(project_root=self.tmp.name, use_real_models=False)
        self.client = TestClient(create_app(store))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_models_expose_mock_default_for_test_store(self) -> None:
        response = self.client.get("/api/v1/models")

        self.assertEqual(response.status_code, 200)
        models = response.json()["models"]
        self.assertEqual([model["id"] for model in models if model["enabled"]], ["auto", "local_mock"])
        self.assertTrue(next(model for model in models if model["id"] == "auto")["is_default"])

    def test_dashscope_defaults_to_qwen_omni_flash(self) -> None:
        env = {
            "MLLMPROJECT_MODEL_BACKEND": "dashscope",
            "MLLMPROJECT_DASHSCOPE_API_KEY": "test-key",
        }
        with patch.dict(os.environ, env, clear=True):
            store = ApiStore(project_root=self.tmp.name, use_real_models=False)

        self.assertEqual(store.model_config.dashscope_chat_model, "qwen3.5-omni-flash")
        self.assertEqual(store.model_config.dashscope_vision_model, "qwen3.5-omni-flash")
        models = model_options(store.model_config)
        self.assertEqual(models[0]["label"], "auto")
        self.assertEqual(models[1]["label"], "Qwen3.5 Omni Flash")
        self.assertEqual(model_label_for("auto"), "auto")
        self.assertEqual(model_label_for("qwen_dashscope"), "Qwen3.5 Omni Flash")

    def test_upload_parse_chunks_page_image_preview_and_query(self) -> None:
        response = self.client.post(
            "/api/v1/files",
            files=[("files", ("sample.png", self._png_bytes(), "image/png"))],
        )
        self.assertEqual(response.status_code, 200)
        file_id = response.json()["files"][0]["file_id"]

        parse = self.client.post(f"/api/v1/files/{file_id}/parse", json={"include_visual": True})
        self.assertEqual(parse.status_code, 200)
        job_id = parse.json()["job_id"]

        job = self.client.get(f"/api/v1/jobs/{job_id}")
        self.assertEqual(job.status_code, 200)
        self.assertEqual(job.json()["status"], "ready")

        files = self.client.get("/api/v1/files")
        self.assertEqual(files.json()["files"][0]["status"], "ready")
        self.assertEqual(files.json()["files"][0]["page_count"], 1)

        chunks = self.client.get(f"/api/v1/files/{file_id}/chunks")
        self.assertEqual(chunks.status_code, 200)
        chunk = chunks.json()["chunks"][0]
        self.assertEqual(chunk["source_type"], "visual")
        self.assertEqual(chunk["file_id"], file_id)

        image = self.client.get(f"/api/v1/files/{file_id}/pages/1/image")
        self.assertEqual(image.status_code, 200)
        self.assertEqual(image.headers["content-type"], "image/png")

        preview = self.client.get(f"/api/v1/evidence/{chunk['evidence_id']}/preview")
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.headers["content-type"], "image/png")

        query = self.client.post(
            "/api/v1/query",
            json={
                "question": "What is in this image?",
                "file_ids": [file_id],
                "selected_chunk_ids": [chunk["chunk_id"]],
                "model": "auto",
                "mode": "auto",
                "top_k": 3,
            },
        )
        self.assertEqual(query.status_code, 200)
        payload = query.json()
        self.assertTrue(payload["answer"])
        self.assertEqual(payload["model_label"], "auto")
        self.assertIn("GRPO-style", payload["route_reason"])
        self.assertTrue(payload["evidences"])

    def test_ready_file_auto_reloads_when_disk_chunks_change(self) -> None:
        project = Path(self.tmp.name)
        file_id = "file_reload_demo"
        upload_dir = project / "data" / "uploads" / file_id
        processed_dir = project / "data" / "processed" / file_id
        upload_dir.mkdir(parents=True)
        processed_dir.mkdir(parents=True)
        source = upload_dir / "original.pdf"
        source.write_bytes(b"%PDF-1.4\n")

        page = Page(doc_id=file_id, page=1, text="demo")
        old_chunk = Chunk(chunk_id="old_text", doc_id=file_id, page=1, source_type="text", content="old")
        new_chunks = [
            Chunk(chunk_id="new_text", doc_id=file_id, page=1, source_type="text", content="new"),
            Chunk(chunk_id="new_formula", doc_id=file_id, page=1, source_type="formula", content="公式1:\n$$\nC = AB\n$$"),
        ]
        document = Document(doc_id=file_id, source_path=str(source), pages=[page], chunks=[old_chunk], file_name="demo.pdf")
        self._write_json(processed_dir / "document.json", document.to_dict())
        self._write_json(processed_dir / "chunks_with_visual.json", [old_chunk.to_dict()])
        self._write_json(
            processed_dir / "api_file.json",
            {
                "file_id": file_id,
                "file_name": "demo.pdf",
                "original_name": "demo.pdf",
                "mime_type": "application/pdf",
                "size_bytes": source.stat().st_size,
                "status": "ready",
                "page_count": 1,
                "chunk_count": 1,
                "visual_region_count": 0,
            },
        )

        store = ApiStore(project_root=project, use_real_models=False)
        first = store.chunks_for_file(file_id, page=1, page_size=100)
        self.assertEqual(first["total"], 1)

        document.chunks = new_chunks
        self._write_json(processed_dir / "document.json", document.to_dict())
        self._write_json(processed_dir / "chunks_with_visual.json", [chunk.to_dict() for chunk in new_chunks])
        self._write_json(
            processed_dir / "api_file.json",
            {
                "file_id": file_id,
                "file_name": "demo.pdf",
                "original_name": "demo.pdf",
                "mime_type": "application/pdf",
                "size_bytes": source.stat().st_size,
                "status": "ready",
                "page_count": 1,
                "chunk_count": 2,
                "visual_region_count": 1,
            },
        )

        refreshed = store.chunks_for_file(file_id, page=1, page_size=100)

        self.assertEqual(refreshed["total"], 2)
        self.assertEqual([chunk["source_type"] for chunk in refreshed["chunks"]], ["text", "formula"])

    def test_chunks_require_ready_file(self) -> None:
        response = self.client.post(
            "/api/v1/files",
            files=[("files", ("sample.png", self._png_bytes(), "image/png"))],
        )
        file_id = response.json()["files"][0]["file_id"]

        chunks = self.client.get(f"/api/v1/files/{file_id}/chunks")

        self.assertEqual(chunks.status_code, 409)
        self.assertEqual(chunks.json()["error"]["code"], "file_not_ready")

    def _png_bytes(self) -> BytesIO:
        output = BytesIO()
        Image.new("RGB", (48, 32), color=(255, 255, 255)).save(output, format="PNG")
        output.seek(0)
        return output

    def _write_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
