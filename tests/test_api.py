from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi.testclient import TestClient
from PIL import Image

from mllmproject.api import ApiStore, create_app


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
        self.assertEqual([model["id"] for model in models if model["enabled"]], ["local_mock"])
        self.assertTrue(next(model for model in models if model["id"] == "local_mock")["is_default"])

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
                "model": "local_mock",
                "mode": "mm-rag",
                "top_k": 3,
            },
        )
        self.assertEqual(query.status_code, 200)
        payload = query.json()
        self.assertTrue(payload["answer"])
        self.assertEqual(payload["model_label"], "Local Mock")
        self.assertTrue(payload["evidences"])

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


if __name__ == "__main__":
    unittest.main()
