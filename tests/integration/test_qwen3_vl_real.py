from __future__ import annotations

from pathlib import Path
import os
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mllmproject.real_models import QWEN3_VL_MODEL_ID, Qwen3VLGenerationConfig, Qwen3VLModel


@unittest.skipUnless(
    os.getenv("MLLMPROJECT_RUN_QWEN3") == "1",
    "Set MLLMPROJECT_RUN_QWEN3=1 to run the real Qwen3-VL integration test.",
)
class RealQwen3VLIntegrationTest(unittest.TestCase):
    def test_real_qwen3_vl_generates_visual_summary(self) -> None:
        model_id = os.getenv("MLLMPROJECT_QWEN3_MODEL_PATH") or QWEN3_VL_MODEL_ID
        model = Qwen3VLModel(Qwen3VLGenerationConfig(model_id=model_id, dtype="bf16"))

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "page.png"
            self._write_png(image_path)
            summary = model.generate_visual_summary(str(image_path))

        self.assertTrue(summary.strip())

    def _write_png(self, path: Path) -> None:
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (128, 80), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.text((12, 20), "Qwen3 VL test page", fill=(0, 0, 0))
        image.save(path)


if __name__ == "__main__":
    unittest.main()
