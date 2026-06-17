from __future__ import annotations

from pathlib import Path
import json
import os
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mllmproject.evaluation import label_failure, normalize_gold_pages_from_sample, run_comparison, run_evaluation
from mllmproject.metrics import citation_accuracy, recall_at_k, reciprocal_rank


class EvaluationFrameworkTest(unittest.TestCase):
    def test_run_evaluation_writes_scores_and_failure_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc, samples = self._write_fixture(root)
            previous = Path.cwd()
            os.chdir(root)
            try:
                result = run_evaluation(
                    doc_path=doc,
                    samples_path=samples,
                    output_dir=root / "results",
                    mode="text-rag",
                    top_k=3,
                )
            finally:
                os.chdir(previous)

            self.assertEqual(result["summary"]["count"], 1.0)
            self.assertIn("case_success_rate", result["summary"])
            self.assertIn("failure_counts", result["breakdowns"])
            self.assertIn(result["scores"][0]["case_status"], {"success", "failure"})
            self.assertTrue((root / "results" / "text_rag_scores.csv").exists())
            self.assertTrue((root / "results" / "text_rag_details.json").exists())
            self.assertTrue((root / "results" / "text_rag_summary.json").exists())

    def test_run_comparison_writes_summary_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc, samples = self._write_fixture(root)
            previous = Path.cwd()
            os.chdir(root)
            try:
                result = run_comparison(
                    doc_path=doc,
                    samples_path=samples,
                    output_dir=root / "results",
                    modes=("text-rag", "auto"),
                    top_k=3,
                )
            finally:
                os.chdir(previous)

            self.assertEqual([row["mode"] for row in result["summary"]], ["text_rag", "auto"])
            self.assertTrue((root / "results" / "comparison_summary.csv").exists())
            self.assertTrue((root / "results" / "comparison_summary.json").exists())

    def test_label_failure_priority(self) -> None:
        self.assertEqual(label_failure({"recall_at_5": 0.0}), "retrieval_miss")
        self.assertEqual(
            label_failure({"recall_at_5": 1.0, "citation_accuracy": 0.0}),
            "citation_miss",
        )
        self.assertEqual(
            label_failure({"recall_at_5": 1.0, "citation_accuracy": 1.0, "recall_at_1": 0.0}),
            "rerank_miss",
        )
        self.assertEqual(
            label_failure(
                {
                    "recall_at_5": 1.0,
                    "citation_accuracy": 1.0,
                    "recall_at_1": 1.0,
                    "answer_match": 0.0,
                }
            ),
            "answer_mismatch",
        )

    def test_metrics_accept_multiple_gold_pages(self) -> None:
        self.assertEqual(recall_at_k([11, 4], gold_page=1, k=1, gold_pages=[1, 11]), 1.0)
        self.assertEqual(reciprocal_rank([4, 10], gold_page=None, gold_pages=[10]), 0.5)
        self.assertEqual(citation_accuracy([11], gold_page=1, gold_pages=[1, 11]), 1.0)
        self.assertEqual(normalize_gold_pages_from_sample({"gold_page": 1, "gold_pages": [1, 11]}), [1, 11])

    def _write_fixture(self, root: Path) -> tuple[Path, Path]:
        doc = root / "demo.txt"
        doc.write_text(
            "期末验收需要提交最终报告、Demo 视频、代码仓库和实验结果。\n\n"
            "系统需要支持 Text-RAG 和 MM-RAG 对比评测。",
            encoding="utf-8",
        )
        samples = root / "samples.json"
        samples.write_text(
            json.dumps(
                [
                    {
                        "sample_id": "sample_001",
                        "question": "期末验收需要提交什么？",
                        "answer": "最终报告、Demo 视频、代码仓库和实验结果",
                        "gold_page": 1,
                        "gold_type": "text",
                        "question_type": "text",
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return doc, samples


if __name__ == "__main__":
    unittest.main()
