# MLLM 文档问答 Demo

当前实现分为 React + FastAPI 联调入口、一个 Gradio 兼容入口和一套命令行/评测入口：

- **React + FastAPI 联调入口**：`frontend/` + `src/mllmproject/api.py`，支持 PDF/图片上传、解析轮询、真实 chunk 列表、页面预览和带引用问答。
- **Gradio 兼容入口**：`app.py`，支持 PDF/图片上传、页面预览、模式选择、规则路由、Top-K evidence 和引用页面高亮。
- **后端服务层**：`src/mllmproject/service.py`，提供无 UI 的文档解析、问答和引用预览接口，前端与评测共用这一层。
- **命令行/评测入口**：`main.py` 与 `scripts/`，用于 Text-RAG baseline、单次问答、批量评测和生成报告表格。

后续可以把本地哈希 embedding 替换为 BGE-M3，把本地索引替换为 FAISS，把 mock answer / mock visual summary 替换为 Qwen/Qwen2.5-VL。

## 环境

```bash
pip install uv
uv sync
```

本 Demo 不强制下载模型。PDF 抽文本会优先使用 `PyMuPDF`，如果未安装则使用系统里的 `pdftotext` 命令。

## 命令行使用

构建索引：

```bash
python main.py build "多模态大模型大作业说明.pdf"
```

提问：

```bash
python main.py ask "多模态大模型大作业说明.pdf" "期末验收需要提交什么？"
```

输出 JSON：

```bash
python main.py ask "多模态大模型大作业说明.pdf" "期末验收需要提交什么？" --json
```

索引和 chunk 会保存到：

```text
data/processed/{doc_id}/chunks.json
data/processed/{doc_id}/index.json
data/processed/{doc_id}/metadata.json
```

当前 Text-RAG baseline 不下载模型，完整流程为：

```text
PDF/TXT/MD -> 文本抽取 -> 按页/段落切 chunk -> 本地哈希 embedding -> 本地余弦检索 -> 词重叠 rerank -> mock 引用式回答
```

## React + FastAPI 前端联调

后端使用 `first` conda 环境和本地真实模型，默认监听 `127.0.0.1:8000`：

```powershell
$env:MLLMPROJECT_USE_REAL_MODELS="1"
$env:MLLMPROJECT_QWEN3_MODEL_PATH="C:\Users\杨毅涵\Desktop\多模态大作业\MLLMProject\model"
& "C:\anaconda\envs\first\python.exe" -m uvicorn mllmproject.api:app --host 127.0.0.1 --port 8000
```

前端使用 Vite，`/api` 会自动代理到后端：

```powershell
cd frontend
npm run dev
```

打开：

```text
http://127.0.0.1:5173
```

## Gradio Demo

运行：

```bash
python app.py
```

打开：

```text
http://127.0.0.1:7860
```

页面支持：

- 上传 PDF / 图片。
- 解析文本 chunk。
- 渲染页面预览。
- 选择 `Text-RAG`、`MM-RAG`、`Auto Router`。
- 输入问题并展示 mock 答案。
- 展示 Top-K evidence。
- 展示路由、citation、页面截图和 bbox 高亮。

注意：React + FastAPI 是当前前后端联调入口；`app.py` 保留为 Gradio 兼容展示入口。Text-RAG baseline 的无界面运行走 `main.py` 或 `scripts/demo_query.py`。

## 接口设计

接口设计文档见：

```text
docs/API_DESIGN.md
```

核心替换点：

- `EmbeddingModel.embed_text` / `MockEmbedder` -> BGE-M3
- `VectorIndex` -> FAISS
- `MockReranker` -> BGE-reranker-v2-m3
- `MockGenerator` / `MockVisualSummarizer` -> Qwen2.5-VL

模块边界：

- `schemas.py`：稳定数据结构。
- `ingest.py`：统一文档导入、页面渲染和 chunk 生成。
- `models.py`：mock 模型实现。
- `model_interfaces.py`：可替换模型抽象接口。
- `pipeline.py`：检索、路由、生成的业务编排。
- `service.py`：供前端、脚本和评测复用的无 UI 服务层。
- `engine.py`：保留给 Gradio 前端使用的兼容包装。

## 评测框架

示例评测集：

```text
data/eval/sample_questions.json
```

运行评测：

```bash
python scripts/run_eval.py --doc "多模态大模型大作业说明.pdf" --samples data/eval/sample_questions.json --mode text-rag
python scripts/run_eval.py --doc "多模态大模型大作业说明.pdf" --samples data/eval/sample_questions.json --mode mm-rag
python scripts/run_eval.py --doc "多模态大模型大作业说明.pdf" --samples data/eval/sample_questions.json --mode auto
python scripts/run_eval.py --doc "多模态大模型大作业说明.pdf" --samples data/eval/sample_questions.json --mode all
```

输出文件：

```text
data/eval/results/{mode}_summary.json
data/eval/results/{mode}_scores.csv
data/eval/results/{mode}_details.json
data/eval/results/comparison_summary.csv
data/eval/results/comparison_summary.json
```

当前支持指标：

- `Recall@1`
- `Recall@5`
- `MRR`
- `EM`
- `ANLS`
- `Citation Accuracy`
- `latency_ms`
- `case_success_rate`
- `failure_label`：`retrieval_miss`、`citation_miss`、`rerank_miss`、`answer_mismatch`、`ok`

评测细节文件会保留每个问题的 top-k evidence、引用页、路由、失败标签和问题类型，后续可以直接筛选 5 个成功案例和 5 个失败案例用于报告。

## 多模态组件

多模态相关的可复用组件集中在：

```text
src/mllmproject/multimodal.py
```

当前提供：

- `make_page_visual_chunks`：把每页渲染图转成页面级 visual evidence。
- `make_mock_region_chunks`：生成占位的 figure/table bbox 结构，供后续真实检测器替换。
- `draw_evidence_preview`：根据 evidence 的 `image_path` 和 `bbox` 生成前端引用高亮图。
- `VisualRegion` / `EvidencePreview`：稳定多模态中间结构，便于接入表格、图表、公式区域。

## 测试

```bash
python -m unittest discover -s tests
```

## 开发日志

开发进展记录见 [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md)。
