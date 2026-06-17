# MLLMProject

MLLMProject 是一个面向文档理解的多模态 RAG Demo。项目支持上传 PDF 或图片，解析页面文本与视觉区域，构建检索索引，并通过 Text-RAG、MM-RAG 或自动路由模式返回带引用的问答结果。

默认配置使用本地 mock 模型，方便在没有模型权重和 API Key 的环境中快速跑通完整流程；需要真实模型时，可以切换到 DashScope / Qwen 或本地模型栈。

## 功能特性

- PDF、图片、TXT、Markdown 文档导入与页面渲染。
- 文本 chunk、页面级视觉 evidence、表格/图表/公式/代码区域 evidence。
- Text-RAG、MM-RAG、Auto Router 三种查询模式。
- React + FastAPI 前后端联调界面。
- Gradio 单文件演示界面。
- 命令行索引构建、单次问答、批量评测和 DocVQA / ChartQA 子集评测脚本。
- 默认 mock 模型可离线运行，真实模型组件可按需替换。

## 项目结构

```text
.
├── app.py                    # Gradio demo 入口
├── main.py                   # 命令行 Text-RAG baseline
├── pyproject.toml            # Python 包与依赖配置
├── frontend/                 # React + Vite 前端
├── scripts/                  # 索引、问答和评测脚本
├── src/mllmproject/          # 核心后端、RAG、模型、评测代码
├── tests/                    # 单元测试和集成测试
├── docs/                     # API 与开发文档
└── data/eval/sample_questions.json
```

运行时生成的上传文件、解析结果、索引、评测输出、日志和模型权重不会作为源码提交，相关路径已经写入 `.gitignore`。

## 环境要求

- Python 3.10+
- Node.js 18+
- 推荐使用 `uv` 管理 Python 环境

安装 Python 依赖：

```bash
pip install uv
uv sync
```

安装前端依赖：

```bash
cd frontend
npm install
```

需要真实模型或基准评测时再安装额外依赖：

```bash
uv sync --extra real
uv sync --extra benchmarks
```

## 快速开始

构建本地索引：

```bash
python main.py build "多模态大模型大作业说明.pdf"
```

对文档提问：

```bash
python main.py ask "多模态大模型大作业说明.pdf" "期末验收需要提交什么？"
```

输出完整 JSON：

```bash
python main.py ask "多模态大模型大作业说明.pdf" "期末验收需要提交什么？" --json
```

默认索引输出到 `data/processed/`，该目录是运行产物，不需要提交。

## FastAPI 后端

启动 API 服务：

```bash
uv run uvicorn mllmproject.api:app --host 127.0.0.1 --port 8000
```

主要接口：

- `GET /api/v1/models`：列出可用模型。
- `POST /api/v1/files`：上传文档。
- `POST /api/v1/files/{file_id}/parse`：解析文档并构建索引。
- `GET /api/v1/files/{file_id}/chunks`：查看 chunk / evidence。
- `POST /api/v1/query`：执行带引用问答。

更完整的接口说明见 [docs/API_DESIGN.md](docs/API_DESIGN.md)。

## React 前端

确保后端运行在 `127.0.0.1:8000`，然后启动前端：

```bash
cd frontend
npm run dev
```

打开：

```text
http://127.0.0.1:5173
```

前端的 `/api` 请求会通过 Vite 代理到 FastAPI 后端。

## Gradio Demo

```bash
python app.py
```

默认地址：

```text
http://127.0.0.1:7860
```

Gradio 入口适合快速演示上传、解析、页面预览、Top-K evidence、路由信息和引用高亮。

## 模型配置

项目默认使用 mock 模型，不需要下载权重。

使用 DashScope 兼容 Qwen 接口：

```bash
export MLLMPROJECT_MODEL_BACKEND=dashscope
export MLLMPROJECT_DASHSCOPE_API_KEY=your_api_key
```

Windows PowerShell：

```powershell
$env:MLLMPROJECT_MODEL_BACKEND = "dashscope"
$env:MLLMPROJECT_DASHSCOPE_API_KEY = "your_api_key"
```

常用可选变量：

- `MLLMPROJECT_DASHSCOPE_CHAT_MODEL`
- `MLLMPROJECT_DASHSCOPE_VISION_MODEL`
- `MLLMPROJECT_QWEN3_MODEL_PATH`
- `MLLMPROJECT_EMBEDDING_MODEL_ID`
- `MLLMPROJECT_RERANKER_MODEL_ID`
- `MLLMPROJECT_VLM_MAX_NEW_TOKENS`
- `MLLMPROJECT_VLM_MAX_IMAGES`

本地真实模型主要面向脚本和可编程调用场景，默认模型组件包括 Qwen3-VL、BGE-M3、BGE reranker 和 FAISS。

## 评测

示例评测集：

```text
data/eval/sample_questions.json
```

运行单模式评测：

```bash
python scripts/run_eval.py --doc "多模态大模型大作业说明.pdf" --samples data/eval/sample_questions.json --mode text-rag
```

比较 Text-RAG、MM-RAG 和 Auto Router：

```bash
python scripts/run_eval.py --doc "多模态大模型大作业说明.pdf" --samples data/eval/sample_questions.json --mode all
```

运行 DocVQA / ChartQA smoke benchmark：

```bash
python scripts/run_benchmark_eval.py --mock --limit-per-dataset 2 --output-dir data/eval/benchmarks/mock_smoke
```

评测输出默认写入 `data/eval/results/` 或 `data/eval/benchmarks/`，这些目录属于生成产物。

## 测试

```bash
python -m unittest discover -s tests
```

前端构建检查：

```bash
cd frontend
npm run build
```

## 开发说明

核心模块边界：

- `schemas.py`：文档、页面、chunk、evidence、answer 等数据结构。
- `ingest.py`：文档导入、页面渲染和 chunk 生成。
- `pipeline.py`：检索、rerank、路由和回答生成编排。
- `model_stack.py`：mock / real 模型组件工厂。
- `api.py`：FastAPI 服务和前端接口。
- `evaluation.py`：文档级评测与指标汇总。

提交前建议确认：

```bash
python -m unittest discover -s tests
cd frontend && npm run build
```

## 许可证

当前仓库尚未声明开源许可证。正式公开前请补充 `LICENSE` 文件，并在本节说明使用条款。
