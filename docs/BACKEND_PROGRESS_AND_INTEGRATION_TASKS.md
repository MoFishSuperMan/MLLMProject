# Backend Progress and Frontend Integration Tasks

Date: 2026-06-15

本文档记录当前后端进展，并说明下一阶段前后端连接需要完成的任务。范围上，当前已完成的是“真实后端能力”，还不是 HTTP API 联调层。

## 1. 当前后端进展

### 1.1 已完成的核心能力

当前后端已经从 mock-only pipeline 升级为可切换的真实 RAG/VLM 后端：

- 文档解析：支持 PDF 文本抽取、页面渲染、文本 chunk 构建。
- 文本检索：已接入 BGE-M3 embedding。
- 向量索引：已接入 FAISS，支持 build/search/save/load。
- 重排：已接入 BGE reranker。
- 多模态模型：已接入本地 `Qwen3-VL-8B-Instruct`，BF16 推理可用。
- 视觉摘要：Qwen3-VL 可以对 PDF 页面截图生成中文视觉摘要。
- 回答生成：Qwen3-VL 可基于检索证据生成答案。
- 完整 MM-RAG：已完成 11 页 PDF 全量视觉摘要索引，并完成 `mm-rag` / `auto` 评测。
- Citation 对齐：模型只输出 `[E1]` 等证据编号，后端统一生成 page/chunk citation，避免模型自写来源和结构化 citation 不一致。
- 评测：支持 `gold_pages` 多页答案，修正了样例页码标注。

### 1.2 当前模型与环境

真实实验使用环境：

- Python: `C:\anaconda\envs\first\python.exe`
- GPU: NVIDIA GeForce RTX 5090 D
- Torch: CUDA 可用，BF16 支持正常
- VLM: `Qwen3-VL-8B-Instruct`，本地路径 `model/`
- Embedding: `BAAI/bge-m3`
- Reranker: `BAAI/bge-reranker-v2-m3`
- Index: `FaissVectorIndex`

已补齐依赖：

- `sentence-transformers`
- `faiss-cpu`
- `pymupdf`

### 1.3 文档实验情况

测试文档：`多模态大模型大作业说明.pdf`

解析结果：

- 页数：11
- 文本 chunks：14
- 页面级 visual chunks：11
- 索引类型：FAISS
- 真实 Text-RAG 构建耗时：约 9-11 秒
- 完整 MM-RAG 构建耗时：约 146.6 秒

Qwen3-VL 页面视觉分析表现：

- 第 1 页：能识别作业基本要求、时间节点、评分表格。
- 第 6 页：能识别“游戏或仿真环境中的多模态智能 Agent”任务说明。
- 第 10 页：能识别“通用评测指标与工具”和指标表格。

单页视觉摘要耗时：

- 冷启动首个页面约 24 秒。
- 后续页面约 8-9 秒/页。

### 1.4 真实 RAG 评测结果

修正 eval 页码和 citation 对齐后，4 条样例上的真实 Text-RAG 结果：

| Metric | Result |
| --- | ---: |
| Recall@1 | 1.0 |
| Recall@5 | 1.0 |
| MRR | 1.0 |
| Citation Accuracy | 1.0 |
| Case Success Rate | 1.0 |

完整 MM-RAG 全流程验收结果：

| Mode | Recall@1 | Recall@5 | MRR | Citation Accuracy | Case Success Rate | Avg Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mm-rag | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 25967 ms |
| auto | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 12673 ms |

完整 MM-RAG 构建输出：

- 总 chunks：25
- 文本 chunks：14
- 页面级 visual chunks：11
- 页面视觉摘要已进入 FAISS 混合索引
- 验收结果已写入 `data/eval/results/real_mm_rag_full_summary.json`

仍为 0 的指标：

- EM
- ANLS

原因：EM/ANLS 适合短答案或抽取式答案，不适合当前 Qwen 生成的长答案。后续建议增加关键词覆盖率或 LLM judge，而不是只依赖字符串完全匹配。

### 1.5 当前后端限制

- Qwen3-VL 全页视觉摘要速度较慢，整篇 11 页预计约 1.5-2 分钟。
- 表格/图像/公式区域检测仍是页面级或占位式能力，还没有接入真实 layout detector。
- HTTP API 尚未实现，React 前端目前仍是 mock 数据。
- 多文件会话管理尚未完成，当前 pipeline 更偏单文档实验。

## 2. 已完成的代码改动概览

新增或改造的后端模块：

- `model_stack.py`：模型栈配置与工厂，支持 mock/real 切换。
- `real_models.py`：BGE-M3、BGE reranker、Qwen3-VL adapter。
- `index.py`：新增 `FaissVectorIndex`。
- `pipeline.py`：支持注入 `ModelStack` 或单独模型组件。
- `service.py`：支持传入真实模型配置。
- `metrics.py`：支持多 gold pages。
- `evaluation.py`：读取并输出 `gold_pages`。
- `sample_questions.json`：修正 eval 页码。

新增测试：

- Qwen3-VL adapter monkeypatch 单测。
- FAISS 检索与 save/load 测试。
- pipeline fake 模型栈端到端测试。
- 多页 gold metrics 测试。
- 可选真实 Qwen3-VL 集成测试。

默认测试命令：

```bash
python -m unittest discover -s tests
```

真实 Qwen3-VL 集成测试命令：

```bash
set MLLMPROJECT_RUN_QWEN3=1
set MLLMPROJECT_QWEN3_MODEL_PATH=C:\Users\杨毅涵\Desktop\多模态大作业\MLLMProject\model
python -m unittest discover -s tests\integration
```

## 3. 前后端连接的任务

前后端连接的目标是让 React 前端从 mock 数据切到真实后端输出。该部分属于接口与状态管理工作，不改变核心 RAG 算法。

### 3.1 后端需要新增 HTTP API 层

建议使用 FastAPI，基础路径为：

```text
/api/v1
```

最小可联调接口：

| Endpoint | Purpose |
| --- | --- |
| `POST /api/v1/files` | 上传一个或多个 PDF/图片文件 |
| `POST /api/v1/files/{file_id}/parse` | 启动解析和索引构建 |
| `GET /api/v1/jobs/{job_id}` | 查询解析进度 |
| `GET /api/v1/files` | 返回已上传文件列表 |
| `GET /api/v1/files/{file_id}/chunks` | 返回分页 evidence chunks |
| `GET /api/v1/files/{file_id}/pages/{page}/image` | 返回页面预览图 |
| `GET /api/v1/models` | 返回可用模型列表 |
| `POST /api/v1/query` | 执行问答并返回 answer/citations/evidences |

可选增强接口：

- `PATCH /api/v1/chunks/{chunk_id}`：启用/禁用 chunk。
- `GET /api/v1/evidence/{evidence_id}/preview`：返回 bbox 高亮图。
- `POST /api/v1/demo/sample-session`：加载样例文件。

### 3.2 后端状态管理

需要新增 session/file/job 管理层：

- 文件上传后生成稳定 `file_id`。
- 文件状态：`uploaded`、`parsing`、`ready`、`failed`。
- 解析任务生成稳定 `job_id`。
- 支持查询 `progress` 和 `message`。
- 每个文件独立保存：
  - 原始文件
  - 页面图
  - chunks
  - index
  - metadata

建议目录结构：

```text
data/uploads/{file_id}/original.pdf
data/processed/{file_id}/document.json
data/processed/{file_id}/chunks.json
data/processed/{file_id}/index.json
data/processed/{file_id}/pages/page_001.png
```

### 3.3 数据结构映射

需要把当前 Python dataclass 映射成前端需要的 JSON shape。

`Document` -> `FileAsset`：

- `doc_id` -> `file_id`
- `file_name` -> `file_name`
- `len(pages)` -> `page_count`
- `len(chunks)` -> `chunk_count`
- `status` 由 job/file store 维护

`Chunk` / `Evidence` -> `EvidenceChunk`：

- `chunk_id`
- `file_id`
- `page`
- `source_type`
- `content`
- `score`
- `bbox`
- `enabled`
- `metadata`
- `image_url`
- `preview_url`

`AnswerResult` -> `/query` response：

- `answer_id`
- `question`
- `answer`
- `model`
- `model_label`
- `route`
- `route_reason`
- `citations`
- `evidences`
- `latency_ms`
- `created_at`

### 3.4 前端需要替换 mock 状态

当前 `frontend/src/App.tsx` 中仍有：

- `sampleFiles`
- `models`
- `chunks`
- 本地 `setTimeout` 模拟 query

需要改为真实请求：

- 进入页面时调用 `GET /api/v1/models`。
- 上传文件时调用 `POST /api/v1/files`。
- 上传成功后调用 `POST /api/v1/files/{file_id}/parse`。
- 轮询 `GET /api/v1/jobs/{job_id}` 直到 ready。
- ready 后调用 `GET /api/v1/files/{file_id}/chunks`。
- 预览图使用 `GET /api/v1/files/{file_id}/pages/{page}/image`。
- 提问时调用 `POST /api/v1/query`。

### 3.5 前后端联调顺序

建议按以下顺序做：

1. 实现 FastAPI app 和 `/api/v1/models`。
2. 实现文件上传和文件列表。
3. 实现单文件 parse，先同步执行，跑通后再改后台任务。
4. 实现 chunks 列表。
5. 实现 page image endpoint。
6. 实现 query endpoint，先支持单文件。
7. 前端替换 mock `models/files/chunks/query`。
8. 支持多文件 query。
9. 加入 parse progress、错误提示、禁用未 ready 文件的 query。
10. 增加 bbox preview 和 sample-session。

### 3.6 验收标准

最小联调成功标准：

- 前端可以上传 PDF。
- 后端完成解析并返回 ready 状态。
- 前端能显示真实 chunk 列表。
- 前端能显示真实页面预览图。
- 前端提问后显示真实 Qwen/BGE RAG 答案。
- 答案引用的 page/chunk 与后端结构化 `citations` 一致。
- 错误响应统一为：

```json
{
  "error": {
    "code": "file_not_found",
    "message": "File not found.",
    "details": {}
  }
}
```

## 4. 建议下一步

优先级最高的下一步不是继续调模型，而是补 HTTP API 层：

1. 新建 `src/mllmproject/api.py`。
2. 新建 `src/mllmproject/session_store.py` 或 `file_store.py`。
3. 将 `RagService` 包装成多文件可管理服务。
4. 前端替换 mock 数据源。

模型侧当前已具备可演示质量；连接层完成后，项目就可以形成完整闭环。
