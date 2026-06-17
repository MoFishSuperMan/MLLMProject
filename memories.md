# Memories

## 用户偏好

- 用户希望沟通和交付都更直接，不喜欢过度包装或花哨 demo。
- 当用户说“demo PDF”时，优先理解为一个正常、朴素、可用于测试的 PDF 文档，而不是装饰性页面或说明型演示材料。
- 用户更喜欢实验报告、课程报告一类的普通文档风格，例如 `\documentclass[a4paper,12pt]{article}` 这种文体。
- PDF 内容应保持简单清晰：正文、表格、图、公式、代码块即可，不要加入复杂色彩、装饰图案、额外答案页或测试说明页。
- 如果用户提供具体 LaTeX 片段，应尽量保留其结构和数据，只做必要的中文编译支持、排版整理和错误修复。

## 当前 PDF 需求上下文

- 用户最终明确需要的是一个矩阵乘法实验报告 PDF。
- 内容包括：
  - 一个矩阵乘法数学公式。
  - 一段矩阵乘法文字说明。
  - 一个 C++ 朴素三层循环矩阵乘法代码块。
  - 一个运行时间表格。
  - 一个不同矩阵规模在不同进程数下的运行时间折线图。
  - 若干性能分析文字。
- 不需要：
  - 系统架构图。
  - 复杂 demo 内容。
  - 问答答案页。
  - 花哨色彩或图案。

## 已生成文件

- TeX 源文件：
  - `E:/GitHub/MLLMProject/output/pdf/matrix_multiplication_experiment.tex`
- PDF 文件：
  - `E:/GitHub/MLLMProject/output/pdf/matrix_multiplication_experiment.pdf`
- 最终 PDF 已用 `xelatex` 编译，并用 `pdftoppm` 渲染检查过，结果为 2 页。

## LaTeX 编译方式

在 PowerShell 中进入 PDF 输出目录：

```powershell
cd E:\GitHub\MLLMProject\output\pdf
xelatex -interaction=nonstopmode matrix_multiplication_experiment.tex
xelatex -interaction=nonstopmode matrix_multiplication_experiment.tex
```

## 项目前端/后端相关上下文

- 用户希望项目中的问答展示从普通文本输出优化为 chat 对话布局。
- 输入框形式保持不变。
- 回车提交问题后，文本框需要清空。
- 等待回答期间，希望 chat 对话框里有加载/思考状态，例如左上角星星图标动画。
- 来源展示不应直接输出数组样式，例如 `[page=1, chunk=file_xxx]`。
- 来源应该按组展示，可以用椭圆形标签包裹。
- `chunk = xxx` 希望支持点击或悬停后弹出弹窗，专门展示 chunk 内容。
- 用户提到希望 chunk 样式接近 RAGFlow 风格，避免纯文本堆叠。
- 用户希望后端能把 PDF 中的表格、图片等区域单独抓取为 chunk，并保留图号、表号等结构信息。

## 模型配置相关上下文

- 用户希望使用 Qwen 的模型：`qwen3.5-omni-flash`。
- 后续如果涉及模型配置，应优先检查现有配置文件和环境变量，再按项目已有模式修改。

## 2026-06-17 期末报告与项目协作记忆

### 协作偏好

- 用户希望直接把问题改好，最后告诉他如何启动即可；除非明确要求，不要主动跑 `uv`。
- 修改报告或代码时，优先落地成可用文件，不要只给建议。
- 最终回复尽量简洁，说明改了什么、产物在哪里、是否编译/验证通过。
- 用户对报告表述比较在意，希望内容贴合开题报告承诺的实现点，而不是泛泛做论文综述。

### 期末报告写作偏好

- 期末报告应按照开题报告里提出的目标和模块展开：多模态 RAG、Embedding/Fusion、向量检索、Reranker、GRPO/路由决策、VLM、多模态证据、PDF 高亮溯源、DocVQA/ChartQA 评测。
- “相关工作”更适合写成“相关技术组件与功能对应”，说明每个组件实现了什么功能，例如：
  - GRPO/路由决策对应自动路由和路径选择。
  - Embedding 与 Fusion 对应统一语义表示和多源证据融合。
  - 向量检索与 Reranker 对应多模态知识问答中的证据召回与排序。
  - VLM 对应图表、图片、表格和局部区域理解。
  - PDF 高亮对应可解释溯源。
- “方法与系统实现”应先给总体结构图，再按照上一章组件顺序逐项说明具体实现。
- 正文中不要大量使用等宽代码字体显示路径，例如 `src/mllmproject/ingest.py` 这种会显得很丑；正文用普通宋体风格写模块名即可，完整路径可以放在代码结构表或附录。

### 前端与产品功能偏好

- Preview 栏目应是完整 PDF 预览，而不是把某个 chunk 单独抽出来展示。
- 知识库中选择某个 chunk 时，右侧 PDF preview 应跳转/滚动到对应页面并高亮该 chunk 对应的文字或区域，类似 RAGFlow 的交互。
- 右侧 PDF 预览应支持连续上下翻页，尽量像真正的 PDF viewer，而不是只能看单页。
- 前端布局应让下面两个主要工作区占据屏幕主体；导航栏和主体之间不要留太多空白，但也不要过度拥挤。
- 文件上传后应可以删除；之前存在“上传后删不掉”的 bug，需要注意。

### 模型与错误处理

- 用户遇到过 DashScope/阿里云模型调用错误：
  - `Error code: 400`
  - `Arrearage`
  - `Access denied, please make sure your account is in good standing`
- 这类错误应识别为账号欠费/权限/账户状态问题，并在前端或后端给友好提示，不要直接把供应商原始 JSON 抛给用户。

### 报告产物状态

- 期末报告目录：`Report/Final/sysu-thesis`
- 期末报告 PDF：`Report/Final/sysu-thesis/main.pdf`
- 当前报告已多次按用户反馈重写，最近一次调整重点：
  - 第 2 章改为“相关技术与功能对应”。
  - 第 3 章改为按组件顺序写系统实现。
  - 正文中的行内等宽路径字体已尽量去除。
- 编译命令使用：
  - `latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex`
- 最近一次编译通过，没有未定义引用、未定义文献或 overfull；只剩模板级普通 warning。

## 2026-06-17 memories.md 维护规则

- 后续更新 `memories.md` 时不要覆盖已有内容，只在文件末尾追加新的小节。
- 如果需要修正旧记忆，优先新增“修正/更新”说明，避免直接删除历史上下文。
- 读取 `memories.md` 时应使用 UTF-8；PowerShell 默认显示可能出现乱码，但文件本身不一定损坏。
- 在 PowerShell 中查看中文内容时，优先使用：

```powershell
Get-Content -LiteralPath 'memories.md' -Encoding UTF8
```

- 追加记忆时应保持简洁，重点记录会影响后续决策的偏好、文件路径、已验证状态、常见坑和用户明确否定过的方向。

## 2026-06-17 多模态后端与 chunk 结构调整上下文

- 用户正在对照开题报告检查后端实现，关心“已经实现”和“没有实现”的差异；整理时应排除视频处理相关内容。
- 对 GRPO 强化学习路由、Qwen-0.5B 轻量路由模型、多 VLM 自动选择等内容，用户不要求真的训练模型；只需要有代码路径、实际接入使用，并在前端模型选择中用 `auto` 表示走自动路由方法。
- `auto` 应作为默认模型选择入口，对应后端的自动路由/策略选择流程；报告表述中应说明这是“GRPO 风格/可训练接口/策略更新钩子”，而不是已经完成真实强化学习训练。
- 用户希望补齐统一跨模态特征空间、ViT 图像特征向量化、表格 OCR 转 Markdown 入库等能力；实现口径应强调图片特征与文本 embedding 被统一到同一检索接口，表格优先保留 Markdown/结构化数据。
- chunk 类型必须分明：
  - `text` chunk 只放普通文本，不应混入 PDF 表格内容。
  - `table` chunk 放结构化表格，优先使用 Markdown table 数据，并在前端渲染成表格。
  - `figure` chunk 放图片/裁剪区域及其视觉描述。
  - `code` chunk 放代码块，属于结构化文本，前端应渲染为代码块。
- PDF 中已识别的表格区域应从普通文本 chunk 中排除，避免同一表格既作为 table chunk 又污染 text chunk。
- 旧的已处理文件不会自动拥有新的 chunk 结构；若要看到新的 table/code/figure/text 分离效果，需要重新上传或重新解析文件。
- 用户希望前端可见英文描述尽量改成中文；`chunk`、`PDF`、`auto` 这类项目术语可以保留。
- 本轮实现涉及的核心文件包括：`src/mllmproject/grpo_router.py`、`src/mllmproject/multimodal_embeddings.py`、`src/mllmproject/table_markdown.py`、`src/mllmproject/chunking.py`、`src/mllmproject/ingest.py`、`src/mllmproject/router.py`、`src/mllmproject/model_stack.py`、`src/mllmproject/index.py`、`src/mllmproject/api.py`、`src/mllmproject/schemas.py`、`frontend/src/App.tsx`。

## 2026-06-17 test_demo chunk 粒度修正

- 用户明确指出 `test_demo.pdf` 的结构块不应该被拆得很碎；目标口径是：图片/图表 chunk 1 个、代码块 chunk 1 个、表格 chunk 1 个、公式 chunk 1 个，并且每个都必须完整。
- 已修正公式检测过宽的问题：普通正文里的复杂度表达式、代码行不应被误判为公式；公式 chunk 应合并为完整 Markdown 公式块。
- 已修正代码识别过宽的问题：PDF 中带缩进的图表坐标轴不能因为有空格缩进就被判为代码；代码块应保留完整嵌套结构和缩进。
- 已修正表格/图表混淆：图表坐标区域不能作为 table chunk；带图题的向量图区域应作为 figure chunk。
- 有结构化 layout chunk 时，不再额外追加整页 page visual chunk，避免出现“先视觉截图，再识别”的体验。
- `test_demo.pdf` 当前验证结果为：`text=2, formula=1, code=1, table=1, figure=1`；text chunk 中不再混入代码主体、公式主体、Markdown 表格或图轴刻度。

## 2026-06-17 聊天记录与证据展示修正

- 用户指出前端问题和回答存在“被截断”的体验；应避免在结果页对用户问题、回答正文和引用证据正文使用强制截断，长问题输入框也应支持换行。
- 本地 mock 回答不应只截取 180 字单条证据；回答应能综合多条证据，真实模型默认输出 token 不应压到 96。
- 自动检索应保证结构化证据可进入回答证据，尤其是 `figure` 图片/图表 chunk、`table`、`formula`、`code`，不要只返回 text。
- 数学公式优先识别为 `formula` chunk；带 `$$ ... $$`、`\sum`、`C = AB` 等公式内容不能被 `code` 规则吃掉。
- 知识库顶部原来显示文件名的蓝色椭圆按钮已改为“聊天记录”入口，用来切换到 chat 界面。
- 前端聊天记录应保存到浏览器 `localStorage`，刷新后仍可查看最近记录。
- 聊天历史中的引用证据和弹窗展示应复用知识库 chunk 的结构化展示方式：表格渲染表格，代码渲染代码块，公式显示 `$$` 包裹的公式块，有 crop/preview 的证据显示截图。

## 2026-06-17 公式抽取与 chunk 分数显示修正

- 用户认为公式识别效果仍然差，且存在重复识别；后续不要再只靠“包含数学符号”或固定公式字符串判断公式。
- 公式应像表格转 Markdown 一样作为结构化文本处理：从 PDF layout 行中识别 display formula 锚点（等号、公式编号、数学结构），再合并上下的求和符号、上下限等连续行，生成 `formula_markdown`。
- `∑`、`n`、`l=1` 这类单独的上下结构行不能作为独立公式 chunk，只能并入主公式块。
- 公式去重应基于归一化后的公式内容 key，避免同一公式被 layout 与文本 chunker 重复识别。
- 前端 chunk 右上角的 `0.800` / score 对用户没有实际意义，已从知识库列表、聊天证据卡片和证据弹窗中删除。

## 2026-06-17 旧 processed 缓存重解析说明

- 用户重启前后端后仍看到旧 chunk，是因为前端读取的是 `data/processed/<file_id>/chunks_with_visual.json` 等持久化结果；重启服务不会自动用新解析逻辑覆盖已上传文件。
- 本次已对 `file_d6026490c820` 从 `data/uploads/file_d6026490c820/original.pdf` 原地重解析，更新了 `chunks.json`、`chunks_with_visual.json`、`document.json`、`index.json` 和 `api_file.json`。
- 当前 `data/processed/file_d6026490c820/chunks_with_visual.json` 已验证为：`text=2, formula=1, code=1, table=1, figure=1`。
- 后续如果继续修改 ingest/chunking/table/formula/code/figure 逻辑，已上传文件需要重新上传或重新解析对应 `file_id`，否则 UI 仍会展示旧 JSON。
- 后端已增加启动恢复逻辑：`ApiStore` 初始化时会从 `data/processed/*/api_file.json` 与 `document.json` 恢复已处理文件，并优先加载 `chunks_with_visual.json` 中的最新 chunk，再重建轻量索引。
- 后端模型选项和常见 API 状态/错误文案已中文化；`auto` 作为模型选择标识保留小写英文，因为这是用户指定的自动路由入口。
- 为避免前端选中其它历史副本造成“还是没变”的误判，本次已将所有 `test_demo*.pdf` 的 processed 目录重解析；`file_4dfd4ebfac94`、`file_5475baaea902`、`file_9ee02561b3af`、`file_c82f8fcac3df`、`file_d6026490c820`、`file_fdb74d97615f` 均为 `text=2, formula=1, code=1, table=1, figure=1`。

## 2026-06-17 PyMuPDF 解析分支根因修复

- 用户重新上传后仍看到公式被拆碎，根因不是旧 processed 数据：新文件 `file_85f358c4e8d9` 的 API 与磁盘都曾生成 `text=3, formula=1, code=2, table=1, figure=1`。
- 真正问题是后端 PyMuPDF 分支：`pdf_page.get_text("text")` 会丢失公式中的 `C = AB` 行，只保留 `n ∑` 和 `l=1 AilBlj`；同时 `detect_code_regions` 把 `C = AB, Cij =` 误识别为代码。
- 已修复 `src/mllmproject/ingest.py`：PyMuPDF 分支继续提供裁剪图，但结构化文本内容优先来自 `pdftotext -layout`，再与 PyMuPDF crop 合并；表格 Markdown、公式、代码、图题以 `pdftotext` 结果为准。
- 已新增回归测试：坏的 PyMuPDF code/formula 区域与好的 pdftotext 结构块合并后，只保留 `formula` 与完整 `code`，不会把 `Cij =` 当代码。
- 当前磁盘 `data/processed/file_85f358c4e8d9/chunks_with_visual.json` 已手动修正并验证为 `text=2, formula=1, code=1, table=1, figure=1`；但正在运行的 8000 后端进程仍持有旧内存状态，必须重启后端后前端才会看到新结果。
- 使用可导入当前代码的 Anaconda Python 实例化 `ApiStore(project_root='.')` 后验证：`file_85f358c4e8d9` 恢复为 `chunk_count=6`，`chunks_for_file` 返回 `text=2, formula=1, code=1, table=1, figure=1`。这证明修复后的后端代码和磁盘数据一致；8000 端口仍显示 8 个 chunk 只代表旧进程未重启。

## 2026-06-17 表格/公式前端渲染修复

- 用户截图指出 table/formula chunk 右侧仍显示“文本 chunk”，且公式以 `$$...$$` 源码显示；这是前端展示问题，不是后端结构化内容缺失。
- 当前 `pdftotext` 结构块可能没有 `crop_url` / `preview_url`，因此右侧缩略区不能只依赖图片；没有 crop 时应直接渲染结构化缩略内容。
- 已修复 `frontend/src/App.tsx`：table chunk 无 crop 时右侧缩略也渲染表格预览；formula chunk 无 crop 时右侧缩略渲染公式预览，不再显示“文本 chunk”。
- 已将 `FormulaChunkPreview` 从 `<code>` 源码显示改为轻量数学渲染：去掉 `$$` 外壳，支持 `\sum_{...}^{...}`、`\quad` 和下标如 `C_{ij}`、`A_{il}`、`B_{lj}`。
- 前端构建 `npm.cmd run build` 已通过。

## 2026-06-17 后端表格/公式 crop 修复

- 用户澄清真正需求是后端应正确得到 table/formula crop，而不是前端处理没有 crop 的兜底。
- 已修复 `src/mllmproject/ingest.py`：PyMuPDF 视觉区域检测现在能把 `C = AB`、`Cij`、`n ∑`、`l=1 AilBlj (1)` 合并为一个 `formula` crop，不再误判为 `code`。
- 已修复 `src/mllmproject/chunking.py`：孤立的 `n ∑` 或中文说明句不再被普通正文 chunker 误判为公式 chunk。
- 当前 `file_85f358c4e8d9` 已用带 PyMuPDF 的 Anaconda 环境重新解析，磁盘结果为 `text=2, formula=1, code=1, table=1, figure=1`，且 `formula` 与 `table` 均有 `image_path`、`bbox`、`crop_url`、`preview_url`。
- 已视觉检查 crop：公式 crop 是干净公式区域，表格 crop 是完整表格区域。
- 验证通过：`test_multimodal_components.py`、`test_api.py`、`test_model_adapters.py`、`test_text_baseline.py`，以及 `npm.cmd run build`。

## 2026-06-17 聊天框滚动与引用证据收纳

- 用户希望 chat 结果页在新问题、思考状态或新回复出现时自动滚动到最下面，避免回答生成后还停留在上方。
- 用户希望每条回复的“引用证据”区域单独提供一个小三角收纳按钮；点击后只收起/展开引用证据卡片列表，不影响回答正文和来源标签。
- 已在 `frontend/src/App.tsx` 的 `ResultPage` 中增加底部滚动锚点与自动滚动 effect，并在 `EvidenceReferencePanel` 中增加每条回复独立的引用证据折叠状态。
- 本次前端修改已通过 `npm.cmd run build` 验证。

## 2026-06-17 引用证据默认收纳

- 用户进一步明确：“引用证据”区域默认应该是收纳状态，而不是默认展开。
- 已将 `frontend/src/App.tsx` 中 `EvidenceReferencePanel` 的 `isEvidenceCollapsed` 初始值改为 `true`，新回复默认只显示标题行和小三角，点击后展开证据卡片。
- 修改后 `npm.cmd run build` 已通过。
