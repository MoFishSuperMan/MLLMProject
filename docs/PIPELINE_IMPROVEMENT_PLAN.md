# DocVQA / ChartQA Pipeline 问题分析与改进计划

## 1. 背景与当前测试结论

本次真实试跑使用：

- 数据集：DocVQA validation 前 5 条，ChartQA test 前 5 条。
- 模型：本地 `model/` 目录中的 Qwen3-VL，BGE-M3 embedding，BGE-reranker-v2-m3。
- 运行参数：`top_k=1`，`max_new_tokens=96`，`max_images=1`，图像最长边缩到 1280，BGE / reranker 放 CPU，Qwen3-VL 跑 GPU。
- 输出目录：`data/eval/benchmarks/real_subset_5/`。

严格脚本指标中 EM / ANLS 全为 0，但这主要是评测格式问题：模型输出中文完整句和引用，gold answer 是短答案。人工语义粗判结果如下：

| 数据集 | 模式 | 样本数 | 语义正确 | 语义错误 |
|---|---:|---:|---:|---:|
| DocVQA | Text-RAG | 5 | 3 | 2 |
| DocVQA | MM-RAG | 5 | 4 | 1 |
| ChartQA | Text-RAG | 5 | 1 | 4 |
| ChartQA | MM-RAG | 5 | 2 | 3 |

错误原因统计：

| 错误原因 | 次数 | 说明 |
|---|---:|---|
| 图表精确数值读错或计算错 | 4 | 如 `0.28` 被读成 `0.2/0.22`，`0.57` 被答成 `0.13`。 |
| 图表结构信息缺失 | 4 | 无法稳定数柱子、找最低柱、建立类别与数值映射。 |
| OCR / 表单 / 手写区域误读 | 1 | Text-RAG 将收件人问题答成错误人名。 |
| 视觉比较证据不足 | 1 | Text-RAG 无法比较 Madagascar 与 Fiji。 |

核心判断：

- memory 是运行瓶颈：原始高分辨率 DocVQA 页面会让 Qwen3-VL 显存接近 32GB，出现 OOM 或极慢。
- 方法是效果瓶颈：当前 pipeline 只是 OCR 文本 + 整页图像理解，没有局部高清 crop、图表结构解析和短答案抽取。

## 2. 当前 Pipeline 的主要问题

### 2.1 评测层问题：指标失真

当前 `scripts/run_benchmark_eval.py` 直接用完整回答计算 EM / ANLS。模型回答类似：

```text
图中显示了14种食品项目... [E1]。
来源：[page=1, chunk=...]
```

但 gold answer 是：

```text
14
```

因此语义正确也会被判为 EM=0、ANLS=0。这导致自动指标无法真实反映 Text-RAG 与 MM-RAG 的差异。

需要改进：

- 增加短答案抽取。
- Prompt 强制输出 `Final answer: <short answer>`。
- 保存 `raw_prediction` 和 `extracted_answer`，用 `extracted_answer` 计算指标。

### 2.2 Text-RAG 问题：OCR 文本不足以表示图表和版面

Text-RAG 当前流程是：

```text
image -> EasyOCR text -> text chunks -> BGE retrieval -> Qwen answer
```

它的问题是 OCR 文本无法稳定表达：

- 柱状图中柱子的数量。
- 类别与数值的对应关系。
- 最低柱 / 最高柱。
- 图例、颜色与系列的关系。
- 表单字段的空间关系。

所以 Text-RAG 在 ChartQA 上错误明显更多。

需要改进：

- OCR 保留 bbox，而不是只保留纯文本。
- 对表格/表单类 DocVQA 保存字段邻近关系。
- 对 ChartQA 不把 OCR 当作唯一结构来源。

### 2.3 MM-RAG 问题：当前视觉 evidence 太粗

当前 MM-RAG 的 visual evidence 是页面级：

```text
整页图片 -> visual summary/page chunk -> retrieval -> Qwen answer
```

它能改善表单、手写、视觉比较问题，但对精确图表题仍然弱。原因是：

- 整页图像被缩放到 1280，细小刻度和数值标签可能丢失。
- 整页视觉输入没有显式定位关键区域。
- 图表题需要结构化读数，而不是泛化描述。

需要改进：

- 先用低清整页定位关键区域。
- 再用高清局部 crop 回答。
- 对 ChartQA 单独做图表主体裁剪和结构化描述。

### 2.4 Memory 问题：整页高清不可行，局部高清可行

测试中发现：

- 原始 DocVQA 高分辨率页面送入 Qwen3-VL 时，显存贴近 32GB，甚至 OOM。
- 降到最长边 1280 后可以跑，但细节损失会影响图表读数。
- BGE/reranker 放 CPU 后，显存压力主要来自 Qwen3-VL。

因此当前显存不适合“整页高清”，但适合“局部高清 crop”。

建议策略：

```text
低清整页图用于定位 -> 高清局部 crop 用于回答
```

而不是直接换小模型。小模型可以作为后续对照，但第一优先级应是图像策略改造。

## 3. 改进目标

短期目标：

- 让自动指标可信。
- 跑通 DocVQA / ChartQA 各 20 条。
- 降低显存风险，保持可复现。

中期目标：

- 提升 ChartQA 数值题和结构题正确率。
- 提升 DocVQA 表单、手写、图表读数能力。
- 让 Text-RAG 与 MM-RAG 的差异更清楚。

最终报告目标：

- 给出严格指标、短答案指标、人工错误分类。
- 展示 Text-RAG 与 MM-RAG 的定量差异。
- 给出代表性成功/失败案例。

## 4. 分阶段改进计划

### 阶段一：修正评测与输出格式

目标：解决 EM / ANLS 全 0 的指标失真问题。

具体改动：

1. 新增 `src/mllmproject/answer_extraction.py`。
   - `extract_short_answer(raw_answer: str, question: str) -> str`
   - 优先解析 `Final answer:` 后面的内容。
   - 如果没有 `Final answer:`，用规则抽取数字、Yes/No、人名/短实体。

2. 修改 `real_models.py` 的 prompt。
   - 要求模型输出中文解释后，最后一行必须是：

   ```text
   Final answer: <short answer>
   ```

   - 对 ChartQA 可进一步要求：

   ```text
   如果问题要求数量、差值、最大值、最小值，只在 Final answer 中输出数字。
   如果问题是 Yes/No，只输出 Yes 或 No。
   ```

3. 修改 `scripts/run_benchmark_eval.py`。
   - details 中保留：
     - `raw_prediction`
     - `extracted_answer`
     - `gold_answer`
     - `gold_answers`
   - EM / ANLS 用 `extracted_answer` 计算。
   - 继续保留 `prediction` 字段兼容现有结果。

4. 增加测试。
   - 输入“图中显示了14种食品项目... Final answer: 14”，应抽取 `14`。
   - 输入“不是... Final answer: No”，应抽取 `No`。
   - 输入“该文档发送给 Paul...”，应能 fallback 抽取 `Paul` 或至少保留完整答案。

验收标准：

- 之前语义正确的 `chartqa_000000` 应从 EM=0 变成 EM=1。
- `docvqa_57349`、`docvqa_24582` 等短答案样例能正确计分。

### 阶段二：增加局部高清 Crop Pipeline

目标：在不增加显存风险的情况下，让 MM-RAG 看清关键区域。

新增模块：

- `src/mllmproject/vision_regions.py`

建议数据结构：

```python
@dataclass
class RegionCandidate:
    region_id: str
    page: int
    source_type: str
    bbox: list[int]
    image_path: str
    reason: str
    score: float
```

具体策略：

1. OCR 阶段保留 bbox。
   - EasyOCR 当前可返回 detail 信息。
   - 保存每个文本框的坐标、文本、置信度。

2. 基于问题做粗定位。
   - 如果问题包含 `to whom`、`sent`、`addressed`，优先裁剪表单 header / recipient 附近区域。
   - 如果问题包含 `value`、`year`、`actual`、`chart`、`bar`，优先裁剪图表主体区域。
   - 如果 OCR 命中关键词，就裁剪关键词附近上下文区域。

3. 生成 crop 图片。
   - 输出到 `data/eval/benchmarks/cache_regions/{dataset}/{sample_id}/`。
   - crop 可以保持比整页更高清，例如最长边 1600 或 2048。
   - 每个样本最多传 1-2 个 crop，避免显存爆。

4. MM-RAG 构造 evidence。
   - 不只加入 page visual chunk。
   - 加入 `source_type="region"` 或 `source_type="chart_region"` 的 crop evidence。
   - retrieval 时 MM-RAG 优先检索 region evidence。

验收标准：

- DocVQA `docvqa_24582` 这类表单收件人问题，crop evidence 应排在 page evidence 前面。
- DocVQA 图表读数题不再只能看整页低清图。
- 显存维持在 24GB 以下或至少低于当前 32GB 贴边状态。

### 阶段三：ChartQA 图表专用处理

目标：解决 ChartQA 主要错误来源：图表结构缺失和精确数值错误。

具体改动：

1. 新增 `src/mllmproject/chart_preprocess.py`。
   - 裁掉空白边缘。
   - 保存图表主体 crop。
   - 尝试提取 OCR 数字、类别标签、图例文字。

2. 新增 chart prompt。
   - 根据问题类型分路：

   | 问题类型 | 识别关键词 | Prompt 要求 |
   |---|---|---|
   | count | `how many bars/items` | 先数可见元素，再给数字 |
   | difference | `difference between` | 先列两个对象的值，再计算差 |
   | compare | `more than`, `higher` | 先列两个值，再输出 Yes/No |
   | min/max | `lowest`, `highest` | 先定位目标柱，再读值 |

3. 对 ChartQA 不只用 RAG 问答。
   - 由于每个样本本来就是单图，retrieval 不是难点。
   - ChartQA 的关键是 chart understanding。
   - 可以走专门的 `chart_qa` route：

   ```text
   chart image crop + OCR labels + calculation prompt -> Qwen3-VL -> short answer
   ```

4. 保存中间结构。
   - `chart_ocr_labels`
   - `chart_numbers`
   - `chart_question_type`
   - `chart_reasoning`

验收标准：

- `chartqa_000000` 数类别应稳定答 `14`。
- `chartqa_000003` 比较题应稳定答 `No`。
- 差值题和最低值题即使仍错，也能在 details 中看到抽取了哪些候选数值，方便误差分析。

### 阶段四：运行配置优化

目标：让实验能稳定扩展到 20+20。

保留当前稳定设置：

```powershell
--top-k 1
--max-images 1
--max-new-tokens 96
--max-image-side 1280
--embedding-device cpu
--reranker-device cpu
```

后续配置建议：

1. Text-RAG 不必调用 Qwen3-VL。
   - 可增加 `--text-generator-model`。
   - 或用 Qwen3-VL 的文本模式，但不传图片。
   - 未来可换成轻量文本 LLM，提高速度。

2. MM-RAG 只传 crop，不传整页高清图。
   - 全页图用于检索和定位。
   - 回答时传相关 crop。

3. 增量写结果已经加入，应保留。
   - 中途失败时不会丢已完成样本。

4. 增加失败恢复。
   - CLI 支持 `--resume`。
   - 已存在 sample_id + mode 的结果则跳过。

验收标准：

- 能稳定跑完 DocVQA 20 + ChartQA 20。
- 中途 OOM 或中断后，重新运行可以接着跑。

### 阶段五：报告与错误分析自动化

目标：减少人工分析成本，让课程报告直接引用结果。

新增脚本：

- `scripts/write_benchmark_report.py`

输入：

```text
data/eval/benchmarks/{run_name}/
```

输出：

- `result_report.md`
- `error_analysis.md`
- `case_studies.md`

自动统计：

- 每个 dataset/mode 的 EM、ANLS、Answer Match。
- 每个问题类型的正确率。
- 每种错误原因的数量。
- Text-RAG 错但 MM-RAG 对的样例。
- Text-RAG 对但 MM-RAG 错的样例。
- 两者都错的样例。

需要人工标注字段：

```text
manual_correct
manual_error_type
manual_note
```

建议错误类型：

| 错误类型 | 含义 |
|---|---|
| `format_mismatch` | 语义正确但短答案指标没抽出来 |
| `chart_value_error` | 图表精确数值读错 |
| `chart_structure_missing` | 没能理解柱子、图例、坐标轴结构 |
| `ocr_noise` | OCR 文字错误导致回答错 |
| `region_miss` | 没定位到关键视觉区域 |
| `calculation_error` | 数值读对但计算错 |
| `insufficient_evidence` | 模型认为证据不足 |

## 5. 推荐实施顺序

### 第 1 步：短答案抽取

优先级最高。否则自动指标继续不可用。

涉及文件：

- `src/mllmproject/answer_extraction.py`
- `src/mllmproject/real_models.py`
- `scripts/run_benchmark_eval.py`
- `tests/test_benchmark_eval.py`

### 第 2 步：OCR bbox 和 region crop

这是提升 DocVQA 和 memory 效率的关键。

涉及文件：

- `src/mllmproject/vision_regions.py`
- `scripts/run_benchmark_eval.py`
- `src/mllmproject/schemas.py`
- `src/mllmproject/multimodal.py`

### 第 3 步：ChartQA 专用 route

这是提升 ChartQA 的关键。

涉及文件：

- `src/mllmproject/chart_preprocess.py`
- `src/mllmproject/router.py`
- `src/mllmproject/real_models.py`
- `scripts/run_benchmark_eval.py`

### 第 4 步：resume 和报告生成

这是扩展实验规模和写报告的关键。

涉及文件：

- `scripts/run_benchmark_eval.py`
- `scripts/write_benchmark_report.py`

## 6. 预期效果

完成阶段一后：

- 严格指标不再全 0。
- 能真实比较 Text-RAG 和 MM-RAG。

完成阶段二后：

- DocVQA 表单、手写、图表区域问题更稳。
- 显存不再靠整页高清硬撑。

完成阶段三后：

- ChartQA 的 count、compare、min/max 题应明显改善。
- difference 精确计算题仍可能难，但会有更可解释的中间结果。

完成阶段四和五后：

- 可以稳定跑 20+20 或更大子集。
- 报告可以直接展示定量结果和错误分析。

## 7. 最终建议

不要急着换小模型。当前主要问题不是模型大小，而是图像策略和评测策略。

推荐主线：

```text
Qwen3-VL 8B + 低清整页定位 + 局部高清 crop + 短答案抽取 + ChartQA 专用结构化 prompt
```

小模型可以作为对照实验，但不应替代主线。当前 32GB 显存足够支撑局部高清，不够支撑整页高清；因此最划算的改进是把视觉 token 花在关键区域上。
