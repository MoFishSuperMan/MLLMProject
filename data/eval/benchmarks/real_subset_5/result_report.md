# DocVQA / ChartQA 小样本真实评测报告

## 实验设置

- 数据集：`lmms-lab/DocVQA` validation 前 5 条，`lmms-lab/ChartQA` test 前 5 条。
- 运行环境：`first` conda 环境，RTX 5090 D 32GB。
- 模型：本地 `model/` 目录中的 Qwen3-VL，BGE-M3 embedding，BGE-reranker-v2-m3。
- 运行参数：`top_k=1`，`max_new_tokens=96`，`max_images=1`，输入图像最长边缩放到 1280，BGE 和 reranker 放在 CPU，Qwen3-VL 使用 GPU。
- 输出目录：`data/eval/benchmarks/real_subset_5/`。

## 严格指标结果

| 数据集 | 模式 | 样本数 | EM | ANLS | Answer Match | 平均生成延迟 ms |
|---|---:|---:|---:|---:|---:|---:|
| ChartQA | MM-RAG | 5 | 0.0000 | 0.0000 | 0.0000 | 1956.83 |
| ChartQA | Text-RAG | 5 | 0.0000 | 0.0000 | 0.0000 | 1962.86 |
| DocVQA | MM-RAG | 5 | 0.0000 | 0.0000 | 0.0000 | 1091.06 |
| DocVQA | Text-RAG | 5 | 0.0000 | 0.0000 | 0.0000 | 4156.14 |

严格 EM/ANLS 全为 0 的主要原因是当前后端按“中文完整回答 + 引用”的形式输出，而数据集 gold answer 是短答案。例如 gold 为 `14`，模型输出“图中显示了14种食品项目...”，语义正确但字符串不完全匹配，因此被严格指标判为 0。

## 人工粗判观察

| 数据集 | Text-RAG | MM-RAG | 主要差异 |
|---|---:|---:|---|
| DocVQA | 约 3/5 | 约 4/5 | MM-RAG 在手写/表单收件人问题上答出 `Paul`，Text-RAG 误答为其他人名；两者都未准确读出图表数值 `0.28`。 |
| ChartQA | 约 1/5 | 约 2/5 | 两者都能回答类别数量 `14`；MM-RAG 在 Madagascar vs Fiji 判断题上直接给出 `No`，Text-RAG 倾向于说证据不足。 |

## 典型样例

- DocVQA `docvqa_24582`，问题是“To whom is the document sent?”，gold 为 `Paul`。Text-RAG 回答为 `Mr. Ms. ESEOSE`，MM-RAG 回答为 `Paul`，说明视觉 evidence 对表单/手写区域有帮助。
- DocVQA `docvqa_57349`，gold 为 `ITC Limited`。Text-RAG 和 MM-RAG 都回答正确，说明 OCR 文本足够时纯文本 RAG 已能解决部分版面问题。
- ChartQA `chartqa_000003`，gold 为 `No`。MM-RAG 给出“马达加斯加小于斐济，因此不是”，Text-RAG 认为证据不足；该样例体现图表视觉信息对比较题有增益。
- ChartQA `chartqa_000001`，gold 为 `0.57`。两种模式都误答为 `0.13`，说明当前 OCR/视觉摘要对精确图表数值读取仍不稳定。

## 结论

在这个 5+5 的小样本真实试跑中，MM-RAG 相比 Text-RAG 的优势主要体现在表单定位、手写/视觉区域理解和图表判断题上；当 OCR 已经提取到关键文本时，Text-RAG 与 MM-RAG 表现接近。ChartQA 的精确数值计算仍是短板，尤其是差值、最低柱值等问题，模型容易依据不完整 OCR 或视觉摘要给出错误数值。

本次严格指标不能直接代表真实语义准确率，因为评测脚本尚未从中文完整回答中抽取短答案再计算 EM/ANLS。下一步应增加 answer extraction，将模型输出规整为短答案，再重新计算 DocVQA ANLS 和 ChartQA EM；同时可以把图表问题的 prompt 改成“只输出最终短答案”，减少指标与生成格式之间的偏差。
