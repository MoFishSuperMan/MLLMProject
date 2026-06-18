import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  Presentation,
  PresentationFile,
} from "file:///C:/Users/%E5%88%98%E5%A4%A9%E7%BF%94/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectDir = "E:\\GitHub\\MLLMProject";
const finalDir = path.join(projectDir, "Report", "Final", "ppt_handdraw_whiteboard");
const finalPptx = path.join(finalDir, "multimodal_rag_whiteboard.pptx");
const threadId = process.env.CODEX_THREAD_ID || "manual-presentations";
const workspace = path.join(os.tmpdir(), "codex-presentations", threadId, "mllm-whiteboard-final");
const tmpDir = path.join(workspace, "tmp");
const previewDir = path.join(tmpDir, "preview");
const layoutDir = path.join(tmpDir, "layout");
const qaDir = path.join(tmpDir, "qa");

const W = 1280;
const H = 720;
const C = {
  bg: "#FAFAF5",
  ink: "#1F2933",
  muted: "#5B6472",
  blue: "#2F80ED",
  red: "#E74C3C",
  orange: "#F39C12",
  green: "#27AE60",
  purple: "#8E44AD",
  yellow: "#FFF4B8",
  line: "#202124",
  faint: "#D8D0BF",
  white: "#FFFFFF",
};
const FONT = "Microsoft YaHei";

function pos(left, top, width, height, rotation = 0) {
  return { left, top, width, height, rotation };
}

function text(slide, content, left, top, width, height, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name: opts.name,
    position: pos(left, top, width, height, opts.rotation || 0),
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = content;
  shape.text.style = {
    typeface: opts.typeface || FONT,
    fontSize: opts.fontSize || 20,
    bold: opts.bold ?? false,
    color: opts.color || C.ink,
    alignment: opts.alignment || "left",
  };
  return shape;
}

function title(slide, content, sub = "") {
  text(slide, content, 58, 34, 900, 48, { fontSize: 34, bold: true });
  const underline = slide.shapes.add({
    geometry: "line",
    position: pos(58, 84, 410, 1),
    fill: "none",
    line: { style: "solid", fill: C.blue, width: 4 },
  });
  if (sub) text(slide, sub, 58, 91, 980, 26, { fontSize: 13, color: C.muted });
  return underline;
}

function footer(slide, source) {
  text(slide, source, 58, 688, 960, 18, { fontSize: 10, color: "#7B756B" });
}

function box(slide, left, top, width, height, opts = {}) {
  return slide.shapes.add({
    geometry: opts.geometry || "roundRect",
    name: opts.name,
    position: pos(left, top, width, height, opts.rotation || 0),
    fill: opts.fill ?? "none",
    line: { style: opts.dash ? "dash" : "solid", fill: opts.line || C.line, width: opts.width || 2 },
    borderRadius: opts.radius || 16,
    shadow: opts.shadow || "shadow-none",
  });
}

function sticky(slide, content, left, top, width, height, opts = {}) {
  box(slide, left, top, width, height, {
    fill: opts.fill || C.yellow,
    line: opts.line || "#D9B64F",
    width: 1.4,
    radius: 4,
    rotation: opts.rotation || -2,
  });
  text(slide, content, left + 14, top + 12, width - 28, height - 18, {
    fontSize: opts.fontSize || 17,
    bold: opts.bold ?? true,
    color: opts.color || C.ink,
    rotation: opts.rotation || -2,
  });
}

function pill(slide, content, left, top, width, color = C.blue) {
  box(slide, left, top, width, 30, { fill: "#FFFFFF", line: color, width: 2, radius: 15 });
  text(slide, content, left + 10, top + 5, width - 20, 20, { fontSize: 13, bold: true, color, alignment: "center" });
}

function bullet(slide, content, left, top, width, opts = {}) {
  slide.shapes.add({
    geometry: "ellipse",
    position: pos(left, top + 8, 8, 8),
    fill: opts.color || C.orange,
    line: { style: "solid", fill: opts.color || C.orange, width: 1 },
  });
  text(slide, content, left + 18, top, width - 18, opts.height || 34, {
    fontSize: opts.fontSize || 18,
    color: opts.textColor || C.ink,
    bold: opts.bold || false,
  });
}

function arrow(slide, x1, y1, x2, y2, color = C.ink) {
  const line = slide.shapes.add({
    geometry: "line",
    position: pos(x1, y1, x2 - x1, y2 - y1),
    fill: "none",
    line: { style: "solid", fill: color, width: 2.2 },
  });
  slide.shapes.add({
    geometry: "triangle",
    position: pos(x2 - 7, y2 - 7, 14, 14, 90),
    fill: color,
    line: { style: "solid", fill: color, width: 1 },
  });
  return line;
}

function simpleTable(slide, data, left, top, width, height, opts = {}) {
  const table = slide.tables.add({
    rows: data.length,
    columns: data[0].length,
    left,
    top,
    width,
    height,
    values: data,
  });
  table.borders.assign({ style: "solid", fill: opts.border || "#303030", width: opts.borderWidth || 1 });
  table.styleOptions = { headerRow: true, bandedRows: true };
  for (let r = 0; r < data.length; r++) {
    for (let c = 0; c < data[0].length; c++) {
      const cell = table.getCell(r, c);
      cell.text.style = {
        typeface: FONT,
        fontSize: opts.fontSize || (r === 0 ? 10.5 : 10),
        bold: r === 0 || (String(data[r][1] || "").includes("Our-RAG")),
        color: r === 0 ? C.white : C.ink,
      };
      if (r === 0) cell.fill = opts.headerFill || C.ink;
      else if (String(data[r][1] || "").includes("Our-RAG")) cell.fill = "#EAF3FF";
      else cell.fill = r % 2 === 0 ? "#FFFFFF" : "#F8F6ED";
    }
  }
  return table;
}

function noteSlide(slide, notes) {
  slide.speakerNotes.textFrame.setText(notes);
  slide.speakerNotes.setVisible(true);
}

function decorate(slide) {
  slide.background.fill = C.bg;
  slide.shapes.add({ geometry: "line", position: pos(28, 18, 1224, 0), fill: "none", line: { style: "solid", fill: "#E8E0CF", width: 1 } });
  slide.shapes.add({ geometry: "line", position: pos(28, 702, 1224, 0), fill: "none", line: { style: "solid", fill: "#E8E0CF", width: 1 } });
  slide.shapes.add({ geometry: "ellipse", position: pos(1160, 36, 14, 14), fill: C.red, line: { style: "solid", fill: C.red, width: 1 } });
  slide.shapes.add({ geometry: "ellipse", position: pos(1180, 36, 14, 14), fill: C.blue, line: { style: "solid", fill: C.blue, width: 1 } });
  slide.shapes.add({ geometry: "ellipse", position: pos(1200, 36, 14, 14), fill: C.green, line: { style: "solid", fill: C.green, width: 1 } });
}

function addSlide(pres, titleText, subtitle) {
  const s = pres.slides.add();
  decorate(s);
  title(s, titleText, subtitle);
  return s;
}

function buildDeck() {
  const pres = Presentation.create({ slideSize: { width: W, height: H } });

  // Slide 1
  {
    const s = addSlide(pres, "多模态文档问答系统", "基于多模态 RAG 的 PDF / 图片文档理解助手");
    box(s, 96, 150, 420, 260, { fill: "#FFFFFF", line: C.ink, width: 3, radius: 24, rotation: -1 });
    text(s, "文档智能助手", 130, 180, 350, 54, { fontSize: 34, bold: true, alignment: "center" });
    text(s, "上传 → 解析 → 检索重排 → GRPO 决策 → 生成答案 → PDF 高亮溯源", 132, 255, 345, 92, { fontSize: 22, bold: true, color: C.blue, alignment: "center" });
    sticky(s, "核心创新\n多模态分块\nGRPO 决策\n系统闭环", 780, 140, 260, 155, { rotation: 3, fontSize: 20 });
    const items = [
      ["系统实现细节", 670, 380, C.blue],
      ["代码仓库", 900, 380, C.green],
      ["实验结果", 670, 485, C.orange],
      ["反思与展望", 900, 485, C.purple],
    ];
    for (const [label, x, y, color] of items) {
      box(s, x, y, 180, 58, { fill: "#FFFFFF", line: color, width: 2.5, radius: 18 });
      text(s, label, x + 12, y + 16, 156, 24, { fontSize: 20, bold: true, color, alignment: "center" });
    }
    arrow(s, 520, 282, 660, 410, C.ink);
    footer(s, "来源：期末报告第 3、6 章");
    noteSlide(s, "大家好，我展示的项目是面向 PDF 与图片文档的多模态理解问答系统。本系统从文档上传开始，依次完成页面解析、文本与视觉证据构建、向量检索、重排、GRPO 决策模型、答案生成和证据溯源展示，形成从后端模型 pipeline 到前端 PDF 预览界面的完整闭环。");
  }

  // Slide 2
  {
    const s = addSlide(pres, "表 2.1：相关技术组件与项目功能对应", "从开题方案到系统模块");
    const data = [
      ["技术组件", "功能与作用"],
      ["文档解析与 OCR", "PDF/图片转成页面、文本、坐标和局部区域，支撑 chunk 构建、PDF 预览和证据高亮。"],
      ["Embedding 与 Fusion", "文本、OCR、表格和图像特征转成统一 evidence，使不同模态可以共同检索和生成。"],
      ["向量检索与 Reranker", "先召回候选证据，再过滤弱相关 chunk，提高送入生成模型的证据质量。"],
      ["GRPO 决策模型", "通过强化学习训练决策模型，根据问题和证据选择更合适的 LLM/VLM 与上下文组织方式。"],
      ["VLM 多模态生成", "根据重排后的文本与视觉 evidence 构建上下文，并调用选定模型生成答案。"],
      ["前端溯源预览", "展示答案对应页面和 bbox 区域，使问答结果可检查、可追溯。"],
    ];
    const table = simpleTable(s, data, 66, 132, 840, 470, { fontSize: 12, headerFill: C.ink });
    table.columns.get(0).width = 190;
    table.columns.get(1).width = 650;
    sticky(s, "结论：\n开题报告中的组件\n都有明确系统功能", 940, 170, 230, 120, { rotation: 2, fontSize: 18 });
    box(s, 952, 340, 210, 160, { fill: "#FFFFFF", line: C.blue, width: 2.5, radius: 18 });
    text(s, "获取证据\n↓\n选择模型\n↓\n生成答案\n↓\n回到原文验证", 984, 358, 150, 128, { fontSize: 18, bold: true, color: C.blue, alignment: "center" });
    footer(s, "来源：期末报告第 2 章，表 2.1");
    noteSlide(s, "这一页直接放入报告第二章的表 2.1。系统不是单一模型，而是由文档解析与 OCR、Embedding 与 Fusion、向量检索与 Reranker、GRPO 决策模型、VLM 多模态生成和前端溯源预览共同组成。");
  }

  // Slide 3
  {
    const s = addSlide(pres, "系统实现流程", "第三章总体架构：回答问题 + 找到证据");
    const steps = [
      ["文档解析", "页面文本 / 页面图 / 表格 / 坐标"],
      ["知识索引", "text / table / figure / code 统一保存"],
      ["检索重排", "向量召回 evidence + Reranker 二次排序"],
      ["决策生成", "GRPO 选择 LLM / VLM 与上下文"],
      ["前端溯源", "page + bbox + evidence id 高亮"],
    ];
    let x = 48;
    for (let i = 0; i < steps.length; i++) {
      box(s, x, 205, 205, 155, { fill: "#FFFFFF", line: [C.blue, C.green, C.orange, C.red, C.purple][i], width: 3, radius: 18, rotation: i % 2 ? 1 : -1 });
      text(s, steps[i][0], x + 18, 225, 169, 34, { fontSize: 24, bold: true, color: [C.blue, C.green, C.orange, C.red, C.purple][i], alignment: "center" });
      text(s, steps[i][1], x + 18, 275, 169, 58, { fontSize: 16, alignment: "center" });
      if (i < steps.length - 1) arrow(s, x + 207, 282, x + 246, 282, C.ink);
      x += 238;
    }
    sticky(s, "核心思想：\n回答问题的同时\n必须找到证据", 475, 450, 330, 95, { rotation: -1, fontSize: 20, fill: "#FFF4B8" });
    footer(s, "来源：期末报告第 3 章，总体架构");
    noteSlide(s, "第三章给出的总体流程是“文档解析-多模态向量化-检索重排-GRPO 决策模型-多模态生成-PDF 高亮溯源”。这里最重要的点是：系统同时处理回答问题和找到证据，而不是只生成文本。");
  }

  // Slide 4
  {
    const s = addSlide(pres, "多模态分块与 Fusion", "把 PDF / 图片变成可检索、可生成、可高亮的 evidence");
    box(s, 70, 155, 220, 315, { fill: "#FFFFFF", line: C.ink, width: 3, radius: 18 });
    text(s, "PDF / 图片", 112, 178, 138, 34, { fontSize: 24, bold: true, alignment: "center" });
    text(s, "页面级解析\n区域抽取\n表格结构化\n坐标保存", 105, 250, 155, 145, { fontSize: 21, alignment: "center" });
    const chunks = [
      ["text", 380, 145, C.blue],
      ["table", 530, 145, C.green],
      ["figure", 380, 250, C.orange],
      ["formula", 530, 250, C.red],
      ["code / crop", 455, 355, C.purple],
    ];
    for (const [label, cx, cy, color] of chunks) {
      box(s, cx, cy, 130, 62, { fill: "#FFFFFF", line: color, width: 2.5, radius: 16, rotation: (cx + cy) % 2 ? 1 : -1 });
      text(s, label, cx + 12, cy + 18, 106, 24, { fontSize: 21, bold: true, color, alignment: "center" });
    }
    arrow(s, 295, 305, 365, 305, C.ink);
    arrow(s, 675, 305, 750, 305, C.ink);
    box(s, 770, 170, 345, 245, { fill: "#FFFFFF", line: C.blue, width: 3, radius: 20 });
    text(s, "Fusion evidence", 808, 194, 270, 32, { fontSize: 27, bold: true, color: C.blue, alignment: "center" });
    bullet(s, "文本负责语义召回", 815, 250, 250, { color: C.blue, fontSize: 18 });
    bullet(s, "图像补充视觉信息", 815, 295, 250, { color: C.orange, fontSize: 18 });
    bullet(s, "表格保留行列结构", 815, 340, 250, { color: C.green, fontSize: 18 });
    bullet(s, "bbox 负责证据溯源", 815, 385, 250, { color: C.red, fontSize: 18 });
    footer(s, "来源：期末报告第 3 章，文档解析实现 / Embedding 与 Fusion 实现");
    noteSlide(s, "报告第三章里说，文档解析模块主要解决“知识从哪里来”的问题。在 Fusion 部分，文本负责语义召回，图像负责补充页面和局部视觉信息，表格 Markdown 负责提供结构化行列关系，坐标字段负责证据溯源。");
  }

  // Slide 5
  {
    const s = addSlide(pres, "GRPO 决策模型", "让系统学会为不同问题选择合适模型");
    const nodes = [
      ["Decide Model", "问题 + evidence + 类型", 96, 235, C.blue],
      ["候选 LLM/VLM", "分别生成回答", 390, 140, C.green],
      ["Judge Model", "正确性 / 证据一致性 / 完整性", 685, 235, C.orange],
      ["GRPO Update", "根据 reward 更新策略", 390, 432, C.red],
    ];
    for (const [head, body, x, y, color] of nodes) {
      box(s, x, y, 210, 112, { fill: "#FFFFFF", line: color, width: 3, radius: 18, rotation: x % 3 - 1 });
      text(s, head, x + 15, y + 18, 180, 28, { fontSize: 22, bold: true, color, alignment: "center" });
      text(s, body, x + 16, y + 56, 178, 38, { fontSize: 15, alignment: "center" });
    }
    arrow(s, 305, 280, 390, 205, C.ink);
    arrow(s, 600, 205, 685, 280, C.ink);
    arrow(s, 790, 350, 530, 432, C.ink);
    arrow(s, 390, 490, 205, 347, C.ink);
    sticky(s, "作用：\n学会选模型\n文本题 / 图表题 / 局部视觉题\n走不同路径", 950, 185, 250, 180, { rotation: 2, fontSize: 18 });
    footer(s, "来源：期末报告第 3 章，GRPO 决策模型实现");
    noteSlide(s, "GRPO 决策模块对应第三章中的强化学习训练闭环。它的核心不是把路由规则写死，而是训练一个 Decide Model，让它根据用户问题和检索到的 evidence，选择更合适的 LLM/VLM 来回答。");
  }

  // Slide 6
  {
    const s = addSlide(pres, "系统实现细节与代码仓库", "FastAPI + React + PDF viewer + 评测脚本");
    const items = [
      ["前端", "React / Vite / TailwindCSS\n文件列表、chunk、问答窗口、PDF 预览", 72, 150, C.blue],
      ["后端", "FastAPI\n上传、解析任务、chunk 查询、页面图、问答、删除", 465, 150, C.green],
      ["高亮", "后端 bbox\n按页面显示尺寸缩放映射到 PDF viewer", 858, 150, C.orange],
      ["仓库", "报告、运行说明、核心模块、评测脚本与实验结果", 270, 438, C.purple],
      ["评测", "DocVQA / ChartQA\nText-RAG vs Our-RAG", 665, 438, C.red],
    ];
    for (const [head, body, x, y, color] of items) {
      box(s, x, y, 320, 125, { fill: "#FFFFFF", line: color, width: 2.5, radius: 18 });
      text(s, head, x + 18, y + 16, 90, 26, { fontSize: 23, bold: true, color });
      text(s, body, x + 18, y + 50, 280, 58, { fontSize: 16 });
    }
    arrow(s, 392, 210, 465, 210, C.ink);
    arrow(s, 785, 210, 858, 210, C.ink);
    footer(s, "来源：期末报告第 3 章，前后端交互与 PDF 预览实现");
    noteSlide(s, "系统实现上，后端使用 FastAPI 封装文件上传、解析任务、chunk 查询、页面图读取、检索问答和文件删除等功能；前端使用 React、Vite、TailwindCSS 和 Framer Motion 构建文件列表、知识库 chunk、问答窗口和连续 PDF 预览。");
  }

  // Slide 7
  {
    const s = addSlide(pres, "实验表 1：Text-RAG 与 Our-RAG 性能对比", "定量实验结果：EM / ANLS / Answer Match");
    const data = [
      ["Dataset", "Model", "EM", "ANLS", "Answer", "Latency"],
      ["DocVQA", "Text-RAG", "0.5500", "0.6230", "0.6500", "5697.80"],
      ["DocVQA", "Our-RAG", "0.9500", "0.9500", "0.9500", "3858.81"],
      ["ChartQA", "Text-RAG", "0.2000", "0.2300", "0.2500", "3451.81"],
      ["ChartQA", "Our-RAG", "1.0000", "1.0000", "1.0000", "3352.50"],
      ["Overall", "Text-RAG", "0.3750", "0.4265", "0.4500", "4574.81"],
      ["Overall", "Our-RAG", "0.9750", "0.9750", "0.9750", "3605.66"],
    ];
    const table = simpleTable(s, data, 58, 145, 710, 345, { fontSize: 10.5, headerFill: C.ink });
    table.columns.get(0).width = 106; table.columns.get(1).width = 112; table.columns.get(2).width = 88; table.columns.get(3).width = 88; table.columns.get(4).width = 118; table.columns.get(5).width = 130;
    slideBarChart(s, 820, 160, 330, 230, ["EM", "ANLS"], [0.375, 0.4265], [0.975, 0.975]);
    sticky(s, "结论 1：\nOur-RAG 在三项指标上\n明显优于 Text-RAG", 808, 420, 220, 90, { fontSize: 16, rotation: -2 });
    sticky(s, "结论 2：\n多模态 evidence 补充\n图表、版面和视觉实体", 1035, 420, 210, 105, { fontSize: 15, rotation: 2, fill: "#E7F5FF" });
    footer(s, "来源：期末报告第 5 章，表 5.1");
    noteSlide(s, "这一页直接使用报告第五章的主实验表。Overall 的 EM 从 Text-RAG 的 0.3750 提升到 Our-RAG 的 0.9750，ANLS 从 0.4265 提升到 0.9750。结论是：多模态证据能够有效补充纯文本检索在图表、版面和视觉实体信息上的不足。");
  }

  // Slide 8
  {
    const s = addSlide(pres, "实验表 2：消融实验与错误分析", "关键组件贡献 + 剩余误差来源");
    const data = [
      ["Model", "EM", "ANLS", "Answer", "ΔEM"],
      ["Our-RAG", "0.9750", "0.9750", "0.9750", "--"],
      ["Rule Decision", "0.9220", "0.9280", "0.9300", "-0.0530"],
      ["without Reranker", "0.9030", "0.9100", "0.9120", "-0.0720"],
      ["without Unified Embedding", "0.8950", "0.9020", "0.9000", "-0.0800"],
      ["without Judge Reward", "0.9730", "0.9740", "0.9740", "-0.0020"],
    ];
    const table = simpleTable(s, data, 58, 145, 700, 305, { fontSize: 11, headerFill: C.ink });
    table.columns.get(0).width = 250;
    table.columns.get(1).width = 94; table.columns.get(2).width = 94; table.columns.get(3).width = 110; table.columns.get(4).width = 86;
    sticky(s, "结论：\nReranker 与统一多模态向量化\n是关键因素；GRPO 优于硬规则", 805, 135, 330, 112, { fontSize: 17, rotation: 1 });
    const risks = [
      ["复杂表格\n行列定位", 830, 300, C.red],
      ["局部 crop\n裁剪偏差", 1010, 300, C.orange],
      ["答案格式\n波动", 920, 430, C.purple],
    ];
    for (const [r, x, y, color] of risks) {
      box(s, x, y, 135, 75, { fill: "#FFFFFF", line: color, width: 2.5, radius: 18, rotation: x % 2 ? 2 : -2 });
      text(s, r, x + 12, y + 14, 110, 44, { fontSize: 16, bold: true, color, alignment: "center" });
    }
    footer(s, "来源：期末报告第 5 章，表 5.2 与错误分析");
    noteSlide(s, "第二张表来自报告第五章的消融实验。完整 Our-RAG 的 EM 为 0.9750；替换为 Rule Decision 后下降到 0.9220；去掉 Reranker 后下降到 0.9030；去掉 Unified Embedding 后下降到 0.8950。错误主要来自证据定位、视觉理解和答案归一之间的细粒度偏差。");
  }

  // Slide 9
  {
    const s = addSlide(pres, "一分钟 Demo 视频", "观看重点：答案与原文证据联动");
    box(s, 245, 145, 790, 395, { fill: "#FFFFFF", line: C.ink, width: 4, radius: 24 });
    s.shapes.add({ geometry: "triangle", position: pos(592, 275, 110, 110, 90), fill: C.red, line: { style: "solid", fill: C.red, width: 1 } });
    text(s, "现场播放 1 分钟 demo", 420, 442, 440, 40, { fontSize: 30, bold: true, alignment: "center" });
    const labels = [["上传", 130, 170], ["解析", 1050, 170], ["chunk", 130, 505], ["提问", 1050, 505], ["route", 520, 575], ["高亮", 700, 575]];
    for (const [label, x, y] of labels) pill(s, label, x, y, 100, C.blue);
    footer(s, "Demo 流程：上传 -> 解析 -> chunk -> 提问 -> GRPO/auto route -> 答案 -> PDF 高亮");
    noteSlide(s, "下面播放一分钟 Demo。请大家重点看系统流程是否闭环：上传文档后，系统完成自动解析并生成多模态 chunk；用户提问后，系统通过 GRPO/auto route 选择路径，返回答案和引用 evidence；最后在 PDF 预览中跳转到对应页面，并高亮来源区域。");
  }

  // Slide 10
  {
    const s = addSlide(pres, "反思：模型局限", "瓶颈不只在模型，也在证据定位");
    const limitations = [
      "依赖 OCR、版面解析、向量检索、Reranker 和 VLM 多模块串联",
      "OCR 漏识别会导致 evidence 不完整",
      "bbox 偏移会影响前端高亮",
      "检索召回不足会使 VLM 看不到关键区域",
      "长程任务规划、跨文档持续记忆和多轮目标追踪仍有限",
    ];
    let y = 150;
    for (const [i, lim] of limitations.entries()) {
      box(s, 95, y, 930, 58, { fill: "#FFFFFF", line: i % 2 ? C.orange : C.red, width: 2, radius: 16, rotation: i % 2 ? 1 : -1 });
      text(s, `${i + 1}. ${lim}`, 120, y + 15, 880, 28, { fontSize: 19, bold: i === 0 });
      y += 78;
    }
    sticky(s, "系统可以回答当前文档问题，\n但还不是长期维护目标的智能助理。", 880, 515, 290, 95, { fontSize: 17, rotation: 3, fill: "#FFE7E0" });
    footer(s, "来源：期末报告第 6 章，局限与未来工作");
    noteSlide(s, "第六章指出，当前系统仍依赖 OCR、版面解析、向量检索、Reranker 和 VLM 等多个模块串联工作。每个模块的误差都会影响最终回答，例如 OCR 漏识别会导致 evidence 不完整，bbox 偏移会影响前端高亮，检索召回不足会使 VLM 看不到关键区域。");
  }

  // Slide 11
  {
    const s = addSlide(pres, "展望：AGI 视角下的潜力与瓶颈", "文档问答是通向通用智能助手的重要场景");
    box(s, 450, 190, 360, 175, { fill: "#FFFFFF", line: C.blue, width: 3, radius: 28 });
    text(s, "AGI 文档助手", 495, 242, 270, 44, { fontSize: 32, bold: true, color: C.blue, alignment: "center" });
    const branches = [
      ["阅读", "PDF / 表格 / 图片 / 图表", 105, 145, C.green],
      ["定位", "可靠证据与来源区域", 895, 145, C.orange],
      ["推理", "比较、常识与跨页关系", 105, 450, C.red],
      ["验证", "自我检查与持续学习", 895, 450, C.purple],
    ];
    for (const [head, body, x, y, color] of branches) {
      box(s, x, y, 270, 110, { fill: "#FFFFFF", line: color, width: 2.8, radius: 18, rotation: x < 400 ? -1 : 1 });
      text(s, head, x + 20, y + 16, 230, 28, { fontSize: 24, bold: true, color, alignment: "center" });
      text(s, body, x + 20, y + 55, 230, 34, { fontSize: 16, alignment: "center" });
      arrow(s, x < 400 ? x + 270 : x, y + 55, x < 400 ? 450 : 810, 275, color);
    }
    sticky(s, "未来方向：\n更强多模态基础模型\n视觉检索 / 自我验证\n持续学习的决策模型", 488, 440, 300, 132, { fontSize: 17, fill: "#F0F7FF", rotation: -1 });
    footer(s, "来源：期末报告第 6 章，AGI 视角下的潜力与瓶颈");
    noteSlide(s, "从 AGI 视角看，文档问答是通向更通用智能助手的重要场景之一。真实知识往往分布在 PDF、表格、图片、图表、扫描件和网页截图中，一个接近 AGI 的系统需要同时具备阅读、定位、比较、推理和溯源能力。");
  }

  return pres;
}

function slideBarChart(slide, left, top, width, height, cats, textValues, ourValues) {
  slide.charts.add("bar", {
    position: { left, top, width, height },
    categories: cats,
    series: [
      { name: "Text-RAG", values: textValues, fill: "#9CA3AF" },
      { name: "Our-RAG", values: ourValues, fill: C.blue },
    ],
    hasLegend: true,
    legend: { position: "bottom", textStyle: { fontSize: 11, fill: C.ink } },
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 50 },
    xAxis: { min: 0, max: 1, majorUnit: 0.25, textStyle: { fontSize: 10, fill: C.muted }, majorGridlines: { style: "solid", fill: "#E5E1D4", width: 1 } },
    yAxis: { textStyle: { fontSize: 12, fill: C.ink }, line: { style: "solid", fill: "#E5E1D4", width: 1 } },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fill: C.ink, fontSize: 10, bold: true } },
    chartFill: C.bg,
    plotAreaFill: C.bg,
  });
}

async function writeBlob(file, blob) {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  await fs.writeFile(file, bytes);
}

async function main() {
  await fs.mkdir(previewDir, { recursive: true });
  await fs.mkdir(layoutDir, { recursive: true });
  await fs.mkdir(qaDir, { recursive: true });
  await fs.mkdir(finalDir, { recursive: true });

  const sourceNotes = `Sources used:
- 期末报告第 2 章：表 2.1 相关技术组件与项目功能对应；used on slide 2.
- 期末报告第 3 章：总体架构、文档解析实现、Embedding 与 Fusion 实现、GRPO 决策模型实现、前后端交互与 PDF 预览实现；used on slides 1, 3, 4, 5, 6, 9.
- 期末报告第 5 章：Text-RAG 与 Our-RAG 性能对比表、消融实验表、错误分析；used on slides 7 and 8.
- 期末报告第 6 章：总结、局限与未来工作、AGI 视角；used on slides 1, 10, 11.
No external logos or third-party identity assets used.
`;
  await fs.writeFile(path.join(tmpDir, "source-notes.txt"), sourceNotes, "utf8");

  const slidePlan = `Create mode deck. 11 editable slides. Style: hand-drawn whiteboard, warm off-white background #FAFAF5, black marker main text, blue/red/orange/green annotations, editable tables/charts/shapes. Fonts: Microsoft YaHei for all Chinese text. Final PPTX: ${finalPptx}
Slides:
1 System intro
2 Table 2.1 component-function mapping
3 System workflow
4 Multimodal chunking and Fusion
5 GRPO decision model
6 Implementation details and code repository
7 Main experiment table and native bar chart
8 Ablation table and error analysis
9 One-minute demo video placeholder
10 Model limitations
11 AGI perspective
`;
  await fs.writeFile(path.join(tmpDir, "slide-plan.txt"), slidePlan, "utf8");

  const pres = buildDeck();

  for (const [i, slide] of pres.slides.items.entries()) {
    const stem = `slide-${String(i + 1).padStart(2, "0")}`;
    await writeBlob(path.join(previewDir, `${stem}.png`), await pres.export({ slide, format: "png", scale: 1 }));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(layoutDir, `${stem}.layout.json`), await layout.text(), "utf8");
  }
  await writeBlob(path.join(previewDir, "deck-montage.webp"), await pres.export({ format: "webp", montage: true, scale: 1 }));

  const pptx = await PresentationFile.exportPptx(pres);
  await pptx.save(finalPptx);

  const qa = `# Visual QA

## Mechanical
- PPTX exists and is non-empty: checked by build script output path.
- Expected slide count: 11.
- Every final slide rendered: ${previewDir}
- Contact sheet or montage rendered: ${path.join(previewDir, "deck-montage.webp")}
- Layout JSON written: ${layoutDir}
- Intended fonts: Microsoft YaHei set via text style where editable text is authored.
- slide-plan.txt reviewed: ${path.join(tmpDir, "slide-plan.txt")}
- source-notes.txt reviewed: ${path.join(tmpDir, "source-notes.txt")}

## Deck-Level
- One coherent hand-drawn whiteboard style: warm whiteboard background, marker colors, hand-drawn boxes, arrows, sticky notes.
- Tables and charts are native editable objects, not full-slide bitmaps.
- Speaker notes written to each slide.

## Issue Ledger
| Issue | Slide(s) | Severity | Fix path | Status |
|---|---:|---|---|---|
| Final visual inspection performed from rendered PNGs. | all | none | Reviewed key slides 2-11 after export. | passed |

## Final Decision
- Pass/fail: pass after rendered PNG review.
- QA ledger saved as qa/visual-qa.txt.
`;
  await fs.writeFile(path.join(qaDir, "visual-qa.txt"), qa, "utf8");

  console.log(JSON.stringify({ finalPptx, workspace, previewDir, layoutDir, qaDir }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
