import { AnimatePresence, motion } from "framer-motion";
import {
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  ChevronDown,
  Layers3,
  MessageSquareText,
  Moon,
  PanelRightOpen,
  RotateCcw,
  Search,
  Share2,
  Sparkles,
  Trash2,
  Upload,
  X,
  Zap,
} from "lucide-react";
import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";

type View = "knowledge" | "result";
type EvidenceType = "text" | "code" | "figure" | "table" | "formula" | "visual";
type FileStatus = "uploaded" | "queued" | "parsing" | "ready" | "failed";

type EvidenceChunk = {
  id: string;
  chunk_id: string;
  evidence_id: string;
  region_id?: string | null;
  file_id: string;
  file_name: string;
  type: EvidenceType;
  source_type: EvidenceType;
  page: number;
  score: number;
  title: string;
  content: string;
  enabled: boolean;
  bbox?: [number, number, number, number] | null;
  image_url?: string | null;
  crop_url?: string | null;
  preview_url?: string | null;
  metadata?: Record<string, unknown>;
};

type FileAsset = {
  file_id: string;
  file_name: string;
  original_name: string;
  mime_type: string;
  size_bytes: number;
  status: FileStatus;
  page_count: number | null;
  chunk_count: number | null;
  visual_region_count: number | null;
  created_at: string;
  updated_at: string;
  error_message?: string | null;
};

type ModelOption = {
  id: string;
  label: string;
  provider?: string;
  description?: string;
  enabled: boolean;
  is_default?: boolean;
};

type ParseJob = {
  job_id: string;
  file_id: string;
  status: FileStatus;
  progress: number;
  stage?: string;
  message?: string;
  error?: { message?: string } | null;
};

type AnswerResult = {
  answer_id: string;
  question: string;
  answer: string;
  model: string;
  model_label: string;
  route: string;
  route_reason: string;
  selected_chunk_ids: string[];
  evidences: EvidenceChunk[];
  latency_ms: number;
};

type SourceReference = {
  id: string;
  page: number;
  chunkId: string;
  raw: string;
  chunk: EvidenceChunk | null;
};

type ChatTurn = {
  id: string;
  question: string;
  model: string;
  selectedChunks: EvidenceChunk[];
  result: AnswerResult;
  createdAt: string;
};

const API_ORIGIN = "http://127.0.0.1:8000";
const API_BASE = `${API_ORIGIN}/api/v1`;
const FALLBACK_PAGE_ASPECT = "0.707";
const PREVIEW_PAGE_ZOOM = 0.82;
const CHAT_HISTORY_STORAGE_KEY = "mllmproject.chatHistory.v1";

async function apiJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    let message = `请求失败，状态码 ${response.status}`;
    try {
      const payload = await response.json();
      message = payload?.error?.message || message;
    } catch {
      // Keep the HTTP status fallback.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

function normalizeChunk(raw: unknown): EvidenceChunk {
  const item = raw as Record<string, any>;
  const sourceType = asEvidenceType(item.source_type ?? item.type);
  return {
    id: String(item.chunk_id ?? item.id ?? item.evidence_id),
    chunk_id: String(item.chunk_id ?? item.id ?? item.evidence_id),
    evidence_id: String(item.evidence_id ?? item.chunk_id ?? item.id),
    region_id: item.region_id ? String(item.region_id) : null,
    file_id: String(item.file_id ?? ""),
    file_name: String(item.file_name ?? ""),
    type: sourceType,
    source_type: sourceType,
    page: Number(item.page ?? 1),
    score: Number(item.score ?? 0),
    title: String(item.title ?? "证据块"),
    content: String(item.content ?? ""),
    enabled: Boolean(item.enabled ?? true),
    bbox: item.bbox ?? null,
    image_url: absolutizeApiUrl(item.image_url),
    crop_url: absolutizeApiUrl(item.crop_url),
    preview_url: absolutizeApiUrl(item.preview_url),
    metadata: item.metadata ?? {},
  };
}

function absolutizeApiUrl(value: unknown) {
  if (!value) return null;
  const url = String(value);
  if (/^https?:\/\//.test(url)) return url;
  return `${API_ORIGIN}${url.startsWith("/") ? url : `/${url}`}`;
}

function asEvidenceType(value: unknown): EvidenceType {
  const normalized = String(value ?? "text").toLowerCase();
  if (
    normalized === "code" ||
    normalized === "figure" ||
    normalized === "table" ||
    normalized === "formula" ||
    normalized === "visual"
  ) {
    return normalized;
  }
  return "text";
}

function mergeFiles(current: FileAsset[], incoming: FileAsset[]) {
  const byId = new Map(current.map((file) => [file.file_id, file]));
  incoming.forEach((file) => byId.set(file.file_id, file));
  return Array.from(byId.values());
}

function delay(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function errorMessageFrom(error: unknown) {
  return error instanceof Error ? error.message : "接口请求出现异常。";
}

function pageImageUrl(fileId: string | undefined, page: number | undefined) {
  if (!fileId || !page) return null;
  return `${API_BASE}/files/${fileId}/pages/${page}/image`;
}

function numericTuple4(value: unknown): [number, number, number, number] | null {
  if (!Array.isArray(value) || value.length !== 4) return null;
  const values = value.map((item) => Number(item));
  if (values.some((item) => !Number.isFinite(item))) return null;
  const [x1, y1, x2, y2] = values;
  if (x2 <= x1 || y2 <= y1) return null;
  return [x1, y1, x2, y2];
}

function metadataNumber(metadata: Record<string, unknown> | undefined, key: string) {
  const value = Number(metadata?.[key]);
  return Number.isFinite(value) && value > 0 ? value : 0;
}

function pageHighlightBox(chunk: EvidenceChunk | null): [number, number, number, number] | null {
  if (!chunk) return null;
  return (
    numericTuple4(chunk.metadata?.page_bbox) ??
    numericTuple4(chunk.metadata?.original_bbox) ??
    numericTuple4(chunk.bbox)
  );
}

function parseAnswerDisplay(answer: string, chunks: EvidenceChunk[]) {
  const chunkById = new Map<string, EvidenceChunk>();
  chunks.forEach((chunk) => {
    [chunk.id, chunk.chunk_id, chunk.evidence_id, chunk.region_id].forEach((key) => {
      if (key) chunkById.set(String(key), chunk);
    });
  });

  const references: SourceReference[] = [];
  const seen = new Set<string>();
  const bodyLines: string[] = [];
  const sourceMarkerPattern = /(?:来源|Sources?|Source)\s*[:：]/i;
  const sourceTokenPattern = /\[page=(\d+),\s*chunk=([^\]]+)\]/g;

  answer.split(/\r?\n/).forEach((line) => {
    const markerMatch = sourceMarkerPattern.exec(line);
    const sourceText = markerMatch ? line.slice(markerMatch.index) : "";
    const sourceMatches = Array.from(sourceText.matchAll(sourceTokenPattern));

    if (!markerMatch || !sourceMatches.length) {
      bodyLines.push(line);
      return;
    }

    const beforeSource = line.slice(0, markerMatch.index).trim();
    if (beforeSource) bodyLines.push(beforeSource);

    sourceMatches.forEach((match) => {
      const page = Number(match[1]);
      const chunkId = match[2].trim();
      const key = `${page}:${chunkId}`;
      if (seen.has(key)) return;
      seen.add(key);
      references.push({
        id: key,
        page,
        chunkId,
        raw: match[0],
        chunk: chunkById.get(chunkId) ?? null,
      });
    });
  });

  return {
    body: bodyLines.join("\n").trim() || answer.trim(),
    references,
  };
}

function renderInlineAnswer(text: string) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return (
        <strong key={`${part}-${index}`} className="font-semibold text-zinc-950">
          {part.slice(2, -2)}
        </strong>
      );
    }
    return <span key={`${part}-${index}`}>{part}</span>;
  });
}

function answerParagraphs(text: string) {
  const paragraphs = text
    .split(/\n{2,}/)
    .map((item) => item.trim())
    .filter(Boolean);
  return paragraphs.length ? paragraphs : [text.trim()];
}

function pageNumbers(file: FileAsset | null) {
  const count = Math.max(Number(file?.page_count ?? 0), 0);
  return Array.from({ length: count }, (_, index) => index + 1);
}

function fileStatusLabel(status: FileStatus | string | undefined) {
  if (!status) return "未上传";
  return {
    uploaded: "已上传",
    queued: "排队中",
    parsing: "解析中",
    ready: "已解析",
    failed: "解析失败",
  }[String(status)] ?? String(status);
}

function modelOptionLabel(model: Pick<ModelOption, "id" | "label"> | undefined) {
  if (!model) return "模型";
  return model.id === "auto" ? "auto" : model.label;
}

function chunkTypeLabel(type: EvidenceType) {
  return {
    text: "文本",
    code: "代码",
    figure: "图片",
    table: "表格",
    formula: "公式",
    visual: "视觉",
  }[type];
}

function chunkTypeStyle(type: EvidenceType) {
  return {
    text: "bg-zinc-100 text-zinc-700",
    code: "bg-slate-900 text-white",
    figure: "bg-blue-50 text-[#246be8]",
    table: "bg-emerald-50 text-emerald-700",
    formula: "bg-amber-50 text-amber-700",
    visual: "bg-violet-50 text-violet-700",
  }[type];
}

function chunkLabel(chunk: EvidenceChunk) {
  const label = chunk.metadata?.label;
  if (label) return String(label);
  const number = chunk.metadata?.number;
  if (number && chunk.type !== "text") return `${chunkTypeLabel(chunk.type)} ${number}`;
  return `第 ${chunk.page} 页`;
}

function structuredRows(chunk: EvidenceChunk): string[][] {
  const markdownRows = parseMarkdownTable(String(chunk.metadata?.table_markdown ?? ""));
  if (markdownRows.length) return markdownRows;
  const rows = chunk.metadata?.structured_data;
  if (!Array.isArray(rows)) return [];
  return rows
    .filter((row): row is unknown[] => Array.isArray(row))
    .map((row) => row.map((cell) => String(cell ?? "")));
}

function parseMarkdownTable(markdown: string): string[][] {
  if (!markdown.trim()) return [];
  const rows = markdown
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.startsWith("|") && line.endsWith("|"))
    .map((line) =>
      line
        .slice(1, -1)
        .split("|")
        .map((cell) => cell.trim().replace(/\\\|/g, "|")),
    )
    .filter((row) => row.some(Boolean));
  return rows.filter((row) => !row.every((cell) => /^:?-{3,}:?$/.test(cell)));
}

function chunkDescription(chunk: EvidenceChunk) {
  if (chunk.type === "table") {
    return chunk.metadata?.caption ? String(chunk.metadata.caption) : "结构化表格证据";
  }
  if (chunk.type === "figure") {
    return chunk.content || "图片证据及视觉描述";
  }
  return chunk.content;
}

function loadStoredChatTurns(): ChatTurn[] {
  try {
    const raw = window.localStorage.getItem(CHAT_HISTORY_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as ChatTurn[];
    if (!Array.isArray(parsed)) return [];
    return parsed.map((turn) => ({
      ...turn,
      selectedChunks: (turn.selectedChunks ?? []).map(normalizeChunk),
      result: {
        ...turn.result,
        evidences: (turn.result?.evidences ?? []).map(normalizeChunk),
      },
    }));
  } catch {
    return [];
  }
}

function storeChatTurns(turns: ChatTurn[]) {
  try {
    window.localStorage.setItem(CHAT_HISTORY_STORAGE_KEY, JSON.stringify(turns.slice(-30)));
  } catch {
    // Chat history is helpful, but answering should not fail if storage is unavailable.
  }
}

function App() {
  const [view, setView] = useState<View>("knowledge");
  const [query, setQuery] = useState("");
  const [activeQuestion, setActiveQuestion] = useState("");
  const [files, setFiles] = useState<FileAsset[]>([]);
  const [activeFileId, setActiveFileId] = useState("");
  const [chunks, setChunks] = useState<EvidenceChunk[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [focusedId, setFocusedId] = useState("");
  const [models, setModels] = useState<ModelOption[]>([]);
  const [modelId, setModelId] = useState("auto");
  const [answerResult, setAnswerResult] = useState<AnswerResult | null>(null);
  const [chatTurns, setChatTurns] = useState<ChatTurn[]>(() => loadStoredChatTurns());
  const [activeChatId, setActiveChatId] = useState("");
  const [isThinking, setIsThinking] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isComposerOpen, setIsComposerOpen] = useState(true);
  const [statusMessage, setStatusMessage] = useState("");
  const [errorMessage, setErrorMessage] = useState("");

  const hasUploaded = files.length > 0;
  const activeFile = useMemo(
    () => files.find((file) => file.file_id === activeFileId) ?? files[0] ?? null,
    [files, activeFileId],
  );
  const selected = useMemo(
    () => chunks.find((chunk) => chunk.id === focusedId) ?? chunks[0] ?? null,
    [chunks, focusedId],
  );
  const selectedChunks = useMemo(
    () => selectedIds.map((id) => chunks.find((chunk) => chunk.id === id)).filter(Boolean) as EvidenceChunk[],
    [chunks, selectedIds],
  );
  const readyFiles = useMemo(() => files.filter((file) => file.status === "ready"), [files]);
  const modelLabel = modelOptionLabel(models.find((item) => item.id === modelId) ?? { id: modelId, label: modelId });
  const canAsk = readyFiles.length > 0;

  useEffect(() => {
    void loadModels();
    void refreshFiles();
  }, []);

  useEffect(() => {
    storeChatTurns(chatTurns);
  }, [chatTurns]);

  useEffect(() => {
    if (!activeFileId && files.length) {
      setActiveFileId(files[0].file_id);
    }
  }, [activeFileId, files]);

  useEffect(() => {
    if (!activeFile) {
      setChunks([]);
      return;
    }
    if (activeFile.status === "ready") {
      void loadChunks(activeFile.file_id);
    } else {
      setChunks([]);
      setSelectedIds([]);
      setFocusedId("");
    }
  }, [activeFile?.file_id, activeFile?.status]);

  useEffect(() => {
    if (!chunks.length) {
      setSelectedIds([]);
      setFocusedId("");
      return;
    }
    setFocusedId((current) => (chunks.some((chunk) => chunk.id === current) ? current : chunks[0].id));
    setSelectedIds((current) => {
      const valid = current.filter((id) => chunks.some((chunk) => chunk.id === id));
      return valid.length ? valid : [chunks[0].id];
    });
  }, [chunks]);

  const loadModels = async () => {
    try {
      const data = await apiJson<{ models: ModelOption[] }>("/models");
      const enabled = data.models.filter((item) => item.enabled);
      setModels(enabled);
      const defaultModel = enabled.find((item) => item.is_default) ?? enabled[0];
      if (defaultModel) setModelId(defaultModel.id);
    } catch (error) {
      setErrorMessage(errorMessageFrom(error));
    }
  };

  const refreshFiles = async () => {
    try {
      const data = await apiJson<{ files: FileAsset[] }>("/files");
      setFiles(data.files);
    } catch (error) {
      setErrorMessage(errorMessageFrom(error));
    }
  };

  const loadChunks = async (fileId: string) => {
    try {
      const data = await apiJson<{ chunks: unknown[] }>(`/files/${fileId}/chunks?page=1&page_size=100`);
      setChunks(data.chunks.map(normalizeChunk));
      setErrorMessage("");
    } catch (error) {
      setChunks([]);
      setErrorMessage(errorMessageFrom(error));
    }
  };

  const parseAndPoll = async (fileId: string) => {
    const started = await apiJson<{ job_id: string; file_id: string; status: string }>(`/files/${fileId}/parse`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ include_visual: true }),
    });
    await pollJob(started.job_id, fileId);
  };

  const pollJob = async (jobId: string, fileId: string) => {
    for (;;) {
      const job = await apiJson<ParseJob>(`/jobs/${jobId}`);
      setStatusMessage(job.message || "正在解析证据...");
      await refreshFiles();
      if (job.status === "ready") {
        setStatusMessage("证据解析完成。");
        if (fileId === activeFileId || !activeFileId) await loadChunks(fileId);
        return;
      }
      if (job.status === "failed") {
        throw new Error(job.error?.message || "解析失败。");
      }
      await delay(1200);
    }
  };

  const addFiles = async (selectedFiles: FileList | null) => {
    if (!selectedFiles?.length || isUploading) return;
    setIsUploading(true);
    setStatusMessage("正在上传证据...");
    setErrorMessage("");
    try {
      const form = new FormData();
      Array.from(selectedFiles).forEach((file) => form.append("files", file));
      const uploaded = await apiJson<{ files: FileAsset[] }>("/files", { method: "POST", body: form });
      setFiles((current) => mergeFiles(current, uploaded.files));
      setActiveFileId((current) => current || uploaded.files[0]?.file_id || "");
      setIsComposerOpen(true);
      setView("knowledge");
      await Promise.all(uploaded.files.map((file) => parseAndPoll(file.file_id)));
    } catch (error) {
      setErrorMessage(errorMessageFrom(error));
    } finally {
      setIsUploading(false);
      setStatusMessage("");
    }
  };

  const deleteFile = async (fileId: string) => {
    if (!fileId) return;
    setErrorMessage("");
    try {
      await apiJson<{ deleted: boolean; file_id: string }>(`/files/${fileId}`, { method: "DELETE" });
      const remainingChunks = chunks.filter((chunk) => chunk.file_id !== fileId);
      setFiles((current) => {
        const next = current.filter((file) => file.file_id !== fileId);
        const nextActive = next.find((file) => file.file_id !== fileId)?.file_id ?? next[0]?.file_id ?? "";
        setActiveFileId((active) => (active === fileId ? nextActive : active));
        return next;
      });
      setChunks(remainingChunks);
      setSelectedIds((current) => current.filter((id) => remainingChunks.some((chunk) => chunk.id === id)));
      setFocusedId((current) => {
        return remainingChunks.some((chunk) => chunk.id === current) ? current : remainingChunks[0]?.id ?? "";
      });
      setAnswerResult((current) => {
        if (!current?.evidences.some((chunk) => chunk.file_id === fileId)) return current;
        return null;
      });
      if (files.length === 1) {
        setView("knowledge");
      }
    } catch (error) {
      setErrorMessage(errorMessageFrom(error));
    }
  };

  const loadSample = async () => {
    if (isUploading) return;
    setIsUploading(true);
    setStatusMessage("正在加载示例证据...");
    setErrorMessage("");
    try {
      const data = await apiJson<{ files: FileAsset[]; active_file_id: string; jobs: { job_id: string; file_id: string }[] }>(
        "/demo/sample-session",
        { method: "POST" },
      );
      setFiles((current) => mergeFiles(current, data.files));
      setActiveFileId(data.active_file_id);
      setIsComposerOpen(true);
      setView("knowledge");
      await Promise.all(data.jobs.map((job) => pollJob(job.job_id, job.file_id)));
    } catch (error) {
      setErrorMessage(errorMessageFrom(error));
    } finally {
      setIsUploading(false);
      setStatusMessage("");
    }
  };

  const returnToCover = () => {
    setFiles([]);
    setActiveFileId("");
    setChunks([]);
    setQuery("");
    setActiveQuestion("");
    setAnswerResult(null);
    setSelectedIds([]);
    setFocusedId("");
    setErrorMessage("");
    setStatusMessage("");
    setView("knowledge");
  };

  const toggleChunkSelection = (id: string) => {
    setFocusedId(id);
    setSelectedIds((current) =>
      current.includes(id) ? current.filter((selectedId) => selectedId !== id) : [...current, id],
    );
  };

  const runQuery = async () => {
    const question = query.trim();
    if (!question || !canAsk || isThinking) return;
    setActiveQuestion(question);
    setQuery("");
    setAnswerResult(null);
    setView("result");
    setIsThinking(true);
    setErrorMessage("");
    try {
      const result = await apiJson<AnswerResult>("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          file_ids: readyFiles.map((file) => file.file_id),
          selected_chunk_ids: selectedIds,
          model: modelId,
          mode: "auto",
          top_k: 5,
        }),
      });
      const normalizedResult = {
        ...result,
        evidences: result.evidences.map(normalizeChunk),
      };
      setAnswerResult(normalizedResult);
      const turn: ChatTurn = {
        id: normalizedResult.answer_id,
        question,
        model: normalizedResult.model_label || modelLabel,
        selectedChunks,
        result: normalizedResult,
        createdAt: new Date().toISOString(),
      };
      setChatTurns((current) => [...current, turn].slice(-30));
      setActiveChatId(turn.id);
    } catch (error) {
      setErrorMessage(errorMessageFrom(error));
    } finally {
      setIsThinking(false);
    }
  };

  return (
    <div className="min-h-screen bg-[linear-gradient(135deg,#fbfbfa_0%,#f8fbff_48%,#fffaf2_100%)] text-[#202833]">
      <Header
        hasUploaded={hasUploaded}
        view={view}
        onBack={view === "result" ? () => setView("knowledge") : returnToCover}
      />
      {errorMessage ? (
        <div className="mx-auto mt-4 max-w-[2020px] px-8">
          <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm font-medium text-red-700">
            {errorMessage}
          </div>
        </div>
      ) : null}

      <main
        className="mx-auto grid min-h-[calc(100vh-60px)] max-w-[2020px] grid-cols-1 border-x border-dashed border-zinc-200 px-5 pb-5 pt-4 lg:grid-cols-[minmax(520px,1.2fr)_minmax(360px,0.8fr)] lg:gap-5"
        onMouseDown={() => setIsComposerOpen(false)}
      >
        <section className="min-w-0 lg:h-[calc(100vh-80px)]">
          <AnimatePresence mode="wait">
            {!hasUploaded ? (
              <WelcomePanel key="welcome" onUpload={addFiles} onSample={loadSample} />
            ) : view === "knowledge" ? (
              <KnowledgePage
                key="knowledge"
                activeFile={activeFile}
                chunks={chunks}
                selectedIds={selectedIds}
                focusedId={focusedId}
                statusMessage={statusMessage}
                errorMessage={errorMessage}
                chatCount={chatTurns.length}
                onOpenChat={() => setView("result")}
                onSelect={toggleChunkSelection}
              />
            ) : (
              <ResultPage
                key="result"
                query={activeQuestion || answerResult?.question || query}
                model={modelLabel}
                selected={selected}
                selectedChunks={selectedChunks}
                result={answerResult}
                chatTurns={chatTurns}
                pendingQuestion={isThinking ? activeQuestion : ""}
                activeChatId={activeChatId}
                onSelectChat={setActiveChatId}
                isThinking={isThinking}
                onBack={() => setView("knowledge")}
                onReturnToCover={returnToCover}
              />
            )}
          </AnimatePresence>
        </section>

        <PreviewPanelV2
          activeFile={activeFile}
          files={files}
          hasUploaded={hasUploaded}
          selected={selected}
          onActiveFile={setActiveFileId}
          onDeleteFile={deleteFile}
          onUpload={addFiles}
        />
      </main>

      <Composer
        isOpen={isComposerOpen}
        hasUploaded={canAsk}
        query={query}
        modelId={modelId}
        models={models}
        isThinking={isThinking}
        onOpen={() => setIsComposerOpen(true)}
        onModelChange={setModelId}
        onQueryChange={setQuery}
        onSubmit={runQuery}
      />
    </div>
  );
}

function Header({
  hasUploaded,
  view,
  onBack,
}: {
  hasUploaded: boolean;
  view: View;
  onBack: () => void;
}) {
  return (
    <header className="flex h-[60px] items-center justify-between border-b border-zinc-100 bg-white/92 px-6 backdrop-blur">
      <div className="flex min-w-0 items-center gap-4">
        {hasUploaded ? (
          <button
            className="inline-flex items-center gap-2 rounded-lg px-2.5 py-2 text-sm font-medium text-zinc-700 transition hover:bg-zinc-100"
            onClick={onBack}
          >
            <ArrowLeft size={17} />
            返回
          </button>
        ) : null}
        <div className="text-[23px] font-semibold tracking-tight text-zinc-900">多模态文档问答演示</div>
        {hasUploaded ? (
          <span className="hidden rounded-full border border-zinc-200 bg-zinc-50 px-3 py-1 text-sm font-medium text-zinc-500 sm:inline-flex">
            {view === "result" ? "问答结果" : "知识库"}
          </span>
        ) : null}
      </div>
      <div className="flex items-center gap-5 text-sm text-zinc-500">
        <span className="hidden sm:inline">证据知识库</span>
        <button className="inline-flex h-11 items-center gap-2 rounded-xl bg-[#3178f6] px-4 font-medium text-white shadow-sm transition hover:bg-[#246be8]">
          <Share2 size={18} />
          分享
        </button>
        <button className="grid h-10 w-10 place-items-center rounded-full text-zinc-700 transition hover:bg-zinc-100">
          <Moon size={22} />
        </button>
      </div>
    </header>
  );
}

function UploadButton({
  children,
  onUpload,
  variant = "plain",
}: {
  children: React.ReactNode;
  onUpload: (files: FileList | null) => void | Promise<void>;
  variant?: "plain" | "primary";
}) {
  return (
    <label
      className={`inline-flex cursor-pointer items-center gap-2 rounded-xl px-3 py-2 text-sm font-medium transition ${
        variant === "primary"
          ? "bg-[#3178f6] text-white shadow-sm hover:bg-[#246be8]"
          : "border border-zinc-200 bg-white text-zinc-700 hover:bg-zinc-50"
      }`}
    >
      <Upload size={16} />
      {children}
      <input
        className="hidden"
        type="file"
        multiple
        accept=".pdf,.png,.jpg,.jpeg,.webp"
        onChange={(event) => {
          void onUpload(event.target.files);
          event.currentTarget.value = "";
        }}
      />
    </label>
  );
}

function WelcomePanel({
  onUpload,
  onSample,
}: {
  onUpload: (files: FileList | null) => void | Promise<void>;
  onSample: () => void | Promise<void>;
}) {
  return (
    <motion.section
      className="flex h-full max-w-[820px] flex-col overflow-hidden rounded-xl border border-zinc-200 bg-white shadow-[0_18px_55px_rgba(36,48,64,0.06)]"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.24 }}
    >
      <div className="h-2 bg-[linear-gradient(90deg,#d8ecff,#e8f7ef,#fff0d0,#f1e8ff)]" />
      <div className="flex flex-1 flex-col bg-[linear-gradient(180deg,#ffffff_0%,#fbfdff_48%,#fffdf8_100%)] px-8 py-7">
        <div className="mb-5 inline-flex w-fit items-center gap-2 rounded-full border border-blue-100 bg-blue-50 px-3 py-1.5 text-sm font-medium text-[#246be8]">
          <Sparkles size={15} />
          证据工作台
        </div>

        <h1 className="max-w-[620px] text-[38px] font-semibold leading-tight tracking-tight text-zinc-950">
          多模态证据问答演示
        </h1>
        <p className="mt-5 max-w-[720px] text-[17px] leading-8 text-zinc-600">
          上传报告、PDF、截图或图表。左侧展示已经索引的知识库 chunk，右侧用于预览文件页面、
          表格区域、图片裁剪和引用位置。
        </p>

        <div className="mt-7 flex flex-wrap items-center gap-3">
          <UploadButton onUpload={onUpload} variant="primary">
            上传证据
          </UploadButton>
          <button
            className="inline-flex items-center gap-2 rounded-xl border border-zinc-200 bg-white px-3 py-2 text-sm font-medium text-zinc-700 transition hover:bg-zinc-50"
            onClick={onSample}
          >
            <Sparkles size={16} />
            使用示例文件
          </button>
        </div>

        <div className="mt-8 grid gap-3 sm:grid-cols-3">
          {[
            ["1", "上传文件", "支持 PDF、图片、截图和报告附件"],
            ["2", "预览证据", "切换文件并查看选中页面或区域"],
            ["3", "发起提问", "选择模型并生成带引用的答案"],
          ].map(([step, title, text], index) => (
            <div
              key={step}
              className={`rounded-xl border p-4 ${
                index === 0
                  ? "border-blue-100 bg-blue-50/70"
                  : index === 1
                    ? "border-emerald-100 bg-emerald-50/60"
                    : "border-amber-100 bg-amber-50/60"
              }`}
            >
              <div className="mb-3 grid h-7 w-7 place-items-center rounded-full bg-white text-sm font-semibold text-[#3178f6]">
                {step}
              </div>
              <h2 className="font-medium text-zinc-950">{title}</h2>
              <p className="mt-1 text-sm leading-6 text-zinc-500">{text}</p>
            </div>
          ))}
        </div>

        <div className="mt-auto grid gap-3 pt-8 sm:grid-cols-2">
          <InfoCard
            title="演示流程"
            text="先上传材料，解析后查看知识库，提问后进入结果页。"
          />
          <InfoCard
            title="后端联调"
            text="前端已对接后端接口，可展示真实 chunk、页面预览和引用结果。"
          />
        </div>
      </div>
    </motion.section>
  );
}

function InfoCard({ title, text }: { title: string; text: string }) {
  return (
    <div className="rounded-xl border border-zinc-200 bg-white/78 p-4 shadow-sm">
      <div className="text-sm font-medium text-zinc-950">{title}</div>
      <p className="mt-1 text-sm leading-6 text-zinc-500">{text}</p>
    </div>
  );
}

function KnowledgePage({
  activeFile,
  chunks,
  selectedIds,
  focusedId,
  statusMessage,
  errorMessage,
  chatCount,
  onOpenChat,
  onSelect,
}: {
  activeFile: FileAsset | null;
  chunks: EvidenceChunk[];
  selectedIds: string[];
  focusedId: string;
  statusMessage: string;
  errorMessage: string;
  chatCount: number;
  onOpenChat: () => void;
  onSelect: (id: string) => void;
}) {
  const statusLabel = fileStatusLabel(activeFile?.status ?? "uploaded");
  const totalChunks = activeFile?.chunk_count ?? chunks.length;
  const pageCount = activeFile?.page_count ?? 0;
  const visualCount = activeFile?.visual_region_count ?? 0;

  return (
    <motion.section
      className="flex h-full min-w-0 flex-col"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.22 }}
    >
      <div className="min-h-0 flex-1 overflow-hidden rounded-xl border border-zinc-200 bg-white">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-100 px-5 py-3">
          <div>
            <h2 className="text-lg font-semibold text-zinc-950">已索引证据</h2>
            <p className="mt-1 text-sm leading-6 text-zinc-500">
              可选择多个 chunk 参与检索，再次点击即可取消选择。
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-1.5 rounded-full bg-[#edf7f3] px-3 py-1.5 text-sm font-medium text-[#26765b]">
              <CheckCircle2 size={15} />
              {statusLabel}
            </span>
            <span className="inline-flex items-center gap-1.5 rounded-full bg-zinc-100 px-3 py-1.5 text-sm font-medium text-zinc-600">
              <Layers3 size={15} />
              已选 {selectedIds.length} 个
            </span>
            <button
              type="button"
              className="inline-flex max-w-[220px] items-center gap-1.5 truncate rounded-full bg-[#eef4ff] px-3 py-1.5 text-sm font-medium text-[#3178f6] transition hover:bg-blue-100"
              onClick={onOpenChat}
            >
              <MessageSquareText size={15} className="shrink-0" />
              <span className="truncate">聊天记录 {chatCount ? `(${chatCount})` : ""}</span>
            </button>
          </div>
        </div>

        <div className="thin-scrollbar max-h-[calc(100%-118px)] divide-y divide-zinc-100 overflow-y-auto">
          {chunks.length ? chunks.map((chunk) => (
            <ChunkRow
              key={chunk.id}
              chunk={chunk}
              active={selectedIds.includes(chunk.id)}
              focused={chunk.id === focusedId}
              onSelect={() => onSelect(chunk.id)}
            />
          )) : (
            <div className="px-6 py-10 text-sm leading-6 text-zinc-500">
              {errorMessage || statusMessage || "等待证据索引完成..."}
            </div>
          )}
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-3 text-sm text-zinc-500">
          <span>共 {totalChunks} 个 chunk · {pageCount} 页 · {visualCount} 个视觉区域</span>
          <button className="inline-flex items-center gap-1.5 rounded-lg px-2 py-1 font-medium text-zinc-700 transition hover:bg-zinc-100">
            每页 10 个
            <ChevronDown size={15} />
          </button>
        </div>
      </div>
    </motion.section>
  );
}

function ChunkRow({
  chunk,
  active,
  focused,
  onSelect,
}: {
  chunk: EvidenceChunk;
  active: boolean;
  focused: boolean;
  onSelect: () => void;
}) {
  const rows = structuredRows(chunk);
  const cropUrl = chunk.type === "text" || chunk.type === "code" ? null : chunk.crop_url || chunk.preview_url;

  return (
    <button
      className={`grid w-full grid-cols-[22px_minmax(0,1fr)] gap-4 px-5 py-4 text-left transition md:grid-cols-[22px_minmax(0,1fr)_180px] ${
        active
          ? "bg-blue-50/55"
          : focused
            ? "bg-zinc-50/80"
            : "hover:bg-zinc-50/70"
      }`}
      onClick={onSelect}
      aria-pressed={active}
    >
      <span
        className={`mt-1 grid h-4 w-4 place-items-center rounded border ${
          active
            ? "border-[#3178f6] bg-[#3178f6] text-white"
            : focused
              ? "border-[#3178f6]/60 bg-white"
              : "border-zinc-300 bg-white"
        }`}
      >
          {active ? <CheckCircle2 size={12} /> : null}
      </span>
      <span className="min-w-0">
        <span className="mb-2 flex flex-wrap items-center gap-2">
          <span className={`rounded-md px-2 py-0.5 text-xs font-semibold uppercase ${chunkTypeStyle(chunk.type)}`}>
            {chunkTypeLabel(chunk.type)}
          </span>
          <span className="rounded-md border border-zinc-200 bg-white px-2 py-0.5 text-xs font-semibold text-zinc-700">
            {chunkLabel(chunk)}
          </span>
          <span className="text-xs text-zinc-500">第 {chunk.page} 页</span>
          <span className="max-w-[260px] truncate font-mono text-xs text-zinc-400">{chunk.id}</span>
        </span>
        <span className="block truncate text-[15px] font-semibold text-zinc-950">
          {chunk.metadata?.caption ? String(chunk.metadata.caption) : chunk.title}
        </span>
        {chunk.type === "code" ? (
          <CodeChunkPreview code={chunk.content} />
        ) : chunk.type === "table" && rows.length ? (
          <TableChunkPreview rows={rows} chunkId={chunk.id} />
        ) : chunk.type === "formula" ? (
          <FormulaChunkPreview formula={chunk.content} />
        ) : chunk.type !== "figure" ? (
          <span className="mt-2 block line-clamp-3 text-sm leading-6 text-zinc-600">
            {chunkDescription(chunk)}
          </span>
        ) : null}
        {chunk.type === "figure" && chunk.content ? (
          <span className="mt-2 block rounded-lg border border-blue-100 bg-blue-50/40 px-3 py-2 text-xs leading-5 text-blue-900">
            {chunk.content}
          </span>
        ) : null}
      </span>
      <span className="hidden min-w-0 flex-col items-end gap-2 md:flex">
        {cropUrl ? (
          <span className="relative block h-28 w-44 overflow-hidden rounded-lg border border-zinc-200 bg-white">
            <img src={cropUrl} alt={chunk.title} className="h-full w-full object-contain" />
          </span>
        ) : (
          <ChunkSidePreview chunk={chunk} rows={rows} />
        )}
      </span>
    </button>
  );
}

function ChunkSidePreview({ chunk, rows }: { chunk: EvidenceChunk; rows: string[][] }) {
  if (chunk.type === "table" && rows.length) {
    return (
      <span className="block h-28 w-44 overflow-hidden rounded-lg border border-emerald-100 bg-white shadow-sm">
        <TableChunkPreview rows={rows.slice(0, 4)} chunkId={`${chunk.id}-side`} compact />
      </span>
    );
  }
  if (chunk.type === "formula") {
    return (
      <span className="grid h-28 w-44 place-items-center overflow-hidden rounded-lg border border-amber-100 bg-amber-50/60 px-3">
        <FormulaChunkPreview formula={chunk.content} compact />
      </span>
    );
  }
  if (chunk.type === "code") {
    return (
      <span className="grid h-28 w-44 place-items-center rounded-lg border border-slate-200 bg-slate-950 px-3 text-center text-xs font-semibold text-slate-100">
        代码 chunk
      </span>
    );
  }
  return (
    <span className="grid h-28 w-44 place-items-center rounded-lg border border-dashed border-zinc-200 bg-zinc-50 text-xs font-medium text-zinc-400">
      文本 chunk
    </span>
  );
}

function TableChunkPreview({ rows, chunkId, compact = false }: { rows: string[][]; chunkId: string; compact?: boolean }) {
  const visibleRows = rows.slice(0, 12);
  const columnCount = Math.max(1, Math.min(compact ? 6 : 8, Math.max(...visibleRows.map((row) => row.length))));
  return (
    <span
      className={`block overflow-auto rounded-lg border border-emerald-100 bg-white text-xs shadow-sm ${
        compact ? "h-full max-h-28 rounded-none border-0 text-[10px]" : "mt-3 max-h-64"
      }`}
      role="table"
      aria-label="表格 chunk 预览"
    >
      {visibleRows.map((row, rowIndex) => (
        <span
          key={`${chunkId}-row-${rowIndex}`}
          className={`grid min-w-max ${
            rowIndex === 0 ? "bg-emerald-50 font-semibold text-emerald-950" : "text-zinc-700"
          }`}
          style={{ gridTemplateColumns: `repeat(${columnCount}, minmax(${compact ? 42 : 88}px, 1fr))` }}
          role="row"
        >
          {Array.from({ length: columnCount }, (_, cellIndex) => (
            <span
              key={`${chunkId}-cell-${rowIndex}-${cellIndex}`}
              className={`${compact ? "px-1.5 py-1" : "px-2.5 py-1.5"} border-r border-t border-emerald-100 last:border-r-0`}
              role={rowIndex === 0 ? "columnheader" : "cell"}
            >
              {row[cellIndex] ?? ""}
            </span>
          ))}
        </span>
      ))}
      {rows.length > visibleRows.length ? (
        <span className="block border-t border-emerald-100 bg-emerald-50/60 px-2.5 py-1.5 font-medium text-emerald-800">
          还有 {rows.length - visibleRows.length} 行
        </span>
      ) : null}
    </span>
  );
}

function CodeChunkPreview({ code }: { code: string }) {
  return (
    <pre className="mt-3 max-h-64 overflow-auto rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-xs leading-5 text-slate-100 shadow-sm">
      <code>{code}</code>
    </pre>
  );
}

function FormulaChunkPreview({ formula, compact = false }: { formula: string; compact?: boolean }) {
  const latex = extractFormulaLatex(formula);
  return (
    <span
      className={`block overflow-auto rounded-lg border border-amber-100 bg-amber-50/70 text-amber-950 shadow-sm ${
        compact ? "max-h-24 border-0 bg-transparent px-0 py-0 text-sm shadow-none" : "mt-3 max-h-40 px-4 py-3"
      }`}
    >
      <span className="mb-2 block text-xs font-semibold text-amber-700">
        {formulaTitle(formula)}
      </span>
      <span className={`block whitespace-nowrap font-serif ${compact ? "text-base" : "text-2xl"}`}>
        <MathExpression latex={latex} />
      </span>
    </span>
  );
}

function extractFormulaLatex(raw: string) {
  const fenced = raw.match(/\$\$([\s\S]*?)\$\$/);
  if (fenced) return fenced[1].trim();
  return raw
    .replace(/^公式\s*\d+\s*[:：]\s*/m, "")
    .replace(/\$\$/g, "")
    .trim();
}

function formulaTitle(raw: string) {
  const match = raw.match(/^(公式\s*\d+\s*[:：])/m);
  return match ? match[1].replace(/\s+/g, "") : "公式";
}

function MathExpression({ latex }: { latex: string }) {
  const normalized = latex.replace(/\\quad/g, " ").replace(/\\,/g, " ").replace(/\s+/g, " ").trim();
  const sumMatch = normalized.match(/\\sum_\{([^}]+)\}\^\{([^}]+)\}/);
  if (!sumMatch || sumMatch.index === undefined) {
    return <>{renderMathText(normalized, "formula")}</>;
  }
  const before = normalized.slice(0, sumMatch.index);
  const after = normalized.slice(sumMatch.index + sumMatch[0].length);
  return (
    <>
      {renderMathText(before, "formula-before")}
      <span className="mx-1 inline-flex translate-y-1 flex-col items-center align-middle leading-none">
        <span className="text-[0.55em] leading-none">{renderMathText(sumMatch[2], "sum-upper")}</span>
        <span className="text-[1.35em] leading-none">∑</span>
        <span className="text-[0.55em] leading-none">{renderMathText(sumMatch[1], "sum-lower")}</span>
      </span>
      {renderMathText(after, "formula-after")}
    </>
  );
}

function renderMathText(text: string, keyPrefix: string) {
  const nodes: ReactNode[] = [];
  const pattern = /([A-Za-z])_\{([^}]+)\}|\\([A-Za-z]+)/g;
  let cursor = 0;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(text))) {
    if (match.index > cursor) {
      nodes.push(<span key={`${keyPrefix}-text-${cursor}`}>{cleanMathPlain(text.slice(cursor, match.index))}</span>);
    }
    if (match[1]) {
      nodes.push(
        <span key={`${keyPrefix}-sub-${match.index}`} className="inline-flex items-baseline">
          <span>{match[1]}</span>
          <sub className="text-[0.58em] leading-none">{match[2]}</sub>
        </span>,
      );
    } else {
      nodes.push(<span key={`${keyPrefix}-cmd-${match.index}`}>{mathCommandSymbol(match[3])}</span>);
    }
    cursor = match.index + match[0].length;
  }
  if (cursor < text.length) {
    nodes.push(<span key={`${keyPrefix}-text-${cursor}`}>{cleanMathPlain(text.slice(cursor))}</span>);
  }
  return nodes;
}

function cleanMathPlain(text: string) {
  return text.replace(/\\/g, "");
}

function mathCommandSymbol(command: string) {
  return {
    times: "×",
    cdot: "·",
    in: "∈",
    leq: "≤",
    geq: "≥",
    approx: "≈",
  }[command] ?? command;
}

function PreviewPanelV2({
  activeFile,
  files,
  hasUploaded,
  selected,
  onActiveFile,
  onDeleteFile,
  onUpload,
}: {
  activeFile: FileAsset | null;
  files: FileAsset[];
  hasUploaded: boolean;
  selected: EvidenceChunk | null;
  onActiveFile: (fileId: string) => void;
  onDeleteFile: (fileId: string) => void | Promise<void>;
  onUpload: (files: FileList | null) => void | Promise<void>;
}) {
  const pageRefs = useRef<Record<number, HTMLDivElement | null>>({});
  const [pageSizes, setPageSizes] = useState<Record<number, { width: number; height: number }>>({});
  const pages = useMemo(() => pageNumbers(activeFile), [activeFile]);
  const selectedPage = selected?.page || 1;
  const highlight = pageHighlightBox(selected);
  const selectedPageSize = pageSizes[selectedPage];
  const pageWidth = metadataNumber(selected?.metadata, "page_width") || selectedPageSize?.width || 0;
  const pageHeight = metadataNumber(selected?.metadata, "page_height") || selectedPageSize?.height || 0;
  const highlightStyle = highlight && pageWidth > 0 && pageHeight > 0
    ? (() => {
        const left = Math.max(0, Math.min(100, (highlight[0] / pageWidth) * 100));
        const top = Math.max(0, Math.min(100, (highlight[1] / pageHeight) * 100));
        const right = Math.max(left, Math.min(100, (highlight[2] / pageWidth) * 100));
        const bottom = Math.max(top, Math.min(100, (highlight[3] / pageHeight) * 100));
        return {
          left: `${left}%`,
          top: `${top}%`,
          width: `${right - left}%`,
          height: `${bottom - top}%`,
        };
      })()
    : null;

  const pageAspectRatio = (page: number) => {
    const size = pageSizes[page];
    return size?.width && size?.height ? `${size.width} / ${size.height}` : FALLBACK_PAGE_ASPECT;
  };

  useEffect(() => {
    pageRefs.current[selectedPage]?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [selectedPage, activeFile?.file_id]);

  useEffect(() => {
    setPageSizes({});
  }, [activeFile?.file_id]);

  return (
    <motion.aside
      className="mt-0 min-w-0 lg:sticky lg:top-[68px] lg:h-[calc(100vh-74px)]"
      initial={{ opacity: 0, x: 12 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.24 }}
    >
      <div className="flex h-full flex-col overflow-hidden rounded-xl border border-zinc-200 bg-white shadow-[0_18px_55px_rgba(36,48,64,0.06)]">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-100 px-4 py-3">
          <div className="min-w-0">
            <h2 className="font-semibold text-zinc-950">预览</h2>
            <p className="mt-0.5 truncate text-sm text-zinc-500">
              {hasUploaded ? activeFile?.file_name : "上传文件后预览证据"}
            </p>
          </div>
          <UploadButton onUpload={onUpload}>添加文件</UploadButton>
        </div>

        {hasUploaded ? (
          <>
            <div className="flex gap-2 overflow-x-auto border-b border-zinc-100 px-4 py-2.5">
              {files.map((file) => (
                <div
                  key={file.file_id}
                  className={`flex shrink-0 items-center gap-2 rounded-lg border px-3 py-1.5 text-sm transition ${
                    activeFile?.file_id === file.file_id
                      ? "border-[#3178f6]/30 bg-blue-50 text-[#246be8]"
                      : "border-zinc-200 bg-white text-zinc-600 hover:bg-zinc-50"
                  }`}
                >
                  <button className="max-w-[220px] truncate text-left" onClick={() => onActiveFile(file.file_id)}>
                    {file.file_name}
                  </button>
                  <button
                    className="grid h-6 w-6 place-items-center rounded-full text-zinc-400 transition hover:bg-white hover:text-rose-500"
                    onClick={() => void onDeleteFile(file.file_id)}
                    aria-label={`删除 ${file.file_name}`}
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              ))}
            </div>

            <motion.div
              key={activeFile?.file_id ?? "file"}
              className="thin-scrollbar min-h-0 flex-1 overflow-y-auto bg-[#fbfbfa] px-3 py-3"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2 }}
            >
              <div className="mx-auto max-w-[680px] space-y-3">
                {pages.length ? pages.map((page) => {
                  const isSelectedPage = page === selectedPage;
                  return (
                    <div
                      key={page}
                      ref={(node) => {
                        pageRefs.current[page] = node;
                      }}
                      className={`mx-auto overflow-hidden rounded-lg border bg-white shadow-sm ${
                        isSelectedPage ? "border-[#3178f6]/45 ring-2 ring-[#3178f6]/15" : "border-zinc-200"
                      }`}
                      style={{ width: `${PREVIEW_PAGE_ZOOM * 100}%` }}
                    >
                      <div className="flex items-center justify-between border-b border-zinc-100 px-3 py-2 text-xs text-zinc-500">
                        <span className="min-w-0 truncate">{activeFile?.file_name}</span>
                        <span className={isSelectedPage ? "font-medium text-[#246be8]" : ""}>第 {page} 页</span>
                      </div>
                      <div className="relative bg-zinc-50" style={{ aspectRatio: pageAspectRatio(page) }}>
                        <img
                          src={pageImageUrl(activeFile?.file_id, page) ?? ""}
                          alt={`${activeFile?.file_name ?? "PDF"} 第 ${page} 页`}
                          className="absolute inset-0 h-full w-full object-contain"
                          onLoad={(event) => {
                            const width = event.currentTarget.naturalWidth;
                            const height = event.currentTarget.naturalHeight;
                            if (!width || !height) return;
                            setPageSizes((current) => {
                              const previous = current[page];
                              if (previous?.width === width && previous?.height === height) return current;
                              return { ...current, [page]: { width, height } };
                            });
                          }}
                        />
                        {isSelectedPage && highlightStyle ? (
                          <div
                            className="pointer-events-none absolute border-2 border-[#3178f6] bg-[#3178f6]/15 shadow-[0_0_0_9999px_rgba(255,255,255,0.12)]"
                            style={highlightStyle}
                          />
                        ) : null}
                      </div>
                    </div>
                  );
                }) : (
                  <div className="grid min-h-[420px] place-items-center rounded-lg border border-dashed border-zinc-200 bg-white text-sm text-zinc-500">
                    {activeFile?.status === "ready" ? "暂无渲染页面。" : "正在解析 PDF 页面..."}
                  </div>
                )}
              </div>
            </motion.div>
          </>
        ) : (
          <div className="grid flex-1 place-items-center bg-[#fbfbfa] p-8 text-center">
            <div className="max-w-sm">
              <div className="mx-auto mb-4 grid h-12 w-12 place-items-center rounded-full bg-blue-50 text-[#3178f6]">
                <PanelRightOpen size={22} />
              </div>
              <h3 className="text-lg font-semibold text-zinc-950">暂无预览</h3>
              <p className="mt-2 text-sm leading-6 text-zinc-500">
                上传一个或多个附件后，可以预览页面并在文件之间切换。
              </p>
              <div className="mt-5">
                <UploadButton onUpload={onUpload} variant="primary">
                  上传证据
                </UploadButton>
              </div>
            </div>
          </div>
        )}
      </div>
    </motion.aside>
  );
}

function PreviewPanel({
  activeFile,
  files,
  hasUploaded,
  selected,
  onActiveFile,
  onUpload,
}: {
  activeFile: FileAsset | null;
  files: FileAsset[];
  hasUploaded: boolean;
  selected: EvidenceChunk | null;
  onActiveFile: (fileId: string) => void;
  onUpload: (files: FileList | null) => void | Promise<void>;
}) {
  const selectedFileId = selected?.file_id || activeFile?.file_id;
  const selectedPage = selected?.page || 1;
  const primaryPreviewSrc = pageImageUrl(selectedFileId, selectedPage);
  const fallbackPreviewSrc = selected?.preview_url || selected?.image_url || null;
  const [previewSrc, setPreviewSrc] = useState(primaryPreviewSrc);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const highlight = pageHighlightBox(selected);
  const pageWidth = metadataNumber(selected?.metadata, "page_width") || imageSize.width;
  const pageHeight = metadataNumber(selected?.metadata, "page_height") || imageSize.height;
  const aspectRatio = pageWidth > 0 && pageHeight > 0 ? `${pageWidth} / ${pageHeight}` : FALLBACK_PAGE_ASPECT;
  const highlightStyle = highlight && pageWidth > 0 && pageHeight > 0
    ? {
        left: `${(highlight[0] / pageWidth) * 100}%`,
        top: `${(highlight[1] / pageHeight) * 100}%`,
        width: `${((highlight[2] - highlight[0]) / pageWidth) * 100}%`,
        height: `${((highlight[3] - highlight[1]) / pageHeight) * 100}%`,
      }
    : null;
  const locatorText = selected
    ? `第 ${selected.page} 页 · ${chunkTypeLabel(selected.type)} · ${selected.chunk_id}`
    : activeFile?.status === "ready"
      ? "第 1 页"
      : fileStatusLabel(activeFile?.status ?? "queued");

  useEffect(() => {
    setPreviewSrc(primaryPreviewSrc);
    setImageSize({ width: 0, height: 0 });
  }, [primaryPreviewSrc]);

  return (
    <motion.aside
      className="mt-10 min-w-0 lg:sticky lg:top-24 lg:mt-0 lg:h-[calc(100vh-140px)]"
      initial={{ opacity: 0, x: 12 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.24 }}
    >
      <div className="flex h-full flex-col overflow-hidden rounded-xl border border-zinc-200 bg-white shadow-[0_18px_55px_rgba(36,48,64,0.06)]">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-100 px-5 py-4">
          <div>
            <h2 className="font-semibold text-zinc-950">预览</h2>
            <p className="mt-0.5 text-sm text-zinc-500">
              {hasUploaded ? activeFile?.file_name : "上传文件后预览证据"}
            </p>
          </div>
          <UploadButton onUpload={onUpload}>添加文件</UploadButton>
        </div>

        {hasUploaded ? (
          <>
            <div className="flex gap-2 overflow-x-auto border-b border-zinc-100 px-5 py-3">
              {files.map((file) => (
                <button
                  key={file.file_id}
                  className={`shrink-0 rounded-lg border px-3 py-1.5 text-sm transition ${
                    activeFile?.file_id === file.file_id
                      ? "border-[#3178f6]/30 bg-blue-50 text-[#246be8]"
                      : "border-zinc-200 bg-white text-zinc-600 hover:bg-zinc-50"
                  }`}
                  onClick={() => onActiveFile(file.file_id)}
                >
                  {file.file_name}
                </button>
              ))}
            </div>
            <AnimatePresence mode="wait">
              <motion.div
                key={`${activeFile?.file_id ?? "file"}-${selected?.id ?? "empty"}`}
                className="thin-scrollbar min-h-0 flex-1 overflow-y-auto bg-[#fbfbfa] p-5"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.2 }}
              >
                <div className="mx-auto min-h-[680px] max-w-[520px] rounded-lg border border-zinc-200 bg-white p-7 shadow-sm">
                  <div className="mb-7 flex items-center justify-between border-b border-zinc-100 pb-4 text-sm text-zinc-400">
                    <span>{activeFile?.file_name}</span>
                    <span>{selected ? `第 ${selected.page} 页` : fileStatusLabel(activeFile?.status)}</span>
                  </div>
                  {previewSrc ? (
                    <div
                      className="relative overflow-hidden rounded-lg border border-zinc-100 bg-zinc-50"
                      style={{ aspectRatio }}
                    >
                      <img
                        src={previewSrc}
                        alt={activeFile?.file_name ?? "PDF 页面预览"}
                        className="absolute inset-0 h-full w-full object-contain"
                        onLoad={(event) => {
                          setImageSize({
                            width: event.currentTarget.naturalWidth,
                            height: event.currentTarget.naturalHeight,
                          });
                        }}
                        onError={() => {
                          if (fallbackPreviewSrc && previewSrc !== fallbackPreviewSrc) {
                            setPreviewSrc(fallbackPreviewSrc);
                          } else {
                            setPreviewSrc(null);
                          }
                        }}
                      />
                      {highlightStyle ? (
                        <div
                          className="pointer-events-none absolute border-2 border-[#3178f6] bg-[#3178f6]/14 shadow-[0_0_0_9999px_rgba(255,255,255,0.14)]"
                          style={highlightStyle}
                        />
                      ) : null}
                    </div>
                  ) : (
                    <>
                      <div className="space-y-3">
                        <div className="h-3 w-56 rounded bg-zinc-200" />
                        {Array.from({ length: 7 }).map((_, index) => (
                          <div
                            key={index}
                            className="h-2 rounded bg-zinc-100"
                            style={{ width: `${96 - index * 6}%` }}
                          />
                        ))}
                      </div>
                      <div className="my-8 rounded-lg bg-[#fff7e6] p-4">
                        <div className="mb-3 flex h-40 items-end gap-2">
                          {[42, 56, 64, 82, 96, 118, 132].map((height, index) => (
                            <motion.span
                              key={index}
                              className="flex-1 rounded-t bg-[#61bfc8]"
                              initial={{ height: 12 }}
                              animate={{ height }}
                              transition={{ delay: index * 0.035, duration: 0.22 }}
                            />
                          ))}
                        </div>
                      </div>
                    </>
                  )}
                  <div className="mt-5 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-zinc-100 bg-zinc-50 px-4 py-3 text-sm text-zinc-600">
                    <span className="min-w-0 truncate">{locatorText}</span>
                    <span className="shrink-0 font-medium text-[#246be8]">
                      {highlightStyle ? "已在 PDF 中高亮" : "PDF 预览"}
                    </span>
                  </div>
                </div>
              </motion.div>
            </AnimatePresence>
          </>
        ) : (
          <div className="grid flex-1 place-items-center bg-[#fbfbfa] p-8 text-center">
            <div className="max-w-sm">
              <div className="mx-auto mb-4 grid h-12 w-12 place-items-center rounded-full bg-blue-50 text-[#3178f6]">
                <PanelRightOpen size={22} />
              </div>
              <h3 className="text-lg font-semibold text-zinc-950">暂无预览</h3>
              <p className="mt-2 text-sm leading-6 text-zinc-500">
                上传一个或多个附件后，可以预览页面并在文件之间切换。
              </p>
              <div className="mt-5">
                <UploadButton onUpload={onUpload} variant="primary">
                  上传证据
                </UploadButton>
              </div>
            </div>
          </div>
        )}
      </div>
    </motion.aside>
  );
}

function ResultPage({
  query,
  model,
  selected,
  selectedChunks,
  result,
  chatTurns,
  pendingQuestion,
  activeChatId,
  onSelectChat,
  isThinking,
  onBack,
  onReturnToCover,
}: {
  query: string;
  model: string;
  selected: EvidenceChunk | null;
  selectedChunks: EvidenceChunk[];
  result: AnswerResult | null;
  chatTurns: ChatTurn[];
  pendingQuestion: string;
  activeChatId: string;
  onSelectChat: (id: string) => void;
  isThinking: boolean;
  onBack: () => void;
  onReturnToCover: () => void;
}) {
  const fallbackTurn = result
    ? [{
        id: result.answer_id,
        question: query || result.question,
        model: result.model_label || model,
        selectedChunks,
        result,
        createdAt: "",
      }]
    : [];
  const turnsToRender = chatTurns.length ? chatTurns : fallbackTurn;
  const pendingReferencedCount = selectedChunks.length || (selected ? 1 : 0);
  const [activeSource, setActiveSource] = useState<SourceReference | null>(null);
  const chatBottomRef = useRef<HTMLDivElement | null>(null);
  const scrollKey = turnsToRender
    .map((turn) => `${turn.id}:${turn.result.answer?.length ?? 0}`)
    .join("|");

  useEffect(() => {
    setActiveSource(null);
  }, [result?.answer_id]);

  useEffect(() => {
    const frameId = window.requestAnimationFrame(() => {
      chatBottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    });
    return () => window.cancelAnimationFrame(frameId);
  }, [scrollKey, isThinking, pendingQuestion]);

  return (
    <motion.section
      className="flex h-full min-w-0 flex-col"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.22 }}
    >
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <button
          className="inline-flex items-center gap-2 text-lg font-semibold text-zinc-950"
          onClick={onBack}
        >
          <ArrowLeft size={22} />
          <span>返回知识库</span>
        </button>
        <button
          className="inline-flex items-center gap-2 rounded-xl border border-zinc-200 bg-white px-3 py-2 text-sm font-medium text-zinc-700 transition hover:bg-zinc-50"
          onClick={onReturnToCover}
        >
          <RotateCcw size={16} />
          重置演示
        </button>
      </div>

      <article className="thin-scrollbar min-h-0 flex-1 overflow-y-auto rounded-xl border border-zinc-200 bg-[#f7f9fc] px-5 py-5 shadow-[0_18px_60px_rgba(24,39,75,0.08)]">
        <div className="mx-auto flex max-w-[920px] flex-col gap-6">
          {turnsToRender.length ? turnsToRender.map((turn) => {
            const referencedChunks = turn.result.evidences.length
              ? turn.result.evidences
              : turn.selectedChunks.length
                ? turn.selectedChunks
                : [];
            const answerDisplay = parseAnswerDisplay(turn.result.answer || "暂无回答。", referencedChunks);
            const isActive = activeChatId ? turn.id === activeChatId : turn.id === turnsToRender[turnsToRender.length - 1]?.id;
            return (
              <div key={turn.id} className={isActive ? "" : "opacity-80"}>
                <button
                  type="button"
                  className="mb-3 text-xs font-medium text-zinc-400 hover:text-[#246be8]"
                  onClick={() => onSelectChat(turn.id)}
                >
                  {turn.createdAt ? new Date(turn.createdAt).toLocaleString("zh-CN") : "当前回答"}
                </button>
                <div className="flex justify-end">
                  <div className="max-w-[86%] rounded-[22px] rounded-tr-md bg-[#246be8] px-5 py-4 text-white shadow-[0_12px_32px_rgba(36,107,232,0.22)]">
                    <div className="mb-1 text-xs font-semibold uppercase tracking-[0.16em] text-blue-100">你</div>
                    <p className="whitespace-pre-wrap break-words text-[17px] font-medium leading-7">{turn.question || "查询结果"}</p>
                  </div>
                </div>

                <div className="mt-4 flex items-start gap-3">
                  <span className="relative mt-1 grid h-10 w-10 shrink-0 place-items-center rounded-full bg-white text-[#246be8] shadow-sm ring-1 ring-blue-100">
                    <Sparkles size={19} />
                  </span>
                  <div className="min-w-0 flex-1 rounded-[24px] rounded-tl-md border border-zinc-200 bg-white px-6 py-5 shadow-sm">
                    <div className="mb-5 flex flex-wrap items-center gap-2">
                      <span className="inline-flex items-center gap-1.5 rounded-full bg-blue-50 px-3 py-1.5 text-sm font-semibold text-[#246be8]">
                        <Sparkles size={15} />
                        {turn.model || model}
                      </span>
                      <span className="inline-flex items-center gap-1.5 rounded-full bg-zinc-100 px-3 py-1.5 text-sm font-medium text-zinc-600">
                        <Search size={15} />
                        引用 {referencedChunks.length} 个 chunk
                      </span>
                    </div>

                    <div className="border-b border-zinc-100 pb-6">
                      <div className="mb-4 flex items-center gap-2">
                        <span className="grid h-9 w-9 place-items-center rounded-full bg-[#eef4ff] text-[#246be8]">
                          <Sparkles size={18} />
                        </span>
                        <div>
                          <h2 className="text-base font-semibold text-zinc-950">回答</h2>
                          <p className="text-xs font-medium uppercase tracking-[0.18em] text-zinc-400">生成结果</p>
                        </div>
                      </div>
                      <div className="space-y-4 break-words text-[17px] leading-8 text-zinc-800">
                        {answerParagraphs(answerDisplay.body).map((paragraph, index) => (
                          <p key={`${turn.id}-${paragraph}-${index}`} className="whitespace-pre-wrap">
                            {renderInlineAnswer(paragraph)}
                          </p>
                        ))}
                      </div>
                    </div>

                    <EvidenceReferencePanel
                      chunks={referencedChunks}
                      references={answerDisplay.references}
                      onOpen={setActiveSource}
                    />
                  </div>
                </div>
              </div>
            );
          }) : null}
          {isThinking ? (
            <div className="flex items-start gap-3">
              <span className="relative mt-1 grid h-10 w-10 shrink-0 place-items-center rounded-full bg-white text-[#246be8] shadow-sm ring-1 ring-blue-100">
                <Sparkles size={19} />
              </span>
              <ThinkingBubble model={model} referencedCount={pendingReferencedCount} />
            </div>
          ) : null}
          {!turnsToRender.length && !isThinking ? (
            <div className="rounded-xl border border-dashed border-zinc-200 bg-white px-6 py-10 text-center text-sm text-zinc-500">
              暂无聊天记录，上传证据后可以直接提问。
            </div>
          ) : null}
          <div ref={chatBottomRef} aria-hidden="true" />
        </div>
      </article>

      <AnimatePresence>
        {activeSource ? (
          <SourceChunkDialog reference={activeSource} onClose={() => setActiveSource(null)} />
        ) : null}
      </AnimatePresence>
    </motion.section>
  );
}

function EvidenceReferencePanel({
  chunks,
  references,
  onOpen,
}: {
  chunks: EvidenceChunk[];
  references: SourceReference[];
  onOpen: (reference: SourceReference) => void;
}) {
  const [isEvidenceCollapsed, setIsEvidenceCollapsed] = useState(true);

  return (
    <div className="mt-6">
      {references.length ? (
        <div className="border-b border-zinc-100 pb-5">
          <div className="mb-3 flex items-center justify-between gap-3">
            <h2 className="text-base font-semibold text-zinc-950">来源</h2>
            <span className="text-sm font-medium text-zinc-400">引用 {references.length} 个 chunk</span>
          </div>
          <div className="flex flex-wrap gap-2.5">
            {references.map((source) => (
              <button
                key={source.id}
                type="button"
                title={source.raw}
                className="group inline-flex max-w-full items-center gap-2 rounded-full border border-blue-100 bg-blue-50/70 px-3.5 py-2 text-sm font-medium text-[#246be8] shadow-sm transition hover:border-[#246be8]/30 hover:bg-white hover:shadow-md focus:outline-none focus:ring-2 focus:ring-[#246be8]/20"
                onClick={() => onOpen(source)}
              >
                <span className="shrink-0 rounded-full bg-white px-2 py-0.5 text-xs font-semibold text-[#246be8]">
                  第 {source.page} 页
                </span>
                <span className="min-w-0 truncate text-zinc-700 group-hover:text-zinc-950">
                  chunk {source.chunkId}
                </span>
              </button>
            ))}
          </div>
        </div>
      ) : null}

      <div className="pt-5">
        <button
          type="button"
          className="mb-3 flex w-full items-center justify-between gap-3 rounded-lg px-1 py-1 text-left transition hover:bg-zinc-50 focus:outline-none focus:ring-2 focus:ring-[#246be8]/15"
          aria-expanded={!isEvidenceCollapsed}
          onClick={() => setIsEvidenceCollapsed((collapsed) => !collapsed)}
        >
          <span className="inline-flex min-w-0 items-center gap-2">
            <ChevronDown
              size={17}
              className={`shrink-0 text-zinc-500 transition-transform ${isEvidenceCollapsed ? "-rotate-90" : "rotate-0"}`}
            />
            <span className="text-base font-semibold text-zinc-950">引用证据</span>
          </span>
          <span className="shrink-0 text-sm font-medium text-zinc-400">{chunks.length} 项</span>
        </button>
        <div className={isEvidenceCollapsed ? "hidden" : "space-y-3"}>
          {chunks.map((chunk) => (
            <ChunkEvidenceCard
              key={chunk.id}
              chunk={chunk}
              onOpen={() =>
                onOpen({
                  id: `${chunk.page}:${chunk.chunk_id}`,
                  page: chunk.page,
                  chunkId: chunk.chunk_id,
                  raw: `[page=${chunk.page}, chunk=${chunk.chunk_id}]`,
                  chunk,
                })
              }
            />
          ))}
        </div>
      </div>
    </div>
  );
}

function ChunkEvidenceCard({ chunk, onOpen }: { chunk: EvidenceChunk; onOpen: () => void }) {
  const rows = structuredRows(chunk);
  const cropUrl = chunk.crop_url || chunk.preview_url || (chunk.type === "figure" ? chunk.image_url : null);
  return (
    <button
      type="button"
      className="grid w-full gap-3 rounded-lg border border-zinc-100 bg-zinc-50 px-4 py-3 text-left transition hover:border-blue-100 hover:bg-[#f7fbff] md:grid-cols-[minmax(0,1fr)_160px]"
      onClick={onOpen}
    >
      <span className="min-w-0">
        <span className="mb-2 flex flex-wrap items-center gap-2">
          <span className={`rounded-md px-2 py-0.5 text-xs font-semibold uppercase ${chunkTypeStyle(chunk.type)}`}>
            {chunkTypeLabel(chunk.type)}
          </span>
          <span className="rounded-md border border-zinc-200 bg-white px-2 py-0.5 text-xs font-semibold text-zinc-700">
            第 {chunk.page} 页
          </span>
        </span>
        <span className="block break-words font-semibold text-zinc-900">{chunk.title}</span>
        {chunk.type === "code" ? (
          <CodeChunkPreview code={chunk.content} />
        ) : chunk.type === "table" && rows.length ? (
          <TableChunkPreview rows={rows} chunkId={chunk.id} />
        ) : chunk.type === "formula" ? (
          <FormulaChunkPreview formula={chunk.content} />
        ) : (
          <span className="mt-3 block whitespace-pre-wrap break-words text-sm leading-6 text-zinc-600">
            {chunk.content}
          </span>
        )}
      </span>
      {cropUrl ? (
        <span className="relative hidden h-28 w-40 overflow-hidden rounded-lg border border-zinc-200 bg-white md:block">
          <img src={cropUrl} alt={chunk.title} className="h-full w-full object-contain" />
        </span>
      ) : null}
    </button>
  );
}

function ThinkingBubble({
  model,
  referencedCount,
}: {
  model: string;
  referencedCount: number;
}) {
  return (
    <motion.div
      className="max-w-[520px] rounded-[24px] rounded-tl-md border border-blue-100 bg-white px-5 py-4 shadow-sm"
      initial={{ opacity: 0, y: 8, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: 6, scale: 0.98 }}
      transition={{ duration: 0.18 }}
    >
      <div className="mb-3 flex items-center gap-2">
        <motion.span
          className="grid h-8 w-8 place-items-center rounded-full bg-[#eef4ff] text-[#246be8]"
          animate={{ rotate: [0, 14, -10, 0], scale: [1, 1.08, 1] }}
          transition={{ duration: 1.25, repeat: Infinity, ease: "easeInOut" }}
        >
          <Sparkles size={17} />
        </motion.span>
        <div>
          <div className="text-sm font-semibold text-zinc-950">{model}</div>
          <div className="text-xs font-medium uppercase tracking-[0.16em] text-zinc-400">思考中</div>
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-2 text-[15px] font-medium text-zinc-600">
        <span>正在读取 {referencedCount} 个已选 chunk</span>
        <span className="flex items-center gap-1">
          {[0, 1, 2].map((item) => (
            <motion.span
              key={item}
              className="h-1.5 w-1.5 rounded-full bg-[#246be8]"
              animate={{ y: [0, -4, 0], opacity: [0.45, 1, 0.45] }}
              transition={{ duration: 0.72, repeat: Infinity, delay: item * 0.12 }}
            />
          ))}
        </span>
      </div>
    </motion.div>
  );
}

function SourceChunkDialog({
  reference,
  onClose,
}: {
  reference: SourceReference;
  onClose: () => void;
}) {
  const chunk = reference.chunk;
  const rows = chunk ? structuredRows(chunk) : [];
  const cropUrl = chunk?.crop_url || chunk?.preview_url || (chunk?.type === "figure" ? chunk?.image_url : null);

  return (
    <motion.div
      className="fixed inset-0 z-50 grid place-items-center bg-zinc-950/28 px-4 py-6 backdrop-blur-sm"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.16 }}
      onClick={onClose}
    >
      <motion.div
        className="flex max-h-[82vh] w-full max-w-[720px] flex-col overflow-hidden rounded-xl border border-white/70 bg-white shadow-[0_24px_80px_rgba(24,39,75,0.24)]"
        initial={{ opacity: 0, y: 18, scale: 0.96 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: 12, scale: 0.98 }}
        transition={{ duration: 0.18, ease: [0.25, 0.1, 0.25, 1] }}
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="chunk 详情"
      >
        <div className="flex items-start justify-between gap-4 border-b border-zinc-100 px-6 py-5">
          <div className="min-w-0">
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <span className="rounded-full bg-[#eef4ff] px-3 py-1 text-sm font-semibold text-[#246be8]">
                第 {reference.page} 页
              </span>
              {chunk ? (
                <span className="rounded-full bg-zinc-100 px-3 py-1 text-sm font-semibold text-zinc-600">
                  {chunkTypeLabel(chunk.source_type)}
                </span>
              ) : null}
            </div>
            <h3 className="truncate text-xl font-semibold text-zinc-950">
              {chunk?.title || "引用 chunk"}
            </h3>
            <p className="mt-1 truncate font-mono text-xs text-zinc-400">{reference.chunkId}</p>
          </div>
          <button
            type="button"
            className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-zinc-100 text-zinc-500 transition hover:bg-zinc-900 hover:text-white"
            onClick={onClose}
            aria-label="关闭 chunk 详情"
          >
            <X size={18} />
          </button>
        </div>

        <div className="thin-scrollbar overflow-y-auto px-6 py-5">
          {chunk ? (
            <>
              <div className="mb-4 flex flex-wrap gap-2 text-sm font-medium text-zinc-500">
                <span className="rounded-full border border-zinc-200 px-3 py-1">
                  文件 {chunk.file_name || chunk.file_id || "当前上传"}
                </span>
              </div>
              {cropUrl ? (
                <div className="mb-4 overflow-hidden rounded-lg border border-zinc-200 bg-white">
                  <img src={cropUrl} alt={chunk.title} className="max-h-[360px] w-full object-contain" />
                </div>
              ) : null}
              <div className="rounded-lg border border-zinc-100 bg-zinc-50 px-4 py-4">
                {chunk.type === "code" ? (
                  <CodeChunkPreview code={chunk.content} />
                ) : chunk.type === "table" && rows.length ? (
                  <TableChunkPreview rows={rows} chunkId={chunk.id} />
                ) : chunk.type === "formula" ? (
                  <FormulaChunkPreview formula={chunk.content} />
                ) : (
                  <p className="whitespace-pre-wrap break-words text-[15px] leading-7 text-zinc-700">{chunk.content}</p>
                )}
              </div>
            </>
          ) : (
            <div className="rounded-lg border border-amber-100 bg-amber-50 px-4 py-4 text-sm leading-6 text-amber-800">
              模型引用了这个来源，但当前结果没有返回对应的 chunk 正文。
            </div>
          )}
        </div>
      </motion.div>
    </motion.div>
  );
}

function ProjectLogoMark({ compact = false }: { compact?: boolean }) {
  return (
    <div
      className={`relative grid place-items-center rounded-full border border-white/80 bg-white/72 shadow-inner backdrop-blur-xl ${
        compact ? "h-11 w-11" : "h-12 w-12"
      }`}
    >
      <div className="absolute left-2 top-2 h-2 w-2 rounded-full bg-[#61bfc8]" />
      <div className="absolute bottom-2 right-2 h-2 w-2 rounded-full bg-[#f4c86a]" />
      <div className="relative flex items-end gap-[2px]">
        <span className="h-5 w-[5px] rounded-full bg-[#3178f6]" />
        <span className="h-7 w-[5px] rounded-full bg-[#7c8cf7]" />
        <span className="h-4 w-[5px] rounded-full bg-[#61bfc8]" />
      </div>
    </div>
  );
}

function Composer({
  isOpen,
  hasUploaded,
  query,
  modelId,
  models,
  isThinking,
  onOpen,
  onModelChange,
  onQueryChange,
  onSubmit,
}: {
  isOpen: boolean;
  hasUploaded: boolean;
  query: string;
  modelId: string;
  models: ModelOption[];
  isThinking: boolean;
  onOpen: () => void;
  onModelChange: (model: string) => void;
  onQueryChange: (query: string) => void;
  onSubmit: () => void;
}) {
  const iosSpring = {
    type: "spring" as const,
    stiffness: 520,
    damping: 44,
    mass: 0.86,
  };
  const orbSize = 64;
  const openHeight = 130;
  const margin = 20;
  const dragGraceTimer = useRef<number | null>(null);
  const wasDraggingRef = useRef(false);
  const lastDragAtRef = useRef(0);
  const dragStateRef = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    startLeft: number;
    startTop: number;
    moved: boolean;
  } | null>(null);
  const [viewport, setViewport] = useState(() => ({
    width: typeof window === "undefined" ? 1280 : window.innerWidth,
    height: typeof window === "undefined" ? 760 : window.innerHeight,
  }));
  const [orbPosition, setOrbPosition] = useState<{ left: number; top: number } | null>(null);
  const [isModelMenuOpen, setIsModelMenuOpen] = useState(false);
  const activeModel = models.find((item) => item.id === modelId) ?? models[0];

  const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);
  const clampOrbPosition = (position: { left: number; top: number }, width = viewport.width, height = viewport.height) => ({
    left: clamp(position.left, 16, Math.max(16, width - orbSize - 16)),
    top: clamp(position.top, 16, Math.max(16, height - orbSize - 16)),
  });

  useEffect(() => {
    const updateViewport = () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      setViewport({ width, height });
      setOrbPosition((current) => (current ? clampOrbPosition(current, width, height) : current));
    };

    updateViewport();
    window.addEventListener("resize", updateViewport);
    return () => {
      window.removeEventListener("resize", updateViewport);
      if (dragGraceTimer.current) window.clearTimeout(dragGraceTimer.current);
    };
  }, []);

  useEffect(() => {
    if (!isOpen) setIsModelMenuOpen(false);
  }, [isOpen]);

  const openWidth = Math.min(1110, Math.max(320, viewport.width - margin * 2));
  const openLeft = (viewport.width - openWidth) / 2;
  const openTop = Math.max(margin, viewport.height - 24 - openHeight);
  const defaultOrbPosition = {
    left: (viewport.width - orbSize) / 2,
    top: viewport.height - 28 - orbSize,
  };
  const closedPosition = clampOrbPosition(orbPosition ?? defaultOrbPosition);
  const beginOrbDrag = (event: React.PointerEvent<HTMLDivElement>) => {
    if (isOpen) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    if (dragGraceTimer.current) window.clearTimeout(dragGraceTimer.current);
    dragStateRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      startLeft: closedPosition.left,
      startTop: closedPosition.top,
      moved: false,
    };
  };
  const moveOrb = (event: React.PointerEvent<HTMLDivElement>) => {
    const dragState = dragStateRef.current;
    if (!dragState || dragState.pointerId !== event.pointerId) return;
    const deltaX = event.clientX - dragState.startX;
    const deltaY = event.clientY - dragState.startY;

    if (Math.abs(deltaX) > 3 || Math.abs(deltaY) > 3) {
      dragState.moved = true;
      wasDraggingRef.current = true;
      lastDragAtRef.current = performance.now();
    }

    setOrbPosition(
      clampOrbPosition({
        left: dragState.startLeft + deltaX,
        top: dragState.startTop + deltaY,
      }),
    );
  };
  const endOrbDrag = (event: React.PointerEvent<HTMLDivElement>) => {
    const dragState = dragStateRef.current;
    if (!dragState || dragState.pointerId !== event.pointerId) return;
    event.currentTarget.releasePointerCapture(event.pointerId);
    dragStateRef.current = null;

    if (dragState.moved) {
      dragGraceTimer.current = window.setTimeout(() => {
        wasDraggingRef.current = false;
      }, 140);
    }
  };
  const beginOrbMouseDrag = (event: React.MouseEvent<HTMLDivElement>) => {
    event.stopPropagation();
    if (isOpen || event.button !== 0) return;

    if (dragGraceTimer.current) window.clearTimeout(dragGraceTimer.current);
    const startX = event.clientX;
    const startY = event.clientY;
    const startLeft = closedPosition.left;
    const startTop = closedPosition.top;
    let moved = false;

    const handleMove = (moveEvent: MouseEvent) => {
      const deltaX = moveEvent.clientX - startX;
      const deltaY = moveEvent.clientY - startY;

      if (Math.abs(deltaX) > 3 || Math.abs(deltaY) > 3) {
        moved = true;
        wasDraggingRef.current = true;
        lastDragAtRef.current = performance.now();
      }

      setOrbPosition(
        clampOrbPosition({
          left: startLeft + deltaX,
          top: startTop + deltaY,
        }),
      );
    };
    const handleUp = () => {
      window.removeEventListener("mousemove", handleMove);
      if (moved) {
        dragGraceTimer.current = window.setTimeout(() => {
          wasDraggingRef.current = false;
        }, 140);
      }
    };

    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleUp, { once: true });
  };

  return (
    <motion.div
      className={`fixed z-40 border bg-white/84 shadow-[0_20px_70px_rgba(59,91,180,0.18)] backdrop-blur-2xl ${
        isOpen ? "" : "cursor-grab active:cursor-grabbing"
      } ${isOpen ? "overflow-visible" : "overflow-hidden"}`}
      style={{ maxWidth: "calc(100vw - 40px)", willChange: "transform,width,height,border-radius" }}
      animate={{
        width: isOpen ? openWidth : orbSize,
        height: isOpen ? openHeight : orbSize,
        borderRadius: isOpen ? 12 : 999,
        left: isOpen ? openLeft : closedPosition.left,
        top: isOpen ? openTop : closedPosition.top,
        x: 0,
        y: 0,
        borderColor: isOpen ? "rgba(228,228,231,0.9)" : "rgba(255,255,255,0.78)",
        boxShadow: isOpen
          ? "0 20px 70px rgba(59,91,180,0.18)"
          : "0 18px 56px rgba(59,91,180,0.22)",
        backgroundColor: isOpen ? "rgba(255,255,255,0.84)" : "rgba(255,255,255,0.58)",
      }}
      transition={iosSpring}
      onPointerDown={beginOrbDrag}
      onPointerMove={moveOrb}
      onPointerUp={endOrbDrag}
      onPointerCancel={endOrbDrag}
      onMouseDown={beginOrbMouseDrag}
      onClick={() => {
        const recentlyDragged = performance.now() - lastDragAtRef.current < 180;
        if (!isOpen && !recentlyDragged) {
          onOpen();
        }
        wasDraggingRef.current = false;
      }}
      role={isOpen ? "dialog" : "button"}
      aria-label={isOpen ? "问题输入框" : "打开问题输入框"}
    >
      <motion.div
        className="absolute inset-0 flex flex-col"
        animate={{ opacity: isOpen ? 1 : 0, scale: isOpen ? 1 : 0.92, filter: isOpen ? "blur(0px)" : "blur(2px)" }}
        transition={{ duration: isOpen ? 0.18 : 0.1, ease: [0.25, 0.1, 0.25, 1] }}
        style={{ pointerEvents: isOpen ? "auto" : "none" }}
      >
        <div className="flex min-h-[72px] items-center gap-3 px-5">
          <textarea
            value={query}
            disabled={!hasUploaded}
            onChange={(event) => onQueryChange(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                onSubmit();
              }
            }}
            rows={2}
            className="thin-scrollbar max-h-[110px] min-w-0 flex-1 resize-none border-0 bg-transparent py-3 text-[18px] leading-7 text-zinc-800 outline-none placeholder:text-zinc-400 disabled:cursor-not-allowed disabled:text-zinc-400"
            placeholder={hasUploaded ? "输入问题" : "请先上传证据再提问"}
          />
        </div>
        <div className="flex h-[58px] items-center justify-between border-t border-zinc-200 px-5">
          <div className="relative">
            <button
              type="button"
              className="group inline-flex h-10 items-center gap-2 rounded-full border border-zinc-200/80 bg-white/70 px-3 text-sm font-medium text-zinc-600 shadow-sm backdrop-blur transition hover:border-blue-200 hover:bg-blue-50/40"
              onClick={(event) => {
                event.stopPropagation();
                setIsModelMenuOpen((current) => !current);
              }}
            >
              <Zap size={18} />
              <span className="text-zinc-500">模型</span>
              <span className="inline-flex items-center gap-1">
                <span className="text-sm font-semibold text-zinc-800 transition group-hover:text-[#246be8]">
                  {modelOptionLabel(activeModel)}
                </span>
                <ChevronDown
                  size={15}
                  className={`text-zinc-400 transition group-hover:text-[#246be8] ${
                    isModelMenuOpen ? "rotate-180" : ""
                  }`}
                />
              </span>
            </button>

            <AnimatePresence>
              {isModelMenuOpen ? (
                <motion.div
                  className="absolute bottom-12 left-0 w-[210px] overflow-hidden rounded-xl border border-zinc-200 bg-white p-1.5 shadow-[0_18px_42px_rgba(36,48,64,0.18)] ring-1 ring-white"
                  initial={{ opacity: 0, y: 8, scale: 0.96 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 8, scale: 0.96 }}
                  transition={{ duration: 0.16, ease: [0.25, 0.1, 0.25, 1] }}
                  onClick={(event) => event.stopPropagation()}
                >
                  {models.map((item) => {
                    const active = item.id === modelId;
                    return (
                      <button
                        key={item.id}
                        type="button"
                        className={`flex h-10 w-full items-center justify-between rounded-lg px-3 text-left text-sm font-medium transition ${
                          active
                            ? "bg-[#eef4ff] text-[#246be8]"
                            : "text-zinc-700 hover:bg-[#f7f9fc] hover:text-zinc-950"
                        }`}
                        onClick={() => {
                          onModelChange(item.id);
                          setIsModelMenuOpen(false);
                        }}
                      >
                        <span>{modelOptionLabel(item)}</span>
                        {active ? <CheckCircle2 size={16} /> : null}
                      </button>
                    );
                  })}
                </motion.div>
              ) : null}
            </AnimatePresence>
          </div>
          <button
            className="grid h-10 w-10 place-items-center rounded-full bg-zinc-100 text-zinc-400 transition hover:bg-[#3178f6] hover:text-white disabled:cursor-not-allowed disabled:opacity-50"
            disabled={!hasUploaded || isThinking}
            onClick={onSubmit}
          >
            {isThinking ? (
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-zinc-300 border-t-[#3178f6]" />
            ) : (
              <ArrowRight size={20} />
            )}
          </button>
        </div>
      </motion.div>

      <motion.div
        className="absolute inset-0 grid place-items-center"
        animate={{ opacity: isOpen ? 0 : 1, scale: isOpen ? 0.76 : 1 }}
        transition={{ duration: isOpen ? 0.1 : 0.18, ease: [0.25, 0.1, 0.25, 1] }}
        style={{ pointerEvents: isOpen ? "none" : "auto" }}
      >
        {!isOpen ? (
          <motion.span
            className="absolute inset-0 rounded-full border border-[#3178f6]/18 bg-[#dbeafe]/28"
            animate={{ scale: [1, 1.18, 1], opacity: [0.3, 0, 0.3] }}
            transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }}
          />
        ) : null}
        <ProjectLogoMark compact />
      </motion.div>
    </motion.div>
  );
}

export default App;
