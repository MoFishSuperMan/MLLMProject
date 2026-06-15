import { AnimatePresence, motion } from "framer-motion";
import {
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  ChevronDown,
  FileText,
  Layers3,
  Moon,
  PanelRightOpen,
  RotateCcw,
  Search,
  Share2,
  Sparkles,
  Upload,
  Zap,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

type View = "knowledge" | "result";
type EvidenceType = "text" | "figure" | "table" | "visual";
type FileStatus = "uploaded" | "queued" | "parsing" | "ready" | "failed";

type EvidenceChunk = {
  id: string;
  chunk_id: string;
  evidence_id: string;
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

const API_BASE = "/api/v1";

async function apiJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    let message = `Request failed with ${response.status}`;
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
    file_id: String(item.file_id ?? ""),
    file_name: String(item.file_name ?? ""),
    type: sourceType,
    source_type: sourceType,
    page: Number(item.page ?? 1),
    score: Number(item.score ?? 0),
    title: String(item.title ?? "Evidence chunk"),
    content: String(item.content ?? ""),
    enabled: Boolean(item.enabled ?? true),
    bbox: item.bbox ?? null,
    image_url: item.image_url ?? null,
    preview_url: item.preview_url ?? null,
    metadata: item.metadata ?? {},
  };
}

function asEvidenceType(value: unknown): EvidenceType {
  const normalized = String(value ?? "text").toLowerCase();
  if (normalized === "figure" || normalized === "table" || normalized === "visual") return normalized;
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
  return error instanceof Error ? error.message : "Unexpected API error.";
}

function App() {
  const [view, setView] = useState<View>("knowledge");
  const [query, setQuery] = useState("");
  const [files, setFiles] = useState<FileAsset[]>([]);
  const [activeFileId, setActiveFileId] = useState("");
  const [chunks, setChunks] = useState<EvidenceChunk[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [focusedId, setFocusedId] = useState("");
  const [models, setModels] = useState<ModelOption[]>([]);
  const [modelId, setModelId] = useState("qwen3_vl_local");
  const [answerResult, setAnswerResult] = useState<AnswerResult | null>(null);
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
  const modelLabel = models.find((item) => item.id === modelId)?.label ?? modelId;
  const canAsk = readyFiles.length > 0;

  useEffect(() => {
    void loadModels();
    void refreshFiles();
  }, []);

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
      setStatusMessage(job.message || "Parsing evidence...");
      await refreshFiles();
      if (job.status === "ready") {
        setStatusMessage("Evidence ready.");
        if (fileId === activeFileId || !activeFileId) await loadChunks(fileId);
        return;
      }
      if (job.status === "failed") {
        throw new Error(job.error?.message || "Parse failed.");
      }
      await delay(1200);
    }
  };

  const addFiles = async (selectedFiles: FileList | null) => {
    if (!selectedFiles?.length || isUploading) return;
    setIsUploading(true);
    setStatusMessage("Uploading evidence...");
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

  const loadSample = async () => {
    if (isUploading) return;
    setIsUploading(true);
    setStatusMessage("Loading sample evidence...");
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
    if (!query.trim() || !canAsk) return;
    setIsThinking(true);
    setErrorMessage("");
    try {
      const result = await apiJson<AnswerResult>("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: query,
          file_ids: readyFiles.map((file) => file.file_id),
          selected_chunk_ids: selectedIds,
          model: modelId,
          mode: "auto",
          top_k: 5,
        }),
      });
      setAnswerResult({
        ...result,
        evidences: result.evidences.map(normalizeChunk),
      });
      setView("result");
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
        className="mx-auto grid min-h-[calc(100vh-68px)] max-w-[2020px] grid-cols-1 border-x border-dashed border-zinc-200 px-8 pb-8 pt-10 lg:grid-cols-[minmax(520px,820px)_minmax(420px,1fr)] lg:gap-10"
        onMouseDown={() => setIsComposerOpen(false)}
      >
        <section className="min-w-0 lg:h-[calc(100vh-140px)]">
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
                onSelect={toggleChunkSelection}
              />
            ) : (
              <ResultPage
                key="result"
                query={query}
                model={modelLabel}
                selected={selected}
                selectedChunks={selectedChunks}
                result={answerResult}
                onBack={() => setView("knowledge")}
                onReturnToCover={returnToCover}
              />
            )}
          </AnimatePresence>
        </section>

        <PreviewPanel
          activeFile={activeFile}
          files={files}
          hasUploaded={hasUploaded}
          selected={selected}
          onActiveFile={setActiveFileId}
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
    <header className="flex h-[68px] items-center justify-between border-b border-zinc-100 bg-white/92 px-9 backdrop-blur">
      <div className="flex min-w-0 items-center gap-4">
        {hasUploaded ? (
          <button
            className="inline-flex items-center gap-2 rounded-lg px-2.5 py-2 text-sm font-medium text-zinc-700 transition hover:bg-zinc-100"
            onClick={onBack}
          >
            <ArrowLeft size={17} />
            Back
          </button>
        ) : null}
        <div className="text-[26px] font-semibold tracking-tight text-zinc-900">MLLM Demo</div>
        {hasUploaded ? (
          <span className="hidden rounded-full border border-zinc-200 bg-zinc-50 px-3 py-1 text-sm font-medium text-zinc-500 sm:inline-flex">
            {view === "result" ? "Query Result" : "Knowledge Base"}
          </span>
        ) : null}
      </div>
      <div className="flex items-center gap-5 text-sm text-zinc-500">
        <span className="hidden sm:inline">Evidence knowledge base</span>
        <button className="inline-flex h-11 items-center gap-2 rounded-xl bg-[#3178f6] px-4 font-medium text-white shadow-sm transition hover:bg-[#246be8]">
          <Share2 size={18} />
          Share
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
          Evidence Workspace
        </div>

        <h1 className="max-w-[620px] text-[38px] font-semibold leading-tight tracking-tight text-zinc-950">
          Multimodal Evidence Demo
        </h1>
        <p className="mt-5 max-w-[720px] text-[17px] leading-8 text-zinc-600">
          Upload reports, PDFs, screenshots, or charts. The left panel will show indexed
          knowledge-base chunks, while the right panel stays focused on file preview and attachment
          switching.
        </p>

        <div className="mt-7 flex flex-wrap items-center gap-3">
          <UploadButton onUpload={onUpload} variant="primary">
            Upload evidence
          </UploadButton>
          <button
            className="inline-flex items-center gap-2 rounded-xl border border-zinc-200 bg-white px-3 py-2 text-sm font-medium text-zinc-700 transition hover:bg-zinc-50"
            onClick={onSample}
          >
            <Sparkles size={16} />
            Use sample files
          </button>
        </div>

        <div className="mt-8 grid gap-3 sm:grid-cols-3">
          {[
            ["1", "Upload files", "PDFs, images, screenshots, or report attachments"],
            ["2", "Preview evidence", "Switch files and inspect the selected page region"],
            ["3", "Ask questions", "Choose a model and generate cited answers"],
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
            title="Demo flow"
            text="Cover first, knowledge base after upload, query result after asking."
          />
          <InfoCard
            title="Backend handoff"
            text="The frontend API contract is documented and ready for backend integration."
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
  onSelect,
}: {
  activeFile: FileAsset | null;
  chunks: EvidenceChunk[];
  selectedIds: string[];
  focusedId: string;
  statusMessage: string;
  errorMessage: string;
  onSelect: (id: string) => void;
}) {
  const statusLabel = activeFile?.status === "ready" ? "Parsed" : activeFile?.status ?? "uploaded";
  const totalChunks = activeFile?.chunk_count ?? chunks.length;
  const pageCount = activeFile?.page_count ?? 0;
  const visualCount = activeFile?.visual_region_count ?? 0;

  return (
    <motion.section
      className="flex h-full max-w-[820px] flex-col"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.22 }}
    >
      <div className="min-h-0 flex-1 overflow-hidden rounded-xl border border-zinc-200 bg-white">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-100 px-6 py-4">
          <div>
            <h2 className="text-lg font-semibold text-zinc-950">Indexed evidence</h2>
            <p className="mt-1 text-sm leading-6 text-zinc-500">
              Select multiple chunks for retrieval. Click again to remove a chunk.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-1.5 rounded-full bg-[#edf7f3] px-3 py-1.5 text-sm font-medium text-[#26765b]">
              <CheckCircle2 size={15} />
              {statusLabel}
            </span>
            <span className="inline-flex items-center gap-1.5 rounded-full bg-zinc-100 px-3 py-1.5 text-sm font-medium text-zinc-600">
              <Layers3 size={15} />
              {selectedIds.length} selected
            </span>
            <span className="inline-flex max-w-[220px] items-center gap-1.5 truncate rounded-full bg-[#eef4ff] px-3 py-1.5 text-sm font-medium text-[#3178f6]">
              <FileText size={15} className="shrink-0" />
              <span className="truncate">{activeFile?.file_name ?? "No active file"}</span>
            </span>
          </div>
        </div>

        <div className="thin-scrollbar max-h-[calc(100%-132px)] divide-y divide-zinc-100 overflow-y-auto">
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
              {errorMessage || statusMessage || "Waiting for indexed evidence..."}
            </div>
          )}
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 px-6 py-4 text-sm text-zinc-500">
          <span>Total {totalChunks} chunks · {pageCount} pages · {visualCount} visual regions</span>
          <button className="inline-flex items-center gap-1.5 rounded-lg px-2 py-1 font-medium text-zinc-700 transition hover:bg-zinc-100">
            10 / page
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
  return (
    <button
      className={`grid w-full grid-cols-[22px_1fr_auto] gap-4 px-6 py-5 text-left transition ${
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
        <span className="mb-1.5 flex flex-wrap items-center gap-2">
          <span className="rounded-md bg-zinc-100 px-2 py-0.5 text-xs font-medium uppercase text-zinc-600">
            {chunk.type}
          </span>
          <span className="text-xs text-zinc-500">Page {chunk.page}</span>
          <span className="text-xs text-zinc-400">{chunk.id}</span>
        </span>
        <span className="block truncate text-[15px] font-medium text-zinc-950">{chunk.title}</span>
        <span className="mt-1 block line-clamp-2 text-sm leading-6 text-zinc-600">
          {chunk.content}
        </span>
      </span>
      <span className="mt-1 text-sm font-medium text-zinc-500">{chunk.score.toFixed(3)}</span>
    </button>
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
  const previewSrc = selected?.preview_url || selected?.image_url || null;

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
            <h2 className="font-semibold text-zinc-950">Preview</h2>
            <p className="mt-0.5 text-sm text-zinc-500">
              {hasUploaded ? activeFile?.file_name : "Upload files to preview evidence"}
            </p>
          </div>
          <UploadButton onUpload={onUpload}>Add file</UploadButton>
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
                    <span>{selected ? `Page ${selected.page}` : activeFile?.status}</span>
                  </div>
                  {previewSrc ? (
                    <img
                      src={previewSrc}
                      alt={selected?.title ?? activeFile?.file_name ?? "Preview"}
                      className="max-h-[520px] w-full rounded-lg border border-zinc-100 object-contain"
                    />
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
                  {selected ? (
                    <motion.div
                      layout
                      className="mt-5 rounded-lg border border-[#3178f6]/30 bg-blue-50/55 p-4 text-sm leading-6 text-zinc-700"
                    >
                      <div className="mb-1 font-medium text-[#246be8]">{selected.title}</div>
                      {selected.content}
                    </motion.div>
                  ) : (
                    <div className="mt-5 rounded-lg border border-zinc-100 bg-zinc-50 p-4 text-sm leading-6 text-zinc-500">
                      {activeFile?.status === "ready" ? "No evidence selected." : "Parsing evidence..."}
                    </div>
                  )}
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
              <h3 className="text-lg font-semibold text-zinc-950">No preview yet</h3>
              <p className="mt-2 text-sm leading-6 text-zinc-500">
                Upload one or more attachments to preview pages and switch between files.
              </p>
              <div className="mt-5">
                <UploadButton onUpload={onUpload} variant="primary">
                  Upload evidence
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
  onBack,
  onReturnToCover,
}: {
  query: string;
  model: string;
  selected: EvidenceChunk | null;
  selectedChunks: EvidenceChunk[];
  result: AnswerResult | null;
  onBack: () => void;
  onReturnToCover: () => void;
}) {
  const referencedChunks = result?.evidences.length
    ? result.evidences
    : selectedChunks.length
      ? selectedChunks
      : selected
        ? [selected]
        : [];
  const answerText = result?.answer || "No answer returned.";
  const resultModel = result?.model_label || model;

  return (
    <motion.section
      className="flex h-full max-w-[820px] flex-col"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.22 }}
    >
      <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
        <button
          className="inline-flex items-center gap-2 text-xl font-medium text-zinc-950"
          onClick={onBack}
        >
          <ArrowLeft size={22} />
          <span>Back to knowledge base</span>
        </button>
        <button
          className="inline-flex items-center gap-2 rounded-xl border border-zinc-200 bg-white px-3 py-2 text-sm font-medium text-zinc-700 transition hover:bg-zinc-50"
          onClick={onReturnToCover}
        >
          <RotateCcw size={16} />
          Reset demo
        </button>
      </div>

      <h1 className="mb-5 text-[34px] font-semibold tracking-tight text-zinc-950">
        {query || "Query result"}
      </h1>

      <div className="mb-5 flex flex-wrap items-center gap-2">
        <span className="inline-flex items-center gap-1.5 rounded-full bg-blue-50 px-3 py-1.5 text-sm font-medium text-[#246be8]">
          <Sparkles size={15} />
          {resultModel}
        </span>
        <span className="inline-flex items-center gap-1.5 rounded-full bg-zinc-100 px-3 py-1.5 text-sm font-medium text-zinc-600">
          <Search size={15} />
          {referencedChunks.length} selected chunks
        </span>
      </div>

      <article className="min-h-0 flex-1 overflow-y-auto rounded-xl border border-zinc-200 bg-white px-7 py-6">
        <p className="whitespace-pre-wrap text-[17px] leading-8 text-zinc-800">{answerText}</p>

        <div className="mt-7">
          <h2 className="mb-3 text-base font-semibold text-zinc-950">Referenced evidence</h2>
          <div className="space-y-3">
            {referencedChunks.map((chunk) => (
              <div key={chunk.id} className="rounded-lg border border-zinc-100 bg-zinc-50 px-4 py-3">
                <div className="flex items-center justify-between gap-3">
                  <span className="font-medium text-zinc-900">{chunk.title}</span>
                  <span className="shrink-0 text-sm text-zinc-500">{chunk.score.toFixed(3)}</span>
                </div>
                <p className="mt-1 text-sm leading-6 text-zinc-600">{chunk.content}</p>
              </div>
            ))}
          </div>
        </div>
      </article>
    </motion.section>
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
      aria-label={isOpen ? "Question composer" : "Open question composer"}
    >
      <motion.div
        className="absolute inset-0 flex flex-col"
        animate={{ opacity: isOpen ? 1 : 0, scale: isOpen ? 1 : 0.92, filter: isOpen ? "blur(0px)" : "blur(2px)" }}
        transition={{ duration: isOpen ? 0.18 : 0.1, ease: [0.25, 0.1, 0.25, 1] }}
        style={{ pointerEvents: isOpen ? "auto" : "none" }}
      >
        <div className="flex min-h-[72px] items-center gap-3 px-5">
          <input
            value={query}
            disabled={!hasUploaded}
            onChange={(event) => onQueryChange(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") onSubmit();
            }}
            className="min-w-0 flex-1 border-0 bg-transparent text-[20px] text-zinc-800 outline-none placeholder:text-zinc-400 disabled:cursor-not-allowed disabled:text-zinc-400"
            placeholder={hasUploaded ? "Ask a follow-up question" : "Upload evidence before asking"}
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
              <span className="text-zinc-500">Model</span>
              <span className="inline-flex items-center gap-1">
                <span className="text-sm font-semibold text-zinc-800 transition group-hover:text-[#246be8]">
                  {activeModel?.label ?? "Model"}
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
                        <span>{item.label}</span>
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
