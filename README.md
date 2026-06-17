# MLLMProject

一个面向多模态文档问答的课程项目 Demo。系统支持上传 PDF，解析文本、表格、公式、代码和图片 chunk，并在前端以聊天形式展示回答和引用证据。

## 快速开始

后端 API：

```powershell
uv sync
uv run uvicorn mllmproject.api:app --reload --host 127.0.0.1 --port 8000
```

前端页面：

```powershell
cd frontend
npm install
npm run dev
```

启动后打开：

```text
http://127.0.0.1:5173/
```
