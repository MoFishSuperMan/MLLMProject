# Video Voiceover Workflow

目标视频：`60sdemo演示视频.mp4`，当前系统识别到时长约 55 秒。

已经准备好的文件：

- `scripts/video_voiceover_segments.json`：55 秒短口播稿和时间轴。
- `scripts/make_video_voiceover.py`：调用 edge-tts 生成分段配音，生成 SRT 字幕，并用 FFmpeg 合成最终 MP4。

## 需要安装

1. FFmpeg

   安装后需要能在 PowerShell 中直接运行：

   ```powershell
   ffmpeg -version
   ffprobe -version
   ```

   Windows 可选安装方式：

   ```powershell
   winget install Gyan.FFmpeg
   ```

   或：

   ```powershell
   choco install ffmpeg
   ```

   或：

   ```powershell
   scoop install ffmpeg
   ```

2. edge-tts

   不需要写入项目依赖，可以用 uv 临时安装运行：

   ```powershell
   $env:UV_CACHE_DIR = "E:\GitHub\MLLMProject\.uv-cache"
   uv run --no-project --with edge-tts python scripts\make_video_voiceover.py
   ```

   如果本机 `uv` 找 Python 有问题，可以指定 Codex 自带 Python：

   ```powershell
   $env:UV_CACHE_DIR = "E:\GitHub\MLLMProject\.uv-cache"
   uv run --no-project --python "C:\Users\刘天翔\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" --with edge-tts python scripts\make_video_voiceover.py
   ```

## 输出

默认会生成：

- `data/video_voiceover/segment_*.mp3`
- `data/video_voiceover/voiceover.srt`
- `data/video_voiceover/voiceover_mix.m4a`
- `output/60sdemo演示视频_voiceover.mp4`

最终 MP4 会包含：

- 原视频画面
- edge-tts 生成的新配音
- 可开关的软字幕

## 调整口播

如果配音太快或太慢，可以改运行参数：

```powershell
uv run --no-project --with edge-tts python scripts\make_video_voiceover.py --rate "-10%"
```

如果想换声音：

```powershell
uv run --no-project --with edge-tts python scripts\make_video_voiceover.py --voice zh-CN-XiaoxiaoNeural
```

推荐声音：

- `zh-CN-YunjianNeural`：男声，偏正式，适合项目汇报。
- `zh-CN-YunxiNeural`：男声，年轻一些。
- `zh-CN-XiaoxiaoNeural`：女声，清晰自然。

## 抽关键帧后精修

装好 FFmpeg 后，可以抽帧给我检查画面节奏：

```powershell
mkdir data\video_voiceover\frames
ffmpeg -y -i "60sdemo演示视频.mp4" -vf "fps=1/5,scale=1280:-1" data\video_voiceover\frames\frame_%03d.jpg
```

我会根据这些帧继续微调 `scripts/video_voiceover_segments.json` 的时间点和文案。
