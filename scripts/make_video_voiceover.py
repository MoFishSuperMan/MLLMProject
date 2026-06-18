from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def resolve_tool(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found

    candidates: list[Path] = []
    if os.name == "nt":
        try:
            import winreg

            for root, subkey in (
                (winreg.HKEY_CURRENT_USER, "Environment"),
                (
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
                ),
            ):
                try:
                    with winreg.OpenKey(root, subkey) as key:
                        value, _ = winreg.QueryValueEx(key, "Path")
                except OSError:
                    continue
                for item in str(value).split(";"):
                    if item:
                        candidates.append(Path(os.path.expandvars(item)) / f"{name}.exe")
        except ImportError:
            pass

        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            winget_root = (
                Path(local_appdata) / "Microsoft" / "WinGet" / "Packages"
            )
            if winget_root.exists():
                candidates.extend(winget_root.glob(f"**/{name}.exe"))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise SystemExit(
        f"Missing required tool: {name}. Install FFmpeg and make sure `{name}` is on PATH."
    )


def load_segments(path: Path) -> tuple[dict[str, Any], list[Segment]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = [
        Segment(float(item["start"]), float(item["end"]), str(item["text"]))
        for item in data["segments"]
    ]
    if not segments:
        raise SystemExit("No segments found.")
    previous_end = 0.0
    for index, segment in enumerate(segments):
        if segment.start < previous_end - 0.001:
            raise SystemExit(f"Segment {index} overlaps the previous segment.")
        if segment.end <= segment.start:
            raise SystemExit(f"Segment {index} has a non-positive duration.")
        previous_end = segment.end
    return data, segments


def probe_duration(ffprobe: str, video: Path) -> float:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def srt_timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def write_srt(path: Path, segments: list[Segment]) -> None:
    lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        lines.extend(
            [
                str(index),
                f"{srt_timestamp(segment.start)} --> {srt_timestamp(segment.end)}",
                segment.text,
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


async def synthesize_segments(
    segments: list[Segment],
    output_dir: Path,
    voice: str,
    rate: str,
    pitch: str,
    volume: str,
) -> list[Path]:
    try:
        import edge_tts
    except ImportError as exc:
        raise SystemExit(
            "Missing Python package: edge-tts. Run with `uv run --with edge-tts ...`."
        ) from exc

    audio_paths: list[Path] = []
    for index, segment in enumerate(segments, start=1):
        audio_path = output_dir / f"segment_{index:02}.mp3"
        print(f"Synthesizing {audio_path.name}: {segment.text}")
        communicate = edge_tts.Communicate(
            segment.text,
            voice=voice,
            rate=rate,
            pitch=pitch,
            volume=volume,
        )
        await communicate.save(str(audio_path))
        audio_paths.append(audio_path)
    return audio_paths


def mix_segments(
    ffmpeg: str,
    audio_paths: list[Path],
    segments: list[Segment],
    output_audio: Path,
    duration: float,
) -> None:
    cmd = [ffmpeg, "-y"]
    for audio_path in audio_paths:
        cmd.extend(["-i", str(audio_path)])

    filters: list[str] = []
    labels: list[str] = []
    for index, segment in enumerate(segments):
        delay_ms = max(0, round(segment.start * 1000))
        label = f"a{index}"
        filters.append(
            f"[{index}:a]atrim=0:{segment.duration:.3f},asetpts=N/SR/TB,"
            f"adelay={delay_ms}|{delay_ms}[{label}]"
        )
        labels.append(f"[{label}]")

    filters.append(
        "".join(labels)
        + f"amix=inputs={len(labels)}:duration=longest:normalize=0,"
        + "loudnorm=I=-16:TP=-1.5:LRA=11,"
        + f"apad=pad_dur={duration:.3f},"
        + f"atrim=0:{duration:.3f}[aout]"
    )

    cmd.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[aout]",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_audio),
        ]
    )
    run(cmd)


def mux_video(
    ffmpeg: str, video: Path, audio: Path, subtitles: Path, output_video: Path
) -> None:
    run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video),
            "-i",
            str(audio),
            "-i",
            str(subtitles),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-map",
            "2:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-c:s",
            "mov_text",
            "-metadata:s:s:0",
            "language=chi",
            str(output_video),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate edge-tts voiceover, SRT subtitles, and a dubbed MP4."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("60sdemo演示视频.mp4"),
        help="Input video path.",
    )
    parser.add_argument(
        "--segments",
        type=Path,
        default=Path("scripts/video_voiceover_segments.json"),
        help="JSON file containing timed voiceover segments.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/video_voiceover"),
        help="Directory for generated audio and subtitle assets.",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=Path("output/60sdemo演示视频_voiceover.mp4"),
        help="Final MP4 path.",
    )
    parser.add_argument("--voice", default=None, help="edge-tts voice name.")
    parser.add_argument("--rate", default="+10%", help="edge-tts speaking rate.")
    parser.add_argument("--pitch", default="+0Hz", help="edge-tts pitch.")
    parser.add_argument("--volume", default="+0%", help="edge-tts volume.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ffmpeg = resolve_tool("ffmpeg")
    ffprobe = resolve_tool("ffprobe")

    data, segments = load_segments(args.segments)
    voice = args.voice or data.get("voice") or "zh-CN-YunjianNeural"

    video = args.video.resolve()
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_video.parent.mkdir(parents=True, exist_ok=True)

    video_duration = probe_duration(ffprobe, video)
    script_duration = max(segment.end for segment in segments)
    duration = min(video_duration, script_duration)
    if abs(video_duration - script_duration) > 1.0:
        print(
            f"Warning: video duration is {video_duration:.2f}s, "
            f"script ends at {script_duration:.2f}s."
        )

    srt_path = args.output_dir / "voiceover.srt"
    mix_path = args.output_dir / "voiceover_mix.m4a"
    write_srt(srt_path, segments)
    audio_paths = asyncio.run(
        synthesize_segments(
            segments,
            args.output_dir,
            voice=voice,
            rate=args.rate,
            pitch=args.pitch,
            volume=args.volume,
        )
    )
    mix_segments(ffmpeg, audio_paths, segments, mix_path, duration)
    mux_video(ffmpeg, video, mix_path, srt_path, args.output_video)

    print(f"Done: {args.output_video}")
    print(f"Subtitle: {srt_path}")
    print(f"Audio: {mix_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
