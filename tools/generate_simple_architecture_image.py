from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "assets" / "system_architecture.png"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc"
    return ImageFont.truetype(path, size=size)


def rounded_box(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, subtitle: str) -> None:
    draw.rounded_rectangle(box, radius=18, fill=(238, 244, 255), outline=(190, 204, 224), width=2)
    x1, y1, x2, y2 = box
    draw.text((x1 + 22, y1 + 18), title, fill=(31, 41, 55), font=font(30, bold=True))
    draw.text((x1 + 22, y1 + 58), subtitle, fill=(100, 116, 139), font=font(22))


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None:
    draw.line([start, end], fill=(37, 99, 235), width=5)
    ex, ey = end
    sx, sy = start
    if ex >= sx:
        points = [(ex, ey), (ex - 16, ey - 9), (ex - 16, ey + 9)]
    else:
        points = [(ex, ey), (ex + 16, ey - 9), (ex + 16, ey + 9)]
    draw.polygon(points, fill=(37, 99, 235))


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1500, 620), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 1499, 619], outline=(226, 232, 240), width=3)
    draw.text((48, 34), "图1 多模态文档问答系统架构图", fill=(31, 41, 55), font=font(34, bold=True))

    boxes = {
        "upload": (70, 145, 340, 255),
        "parse": (430, 145, 700, 255),
        "detect": (790, 145, 1060, 255),
        "index": (430, 365, 700, 475),
        "qa": (790, 365, 1060, 475),
    }
    rounded_box(draw, boxes["upload"], "文档上传", "PDF / 页面截图")
    rounded_box(draw, boxes["parse"], "页面解析", "文本与页面图像")
    rounded_box(draw, boxes["detect"], "区域检测", "图片 / 表格")
    rounded_box(draw, boxes["index"], "索引构建", "Text / Visual chunks")
    rounded_box(draw, boxes["qa"], "问答生成", "回答与来源")

    arrow(draw, (340, 200), (430, 200))
    arrow(draw, (700, 200), (790, 200))
    arrow(draw, (565, 255), (565, 365))
    arrow(draw, (925, 255), (925, 365))
    arrow(draw, (700, 420), (790, 420))

    draw.text((1110, 174), "图片 chunk: 保留图号、caption、裁剪图", fill=(71, 85, 105), font=font(24))
    draw.text((1110, 220), "表格 chunk: 保留表号、行列结构", fill=(71, 85, 105), font=font(24))
    draw.text((1110, 408), "文本 chunk: 保留正文段落", fill=(71, 85, 105), font=font(24))
    image.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
