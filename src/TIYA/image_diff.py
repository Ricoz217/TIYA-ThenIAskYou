"""从原图与标注图的像素差异提取透明 PNG 图层，不依赖 LLM 接口。"""

from __future__ import annotations

import asyncio

from functools import partial
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageChops, ImageFilter, ImageOps

from TIYA.executor import GLOBAL_EXECUTOR

def extract_markup(
    original: Path | bytes,
    annotated: Path | bytes,
    *,
    tolerance: int = 60,
    resize: bool = True,
    compare_alpha: bool | None = None,
    despeckle: bool = False,
) -> bytes:
    """提取新增笔迹，返回与原图显示尺寸一致的 RGBA PNG 字节。

    original 为未标注的原图，annotated 为在原图上圈画后的图片。输入支持
    Path 或编码后的图片 bytes，先按 EXIF 方向解码，不修改源文件。多帧图片只取首帧。

    tolerance 为 0~255 的整数。参与比较的通道中只要有一个差值超过容差，
    就保留标注图的像素及其透明度，其余像素清零为透明。压缩、缩放产生的
    差异也可能超过容差，需按实际传输情况调整。

    compare_alpha 默认自动判断：标注图全不透明时只比 RGB，以兼容传输丢失
    透明度的情况。True 始终比较 alpha，False 始终忽略。despeckle=True
    清除仅由一个或两个像素组成的孤立碎片，相邻的对角像素也算连在一起。

    resize=True 时将标注图缩放回原图的显示尺寸；False 时尺寸不一致抛
    ValueError。不做裁剪或图像配准，调用方须保证两图内容位置对应。
    文件缺失和图片解码错误原样抛出。同步 CPU 操作，不应直接阻塞异步业务。
    """
    if isinstance(tolerance, bool) or not isinstance(tolerance, int) or not 0 <= tolerance <= 255:
        raise ValueError("tolerance 必须是 0~255 的整数")

    with _load_rgba(original) as original_image:
        annotated_image = _load_rgba(annotated)
        try:
            if annotated_image.size != original_image.size:
                if not resize:
                    raise ValueError("原图与标注图尺寸不一致")

                resized = annotated_image.resize(original_image.size, Image.Resampling.LANCZOS)
                annotated_image.close()
                annotated_image = resized

            if compare_alpha is None:
                compare_alpha = annotated_image.getchannel("A").getextrema()[0] < 255

            selection = _difference_mask(original_image, annotated_image, tolerance, compare_alpha)
            if despeckle:
                selection = _remove_speckles(selection)

            with Image.new("RGBA", original_image.size, (0, 0, 0, 0)) as layer:
                # 清除未选区的 RGB，避免透明 PNG 仍携带整张原图的背景像素。
                layer.paste(annotated_image, (0, 0), selection)
                output = BytesIO()
                layer.save(output, format="PNG")
                return output.getvalue()

        finally:
            annotated_image.close()


async def extract_markup_async(
    original: Path | bytes,
    annotated: Path | bytes,
    *,
    tolerance: int = 60,
    resize: bool = True,
    compare_alpha: bool | None = None,
    despeckle: bool = False,
) -> bytes:
    """在线程池执行 extract_markup()，接收图片路径或字节并返回 PNG 字节。"""
    extract = partial(
        extract_markup,
        original,
        annotated,
        tolerance=tolerance,
        resize=resize,
        compare_alpha=compare_alpha,
        despeckle=despeckle,
    )
    content = await asyncio.get_running_loop().run_in_executor(GLOBAL_EXECUTOR, extract)
    return content


def _load_rgba(source: Path | bytes) -> Image.Image:
    if isinstance(source, bytes):
        input_file = BytesIO(source)

    elif isinstance(source, Path):
        input_file = source

    else:
        raise TypeError("图片输入需要 Path 或 bytes，请先将缓存 ID 转成文件路径")

    with Image.open(input_file) as image:
        ImageOps.exif_transpose(image, in_place=True)
        return image.convert("RGBA")


def _difference_mask(
    original: Image.Image,
    annotated: Image.Image,
    tolerance: int,
    compare_alpha: bool,
) -> Image.Image:
    # Pillow 按通道批量计算，不在 Python 中逐像素循环。
    difference = ImageChops.difference(original, annotated)
    strength = difference.getchannel("R")
    channels = ("G", "B", "A") if compare_alpha else ("G", "B")
    for channel in channels:
        strength = ImageChops.lighter(strength, difference.getchannel(channel))

    return strength.point([255 if value > tolerance else 0 for value in range(256)])


def _remove_speckles(selection: Image.Image) -> Image.Image:
    """清除单点和双点碎片，保留较长笔迹的轮廓和端点。"""
    padded = ImageOps.expand(selection, border=1, fill=0)
    kernel = ImageFilter.Kernel((3, 3), [1] * 9, scale=255)
    counts = padded.filter(kernel).crop((1, 1, selection.width + 1, selection.height + 1))

    # 先保留至少有两个邻居的像素，每个较大连通区域都至少有一个。
    supported = ImageChops.darker(selection, counts.point([0] * 3 + [255] * 253))
    # 恢复只连一个邻居的端点，交回原选区确保笔迹不会变粗。
    return ImageChops.darker(selection, supported.filter(ImageFilter.MaxFilter(3)))
