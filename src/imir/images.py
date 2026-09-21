"""RGB image loading and checkpoint-compatible resizing."""

import math
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def load_image(image: Image.Image | str | Path) -> Image.Image:
    if isinstance(image, Image.Image):
        return ImageOps.exif_transpose(image).convert("RGB")
    with Image.open(image) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def restoration_size(size: tuple[int, int], max_pixels: int) -> tuple[int, int]:
    w, h = size
    if min(w, h) < 16 or max_pixels < 256:
        raise ValueError("Images must be at least 16 pixels per side; budget must be >=256")
    scale = min(1.0, math.sqrt(max_pixels / (w * h)))
    width, height = int(w * scale) // 16 * 16, int(h * scale) // 16 * 16
    if min(width, height) < 16:
        raise ValueError("Image is too elongated for this pixel budget")
    return width, height


def instruction_size(size: tuple[int, int], pixels: int) -> tuple[int, int]:
    ratio = size[0] / size[1]
    width = round(math.sqrt(pixels * ratio) / 32) * 32
    height = round(math.sqrt(pixels / ratio) / 32) * 32
    if min(width, height) < 32:
        raise ValueError("Image is too elongated for the instruction pixel budget")
    return width, height


def prepare_image(image: Image.Image, max_pixels: int) -> Image.Image:
    return image.resize(restoration_size(image.size, max_pixels), Image.Resampling.LANCZOS)
