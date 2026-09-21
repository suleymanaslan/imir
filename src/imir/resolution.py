"""Checkpoint-controlled canvas policy; legacy preprocessing remains unchanged."""

import math

from PIL import Image

from .images import prepare_image


def target_area_size(size: tuple[int, int], pixels: int) -> tuple[int, int]:
    """Preserve aspect ratio up to 16-pixel alignment, allowing up/downscaling."""
    width, height = size
    if min(width, height) < 16 or pixels < 256:
        raise ValueError("Images and target budget must accommodate a 16-pixel grid")
    scale = math.sqrt(pixels / (width * height))
    result = (int(width * scale) // 16 * 16, int(height * scale) // 16 * 16)
    if min(result) < 16:
        raise ValueError("Image is too elongated for the target pixel budget")
    return result


def prepare_canvas(image: Image.Image, pixels: int, resize_mode="downscale_only"):
    if resize_mode == "downscale_only":
        return prepare_image(image, pixels)
    if resize_mode == "target_area":
        return image.resize(target_area_size(image.size, pixels), Image.Resampling.LANCZOS)
    raise ValueError(f"Unknown resize mode: {resize_mode}")
