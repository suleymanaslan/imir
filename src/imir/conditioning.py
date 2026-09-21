"""Vision-language encoding for image-derived restoration instructions."""

import torch
from PIL import Image

from .images import instruction_size, load_image


@torch.no_grad()
def encode_image(backbone, image: Image.Image, instruction_pixels: int):
    image = load_image(image)
    image = image.resize(instruction_size(image.size, instruction_pixels), Image.Resampling.LANCZOS)
    embeddings, mask = backbone.encode_prompt(
        prompt="",
        image=[image],
        device=backbone._execution_device,
    )
    if mask is None:
        mask = torch.ones(embeddings.shape[:2], dtype=torch.bool, device=embeddings.device)
    return embeddings, mask.bool()
