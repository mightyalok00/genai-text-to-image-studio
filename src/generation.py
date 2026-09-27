"""Reusable text-to-image inference function."""

from __future__ import annotations

import torch
from PIL import Image


def generate_image(
    pipe,
    prompt: str,
    negative_prompt: str = "",
    steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = 42,
    width: int = 512,
    height: int = 512,
) -> Image.Image:
    """Generate one image with a reproducible or random seed."""
    if seed >= 0:
        generator_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(
            device=generator_device
        ).manual_seed(seed)
    else:
        generator = None

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    )

    return result.images[0]
