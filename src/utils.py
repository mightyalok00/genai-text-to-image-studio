"""Small application utilities."""

from __future__ import annotations

import io


def image_to_bytes(image) -> bytes:
    """Convert a PIL image to PNG bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def format_duration(seconds: float) -> str:
    """Format generation time for a compact UI metric."""
    return f"{seconds:.1f} s"


def format_vram(bytes_used: int) -> str:
    """Format GPU memory usage."""
    if not bytes_used:
        return "N/A"
    return f"{bytes_used / (1024**3):.2f} GB"


def validate_generation_settings(
    prompt: str,
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
) -> str | None:
    """Return a user-facing validation error, or None when valid."""
    if not prompt.strip():
        return "Enter a prompt before generating."
    if not 10 <= steps <= 50:
        return "Inference steps must be between 10 and 50."
    if not 1.0 <= guidance_scale <= 15.0:
        return "Guidance scale must be between 1 and 15."
    if width % 8 != 0 or height % 8 != 0:
        return "Width and height must be divisible by 8."
    if width < 256 or height < 256:
        return "Resolution must be at least 256 × 256."
    return None
