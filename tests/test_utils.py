from __future__ import annotations

from io import BytesIO

from PIL import Image

from src.utils import (
    format_duration,
    format_vram,
    image_to_bytes,
    validate_generation_settings,
)


def test_image_to_bytes_returns_valid_png():
    image = Image.new("RGB", (8, 8), "white")

    data = image_to_bytes(image)

    assert data.startswith(b"\x89PNG\r\n\x1a\n")

    decoded = Image.open(BytesIO(data))
    assert decoded.size == (8, 8)
    assert decoded.format == "PNG"


def test_format_helpers():
    assert format_duration(7.234) == "7.2 s"
    assert format_vram(2 * 1024**3) == "2.00 GB"
    assert format_vram(0) == "N/A"


def test_validate_generation_settings():
    assert validate_generation_settings("", 30, 7.5, 512, 512)
    assert validate_generation_settings("test", 9, 7.5, 512, 512)
    assert validate_generation_settings("test", 30, 16, 512, 512)
    assert validate_generation_settings("test", 30, 7.5, 511, 512) 
    assert validate_generation_settings("test", 30, 7.5, 128, 128)
    assert validate_generation_settings("test", 30, 7.5, 512, 512) is None
