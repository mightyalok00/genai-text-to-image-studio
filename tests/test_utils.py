from __future__ import annotations

from io import BytesIO

from PIL import Image

from src.utils import image_to_bytes


def test_image_to_bytes_returns_valid_png():
    image = Image.new("RGB", (8, 8), "white")
    data = image_to_bytes(image)
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    decoded = Image.open(BytesIO(data))
    assert decoded.size == (8, 8)
    assert decoded.format == "PNG"
