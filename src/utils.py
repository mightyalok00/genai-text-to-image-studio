"""Small application utilities."""

from __future__ import annotations

import io


def image_to_bytes(image) -> bytes:
    """Convert a PIL image to PNG bytes for Streamlit."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
