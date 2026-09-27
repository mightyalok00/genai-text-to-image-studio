from __future__ import annotations

import os

from src.pipeline import configure_huggingface_cache, get_device


def test_configure_huggingface_cache_respects_existing_environment(monkeypatch):
    monkeypatch.setenv("HF_HOME", "/tmp/test-hf")
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    cache_root = configure_huggingface_cache()
    assert cache_root == "/tmp/test-hf"
    assert os.environ["HF_HUB_CACHE"] == "/tmp/test-hf/hub"


def test_get_device_returns_supported_device():
    assert get_device() in {"cuda", "cpu"}
