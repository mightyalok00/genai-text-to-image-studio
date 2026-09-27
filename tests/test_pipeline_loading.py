from __future__ import annotations

import sys
import types

from src import pipeline


class FakeCuda:
    @staticmethod
    def is_available():
        return True


class FakeTorch:
    float16 = "float16"
    float32 = "float32"
    cuda = FakeCuda()


class FakePipeline:
    def __init__(self):
        self.offloaded = False

    def enable_model_cpu_offload(self):
        self.offloaded = True


def test_load_pipeline_uses_pinned_model_revision(monkeypatch):
    calls = {}

    class FakeStableDiffusionPipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["args"] = args
            calls["kwargs"] = kwargs
            return FakePipeline()

    fake_diffusers = types.SimpleNamespace(
        StableDiffusionPipeline=FakeStableDiffusionPipeline
    )

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setattr(pipeline, "configure_huggingface_cache", lambda: "test-cache")

    loaded = pipeline.load_pipeline()

    assert calls["args"] == (pipeline.MODEL_ID,)
    assert calls["kwargs"]["revision"] == pipeline.MODEL_REVISION
    assert calls["kwargs"]["dtype"] == FakeTorch.float16
    assert calls["kwargs"]["use_safetensors"] is True
    assert loaded.offloaded is True
