"""Model loading and hardware/cache configuration."""

from __future__ import annotations

import os

MODEL_ID = "sd-legacy/stable-diffusion-v1-5"
MODEL_REVISION = "f03de32"


def configure_huggingface_cache() -> str:
    """Configure the Hugging Face cache location."""
    cache_root = os.environ.setdefault(
        "HF_HOME",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
    )
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(cache_root, "hub"))
    return cache_root


def get_device() -> str:
    """Use CUDA when an NVIDIA GPU is available."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_pipeline():
    """Load Stable Diffusion with settings suitable for a 6 GB GPU."""
    import torch
    from diffusers import StableDiffusionPipeline

    configure_huggingface_cache()

    device = get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=dtype,
        use_safetensors=True,
    )

    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cpu")

    return pipe
