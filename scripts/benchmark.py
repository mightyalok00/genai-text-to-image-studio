"""Benchmark local Stable Diffusion inference.

Run this on the target GPU after the model is available in the Hugging Face cache.
Results are printed as CSV so they can be copied into docs/benchmarks.md.
"""

from __future__ import annotations

import csv
import io
import time
from contextlib import redirect_stdout

import torch

from src.generation import generate_image
from src.pipeline import load_pipeline


CASES = [
    (512, 512, 20),
    (512, 512, 30),
    (512, 768, 30),
    (768, 512, 30),
]


def benchmark_case(pipe, width: int, height: int, steps: int) -> dict[str, object]:
    prompt = "A cinematic futuristic Indian city at sunset, highly detailed, realistic lighting"
    seed = 42

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Warm-up / first-run overhead is excluded from the measured case.
    with redirect_stdout(io.StringIO()):
        generate_image(
            pipe,
            prompt=prompt,
            steps=5,
            seed=seed,
            width=width,
            height=height,
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    with redirect_stdout(io.StringIO()):
        generate_image(
            pipe,
            prompt=prompt,
            steps=steps,
            seed=seed,
            width=width,
            height=height,
        )
    elapsed = time.perf_counter() - start

    peak_vram_gb = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )

    return {
        "resolution": f"{width}x{height}",
        "steps": steps,
        "seed": seed,
        "time_seconds": round(elapsed, 3),
        "peak_vram_gb": round(peak_vram_gb, 3) if torch.cuda.is_available() else "",
    }


def main() -> None:
    print("Loading pipeline...")
    pipe = load_pipeline()

    print("resolution,steps,seed,time_seconds,peak_vram_gb")
    for width, height, steps in CASES:
        result = benchmark_case(pipe, width, height, steps)
        print(
            f"{result['resolution']},{result['steps']},{result['seed']},"
            f"{result['time_seconds']},{result['peak_vram_gb']}"
        )


if __name__ == "__main__":
    main()
