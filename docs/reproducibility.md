# Reproducibility

## Goal

The project supports reproducible local image generation when the same model revision, prompt, generation settings, seed, software environment, and hardware/runtime are used.

## Tested baseline

| Component | Tested baseline |
|---|---|
| Python | 3.14 |
| PyTorch | 2.14.0+cu126 |
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| Diffusers | 0.40.0 |
| Transformers | 5.17.0 |
| Accelerate | 1.15.0 |
| Precision | FP16 on CUDA |
| Model | sd-legacy/stable-diffusion-v1-5 |

The CI test environment uses pinned direct dependency versions in requirements-ci.txt. The local CUDA environment is kept separate because the CUDA-specific PyTorch wheel depends on the target platform and driver stack.

## Deterministic generation

Use a fixed seed such as 42. The application passes that seed to a PyTorch Generator. Keeping the prompt, negative prompt, resolution, inference steps, guidance scale, seed, model revision, and compatible runtime unchanged makes results comparable.

Exact bit-for-bit reproducibility is not guaranteed across different GPU architectures, CUDA/cuDNN versions, PyTorch versions, Diffusers versions, model revisions, schedulers, or CPU versus CUDA execution.

## Model revision

For stronger experiment reproducibility, record the exact model revision used rather than relying indefinitely on a moving model reference. The application currently uses sd-legacy/stable-diffusion-v1-5.

## Environment and cache

Keep Hugging Face model caches outside Git. Set HF_HOME and HF_HUB_CACHE explicitly when a known cache location is required. Never commit model weights, credentials, or private generated content.

## Validation

Run the unit tests with:

    python -m pytest -q

Run the syntax check with:

    python -m compileall -q src tests

The CI workflow runs both checks on pushes and pull requests targeting main.
