# Performance Baseline

This document records the local development baseline for the current Stable Diffusion v1.5 implementation.

## Development hardware

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- VRAM: 6 GB
- Precision: FP16
- PyTorch: 2.14.0+cu126
- Model: `sd-legacy/stable-diffusion-v1-5`

## Baseline generation

A representative local notebook run produced an image in approximately **7 seconds** using:

| Parameter | Value |
|---|---:|
| Resolution | 512 × 512 |
| Inference steps | 25 |
| Guidance scale | 7.5 |
| Seed | 42 |
| Precision | FP16 |
| Device | RTX 3050 6 GB |

Actual time varies with GPU power mode, thermals, background processes, software versions, and memory pressure.

## Measuring local inference

The notebook workflow can record generation time and inspect peak CUDA memory when running on an NVIDIA GPU.

Use those measurements to build a machine-specific benchmark table rather than treating the baseline above as a universal performance claim.

## Recommended benchmark matrix

For future comparisons, record:

| Resolution | Steps | Seed | Time | Peak VRAM |
|---|---:|---:|---:|---:|
| 512 × 512 | 20 | 42 | — | — |
| 512 × 512 | 30 | 42 | — | — |
| 512 × 768 | 30 | 42 | — | — |
| 768 × 512 | 30 | 42 | — | — |

Run each configuration more than once if you want a stable average. Keep the model revision and software environment fixed when comparing changes.
