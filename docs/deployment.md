# Local Execution

## Current status

This repository is a **local-first text-to-image inference project**.

There is no hosted application or web UI requirement. The core workflow runs directly on the developer's machine using Python, PyTorch, Hugging Face Diffusers, and Stable Diffusion v1.5.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Git Bash:

```bash
source .venv/Scripts/activate
```

Install a PyTorch build appropriate for the local CPU/GPU, then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the experiment notebook

Start Jupyter:

```bash
python -m jupyter notebook
```

Open:

```text
notebooks/text_to_image_experiments.ipynb
```

The notebook demonstrates environment verification, Hugging Face cache configuration, model loading, prompt configuration, inference, and image output.

## GPU requirements

Stable Diffusion v1.5 is resource-intensive.

For approximately 6 GB VRAM:

| Setting | Starting point |
|---|---:|
| Resolution | 512 × 512 |
| Inference steps | 20–30 |
| Batch size | 1 |
| Precision | FP16 |
| Offloading | Enabled |

## Environment variables

The project respects:

- `HF_HOME`
- `HF_HUB_CACHE`

These can be used to place model downloads on a larger drive.

## Reproducibility

For repeatable experiments:

- Pin the software environment.
- Keep the model revision fixed.
- Record resolution, steps, guidance scale, and seed.
- Use the same hardware when comparing performance.
- Keep generated outputs outside Git unless they are intentionally selected as project assets.

## Security

Do not commit:

- API keys
- access tokens
- `.env` files
- virtual environments
- Hugging Face caches
- model weights
- private generated data
