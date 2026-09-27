# 🎨 GenAI Text-to-Image Studio

A local GenAI text-to-image application built with Python, PyTorch,
Hugging Face Diffusers, and Streamlit.

## Features

- Text-to-image generation
- Negative prompts
- Stable Diffusion
- CUDA detection
- FP16 on NVIDIA GPU
- CPU fallback
- Accelerate model CPU offloading
- Seed control
- CFG/guidance scale
- Inference-step control
- Multiple resolutions
- PNG download
- Modular source code
- Commented Jupyter notebook

## Optional D: drive model storage

By default, Hugging Face model files use the standard per-user cache at
`~/.cache/huggingface`. On Windows, you can move the cache to D: by setting
these environment variables:

```text
D:\huggingface-cache
D:\huggingface-cache\hub
```

The application honors `HF_HOME` and uses its `hub` subdirectory for
`HF_HUB_CACHE` unless you have already configured `HF_HUB_CACHE` separately.

For a permanent Windows configuration:

```powershell
[Environment]::SetEnvironmentVariable(
    "HF_HOME",
    "D:\huggingface-cache",
    "User"
)

[Environment]::SetEnvironmentVariable(
    "HF_HUB_CACHE",
    "D:\huggingface-cache\hub",
    "User"
)
```

Close and reopen VS Code after changing these variables.

## Install

```powershell
cd D:\GenAI-Text-to-Image-Studio

python -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The portable `requirements.txt` intentionally omits PyTorch and torchvision
so it does not replace an existing CUDA-enabled PyTorch installation. For a
new GPU environment, install the PyTorch build that matches your CUDA setup
before installing the application requirements. `requirements-local-cuda.txt`
records the local GPU dependency set; do not install it over an environment
that already has a working CUDA-enabled PyTorch build.

## Verify GPU

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Run

```powershell
streamlit run app.py
```

## Hardware guidance

For an NVIDIA RTX 3050 6 GB, start with:

- 512 × 512
- 20–30 inference steps
- Batch size 1

The model uses FP16 and Accelerate CPU offloading on CUDA.

## Model

The project uses the pretrained
`sd-legacy/stable-diffusion-v1-5` checkpoint through Diffusers.
The application does not train a foundation model from scratch.

## Architecture

```text
Prompt
  ↓
Text Encoder
  ↓
Diffusion / U-Net
  ↓
Latent Representation
  ↓
VAE Decoder
  ↓
Generated Image
  ↓
Streamlit UI
```

## Responsible use

Review generated content before publishing and respect model licenses,
copyright, privacy, platform rules, and applicable laws.
