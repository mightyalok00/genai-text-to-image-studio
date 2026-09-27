# 🎨 GenAI Text-to-Image Studio

[![Python](https://img.shields.io/badge/Python-3.14%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20ready-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/Hugging%20Face-Diffusers-yellow?logo=huggingface&logoColor=black)](https://huggingface.co/docs/diffusers)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python checks](https://github.com/mightyalok00/genai-text-to-image-studio/actions/workflows/python-checks.yml/badge.svg)](https://github.com/mightyalok00/genai-text-to-image-studio/actions/workflows/python-checks.yml)

A modular **local text-to-image GenAI application** built with Python, PyTorch, Hugging Face Diffusers, Stable Diffusion v1.5, and Streamlit.

Generate images from natural-language prompts while controlling inference steps, guidance scale, seed, negative prompts, and resolution. The application is designed for local NVIDIA GPU inference and includes CPU offloading to reduce peak VRAM usage.

> **Project scope:** inference-focused text-to-image generation. This repository does not train or clone a foundation model.

## 🚀 Live Demo

**Try the deployed Streamlit app:**  
👉 https://alok-text-to-image.streamlit.app/

## ✨ Features

- 🖼️ Text-to-image generation
- ✍️ Positive and negative prompts
- 🎛️ Inference-step control
- 🎚️ CFG / guidance-scale control
- 🎲 Reproducible seeds or random generation
- 📐 512×512, 512×768, and 768×512 output options
- ⚡ CUDA acceleration with FP16
- 🧠 Accelerate CPU offloading for lower peak VRAM usage
- 💻 CPU fallback
- 📥 PNG download
- ♻️ Streamlit model caching to avoid repeated model initialization
- 🧩 Modular Python source code
- 📓 Reproducible Jupyter experiment notebook
- 🔒 No API key required for local inference

## 🧱 Architecture

```text
User prompt
    │
    ▼
Streamlit UI
    │
    ▼
Generation parameters
    │
    ▼
src/generation.py
    │
    ▼
src/pipeline.py
    │
    ├── Hugging Face cache configuration
    ├── Device detection
    └── Stable Diffusion pipeline
            │
            ▼
     sd-legacy/stable-diffusion-v1-5
            │
            ▼
      Generated PIL Image
            │
            ├── Preview in Streamlit
            └── Download as PNG
```

See [docs/architecture.md](docs/architecture.md) for the component breakdown.

## 📁 Project structure

```text
genai-text-to-image-studio/
├── app.py                         # Streamlit application
├── src/
│   ├── __init__.py
│   ├── generation.py              # Image generation logic
│   ├── pipeline.py                # Model, device, and cache management
│   └── utils.py                   # Small application utilities
├── notebooks/
│   └── text_to_image_experiments.ipynb
├── assets/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
├── docs/
│   ├── architecture.md
│   └── deployment.md
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   ├── pull_request_template.md
│   └── workflows/
│       └── python-checks.yml
├── .gitignore
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── requirements.txt
└── requirements-local-cuda.txt
```

## 🚀 Quick start

### 1. Clone

```bash
git clone https://github.com/mightyalok00/genai-text-to-image-studio.git
cd genai-text-to-image-studio
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
```

Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
```

### 3. Install PyTorch

Install a PyTorch build appropriate for your operating system and NVIDIA/CUDA configuration using the official [PyTorch installation selector](https://pytorch.org/get-started/locally/).

This project intentionally keeps PyTorch out of the portable `requirements.txt` so a generic application install does not replace an existing CUDA-enabled PyTorch build.

For the local GPU environment used while developing this project, `requirements-local-cuda.txt` records the dependency set.

### 4. Install application dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Verify CUDA

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 6. Run

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, normally:

```text
http://localhost:8501
```

## ⚙️ Hugging Face cache

The application uses the standard Hugging Face cache by default.

On Windows, you can optionally move model downloads to another drive:

```powershell
[Environment]::SetEnvironmentVariable("HF_HOME", "D:\\huggingface-cache", "User")
[Environment]::SetEnvironmentVariable("HF_HUB_CACHE", "D:\\huggingface-cache\\hub", "User")
```

Restart your terminal/VS Code after changing persistent environment variables.

The cache and model weights are intentionally excluded from Git by `.gitignore`.

## 🖥️ Hardware guidance

For a GPU with approximately **6 GB VRAM**, start with:

| Setting | Suggested starting point |
|---|---:|
| Resolution | 512 × 512 |
| Inference steps | 20–30 |
| Batch size | 1 |
| Precision | FP16 |
| Offloading | Enabled |

Higher resolutions and more inference steps increase memory use and generation time.

## 🤖 Model

The application loads:

```text
sd-legacy/stable-diffusion-v1-5
```

The model is downloaded at runtime through Hugging Face Diffusers rather than stored in this repository. The model card identifies the checkpoint as a Stable Diffusion v1.5 model and lists the CreativeML OpenRAIL-M license. Review the model's current terms before distributing or deploying generated content.

- [Stable Diffusion v1.5 model card](https://huggingface.co/sd-legacy/stable-diffusion-v1-5)
- [Diffusers documentation](https://huggingface.co/docs/diffusers)
- [PyTorch](https://pytorch.org/)

## 🧪 Testing and reproducibility

The repository includes unit tests for generation parameters, seed handling, Hugging Face cache configuration, device detection, and PNG output.

Run locally with:

```bash
python -m pytest -q
python -m compileall -q app.py src tests
```

CI installs pinned test dependencies from `requirements-ci.txt` and runs the same tests and compile checks on pushes and pull requests. See [docs/reproducibility.md](docs/reproducibility.md) for the tested environment and reproducibility limitations.

## 📓 Experiments

The notebook in `notebooks/text_to_image_experiments.ipynb` demonstrates the same inference flow used by the application:

1. Verify the Python/CUDA environment
2. Configure the Hugging Face cache
3. Select hardware and precision
4. Load Stable Diffusion
5. Define prompts
6. Configure generation parameters
7. Generate an image
8. Display and save the result

The notebook is intended for experimentation and reproducibility; the Streamlit application is the primary user interface.

## 🔐 Security and repository hygiene

Do not commit:

- API keys or access tokens
- `.env` files
- virtual environments
- Hugging Face caches
- Stable Diffusion model weights
- generated local output files

See [SECURITY.md](SECURITY.md) for reporting security issues.

## 🤝 Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening an issue or pull request.

## 📄 License

The application source code is released under the [MIT License](LICENSE).

The pretrained Stable Diffusion model is a separate dependency and is governed by its own model license and terms.

## ⚠️ Responsible use

Generated images can contain errors, artifacts, or unintended content. Review outputs before publishing or distributing them. Respect privacy, copyright, model terms, platform policies, and applicable laws.

## 👤 Maintainer

**Alok Agarwal**

- GitHub: [@mightyalok00](https://github.com/mightyalok00)
- Project: [genai-text-to-image-studio](https://github.com/mightyalok00/genai-text-to-image-studio)
