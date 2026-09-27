# Architecture

## Overview

GenAI Text-to-Image Studio separates the user interface, model lifecycle, generation logic, and small utility functions.

## Components

### `app.py`

The Streamlit entry point.

Responsibilities:

- Page configuration
- User controls
- Prompt and negative-prompt input
- Model resource caching
- Generation requests
- Error handling
- Image preview and PNG download

### `src/pipeline.py`

Owns model and hardware configuration.

Responsibilities:

- Define the pretrained model ID
- Configure Hugging Face caching
- Detect CUDA versus CPU
- Select FP16 for CUDA and FP32 for CPU
- Load the Diffusers pipeline
- Enable Accelerate CPU offloading on CUDA

### `src/generation.py`

Contains the reusable inference function.

Responsibilities:

- Handle fixed or random seeds
- Create the appropriate PyTorch generator
- Pass prompt and generation parameters to the pipeline
- Return the generated PIL image

### `src/utils.py`

Contains small application utilities.

Currently it converts a PIL image into PNG bytes for Streamlit downloads.

### Notebook

`notebooks/text_to_image_experiments.ipynb` provides an interactive view of the same local inference workflow and is useful for experimentation and debugging.

## Data flow

```text
Prompt + settings
      │
      ▼
Streamlit app.py
      │
      ▼
generate_image()
      │
      ▼
StableDiffusionPipeline
      │
      ├── Text conditioning
      ├── Diffusion denoising
      └── VAE decoding
      │
      ▼
PIL Image
      │
      ├── Streamlit preview
      └── PNG bytes
```

## Design goals

- Keep UI code separate from inference code.
- Keep model initialization in one place.
- Avoid hard-coded personal filesystem paths.
- Avoid storing model weights in Git.
- Keep local GPU setup separate from portable application dependencies.
- Make the inference function reusable outside Streamlit.
