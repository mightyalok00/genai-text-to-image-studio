# Deployment

## Current status

The project is currently **local-first**.

The previous Streamlit Community Cloud deployment has been removed. No public hosted demo is maintained by this repository at this time.

The recommended development workflow is to run the application locally on hardware with sufficient CPU RAM and, ideally, an NVIDIA GPU with enough VRAM for the selected Stable Diffusion configuration.

## Local execution

Create and activate a virtual environment, install a compatible PyTorch build, then install the application dependencies:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The local interface is normally available at:

```text
http://localhost:8501
```

For CUDA-enabled development, verify the GPU before starting the application:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## GPU requirements

Stable Diffusion v1.5 is substantially more resource-intensive than a typical CPU-only application.

For a GPU with approximately 6 GB VRAM, begin with:

| Setting | Starting point |
|---|---:|
| Resolution | 512 × 512 |
| Inference steps | 20–30 |
| Batch size | 1 |
| Precision | FP16 |
| Offloading | Enabled |

Higher resolutions and inference steps can increase VRAM usage and generation time.

## Environment variables

The application respects:

- `HF_HOME`
- `HF_HUB_CACHE`

These can be used to move model downloads to a larger drive.

No application API key is required for the current local inference workflow.

## If public hosting is added later

A future hosted deployment should be treated as a separate deployment target rather than assuming that a general-purpose CPU host can run the current model efficiently.

Before deploying:

- [ ] Confirm CPU RAM and GPU VRAM capacity.
- [ ] Confirm the target's PyTorch/CUDA compatibility.
- [ ] Confirm model access and licensing requirements.
- [ ] Do not upload model weights to Git.
- [ ] Do not commit secrets.
- [ ] Test a 512×512 generation first.
- [ ] Configure model caching where supported.
- [ ] Review generated-content and platform policies.
- [ ] Record the deployment environment for reproducibility.

## Containerization

The existing local application can be containerized around the Streamlit entry point:

```text
streamlit run app.py --server.address=0.0.0.0 --server.port=<PORT>
```

A production container should install a PyTorch build appropriate for the target GPU/runtime rather than assuming the developer's local CUDA environment.
