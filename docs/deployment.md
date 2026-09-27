# Deployment

## Local deployment

The supported baseline is local execution with Streamlit.

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

A compatible PyTorch installation should be configured before installing the application dependencies.

## GPU deployment

Stable Diffusion v1.5 is substantially more resource-intensive than a typical CPU-only Streamlit application. A deployment target should therefore provide enough RAM/VRAM for the selected model, resolution, and inference settings.

For GPU hosting:

1. Install a compatible CUDA-enabled PyTorch build.
2. Install the packages in `requirements.txt`.
3. Make sure the runtime can access the Hugging Face model.
4. Expose the Streamlit port used by the hosting platform.
5. Cache the model so it is not downloaded on every request.

## Environment variables

The application respects:

- `HF_HOME`
- `HF_HUB_CACHE`

No application API key is required for the current local inference workflow.

## Deployment checklist

Before deploying:

- [ ] Confirm the target has sufficient CPU RAM.
- [ ] Confirm the target has sufficient GPU VRAM if using CUDA.
- [ ] Confirm model access and licensing requirements.
- [ ] Do not upload model weights to the Git repository.
- [ ] Do not commit secrets.
- [ ] Test a 512×512 generation first.
- [ ] Configure persistent caching where supported.
- [ ] Review generated-content and platform policies.

## Containerization

The application can be containerized around the existing Streamlit entry point:

```text
streamlit run app.py --server.address=0.0.0.0 --server.port=<PORT>
```

A production container should install a PyTorch build appropriate for the target GPU/runtime rather than assuming the developer's local CUDA environment.
