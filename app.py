"""GenAI Text-to-Image Studio Streamlit application."""

from __future__ import annotations

import os

import streamlit as st
import torch

from src.generation import generate_image
from src.pipeline import (
    configure_huggingface_cache,
    get_device,
    load_pipeline,
)
from src.utils import image_to_bytes


# ---------- Page configuration ----------

st.set_page_config(
    page_title="GenAI Text-to-Image Studio",
    page_icon="🎨",
    layout="wide",
)


# ---------- Hugging Face cache ----------

# WHY: Configure the D: drive before any model download occurs.
configure_huggingface_cache()


# ---------- Cached model ----------

@st.cache_resource(show_spinner=False)
def get_model():
    """Load the model once and reuse it across Streamlit reruns."""
    return load_pipeline()


# ---------- Header ----------

st.title("🎨 GenAI Text-to-Image Studio")
st.caption("Local Stable Diffusion • PyTorch • Diffusers • Streamlit")


# ---------- Sidebar ----------

with st.sidebar:
    st.header("Generation Settings")

    steps = st.slider(
        "Inference steps",
        min_value=10,
        max_value=50,
        value=30,
        help="More steps usually improve refinement but increase runtime.",
    )

    guidance = st.slider(
        "CFG / Guidance scale",
        min_value=1.0,
        max_value=15.0,
        value=7.5,
        step=0.5,
        help="Controls how strongly the image follows the prompt.",
    )

    seed = st.number_input(
        "Seed (-1 = random)",
        min_value=-1,
        max_value=2_147_483_647,
        value=42,
        step=1,
        help="A fixed seed makes generation reproducible.",
    )

    resolution = st.selectbox(
        "Resolution",
        [(512, 512), (512, 768), (768, 512)],
        format_func=lambda value: f"{value[0]} × {value[1]}",
    )

    st.divider()
    st.write(f"**Device:** `{get_device()}`")
    st.write(f"**HF cache:** `{os.environ.get('HF_HOME')}`")


# ---------- Prompt inputs ----------

prompt = st.text_area(
    "Prompt",
    placeholder=(
        "A cinematic portrait of a young South Asian man, "
        "golden-hour lighting, realistic photography, highly detailed"
    ),
    height=120,
)

negative_prompt = st.text_area(
    "Negative prompt",
    value="blurry, low quality, distorted, deformed, bad anatomy",
    height=80,
)


# ---------- Generate ----------

generate = st.button(
    "✨ Generate Image",
    type="primary",
    width="stretch",
)


if generate:
    if not prompt.strip():
        st.warning("Enter a prompt before generating.")
        st.stop()

    try:
        with st.spinner("Loading model and generating image..."):
            image = generate_image(
                pipe=get_model(),
                prompt=prompt.strip(),
                negative_prompt=negative_prompt.strip(),
                steps=steps,
                guidance_scale=guidance,
                seed=int(seed),
                width=resolution[0],
                height=resolution[1],
            )

        st.session_state["image"] = image
        st.session_state["prompt"] = prompt

    except torch.cuda.OutOfMemoryError:
        st.error(
            "GPU memory is insufficient. Try 512×512 and "
            "20–25 inference steps."
        )

    except Exception as exc:
        st.exception(exc)


# ---------- Result ----------

if "image" in st.session_state:
    st.subheader("Generated Image")

    st.image(
        st.session_state["image"],
        caption=st.session_state.get("prompt", ""),
        width="stretch",
    )

    st.download_button(
        "⬇️ Download PNG",
        data=image_to_bytes(st.session_state["image"]),
        file_name="generated_image.png",
        mime="image/png",
        width="stretch",
    )
