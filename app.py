"""GenAI Text-to-Image Studio Streamlit application."""

from __future__ import annotations

import os
import random
import time
from datetime import datetime

import streamlit as st
import torch

from src.generation import generate_image
from src.pipeline import configure_huggingface_cache, get_device, load_pipeline
from src.utils import format_duration, format_vram, image_to_bytes, validate_generation_settings

st.set_page_config(
    page_title="GenAI Text-to-Image Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

configure_huggingface_cache()

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    .hero {
        padding: 1.4rem 1.6rem;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 18px;
        margin-bottom: 1.2rem;
    }
    .hero h1 {margin-bottom: .25rem;}
    .muted {opacity: .72; font-size: .88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner=False)
def get_model():
    """Load the model once and reuse it across Streamlit reruns."""
    return load_pipeline()

def init_state() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("prompt_input", "")
    st.session_state.setdefault(
        "negative_prompt_input",
        "blurry, low quality, distorted, deformed, bad anatomy",
    )

init_state()

PROMPT_PRESETS = [
    "A cinematic futuristic city at sunset, volumetric lighting, ultra detailed",
    "A photorealistic mountain landscape at sunrise, dramatic clouds, 35mm photography",
    "A luxury product photograph on a studio set, soft lighting, sharp details",
    "A fantasy castle above the clouds, cinematic composition, atmospheric perspective",
]

def runtime_info() -> tuple[str, str]:
    device = get_device()
    if device == "cuda":
        return device, torch.cuda.get_device_name(0)
    return device, "CPU"

def generate() -> None:
    prompt = st.session_state.prompt_input.strip()
    negative = st.session_state.negative_prompt_input.strip()
    steps = int(st.session_state.steps)
    guidance = float(st.session_state.guidance)
    seed = int(st.session_state.seed)
    width, height = st.session_state.resolution

    error = validate_generation_settings(
        prompt=prompt,
        steps=steps,
        guidance_scale=guidance,
        width=width,
        height=height,
    )
    if error:
        st.error(error)
        return

    try:
        pipe = get_model()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        with st.spinner("Generating your image…"):
            image = generate_image(
                pipe=pipe,
                prompt=prompt,
                negative_prompt=negative,
                steps=steps,
                guidance_scale=guidance,
                seed=seed,
                width=width,
                height=height,
            )
        elapsed = time.perf_counter() - start

        peak_vram = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        )
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative,
            "steps": steps,
            "guidance_scale": guidance,
            "seed": seed,
            "width": width,
            "height": height,
            "duration": elapsed,
            "peak_vram": peak_vram,
            "device": runtime_info()[1],
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        st.session_state.history.insert(
            0,
            {"image": image_to_bytes(image), "metadata": metadata},
        )
        st.session_state.history = st.session_state.history[:12]

    except torch.cuda.OutOfMemoryError:
        st.error(
            "GPU memory is insufficient. Try 512 × 512 and 20–25 inference steps."
        )
    except Exception as exc:
        st.exception(exc)

st.markdown(
    """
    <div class="hero">
        <h1>🎨 GenAI Text-to-Image Studio</h1>
        <div class="muted">
            Local Stable Diffusion • PyTorch • Diffusers • CUDA • Streamlit
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

device, device_name = runtime_info()

with st.sidebar:
    st.header("⚙️ Generation")

    steps = st.slider(
        "Inference steps", 10, 50, 30,
        help="Higher values can improve refinement but increase generation time.",
        key="steps",
    )
    guidance = st.slider(
        "CFG / Guidance", 1.0, 15.0, 7.5, 0.5,
        help="How strongly the output follows the prompt.",
        key="guidance",
    )
    seed = st.number_input(
        "Seed (-1 = random)", -1, 2_147_483_647, 42, 1, key="seed",
    )
    resolution = st.selectbox(
        "Resolution",
        [(512, 512), (512, 768), (768, 512)],
        format_func=lambda value: f"{value[0]} × {value[1]}",
        key="resolution",
    )

    with st.expander("🧠 Runtime"):
        st.write(f"**Device:** `{device}`")
        st.write(f"**Hardware:** `{device_name}`")
        st.write(f"**HF cache:** `{os.environ.get('HF_HOME')}`")
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            st.write(f"**VRAM:** `{vram_gb:.1f} GB`")

    if st.button("🎲 Random prompt", width="stretch"):
        st.session_state.prompt_input = random.choice(PROMPT_PRESETS)
        st.rerun()

    if st.button("🗑️ Clear history", width="stretch"):
        st.session_state.history = []
        st.rerun()

left, right = st.columns([1.25, 1], gap="large")

with left:
    st.subheader("✍️ Create")
    st.text_area(
        "Prompt",
        height=140,
        placeholder="Describe the image you want to create…",
        key="prompt_input",
    )
    st.text_area(
        "Negative prompt",
        height=90,
        key="negative_prompt_input",
    )
    if st.button("✨ Generate Image", type="primary", width="stretch"):
        generate()

with right:
    st.subheader("🎛️ Generation profile")
    st.info(
        f"**{resolution[0]} × {resolution[1]}**  •  "
        f"**{steps} steps**  •  **CFG {guidance:g}**  •  **Seed {seed}**"
    )
    st.caption(
        "Fixed seeds make local experiments reproducible. Use -1 for a random result."
    )

if st.session_state.history:
    latest = st.session_state.history[0]
    meta = latest["metadata"]

    st.divider()
    st.subheader("🖼️ Latest generation")
    image_col, metrics_col = st.columns([1.55, 1], gap="large")

    with image_col:
        st.image(latest["image"], width="stretch")
        st.download_button(
            "⬇️ Download PNG",
            data=latest["image"],
            file_name=f"generated_{meta['seed']}_{meta['width']}x{meta['height']}.png",
            mime="image/png",
            width="stretch",
        )

    with metrics_col:
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Generation time", format_duration(meta["duration"]))
        with m2:
            st.metric("Peak VRAM", format_vram(meta["peak_vram"]))

        m3, m4 = st.columns(2)
        with m3:
            st.metric("Steps", meta["steps"])
        with m4:
            st.metric("Seed", meta["seed"])

        st.caption(f"Hardware: {meta['device']}")
        st.caption(f"Created: {meta['created_at']}")
        with st.expander("View generation details"):
            st.write(f"**Prompt:** {meta['prompt']}")
            st.write(f"**Negative:** {meta['negative_prompt']}")
            st.write(f"**Resolution:** {meta['width']} × {meta['height']}")
            st.write(f"**Guidance:** {meta['guidance_scale']}")

if len(st.session_state.history) > 1:
    st.divider()
    st.subheader(f"🕘 Generation history ({len(st.session_state.history)})")

    history_items = st.session_state.history[1:]
    columns = st.columns(3)
    for index, item in enumerate(history_items):
        meta = item["metadata"]
        with columns[index % 3]:
            st.image(item["image"], width="stretch")
            st.caption(
                f"{meta['width']}×{meta['height']} • "
                f"{format_duration(meta['duration'])} • seed {meta['seed']}"
            )
            st.download_button(
                "Download",
                data=item["image"],
                file_name=f"generated_{meta['seed']}_{meta['width']}x{meta['height']}.png",
                mime="image/png",
                key=f"download_{index}",
                width="stretch",
            )
            if st.button("Use prompt", key=f"use_prompt_{index}"):
                st.session_state.prompt_input = meta["prompt"]
                st.session_state.negative_prompt_input = meta["negative_prompt"]
                st.session_state.steps = meta["steps"]
                st.session_state.guidance = meta["guidance_scale"]
                st.session_state.seed = meta["seed"]
                st.session_state.resolution = (meta["width"], meta["height"])
                st.rerun()

st.divider()
st.caption(
    "Local-first GenAI • Stable Diffusion v1.5 • No API key required for local inference"
)
