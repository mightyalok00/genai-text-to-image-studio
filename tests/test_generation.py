from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from src.generation import generate_image


@dataclass
class FakeResult:
    images: list[Image.Image]


class FakePipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResult([Image.new("RGB", (kwargs["width"], kwargs["height"]))])


def test_generate_image_passes_generation_parameters():
    pipe = FakePipeline()
    image = generate_image(pipe, "a test image", "blurry", 20, 6.5, 42, 512, 768)
    assert image.size == (512, 768)
    call = pipe.calls[0]
    assert call["prompt"] == "a test image"
    assert call["negative_prompt"] == "blurry"
    assert call["num_inference_steps"] == 20
    assert call["guidance_scale"] == 6.5
    assert call["width"] == 512
    assert call["height"] == 768
    assert call["generator"].initial_seed() == 42


def test_generate_image_uses_none_for_random_seed():
    pipe = FakePipeline()
    generate_image(pipe, "random test", seed=-1)
    assert pipe.calls[0]["generator"] is None
