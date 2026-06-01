from __future__ import annotations

from PIL import Image, ImageEnhance, ImageFilter
import numpy as np


def apply_distortion(image: Image.Image, kind: str, value: float) -> Image.Image:
    if kind == "brightness":
        return ImageEnhance.Brightness(image).enhance(value)
    if kind == "contrast":
        return ImageEnhance.Contrast(image).enhance(value)
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=value))
    if kind == "noise":
        arr = np.asarray(image).astype("float32") / 255.0
        noise = np.random.normal(0.0, value, arr.shape).astype("float32")
        return Image.fromarray(np.clip((arr + noise) * 255.0, 0, 255).astype("uint8"))
    raise ValueError(f"Unknown distortion kind: {kind}")
