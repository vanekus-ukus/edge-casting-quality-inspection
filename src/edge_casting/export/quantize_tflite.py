from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .export_tflite import build_keras_micro_model


def representative_dataset(image_paths: list[Path], input_size: int):
    def gen():
        for path in image_paths[:100]:
            image = Image.open(path).convert("RGB").resize((input_size, input_size))
            arr = np.asarray(image, dtype=np.float32)[None, ...]
            yield [arr]

    return gen


def quantize_micro_edge_int8(
    output_path: str | Path,
    train_dir: str | Path,
    input_size: int = 96,
    keras_weights: str | Path | None = None,
) -> Path:
    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError("TensorFlow is required for INT8 quantization. Install with: pip install '.[tflite]'") from exc

    train_path = Path(train_dir)
    image_paths = [p for p in train_path.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    if not image_paths:
        raise FileNotFoundError("Representative images not found. Run make prepare-data first.")

    model = build_keras_micro_model(input_size)
    if keras_weights:
        model.load_weights(str(keras_weights))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset(image_paths, input_size)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(converter.convert())
    return out
