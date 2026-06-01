from __future__ import annotations

import yaml

from edge_casting.export.export_tflite import export_micro_edge_fp32, train_keras_micro_model
from edge_casting.export.quantize_tflite import quantize_micro_edge_int8
from edge_casting.paths import project_path

models_cfg = yaml.safe_load(project_path("configs", "models.yaml").read_text(encoding="utf-8"))["profiles"]["micro_edge"]
dataset_cfg = yaml.safe_load(project_path("configs", "dataset.yaml").read_text(encoding="utf-8"))["dataset"]
input_size = int(models_cfg["input_size"])
processed = project_path(dataset_cfg["processed_dir"])
weights = project_path("models_exported", "micro_edge_tf.weights.h5")
if not weights.exists():
    print(train_keras_micro_model(processed / "train", processed / "val", weights, input_size=input_size, epochs=5))
print(export_micro_edge_fp32(project_path("models_exported", "micro_edge_fp32.tflite"), input_size, weights))
print(quantize_micro_edge_int8(project_path("models_exported", "micro_edge_int8.tflite"), processed / "train", input_size, weights))
