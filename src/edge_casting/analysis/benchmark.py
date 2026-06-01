from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from edge_casting.models.registry import build_model
from edge_casting.paths import project_path
from edge_casting.utils.metrics import count_parameters


def file_size_mb(path: Path) -> float | None:
    return round(path.stat().st_size / (1024 * 1024), 4) if path.exists() else None


def benchmark_profile(profile: str) -> dict:
    models_cfg = yaml.safe_load(project_path("configs", "models.yaml").read_text(encoding="utf-8"))["profiles"][profile]
    bench_cfg = yaml.safe_load(project_path("configs", "experiments.yaml").read_text(encoding="utf-8"))["benchmark"]
    model = build_model(models_cfg["model_name"], pretrained=False).eval()
    checkpoint = project_path("models_trained", profile, "best.pt")
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model_state"])
    x = torch.randn(1, 3, int(models_cfg["input_size"]), int(models_cfg["input_size"]))
    with torch.no_grad():
        for _ in range(int(bench_cfg["warmup_runs"])):
            _ = model(x)
        start = time.perf_counter()
        for _ in range(int(bench_cfg["timed_runs"])):
            _ = model(x)
        elapsed = time.perf_counter() - start
    latency_ms = elapsed / int(bench_cfg["timed_runs"]) * 1000.0
    tflite_path = project_path("models_exported", "micro_edge_int8.tflite")
    return {
        "profile": profile,
        "model_name": models_cfg["model_name"],
        "input_size": models_cfg["input_size"],
        "format": "pytorch",
        "params": count_parameters(model),
        "model_size_mb": file_size_mb(checkpoint) if checkpoint.exists() else None,
        "latency_ms": round(latency_ms, 3),
        "fps": round(1000.0 / latency_ms, 2) if latency_ms > 0 else None,
        "tflite_size_mb": file_size_mb(tflite_path) if profile == "micro_edge" else None,
        "notes": bench_cfg["profiles"][profile]["note"],
    }


def benchmark_tflite(model_path: Path, label: str, input_size: int) -> dict:
    bench_cfg = yaml.safe_load(project_path("configs", "experiments.yaml").read_text(encoding="utf-8"))["benchmark"]
    row = {
        "profile": label,
        "model_name": "small_cnn",
        "input_size": input_size,
        "format": label.replace("micro_edge_", ""),
        "params": None,
        "model_size_mb": file_size_mb(model_path),
        "latency_ms": None,
        "fps": None,
        "tflite_size_mb": file_size_mb(model_path),
        "notes": "TensorFlow Lite proxy timing on current workstation.",
    }
    if not model_path.exists():
        row["notes"] = "TFLite file not found."
        return row
    try:
        import numpy as np
        import tensorflow as tf
    except Exception:
        row["notes"] = "TensorFlow not available; size only."
        return row

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    shape = input_info["shape"]
    dtype = input_info["dtype"]
    if dtype == np.float32:
        x = np.random.rand(*shape).astype(np.float32) * 255.0
    else:
        low, high = (0, 255) if dtype == np.uint8 else (-128, 127)
        x = np.random.randint(low, high + 1, size=shape, dtype=dtype)
    for _ in range(int(bench_cfg["warmup_runs"])):
        interpreter.set_tensor(input_info["index"], x)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_info["index"])
    start = time.perf_counter()
    for _ in range(int(bench_cfg["timed_runs"])):
        interpreter.set_tensor(input_info["index"], x)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_info["index"])
    latency_ms = (time.perf_counter() - start) / int(bench_cfg["timed_runs"]) * 1000.0
    row["latency_ms"] = round(latency_ms, 3)
    row["fps"] = round(1000.0 / latency_ms, 2) if latency_ms > 0 else None
    return row


def run_benchmarks(profiles: list[str] | None = None) -> Path:
    profiles = profiles or ["baseline_pc", "edge_sbc", "micro_edge"]
    rows = [benchmark_profile(profile) for profile in profiles]
    micro_cfg = yaml.safe_load(project_path("configs", "models.yaml").read_text(encoding="utf-8"))["profiles"]["micro_edge"]
    input_size = int(micro_cfg["input_size"])
    rows.extend([
        benchmark_tflite(project_path("models_exported", "micro_edge_fp32.tflite"), "micro_edge_tflite_fp32", input_size),
        benchmark_tflite(project_path("models_exported", "micro_edge_int8.tflite"), "micro_edge_tflite_int8", input_size),
    ])
    out = project_path("reports", "tables", "edge_benchmark.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out
