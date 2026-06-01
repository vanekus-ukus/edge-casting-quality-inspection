from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from edge_casting.paths import project_path
from edge_casting.utils.metrics import load_json


def collect_results() -> list[Path]:
    profiles_cfg = yaml.safe_load(project_path("configs", "models.yaml").read_text(encoding="utf-8"))["profiles"]
    rows = []
    bench_path = project_path("reports", "tables", "edge_benchmark.csv")
    bench = pd.read_csv(bench_path) if bench_path.exists() else pd.DataFrame()
    for profile, cfg in profiles_cfg.items():
        metrics_path = project_path("reports", "metrics", profile, "metrics.json")
        metrics = load_json(metrics_path) if metrics_path.exists() else {}
        bench_row = bench[bench["profile"] == profile].iloc[0].to_dict() if not bench.empty and profile in set(bench["profile"]) else {}
        rows.append({
            "profile": profile,
            "model_name": cfg["model_name"],
            "input_size": cfg["input_size"],
            "format": bench_row.get("format", "pytorch"),
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "defective_recall": metrics.get("defective_recall"),
            "model_size_mb": bench_row.get("model_size_mb"),
            "params": bench_row.get("params"),
            "latency_ms": bench_row.get("latency_ms"),
            "fps": bench_row.get("fps"),
            "tflite_size_mb": bench_row.get("tflite_size_mb"),
            "notes": cfg.get("notes"),
        })
    out_dir = project_path("reports", "tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    model_comparison = out_dir / "model_comparison.csv"
    pd.DataFrame(rows).to_csv(model_comparison, index=False)
    outputs.append(model_comparison)
    for src_name, dst_name in [
        ("confidence_triage.csv", "triage_summary.csv"),
        ("robustness.csv", "robustness_summary.csv"),
    ]:
        src = out_dir / src_name
        dst = out_dir / dst_name
        if src.exists():
            pd.read_csv(src).to_csv(dst, index=False)
            outputs.append(dst)
    if bench_path.exists():
        outputs.append(bench_path)
    return outputs
