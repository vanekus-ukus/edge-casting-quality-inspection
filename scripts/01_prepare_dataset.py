from __future__ import annotations

import yaml

from edge_casting.data.prepare_dataset import prepare_dataset
from edge_casting.data.validate_dataset import validate_raw_dataset
from edge_casting.paths import project_path

cfg = yaml.safe_load(project_path("configs", "dataset.yaml").read_text(encoding="utf-8"))["dataset"]
counts = validate_raw_dataset(project_path(cfg["raw_dir"]), cfg["image_extensions"])
print(f"Raw dataset counts: {counts}")
summary = prepare_dataset(project_path(cfg["raw_dir"]), project_path(cfg["processed_dir"]), cfg["image_extensions"], cfg["seed"], cfg["val_fraction"])
print(f"Prepared split summary: {summary}")
