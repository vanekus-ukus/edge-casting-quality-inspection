from __future__ import annotations

import yaml

from edge_casting.data.download_kaggle import download_dataset
from edge_casting.paths import project_path

cfg = yaml.safe_load(project_path("configs", "dataset.yaml").read_text(encoding="utf-8"))["dataset"]
download_dataset(cfg["kaggle_slug"], project_path(cfg["raw_dir"]))
print(f"Dataset downloaded to {cfg['raw_dir']}")
