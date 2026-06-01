from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from edge_casting.paths import project_path
from edge_casting.training.evaluate import evaluate_model, load_checkpoint_model
from edge_casting.utils.image_transforms import apply_distortion


class DistortedImageFolder(Dataset):
    def __init__(self, root: Path, input_size: int, kind: str, value: float, max_samples: int | None = None) -> None:
        self.samples = []
        per_class_limit = max_samples // 2 if max_samples is not None else None
        for class_name, label in [("ok", 0), ("defective", 1)]:
            class_samples = []
            for path in (root / class_name).rglob("*"):
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    class_samples.append((path, label))
            self.samples.extend(class_samples[:per_class_limit] if per_class_limit is not None else class_samples)
        self.kind = kind
        self.value = value
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        image = apply_distortion(image, self.kind, self.value)
        return self.transform(image), label


def robustness_sweep(profile: str | list[str] = "micro_edge") -> Path:
    cfg = yaml.safe_load(project_path("configs", "experiments.yaml").read_text(encoding="utf-8"))
    profiles = profile if isinstance(profile, list) else [profile]
    all_models_cfg = yaml.safe_load(project_path("configs", "models.yaml").read_text(encoding="utf-8"))["profiles"]
    dataset_cfg = yaml.safe_load(project_path("configs", "dataset.yaml").read_text(encoding="utf-8"))["dataset"]
    distortions = []
    robust = cfg["robustness"]
    max_samples = int(robust.get("max_samples", 256))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distortions += [("brightness", v) for v in robust["brightness_factors"]]
    distortions += [("contrast", v) for v in robust["contrast_factors"]]
    distortions += [("blur", v) for v in robust["blur_radii"]]
    distortions += [("noise", v) for v in robust["noise_sigmas"]]
    rows = []
    for one_profile in profiles:
        models_cfg = all_models_cfg[one_profile]
        checkpoint = project_path("models_trained", one_profile, "best.pt")
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Train the model first.")
        model = load_checkpoint_model(checkpoint, models_cfg["model_name"], device=str(device)).eval()
        for kind, value in distortions:
            dataset = DistortedImageFolder(project_path(dataset_cfg["processed_dir"], "test"), models_cfg["input_size"], kind, float(value), max_samples)
            metrics = evaluate_model(model, DataLoader(dataset, batch_size=32), device)
            rows.append({
                "profile": one_profile,
                "distortion": kind,
                "value": value,
                **{k: metrics[k] for k in ["accuracy", "precision", "recall", "f1", "defective_recall"]},
                "notes": f"quick subset max_samples={max_samples}",
            })
    out = project_path("reports", "tables", "robustness.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="micro_edge")
    args = parser.parse_args()
    profiles = ["baseline_pc", "edge_sbc", "micro_edge"] if args.profile == "all" else args.profile
    print(robustness_sweep(profiles))


if __name__ == "__main__":
    main()
