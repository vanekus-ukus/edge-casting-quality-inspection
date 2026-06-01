from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import yaml

from edge_casting.paths import project_path
from edge_casting.training.evaluate import load_checkpoint_model, make_loader


@torch.no_grad()
def collect_predictions(profile: str) -> tuple[list[int], list[float]]:
    models_cfg = yaml.safe_load(project_path("configs", "models.yaml").read_text(encoding="utf-8"))["profiles"][profile]
    dataset_cfg = yaml.safe_load(project_path("configs", "dataset.yaml").read_text(encoding="utf-8"))["dataset"]
    checkpoint = project_path("models_trained", profile, "best.pt")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Train the model first.")
    model = load_checkpoint_model(checkpoint, models_cfg["model_name"], pretrained=False, device="cpu").eval()
    loader = make_loader(project_path(dataset_cfg["processed_dir"], "test"), models_cfg["input_size"], 32)
    y_true: list[int] = []
    y_score: list[float] = []
    for images, labels in loader:
        probs = torch.softmax(model(images), dim=1)[:, 1]
        y_true.extend(labels.tolist())
        y_score.extend(probs.tolist())
    return y_true, y_score


def triage_sweep(profile: str | list[str] = "micro_edge") -> Path:
    thresholds = yaml.safe_load(project_path("configs", "experiments.yaml").read_text(encoding="utf-8"))["triage"]["thresholds"]
    profiles = profile if isinstance(profile, list) else [profile]
    rows = []
    for one_profile in profiles:
        y_true, y_score = collect_predictions(one_profile)
        for threshold in thresholds:
            accepted = [max(p, 1.0 - p) >= threshold for p in y_score]
            pred = [1 if p >= 0.5 else 0 for p in y_score]
            accepted_idx = [i for i, flag in enumerate(accepted) if flag]
            accepted_correct = sum(pred[i] == y_true[i] for i in accepted_idx)
            accepted_defective = [i for i in accepted_idx if y_true[i] == 1]
            accepted_defective_hits = sum(pred[i] == 1 for i in accepted_defective)
            unsafe_errors = sum(1 for i in accepted_idx if y_true[i] == 1 and pred[i] == 0)
            rows.append({
                "profile": one_profile,
                "threshold": threshold,
                "coverage": len(accepted_idx) / max(1, len(y_true)),
                "review_rate": 1.0 - len(accepted_idx) / max(1, len(y_true)),
                "accepted_accuracy": accepted_correct / max(1, len(accepted_idx)),
                "accepted_defective_recall": accepted_defective_hits / max(1, len(accepted_defective)),
                "unsafe_errors": unsafe_errors,
            })
    out = project_path("reports", "tables", "confidence_triage.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="micro_edge")
    args = parser.parse_args()
    profiles = ["baseline_pc", "edge_sbc", "micro_edge"] if args.profile == "all" else args.profile
    print(triage_sweep(profiles))


if __name__ == "__main__":
    main()
