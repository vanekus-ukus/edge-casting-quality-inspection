from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from edge_casting.models.registry import build_model
from edge_casting.paths import ensure_dir, project_path
from edge_casting.training.evaluate import evaluate_model, make_loader
from edge_casting.utils.metrics import save_json
from edge_casting.utils.seed import set_seed


@torch.no_grad()
def prediction_rows(model: torch.nn.Module, loader, device: torch.device) -> list[dict]:
    rows = []
    model.eval()
    for images, labels in loader:
        probs = torch.softmax(model(images.to(device)), dim=1)[:, 1].detach().cpu()
        for label, prob in zip(labels.tolist(), probs.tolist()):
            pred = int(prob >= 0.5)
            rows.append({
                "y_true": int(label),
                "y_pred": pred,
                "prob_defective": float(prob),
                "confidence": float(max(prob, 1.0 - prob)),
            })
    return rows


def train_profile(
    profile: str,
    output_dir: str | Path | None = None,
    metrics_dir: str | Path | None = None,
    seed: int | None = None,
    epochs: int | None = None,
    device_override: str | None = None,
) -> Path:
    dataset_cfg = yaml.safe_load(project_path("configs", "dataset.yaml").read_text(encoding="utf-8"))["dataset"]
    models_cfg = yaml.safe_load(project_path("configs", "models.yaml").read_text(encoding="utf-8"))["profiles"][profile]
    exp_cfg = yaml.safe_load(project_path("configs", "experiments.yaml").read_text(encoding="utf-8"))["experiment"]
    set_seed(int(seed if seed is not None else exp_cfg["seed"]))

    device_name = str(device_override or exp_cfg["device"])
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Config requests CUDA, but torch.cuda.is_available() is False in this process.")
        device = torch.device(device_name)
    else:
        device = torch.device(device_name)
    processed = project_path(dataset_cfg["processed_dir"])
    if not (processed / "train").exists():
        raise FileNotFoundError("Prepared dataset not found. Run make prepare-data first.")

    train_loader = make_loader(processed / "train", models_cfg["input_size"], models_cfg["batch_size"], exp_cfg["num_workers"], True)
    val_loader = make_loader(processed / "val", models_cfg["input_size"], models_cfg["batch_size"], exp_cfg["num_workers"], False)
    test_loader = make_loader(processed / "test", models_cfg["input_size"], models_cfg["batch_size"], exp_cfg["num_workers"], False)

    run_cfg = dict(models_cfg)
    if epochs is not None:
        run_cfg["epochs"] = int(epochs)
    model = build_model(run_cfg["model_name"], pretrained=bool(run_cfg["pretrained"])).to(device)
    optimizer = AdamW(model.parameters(), lr=float(models_cfg["learning_rate"]))
    labels = [label for _, label in train_loader.dataset.samples]
    counts = torch.bincount(torch.tensor(labels), minlength=2).float()
    weights = (counts.sum() / counts.clamp_min(1.0)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    history: list[dict] = []
    best_f1 = -1.0
    out_dir = ensure_dir(Path(output_dir) if output_dir else project_path(exp_cfg["output_dir"], profile))
    best_path = out_dir / "best.pt"

    for epoch in range(int(run_cfg["epochs"])):
        model.train()
        losses: list[float] = []
        for images, labels in tqdm(train_loader, desc=f"{profile} epoch {epoch + 1}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_metrics = evaluate_model(model, val_loader, device)
        row = {"epoch": epoch + 1, "train_loss": sum(losses) / max(1, len(losses)), **val_metrics}
        history.append(row)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save({"model_state": model.state_dict(), "profile": profile, "model_config": run_cfg}, best_path)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate_model(model, test_loader, device)
    metrics_path = ensure_dir(Path(metrics_dir) if metrics_dir else project_path(exp_cfg["metrics_dir"], profile))
    save_json({"profile": profile, "seed": seed, "model_config": run_cfg, **test_metrics}, metrics_path / "metrics.json")
    save_json({"history": history}, metrics_path / "history.json")
    save_json({"confusion_matrix": test_metrics["confusion_matrix"]}, metrics_path / "confusion_matrix.json")
    save_json(test_metrics["classification_report"], metrics_path / "classification_report.json")
    pd.DataFrame(test_metrics["confusion_matrix"], index=["ok", "defective"], columns=["ok", "defective"]).to_csv(metrics_path / "confusion_matrix.csv")
    pd.DataFrame(test_metrics["classification_report"]).T.to_csv(metrics_path / "classification_report.csv")
    pd.DataFrame(prediction_rows(model, test_loader, device)).to_csv(metrics_path / "predictions.csv", index=False)
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=["baseline_pc", "edge_sbc", "micro_edge"])
    parser.add_argument("--output-dir")
    parser.add_argument("--metrics-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    print(f"Saved checkpoint: {train_profile(args.profile, args.output_dir, args.metrics_dir, args.seed, args.epochs, args.device)}")


if __name__ == "__main__":
    main()
