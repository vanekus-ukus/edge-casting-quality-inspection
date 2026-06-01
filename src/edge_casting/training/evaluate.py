from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from edge_casting.models.registry import build_model
from edge_casting.utils.metrics import binary_metrics


class CastingImageFolder(datasets.ImageFolder):
    def find_classes(self, directory: str):
        classes = ["ok", "defective"]
        class_to_idx = {"ok": 0, "defective": 1}
        return classes, class_to_idx


def make_loader(data_dir: str | Path, input_size: int, batch_size: int, num_workers: int = 0, shuffle: bool = False) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = CastingImageFolder(str(data_dir), transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu()
        preds = (probs >= 0.5).long()
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())
        y_score.extend(probs.tolist())
    return binary_metrics(y_true, y_pred, y_score)


def load_checkpoint_model(checkpoint: str | Path, model_name: str, pretrained: bool = False, device: str = "cpu") -> torch.nn.Module:
    model = build_model(model_name, pretrained=pretrained)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    return model.to(device)
