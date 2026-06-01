from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true: list[int], y_pred: list[int], y_score: list[float] | None = None) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "defective_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["ok", "defective"], zero_division=0, output_dict=True
        ),
    }
    if y_score is not None and len(set(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def save_json(data: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def count_parameters(model: Any) -> int:
    return int(sum(np.prod(tuple(p.shape)) for p in model.parameters()))
