from __future__ import annotations

from torch import nn

from .mobilenet import mobilenet_v2, mobilenet_v3_small
from .small_cnn import SmallCNN


def build_model(model_name: str, num_classes: int = 2, pretrained: bool = False) -> nn.Module:
    if model_name == "small_cnn":
        return SmallCNN(num_classes=num_classes)
    if model_name == "mobilenet_v2":
        return mobilenet_v2(num_classes=num_classes, pretrained=pretrained)
    if model_name == "mobilenet_v3_small":
        return mobilenet_v3_small(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown model_name: {model_name}")
