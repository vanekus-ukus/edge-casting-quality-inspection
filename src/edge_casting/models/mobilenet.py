from __future__ import annotations

from torch import nn
from torchvision import models


def mobilenet_v2(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def mobilenet_v3_small(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model
