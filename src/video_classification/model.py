from __future__ import annotations

import torch
from torch import nn
from torchvision.models.video import R3D_18_Weights, r3d_18


def build_r3d18(num_classes: int, pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    if num_classes < 2:
        raise ValueError("At least two classes are required")
    model = r3d_18(weights=R3D_18_Weights.DEFAULT if pretrained else None)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, classes: list[str], config: dict[str, object]) -> None:
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                "classes": classes, "config": config}, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint
