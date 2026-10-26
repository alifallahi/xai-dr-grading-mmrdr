"""Model construction (timm) and access to layers needed for explanations."""

from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn


def build_model(cfg: dict, num_classes: int = 5, pretrained: bool | None = None) -> nn.Module:
    """timm model with a fresh ``num_classes`` head. ``num_classes=0`` returns a pooled feature extractor."""
    m = cfg["model"]
    return timm.create_model(
        m["name"],
        pretrained=m.get("pretrained", True) if pretrained is None else pretrained,
        num_classes=num_classes,
        drop_rate=m.get("drop_rate", 0.0),
    )


def gradcam_layer(model: nn.Module) -> nn.Module:
    """Last convolutional stage (spatial feature map) for Grad-CAM."""
    for attr in ("layer4", "stages", "conv_head", "features"):
        if hasattr(model, attr):
            layer = getattr(model, attr)
            if isinstance(layer, (nn.Sequential, nn.ModuleList)) and attr in ("stages", "features"):
                return layer[-1]
            return layer
    raise ValueError(f"Do not know the Grad-CAM layer for {type(model).__name__}; add it to gradcam_layer()")


def randomisation_stages(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Modules re-initialised top-down in the cascading model-randomisation sanity check."""
    if all(hasattr(model, f"layer{i}") for i in range(1, 5)):
        return [("fc", model.get_classifier()), ("layer4", model.layer4), ("layer3", model.layer3),
                ("layer2", model.layer2), ("layer1", model.layer1)]
    if hasattr(model, "stages"):
        stages = [(f"stage{i}", s) for i, s in enumerate(model.stages)][::-1]
        return [("head", model.get_classifier())] + stages
    raise ValueError("randomisation_stages() supports ResNet- and ConvNeXt-style timm models")


@torch.no_grad()
def pooled_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Penultimate (globally pooled) features of a timm classifier."""
    return model.forward_head(model.forward_features(x), pre_logits=True)


def save_checkpoint(path: str | Path, model: nn.Module, cfg: dict, epoch: int, val_metrics: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "config": cfg, "epoch": epoch, "val_metrics": val_metrics}, path)


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[nn.Module, dict, dict]:
    """Rebuild the model from the config stored in the checkpoint (no pretrained download needed)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["config"], ckpt
