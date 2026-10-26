"""Grad-CAM and Integrated Gradients, implemented directly so their behaviour is transparent.

Grad-CAM (Selvaraju et al., 2017): class-specific weighting of the last convolutional
feature map. Coarse (16 x 16 at 512 px for ResNet-50), so it shows *regions*, not lesions.

Integrated Gradients (Sundararajan et al., 2017): path integral of input gradients from a
baseline to the image. Pixel-level, satisfies completeness (attributions sum to the logit
difference between image and baseline). The baseline is a heavily blurred copy of the image
(Sturmfels et al., 2020): it keeps global colour and illumination but removes small
structures such as microaneurysms and haemorrhages, so attribution concentrates on the
fine detail the grade depends on. A black baseline would assign near-zero attribution to
dark lesions (haemorrhages) because attribution scales with (image - baseline).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model = model
        self._acts: torch.Tensor | None = None
        self._handle = layer.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        self._acts = output

    def remove(self) -> None:
        self._handle.remove()

    def __call__(self, x: torch.Tensor, target: torch.Tensor | int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cams [B,H,W], non-negative, *not* normalised; logits [B,K])."""
        self.model.eval()
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)  # builds the graph even if model weights are frozen
            logits = self.model(x)
            if target is None:
                target = logits.argmax(1)
            elif isinstance(target, int):
                target = torch.full((x.shape[0],), target, device=x.device, dtype=torch.long)
            score = logits.gather(1, target.view(-1, 1)).sum()
            grads = torch.autograd.grad(score, self._acts)[0]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        return cam.detach(), logits.detach()


def blurred_baseline(rgb: np.ndarray, sigma: float, fov: np.ndarray | None = None) -> np.ndarray:
    """Heavily blurred copy of the image, computed only from pixels inside the field of view.

    A plain Gaussian blur mixes the black surround into the retina near its edge, so the baseline
    is darker there and (image - baseline) is about three times larger on the FOV rim than inside.
    IG then attributes to the rim whatever the model does. Normalised convolution
    blur(x * m) / blur(m) avoids this: each baseline pixel is a local average of retina only.
    """
    m = (rgb.max(2) > 0) if fov is None else fov.astype(bool)
    mf = m.astype(np.float32)
    num = cv2.GaussianBlur(rgb.astype(np.float32) * mf[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(mf, (0, 0), sigma)[..., None]
    out = num / np.maximum(den, 1e-3)
    out[~m] = 0
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def integrated_gradients(
    model: nn.Module, x: torch.Tensor, baseline: torch.Tensor, target: int, steps: int = 32, chunk: int = 8
) -> tuple[torch.Tensor, float]:
    """IG for one image (1xCxHxW). Returns (attributions CxHxW, completeness gap).

    The completeness gap = sum(attributions) - (f(x) - f(baseline)), relative to |f(x) - f(baseline)|;
    a small gap means enough integration steps were used.
    """
    model.eval()
    alphas = (torch.arange(steps, device=x.device, dtype=x.dtype) + 0.5) / steps  # midpoint rule
    diff = x - baseline
    total = torch.zeros_like(x)
    for start in range(0, steps, chunk):
        a = alphas[start:start + chunk].view(-1, 1, 1, 1)
        xs = (baseline + a * diff).detach().requires_grad_(True)
        with torch.enable_grad():
            out = model(xs)[:, target].sum()
            g = torch.autograd.grad(out, xs)[0]
        total += g.sum(0, keepdim=True)
    attr = (diff * total / steps)[0]
    with torch.no_grad():
        fx = model(x)[0, target].item()
        fb = model(baseline)[0, target].item()
    gap = (attr.sum().item() - (fx - fb)) / (abs(fx - fb) + 1e-8)
    return attr.detach(), float(gap)


def ig_saliency(attr: torch.Tensor, smooth_sigma: float = 2.0) -> np.ndarray:
    """Positive evidence map from IG: channel-summed attribution, positive part, lightly smoothed."""
    m = attr.sum(0).clamp(min=0).cpu().numpy().astype(np.float32)
    return cv2.GaussianBlur(m, (0, 0), smooth_sigma) if smooth_sigma > 0 else m


def normalise_map(m: np.ndarray, pct: float = 99.5) -> np.ndarray:
    """Scale a non-negative map to [0, 1], clipping at a high percentile for display."""
    top = np.percentile(m, pct)
    return np.clip(m / top, 0, 1) if top > 0 else np.zeros_like(m)
