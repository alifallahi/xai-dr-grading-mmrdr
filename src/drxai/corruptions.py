"""Synthetic image-quality degradations applied inside the field of view.

MMRDR excluded poor-quality images during curation, so the test set contains few real
examples of blur, poor exposure or compression. These corruptions are a controlled proxy
for the image-quality variation a community setting would produce; they are not a
substitute for real low-quality images.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import cv2
import numpy as np


def _restore_background(out: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out[~mask] = 0
    return out


def gaussian_blur(rgb: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Defocus / motion-free blur. sigma is in pixels of the 512 px image (scaled to the input size)."""
    s = sigma * rgb.shape[0] / 512
    return _restore_background(cv2.GaussianBlur(rgb, (0, 0), s), mask)


def darken(rgb: np.ndarray, mask: np.ndarray, gamma: float) -> np.ndarray:
    """Under-exposure via a gamma curve (gamma > 1 darkens mid-tones)."""
    out = (255.0 * (rgb / 255.0) ** gamma).astype(np.uint8)
    return _restore_background(out, mask)


def low_contrast(rgb: np.ndarray, mask: np.ndarray, factor: float) -> np.ndarray:
    """Compress intensities towards the FOV mean (e.g. media opacity, flash glare)."""
    mean = rgb[mask].mean(0) if mask.any() else rgb.reshape(-1, 3).mean(0)
    out = np.clip(mean + factor * (rgb.astype(np.float32) - mean), 0, 255).astype(np.uint8)
    return _restore_background(out, mask)


def jpeg(rgb: np.ndarray, mask: np.ndarray, quality: int) -> np.ndarray:
    """Re-encode at low JPEG quality (blocking artefacts, loss of fine detail)."""
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    out = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return _restore_background(out, mask)


def corruption_grid(cfg: dict) -> list[tuple[str, int, float, Callable | None]]:
    """[(name, level index, parameter, function)], starting with the uncorrupted reference."""
    rob = cfg["robustness"]
    grid: list[tuple[str, int, float, Callable | None]] = [("none", 0, 0.0, None)]
    for name, fn, key in (("gaussian_blur", gaussian_blur, "sigma"), ("darken", darken, "gamma"),
                          ("low_contrast", low_contrast, "factor"), ("jpeg", jpeg, "quality")):
        cfg_key = {"gaussian_blur": "gaussian_blur", "darken": "darken_gamma",
                   "low_contrast": "low_contrast", "jpeg": "jpeg_quality"}[name]
        for level, value in enumerate(rob.get(cfg_key, []), start=1):
            grid.append((name, level, float(value), partial(fn, **{key: value})))
    return grid
