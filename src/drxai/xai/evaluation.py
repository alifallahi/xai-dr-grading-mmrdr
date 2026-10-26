"""Checks on explanations: faithfulness, sanity, shortcut and localisation metrics.

* Deletion / insertion (Petsiuk et al., 2018): perturb image cells in order of attribution and
  track the predicted-class probability. A faithful map makes the probability fall fast when its
  top cells are removed (low deletion AUC) and rise fast when they are inserted (high insertion
  AUC), relative to a random ordering. Removed cells are replaced by the blurred baseline rather
  than black, to limit out-of-distribution artefacts (they are not eliminated).
* Model randomisation (Adebayo et al., 2018): if a map barely changes when trained weights are
  replaced by random ones, it is not explaining what the model learned.
* Attribution outside the FOV: mass on the black border or rim suggests a shortcut.
* Localisation against pixel-level lesion masks: pointing game (with tolerance) and the
  share of attribution mass inside lesions, compared with the lesion area (chance level).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from scipy.stats import spearmanr


# ------------------------------------------------------------------ deletion / insertion
def cell_order(saliency: np.ndarray, fov: np.ndarray, cell: int, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Rank grid cells inside the FOV by mean saliency (or randomly if ``rng`` is given).

    Returns (ordered cell indices, cell-grid shape as array [rows, cols]).
    """
    h, w = saliency.shape
    rows, cols = h // cell, w // cell
    sal = saliency[: rows * cell, : cols * cell].reshape(rows, cell, cols, cell).mean((1, 3)).ravel()
    inside = fov[: rows * cell, : cols * cell].reshape(rows, cell, cols, cell).mean((1, 3)).ravel() >= 0.5
    candidates = np.flatnonzero(inside)
    if rng is not None:
        order = rng.permutation(candidates)
    else:
        order = candidates[np.argsort(-sal[candidates], kind="stable")]
    return order, np.array([rows, cols])


def perturbation_curve(
    model, x: torch.Tensor, baseline: torch.Tensor, order: np.ndarray, grid: np.ndarray, cell: int,
    target: int, steps: int, mode: str, temperature: float = 1.0,
) -> np.ndarray:
    """Predicted-class probability as cells are deleted (x -> baseline) or inserted (baseline -> x)."""
    rows, cols = int(grid[0]), int(grid[1])
    n = len(order)
    counts = np.round(np.linspace(0, n, steps + 1)).astype(int)
    masks = torch.zeros(steps + 1, 1, *x.shape[-2:], device=x.device)
    for s, k in enumerate(counts):
        m = np.zeros(rows * cols, np.float32)
        m[order[:k]] = 1
        m = torch.from_numpy(m.reshape(rows, cols)).to(x.device)
        m = m.repeat_interleave(cell, 0).repeat_interleave(cell, 1)
        masks[s, 0, : m.shape[0], : m.shape[1]] = m
    if mode == "deletion":
        batch = x * (1 - masks) + baseline * masks
    elif mode == "insertion":
        batch = baseline * (1 - masks) + x * masks
    else:
        raise ValueError(mode)
    with torch.no_grad():
        probs = torch.softmax(model(batch) / temperature, 1)[:, target]
    return probs.cpu().numpy()


def curve_auc(curve: np.ndarray) -> float:
    xs = np.linspace(0, 1, len(curve))
    return float(np.sum((curve[1:] + curve[:-1]) / 2 * np.diff(xs)))


# ------------------------------------------------------------------- sanity check
def map_similarity(a: np.ndarray, b: np.ndarray, fov: np.ndarray | None = None, size: int = 64) -> float:
    """Spearman rank correlation between two maps (downsampled), inside the FOV only.

    Restricting to the FOV matters: maps that are both zero on the black border would
    otherwise look similar regardless of what the model does. NaN if either map is constant.
    """
    a = cv2.resize(a.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    keep = np.ones((size, size), bool) if fov is None else \
        cv2.resize(fov.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA) > 0.99
    a, b = a[keep], b[keep]
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def reinitialise(module: torch.nn.Module, seed: int) -> None:
    torch.manual_seed(seed)
    for m in module.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
        if hasattr(m, "reset_running_stats"):
            m.reset_running_stats()


# ------------------------------------------------------------- shortcut / location
def mass_outside_fov(saliency: np.ndarray, fov: np.ndarray, rim_px: int) -> dict:
    """Share of attribution mass outside the FOV and on its outer rim, with the area shares (chance)."""
    total = saliency.sum()
    inner = cv2.erode(fov.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=max(1, rim_px)).astype(bool)
    rim = fov & ~inner
    if total <= 0:
        return {"mass_outside": np.nan, "area_outside": float((~fov).mean()), "mass_rim": np.nan, "area_rim": float(rim.mean())}
    return {
        "mass_outside": float(saliency[~fov].sum() / total),
        "area_outside": float((~fov).mean()),
        "mass_rim": float(saliency[rim].sum() / total),
        "area_rim": float(rim.mean()),
    }


def localisation_scores(saliency: np.ndarray, lesion: np.ndarray, fov: np.ndarray, tolerance_px: int) -> dict:
    """Pointing-game hit (peak within ``tolerance_px`` of a lesion) and mass inside lesions vs lesion area."""
    sal = saliency * fov
    total = sal.sum()
    lesion = lesion & fov
    if total <= 0 or not lesion.any():
        return {"hit": np.nan, "mass_in_lesion": np.nan, "area_lesion": float(lesion.sum() / max(fov.sum(), 1))}
    dilated = cv2.dilate(lesion.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance_px + 1,) * 2)).astype(bool)
    peak = np.unravel_index(np.argmax(sal), sal.shape)
    return {
        "hit": float(dilated[peak]),
        "mass_in_lesion": float(sal[dilated].sum() / total),
        "area_lesion": float(dilated[fov].mean()),
    }
