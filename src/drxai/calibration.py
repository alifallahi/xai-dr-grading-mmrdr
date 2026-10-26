"""Confidence calibration: temperature scaling (Guo et al., 2017), ECE, NLL, Brier score.

The temperature is fitted on the validation split only and then applied unchanged to the
test split. Temperature scaling does not change the predicted grade (argmax), only the
confidence attached to it.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    """Single scalar T > 0 minimising validation NLL (optimised in log space)."""
    z = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1)).mean())


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[labels]
    return float(((probs - onehot) ** 2).sum(1).mean())


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    """Top-label confidence vs accuracy in equal-width bins (for ECE and reliability diagrams)."""
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    conf_sum = np.bincount(idx, weights=conf, minlength=n_bins)
    acc_sum = np.bincount(idx, weights=correct, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_conf = conf_sum / counts
        mean_acc = acc_sum / counts
    return {"edges": edges, "counts": counts, "confidence": mean_conf, "accuracy": mean_acc}


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    b = reliability_bins(probs, labels, n_bins)
    w = b["counts"] / b["counts"].sum()
    gap = np.nan_to_num(np.abs(b["accuracy"] - b["confidence"]))
    return float((w * gap).sum())


def calibration_summary(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    return {
        "ece": ece(probs, labels, n_bins),
        "nll": nll(probs, labels),
        "brier": brier(probs, labels),
        "mean_confidence": float(probs.max(1).mean()),
        "accuracy": float((probs.argmax(1) == labels).mean()),
    }
