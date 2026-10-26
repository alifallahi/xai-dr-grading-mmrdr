"""Classification, ordinal and clinically framed metrics for five-grade DR.

Clinically framed binary views (thresholds on the ICDR grade):
  * any DR          : grade >= 1
  * referable DR    : grade >= 2 (moderate NPDR or worse; DME cannot be assessed from these labels)
  * severe DR       : grade >= 3 (severe NPDR or PDR)
Under-grading (prediction below the reference grade) is reported separately because
in triage it delays care, whereas over-grading mainly costs clinic time.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

# Small subgroups can contain a single grade; kappa/F1 are then undefined and reported as NaN or 0.
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

N_CLASSES = 5
BINARY_TASKS = {"any_dr": 1, "referable": 2, "severe": 3}


def qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Quadratic weighted kappa; NaN (silently) when undefined, e.g. a subgroup with a single grade."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=list(range(N_CLASSES))))


def core_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Metrics that need only hard predictions (cheap; used inside the bootstrap)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    diseased = y_true >= 1
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(range(N_CLASSES)), zero_division=0)),
        "qwk": qwk(y_true, y_pred),
        "mae_grade": float(np.abs(y_true - y_pred).mean()),
        "undergrade_rate": float((y_pred < y_true).mean()),
        "undergrade_rate_diseased": float((y_pred[diseased] < y_true[diseased]).mean()) if diseased.any() else np.nan,
        "error_2plus_steps": float((np.abs(y_true - y_pred) >= 2).mean()),
    }
    for name, thr in BINARY_TASKS.items():
        pos, hit = y_true >= thr, y_pred >= thr
        out[f"{name}_sensitivity"] = float(hit[pos].mean()) if pos.any() else np.nan
        out[f"{name}_specificity"] = float((~hit[~pos]).mean()) if (~pos).any() else np.nan
    return out


def full_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray | None = None) -> dict:
    """Core metrics + per-class table, confusion matrix and probability-based AUCs."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = core_metrics(y_true, y_pred)
    labels = list(range(N_CLASSES))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out["per_class"] = {
        str(k): {"precision": float(p[k]), "recall": float(r[k]), "f1": float(f[k]), "support": int(s[k])}
        for k in labels
    }
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    out["linear_kappa"] = float(cohen_kappa_score(y_true, y_pred, weights="linear", labels=labels))
    for name, thr in BINARY_TASKS.items():
        pos, hit = y_true >= thr, y_pred >= thr
        out[f"{name}_ppv"] = float(pos[hit].mean()) if hit.any() else np.nan
        out[f"{name}_npv"] = float((~pos[~hit]).mean()) if (~hit).any() else np.nan
        out[f"{name}_missed"] = int((pos & ~hit).sum())
        out[f"{name}_n_positive"] = int(pos.sum())
    if probs is not None:
        for name, thr in BINARY_TASKS.items():
            pos = y_true >= thr
            score = probs[:, thr:].sum(1)
            out[f"{name}_auc"] = float(roc_auc_score(pos, score)) if 0 < pos.sum() < len(pos) else np.nan
        present = np.unique(y_true)
        if len(present) == N_CLASSES:
            out["macro_auc_ovr"] = float(roc_auc_score(y_true, probs, multi_class="ovr", labels=labels))
            out["per_class_auc"] = {
                str(k): float(roc_auc_score(y_true == k, probs[:, k])) for k in labels
            }
    # Under-grading by reference grade (row-wise view of the confusion matrix).
    out["undergrade_by_grade"] = {
        str(g): float((y_pred[y_true == g] < g).mean()) if (y_true == g).any() else np.nan for g in range(1, N_CLASSES)
    }
    return out


def bootstrap_ci(
    y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000, seed: int = 0,
    keys: tuple[str, ...] = ("accuracy", "macro_f1", "qwk", "undergrade_rate_diseased",
                             "referable_sensitivity", "referable_specificity"),
) -> dict:
    """Percentile 95% confidence intervals by resampling test images with replacement.

    Note: images, not patients, are resampled (no patient ids), so CIs are optimistic
    if fellow-eye images are correlated.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = core_metrics(y_true[idx], y_pred[idx])
        for k in keys:
            samples[k].append(m[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in samples.items()}
