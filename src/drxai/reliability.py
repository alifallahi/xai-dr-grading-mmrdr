"""Ordinal decision rule, uncertainty scores and selective referral (risk-coverage).

Cost-sensitive decision
    With calibrated class probabilities p(t|x) and a cost matrix C[t, k] (true t, predicted k),
    the Bayes decision is argmin_k sum_t p(t|x) C[t, k]. Here C[t, k] = |t - k|, multiplied by
    ``undergrade_cost`` when k < t, so under-grading costs more than over-grading. The
    multiplier is fixed a priori (not tuned on test); in deployment it should be set with clinicians.

Uncertainty scores (higher = less certain)
    * msp      : 1 - max probability
    * entropy  : predictive entropy
    * exp_cost : expected misgrading cost of the chosen grade under C (ordinal-aware: mass on
                 distant grades counts more than mass on the neighbouring grade)

Selective referral
    Cases whose uncertainty exceeds a threshold are referred to a clinician; the threshold is the
    validation quantile that auto-accepts the target fraction (coverage) of cases.
"""

from __future__ import annotations

import numpy as np

from .metrics import core_metrics


def cost_matrix(n_classes: int = 5, undergrade_cost: float = 2.0) -> np.ndarray:
    t = np.arange(n_classes)[:, None]
    k = np.arange(n_classes)[None, :]
    c = np.abs(t - k).astype(float)
    c[k < t] *= undergrade_cost
    return c


def expected_costs(probs: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """(n, K) expected cost of predicting each grade."""
    return probs @ cost


def cost_sensitive_predict(probs: np.ndarray, cost: np.ndarray) -> np.ndarray:
    return expected_costs(probs, cost).argmin(1)


def uncertainty_scores(probs: np.ndarray, cost: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray]:
    p = np.clip(probs, 1e-12, 1)
    return {
        "msp": 1 - probs.max(1),
        "entropy": -(p * np.log(p)).sum(1),
        "exp_cost": expected_costs(probs, cost)[np.arange(len(pred)), pred],
    }


def risk_coverage(uncertainty: np.ndarray, error: np.ndarray) -> dict:
    """Error rate among the most-certain fraction of cases, for every coverage level."""
    order = np.argsort(uncertainty, kind="stable")
    err = error[order].astype(float)
    n = len(err)
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(err) / np.arange(1, n + 1)
    return {"coverage": coverage, "risk": risk, "aurc": float(risk.mean())}


def threshold_for_coverage(val_uncertainty: np.ndarray, coverage: float) -> float:
    """Uncertainty threshold that auto-accepts ``coverage`` of validation cases."""
    return float(np.quantile(val_uncertainty, coverage))


def conformal_threshold(val_probs: np.ndarray, val_labels: np.ndarray, alpha: float = 0.1) -> float:
    """Split-conformal (LAC) score threshold from validation: sets {k : 1 - p_k <= q} cover the true
    grade with probability >= 1 - alpha (marginally, assuming validation and test are exchangeable)."""
    scores = 1 - val_probs[np.arange(len(val_labels)), val_labels]
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def prediction_sets(probs: np.ndarray, q: float) -> np.ndarray:
    """Boolean (n, K) membership of each grade in the conformal prediction set (never empty)."""
    sets = (1 - probs) <= q
    sets[np.arange(len(probs)), probs.argmax(1)] = True
    return sets


def referral_report(y_true: np.ndarray, y_pred: np.ndarray, uncertainty: np.ndarray, threshold: float) -> dict:
    """Effect of referring cases with uncertainty > threshold."""
    accept = uncertainty <= threshold
    wrong = y_pred != y_true
    under = y_pred < y_true
    missed_ref = (y_true >= 2) & (y_pred < 2)
    out = {
        "threshold": threshold,
        "coverage": float(accept.mean()),
        "n_referred": int((~accept).sum()),
        "errors_captured": float(wrong[~accept].sum() / max(wrong.sum(), 1)),
        "undergrades_captured": float(under[~accept].sum() / max(under.sum(), 1)),
        "missed_referable_all": int(missed_ref.sum()),
        "missed_referable_accepted": int(missed_ref[accept].sum()),
    }
    if accept.any():
        acc_m = core_metrics(y_true[accept], y_pred[accept])
        out.update({f"accepted_{k}": acc_m[k] for k in ("accuracy", "qwk", "macro_f1", "undergrade_rate_diseased")})
    if (~accept).any():
        out["referred_accuracy"] = float((~wrong[~accept]).mean())
    return out
