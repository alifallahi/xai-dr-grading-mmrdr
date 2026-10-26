import numpy as np
from sklearn.metrics import cohen_kappa_score

from drxai.calibration import ece, fit_temperature, softmax
from drxai.metrics import bootstrap_ci, core_metrics, full_metrics, qwk
from drxai.reliability import (conformal_threshold, cost_matrix, cost_sensitive_predict, prediction_sets,
                               referral_report, risk_coverage)


def test_qwk_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 500)
    p = np.clip(y + rng.integers(-1, 2, 500), 0, 4)
    assert np.isclose(qwk(y, p), cohen_kappa_score(y, p, weights="quadratic"))


def test_perfect_predictions():
    y = np.array([0, 1, 2, 3, 4] * 10)
    m = core_metrics(y, y)
    assert m["accuracy"] == 1 and m["qwk"] == 1 and m["undergrade_rate"] == 0
    assert m["referable_sensitivity"] == 1 and m["referable_specificity"] == 1


def test_undergrading_and_binary_views():
    y = np.array([0, 2, 3, 4])
    p = np.array([0, 1, 3, 2])  # two under-gradings; one referable case (2) missed
    m = full_metrics(y, p)
    assert np.isclose(m["undergrade_rate_diseased"], 2 / 3)
    assert m["referable_missed"] == 1 and m["referable_n_positive"] == 3
    assert np.isclose(m["error_2plus_steps"], 0.25)


def test_temperature_recovers_known_value():
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 3, (4000, 5))
    true_t = 2.0
    labels = np.array([rng.choice(5, p=q) for q in softmax(logits, true_t)])
    assert abs(fit_temperature(logits, labels) - true_t) < 0.15


def test_ece_zero_for_calibrated_and_large_for_overconfident():
    rng = np.random.default_rng(2)
    probs = np.full((20000, 5), 0.1)
    probs[:, 0] = 0.6
    labels = np.array([rng.choice(5, p=q) for q in probs])
    assert ece(probs, labels) < 0.02
    over = np.full((1000, 5), 0.0)
    over[:, 0] = 1.0
    assert ece(over, np.ones(1000, int)) > 0.99


def test_cost_sensitive_rule_shifts_up_under_asymmetric_cost():
    probs = np.array([[0.5, 0.0, 0.5, 0.0, 0.0]])
    assert cost_sensitive_predict(probs, cost_matrix(5, 1.0))[0] in (0, 1, 2)
    assert cost_sensitive_predict(probs, cost_matrix(5, 3.0))[0] == 2  # under-grading now costlier
    assert cost_matrix(5, 2.0)[3, 1] == 4.0 and cost_matrix(5, 2.0)[1, 3] == 2.0


def test_risk_coverage_and_referral():
    err = np.array([0, 0, 0, 1, 1])
    unc = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    rc = risk_coverage(unc, err)
    assert rc["risk"][2] == 0 and np.isclose(rc["risk"][-1], 0.4)
    y = np.array([0, 1, 2, 3, 2])
    p = np.array([0, 1, 2, 1, 0])
    rep = referral_report(y, p, unc, threshold=0.5)
    assert rep["coverage"] == 0.6 and rep["errors_captured"] == 1.0 and rep["missed_referable_accepted"] == 0


def test_conformal_sets_reach_target_coverage():
    rng = np.random.default_rng(3)
    logits = rng.normal(0, 2, (6000, 5))
    probs = softmax(logits)
    labels = np.array([rng.choice(5, p=q) for q in probs])
    q = conformal_threshold(probs[:3000], labels[:3000], alpha=0.1)
    sets = prediction_sets(probs[3000:], q)
    coverage = sets[np.arange(3000), labels[3000:]].mean()
    assert 0.87 <= coverage <= 0.93
    assert sets.sum(1).min() >= 1


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 5, 400)
    p = np.where(rng.random(400) < 0.7, y, rng.integers(0, 5, 400))
    ci = bootstrap_ci(y, p, n_boot=200)
    acc = core_metrics(y, p)["accuracy"]
    assert ci["accuracy"][0] <= acc <= ci["accuracy"][1]
