"""Step 4 - evaluate on the held-out test split (after model selection is final).

For the selected run, any other fine-tuned runs (secondary comparisons) and the baselines:
  * fits the calibration temperature on VALIDATION logits, applies it to test
  * reports accuracy, macro-F1, QWK, per-class metrics, confusion matrix, clinically framed
    binary metrics and ordinal error rates, with bootstrap 95% CIs
  * evaluates an a-priori cost-sensitive rule that penalises under-grading
  * evaluates calibration (ECE/NLL/Brier) and selective referral (risk-coverage, threshold from validation)
  * stratifies errors by annotated lesions, image quality, duplicate leakage and label consistency
Outputs go to <output_dir>/evaluation/<run>/.

Usage:  python scripts/evaluate.py --config configs/default.yaml            # all runs + baselines
        python scripts/evaluate.py --config configs/default.yaml --run resnet50_ce_512
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drxai import plotting  # noqa: E402
from drxai.calibration import calibration_summary, fit_temperature, reliability_bins, softmax  # noqa: E402
from drxai.config import base_parser, load_config  # noqa: E402
from drxai.data.dataset import FundusDataset, build_transforms, load_metadata, make_loader  # noqa: E402
from drxai.data.mmrdr import LESION_KEYS  # noqa: E402
from drxai.metrics import bootstrap_ci, core_metrics, full_metrics  # noqa: E402
from drxai.models import load_checkpoint  # noqa: E402
from drxai.reliability import (conformal_threshold, cost_matrix, cost_sensitive_predict,  # noqa: E402
                               prediction_sets, referral_report, risk_coverage, threshold_for_coverage,
                               uncertainty_scores)
from drxai.train import predict_logits  # noqa: E402
from drxai.utils import get_device, load_json, save_json, selected_run_name  # noqa: E402

# Published reference (Tang et al. 2026, Table 3, ResNet-50 fine-tuned at 512 px; their own
# random 7:1 validation split, so only approximately comparable).
PUBLISHED = {"run": "MMRDR paper: ResNet-50 FT (reported)", "accuracy": 0.823, "macro_f1": 0.702}


# ----------------------------------------------------------------------------- logits
def run_logits(cfg: dict, run: str, meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Validation and test logits for a run (computed once and cached in the run folder)."""
    rdir = Path(cfg["paths"]["output_dir"]) / "runs" / run
    if (rdir / "logits_val.npy").exists() and (rdir / "logits_test.npy").exists():
        return np.load(rdir / "logits_val.npy"), np.load(rdir / "logits_test.npy")
    device = get_device()
    model, run_cfg, _ = load_checkpoint(rdir / "best.pt", device)
    run_cfg["paths"]["prepared_dir"] = cfg["paths"]["prepared_dir"]
    out = []
    for split in ("val", "test"):
        df = meta[meta["split"] == split].reset_index(drop=True)
        loader = make_loader(FundusDataset(df, cfg["paths"]["prepared_dir"], build_transforms(run_cfg, False)),
                             cfg["evaluation"]["batch_size"], False, cfg["data"]["num_workers"])
        logits = predict_logits(model, loader, device, amp=False)
        np.save(rdir / f"logits_{split}.npy", logits)
        out.append(logits)
    return out[0], out[1]


def majority_logits(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Always predicts the commonest training grade; probabilities = training class frequencies."""
    freq = np.bincount(meta.loc[meta.split == "train", "grade"], minlength=5) / (meta.split == "train").sum()
    logit = np.log(np.clip(freq, 1e-6, 1))
    n_val, n_test = (meta.split == "val").sum(), (meta.split == "test").sum()
    return np.tile(logit, (n_val, 1)), np.tile(logit, (n_test, 1))


# ------------------------------------------------------------------------ subgroups
def subgroup_row(name: str, mask: np.ndarray, y: np.ndarray, pred: np.ndarray) -> dict:
    n = int(mask.sum())
    row = {"subgroup": name, "n": n}
    if n == 0:
        return row
    m = core_metrics(y[mask], pred[mask])
    row.update(accuracy=m["accuracy"], undergrade_rate_diseased=m["undergrade_rate_diseased"],
               mean_grade=float(y[mask].mean()))
    row["qwk"] = m["qwk"] if n >= 20 and len(np.unique(y[mask])) > 1 else np.nan
    return row


def stratified_analysis(test: pd.DataFrame, pred: np.ndarray, train: pd.DataFrame) -> pd.DataFrame:
    y = test["grade"].to_numpy()
    rows = [subgroup_row("all test images", np.ones(len(y), bool), y, pred)]
    for k in LESION_KEYS:
        has = (test[f"lesion_{k}"] == 1).to_numpy()
        rows.append(subgroup_row(f"lesion present: {k}", has & (y >= 1), y, pred))
    vb = (test["lesion_VB_IRMA"] == 1).to_numpy()
    nv = (test["lesion_NV"] == 1).to_numpy()
    rows.append(subgroup_row("severe NPDR with VB/IRMA", (y == 3) & vb, y, pred))
    rows.append(subgroup_row("severe NPDR without VB/IRMA", (y == 3) & ~vb, y, pred))
    rows.append(subgroup_row("PDR with NV annotated", (y == 4) & nv, y, pred))
    rows.append(subgroup_row("PDR without NV (VH/RD only)", (y == 4) & ~nv, y, pred))
    for col in ("sharpness", "brightness"):
        edges = np.quantile(train[col], [0.25, 0.5, 0.75])  # quartile edges from the TRAINING set
        q = np.digitize(test[col], edges)
        for i, label in enumerate(["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]):
            rows.append(subgroup_row(f"{col} {label}", q == i, y, pred))
    dup = test["has_dup_in_train"].astype(bool).to_numpy()
    rows.append(subgroup_row("near-duplicate in training set", dup, y, pred))
    rows.append(subgroup_row("no near-duplicate in training set", ~dup, y, pred))
    # Camera proxy: original image resolution (MMRDR-CFP has no device metadata).
    res = (test["orig_w"].astype(str) + "x" + test["orig_h"].astype(str)).to_numpy()
    for r, n in pd.Series(res).value_counts().items():
        if n >= 30:
            rows.append(subgroup_row(f"resolution {r} (camera proxy)", res == r, y, pred))
    inc = test["label_inconsistent"].astype(bool).to_numpy()
    rows.append(subgroup_row("grade/lesion labels inconsistent", inc, y, pred))
    rows.append(subgroup_row("grade/lesion labels consistent", ~inc, y, pred))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- evaluate
def evaluate_run(cfg: dict, name: str, logits_val: np.ndarray, logits_test: np.ndarray,
                 meta: pd.DataFrame, figures: bool = True) -> dict:
    ecfg = cfg["evaluation"]
    out = Path(cfg["paths"]["output_dir"]) / "evaluation" / name
    out.mkdir(parents=True, exist_ok=True)
    val = meta[meta.split == "val"].reset_index(drop=True)
    test = meta[meta.split == "test"].reset_index(drop=True)
    train = meta[meta.split == "train"]
    y_val, y = val["grade"].to_numpy(), test["grade"].to_numpy()

    # Calibration (temperature from validation only).
    temperature = fit_temperature(logits_val, y_val)
    p_val, p_raw, p = softmax(logits_val, temperature), softmax(logits_test), softmax(logits_test, temperature)
    pred = p.argmax(1)

    res = {"run": name, "temperature": temperature, "n_test": len(y)}
    res["test"] = full_metrics(y, pred, p)
    res["test_ci95"] = bootstrap_ci(y, pred, ecfg["n_bootstrap"], cfg["seed"])
    res["val"] = core_metrics(y_val, p_val.argmax(1))
    dedup = ~test["has_dup_in_train"].astype(bool).to_numpy()
    res["test_without_train_duplicates"] = {**core_metrics(y[dedup], pred[dedup]), "n": int(dedup.sum())}

    nb = ecfg["ece_bins"]
    res["calibration"] = {"test_before": calibration_summary(p_raw, y, nb), "test_after": calibration_summary(p, y, nb),
                          "val_after": calibration_summary(p_val, y_val, nb)}

    # Cost-sensitive decision (fixed a-priori cost; calibrated probabilities).
    cost = cost_matrix(5, ecfg["undergrade_cost"])
    pred_cs = cost_sensitive_predict(p, cost)
    res["cost_sensitive"] = {"undergrade_cost": ecfg["undergrade_cost"], "test": full_metrics(y, pred_cs, p)}

    # Uncertainty and selective referral (threshold from validation).
    u_val = uncertainty_scores(p_val, cost, p_val.argmax(1))
    u = uncertainty_scores(p, cost, pred)
    cov = ecfg["referral_coverage"]
    res["referral"] = {}
    curves_err, curves_ref = {}, {}
    for key in u:
        thr = threshold_for_coverage(u_val[key], cov)
        res["referral"][key] = referral_report(y, pred, u[key], thr)
        curves_err[key] = risk_coverage(u[key], pred != y)
        curves_ref[key] = risk_coverage(u[key], (y >= 2) & (pred < 2))
        res["referral"][key]["aurc_error"] = curves_err[key]["aurc"]
        res["referral"][key]["aurc_missed_referable"] = curves_ref[key]["aurc"]

    # Conformal prediction sets (90% target coverage, calibrated on validation).
    q = conformal_threshold(p_val, y_val, alpha=0.1)
    sets = prediction_sets(p, q)
    covered = sets[np.arange(len(y)), y]
    size = sets.sum(1)
    res["conformal"] = {"alpha": 0.1, "threshold": q, "test_coverage": float(covered.mean()),
                        "mean_set_size": float(size.mean()), "singleton_rate": float((size == 1).mean()),
                        "coverage_by_grade": {str(g): float(covered[y == g].mean()) for g in range(5) if (y == g).any()},
                        "set_size_by_grade": {str(g): float(size[y == g].mean()) for g in range(5) if (y == g).any()}}

    strata = stratified_analysis(test, pred, train)
    strata.to_csv(out / "stratified_test.csv", index=False)
    res["stratified"] = strata.to_dict(orient="records")

    # Per-image predictions for error analysis and explanation selection.
    referred_key = "exp_cost"
    thr = res["referral"][referred_key]["threshold"]
    preds = pd.DataFrame({"image_id": test["image_id"], "grade": y, "pred": pred, "pred_cost_sensitive": pred_cs,
                          "confidence": p.max(1), **{f"p{k}": p[:, k] for k in range(5)},
                          **{f"u_{k}": v for k, v in u.items()}, "referred": u[referred_key] > thr,
                          "conformal_set": ["".join(str(k) for k in range(5) if s[k]) for s in sets]})
    keep = [c for c in test.columns if c.startswith("lesion_") or c in
            ("sharpness", "brightness", "contrast", "fov_fraction", "orig_w", "orig_h",
             "has_dup_in_train", "label_inconsistent", "lr")]
    preds = pd.concat([preds, test[keep].reset_index(drop=True)], axis=1)
    preds["error"] = preds["pred"] - preds["grade"]
    preds.to_csv(out / "test_predictions.csv", index=False)
    save_json(res, out / "metrics.json")

    if figures:
        plotting.confusion(res["test"]["confusion_matrix"], out / "confusion_matrix.png", f"{name}: test (argmax)")
        plotting.confusion(res["cost_sensitive"]["test"]["confusion_matrix"], out / "confusion_matrix_cost_sensitive.png",
                           f"{name}: test (cost-sensitive)")
        plotting.reliability(reliability_bins(p_raw, y, nb), reliability_bins(p, y, nb),
                             res["calibration"]["test_before"]["ece"], res["calibration"]["test_after"]["ece"],
                             out / "reliability.png")
        plotting.risk_coverage(curves_err, out / "risk_coverage_error.png", "error rate among accepted", cov)
        plotting.risk_coverage(curves_ref, out / "risk_coverage_missed_referable.png",
                               "missed referable DR among accepted", cov)
        hist = Path(cfg["paths"]["output_dir"]) / "runs" / name / "history.csv"
        if hist.exists():
            plotting.training_curves(pd.read_csv(hist), out / "training_curves.png")

    t = res["test"]
    ci = res["test_ci95"]
    print(f"[evaluate] {name}: T={temperature:.2f} | acc {t['accuracy']:.3f} {np.round(ci['accuracy'], 3)} | "
          f"macro-F1 {t['macro_f1']:.3f} | QWK {t['qwk']:.3f} {np.round(ci['qwk'], 3)} | "
          f"referable sens {t['referable_sensitivity']:.3f} spec {t['referable_specificity']:.3f} | "
          f"ECE {res['calibration']['test_before']['ece']:.3f} -> {res['calibration']['test_after']['ece']:.3f}")
    return res


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--run", nargs="*", default=None,
                        help="run name(s); default = the selected run plus every other fine-tuned run")
    parser.add_argument("--no-baselines", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    torch.set_grad_enabled(False)
    meta = load_metadata(cfg["paths"]["prepared_dir"])
    out = Path(cfg["paths"]["output_dir"]) / "evaluation"

    selected = selected_run_name(cfg)
    if args.run:
        runs = args.run
    else:
        if selected is None:
            raise SystemExit("No selected run. Run scripts/select_model.py first (or pass --run).")
        # The selected model plus every other finished fine-tuned run (grid points, seed replicates).
        # A run without train_summary.json was interrupted and is ignored.
        others = sorted(p.parent.name for p in (Path(cfg["paths"]["output_dir"]) / "runs").glob("*/best.pt")
                        if p.parent.name != selected and (p.parent / "train_summary.json").exists())
        runs = [selected] + others
    results = []
    if not args.no_baselines:
        results.append(evaluate_run(cfg, "majority_class", *majority_logits(meta), meta, figures=False))
        if (Path(cfg["paths"]["output_dir"]) / "runs" / "linear_probe" / "logits_test.npy").exists():
            results.append(evaluate_run(cfg, "linear_probe", *run_logits(cfg, "linear_probe", meta), meta))
    for run in runs:
        results.append(evaluate_run(cfg, run, *run_logits(cfg, run, meta), meta))

    rows = []
    for r in results:
        t, ci = r["test"], r["test_ci95"]
        rows.append({"run": r["run"], "selected": r["run"] == selected, "accuracy": t["accuracy"], "acc_ci": ci["accuracy"], "macro_f1": t["macro_f1"],
                     "f1_ci": ci["macro_f1"], "qwk": t["qwk"], "qwk_ci": ci["qwk"],
                     "referable_sens": t["referable_sensitivity"], "referable_spec": t["referable_specificity"],
                     "referable_auc": t.get("referable_auc"), "undergrade_diseased": t["undergrade_rate_diseased"],
                     "ece_after": r["calibration"]["test_after"]["ece"]})
    seeds = seed_summary(cfg, rows)
    rows.append(PUBLISHED)
    pd.DataFrame(rows).to_csv(out / "summary_test.csv", index=False)
    print(f"[evaluate] summary written to {out / 'summary_test.csv'}")
    if seeds is not None:
        seeds.to_csv(out / "summary_seeds.csv", index=False)
        print(seeds.round(3).to_string(index=False))


def seed_summary(cfg: dict, rows: list[dict]) -> pd.DataFrame | None:
    """Mean and SD over a configuration's seed replicates (the run itself + train_seeds.py runs)."""
    runs_dir = Path(cfg["paths"]["output_dir"]) / "runs"
    groups: dict[str, list[dict]] = {}
    for row in rows:
        s = runs_dir / row["run"] / "train_summary.json"
        if not s.exists() or not (runs_dir / row["run"] / "best.pt").exists():
            continue
        summary = load_json(s)
        groups.setdefault(summary.get("replicate_of") or row["run"], []).append({**row, **{
            "val_qwk": summary["best_val_metrics"].get("val_qwk"),
            "val_macro_f1": summary["best_val_metrics"].get("val_macro_f1")}})
    out = []
    for name, members in groups.items():
        if len(members) < 2:
            continue
        rec = {"configuration": name, "n_seeds": len(members)}
        for k in ("val_qwk", "val_macro_f1", "accuracy", "macro_f1", "qwk", "referable_sens", "referable_spec", "ece_after"):
            vals = np.array([m[k] for m in members], dtype=float)
            rec[f"{k}_mean"], rec[f"{k}_sd"] = float(vals.mean()), float(vals.std(ddof=1))
        out.append(rec)
    return pd.DataFrame(out) if out else None


if __name__ == "__main__":
    main()
