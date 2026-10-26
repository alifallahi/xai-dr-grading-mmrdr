"""Step 6 - sensitivity to image quality (synthetic degradations of the test images).

For each corruption and severity: grading performance, under-grading, mean confidence and the
share of cases the validation-fitted uncertainty threshold would refer. The question is whether
uncertainty rises as quality falls (the system 'knows it does not know') or whether performance
degrades silently with confidence unchanged.

Usage:  python scripts/robustness.py --config configs/default.yaml
Output: <output_dir>/robustness/<run>/{robustness.csv, *.png}
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drxai import plotting  # noqa: E402
from drxai.calibration import softmax  # noqa: E402
from drxai.config import base_parser, load_config  # noqa: E402
from drxai.corruptions import corruption_grid  # noqa: E402
from drxai.data.dataset import FundusDataset, build_transforms, load_metadata, make_loader  # noqa: E402
from drxai.metrics import core_metrics  # noqa: E402
from drxai.models import load_checkpoint  # noqa: E402
from drxai.reliability import cost_matrix, uncertainty_scores  # noqa: E402
from drxai.train import predict_logits  # noqa: E402
from drxai.utils import get_device, load_json, selected_run_name  # noqa: E402


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--run", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    torch.set_grad_enabled(False)
    run = args.run or selected_run_name(cfg)
    root = Path(cfg["paths"]["output_dir"])
    out = root / "robustness" / run
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    model, run_cfg, _ = load_checkpoint(root / "runs" / run / "best.pt", device)
    metrics = load_json(root / "evaluation" / run / "metrics.json")
    temperature = metrics["temperature"]
    threshold = metrics["referral"]["exp_cost"]["threshold"]  # fitted on validation in evaluate.py
    cost = cost_matrix(5, cfg["evaluation"]["undergrade_cost"])

    test = load_metadata(cfg["paths"]["prepared_dir"]).query("split == 'test'").reset_index(drop=True)
    y = test["grade"].to_numpy()
    rows = []
    for name, level, value, fn in corruption_grid(cfg):
        ds = FundusDataset(test, cfg["paths"]["prepared_dir"], build_transforms(run_cfg, False), corruption=fn)
        loader = make_loader(ds, cfg["evaluation"]["batch_size"], False, cfg["data"]["num_workers"])
        p = softmax(predict_logits(model, loader, device, amp=False), temperature)
        pred = p.argmax(1)
        u = uncertainty_scores(p, cost, pred)["exp_cost"]
        m = core_metrics(y, pred)
        rows.append({"corruption": name, "level": level, "parameter": value, "accuracy": m["accuracy"],
                     "qwk": m["qwk"], "macro_f1": m["macro_f1"], "undergrade_rate_diseased": m["undergrade_rate_diseased"],
                     "referable_sensitivity": m["referable_sensitivity"], "mean_pred_grade": float(pred.mean()),
                     "mean_confidence": float(p.max(1).mean()), "referral_rate": float((u > threshold).mean()),
                     "errors_referred": float(((u > threshold) & (pred != y)).sum() / max((pred != y).sum(), 1))})
        print(f"[robustness] {name:14s} level {level} ({value:g}): QWK {m['qwk']:.3f} | under-grade {m['undergrade_rate_diseased']:.3f}"
              f" | confidence {rows[-1]['mean_confidence']:.3f} | referred {rows[-1]['referral_rate']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "robustness.csv", index=False)
    plotting.robustness_curves(df, "qwk", out / "robustness_qwk.png", "quadratic weighted kappa")
    plotting.robustness_curves(df, "undergrade_rate_diseased", out / "robustness_undergrading.png",
                               "under-grading rate (grade >= 1)")
    plotting.robustness_curves(df, "referral_rate", out / "robustness_referral.png", "share referred as uncertain")
    plotting.robustness_curves(df, "mean_confidence", out / "robustness_confidence.png", "mean confidence")
    print(f"[robustness] outputs in {out}")


if __name__ == "__main__":
    main()
