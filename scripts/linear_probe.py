"""Baseline - logistic regression on frozen ImageNet ResNet-50 features (no fine-tuning).

Shows how much of the task is solved by generic ImageNet features, and gives the reference
representation for the lesion concept-probe comparison in explain.py. The regularisation
strength C is chosen on the validation split (QWK); the test split is only scored.

Usage:  python scripts/linear_probe.py --config configs/default.yaml
Output: <output_dir>/runs/linear_probe/{features_*.npz, logits_val.npy, logits_test.npy, train_summary.json}
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drxai.config import base_parser, load_config  # noqa: E402
from drxai.data.dataset import FundusDataset, build_transforms, load_metadata, make_loader  # noqa: E402
from drxai.metrics import core_metrics  # noqa: E402
from drxai.models import build_model, pooled_features  # noqa: E402
from drxai.utils import get_device, run_dir, save_json, seed_everything  # noqa: E402

C_GRID = [1e-3, 1e-2, 1e-1, 1.0]


def extract(model, df, cfg, device) -> np.ndarray:
    loader = make_loader(FundusDataset(df, cfg["paths"]["prepared_dir"], build_transforms(cfg, False)),
                         cfg["evaluation"]["batch_size"], False, cfg["data"]["num_workers"])
    feats = []
    with torch.no_grad():
        for batch in loader:
            feats.append(pooled_features(model, batch["image"].to(device)).float().cpu().numpy())
    return np.concatenate(feats)


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(cfg["seed"])
    device = get_device()
    out = run_dir(cfg, "linear_probe")
    meta = load_metadata(cfg["paths"]["prepared_dir"])
    model = build_model(cfg, num_classes=0).to(device).eval()  # pretrained backbone, no head

    data = {}
    for split in ("train", "val", "test"):
        df = meta[meta["split"] == split].reset_index(drop=True)
        cache = out / f"features_{split}.npz"
        if cache.exists():
            feats = np.load(cache)["features"]
        else:
            feats = extract(model, df, cfg, device)
            np.savez_compressed(cache, features=feats, image_id=df["image_id"].to_numpy())
        data[split] = (feats, df["grade"].to_numpy())
        print(f"[linear probe] {split}: features {feats.shape}")

    scaler = StandardScaler().fit(data["train"][0])
    xs = {k: scaler.transform(v[0]) for k, v in data.items()}
    results = []
    for c in C_GRID:
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=3000)
        clf.fit(xs["train"], data["train"][1])
        m = core_metrics(data["val"][1], clf.predict(xs["val"]))
        results.append((m["qwk"], c, clf, m))
        print(f"[linear probe] C={c:g}: val QWK {m['qwk']:.4f}, macro-F1 {m['macro_f1']:.4f}")
    best_qwk, best_c, clf, best_m = max(results, key=lambda r: r[0])
    np.save(out / "logits_val.npy", clf.decision_function(xs["val"]))
    np.save(out / "logits_test.npy", clf.decision_function(xs["test"]))
    save_json({"run": "linear_probe", "backbone": cfg["model"]["name"], "C": best_c, "selection_metric": "qwk",
               "best_val_score": best_qwk, "best_val_metrics": best_m,
               "grid": [{"C": r[1], "val_qwk": r[0]} for r in results]}, out / "train_summary.json")
    print(f"[linear probe] selected C={best_c:g} (val QWK {best_qwk:.4f}); logits saved in {out}")


if __name__ == "__main__":
    main()
