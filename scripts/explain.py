"""Step 5 - explanations for the selected model, and checks on whether they can be trusted.

1. Panels: Grad-CAM and Integrated Gradients on representative test images (correct cases per
   grade, worst under-grading errors, an over-grading error, a referred case), with
   annotated lesions in the caption; for errors, Grad-CAM for the reference grade as well.
2. Decision-card prototypes (prediction, calibrated probabilities, conformal set, heatmap, action).
3. Faithfulness: deletion/insertion curves for Grad-CAM, IG and random orderings.
4. Sanity: cascading model-randomisation test.
5. Shortcut check: attribution mass outside the field of view and on its rim.
6. Concept probes: do the model's features linearly encode each annotated lesion, overall and
   within grade? Fine-tuned features vs frozen ImageNet features.

Requires: scripts/evaluate.py has been run for the selected model (for temperature + predictions).
Usage:    python scripts/explain.py --config configs/default.yaml
Outputs:  <output_dir>/explanations/<run>/
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drxai import plotting  # noqa: E402
from drxai.config import base_parser, load_config  # noqa: E402
from drxai.data.dataset import FundusDataset, build_transforms, load_cached, load_metadata, make_loader, normalise  # noqa: E402
from drxai.data.mmrdr import GRADE_SHORT, LESION_KEYS, LESION_NAMES  # noqa: E402
from drxai.models import gradcam_layer, load_checkpoint, pooled_features, randomisation_stages  # noqa: E402
from drxai.utils import get_device, load_json, save_json, seed_everything, selected_run_name  # noqa: E402
from drxai.xai.attribution import GradCAM, blurred_baseline, ig_saliency, integrated_gradients, normalise_map  # noqa: E402
from drxai.xai.evaluation import (cell_order, curve_auc, mass_outside_fov, map_similarity,  # noqa: E402
                                  perturbation_curve, reinitialise)


class Explainer:
    """Holds the model and produces Grad-CAM / IG maps for cached test images."""

    def __init__(self, model, cfg: dict, device: torch.device):
        self.model, self.cfg, self.device = model, cfg, device
        self.size = cfg["data"]["image_size"]
        self.sigma = cfg["explain"]["ig_baseline_blur_sigma"] * self.size / 512
        self.cam = GradCAM(model, gradcam_layer(model))

    def load(self, image_id: str) -> dict:
        rgb, fov = load_cached(self.cfg["paths"]["prepared_dir"], image_id)
        if rgb.shape[0] != self.size:
            rgb = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_AREA)
            fov = cv2.resize(fov.astype(np.uint8), (self.size, self.size), interpolation=cv2.INTER_NEAREST) > 0
        x = normalise(rgb).to(self.device)
        b = normalise(blurred_baseline(rgb, self.sigma, fov)).to(self.device)
        return {"rgb": rgb, "fov": fov, "x": x, "baseline": b}

    def gradcam(self, x: torch.Tensor, target: int | None = None) -> np.ndarray:
        cam, _ = self.cam(x, target)
        return cam[0].cpu().numpy()

    def ig(self, x, baseline, target: int) -> tuple[np.ndarray, float]:
        attr, gap = integrated_gradients(self.model, x, baseline, target, self.cfg["explain"]["ig_steps"])
        return ig_saliency(attr, smooth_sigma=2.0 * self.size / 512), gap


def lesion_list(row: pd.Series) -> list[str]:
    return [LESION_NAMES[k] for k in LESION_KEYS if row.get(f"lesion_{k}", 0) == 1]


def caption(row: pd.Series) -> str:
    status = "REFER (uncertain)" if row["referred"] else "auto-accept"
    lesions = ", ".join(k for k in LESION_KEYS if row.get(f"lesion_{k}", 0) == 1) or "none"
    return (f"{row['image_id']}  ref: {GRADE_SHORT[int(row['grade'])]}\n"
            f"pred: {GRADE_SHORT[int(row['pred'])]} (p={row['confidence']:.2f}), {status}\nlesions: {lesions}")


def select_examples(preds: pd.DataFrame) -> dict[str, list[str]]:
    """Deterministic choice of representative cases (no cherry-picking by eye)."""
    correct = []
    for g in range(5):
        sub = preds[(preds.grade == g) & (preds.pred == g)]
        if len(sub):
            correct.append(sub.sort_values("confidence", ascending=False).iloc[0]["image_id"])
    under = preds[preds.error < 0].sort_values(["error", "confidence"], ascending=[True, False]).head(3)
    over = preds[preds.error > 0].sort_values(["error", "confidence"], ascending=[False, False]).head(1)
    referred = preds[(preds.referred) & (preds.error == 0)].sort_values("u_exp_cost", ascending=False).head(1)
    return {"correct": correct, "errors": list(under.image_id) + list(over.image_id), "referred": list(referred.image_id)}


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--run", default=None)
    parser.add_argument("--skip", nargs="*", default=[], choices=["faithfulness", "sanity", "fov", "concepts"])
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(cfg["seed"])
    ecfg = cfg["explain"]
    run = args.run or selected_run_name(cfg)
    out_root = Path(cfg["paths"]["output_dir"])
    out = out_root / "explanations" / run
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    model, run_cfg, _ = load_checkpoint(out_root / "runs" / run / "best.pt", device)
    for p in model.parameters():
        p.requires_grad_(False)  # gradients w.r.t. inputs/activations only
    run_cfg["paths"] = cfg["paths"]
    run_cfg["explain"] = ecfg
    metrics = load_json(out_root / "evaluation" / run / "metrics.json")
    temperature = metrics["temperature"]
    preds = pd.read_csv(out_root / "evaluation" / run / "test_predictions.csv")
    conformal_q = metrics["conformal"]["threshold"]
    ex = Explainer(model, run_cfg, device)
    summary: dict = {"run": run, "temperature": temperature}

    # ------------------------------------------------------------ 1-2. panels + cards
    chosen = select_examples(preds)
    summary["examples"] = chosen
    by_id = preds.set_index("image_id")
    for group in ("correct", "errors"):
        rows = []
        for image_id in chosen[group] + (chosen["referred"] if group == "correct" else []):
            r = by_id.loc[image_id]
            d = ex.load(image_id)
            cam = ex.gradcam(d["x"], int(r["pred"]))
            ig, _ = ex.ig(d["x"], d["baseline"], int(r["pred"]))
            cam_ref = normalise_map(ex.gradcam(d["x"], int(r["grade"]))) if group == "errors" else None
            row = r.copy()
            row["image_id"] = image_id
            rows.append({"image": d["rgb"], "gradcam": normalise_map(cam), "gradcam_ref": cam_ref,
                         "ig": normalise_map(ig), "caption": caption(row)})
        if rows:
            title = ("Correct predictions (highest confidence per grade) and one referred case" if group == "correct"
                     else "Largest under-grading errors and largest over-grading error")
            plotting.explanation_panel(rows, out / f"panel_{group}.png", title)

    card_ids = (chosen["correct"][-1:] + chosen["referred"][:1] + chosen["errors"][:1])
    for i, image_id in enumerate(card_ids):
        r = by_id.loc[image_id]
        d = ex.load(image_id)
        probs = r[[f"p{k}" for k in range(5)]].to_numpy(float)
        plotting.decision_card({
            "image": d["rgb"], "gradcam": normalise_map(ex.gradcam(d["x"], int(r["pred"]))), "probs": probs,
            "set": [k for k in range(5) if 1 - probs[k] <= conformal_q] or [int(r["pred"])],
            "pred": int(r["pred"]), "grade": int(r["grade"]), "referred": bool(r["referred"]),
            "lesions": lesion_list(r),
        }, out / f"decision_card_{i + 1}_{image_id}.png")

    # -------------------------------------------------------- 3. faithfulness
    rng = np.random.default_rng(cfg["seed"])
    faith_ids = []
    for g in range(5):
        ids = preds[preds.grade == g].image_id.to_numpy()
        faith_ids += list(rng.choice(ids, size=min(len(ids), ecfg["faithfulness_n_per_grade"]), replace=False))
    if "faithfulness" not in args.skip:
        cell, steps = ecfg["faithfulness_cell"] * ex.size // 512 or 1, ecfg["faithfulness_steps"]
        curves = {(m, mode): [] for m in ("Grad-CAM", "Integrated Gradients", "Random") for mode in ("deletion", "insertion")}
        gaps, fov_rows = [], []
        for image_id in tqdm(faith_ids, desc="faithfulness"):
            d = ex.load(image_id)
            target = int(by_id.loc[image_id, "pred"])
            maps = {"Grad-CAM": ex.gradcam(d["x"], target)}
            maps["Integrated Gradients"], gap = ex.ig(d["x"], d["baseline"], target)
            gaps.append(gap)
            fov_rows.append({"image_id": image_id, "method": "Integrated Gradients",
                             **mass_outside_fov(maps["Integrated Gradients"], d["fov"], ex.size // 25)})
            for mode in ("deletion", "insertion"):
                for name, sal in maps.items():
                    order, grid = cell_order(sal, d["fov"], cell)
                    curves[(name, mode)].append(perturbation_curve(model, d["x"], d["baseline"], order, grid, cell,
                                                                   target, steps, mode, temperature))
                rand = [perturbation_curve(model, d["x"], d["baseline"], *cell_order(maps["Grad-CAM"], d["fov"], cell, rng),
                                           cell, target, steps, mode, temperature) for _ in range(3)]
                curves[("Random", mode)].append(np.mean(rand, 0))
        rows = []
        for (name, mode), cs in curves.items():
            aucs = np.array([curve_auc(c) for c in cs])
            rows.append({"method": name, "mode": mode, "auc_mean": aucs.mean(), "auc_se": aucs.std() / np.sqrt(len(aucs)),
                         "n_images": len(aucs)})
        faith = pd.DataFrame(rows)
        # Paired comparison with random ordering (per image), the quantity that matters.
        for name in ("Grad-CAM", "Integrated Gradients"):
            for mode in ("deletion", "insertion"):
                diff = np.array([curve_auc(a) - curve_auc(b) for a, b in zip(curves[(name, mode)], curves[("Random", mode)])])
                faith.loc[(faith.method == name) & (faith["mode"] == mode), "minus_random_mean"] = diff.mean()
                faith.loc[(faith.method == name) & (faith["mode"] == mode), "minus_random_se"] = diff.std() / np.sqrt(len(diff))
        faith.to_csv(out / "faithfulness.csv", index=False)
        summary["faithfulness"] = faith.to_dict(orient="records")
        summary["ig_completeness_gap_median_abs"] = float(np.median(np.abs(gaps)))
        for mode, ylabel in (("deletion", "P(predicted grade)  - lower is more faithful"),
                             ("insertion", "P(predicted grade)  - higher is more faithful")):
            plotting.mean_curves({n: np.array(curves[(n, mode)]) for n in ("Grad-CAM", "Integrated Gradients", "Random")},
                                 out / f"{mode}_curves.png", ylabel, f"{mode.capitalize()} (n = {len(faith_ids)} test images)")
        pd.DataFrame(fov_rows).to_csv(out / "ig_fov_mass.csv", index=False)
        print(faith.round(3).to_string(index=False))

    # ------------------------------------------------------------- 4. sanity check
    if "sanity" not in args.skip:
        sanity_ids = faith_ids[:: max(1, len(faith_ids) // ecfg["sanity_n"])][: ecfg["sanity_n"]]
        originals = {}
        for image_id in sanity_ids:
            d = ex.load(image_id)
            t = int(by_id.loc[image_id, "pred"])
            originals[image_id] = (d, t, ex.gradcam(d["x"], t), ex.ig(d["x"], d["baseline"], t)[0])
        rand_model = copy.deepcopy(model)
        rand_ex = Explainer(rand_model, run_cfg, device)
        stage_names, sims = [], {"Grad-CAM": [], "Integrated Gradients": []}
        for i, (stage, _) in enumerate(randomisation_stages(model)):
            module = dict(randomisation_stages(rand_model))[stage]
            reinitialise(module, cfg["seed"] + i)
            for p in rand_model.parameters():
                p.requires_grad_(False)
            stage_names.append(stage)
            s_cam, s_ig = [], []
            for image_id, (d, t, cam0, ig0) in originals.items():
                s_cam.append(map_similarity(cam0, rand_ex.gradcam(d["x"], t), d["fov"]))
                s_ig.append(map_similarity(ig0, rand_ex.ig(d["x"], d["baseline"], t)[0], d["fov"]))
            sims["Grad-CAM"].append(float(np.nanmean(s_cam)) if not np.all(np.isnan(s_cam)) else float("nan"))
            sims["Integrated Gradients"].append(float(np.nanmean(s_ig)) if not np.all(np.isnan(s_ig)) else float("nan"))
        summary["sanity"] = {"stages_cumulative": stage_names, "spearman": sims, "n_images": len(originals)}
        plotting.sanity_curve(stage_names, sims, out / "sanity_randomisation.png")
        print(f"[explain] sanity (Spearman vs trained maps): {sims}")

    # ------------------------------------------------------------- 5. FOV mass (all test images)
    if "fov" not in args.skip:
        rows = []
        for image_id in tqdm(preds.image_id, desc="Grad-CAM FOV check"):
            d = ex.load(image_id)
            rows.append({"image_id": image_id, **mass_outside_fov(ex.gradcam(d["x"]), d["fov"], ex.size // 25)})
        fov = pd.DataFrame(rows)
        fov.to_csv(out / "gradcam_fov_mass.csv", index=False)
        summary["gradcam_fov"] = fov.drop(columns="image_id").describe().loc[["mean", "50%", "max"]].to_dict()
        summary["gradcam_fov_images_over_10pct_outside"] = int((fov["mass_outside"] > 0.10).sum())

    # ------------------------------------------------------------- 6. concept probes
    if "concepts" not in args.skip:
        summary["concept_probes"] = concept_probes(model, run_cfg, cfg, device, out_root, out)

    save_json(summary, out / "explain_summary.json")
    print(f"[explain] outputs in {out}")


def concept_probes(model, run_cfg: dict, cfg: dict, device, out_root: Path, out: Path) -> list[dict]:
    """Logistic-regression probes for each annotated lesion on pooled features.

    'within-grade AUC' averages AUCs computed inside each grade (where both classes have >= 10
    images), so a probe cannot score well merely by detecting the grade, which lesions track.
    """
    meta = load_metadata(cfg["paths"]["prepared_dir"])
    feats = {}
    for split in ("train", "val", "test"):
        df = meta[meta.split == split].reset_index(drop=True)
        loader = make_loader(FundusDataset(df, cfg["paths"]["prepared_dir"], build_transforms(run_cfg, False)),
                             cfg["evaluation"]["batch_size"], False, cfg["data"]["num_workers"])
        with torch.no_grad():
            f = np.concatenate([pooled_features(model, b["image"].to(device)).float().cpu().numpy() for b in loader])
        feats[split] = (f, df)
    sources = {"fine-tuned": {s: v[0] for s, v in feats.items()}}
    lp = out_root / "runs" / "linear_probe"
    if all((lp / f"features_{s}.npz").exists() for s in ("train", "val", "test")):
        sources["ImageNet (frozen)"] = {s: np.load(lp / f"features_{s}.npz")["features"] for s in ("train", "val", "test")}
    rows = []
    for key in LESION_KEYS:
        col = f"lesion_{key}"
        y = {s: feats[s][1][col].to_numpy() for s in feats}
        g_test = feats["test"][1]["grade"].to_numpy()
        if any(np.nansum(y[s]) < 5 for s in ("train", "test")):
            continue
        row = {"lesion": key, "n_pos_test": int(np.nansum(y["test"]))}
        for name, f in sources.items():
            ok = {s: ~np.isnan(y[s]) for s in y}
            scaler = StandardScaler().fit(f["train"][ok["train"]])
            best, best_auc = None, -1
            for c in (1e-3, 1e-2, 1e-1):
                clf = LogisticRegression(C=c, class_weight="balanced", max_iter=3000)
                clf.fit(scaler.transform(f["train"][ok["train"]]), y["train"][ok["train"]])
                if np.unique(y["val"][ok["val"]]).size < 2:
                    best = clf
                    break
                auc = roc_auc_score(y["val"][ok["val"]], clf.decision_function(scaler.transform(f["val"][ok["val"]])))
                if auc > best_auc:
                    best, best_auc = clf, auc
            s = best.decision_function(scaler.transform(f["test"]))
            yt = y["test"]
            row[f"auc_{name}"] = float(roc_auc_score(yt[ok["test"]], s[ok["test"]]))
            within, weights = [], []
            for g in range(5):
                m = (g_test == g) & ok["test"]
                if m.sum() and min(yt[m].sum(), (1 - yt[m]).sum()) >= 10:
                    within.append(roc_auc_score(yt[m], s[m]))
                    weights.append(m.sum())
            row[f"within_grade_auc_{name}"] = float(np.average(within, weights=weights)) if within else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / "concept_probes.csv", index=False)
    print(table.round(3).to_string(index=False))
    return table.to_dict(orient="records")


if __name__ == "__main__":
    main()
