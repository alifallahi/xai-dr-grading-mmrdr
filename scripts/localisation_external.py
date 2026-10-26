"""Optional - compare explanations with PIXEL-level lesion masks from an external dataset.

MMRDR's lesion labels are image-level (present/absent), so they cannot say *where* a lesion is.
Datasets with pixel masks (e.g. IDRiD segmentation subset, Porwal et al. 2018; the DDR lesion
segmentation subset, Li et al. 2019) can. This script preprocesses the images and masks with the
same FOV crop/resize as training, then scores Grad-CAM and IG maps per lesion type with:
  * pointing game: is the map's peak within `pointing_tolerance_px` of an annotated lesion?
  * mass in lesion: share of the map inside the (dilated) lesion mask, vs the lesion area (chance)
and compares them with a 'prior' map = the average Grad-CAM over all evaluated images (a map
that ignores the specific image). An explanation is only image-specific if it beats that prior.

Naming caution: IDRiD/DDR use EX = hard exudate and HE = haemorrhage; MMRDR uses HE = hard exudate.

Expected layout:  --images DIR (fundus images)   --masks DIR/<LESION>/<image stem>[_<LESION>].<ext>
Usage:  python scripts/localisation_external.py --config configs/default.yaml \
            --images /path/IDRiD/images --masks /path/IDRiD/masks --lesions MA HE EX SE
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drxai.config import base_parser, load_config  # noqa: E402
from drxai.data.dataset import normalise  # noqa: E402
from drxai.data.preprocess import apply_geometry, preprocess_fundus, read_rgb  # noqa: E402
from drxai.models import gradcam_layer, load_checkpoint  # noqa: E402
from drxai.utils import get_device, save_json, selected_run_name  # noqa: E402
from drxai.xai.attribution import GradCAM, blurred_baseline, ig_saliency, integrated_gradients  # noqa: E402
from drxai.xai.evaluation import localisation_scores  # noqa: E402

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def find_mask(lesion_dir: Path, stem: str) -> Path | None:
    for p in sorted(lesion_dir.glob(f"{stem}*")):
        if p.suffix.lower() in IMAGE_EXT and (p.stem == stem or p.stem.startswith(stem + "_")):
            return p
    return None


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--lesions", nargs="*", default=None, help="lesion sub-folders (default: all)")
    parser.add_argument("--run", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--name", default="external")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    run = args.run or selected_run_name(cfg)
    root = Path(cfg["paths"]["output_dir"])
    out = root / "localisation" / run / args.name
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    model, run_cfg, _ = load_checkpoint(root / "runs" / run / "best.pt", device)
    for p in model.parameters():
        p.requires_grad_(False)
    size = cfg["data"]["image_size"]
    tol = max(1, round(cfg["explain"]["pointing_tolerance_px"] * size / 512))
    sigma = cfg["explain"]["ig_baseline_blur_sigma"] * size / 512
    cam = GradCAM(model, gradcam_layer(model))

    mask_root = Path(args.masks)
    lesions = args.lesions or sorted(d.name for d in mask_root.iterdir() if d.is_dir())
    images = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_EXT)[: args.max_images]

    cases = []
    for path in tqdm(images, desc="explaining"):
        masks = {les: find_mask(mask_root / les, path.stem) for les in lesions}
        if not any(masks.values()):
            continue
        rgb_full = read_rgb(str(path))
        rgb, fov, geom = preprocess_fundus(rgb_full, size, cfg["data"]["fov_threshold"])
        x = normalise(rgb).to(device)
        g, logits = cam(x)
        pred = int(logits.argmax(1).item())
        attr, _ = integrated_gradients(model, x, normalise(blurred_baseline(rgb, sigma, fov)).to(device), pred,
                                       cfg["explain"]["ig_steps"])
        lesion_masks = {}
        for les, mpath in masks.items():
            if mpath is None:
                continue
            m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
            if m.shape != rgb_full.shape[:2]:
                m = cv2.resize(m, (rgb_full.shape[1], rgb_full.shape[0]), interpolation=cv2.INTER_NEAREST)
            lesion_masks[les] = apply_geometry((m > 0).astype(np.uint8) * 255, geom, cv2.INTER_NEAREST) > 127
        cases.append({"image": path.stem, "pred": pred, "fov": fov, "gradcam": g[0].cpu().numpy(),
                      "ig": ig_saliency(attr, 2.0 * size / 512), "masks": lesion_masks})
    if not cases:
        raise SystemExit("No image had a matching mask; check --images/--masks layout.")

    prior = np.mean([c["gradcam"] / (c["gradcam"].sum() + 1e-8) for c in cases], 0)
    rows = []
    for c in cases:
        for les, m in c["masks"].items():
            for method, sal in (("Grad-CAM", c["gradcam"]), ("Integrated Gradients", c["ig"]), ("Prior (mean Grad-CAM)", prior)):
                rows.append({"image": c["image"], "lesion": les, "method": method, "pred": c["pred"],
                             **localisation_scores(sal, m, c["fov"], tol)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "localisation_per_image.csv", index=False)
    summary = (df.groupby(["lesion", "method"])
               .agg(n=("hit", "count"), pointing_hit_rate=("hit", "mean"), mass_in_lesion=("mass_in_lesion", "mean"),
                    lesion_area_chance=("area_lesion", "mean")).reset_index())
    summary["mass_enrichment"] = summary["mass_in_lesion"] / summary["lesion_area_chance"]
    summary.to_csv(out / "localisation_summary.csv", index=False)
    save_json({"run": run, "images": len(cases), "tolerance_px": tol, "summary": summary.to_dict(orient="records")},
              out / "localisation_summary.json")
    print(summary.round(3).to_string(index=False))
    print(f"[localisation] outputs in {out}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    main()
