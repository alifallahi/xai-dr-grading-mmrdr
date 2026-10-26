"""Create a tiny synthetic dataset with the same layout as MMRDR-CFP (for smoke tests only).

The images are cartoon fundus photographs: an orange disc with an optic disc, vessels and
grade-dependent "lesions" (dark dots, yellow spots, blobs, vessel tufts). They exist so the
whole pipeline can be exercised without the real data; they say nothing about real
performance. Pixel masks of the drawn lesions are written too, to test the localisation script.

Usage:  python scripts/make_synthetic_mmrdr.py --out data/synthetic --n-train 400 --n-test 100
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

GRADE_P = [0.59, 0.12, 0.18, 0.055, 0.055]
LESIONS = ["MA", "HE", "IH", "VB_IRMA", "NV", "VH", "RD"]


def _fundus(rng: np.random.Generator, right_eye: bool) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    w = int(rng.choice([640, 720]))
    h = int(w * 0.85)
    img = np.zeros((h, w, 3), np.uint8)
    cx, cy = w // 2 + int(rng.integers(-15, 15)), h // 2 + int(rng.integers(-10, 10))
    r = int(h * rng.uniform(0.44, 0.52))  # sometimes truncated at top/bottom
    yy, xx = np.mgrid[:h, :w]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
    fov = d <= 1
    base = np.array([205, 95, 45], np.float32) * rng.uniform(0.85, 1.1)
    vignette = np.clip(1.05 - 0.45 * d ** 2, 0, 1)[..., None]
    img[fov] = np.clip(base * vignette, 0, 255)[fov].astype(np.uint8)
    disc = (cx + int((0.38 if right_eye else -0.38) * r), cy)
    for _ in range(7):  # vessels
        end = (int(cx + rng.uniform(-0.9, 0.9) * r), int(cy + rng.uniform(-0.9, 0.9) * r))
        ctrl = (int((disc[0] + end[0]) / 2 + rng.uniform(-60, 60)), int((disc[1] + end[1]) / 2 + rng.uniform(-60, 60)))
        t = np.linspace(0, 1, 40)[:, None]
        pts = ((1 - t) ** 2 * np.array(disc) + 2 * (1 - t) * t * np.array(ctrl) + t ** 2 * np.array(end)).astype(np.int32)
        cv2.polylines(img, [pts], False, (120, 25, 20), int(rng.integers(2, 5)), cv2.LINE_AA)
    cv2.circle(img, disc, int(0.12 * r), (250, 210, 140), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), int(0.1 * r), (150, 60, 30), -1, cv2.LINE_AA)
    img[~fov] = 0
    return img, fov, (cx, cy, r)


def _random_point(rng, cx, cy, r, fov):
    while True:
        a, rho = rng.uniform(0, 2 * np.pi), r * np.sqrt(rng.uniform(0.02, 0.8))
        x, y = int(cx + rho * np.cos(a)), int(cy + rho * np.sin(a))
        if 0 <= y < fov.shape[0] and 0 <= x < fov.shape[1] and fov[y, x]:
            return x, y


def _draw_lesions(rng, img, fov, geom, grade):
    cx, cy, r = geom
    masks = {k: np.zeros(fov.shape, np.uint8) for k in LESIONS}
    present = dict.fromkeys(LESIONS, 0)

    def dots(key, n, radius, colour):
        for _ in range(n):
            p = _random_point(rng, cx, cy, r, fov)
            rad = int(rng.integers(*radius))
            cv2.circle(img, p, rad, colour, -1, cv2.LINE_AA)
            cv2.circle(masks[key], p, rad, 1, -1)
        present[key] = int(n > 0)

    if grade >= 1:
        dots("MA", int(rng.integers(3, 9) if grade == 1 else rng.integers(8, 25)), (2, 4), (110, 15, 15))
    if grade >= 2 and rng.random() < 0.7:
        dots("HE", int(rng.integers(5, 20)), (3, 6), (235, 215, 90))
    if grade >= 2 and rng.random() < (0.6 if grade == 2 else 0.95):
        dots("IH", int(rng.integers(2, 6) if grade == 2 else rng.integers(10, 22)), (6, 11), (95, 15, 15))
    if grade >= 3 and rng.random() < 0.5:
        p = _random_point(rng, cx, cy, r, fov)
        pts = np.array([[p[0] + 6 * i, p[1] + int(6 * np.sin(i))] for i in range(12)], np.int32)
        cv2.polylines(img, [pts], False, (100, 20, 20), 2, cv2.LINE_AA)
        cv2.polylines(masks["VB_IRMA"], [pts], False, 1, 2)
        present["VB_IRMA"] = 1
    if grade == 4:
        if rng.random() < 0.85:
            p = _random_point(rng, cx, cy, r * 0.6, fov)
            for _ in range(25):
                q = (p[0] + int(rng.integers(-25, 25)), p[1] + int(rng.integers(-25, 25)))
                cv2.line(img, p, q, (140, 30, 40), 1, cv2.LINE_AA)
                cv2.line(masks["NV"], p, q, 1, 1)
            present["NV"] = 1
        if rng.random() < 0.4 or not present["NV"]:
            p = _random_point(rng, cx, cy, r, fov)
            overlay = img.copy()
            cv2.ellipse(overlay, p, (int(0.25 * r), int(0.15 * r)), float(rng.uniform(0, 180)), 0, 360, (70, 10, 10), -1)
            cv2.ellipse(masks["VH"], p, (int(0.25 * r), int(0.15 * r)), 0, 0, 360, 1, -1)
            img[:] = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
            present["VH"] = 1
    img[~fov] = 0
    return present, masks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--label-noise", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    root = Path(args.out)
    cfp = root / "MMRDR" / "MMRDR-CFP"
    (cfp / "img").mkdir(parents=True, exist_ok=True)
    rows = []
    for split, n in (("tr", args.n_train), ("ts", args.n_test)):
        for k in range(1, n + 1):
            image_id = f"{split}{k:06d}"
            grade = int(rng.choice(5, p=np.array(GRADE_P) / sum(GRADE_P)))
            right = bool(rng.random() < 0.5)
            img, fov, geom = _fundus(rng, right)
            present, masks = _draw_lesions(rng, img, fov, geom, grade)
            noise = rng.normal(0, 4, img.shape)
            img = np.clip(img + noise * fov[..., None], 0, 255).astype(np.uint8)
            label = grade
            if rng.random() < args.label_noise:
                label = int(np.clip(grade + rng.choice([-1, 1]), 0, 4))
            cv2.imwrite(str(cfp / "img" / f"{image_id}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            for key, m in masks.items():
                if m.any():
                    (root / "lesion_masks" / key).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(root / "lesion_masks" / key / f"{image_id}.png"), m * 255)
            rows.append({"type": "CFP", "image": f"img/{image_id}.jpg", "grade": label,
                         "lesion": str([present[k] for k in LESIONS]), "lr": int(right)})
    # Plant three train->test near-duplicates (re-encoded copies) to exercise the leakage audit.
    for k in range(3):
        src, dst = rows[k], rows[args.n_train + k]
        img = cv2.imread(str(cfp / src["image"]))
        cv2.imwrite(str(cfp / dst["image"]), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        dst.update(grade=src["grade"], lesion=src["lesion"], lr=src["lr"])
        for mask_file in (root / "lesion_masks").glob(f"*/{Path(dst['image']).stem}.png"):
            mask_file.unlink()  # masks drawn for the overwritten image no longer apply
    pd.DataFrame(rows).to_csv(cfp / "FP.csv", index=False)
    print(f"Wrote {len(rows)} synthetic images to {cfp}")
    write_idrid_like(root, cfp, rows)


def write_idrid_like(root: Path, cfp: Path, rows: list[dict], n: int = 30) -> None:
    """Copy test images with drawn lesions into an IDRiD-style nested layout (tests prepare_idrid.py).

    Synthetic MA -> IDRiD MA, IH (haemorrhage) -> IDRiD HE, HE (hard exudate) -> IDRiD EX.
    """
    seg = root / "idrid_like" / "A. Segmentation"
    img_dir = seg / "1. Original Images" / "b. Testing Set"
    gt = seg / "2. All Segmentation Groundtruths" / "b. Testing Set"
    folders = {"MA": ("1. Microaneurysms", "MA"), "IH": ("2. Haemorrhages", "HE"), "HE": ("3. Hard Exudates", "EX")}
    written = 0
    for row in rows:
        stem = Path(row["image"]).stem
        if not stem.startswith("ts") or written >= n:
            continue
        found = {k: root / "lesion_masks" / k / f"{stem}.png" for k in folders}
        found = {k: p for k, p in found.items() if p.exists()}
        if not found:
            continue
        written += 1
        idrid = f"IDRiD_{written:02d}"
        img_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(cfp / row["image"], img_dir / f"{idrid}.jpg")
        for key, p in found.items():
            folder, code = folders[key]
            (gt / folder).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(gt / folder / f"{idrid}_{code}.tif"), cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
    print(f"Wrote {written} IDRiD-style images with masks to {seg}")


if __name__ == "__main__":
    main()
