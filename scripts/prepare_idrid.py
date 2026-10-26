"""Optional - flatten the IDRiD segmentation subset for scripts/localisation_external.py.

IDRiD (Porwal et al., 2018) provides 81 fundus images with pixel-level masks for microaneurysms,
haemorrhages, hard exudates and soft exudates. Download "A. Segmentation.zip" from IEEE DataPort
(free account needed), unzip it, and point --src at the extracted folder. Files are found by name
anywhere under --src, so the exact folder layout does not matter:
    IDRiD_XX.jpg            fundus image
    IDRiD_XX_MA.tif / _HE / _EX / _SE   lesion masks (any image format)
Only images that have at least one mask are kept. Masks are binarised and saved as PNG.

Naming: in IDRiD, HE = haemorrhage and EX = hard exudate (in MMRDR, HE = hard exudate).

Usage:  python scripts/prepare_idrid.py --src "data/idrid_raw/A. Segmentation" --out data/idrid
Output: data/idrid/images/IDRiD_XX.jpg and data/idrid/masks/{MA,HE,EX,SE}/IDRiD_XX.png
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_RE = re.compile(r"(IDRiD_\d+)\.(jpe?g|png|tiff?)", re.IGNORECASE)
MASK_RE = re.compile(r"(IDRiD_\d+)_(MA|HE|EX|SE)\.(tiff?|png|jpe?g|bmp|gif)", re.IGNORECASE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="extracted 'A. Segmentation' folder (or any parent of it)")
    ap.add_argument("--out", default="data/idrid")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    if not src.exists():
        raise SystemExit(f"{src} does not exist")

    images: dict[str, Path] = {}
    masks: dict[tuple[str, str], Path] = {}
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        if m := MASK_RE.fullmatch(p.name):
            masks[(m.group(1), m.group(2).upper())] = p
        elif m := IMAGE_RE.fullmatch(p.name):
            images[m.group(1)] = p

    with_masks = sorted({stem for stem, _ in masks if stem in images})
    if not with_masks:
        raise SystemExit(f"No IDRiD_XX images with IDRiD_XX_<LESION> masks found under {src}")
    (out / "images").mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    for stem in with_masks:
        shutil.copy(images[stem], out / "images" / f"{stem}{images[stem].suffix.lower()}")
        for lesion in ("MA", "HE", "EX", "SE"):
            path = masks.get((stem, lesion))
            if path is None:
                continue
            arr = np.array(Image.open(path).convert("L")) > 0
            if not arr.any():
                continue
            (out / "masks" / lesion).mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr.astype(np.uint8) * 255).save(out / "masks" / lesion / f"{stem}.png")
            counts[lesion] += 1
    print(f"[idrid] {len(with_masks)} images; non-empty masks per lesion: {dict(counts)} -> {out}")


if __name__ == "__main__":
    main()
