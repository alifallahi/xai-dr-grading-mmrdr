"""Step 1 - inspect and prepare MMRDR-CFP.

* reads FP.csv and checks counts against the paper (verifies lesion order / grade coding)
* crops each image to its field of view, pads to square, resizes, caches it (+ FOV mask)
* computes simple quality descriptors (sharpness, brightness, contrast)
* finds near-duplicate images (exhaustive high-pass NCC search) and measures train-test overlap
* checks grade/lesion consistency against the grading rules
* makes a grade-stratified, duplicate-grouped train/validation split of the official training set

Usage:  python scripts/prepare_data.py --config configs/default.yaml
Outputs (in paths.prepared_dir): images/, masks/, metadata.csv, splits.csv, audit/
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drxai import plotting  # noqa: E402
from drxai.config import base_parser, load_config  # noqa: E402
from drxai.data import audit  # noqa: E402
from drxai.data.dataset import cached_image_path, cached_mask_path  # noqa: E402
from drxai.data.mmrdr import check_against_paper, load_cfp_annotations  # noqa: E402
from drxai.data.preprocess import (highpass_thumbnail, preprocess_fundus, quality_metrics,  # noqa: E402
                                   read_rgb, write_rgb)
from drxai.data.splits import make_train_val_split  # noqa: E402
from drxai.utils import df_to_markdown, save_json  # noqa: E402


def _process(job: tuple) -> dict:
    """Preprocess one image (or reuse the cache) and return metadata + duplicate-search thumbnail."""
    image_id, path, prepared_dir, size, threshold, overwrite = job
    cv2.setNumThreads(0)
    img_path, mask_path = cached_image_path(prepared_dir, image_id), cached_mask_path(prepared_dir, image_id)
    with Image.open(path) as im:  # reads the header only
        orig_w, orig_h = im.size
    if img_path.exists() and mask_path.exists() and not overwrite:
        rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
    else:
        rgb, mask, _ = preprocess_fundus(read_rgb(path), size, threshold)
        write_rgb(str(img_path), rgb)
        cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
    record = {"image_id": image_id, "orig_w": orig_w, "orig_h": orig_h, **quality_metrics(rgb, mask)}
    return {"record": record, "thumb": highpass_thumbnail(rgb, mask)}


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--overwrite", action="store_true", help="re-create cached images")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    dcfg = cfg["data"]
    prepared = Path(cfg["paths"]["prepared_dir"])
    audit_dir = prepared / "audit"
    for sub in ("images", "masks", "audit"):
        (prepared / sub).mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- annotations
    df = load_cfp_annotations(cfg["paths"]["mmrdr_cfp_dir"])
    paper = check_against_paper(df)
    print(f"[prepare] {len(df)} images; grade counts {paper['grade_counts']} "
          f"(paper: {paper['paper_grade_counts']}); lesion counts match paper: {paper['lesion_counts_match_paper']}")
    if not paper["lesion_counts_match_paper"]:
        print("[prepare] WARNING: lesion totals differ from the paper's Table 2 - check the lesion column order.")

    # ------------------------------------------------------------- preprocessing
    jobs = [(r.image_id, r.path, prepared, dcfg["image_size"], dcfg["fov_threshold"], args.overwrite)
            for r in df.itertuples()]
    workers = max(1, dcfg.get("num_workers", 2))
    if workers > 1:
        with Pool(workers) as pool:
            results = list(tqdm(pool.imap(_process, jobs, chunksize=16), total=len(jobs), desc="preprocess"))
    else:
        results = [_process(j) for j in tqdm(jobs, desc="preprocess")]
    meta = df.merge(pd.DataFrame([r["record"] for r in results]), on="image_id", validate="one_to_one")
    thumbs = np.stack([r["thumb"] for r in results])

    # ------------------------------------------------------------- duplicates
    nn_idx, nn_val = audit.nearest_neighbours_ncc(thumbs)
    meta["nn_ncc"] = nn_val
    meta["nn_image_id"] = meta["image_id"].to_numpy()[nn_idx]
    pairs = audit.duplicate_pairs(thumbs, dcfg["dup_ncc_min"])
    meta["dup_group"] = audit.group_duplicates(len(meta), pairs)
    split_arr = meta["official_split"].to_numpy()
    grade_arr = meta["grade"].to_numpy()
    pairs["split_i"], pairs["split_j"] = split_arr[pairs["i"]], split_arr[pairs["j"]]
    pairs["grade_i"], pairs["grade_j"] = grade_arr[pairs["i"]], grade_arr[pairs["j"]]
    pairs["kind"] = ["-".join(sorted((a, b))) for a, b in zip(pairs["split_i"], pairs["split_j"])]
    has_train_dup = np.zeros(len(meta), bool)
    for i, j, si, sj in zip(pairs["i"], pairs["j"], pairs["split_i"], pairs["split_j"]):
        if si == "train":
            has_train_dup[j] = True
        if sj == "train":
            has_train_dup[i] = True
    meta["has_dup_in_train"] = has_train_dup & (meta["official_split"] == "test")
    pairs.assign(image_i=meta["image_id"].to_numpy()[pairs["i"]], image_j=meta["image_id"].to_numpy()[pairs["j"]]) \
        .to_csv(audit_dir / "duplicate_pairs.csv", index=False)

    # ------------------------------------------------------------ consistency
    flags = audit.consistency_flags(meta)
    meta = pd.concat([meta, flags], axis=1)

    meta.drop(columns=["path"]).to_csv(prepared / "metadata.csv", index=False)
    splits = make_train_val_split(meta, dcfg["val_n_splits"], cfg["seed"])
    splits.to_csv(prepared / "splits.csv", index=False)
    meta = meta.merge(splits, on="image_id")

    # ------------------------------------------------------------ audit report
    dup_summary = {}
    for kind, sub in pairs.groupby("kind"):
        dup_summary[kind] = {
            "pairs": int(len(sub)),
            "grade_discordant_pairs": int((sub["grade_i"] != sub["grade_j"]).sum()),
            "mean_abs_grade_difference": float((sub["grade_i"] - sub["grade_j"]).abs().mean()),
        }
    test = meta[meta["official_split"] == "test"]
    report = {
        "paper_check": paper,
        "class_table_official": audit.class_table(meta, "official_split").to_dict(),
        "class_table_model_splits": audit.class_table(meta, "split").to_dict(),
        "imbalance_ratio_train": float(meta[meta.split == "train"].grade.value_counts().max()
                                       / meta[meta.split == "train"].grade.value_counts().min()),
        "lesion_prevalence_by_grade_pct": audit.lesion_by_grade(meta).to_dict(),
        "resolutions": audit.resolution_summary(meta),
        "laterality_counts": meta["lr"].value_counts().to_dict(),
        "quality_median_by_grade": meta.groupby("grade")[["sharpness", "brightness", "contrast"]].median().round(2).to_dict(),
        "fov_detection_suspect": int((meta["fov_fraction"] < 0.3).sum()),
        # ~0.785 is a circle filling its square; ~1.0 means no surround was found (retina fills the frame,
        # or detection failed - inspect these images if there are many).
        "fov_full_frame": int((meta["fov_fraction"] > 0.97).sum()),
        "fov_full_frame_by_resolution": (meta.loc[meta["fov_fraction"] > 0.97, "orig_w"].astype(str) + "x"
                                         + meta.loc[meta["fov_fraction"] > 0.97, "orig_h"].astype(str)).value_counts().head(5).to_dict(),
        "duplicates": {
            "ncc_threshold": dcfg["dup_ncc_min"],
            "pairs_by_split": dup_summary,
            "images_in_duplicate_groups": int(meta["dup_group"].duplicated(keep=False).sum()),
            "test_images_with_near_duplicate_in_train": int(test["has_dup_in_train"].sum()),
            "test_images_total": int(len(test)),
        },
        "label_consistency": {
            "rules": audit.CONSISTENCY_RULES,
            "counts": {k: int(meta[k].sum()) for k in list(audit.CONSISTENCY_RULES) + ["label_inconsistent"]},
            "inconsistent_by_split": meta.groupby("split")["label_inconsistent"].sum().astype(int).to_dict(),
        },
    }
    save_json(report, audit_dir / "audit.json")

    counts = {s: [int(((meta.split == s) & (meta.grade == g)).sum()) for g in range(5)] for s in ("train", "val", "test")}
    plotting.class_distribution(counts, audit_dir / "class_distribution.png")
    plotting.histogram(meta["nn_ncc"].to_numpy(), audit_dir / "nearest_neighbour_ncc.png",
                       "NCC with most similar other image", "Near-duplicate search", vline=dcfg["dup_ncc_min"])
    top = pairs[pairs["kind"] == "test-train"].sort_values("ncc", ascending=False).head(6)
    if len(top) == 0:
        top = pairs.sort_values("ncc", ascending=False).head(6)
    if len(top):
        ids = meta["image_id"].to_numpy()
        shots = []
        for r in top.itertuples():
            a = cv2.cvtColor(cv2.imread(str(cached_image_path(prepared, ids[r.i]))), cv2.COLOR_BGR2RGB)
            b = cv2.cvtColor(cv2.imread(str(cached_image_path(prepared, ids[r.j]))), cv2.COLOR_BGR2RGB)
            shots.append((a, b, f"{ids[r.i]} grade {r.grade_i}", f"{ids[r.j]} grade {r.grade_j} (NCC {r.ncc:.2f})"))
        plotting.image_pairs(shots, audit_dir / "duplicate_examples.png", "Most similar image pairs - check by eye")

    lines = [
        "# MMRDR-CFP audit", "",
        f"Images: {len(meta)} (train {counts['train']}, val {counts['val']}, test {counts['test']} per grade).",
        f"Grade counts match paper: {paper['grade_counts_match_paper']}; lesion counts match paper: {paper['lesion_counts_match_paper']}.",
        f"Distinct original resolutions (camera proxy): {report['resolutions']['n_distinct_resolutions']}.",
        f"Near-duplicate pairs (NCC >= {dcfg['dup_ncc_min']}): {dup_summary}.",
        f"Test images with a near-duplicate in the official training set: {report['duplicates']['test_images_with_near_duplicate_in_train']} / {len(test)}.",
        f"Grade/lesion inconsistencies: {report['label_consistency']['counts']}.",
        "", "Lesion prevalence by grade (%):", "", df_to_markdown(audit.lesion_by_grade(meta)),
    ]
    (audit_dir / "audit.md").write_text("\n".join(lines))
    print("\n".join(lines[:7]))
    print(f"[prepare] done. Metadata: {prepared / 'metadata.csv'}  splits: {prepared / 'splits.csv'}  audit: {audit_dir}")


if __name__ == "__main__":
    main()
