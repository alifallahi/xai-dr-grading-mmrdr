import cv2
import numpy as np
import pandas as pd
import pytest

from drxai.data.audit import consistency_flags, duplicate_pairs, group_duplicates
from drxai.data.mmrdr import load_cfp_annotations, parse_lesions
from drxai.data.preprocess import apply_geometry, highpass_thumbnail, preprocess_fundus
from drxai.data.splits import make_train_val_split


def fake_fundus(h=300, w=400, cx=210, cy=150, r=140, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    cv2.circle(img, (cx, cy), r, (200, 90, 40), -1)
    for _ in range(30):
        cv2.circle(img, (int(cx + rng.uniform(-90, 90)), int(cy + rng.uniform(-90, 90))), 3, (90, 20, 20), -1)
    return img


def test_preprocess_centres_and_squares_the_fov():
    rgb, mask, geom = preprocess_fundus(fake_fundus(), 128)
    assert rgb.shape == (128, 128, 3) and mask.shape == (128, 128)
    assert abs(mask.mean() - np.pi / 4) < 0.05  # a circle filling its bounding square
    ys, xs = np.where(mask)
    assert abs(ys.mean() - 63.5) < 2 and abs(xs.mean() - 63.5) < 2
    m = np.zeros((300, 400), np.uint8)
    m[150, 210] = 255  # FOV centre must map to the image centre
    out = apply_geometry(cv2.dilate(m, np.ones((9, 9), np.uint8)), geom, cv2.INTER_NEAREST)
    yy, xx = np.where(out > 0)
    assert abs(yy.mean() - 64) < 3 and abs(xx.mean() - 64) < 3


def test_fov_mask_handles_grey_surround_and_full_frame():
    grey = fake_fundus()
    grey[grey.max(axis=2) == 0] = 28  # dark-grey camera surround instead of black
    noise = np.random.default_rng(0).integers(-4, 5, grey.shape)
    grey = np.clip(grey.astype(int) + noise, 0, 255).astype(np.uint8)
    rgb, mask, _ = preprocess_fundus(grey, 128)
    assert abs(mask.mean() - np.pi / 4) < 0.05
    assert rgb[~mask].max() == 0  # surround removed
    full = np.full((200, 200, 3), (190, 90, 40), np.uint8)  # retina fills the frame
    assert preprocess_fundus(full, 64)[1].all()


def test_duplicate_search_finds_reencoded_copy_only():
    a = preprocess_fundus(fake_fundus(seed=1), 128)
    b = preprocess_fundus(fake_fundus(seed=2), 128)
    ok, buf = cv2.imencode(".jpg", a[0], [cv2.IMWRITE_JPEG_QUALITY, 70])
    a2 = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    thumbs = np.stack([highpass_thumbnail(a[0], a[1]), highpass_thumbnail(b[0], b[1]), highpass_thumbnail(a2, a[1])])
    pairs = duplicate_pairs(thumbs, 0.9)
    assert list(zip(pairs.i, pairs.j)) == [(0, 2)]
    groups = group_duplicates(3, pairs)
    assert groups[0] == groups[2] != groups[1]


def test_parse_lesions():
    assert parse_lesions("[0, 1, 0, 0, 1, 0, 0]") == [0, 1, 0, 0, 1, 0, 0]
    assert parse_lesions("0 1 0 0 1 0 0") == [0, 1, 0, 0, 1, 0, 0]
    assert np.isnan(parse_lesions("[1, 0]")).all()
    assert np.isnan(parse_lesions(float("nan"))).all()


def test_load_annotations_and_split(tmp_path):
    cfp = tmp_path / "MMRDR-CFP"
    (cfp / "img").mkdir(parents=True)
    rows = []
    rng = np.random.default_rng(0)
    for split, n in (("tr", 80), ("ts", 20)):
        for k in range(1, n + 1):
            name = f"{split}{k:06d}.jpg"
            cv2.imwrite(str(cfp / "img" / name), fake_fundus(seed=k))
            g = int(rng.integers(0, 5))
            rows.append({"type": "CFP", "image": f"img/{name}", "grade": g,
                         "lesion": str([int(g > 0)] + [0] * 6), "lr": k % 2})
    pd.DataFrame(rows).to_csv(cfp / "FP.csv", index=False)
    df = load_cfp_annotations(cfp)
    assert (df.official_split == "train").sum() == 80 and (df.official_split == "test").sum() == 20
    assert df.lesion_MA.sum() == (df.grade > 0).sum()
    df["dup_group"] = np.arange(len(df))
    df.loc[[0, 1], "dup_group"] = 0  # a duplicate pair must stay together
    splits = make_train_val_split(df, 5, seed=0)
    s = splits.set_index("image_id").split
    assert s[df.image_id[0]] == s[df.image_id[1]]
    assert (s[df.official_split.values == "test"] == "test").all()


def test_consistency_rules():
    df = pd.DataFrame({"grade": [0, 1, 4, 2],
                       **{f"lesion_{k}": v for k, v in zip(["MA", "HE", "IH", "VB_IRMA", "NV", "VH", "RD"],
                                                          [[1, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                                           [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])}})
    f = consistency_flags(df)
    assert f.R1_noDR_with_lesion.tolist() == [True, False, False, False]
    assert f.R3_mild_with_non_MA.tolist() == [False, True, False, False]
    assert f.R4_PDR_without_PDR_sign.tolist() == [False, False, True, False]
    assert f.R5_PDR_sign_below_PDR.tolist() == [False, False, False, True]


def test_missing_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cfp_annotations(tmp_path)
