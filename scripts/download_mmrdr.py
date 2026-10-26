"""Download MMRDR from figshare and extract only the colour fundus (CFP) subset.

MMRDR (Tang et al., Sci Data 2026; figshare article 29423747, CC BY 4.0) is distributed as a
single zip split into 9 parts (~18.6 GB in total, all three modalities). This script:
  1. lists the files through the public figshare API,
  2. downloads each part with resume support and checks its MD5,
  3. concatenates the parts into MMRDR.zip (deleting each part once appended),
  4. extracts only members under MMRDR-CFP/ into --out.
Peak disk use is roughly 2 x 18.6 GB during step 3. Use --keep-zip to keep MMRDR.zip.

Note: the CFP images originate from OIA-DDR (Li et al., 2019), whose own licence is
CC BY-NC-SA 4.0; check both licences before any non-academic use.

Usage:  python scripts/download_mmrdr.py --out data/MMRDR --work data/mmrdr_download
If the zip cannot be read directly, extract the parts with 7-Zip instead:  7z x MMRDR.zip.001
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

API = "https://api.figshare.com/v2/articles/29423747/files"


def md5sum(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, size: int) -> None:
    done = dest.stat().st_size if dest.exists() else 0
    if done == size:
        return
    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        mode = "ab" if done and r.status_code == 206 else "wb"
        with open(dest, mode) as f, tqdm(total=size, initial=done if mode == "ab" else 0, unit="B",
                                         unit_scale=True, desc=dest.name) as bar:
            for block in r.iter_content(1 << 20):
                f.write(block)
                bar.update(len(block))


def join_parts(parts: list[Path], big: Path, delete_parts: bool) -> None:
    """Concatenate raw split parts (.zip.001, .zip.002, ...) into one zip file."""
    tmp = big.with_suffix(".zip.partial")
    with open(tmp, "wb") as joined:
        for part in parts:
            with open(part, "rb") as src:
                shutil.copyfileobj(src, joined, 1 << 24)
            if delete_parts:
                part.unlink()
    tmp.rename(big)


def extract_cfp(big: Path, out: Path) -> None:
    with zipfile.ZipFile(big) as z:
        members = [m for m in z.namelist() if "MMRDR-CFP/" in m]
        if not members:
            raise SystemExit("No MMRDR-CFP/ entries found in the archive")
        prefix = members[0].split("MMRDR-CFP/")[0]
        for m in tqdm(members, desc="extracting CFP"):
            target = out / m[len(prefix):]
            if m.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/MMRDR", help="where MMRDR-CFP/ will be extracted")
    ap.add_argument("--work", default="data/mmrdr_download", help="temporary download folder")
    ap.add_argument("--parts-dir", default=None,
                    help="use MMRDR.zip.001... already downloaded here (e.g. via a browser) instead of downloading")
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()
    work, out = Path(args.work), Path(args.out)
    work.mkdir(parents=True, exist_ok=True)
    big = work / "MMRDR.zip"

    if not big.exists():
        if args.parts_dir:
            parts = sorted(Path(args.parts_dir).glob("MMRDR.zip.[0-9]*"))
            if not parts:
                raise SystemExit(f"No MMRDR.zip.NNN files in {args.parts_dir}")
            join_parts(parts, big, delete_parts=False)
        else:
            files = requests.get(API, params={"page_size": 100}, timeout=60).json()
            meta = sorted((f for f in files if ".zip." in f["name"]), key=lambda f: f["name"])
            print(f"{len(meta)} archive parts, {sum(f['size'] for f in meta) / 1e9:.1f} GB in total")
            for f in meta:
                dest = work / f["name"]
                download(f["download_url"], dest, f["size"])
                expected = f.get("computed_md5") or f.get("supplied_md5")
                if expected and md5sum(dest) != expected:
                    dest.unlink()
                    raise SystemExit(f"MD5 mismatch for {f['name']}; deleted it - re-run to download again")
            join_parts([work / f["name"] for f in meta], big, delete_parts=True)

    extract_cfp(big, out)
    if not args.keep_zip:
        big.unlink()
    print(f"Done. Set paths.mmrdr_cfp_dir to {out / 'MMRDR-CFP'} (contains FP.csv and img/).")


if __name__ == "__main__":
    main()
