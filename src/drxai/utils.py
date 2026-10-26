"""Small shared helpers: seeding, device selection, JSON I/O, run folders."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch. ``deterministic`` trades speed for bitwise repeatability."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=_NumpyEncoder)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def df_to_markdown(df, floatfmt: str = ".3f", index: bool = True) -> str:
    """Minimal pandas -> GitHub-markdown table (avoids an extra dependency)."""
    def fmt(v: Any) -> str:
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else format(float(v), floatfmt)
        return str(v)

    cols = ([df.index.name or ""] if index else []) + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for idx, row in df.iterrows():
        cells = ([fmt(idx)] if index else []) + [fmt(v) for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_dir(cfg: dict, name: str | None = None) -> Path:
    """Folder for one training run: <output_dir>/runs/<experiment_name>."""
    path = Path(cfg["paths"]["output_dir"]) / "runs" / (name or cfg["experiment_name"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def selected_run_name(cfg: dict) -> str | None:
    """Name of the run chosen by scripts/select_model.py (validation only), if any."""
    marker = Path(cfg["paths"]["output_dir"]) / "selected_run.txt"
    return marker.read_text().strip() if marker.exists() else None
