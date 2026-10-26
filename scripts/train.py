"""Step 2 - fine-tune a CNN on the training split; select the checkpoint on validation.

Usage:
  python scripts/train.py --config configs/default.yaml
  python scripts/train.py --config configs/ablation_unweighted.yaml
  python scripts/train.py --config configs/default.yaml --set train.lr=1e-4 experiment_name=resnet50_lr1e-4
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drxai.config import base_parser, load_config  # noqa: E402
from drxai.train import train  # noqa: E402

if __name__ == "__main__":
    args = base_parser(__doc__).parse_args()
    train(load_config(args.config, args.set))
