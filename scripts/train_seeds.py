"""Step 5 - retrain the selected configuration with extra random seeds.

The replicates measure run-to-run variability (initialisation of the new head, data order,
augmentation, dropout); they are not candidates for selection. The final model stays the
selected seed-42 run, so explanations and robustness analyses refer to one fixed model.
The train/validation split is fixed by prepare_data.py and does not change with the seed.

Finished replicates are skipped, so the step can be re-run after a disconnect.

Usage:  python scripts/train_seeds.py --config configs/default.yaml --seeds 1 2
Output: <output_dir>/runs/<selected>_seed<k>/
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drxai.config import base_parser, load_config  # noqa: E402
from drxai.train import train  # noqa: E402
from drxai.utils import selected_run_name  # noqa: E402


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    selected = selected_run_name(cfg)
    if selected is None:
        raise SystemExit("No selected run. Run scripts/select_model.py first.")
    runs = Path(cfg["paths"]["output_dir"]) / "runs"
    base = load_config(runs / selected / "config.yaml")
    for seed in args.seeds:
        name = f"{selected}_seed{seed}"
        if (runs / name / "train_summary.json").exists():
            print(f"[seeds] {name} already finished, skipping")
            continue
        run_cfg = copy.deepcopy(base)
        run_cfg.update(seed=seed, experiment_name=name, replicate_of=selected, paths=cfg["paths"])
        print(f"[seeds] training {name}")
        train(run_cfg)


if __name__ == "__main__":
    main()
