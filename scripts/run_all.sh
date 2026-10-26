#!/usr/bin/env bash
# Full pipeline on MMRDR-CFP, in the order used for the reported results.
# Roughly 7-8 h on one T4 GPU (six ~1 h training runs); much less on an L4/A100.
# Every step writes to disk, so steps can be re-run individually after a disconnect.
set -euo pipefail
CONFIG=${1:-configs/default.yaml}

# 1. Data
python scripts/prepare_data.py   --config "$CONFIG"   # audit, FOV crop, duplicate search, splits

# 2. Baselines (majority class is computed inside evaluate.py)
python scripts/linear_probe.py   --config "$CONFIG"

# 3. Hyperparameter grid: {lr 3e-4, 1e-4} x {class-weighted, unweighted}, seed 42
python scripts/train.py --config configs/default.yaml
python scripts/train.py --config configs/ablation_unweighted.yaml
python scripts/train.py --config configs/lr1e-4_weighted.yaml
python scripts/train.py --config configs/lr1e-4_unweighted.yaml

# 4. Selection on validation QWK only
python scripts/select_model.py   --config "$CONFIG"

# 5. Seed replicates of the selected configuration (variability; not selection candidates)
python scripts/train_seeds.py    --config "$CONFIG" --seeds 1 2

# 6. First and only use of the test split, then analyses of the selected model
python scripts/evaluate.py       --config "$CONFIG"
python scripts/explain.py        --config "$CONFIG"
python scripts/robustness.py     --config "$CONFIG"

# 7. Optional: localisation against IDRiD pixel masks (see README)
if [ -d data/idrid/masks ]; then
  python scripts/localisation_external.py --config "$CONFIG" --images data/idrid/images \
         --masks data/idrid/masks --lesions MA HE EX SE --name idrid
fi

# 8. Tables and figures for the report
python scripts/collect_results.py --config "$CONFIG"
